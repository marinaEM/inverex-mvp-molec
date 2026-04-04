"""
Drug-target interaction features for INVEREX.

Constructs drug-specific vulnerability features using known drug targets
from GDSC2's PUTATIVE_TARGET column.  Three feature families are computed
per (patient, drug) pair and evaluated with a LODO ablation study:

Feature families
----------------
1. **Target expression** -- mean, max, and n_dysregulated (|z| > 2) of a
   drug's direct target genes in the patient expression profile.
2. **Pathway context** -- mean Hallmark-ssGSEA pathway score across
   pathways that contain any of the drug's target genes.
3. **Drug-patient compatibility** -- dot product of a binary drug-target
   mask with patient z-scores (a drug-specific vulnerability score).

Ablation study (LODO on CTR-DB)
-------------------------------
A. Gene reversal only (baseline)
B. Gene reversal + target expression features
C. Gene reversal + target expression + pathway context
D. Gene reversal + all drug-target features (target expr + pathway ctx +
   compatibility)

Each fold uses L1-logistic regression (C=0.05, solver=liblinear).

Entry point: ``run_drug_target_pipeline()`` or ``python -m src.features.drug_target_interactions``
"""

import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CACHE, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CTRDB_DIR = DATA_RAW / "ctrdb"
META_COLS = frozenset(
    {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
)


# ===================================================================
# 1. Parse drug targets from GDSC2
# ===================================================================

def parse_drug_targets(
    ref_path: Optional[Path] = None,
) -> dict[str, list[str]]:
    """
    Parse GDSC2 putative_target column to build a drug -> [target genes] map.

    Comma- and semicolon-separated targets are split.  Only tokens that look
    like gene symbols (upper-case alphanumeric, possibly with hyphens) are
    kept; descriptive phrases like 'Microtubule destabiliser' are dropped.

    Returns
    -------
    dict : drug_name (lower-cased) -> sorted list of target gene symbols.
    """
    if ref_path is None:
        ref_path = DATA_CACHE / "breast_dose_response_ref.parquet"
    ref = pd.read_parquet(ref_path)

    drug_targets: dict[str, set[str]] = {}
    for _, row in (
        ref[["drug_name", "putative_target"]]
        .drop_duplicates()
        .iterrows()
    ):
        drug = str(row["drug_name"]).strip().lower()
        raw = str(row["putative_target"]).strip()
        tokens = re.split(r"[,;]\s*", raw)
        gene_symbols = []
        for tok in tokens:
            tok = tok.strip()
            # Keep tokens that look like gene symbols:
            #   - all upper-case (with digits, hyphens, slashes allowed)
            #   - e.g. EGFR, BCL2, BCL-XL, PARP1, FGFR1
            # Drop descriptive phrases like 'Microtubule destabiliser'
            if re.match(r"^[A-Z][A-Z0-9]", tok) and " " not in tok:
                # Normalise: remove parentheses
                tok = tok.replace("(", "").replace(")", "")
                gene_symbols.append(tok)
        if gene_symbols:
            drug_targets.setdefault(drug, set()).update(gene_symbols)

    result = {d: sorted(gs) for d, gs in drug_targets.items()}
    logger.info(
        "Parsed drug targets: %d drugs with gene-level targets "
        "(from %d total GDSC2 drugs)",
        len(result),
        ref["drug_name"].nunique(),
    )
    return result


def parse_drug_pathways(
    ref_path: Optional[Path] = None,
) -> dict[str, list[str]]:
    """
    Build a drug -> [GDSC2 pathway_name] map.

    Returns
    -------
    dict : drug_name (lower-cased) -> list of pathway_name strings.
    """
    if ref_path is None:
        ref_path = DATA_CACHE / "breast_dose_response_ref.parquet"
    ref = pd.read_parquet(ref_path)

    drug_pw: dict[str, set[str]] = {}
    for _, row in (
        ref[["drug_name", "pathway_name"]]
        .drop_duplicates()
        .iterrows()
    ):
        drug = str(row["drug_name"]).strip().lower()
        pw = str(row["pathway_name"]).strip()
        if pw and pw.lower() not in ("nan", "other", "unclassified"):
            drug_pw.setdefault(drug, set()).add(pw)

    return {d: sorted(pws) for d, pws in drug_pw.items()}


# ===================================================================
# 2. Hallmark gene-set membership lookup
# ===================================================================

def load_hallmark_gene_sets() -> dict[str, set[str]]:
    """
    Load MSigDB Hallmark 2020 gene sets via gseapy.

    Returns
    -------
    dict : pathway_name -> set of gene symbols.
    """
    import gseapy as gp

    lib = gp.get_library("MSigDB_Hallmark_2020")
    return {name: set(genes) for name, genes in lib.items()}


def find_pathways_for_targets(
    targets: list[str],
    hallmark: dict[str, set[str]],
) -> list[str]:
    """
    Return Hallmark pathway names that contain at least one of the targets.
    """
    target_set = set(targets)
    return sorted(
        pw for pw, genes in hallmark.items() if target_set & genes
    )


# ===================================================================
# 3. Feature computation per (patient-set, drug) pair
# ===================================================================

def compute_target_expression_features(
    patient_z: np.ndarray,
    gene_names: list[str],
    target_genes: list[str],
    prefix: str = "tgt_",
) -> np.ndarray:
    """
    Feature family 1: target expression statistics.

    For each patient (row), compute over the drug's target genes:
      - mean expression z-score
      - max absolute z-score
      - n_dysregulated (count of |z| > 2)

    Parameters
    ----------
    patient_z : ndarray, shape (n_patients, n_genes)
        Patient expression z-scores.
    gene_names : list[str]
        Column names for patient_z.
    target_genes : list[str]
        Drug target gene symbols.

    Returns
    -------
    ndarray, shape (n_patients, 3) -- [mean_z, max_abs_z, n_dysreg]
    """
    gene_idx = [i for i, g in enumerate(gene_names) if g in set(target_genes)]

    n = patient_z.shape[0]
    if len(gene_idx) == 0:
        return np.zeros((n, 3), dtype=np.float64)

    sub = patient_z[:, gene_idx]  # (n_patients, n_targets_found)
    mean_z = np.nanmean(sub, axis=1)
    max_abs_z = np.nanmax(np.abs(sub), axis=1)
    n_dysreg = np.sum(np.abs(sub) > 2.0, axis=1).astype(np.float64)

    return np.column_stack([mean_z, max_abs_z, n_dysreg])


def compute_pathway_context_features(
    patient_z: np.ndarray,
    gene_names: list[str],
    target_genes: list[str],
    hallmark: dict[str, set[str]],
) -> np.ndarray:
    """
    Feature family 2: pathway context.

    For each patient, compute the mean z-score across genes in Hallmark
    pathways that contain any of the drug's target genes.  Returns a single
    scalar per patient (mean of per-pathway mean z-scores).

    Parameters
    ----------
    patient_z : ndarray, shape (n_patients, n_genes)
    gene_names : list[str]
    target_genes : list[str]
    hallmark : dict, pathway -> set of genes

    Returns
    -------
    ndarray, shape (n_patients, 1)
    """
    relevant_pws = find_pathways_for_targets(target_genes, hallmark)

    n = patient_z.shape[0]
    if not relevant_pws:
        return np.zeros((n, 1), dtype=np.float64)

    gene_name_set = set(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    pw_means = []
    for pw_name in relevant_pws:
        pw_genes = hallmark[pw_name] & gene_name_set
        if not pw_genes:
            continue
        idxs = [gene_to_idx[g] for g in pw_genes]
        pw_means.append(np.nanmean(patient_z[:, idxs], axis=1))

    if not pw_means:
        return np.zeros((n, 1), dtype=np.float64)

    # Mean of per-pathway mean z-scores
    stacked = np.column_stack(pw_means)  # (n_patients, n_pathways)
    result = np.nanmean(stacked, axis=1, keepdims=True)
    return result


def compute_compatibility_features(
    patient_z: np.ndarray,
    gene_names: list[str],
    target_genes: list[str],
) -> np.ndarray:
    """
    Feature family 3: drug-patient compatibility score.

    Dot product of a binary drug-target mask with patient z-scores.
    This gives a drug-specific vulnerability score: high when the patient
    over-expresses the drug's targets.

    Parameters
    ----------
    patient_z : ndarray, shape (n_patients, n_genes)
    gene_names : list[str]
    target_genes : list[str]

    Returns
    -------
    ndarray, shape (n_patients, 1)
    """
    mask = np.array(
        [1.0 if g in set(target_genes) else 0.0 for g in gene_names],
        dtype=np.float64,
    )
    # Dot product: sum of patient z-scores at target gene positions
    score = patient_z @ mask  # (n_patients,)
    return score.reshape(-1, 1)


# ===================================================================
# 4. CTR-DB loading and drug matching (self-contained)
# ===================================================================

def _normalize_drug_name(name: str) -> str:
    """Lower-case, strip whitespace/hyphens for fuzzy matching."""
    return re.sub(r"[\s\-_]+", "", name.strip().lower())


def _parse_combination_drugs(drug_string: str) -> list[str]:
    """
    Parse a CTR-DB combination drug string into individual drug names.
    """
    parts = []
    paren_re = re.compile(r"\([^)]+\)")
    parens = paren_re.findall(drug_string)
    remaining = paren_re.sub("", drug_string).strip()

    for p in parens:
        inner = p.strip("()")
        parts.extend([x.strip() for x in inner.split("+") if x.strip()])

    for token in re.split(r"\+|/", remaining):
        token = token.strip()
        if not token:
            continue
        if re.match(r"^[A-Z]{1,6}$", token):
            continue
        parts.append(token)

    if not parts:
        parts = [x.strip() for x in drug_string.split("+") if x.strip()]

    return parts


_CLASS_TO_DRUGS = {
    "anthracycline": ["doxorubicin", "epirubicin", "daunorubicin"],
    "taxane": ["paclitaxel", "docetaxel"],
    "platinum": ["cisplatin", "carboplatin", "oxaliplatin"],
    "glucocorticoids": ["dexamethasone", "prednisone", "prednisolone"],
}


def match_drugs_to_lincs(
    drug_string: str,
    lincs_drug_names: set[str],
) -> list[str]:
    """
    Given a CTR-DB drug string, return matching LINCS pert_iname values.
    """
    components = _parse_combination_drugs(drug_string)
    lincs_norm = {_normalize_drug_name(n): n for n in lincs_drug_names}

    matched = []
    for comp in components:
        comp_norm = _normalize_drug_name(comp)
        if comp_norm in lincs_norm:
            matched.append(lincs_norm[comp_norm])
            continue
        if comp_norm in _CLASS_TO_DRUGS:
            for specific in _CLASS_TO_DRUGS[comp_norm]:
                sn = _normalize_drug_name(specific)
                if sn in lincs_norm:
                    matched.append(lincs_norm[sn])
            continue
        for ln, orig in lincs_norm.items():
            if comp_norm in ln or ln in comp_norm:
                matched.append(orig)
                break

    return list(dict.fromkeys(matched))


def match_ctrdb_drug_to_gdsc(
    drug_string: str,
    gdsc_drug_names: set[str],
) -> list[str]:
    """
    Match a CTR-DB drug string to GDSC2 drug names (for target lookup).
    """
    components = _parse_combination_drugs(drug_string)
    gdsc_norm = {_normalize_drug_name(n): n for n in gdsc_drug_names}

    matched = []
    for comp in components:
        comp_norm = _normalize_drug_name(comp)
        if comp_norm in gdsc_norm:
            matched.append(gdsc_norm[comp_norm])
            continue
        if comp_norm in _CLASS_TO_DRUGS:
            for specific in _CLASS_TO_DRUGS[comp_norm]:
                sn = _normalize_drug_name(specific)
                if sn in gdsc_norm:
                    matched.append(gdsc_norm[sn])
            continue
        for gn, orig in gdsc_norm.items():
            if comp_norm in gn or gn in comp_norm:
                matched.append(orig)
                break

    return list(dict.fromkeys(matched))


def load_ctrdb_datasets() -> list[dict]:
    """
    Load all available CTR-DB datasets (expression + response labels).

    Returns a list of dicts: gse_id, drug_string, expression (DataFrame),
    response (Series 0/1).
    """
    gse_drug_map: dict[str, str] = {}
    for catalog_name in ["catalog.csv", "pan_cancer_catalog.csv"]:
        cat_path = CTRDB_DIR / catalog_name
        if cat_path.exists():
            cat = pd.read_csv(cat_path)
            for _, row in cat.iterrows():
                gse = row["geo_source"]
                if gse not in gse_drug_map:
                    gse_drug_map[gse] = row["drug"]

    datasets = []
    for gse_id in sorted(os.listdir(CTRDB_DIR)):
        gse_dir = CTRDB_DIR / gse_id
        if not gse_dir.is_dir() or not gse_id.startswith("GSE"):
            continue

        expr_path = gse_dir / f"{gse_id}_expression.parquet"
        resp_path = gse_dir / "response_labels.parquet"
        if not expr_path.exists() or not resp_path.exists():
            continue

        drug_string = gse_drug_map.get(gse_id)
        if drug_string is None:
            continue

        try:
            expression = pd.read_parquet(expr_path)
            response = pd.read_parquet(resp_path).squeeze()

            common = expression.index.intersection(response.index)
            if len(common) < 10:
                continue

            expression = expression.loc[common]
            response = response.loc[common]

            n_pos = int(response.sum())
            n_neg = len(response) - n_pos
            if n_pos < 3 or n_neg < 3:
                continue

            datasets.append({
                "gse_id": gse_id,
                "drug_string": drug_string,
                "expression": expression,
                "response": response,
            })
            logger.info(
                "Loaded %s: %d samples, drug=%s, resp=%d/%d",
                gse_id, len(common), drug_string, n_pos, len(response),
            )
        except Exception as e:
            logger.warning("Failed to load %s: %s", gse_id, e)

    logger.info("Loaded %d CTR-DB datasets with drug annotations", len(datasets))
    return datasets


# ===================================================================
# 5. Build per-dataset feature matrices for all ablation variants
# ===================================================================

def build_all_features_for_dataset(
    ds: dict,
    lincs_drugs: set[str],
    avg_sigs: pd.DataFrame,
    gene_cols: list[str],
    drug_targets: dict[str, list[str]],
    hallmark: dict[str, set[str]],
    gdsc_drug_names: set[str],
) -> Optional[dict]:
    """
    Build feature matrices for a single CTR-DB dataset.

    Returns a dict with keys:
        X_reversal, X_reversal_tgt, X_reversal_tgt_pw, X_all, y
    or None if the dataset cannot be used.
    """
    expr = ds["expression"]
    resp = ds["response"]
    drug_string = ds["drug_string"]

    # Match to LINCS
    lincs_matched = match_drugs_to_lincs(drug_string, lincs_drugs)
    if not lincs_matched:
        return None

    # Match to GDSC for targets
    gdsc_matched = match_ctrdb_drug_to_gdsc(drug_string, gdsc_drug_names)

    # Collect all target genes for the matched GDSC drugs
    all_targets: set[str] = set()
    for gd in gdsc_matched:
        gd_lower = gd.lower()
        if gd_lower in drug_targets:
            all_targets.update(drug_targets[gd_lower])

    # Common genes between patient expression and LINCS
    common_genes = sorted(set(gene_cols) & set(expr.columns))
    if len(common_genes) < 50:
        return None

    # Z-score patient expression (per gene across patients in this dataset)
    patient_mat = expr[common_genes].values.astype(np.float64)
    patient_mat = np.nan_to_num(patient_mat, nan=0.0)
    p_mean = patient_mat.mean(axis=0)
    p_std = patient_mat.std(axis=0)
    p_std[p_std == 0] = 1.0
    patient_z = (patient_mat - p_mean) / p_std

    # Mean LINCS drug signature across matched drugs
    sig_lookup = avg_sigs.set_index("pert_iname")
    drug_sig = np.zeros(len(common_genes), dtype=np.float64)
    n_d = 0
    for d in lincs_matched:
        if d in sig_lookup.index:
            vals = sig_lookup.loc[d].reindex(common_genes).values.astype(
                np.float64
            )
            drug_sig += np.nan_to_num(vals, nan=0.0)
            n_d += 1
    if n_d == 0:
        return None
    drug_sig /= n_d

    # --- Feature A: Gene reversal (element-wise product) ---
    # Reduce dimensionality: compute a single reversal score per patient
    # (negative Pearson correlation with drug signature)
    from scipy.stats import pearsonr

    n_patients = patient_z.shape[0]

    # Element-wise reversal features (patient z * drug sig)
    reversal_product = patient_z * drug_sig[np.newaxis, :]

    # Use summary statistics of the reversal product as features:
    # mean, std, min (most reversed), proportion negative
    rev_mean = np.nanmean(reversal_product, axis=1, keepdims=True)
    rev_std = np.nanstd(reversal_product, axis=1, keepdims=True)
    rev_min = np.nanmin(reversal_product, axis=1, keepdims=True)
    rev_neg_frac = (
        np.sum(reversal_product < 0, axis=1, keepdims=True).astype(np.float64)
        / reversal_product.shape[1]
    )

    # Full reversal score (negative Pearson r)
    rev_scores = np.array([
        -pearsonr(patient_z[i], drug_sig)[0]
        if np.std(patient_z[i]) > 0 else 0.0
        for i in range(n_patients)
    ]).reshape(-1, 1)

    X_reversal = np.hstack([rev_mean, rev_std, rev_min, rev_neg_frac, rev_scores])

    # --- Feature B: Target expression ---
    target_list = sorted(all_targets)
    if target_list:
        X_target_expr = compute_target_expression_features(
            patient_z, common_genes, target_list
        )
    else:
        X_target_expr = np.zeros((n_patients, 3), dtype=np.float64)

    # --- Feature C: Pathway context ---
    if target_list:
        X_pathway_ctx = compute_pathway_context_features(
            patient_z, common_genes, target_list, hallmark
        )
    else:
        X_pathway_ctx = np.zeros((n_patients, 1), dtype=np.float64)

    # --- Feature D: Drug-patient compatibility ---
    if target_list:
        X_compat = compute_compatibility_features(
            patient_z, common_genes, target_list
        )
    else:
        X_compat = np.zeros((n_patients, 1), dtype=np.float64)

    # Assemble ablation variants
    # A: reversal only
    Xa = X_reversal
    # B: reversal + target expression
    Xb = np.hstack([X_reversal, X_target_expr])
    # C: reversal + target expression + pathway context
    Xc = np.hstack([X_reversal, X_target_expr, X_pathway_ctx])
    # D: reversal + all drug-target features
    Xd = np.hstack([X_reversal, X_target_expr, X_pathway_ctx, X_compat])

    y = resp.values.astype(int)

    return {
        "X_reversal": Xa,
        "X_reversal_tgt": Xb,
        "X_reversal_tgt_pw": Xc,
        "X_all": Xd,
        "y": y,
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "drug": drug_string,
        "n_targets": len(target_list),
        "targets_found": target_list,
    }


# ===================================================================
# 6. LODO ablation evaluation
# ===================================================================

ABLATION_VARIANTS = {
    "A_reversal_only": "X_reversal",
    "B_reversal_plus_target_expr": "X_reversal_tgt",
    "C_reversal_plus_target_expr_pw": "X_reversal_tgt_pw",
    "D_reversal_plus_all_dt": "X_all",
}


def run_lodo_ablation(
    dataset_features: dict[str, dict],
    C: float = 0.05,
) -> pd.DataFrame:
    """
    Leave-One-Dataset-Out ablation across the four feature variants.

    Parameters
    ----------
    dataset_features : dict
        gse_id -> dict with keys X_reversal, X_reversal_tgt,
        X_reversal_tgt_pw, X_all, y, etc.
    C : float
        Inverse regularisation strength for L1-logistic.

    Returns
    -------
    DataFrame with per-fold, per-variant results.
    """
    all_geos = sorted(dataset_features.keys())
    if len(all_geos) < 2:
        logger.warning("Need >= 2 datasets for LODO; got %d", len(all_geos))
        return pd.DataFrame()

    results = []
    for held_out in all_geos:
        train_geos = [g for g in all_geos if g != held_out]

        for variant_name, feat_key in ABLATION_VARIANTS.items():
            # Assemble training data
            X_train_parts, y_train_parts = [], []
            for tg in train_geos:
                X_train_parts.append(dataset_features[tg][feat_key])
                y_train_parts.append(dataset_features[tg]["y"])

            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            X_test = dataset_features[held_out][feat_key]
            y_test = dataset_features[held_out]["y"]

            # Standardise
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_train_s = np.nan_to_num(X_train_s, nan=0.0)
            X_test_s = np.nan_to_num(X_test_s, nan=0.0)

            # Train L1-logistic regression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = LogisticRegression(
                    C=C,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                    l1_ratio=1.0,
                )
                try:
                    clf.fit(X_train_s, y_train)
                except Exception as e:
                    logger.warning(
                        "LODO %s / %s: fit failed (%s)",
                        held_out, variant_name, e,
                    )
                    continue

            # Predict and score
            try:
                proba = clf.predict_proba(X_test_s)
                if proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    auc = 0.5
            except Exception:
                auc = 0.5

            results.append({
                "held_out_dataset": held_out,
                "feature_set": variant_name,
                "auc": round(auc, 4),
                "n_test": len(y_test),
                "n_test_pos": int(y_test.sum()),
                "n_train": len(y_train),
                "n_train_pos": int(y_train.sum()),
                "n_features": X_train.shape[1],
                "drug": dataset_features[held_out]["drug"],
                "n_targets": dataset_features[held_out]["n_targets"],
            })

            logger.info(
                "  LODO %s [%s]: AUC=%.3f (test=%d, feats=%d, targets=%d)",
                held_out,
                variant_name,
                auc,
                len(y_test),
                X_train.shape[1],
                dataset_features[held_out]["n_targets"],
            )

    return pd.DataFrame(results)


# ===================================================================
# 7. Pipeline entry point
# ===================================================================

def run_drug_target_pipeline():
    """
    End-to-end pipeline:
      1. Parse drug targets from GDSC2
      2. Load LINCS signatures, compute dose-averaged drug profiles
      3. Load Hallmark gene sets
      4. Load CTR-DB datasets
      5. Build features per dataset
      6. Run LODO ablation
      7. Save results
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.time()

    logger.info("=" * 60)
    logger.info("DRUG-TARGET INTERACTION FEATURES PIPELINE")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Parse drug targets from GDSC2                               #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 1: Parsing drug targets from GDSC2 ...")
    drug_targets = parse_drug_targets()
    drug_pathways = parse_drug_pathways()

    n_with_targets = sum(1 for ts in drug_targets.values() if ts)
    logger.info(
        "  %d drugs with gene-level targets, %d with pathway annotations",
        n_with_targets,
        len(drug_pathways),
    )

    # Print a sample
    for drug, targets in sorted(drug_targets.items())[:5]:
        logger.info("  %s: %s", drug, targets)

    # ------------------------------------------------------------------ #
    # Step 2: Load LINCS signatures                                       #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 2: Loading LINCS signatures ...")

    # Try all-cell-line first, fallback to breast-only
    lincs_path = DATA_CACHE / "all_cellline_drug_signatures.parquet"
    if not lincs_path.exists():
        lincs_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    if not lincs_path.exists():
        logger.error("No LINCS signatures found.")
        return

    lincs_sigs = pd.read_parquet(lincs_path)
    gene_cols = [c for c in lincs_sigs.columns if c not in META_COLS]
    lincs_drugs = set(lincs_sigs["pert_iname"].unique())
    logger.info(
        "  %d signatures, %d drugs, %d genes",
        len(lincs_sigs),
        len(lincs_drugs),
        len(gene_cols),
    )

    # Compute dose-averaged signatures
    avg_sigs = (
        lincs_sigs.groupby("pert_iname")[gene_cols]
        .mean()
        .reset_index()
    )
    logger.info("  %d dose-averaged drug profiles", len(avg_sigs))

    # ------------------------------------------------------------------ #
    # Step 3: Load Hallmark gene sets                                     #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 3: Loading Hallmark gene sets ...")
    hallmark = load_hallmark_gene_sets()
    logger.info("  %d Hallmark pathways loaded", len(hallmark))

    # ------------------------------------------------------------------ #
    # Step 4: Load CTR-DB datasets                                        #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 4: Loading CTR-DB datasets ...")
    datasets = load_ctrdb_datasets()

    if len(datasets) < 2:
        logger.error("Not enough CTR-DB datasets for LODO evaluation.")
        return

    # ------------------------------------------------------------------ #
    # Step 5: Build features per dataset                                  #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 5: Building drug-target features per dataset ...")

    gdsc_drug_names = set()
    ref = pd.read_parquet(DATA_CACHE / "breast_dose_response_ref.parquet")
    gdsc_drug_names = set(ref["drug_name"].unique())

    dataset_features: dict[str, dict] = {}
    for ds in datasets:
        gse_id = ds["gse_id"]
        feats = build_all_features_for_dataset(
            ds=ds,
            lincs_drugs=lincs_drugs,
            avg_sigs=avg_sigs,
            gene_cols=gene_cols,
            drug_targets=drug_targets,
            hallmark=hallmark,
            gdsc_drug_names=gdsc_drug_names,
        )
        if feats is not None:
            dataset_features[gse_id] = feats
            logger.info(
                "  %s: reversal=%d, +tgt=%d, +pw=%d, all=%d feats "
                "(targets=%s)",
                gse_id,
                feats["X_reversal"].shape[1],
                feats["X_reversal_tgt"].shape[1],
                feats["X_reversal_tgt_pw"].shape[1],
                feats["X_all"].shape[1],
                feats["targets_found"][:5],
            )
        else:
            logger.info("  %s: skipped (no LINCS match or too few genes)", gse_id)

    logger.info(
        "\n  %d / %d datasets ready for LODO",
        len(dataset_features),
        len(datasets),
    )

    if len(dataset_features) < 2:
        logger.error("Not enough datasets with valid features for LODO.")
        return

    # ------------------------------------------------------------------ #
    # Step 6: LODO ablation                                               #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: LODO ablation evaluation")
    logger.info("=" * 60)

    lodo_results = run_lodo_ablation(dataset_features, C=0.05)

    # ------------------------------------------------------------------ #
    # Step 7: Save results                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Saving results")
    logger.info("=" * 60)

    RESULTS.mkdir(parents=True, exist_ok=True)

    if not lodo_results.empty:
        out_path = RESULTS / "ablation_drug_target_features.tsv"
        lodo_results.to_csv(out_path, sep="\t", index=False)
        logger.info("  Saved per-fold results to %s", out_path)

        # Summary table
        summary = (
            lodo_results.groupby("feature_set")["auc"]
            .agg(["mean", "std", "median", "count"])
            .round(4)
            .reset_index()
        )
        summary.columns = [
            "feature_set",
            "mean_auc",
            "std_auc",
            "median_auc",
            "n_datasets",
        ]

        logger.info("\n" + "=" * 60)
        logger.info("ABLATION SUMMARY")
        logger.info("=" * 60)
        for _, row in summary.iterrows():
            logger.info(
                "  %-40s: AUC = %.4f +/- %.4f  (median=%.4f, n=%d)",
                row["feature_set"],
                row["mean_auc"],
                row["std_auc"],
                row["median_auc"],
                int(row["n_datasets"]),
            )
    else:
        logger.warning("No LODO results to save.")

    elapsed = time.time() - t_start
    logger.info("\nTotal pipeline time: %.0fs", elapsed)


if __name__ == "__main__":
    run_drug_target_pipeline()
