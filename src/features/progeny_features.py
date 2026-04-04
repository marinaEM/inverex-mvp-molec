"""
PROGENy pathway activity features for INVEREX.

PROGENy estimates 14 signaling pathway activities from gene expression using
footprint-based inference (downstream transcriptional targets, not pathway
member genes).  This is complementary to ssGSEA Hallmark features which use
gene set membership.

Pathways
--------
EGFR, Hypoxia, JAK-STAT, MAPK, NFkB, p53, PI3K, TNFa, TGFb, Trail,
VEGF, WNT, Androgen, Estrogen.

Drug-pathway mapping
--------------------
Built from GDSC2 PATHWAY_NAME + manual curation to link CTR-DB drugs to
their primary PROGENy pathway targets.

Ablation (LODO on CTR-DB)
-------------------------
A. ssGSEA Hallmark (48 pathways) -- baseline
B. PROGENy only (14 pathways)
C. ssGSEA + PROGENy combined (62 pathways)
D. PROGENy + drug-specific pathway features (using mapping)

L1-logistic regression, C=0.05, LODO.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_CACHE, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PROGENy pathway list (14 pathways)
# ---------------------------------------------------------------------------
PROGENY_PATHWAYS = [
    "Androgen", "EGFR", "Estrogen", "Hypoxia", "JAK-STAT", "MAPK",
    "NFkB", "PI3K", "TGFb", "TNFa", "Trail", "VEGF", "WNT", "p53",
]

# ---------------------------------------------------------------------------
# Drug -> PROGENy pathway mapping
# Built from GDSC2 PATHWAY_NAME + manual curation for breast cancer drugs
# ---------------------------------------------------------------------------
DRUG_PROGENY_PATHWAY: dict[str, list[str]] = {
    # EGFR signaling
    "lapatinib":    ["EGFR"],
    "neratinib":    ["EGFR"],
    "afatinib":     ["EGFR"],
    "gefitinib":    ["EGFR"],
    "erlotinib":    ["EGFR"],
    # PI3K/mTOR signaling
    "alpelisib":    ["PI3K"],
    "everolimus":   ["PI3K"],
    "wortmannin":   ["PI3K"],
    "sirolimus":    ["PI3K"],
    "ly-294002":    ["PI3K"],
    # Estrogen / Hormone-related
    "tamoxifen":    ["Estrogen"],
    "fulvestrant":  ["Estrogen"],
    "raloxifene":   ["Estrogen"],
    "letrozole":    ["Estrogen"],
    "anastrozole":  ["Estrogen"],
    "exemestane":   ["Estrogen"],
    # MAPK / ERK signaling
    "trametinib":   ["MAPK"],
    "selumetinib":  ["MAPK"],
    "refametinib":  ["MAPK"],
    "plx-4720":     ["MAPK"],
    # p53 / DNA damage
    "cisplatin":    ["p53"],
    "doxorubicin":  ["p53"],
    "olaparib":     ["p53"],
    "veliparib":    ["p53"],
    "epirubicin":   ["p53"],
    "carboplatin":  ["p53"],
    "fluorouracil": ["p53"],
    "methotrexate": ["p53"],
    "cyclophosphamide": ["p53"],
    "capecitabine": ["p53"],
    # TNFa / NFkB
    "bortezomib":   ["NFkB"],
    # JAK-STAT
    "tofacitinib":  ["JAK-STAT"],
    "ruxolitinib":  ["JAK-STAT"],
    # Microtubule agents -- affect p53 indirectly via mitotic stress
    "paclitaxel":   ["p53"],
    "docetaxel":    ["p53"],
    "ixabepilone":  ["p53"],
    # VEGF
    "bevacizumab":  ["VEGF"],
    # WNT
    "sb216763":     ["WNT"],
}


# ---------------------------------------------------------------------------
# Core PROGENy computation
# ---------------------------------------------------------------------------

def compute_progeny_activities(
    expr_df: pd.DataFrame,
    organism: str = "human",
    top: int = 500,
) -> pd.DataFrame:
    """
    Compute PROGENy pathway activity scores for a samples x genes matrix.

    Parameters
    ----------
    expr_df : DataFrame
        Samples (rows) x genes (columns).  Gene symbols as column names.
    organism : str
        'human' or 'mouse'.
    top : int
        Number of top genes per pathway to use from PROGENy model.

    Returns
    -------
    DataFrame : samples (rows) x 14 pathways (columns), float64.
    """
    import decoupler as dc

    n_samples, n_genes = expr_df.shape
    logger.info(
        f"Running PROGENy: {n_samples} samples x {n_genes} genes, "
        f"organism={organism}, top={top}"
    )

    t0 = time.time()

    # Get PROGENy model (source=pathway, target=gene, weight=coefficient)
    progeny_model = dc.op.progeny(organism=organism, top=top)

    # Run multivariate linear model (MLM) -- recommended method for PROGENy
    result = dc.mt.mlm(data=expr_df, net=progeny_model)

    # result is a tuple of (estimate, pvalue) DataFrames
    # Both are samples x pathways
    if isinstance(result, tuple):
        activities = result[0]
    else:
        activities = result

    activities = activities.astype(np.float64)

    elapsed = time.time() - t0
    logger.info(
        f"PROGENy complete: {activities.shape[1]} pathways scored "
        f"in {elapsed:.1f}s"
    )

    return activities


# ---------------------------------------------------------------------------
# Apply to LINCS drug signatures
# ---------------------------------------------------------------------------

def transform_lincs_to_progeny(
    lincs_df: Optional[pd.DataFrame] = None,
    cache_path: Optional[Path] = None,
    top: int = 500,
) -> pd.DataFrame:
    """
    Transform LINCS drug z-score signatures into PROGENy pathway scores.

    Parameters
    ----------
    lincs_df : DataFrame, optional
        Raw LINCS signatures (samples x [meta + genes]).  If None, loaded
        from cache.
    cache_path : Path, optional
        Where to save / load cached scores.
    top : int
        Number of top PROGENy genes per pathway.

    Returns
    -------
    DataFrame with meta columns + 14 PROGENy pathway score columns.
    """
    if cache_path is None:
        cache_path = DATA_CACHE / "lincs_progeny.parquet"

    if cache_path.exists():
        logger.info(f"Loading cached LINCS PROGENy scores from {cache_path}")
        return pd.read_parquet(cache_path)

    # Load raw signatures if not provided
    if lincs_df is None:
        raw_path = DATA_CACHE / "breast_l1000_signatures.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"LINCS signatures not found at {raw_path}"
            )
        lincs_df = pd.read_parquet(raw_path)

    # Separate meta and gene columns
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id",
                 "dose_um", "pert_idose"}
    gene_cols = [c for c in lincs_df.columns if c not in meta_cols]
    meta = lincs_df[[c for c in lincs_df.columns if c in meta_cols]].copy()
    expr = lincs_df[gene_cols].astype(np.float64)

    # Compute PROGENy scores
    progeny_scores = compute_progeny_activities(expr, top=top)

    # Re-attach metadata
    result = pd.concat(
        [meta.reset_index(drop=True),
         progeny_scores.reset_index(drop=True)],
        axis=1,
    )

    result.to_parquet(cache_path, index=False)
    logger.info(f"Saved LINCS PROGENy scores to {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Apply to CTR-DB patient expression
# ---------------------------------------------------------------------------

def transform_ctrdb_to_progeny(
    ctrdb_datasets: Optional[dict] = None,
    top: int = 500,
    cache_dir: Optional[Path] = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Transform CTR-DB patient expression datasets to PROGENy pathway scores.

    Parameters
    ----------
    ctrdb_datasets : dict, optional
        geo_id -> (expression_df, response_series).  If None, loaded via
        ``load_all_breast_ctrdb``.
    top : int
        Number of top PROGENy genes per pathway.
    cache_dir : Path, optional
        Where to cache results.

    Returns
    -------
    dict : geo_id -> (progeny_scores_df, response_series)
    """
    if cache_dir is None:
        cache_dir = DATA_CACHE / "progeny_ctrdb"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if ctrdb_datasets is None:
        from src.data_ingestion.ctrdb import load_all_breast_ctrdb
        ctrdb_datasets = load_all_breast_ctrdb()

    transformed = {}
    for geo_id, (expr, labels) in sorted(ctrdb_datasets.items()):
        cache_path = cache_dir / f"{geo_id}_progeny.parquet"

        if cache_path.exists():
            pw_scores = pd.read_parquet(cache_path)
            pw_scores.index = pw_scores.index.astype(str)
            common = labels.index.intersection(pw_scores.index)
            if len(common) > 0:
                transformed[geo_id] = (
                    pw_scores.loc[common], labels.loc[common]
                )
                logger.info(
                    f"  {geo_id}: loaded cached PROGENy scores "
                    f"({pw_scores.shape[1]} pathways, {len(common)} samples)"
                )
            continue

        try:
            pw_scores = compute_progeny_activities(expr, top=top)
            pw_scores = pw_scores.dropna(axis=1, how="all")
            pw_scores = pw_scores.fillna(0.0)

            pw_scores.to_parquet(cache_path)

            common = labels.index.intersection(pw_scores.index)
            if len(common) > 0:
                transformed[geo_id] = (
                    pw_scores.loc[common], labels.loc[common]
                )
                logger.info(
                    f"  {geo_id}: {pw_scores.shape[1]} PROGENy pathways, "
                    f"{len(common)} samples"
                )
            else:
                logger.warning(
                    f"  {geo_id}: no sample overlap after PROGENy"
                )
        except Exception as e:
            logger.warning(f"  {geo_id}: PROGENy failed ({e})")

    logger.info(
        f"Transformed {len(transformed)}/{len(ctrdb_datasets)} "
        f"CTR-DB datasets to PROGENy scores"
    )
    return transformed


# ---------------------------------------------------------------------------
# Apply to TCGA-BRCA expression
# ---------------------------------------------------------------------------

def transform_tcga_to_progeny(
    tcga_expr: Optional[pd.DataFrame] = None,
    cache_path: Optional[Path] = None,
    top: int = 500,
) -> pd.DataFrame:
    """
    Transform TCGA-BRCA expression to PROGENy pathway activity scores.

    Parameters
    ----------
    tcga_expr : DataFrame, optional
        Samples x genes.  If None, loaded from cache.
    cache_path : Path, optional
        Where to save / load cached scores.
    top : int
        Number of top PROGENy genes.

    Returns
    -------
    DataFrame : samples (rows) x 14 PROGENy pathways (columns).
    """
    if cache_path is None:
        cache_path = DATA_CACHE / "tcga_brca_progeny.parquet"

    if cache_path.exists():
        logger.info(f"Loading cached TCGA PROGENy scores from {cache_path}")
        return pd.read_parquet(cache_path)

    if tcga_expr is None:
        expr_path = DATA_CACHE / "tcga_brca_expression.parquet"
        if not expr_path.exists():
            raise FileNotFoundError(
                f"TCGA-BRCA expression not found at {expr_path}"
            )
        tcga_expr = pd.read_parquet(expr_path)

    progeny_scores = compute_progeny_activities(tcga_expr, top=top)
    progeny_scores.to_parquet(cache_path)
    logger.info(f"Saved TCGA PROGENy scores to {cache_path}")
    return progeny_scores


# ---------------------------------------------------------------------------
# Drug-specific PROGENy pathway feature
# ---------------------------------------------------------------------------

def get_drug_progeny_pathways(drug_name: str) -> list[str]:
    """
    Return PROGENy pathways targeted by a drug.

    Uses the DRUG_PROGENY_PATHWAY mapping (GDSC2 + manual curation).
    Falls back to GDSC2 PATHWAY_NAME column if not in the curated map.
    """
    norm = drug_name.strip().lower().replace("-", "").replace(" ", "")
    # Direct lookup
    for key, pathways in DRUG_PROGENY_PATHWAY.items():
        if norm == key.replace("-", "").replace(" ", ""):
            return pathways
    return []


def build_drug_specific_progeny_features(
    patient_progeny: np.ndarray,
    drug_progeny: np.ndarray,
    pathway_names: list[str],
    drug_components: list[str],
) -> np.ndarray:
    """
    Build drug-specific PROGENy features by amplifying pathway columns
    that the drug targets.

    For each sample, the feature vector is:
        reversal = patient_progeny * drug_progeny
        drug_specific = reversal * on_target_mask

    where on_target_mask is 2.0 for pathways targeted by the drug and 1.0
    for others.  This preserves the full reversal signal while up-weighting
    the pharmacologically relevant pathways.

    Parameters
    ----------
    patient_progeny : ndarray, shape (n_samples, n_pathways)
        Patient PROGENy scores.
    drug_progeny : ndarray, shape (n_pathways,)
        Mean drug PROGENy perturbation signature.
    pathway_names : list of str
        Names of the pathways (columns).
    drug_components : list of str
        Normalised drug component names.

    Returns
    -------
    ndarray, shape (n_samples, n_pathways)
        Drug-specific reversal features.
    """
    # Get targeted pathways for all components of the drug regimen
    targeted = set()
    for comp in drug_components:
        targeted.update(get_drug_progeny_pathways(comp))

    # Build on-target mask: 2x weight for targeted pathways
    mask = np.ones(len(pathway_names), dtype=np.float64)
    for i, pw in enumerate(pathway_names):
        if pw in targeted:
            mask[i] = 2.0

    # Reversal features with drug-specific amplification
    reversal = patient_progeny * drug_progeny[np.newaxis, :]
    return reversal * mask[np.newaxis, :]


# ---------------------------------------------------------------------------
# LODO ablation: ssGSEA Hallmark vs PROGENy vs combined vs drug-specific
# ---------------------------------------------------------------------------

def run_progeny_ablation(
    ctrdb_gene: dict[str, tuple[pd.DataFrame, pd.Series]],
    ctrdb_hallmark: dict[str, tuple[pd.DataFrame, pd.Series]],
    ctrdb_progeny: dict[str, tuple[pd.DataFrame, pd.Series]],
    lincs_sigs: pd.DataFrame,
    lincs_hallmark: pd.DataFrame,
    lincs_progeny: pd.DataFrame,
    catalog: pd.DataFrame,
    C: float = 0.05,
) -> pd.DataFrame:
    """
    Leave-One-Dataset-Out ablation comparing feature sets:

    A. ssGSEA Hallmark only (48 pathways) -- baseline
    B. PROGENy only (14 pathways)
    C. ssGSEA + PROGENy combined (62 pathways)
    D. PROGENy + drug-specific pathway features

    Parameters
    ----------
    ctrdb_gene : dict
        geo_id -> (gene_expression_df, labels).
    ctrdb_hallmark : dict
        geo_id -> (hallmark_pathway_scores_df, labels).
    ctrdb_progeny : dict
        geo_id -> (progeny_pathway_scores_df, labels).
    lincs_sigs : DataFrame
        Raw LINCS signatures.
    lincs_hallmark : DataFrame
        LINCS Hallmark pathway scores.
    lincs_progeny : DataFrame
        LINCS PROGENy pathway scores.
    catalog : DataFrame
        CTR-DB catalog with geo_source and drug columns.
    C : float
        Inverse regularisation for L1-logistic.

    Returns
    -------
    DataFrame with per-dataset AUC for each feature set.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from src.models.recalibrate_signatures import (
        _normalise_drug_name,
        match_drugs_to_lincs,
        parse_regimen_components,
    )

    # ----- Build per-drug mean LINCS signatures for each feature space -----
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id",
                 "dose_um", "pert_idose"}

    # Hallmark
    hallmark_cols = [c for c in lincs_hallmark.columns if c not in meta_cols]
    drug_mean_hallmark = {}
    for drug, grp in lincs_hallmark.groupby(
        lincs_hallmark["pert_iname"].str.lower()
    ):
        drug_mean_hallmark[drug] = grp[hallmark_cols].mean()

    # PROGENy
    progeny_cols = [c for c in lincs_progeny.columns if c not in meta_cols]
    drug_mean_progeny = {}
    for drug, grp in lincs_progeny.groupby(
        lincs_progeny["pert_iname"].str.lower()
    ):
        drug_mean_progeny[drug] = grp[progeny_cols].mean()

    lincs_drug_set = set(lincs_sigs["pert_iname"].str.lower().unique())

    # ----- Get drug for each dataset -----
    def _get_drug(geo_id):
        row = catalog[catalog["geo_source"] == geo_id]
        if row.empty:
            return ""
        return str(row.iloc[0]["drug"])

    # ----- Identify eligible datasets (present in all three) -----
    common_datasets = sorted(
        set(ctrdb_hallmark.keys()) & set(ctrdb_progeny.keys())
    )
    logger.info(
        f"LODO ablation: {len(common_datasets)} datasets "
        f"with both Hallmark and PROGENy features"
    )

    # ----- Determine global feature sets -----
    # Hallmark pathways: intersect across all datasets
    global_hallmark = None
    for geo_id in common_datasets:
        ds_cols = set(ctrdb_hallmark[geo_id][0].columns) & set(hallmark_cols)
        if global_hallmark is None:
            global_hallmark = ds_cols
        else:
            global_hallmark = global_hallmark & ds_cols
    global_hallmark = sorted(global_hallmark) if global_hallmark else []

    # PROGENy pathways: intersect across all datasets
    global_progeny = None
    for geo_id in common_datasets:
        ds_cols = set(ctrdb_progeny[geo_id][0].columns) & set(progeny_cols)
        if global_progeny is None:
            global_progeny = ds_cols
        else:
            global_progeny = global_progeny & ds_cols
    global_progeny = sorted(global_progeny) if global_progeny else []

    logger.info(
        f"Global feature sets: "
        f"Hallmark={len(global_hallmark)}, PROGENy={len(global_progeny)}, "
        f"Combined={len(global_hallmark) + len(global_progeny)}"
    )

    if len(global_hallmark) < 3 or len(global_progeny) < 3:
        logger.warning("Too few common pathways for ablation")
        return pd.DataFrame()

    # ----- Pre-compute per-dataset features -----
    dataset_features = {}
    eligible_datasets = []

    for geo_id in common_datasets:
        drug_str = _get_drug(geo_id)
        components = parse_regimen_components(drug_str)
        matched = match_drugs_to_lincs(components, lincs_drug_set)
        if not matched:
            logger.info(f"  {geo_id}: no LINCS-matched drugs, skipping")
            continue

        expr_hallmark, labels_hallmark = ctrdb_hallmark[geo_id]
        expr_progeny, labels_progeny = ctrdb_progeny[geo_id]

        common_samples = (
            labels_hallmark.index
            .intersection(labels_progeny.index)
            .intersection(expr_hallmark.index)
            .intersection(expr_progeny.index)
        )

        if len(common_samples) < 10:
            continue

        labels = labels_hallmark.loc[common_samples]
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos < 3 or n_neg < 3:
            continue

        # --- Hallmark reversal features ---
        drug_sig_hallmark = np.zeros(len(global_hallmark), dtype=np.float64)
        n_d = 0
        for d in matched:
            dl = d.lower()
            if dl in drug_mean_hallmark:
                sig = drug_mean_hallmark[dl]
                vals = sig.reindex(global_hallmark).values.astype(np.float64)
                drug_sig_hallmark += np.nan_to_num(vals, 0.0)
                n_d += 1
        if n_d > 0:
            drug_sig_hallmark /= n_d
        else:
            continue

        X_hallmark = np.zeros(
            (len(common_samples), len(global_hallmark)), dtype=np.float64
        )
        for i, p in enumerate(global_hallmark):
            if p in expr_hallmark.columns:
                X_hallmark[:, i] = (
                    expr_hallmark.loc[common_samples, p]
                    .values.astype(np.float64)
                )
        X_hallmark = np.nan_to_num(X_hallmark, 0.0)
        X_hallmark_rev = X_hallmark * drug_sig_hallmark[np.newaxis, :]

        # --- PROGENy reversal features ---
        drug_sig_progeny = np.zeros(len(global_progeny), dtype=np.float64)
        n_d = 0
        for d in matched:
            dl = d.lower()
            if dl in drug_mean_progeny:
                sig = drug_mean_progeny[dl]
                vals = sig.reindex(global_progeny).values.astype(np.float64)
                drug_sig_progeny += np.nan_to_num(vals, 0.0)
                n_d += 1
        if n_d > 0:
            drug_sig_progeny /= n_d
        else:
            continue

        X_progeny = np.zeros(
            (len(common_samples), len(global_progeny)), dtype=np.float64
        )
        for i, p in enumerate(global_progeny):
            if p in expr_progeny.columns:
                X_progeny[:, i] = (
                    expr_progeny.loc[common_samples, p]
                    .values.astype(np.float64)
                )
        X_progeny = np.nan_to_num(X_progeny, 0.0)
        X_progeny_rev = X_progeny * drug_sig_progeny[np.newaxis, :]

        # --- Combined: Hallmark + PROGENy ---
        X_combined = np.hstack([X_hallmark_rev, X_progeny_rev])

        # --- Drug-specific PROGENy features ---
        X_drug_specific = build_drug_specific_progeny_features(
            patient_progeny=X_progeny,
            drug_progeny=drug_sig_progeny,
            pathway_names=global_progeny,
            drug_components=components,
        )

        y = labels.values.astype(int)

        dataset_features[geo_id] = {
            "X_hallmark": X_hallmark_rev,
            "X_progeny": X_progeny_rev,
            "X_combined": X_combined,
            "X_drug_specific": X_drug_specific,
            "y": y,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "drug": drug_str,
        }
        eligible_datasets.append(geo_id)

    logger.info(
        f"LODO ablation: {len(dataset_features)} eligible datasets "
        f"(Hallmark={len(global_hallmark)}, PROGENy={len(global_progeny)}, "
        f"Combined={len(global_hallmark)+len(global_progeny)})"
    )

    if len(dataset_features) < 2:
        logger.warning("Not enough datasets for LODO ablation")
        return pd.DataFrame()

    # ----- LODO loop -----
    results = []
    all_geos = sorted(dataset_features.keys())

    feature_sets = [
        ("A_ssGSEA_Hallmark", "X_hallmark"),
        ("B_PROGENy_only", "X_progeny"),
        ("C_ssGSEA_plus_PROGENy", "X_combined"),
        ("D_PROGENy_drug_specific", "X_drug_specific"),
    ]

    for held_out in all_geos:
        train_geos = [g for g in all_geos if g != held_out]

        for feat_name, feat_key in feature_sets:
            # Assemble training data
            X_train_parts = []
            y_train_parts = []
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
            X_train_s = np.nan_to_num(X_train_s, 0.0)
            X_test_s = np.nan_to_num(X_test_s, 0.0)

            # Train L1-logistic
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = LogisticRegression(
                    C=C,
                    l1_ratio=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                )
                try:
                    clf.fit(X_train_s, y_train)
                except Exception as e:
                    logger.warning(
                        f"  LODO {held_out}/{feat_name}: fit failed ({e})"
                    )
                    continue

            # Predict
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
                "feature_set": feat_name,
                "auc": round(auc, 4),
                "n_test": len(y_test),
                "n_test_pos": int(y_test.sum()),
                "n_train": len(y_train),
                "n_train_pos": int(y_train.sum()),
                "n_features": X_train.shape[1],
                "drug": dataset_features[held_out]["drug"],
            })

            logger.info(
                f"  LODO {held_out} [{feat_name}]: "
                f"AUC={auc:.3f} "
                f"(test={len(y_test)}, features={X_train.shape[1]})"
            )

    results_df = pd.DataFrame(results)
    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_progeny_pipeline():
    """
    End-to-end PROGENy features pipeline:
    1. Transform LINCS signatures to PROGENy scores
    2. Transform CTR-DB patient expression to PROGENy scores
    3. Transform TCGA-BRCA expression to PROGENy scores
    4. Run LODO ablation (Hallmark vs PROGENy vs combined vs drug-specific)
    5. Save results
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.time()

    # ------------------------------------------------------------------ #
    # Step 0: Load raw data                                               #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("PROGENy FEATURES PIPELINE")
    logger.info("=" * 60)

    logger.info("Loading LINCS breast signatures ...")
    lincs_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    if not lincs_path.exists():
        logger.error(f"LINCS signatures not found at {lincs_path}")
        return
    lincs_sigs = pd.read_parquet(lincs_path)
    logger.info(
        f"  {lincs_sigs.shape[0]} signatures, "
        f"{lincs_sigs['pert_iname'].nunique()} drugs"
    )

    logger.info("Loading CTR-DB patient datasets ...")
    from src.data_ingestion.ctrdb import load_all_breast_ctrdb
    ctrdb_gene = load_all_breast_ctrdb()
    logger.info(f"  {len(ctrdb_gene)} datasets loaded")

    if len(ctrdb_gene) == 0:
        logger.error("No CTR-DB datasets available.")
        return

    # Load catalog
    cat_path = DATA_RAW / "ctrdb" / "catalog.csv"
    if cat_path.exists():
        catalog = pd.read_csv(cat_path)
    else:
        logger.error("CTR-DB catalog not found.")
        return

    # ------------------------------------------------------------------ #
    # Step 1: Transform LINCS to PROGENy scores                          #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Transforming LINCS signatures to PROGENy scores")
    logger.info("=" * 60)

    lincs_progeny = transform_lincs_to_progeny(lincs_df=lincs_sigs, top=500)
    progeny_meta = {"sig_id", "pert_id", "pert_iname", "cell_id",
                    "dose_um", "pert_idose"}
    progeny_feat_cols = [
        c for c in lincs_progeny.columns if c not in progeny_meta
    ]
    logger.info(f"  LINCS PROGENy features: {len(progeny_feat_cols)} pathways")

    # Also load existing Hallmark pathway scores
    logger.info("Loading cached LINCS Hallmark pathway scores ...")
    hallmark_path = DATA_CACHE / "lincs_pathway_MSigDB_Hallmark_2020.parquet"
    if hallmark_path.exists():
        lincs_hallmark = pd.read_parquet(hallmark_path)
    else:
        logger.info("Hallmark cache not found, computing ...")
        from src.features.pathway_features import transform_lincs_to_pathways
        lincs_hallmark = transform_lincs_to_pathways(lincs_df=lincs_sigs)

    # ------------------------------------------------------------------ #
    # Step 2: Transform CTR-DB to PROGENy scores                         #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Transforming CTR-DB datasets to PROGENy scores")
    logger.info("=" * 60)

    ctrdb_progeny = transform_ctrdb_to_progeny(
        ctrdb_datasets=ctrdb_gene, top=500
    )

    # Load existing Hallmark pathway scores for CTR-DB
    logger.info("Loading cached CTR-DB Hallmark pathway scores ...")
    from src.features.pathway_features import transform_ctrdb_to_pathways
    ctrdb_hallmark = transform_ctrdb_to_pathways(ctrdb_datasets=ctrdb_gene)

    # ------------------------------------------------------------------ #
    # Step 3: Transform TCGA-BRCA to PROGENy scores                      #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Transforming TCGA-BRCA expression to PROGENy scores")
    logger.info("=" * 60)

    try:
        tcga_progeny = transform_tcga_to_progeny(top=500)
        logger.info(
            f"  TCGA-BRCA PROGENy scores: {tcga_progeny.shape[0]} samples x "
            f"{tcga_progeny.shape[1]} pathways"
        )
    except FileNotFoundError as e:
        logger.warning(f"  TCGA-BRCA expression not available: {e}")
        tcga_progeny = None

    # ------------------------------------------------------------------ #
    # Step 4: LODO ablation                                              #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info(
        "Step 4: LODO ablation "
        "(Hallmark vs PROGENy vs combined vs drug-specific)"
    )
    logger.info("=" * 60)

    ablation_results = run_progeny_ablation(
        ctrdb_gene=ctrdb_gene,
        ctrdb_hallmark=ctrdb_hallmark,
        ctrdb_progeny=ctrdb_progeny,
        lincs_sigs=lincs_sigs,
        lincs_hallmark=lincs_hallmark,
        lincs_progeny=lincs_progeny,
        catalog=catalog,
        C=0.05,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Save results                                               #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Saving results")
    logger.info("=" * 60)

    RESULTS.mkdir(parents=True, exist_ok=True)

    if not ablation_results.empty:
        # Save detailed per-dataset results
        detail_path = RESULTS / "ablation_progeny_features.tsv"
        ablation_results.to_csv(detail_path, sep="\t", index=False)
        logger.info(f"  Saved detailed ablation results to {detail_path}")

        # Compute summary across datasets
        summary = (
            ablation_results
            .groupby("feature_set")["auc"]
            .agg(["mean", "std", "median", "count"])
            .round(4)
            .reset_index()
        )
        summary.columns = [
            "feature_set", "mean_auc", "std_auc", "median_auc", "n_datasets"
        ]
        summary = summary.sort_values("feature_set").reset_index(drop=True)

        summary_path = RESULTS / "ablation_progeny_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        logger.info(f"  Saved summary to {summary_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("ABLATION RESULTS SUMMARY")
        logger.info("=" * 60)
        for _, row in summary.iterrows():
            logger.info(
                f"  {row['feature_set']:30s}: "
                f"AUC = {row['mean_auc']:.4f} +/- {row['std_auc']:.4f} "
                f"(median={row['median_auc']:.4f}, n={int(row['n_datasets'])})"
            )
    else:
        logger.warning("No ablation results to save")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal pipeline time: {elapsed:.0f}s")


if __name__ == "__main__":
    run_progeny_pipeline()
