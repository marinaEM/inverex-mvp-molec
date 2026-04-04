"""
Dose-aware LINCS L1000 signatures and LODO evaluation.

Instead of averaging all LINCS signatures for a drug across doses, we compute
separate signatures for low / medium / high dose bins, then derive
dose-stratified reversal features for CTR-DB patient response prediction.

Three modelling approaches are compared via Leave-One-Dataset-Out (LODO):
  1. Dose-averaged reversal  (single reversal score per drug — current baseline)
  2. Dose-stratified reversal (3 features per drug: low/medium/high)
  3. Dose-stratified + slope  (5 features per drug: low/medium/high + slope + max)
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Dose bins ──────────────────────────────────────────────────────────────
DOSE_BINS = {
    "low":    (0, 0.5),
    "medium": (0.5, 5.0),
    "high":   (5.0, float('inf')),
}

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CACHE = ROOT / "data" / "cache"
DATA_RAW = ROOT / "data" / "raw"
CTRDB_DIR = DATA_RAW / "ctrdb"
RESULTS_DIR = ROOT / "results"


# =====================================================================
# 1. Load LINCS data
# =====================================================================

META_COLS = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}


def load_lincs_signatures(path: Path | None = None) -> pd.DataFrame:
    """Load the all-cell-line LINCS signature matrix."""
    if path is None:
        path = DATA_CACHE / "all_cellline_drug_signatures.parquet"
    df = pd.read_parquet(path)
    logger.info(
        "Loaded LINCS signatures: %d sigs, %d drugs, %d cell lines",
        len(df), df["pert_iname"].nunique(), df["cell_id"].nunique(),
    )
    return df


def gene_columns(df: pd.DataFrame) -> list[str]:
    """Return the gene z-score column names (everything not in META_COLS)."""
    return [c for c in df.columns if c not in META_COLS]


# =====================================================================
# 2. Compute dose-stratified average signatures
# =====================================================================

def assign_dose_bin(dose_um: float) -> str | None:
    """Map a numeric dose (µM) to a bin label."""
    for label, (lo, hi) in DOSE_BINS.items():
        if lo < dose_um <= hi:
            return label
    # dose_um == 0 is unlikely but handle it
    if dose_um == 0:
        return "low"
    return None


def compute_dose_stratified_signatures(
    lincs: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each drug, compute the mean gene z-score at each dose bin.

    Returns a DataFrame with columns:
        pert_iname, dose_bin, <gene_cols...>
    """
    genes = gene_columns(lincs)
    lincs = lincs.copy()
    lincs["dose_bin"] = lincs["dose_um"].apply(assign_dose_bin)
    lincs = lincs.dropna(subset=["dose_bin"])

    grouped = lincs.groupby(["pert_iname", "dose_bin"])[genes].mean()
    result = grouped.reset_index()

    n_drugs = result["pert_iname"].nunique()
    bin_counts = result["dose_bin"].value_counts().to_dict()
    logger.info(
        "Dose-stratified signatures: %d drug-bin pairs across %d drugs. "
        "Bins: %s",
        len(result), n_drugs, bin_counts,
    )
    return result


def compute_dose_averaged_signatures(
    lincs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Baseline: average all signatures for each drug regardless of dose.

    Returns a DataFrame with columns: pert_iname, <gene_cols...>
    """
    genes = gene_columns(lincs)
    result = lincs.groupby("pert_iname")[genes].mean().reset_index()
    logger.info("Dose-averaged signatures: %d drugs", len(result))
    return result


# =====================================================================
# 3. CTR-DB loading and drug matching
# =====================================================================

def _normalize_drug_name(name: str) -> str:
    """Lowercase, strip whitespace, remove hyphens for fuzzy matching."""
    return re.sub(r"[\s\-_]+", "", name.strip().lower())


def _parse_combination_drugs(drug_string: str) -> list[str]:
    """
    Parse a CTR-DB combination drug string into individual drug names.

    Examples:
        "TFAC (Cyclophosphamide+Doxorubicin+Fluorouracil+Paclitaxel)"
          -> ["Cyclophosphamide", "Doxorubicin", "Fluorouracil", "Paclitaxel"]
        "Anthracycline+Taxane"
          -> ["Anthracycline", "Taxane"]
        "AC (Cyclophosphamide+Doxorubicin)+Ixabepilone"
          -> ["Cyclophosphamide", "Doxorubicin", "Ixabepilone"]
    """
    # Remove abbreviation prefixes like "TFAC (...)", "AC (...)", "FAC (...)"
    # Extract content in parentheses and anything outside
    parts = []

    # Find all parenthesised groups and replace them inline
    paren_re = re.compile(r"\([^)]+\)")
    parens = paren_re.findall(drug_string)
    remaining = paren_re.sub("", drug_string).strip()

    # From parentheses: split on "+"
    for p in parens:
        inner = p.strip("()")
        parts.extend([x.strip() for x in inner.split("+") if x.strip()])

    # From remaining text after removing abbreviation labels
    # e.g. "TFAC   +Ixabepilone" -> ["Ixabepilone"]
    # Remove leading abbreviation tokens (all-caps or short)
    for token in re.split(r"\+|/", remaining):
        token = token.strip()
        if not token:
            continue
        # Skip abbreviation-only tokens like "TFAC", "AC", "FAC", "AT", "CMF"
        if re.match(r"^[A-Z]{1,6}$", token):
            continue
        parts.append(token)

    if not parts:
        # Fallback: split the entire string on "+"
        parts = [x.strip() for x in drug_string.split("+") if x.strip()]

    return parts


# Drug-class-to-specific-drug mapping for generic class names
_CLASS_TO_DRUGS = {
    "anthracycline": ["doxorubicin", "epirubicin", "daunorubicin"],
    "taxane":        ["paclitaxel", "docetaxel"],
    "platinum":      ["cisplatin", "carboplatin", "oxaliplatin"],
    "glucocorticoids": ["dexamethasone", "prednisone", "prednisolone"],
}


def match_drugs_to_lincs(
    drug_string: str,
    lincs_drug_names: set[str],
) -> list[str]:
    """
    Given a CTR-DB drug string, return the list of LINCS pert_iname values
    that match any individual component.
    """
    components = _parse_combination_drugs(drug_string)
    lincs_norm = {_normalize_drug_name(n): n for n in lincs_drug_names}

    matched = []
    for comp in components:
        comp_norm = _normalize_drug_name(comp)

        # Direct match
        if comp_norm in lincs_norm:
            matched.append(lincs_norm[comp_norm])
            continue

        # Class expansion
        if comp_norm in _CLASS_TO_DRUGS:
            for specific in _CLASS_TO_DRUGS[comp_norm]:
                specific_norm = _normalize_drug_name(specific)
                if specific_norm in lincs_norm:
                    matched.append(lincs_norm[specific_norm])
            continue

        # Substring match (e.g. "fluorouracil" in LINCS as "fluorouracil")
        for ln, orig in lincs_norm.items():
            if comp_norm in ln or ln in comp_norm:
                matched.append(orig)
                break

    return list(dict.fromkeys(matched))  # deduplicate preserving order


def load_ctrdb_datasets() -> list[dict]:
    """
    Load all available CTR-DB datasets (expression + response labels).

    Returns a list of dicts with keys:
        gse_id, drug_string, expression (DataFrame), response (Series 0/1)
    """
    # Build drug mapping from both catalogs
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

        # Find expression and response files
        expr_path = gse_dir / f"{gse_id}_expression.parquet"
        resp_path = gse_dir / "response_labels.parquet"
        if not expr_path.exists() or not resp_path.exists():
            continue

        drug_string = gse_drug_map.get(gse_id)
        if drug_string is None:
            logger.debug("Skipping %s — no drug annotation in catalog", gse_id)
            continue

        try:
            expression = pd.read_parquet(expr_path)
            response = pd.read_parquet(resp_path).squeeze()

            # Align samples
            common = expression.index.intersection(response.index)
            if len(common) < 10:
                logger.debug(
                    "Skipping %s — only %d common samples", gse_id, len(common)
                )
                continue

            expression = expression.loc[common]
            response = response.loc[common]

            datasets.append({
                "gse_id": gse_id,
                "drug_string": drug_string,
                "expression": expression,
                "response": response,
            })
            logger.info(
                "Loaded %s: %d samples, drug=%s, responders=%d/%d",
                gse_id, len(common), drug_string,
                int(response.sum()), len(response),
            )
        except Exception as e:
            logger.warning("Failed to load %s: %s", gse_id, e)

    logger.info("Loaded %d CTR-DB datasets with drug annotations", len(datasets))
    return datasets


# =====================================================================
# 4. Reversal score computation
# =====================================================================

def _reversal_score(
    patient_profile: np.ndarray,
    drug_signature: np.ndarray,
) -> float:
    """
    Compute reversal score as negative Pearson correlation.

    A high reversal score means the drug signature *reverses* the
    patient's expression profile (anti-correlated).
    """
    mask = np.isfinite(patient_profile) & np.isfinite(drug_signature)
    if mask.sum() < 10:
        return np.nan
    r, _ = pearsonr(patient_profile[mask], drug_signature[mask])
    return -r  # negate so that anti-correlation -> positive reversal


def compute_reversal_features_averaged(
    patient_expr: pd.DataFrame,
    avg_sigs: pd.DataFrame,
    drugs: list[str],
) -> pd.DataFrame:
    """
    Baseline: one reversal score per drug (dose-averaged signature).

    Returns DataFrame with shape (n_patients, n_drugs).
    """
    genes = [c for c in avg_sigs.columns if c != "pert_iname"]
    common_genes = sorted(set(genes) & set(patient_expr.columns))
    if not common_genes:
        logger.warning("No overlapping genes between patient and LINCS data")
        return pd.DataFrame(index=patient_expr.index)

    patient_mat = patient_expr[common_genes].values
    # z-score patient expression (per gene across patients)
    patient_mean = np.nanmean(patient_mat, axis=0)
    patient_std = np.nanstd(patient_mat, axis=0)
    patient_std[patient_std == 0] = 1
    patient_z = (patient_mat - patient_mean) / patient_std

    features = {}
    sig_lookup = avg_sigs.set_index("pert_iname")
    for drug in drugs:
        if drug not in sig_lookup.index:
            continue
        drug_sig = sig_lookup.loc[drug, common_genes].values.astype(float)
        scores = np.array([
            _reversal_score(patient_z[i], drug_sig)
            for i in range(len(patient_z))
        ])
        features[f"reversal_{drug}"] = scores

    return pd.DataFrame(features, index=patient_expr.index)


def compute_reversal_features_stratified(
    patient_expr: pd.DataFrame,
    strat_sigs: pd.DataFrame,
    drugs: list[str],
    include_slope: bool = False,
) -> pd.DataFrame:
    """
    Dose-stratified reversal features.

    For each drug, compute reversal_low, reversal_medium, reversal_high.
    If include_slope=True, also add reversal_slope and reversal_max.

    Returns DataFrame with shape (n_patients, n_drugs * {3 or 5}).
    """
    genes = [c for c in strat_sigs.columns if c not in {"pert_iname", "dose_bin"}]
    common_genes = sorted(set(genes) & set(patient_expr.columns))
    if not common_genes:
        logger.warning("No overlapping genes between patient and LINCS data")
        return pd.DataFrame(index=patient_expr.index)

    patient_mat = patient_expr[common_genes].values
    patient_mean = np.nanmean(patient_mat, axis=0)
    patient_std = np.nanstd(patient_mat, axis=0)
    patient_std[patient_std == 0] = 1
    patient_z = (patient_mat - patient_mean) / patient_std

    sig_lookup = strat_sigs.set_index(["pert_iname", "dose_bin"])
    dose_levels = ["low", "medium", "high"]
    # numeric positions for slope calculation
    dose_numeric = {"low": 0.25, "medium": 2.75, "high": 7.5}

    features = {}
    for drug in drugs:
        bin_scores: dict[str, np.ndarray] = {}
        for dbin in dose_levels:
            if (drug, dbin) not in sig_lookup.index:
                continue
            drug_sig = sig_lookup.loc[(drug, dbin), common_genes].values.astype(float)
            scores = np.array([
                _reversal_score(patient_z[i], drug_sig)
                for i in range(len(patient_z))
            ])
            features[f"reversal_{dbin}_{drug}"] = scores
            bin_scores[dbin] = scores

        if not bin_scores:
            continue

        if include_slope:
            # reversal_max
            all_scores = np.column_stack(list(bin_scores.values()))
            features[f"reversal_max_{drug}"] = np.nanmax(all_scores, axis=1)

            # reversal_slope: linear fit of reversal vs dose level
            if len(bin_scores) >= 2:
                x_vals = np.array([dose_numeric[b] for b in bin_scores])
                y_mat = np.column_stack([bin_scores[b] for b in bin_scores])
                slopes = np.array([
                    np.polyfit(x_vals, y_mat[i], 1)[0]
                    if np.all(np.isfinite(y_mat[i]))
                    else np.nan
                    for i in range(len(patient_z))
                ])
                features[f"reversal_slope_{drug}"] = slopes
            else:
                features[f"reversal_slope_{drug}"] = np.full(len(patient_z), np.nan)

    return pd.DataFrame(features, index=patient_expr.index)


# =====================================================================
# 5. LODO evaluation
# =====================================================================

def lodo_evaluate(
    datasets: list[dict],
    lincs: pd.DataFrame,
    C: float = 0.05,
) -> pd.DataFrame:
    """
    Leave-One-Dataset-Out evaluation comparing three approaches.

    Returns a DataFrame with columns:
        dataset, approach, auc, n_train, n_test, n_features, matched_drugs
    """
    lincs_drugs = set(lincs["pert_iname"].unique())
    avg_sigs = compute_dose_averaged_signatures(lincs)
    strat_sigs = compute_dose_stratified_signatures(lincs)

    results = []

    for i, test_ds in enumerate(datasets):
        test_gse = test_ds["gse_id"]
        test_drugs = match_drugs_to_lincs(test_ds["drug_string"], lincs_drugs)

        if not test_drugs:
            logger.warning(
                "LODO: %s — no LINCS matches for '%s', skipping",
                test_gse, test_ds["drug_string"],
            )
            continue

        logger.info(
            "LODO fold %d/%d: test=%s, drugs=%s",
            i + 1, len(datasets), test_gse, test_drugs,
        )

        # Train datasets = all except current test
        train_datasets = [d for d in datasets if d["gse_id"] != test_gse]

        for approach in ["dose_averaged", "dose_stratified", "dose_stratified_slope"]:
            try:
                auc, n_train, n_test, n_feats = _run_one_fold(
                    train_datasets, test_ds, lincs_drugs,
                    avg_sigs, strat_sigs, test_drugs, approach, C,
                )
                results.append({
                    "dataset": test_gse,
                    "drug_string": test_ds["drug_string"],
                    "approach": approach,
                    "auc": auc,
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_features": n_feats,
                    "matched_drugs": ", ".join(test_drugs),
                })
                logger.info(
                    "  %s: AUC=%.3f (train=%d, test=%d, feats=%d)",
                    approach, auc, n_train, n_test, n_feats,
                )
            except Exception as e:
                logger.warning("  %s: FAILED — %s", approach, e)
                results.append({
                    "dataset": test_gse,
                    "drug_string": test_ds["drug_string"],
                    "approach": approach,
                    "auc": np.nan,
                    "n_train": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "matched_drugs": ", ".join(test_drugs),
                })

    return pd.DataFrame(results)


def _build_features_for_dataset(
    ds: dict,
    lincs_drugs: set[str],
    avg_sigs: pd.DataFrame,
    strat_sigs: pd.DataFrame,
    target_drugs: list[str],
    approach: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and response for one dataset."""
    expr = ds["expression"]
    resp = ds["response"]

    # The dataset's own drugs may differ from target_drugs,
    # but we use the target_drugs (test-dataset drugs) so features are aligned.
    if approach == "dose_averaged":
        X = compute_reversal_features_averaged(expr, avg_sigs, target_drugs)
    elif approach == "dose_stratified":
        X = compute_reversal_features_stratified(
            expr, strat_sigs, target_drugs, include_slope=False
        )
    elif approach == "dose_stratified_slope":
        X = compute_reversal_features_stratified(
            expr, strat_sigs, target_drugs, include_slope=True
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Drop columns that are all NaN
    X = X.dropna(axis=1, how="all")
    # Fill remaining NaN with 0
    X = X.fillna(0)

    # Align
    common = X.index.intersection(resp.index)
    X = X.loc[common]
    y = resp.loc[common]

    return X, y


def _run_one_fold(
    train_datasets: list[dict],
    test_ds: dict,
    lincs_drugs: set[str],
    avg_sigs: pd.DataFrame,
    strat_sigs: pd.DataFrame,
    test_drugs: list[str],
    approach: str,
    C: float,
) -> tuple[float, int, int, int]:
    """Run one LODO fold: train on all-but-one, test on held-out."""

    # Build train set
    X_train_parts, y_train_parts = [], []
    for ds in train_datasets:
        # Use the test_drugs for feature construction so columns align
        X_ds, y_ds = _build_features_for_dataset(
            ds, lincs_drugs, avg_sigs, strat_sigs, test_drugs, approach,
        )
        if len(X_ds) > 0 and X_ds.shape[1] > 0:
            X_train_parts.append(X_ds)
            y_train_parts.append(y_ds)

    if not X_train_parts:
        raise ValueError("No valid training data")

    X_train = pd.concat(X_train_parts, axis=0)
    y_train = pd.concat(y_train_parts, axis=0)

    # Build test set
    X_test, y_test = _build_features_for_dataset(
        test_ds, lincs_drugs, avg_sigs, strat_sigs, test_drugs, approach,
    )

    if X_test.shape[1] == 0:
        raise ValueError("No features for test set")

    # Align columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    if not common_cols:
        raise ValueError("No common features between train and test")
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Check we have both classes
    if y_train.nunique() < 2:
        raise ValueError("Training set has only one class")
    if y_test.nunique() < 2:
        raise ValueError("Test set has only one class")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # L1 logistic regression
    clf = LogisticRegression(
        penalty="l1", C=C, solver="liblinear",
        random_state=42, max_iter=1000,
    )
    clf.fit(X_train_s, y_train.values)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test.values, y_prob)

    return auc, len(X_train), len(X_test), len(common_cols)


# =====================================================================
# 6. Main entry point
# =====================================================================

def main() -> None:
    """Run the full dose-aware signature analysis and save results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    logger.info("=" * 70)
    logger.info("DOSE-AWARE LINCS SIGNATURE ANALYSIS")
    logger.info("=" * 70)

    # ── Load LINCS ──
    lincs = load_lincs_signatures()
    genes = gene_columns(lincs)
    logger.info("Gene columns: %d", len(genes))
    logger.info(
        "Dose distribution: low=%d, medium=%d, high=%d",
        (lincs["dose_um"] <= 0.5).sum(),
        ((lincs["dose_um"] > 0.5) & (lincs["dose_um"] <= 5.0)).sum(),
        (lincs["dose_um"] > 5.0).sum(),
    )

    # ── Compute signatures ──
    strat_sigs = compute_dose_stratified_signatures(lincs)
    avg_sigs = compute_dose_averaged_signatures(lincs)

    logger.info(
        "Stratified sigs: %d drug-bin combos; Averaged sigs: %d drugs",
        len(strat_sigs), len(avg_sigs),
    )

    # ── Load CTR-DB ──
    datasets = load_ctrdb_datasets()
    if not datasets:
        logger.error("No CTR-DB datasets available. Aborting.")
        return

    # Log drug matching
    lincs_drugs = set(lincs["pert_iname"].unique())
    for ds in datasets:
        matched = match_drugs_to_lincs(ds["drug_string"], lincs_drugs)
        logger.info(
            "  %s: '%s' -> LINCS matches: %s",
            ds["gse_id"], ds["drug_string"],
            matched if matched else "NONE",
        )

    # Filter to datasets with at least one LINCS match
    datasets_with_match = [
        ds for ds in datasets
        if match_drugs_to_lincs(ds["drug_string"], lincs_drugs)
    ]
    logger.info(
        "Datasets with LINCS drug matches: %d / %d",
        len(datasets_with_match), len(datasets),
    )

    if len(datasets_with_match) < 2:
        logger.error(
            "Need at least 2 datasets for LODO. Only have %d. Aborting.",
            len(datasets_with_match),
        )
        return

    # ── LODO evaluation ──
    logger.info("Starting LODO evaluation...")
    results = lodo_evaluate(datasets_with_match, lincs)

    # ── Save detailed results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = RESULTS_DIR / "dose_aware_comparison.csv"
    results.to_csv(detail_path, index=False)
    logger.info("Saved per-dataset results to %s", detail_path)

    # ── Summary ──
    summary = (
        results
        .groupby("approach")["auc"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_auc",
            "std": "std_auc",
            "min": "min_auc",
            "max": "max_auc",
            "count": "n_datasets",
        })
        .sort_values("mean_auc", ascending=False)
    )
    summary_path = RESULTS_DIR / "dose_aware_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    logger.info("\n%s", "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for _, row in summary.iterrows():
        logger.info(
            "  %-30s  mean AUC = %.3f +/- %.3f  (n=%d datasets)",
            row["approach"], row["mean_auc"], row["std_auc"], row["n_datasets"],
        )
    logger.info("=" * 70)

    # Also print to stdout for visibility
    print("\n" + "=" * 70)
    print("DOSE-AWARE SIGNATURE COMPARISON — LODO RESULTS")
    print("=" * 70)
    print(f"\nDetailed results: {detail_path}")
    print(f"Summary:          {summary_path}\n")
    print(summary.to_string(index=False))
    print()

    print("\nPer-dataset breakdown:")
    pivot = results.pivot(
        index="dataset", columns="approach", values="auc"
    )
    print(pivot.to_string())
    print()


if __name__ == "__main__":
    main()
