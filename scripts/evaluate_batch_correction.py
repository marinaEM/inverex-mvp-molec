"""
Evaluate cross-platform batch correction methods for the INVEREX pipeline.

For each batch correction method:
1. Load all CTR-DB datasets (breast + pan-cancer on disk).
2. Load LINCS drug signatures.
3. Match CTR-DB drugs to LINCS drugs.
4. Apply the batch correction method.
5. Run LODO cross-validation with L1-logistic regression on reversal features.
6. Report per-held-out-dataset AUC and mean AUC.

Usage:
    pixi run python scripts/evaluate_batch_correction.py
"""

import json
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_CACHE, DATA_RAW, DATA_PROCESSED, RESULTS
from src.preprocessing.batch_correction import (
    BATCH_METHODS,
    apply_batch_correction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("batch_eval")


# ---------------------------------------------------------------------------
# Drug name parsing and matching (from recalibrate_signatures.py)
# ---------------------------------------------------------------------------

DRUG_ALIASES = {
    "5-fluorouracil": "fluorouracil",
    "5-fu": "fluorouracil",
    "5fu": "fluorouracil",
    "adriamycin": "doxorubicin",
    "taxol": "paclitaxel",
    "nolvadex": "tamoxifen",
    "xeloda": "capecitabine",
    "taxotere": "docetaxel",
    "gemzar": "gemcitabine",
    "mtx": "methotrexate",
    "ctx": "cyclophosphamide",
    "cytoxan": "cyclophosphamide",
    "epirubicin": "epirubicin",
    "ixabepilone": "ixabepilone",
    "pegfilgrastim": "pegfilgrastim",
    "mk-2206": "MK-2206",
    "trastuzumab": "trastuzumab",
}


def _normalise_drug_name(name: str) -> str:
    """Lower-case, strip hyphens/spaces, apply aliases."""
    s = name.lower().strip()
    s = re.sub(r"[\s\-]+", "", s)
    for alias, canonical in DRUG_ALIASES.items():
        if s == re.sub(r"[\s\-]+", "", alias):
            return canonical.lower()
    return s


def parse_regimen_components(drug_string: str) -> list[str]:
    """Parse a CTR-DB drug string into individual component names."""
    s = drug_string.strip()
    paren_match = re.search(r"\(([^)]+)\)", s)
    if paren_match:
        inner = paren_match.group(1)
    else:
        inner = s
    parts = re.split(r"[+/]", inner)
    components = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^[A-Z]{1,5}\s+", "", p)
        norm = _normalise_drug_name(p)
        if norm and len(norm) > 1:
            components.append(norm)
    return list(dict.fromkeys(components))


def match_drugs_to_lincs(
    components: list[str],
    lincs_drug_set: set[str],
) -> list[str]:
    """Return the subset of components found in LINCS (normalised)."""
    lincs_norm = {_normalise_drug_name(d): d for d in lincs_drug_set}
    matched = []
    for comp in components:
        cn = _normalise_drug_name(comp)
        if cn in lincs_norm:
            matched.append(lincs_norm[cn])
    return matched


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all_datasets() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.Series],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load all CTR-DB datasets from disk (breast + pan-cancer).

    Returns:
        expr_datasets: geo_id -> expression DataFrame
        label_datasets: geo_id -> binary response Series
        catalog: breast catalog DataFrame
        pan_catalog: pan-cancer catalog DataFrame
    """
    ctrdb_dir = DATA_RAW / "ctrdb"

    # Load catalogs
    cat_path = ctrdb_dir / "catalog.csv"
    pan_cat_path = ctrdb_dir / "pan_cancer_catalog.csv"

    catalog = pd.read_csv(cat_path) if cat_path.exists() else pd.DataFrame()
    pan_catalog = pd.read_csv(pan_cat_path) if pan_cat_path.exists() else pd.DataFrame()

    expr_datasets = {}
    label_datasets = {}

    for ds_dir in sorted(ctrdb_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        geo_id = ds_dir.name
        if not geo_id.startswith("GSE"):
            continue

        expr_files = list(ds_dir.glob("*_expression.parquet"))
        label_file = ds_dir / "response_labels.parquet"

        if not expr_files or not label_file.exists():
            continue

        expr = pd.read_parquet(expr_files[0])
        labels = pd.read_parquet(label_file)["response"]

        common_samples = expr.index.intersection(labels.index)
        if len(common_samples) < 5:
            continue

        expr_datasets[geo_id] = expr.loc[common_samples]
        label_datasets[geo_id] = labels.loc[common_samples]

    logger.info(f"Loaded {len(expr_datasets)} CTR-DB datasets from disk")
    return expr_datasets, label_datasets, catalog, pan_catalog


def load_lincs_signatures() -> tuple[pd.DataFrame, list[str], set[str]]:
    """
    Load LINCS drug signatures and landmark genes.

    Returns:
        lincs_sigs: DataFrame of signatures
        gene_cols: list of gene column names
        drug_set: set of drug names in LINCS
    """
    sigs = pd.read_parquet(DATA_CACHE / "all_cellline_drug_signatures.parquet")
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
    gene_cols = [c for c in sigs.columns if c not in meta_cols]
    drug_set = set(sigs["pert_iname"].str.lower().unique())

    logger.info(f"LINCS: {len(sigs)} sigs, {len(gene_cols)} genes, {len(drug_set)} drugs")
    return sigs, gene_cols, drug_set


def load_landmark_genes() -> list[str]:
    """Load landmark gene symbols."""
    gi = pd.read_csv(DATA_CACHE / "geneinfo_beta_input.txt", sep="\t")
    genes = gi["gene_symbol"].tolist()
    logger.info(f"Landmark genes: {len(genes)}")
    return genes


def build_drug_mean_signatures(
    lincs_sigs: pd.DataFrame,
    gene_cols: list[str],
) -> dict[str, pd.Series]:
    """Build per-drug mean signatures (averaged over all cell lines/doses)."""
    drug_sigs = {}
    for drug, grp in lincs_sigs.groupby(lincs_sigs["pert_iname"].str.lower()):
        drug_sigs[drug] = grp[gene_cols].mean(axis=0)
    logger.info(f"Built mean signatures for {len(drug_sigs)} drugs")
    return drug_sigs


def get_drug_for_dataset(
    geo_id: str,
    catalog: pd.DataFrame,
    pan_catalog: pd.DataFrame,
) -> str:
    """Return the drug string for a CTR-DB dataset from catalogs."""
    for cat in [catalog, pan_catalog]:
        if cat.empty:
            continue
        row = cat[cat["geo_source"] == geo_id]
        if not row.empty:
            return str(row.iloc[0]["drug"])
    return ""


# ---------------------------------------------------------------------------
# LODO evaluation
# ---------------------------------------------------------------------------

def run_lodo_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_ids: pd.Series,
    common_genes: list[str],
    drug_mean_sigs: dict[str, pd.Series],
    lincs_drug_set: set[str],
    catalog: pd.DataFrame,
    pan_catalog: pd.DataFrame,
    method_name: str,
    min_patients: int = 20,
    C: float = 0.05,
) -> pd.DataFrame:
    """
    Run leave-one-dataset-out cross-validation using reversal features.

    For each held-out dataset:
    1. Get the drug(s) for that dataset.
    2. Match to LINCS drug signatures.
    3. Compute reversal features: patient_zscore * drug_avg_signature.
    4. Train L1-logistic regression on all OTHER datasets.
    5. Predict on the held-out dataset.
    6. Compute AUC.
    """
    unique_datasets = dataset_ids.unique()
    results = []

    # Pre-compute per-dataset drug signatures aligned to common_genes
    dataset_drug_sigs = {}
    for geo_id in unique_datasets:
        drug_str = get_drug_for_dataset(geo_id, catalog, pan_catalog)
        if not drug_str:
            continue
        components = parse_regimen_components(drug_str)
        matched = match_drugs_to_lincs(components, lincs_drug_set)
        if not matched:
            continue

        # Average matched drug signatures
        sig_parts = []
        for d in matched:
            d_lower = d.lower()
            if d_lower in drug_mean_sigs:
                sig = drug_mean_sigs[d_lower].reindex(common_genes).values.astype(np.float64)
                sig_parts.append(np.nan_to_num(sig, 0.0))
        if sig_parts:
            dataset_drug_sigs[geo_id] = np.mean(sig_parts, axis=0)

    logger.info(
        f"[{method_name}] Drug signatures matched for "
        f"{len(dataset_drug_sigs)}/{len(unique_datasets)} datasets"
    )

    # Pre-compute reversal features for all datasets
    # reversal_features[geo_id] = (X_rev, y_rev)
    reversal_data = {}
    for geo_id in unique_datasets:
        if geo_id not in dataset_drug_sigs:
            continue
        mask = dataset_ids == geo_id
        X_ds = X[mask].values
        y_ds = y[mask].values
        drug_sig = dataset_drug_sigs[geo_id]

        # Reversal features: element-wise product
        X_rev = X_ds * drug_sig[np.newaxis, :]
        reversal_data[geo_id] = (X_rev, y_ds)

    # LODO: for each dataset, train on all others, test on held-out
    for held_out in unique_datasets:
        if held_out not in reversal_data:
            continue

        X_test, y_test = reversal_data[held_out]
        n_test = len(y_test)

        if n_test < min_patients:
            continue
        if len(np.unique(y_test)) < 2:
            continue
        n_pos = int(y_test.sum())
        n_neg = n_test - n_pos
        if n_pos < 3 or n_neg < 3:
            continue

        # Pool training data from all other datasets
        X_train_parts = []
        y_train_parts = []
        for geo_id, (X_rev, y_rev) in reversal_data.items():
            if geo_id == held_out:
                continue
            X_train_parts.append(X_rev)
            y_train_parts.append(y_rev)

        if not X_train_parts:
            continue

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)

        if len(np.unique(y_train)) < 2:
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_train_s = np.nan_to_num(X_train_s, 0.0)
        X_test_s = np.nan_to_num(X_test_s, 0.0)

        # Train L1-logistic regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(
                penalty="l1",
                C=C,
                solver="liblinear",
                max_iter=2000,
                random_state=42,
                class_weight="balanced",
            )
            clf.fit(X_train_s, y_train)

        # Predict
        y_prob = clf.predict_proba(X_test_s)
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob[:, 0]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        drug_str = get_drug_for_dataset(held_out, catalog, pan_catalog)
        n_nonzero = int((np.abs(clf.coef_[0]) > 1e-8).sum())

        results.append({
            "method": method_name,
            "dataset": held_out,
            "drug": drug_str,
            "n_patients": n_test,
            "n_responders": n_pos,
            "n_nonresponders": n_neg,
            "auc": round(auc, 4),
            "n_nonzero_genes": n_nonzero,
            "n_train": len(y_train),
        })

        logger.info(
            f"  [{method_name}] {held_out}: AUC={auc:.3f} "
            f"(n={n_test}, R={n_pos}, NR={n_neg}, "
            f"nonzero={n_nonzero}, train={len(y_train)})"
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("BATCH CORRECTION EVALUATION FOR INVEREX PIPELINE")
    logger.info("=" * 70)

    # Load data
    expr_datasets, label_datasets, catalog, pan_catalog = load_all_datasets()
    lincs_sigs, gene_cols, lincs_drug_set = load_lincs_signatures()
    landmark_genes = load_landmark_genes()
    drug_mean_sigs = build_drug_mean_signatures(lincs_sigs, gene_cols)

    # Filter datasets with >= 20 patients
    MIN_PATIENTS = 20
    valid_datasets = {}
    valid_labels = {}
    for geo_id in expr_datasets:
        if geo_id in label_datasets and len(label_datasets[geo_id]) >= MIN_PATIENTS:
            valid_datasets[geo_id] = expr_datasets[geo_id]
            valid_labels[geo_id] = label_datasets[geo_id]

    logger.info(f"Datasets with >= {MIN_PATIENTS} patients: {len(valid_datasets)}")

    # Run evaluation for each batch correction method
    all_results = []
    methods = list(BATCH_METHODS.keys())

    for method_name in methods:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"METHOD: {method_name}")
        logger.info("=" * 70)

        try:
            X, y_labels, ds_ids, common_genes = apply_batch_correction(
                method_name=method_name,
                datasets=valid_datasets,
                labels=valid_labels,
                landmark_genes=landmark_genes,
            )

            if X.empty:
                logger.warning(f"[{method_name}] Empty result -- skipping")
                continue

            logger.info(
                f"[{method_name}] Matrix: {X.shape[0]} patients x {X.shape[1]} genes, "
                f"{ds_ids.nunique()} datasets"
            )

            results = run_lodo_evaluation(
                X=X,
                y=y_labels,
                dataset_ids=ds_ids,
                common_genes=common_genes,
                drug_mean_sigs=drug_mean_sigs,
                lincs_drug_set=lincs_drug_set,
                catalog=catalog,
                pan_catalog=pan_catalog,
                method_name=method_name,
                min_patients=MIN_PATIENTS,
            )

            if not results.empty:
                all_results.append(results)

        except Exception as exc:
            logger.error(f"[{method_name}] Failed: {exc}", exc_info=True)
            continue

    if not all_results:
        logger.error("No successful methods -- aborting")
        return

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Save per-dataset results
    results_path = RESULTS / "batch_correction_comparison.csv"
    combined.to_csv(results_path, index=False)
    logger.info(f"\nPer-dataset results saved to {results_path}")

    # Build summary
    summary_rows = []
    for method in combined["method"].unique():
        method_results = combined[combined["method"] == method]
        mean_auc = method_results["auc"].mean()
        median_auc = method_results["auc"].median()
        std_auc = method_results["auc"].std()
        n_datasets = len(method_results)

        # How many improved vs baseline (per_dataset_zscore)?
        baseline = combined[combined["method"] == "per_dataset_zscore"]
        if not baseline.empty and method != "per_dataset_zscore":
            common_ds = set(method_results["dataset"]) & set(baseline["dataset"])
            n_improved = 0
            n_worsened = 0
            for ds in common_ds:
                m_auc = method_results.loc[method_results["dataset"] == ds, "auc"].values[0]
                b_auc = baseline.loc[baseline["dataset"] == ds, "auc"].values[0]
                if m_auc > b_auc + 0.001:
                    n_improved += 1
                elif m_auc < b_auc - 0.001:
                    n_worsened += 1
        else:
            n_improved = 0
            n_worsened = 0

        summary_rows.append({
            "method": method,
            "mean_auc": round(mean_auc, 4),
            "median_auc": round(median_auc, 4),
            "std_auc": round(std_auc, 4),
            "n_datasets": n_datasets,
            "n_improved": n_improved,
            "n_worsened": n_worsened,
        })

    summary = pd.DataFrame(summary_rows).sort_values("mean_auc", ascending=False)
    summary_path = RESULTS / "batch_correction_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("BATCH CORRECTION COMPARISON")
    logger.info("=" * 70)
    logger.info(f"\n{'Method':<25} {'Mean AUC':>10} {'Median':>10} {'Std':>8} {'N_ds':>6} {'Improved':>10} {'Worsened':>10}")
    logger.info("-" * 90)
    for _, row in summary.iterrows():
        logger.info(
            f"{row['method']:<25} {row['mean_auc']:>10.4f} {row['median_auc']:>10.4f} "
            f"{row['std_auc']:>8.4f} {row['n_datasets']:>6} "
            f"{row['n_improved']:>10} {row['n_worsened']:>10}"
        )

    # Pick the winner
    best = summary.iloc[0]
    logger.info(f"\nWINNER: {best['method']} (mean AUC = {best['mean_auc']:.4f})")

    # Save the best method's corrected expression
    best_method = best["method"]
    logger.info(f"\nRe-running {best_method} to save corrected expression ...")

    X_best, y_best, ds_best, genes_best = apply_batch_correction(
        method_name=best_method,
        datasets=valid_datasets,
        labels=valid_labels,
        landmark_genes=landmark_genes,
    )

    output_path = DATA_PROCESSED / "batch_corrected_expression.parquet"
    X_best.to_parquet(output_path)
    logger.info(f"Best corrected expression saved to {output_path}")

    # Also save the labels and dataset IDs alongside
    meta = pd.DataFrame({
        "response": y_best,
        "dataset_id": ds_best,
    })
    meta.to_parquet(DATA_PROCESSED / "batch_corrected_meta.parquet")
    logger.info(f"Metadata saved to {DATA_PROCESSED / 'batch_corrected_meta.parquet'}")

    # Per-dataset AUC table for the winning method
    logger.info(f"\n{'='*70}")
    logger.info(f"PER-DATASET AUC FOR WINNER: {best_method}")
    logger.info(f"{'='*70}")
    winner_results = combined[combined["method"] == best_method].sort_values("auc", ascending=False)
    logger.info(f"\n{'Dataset':<15} {'Drug':<55} {'N':>5} {'AUC':>7}")
    logger.info("-" * 90)
    for _, row in winner_results.iterrows():
        drug_short = row["drug"][:52] + "..." if len(str(row["drug"])) > 55 else row["drug"]
        logger.info(f"{row['dataset']:<15} {drug_short:<55} {row['n_patients']:>5} {row['auc']:>7.3f}")

    logger.info(f"\nMean AUC: {winner_results['auc'].mean():.4f}")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
