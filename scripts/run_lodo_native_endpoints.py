#!/usr/bin/env python
"""
LODO evaluation with endpoint-aware metrics and per-family aggregation.

For each CTR-DB dataset:
  1. Load expression + binary response labels.
  2. Look up endpoint family from metadata.
  3. Apply the appropriate harmonization policy.
  4. Compute patient disease signatures (L1000 landmark gene z-scores).
  5. Build reversal features using LINCS drug signatures.
  6. Evaluate with LODO: train on all-but-one, test on held-out.
  7. Use appropriate metrics per endpoint type:
     - Binary (pathologic/radiographic/pharmacodynamic): AUROC, AUPRC
     - Continuous: Spearman correlation
     - Survival: concordance index (not present in current data)

Saves:
  - results/lodo_by_dataset.tsv
  - results/lodo_by_endpoint_family.tsv
  - results/lodo_summary.tsv

Usage:
    pixi run python scripts/run_lodo_native_endpoints.py
"""
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_CACHE, DATA_RAW, RESULTS
from src.preprocessing.response_handler import load_response_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lodo_native_endpoints")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

CTRDB_DIR = DATA_RAW / "ctrdb"
LODO_C = 0.05
META_COLS = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}

# ── Drug name utilities (from run_combined_pipeline.py) ──────────────────

import re

DRUG_ALIASES = {
    "5-fluorouracil": "fluorouracil", "5-fu": "fluorouracil",
    "5fu": "fluorouracil", "adriamycin": "doxorubicin",
    "taxol": "paclitaxel", "nolvadex": "tamoxifen",
    "xeloda": "capecitabine", "taxotere": "docetaxel",
    "gemzar": "gemcitabine", "mtx": "methotrexate",
    "ctx": "cyclophosphamide", "cytoxan": "cyclophosphamide",
    "epirubicin": "epirubicin", "ixabepilone": "ixabepilone",
    "mk-2206": "MK-2206", "trastuzumab": "trastuzumab",
}


def _normalise_drug_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[\s\-]+", "", s)
    for alias, canonical in DRUG_ALIASES.items():
        if s == re.sub(r"[\s\-]+", "", alias):
            return canonical.lower()
    return s


def parse_regimen_components(drug_string: str) -> list[str]:
    s = drug_string.strip()
    paren_match = re.search(r"\(([^)]+)\)", s)
    inner = paren_match.group(1) if paren_match else s
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


def match_drugs_to_lincs(components: list[str], lincs_drug_set: set[str]) -> list[str]:
    lincs_norm = {_normalise_drug_name(d): d for d in lincs_drug_set}
    matched = []
    for comp in components:
        cn = _normalise_drug_name(comp)
        if cn in lincs_norm:
            matched.append(lincs_norm[cn])
    return matched


# ── Data loading ─────────────────────────────────────────────────────────

def load_landmark_genes() -> list[str]:
    gi = pd.read_csv(DATA_CACHE / "geneinfo_beta_input.txt", sep="\t")
    return gi["gene_symbol"].tolist()


def load_lincs_signatures():
    """Load LINCS signatures, preferring all-cellline, falling back to breast."""
    all_path = DATA_CACHE / "all_cellline_drug_signatures.parquet"
    breast_path = DATA_CACHE / "breast_l1000_signatures.parquet"

    if all_path.exists():
        sigs = pd.read_parquet(all_path)
        logger.info(f"Loaded all-cellline LINCS: {len(sigs)} signatures")
    elif breast_path.exists():
        sigs = pd.read_parquet(breast_path)
        logger.info(f"Loaded breast LINCS: {len(sigs)} signatures")
    else:
        logger.error("No LINCS signatures found in data/cache/")
        return None, [], set()

    gene_cols = [c for c in sigs.columns if c not in META_COLS]
    drug_set = set(sigs["pert_iname"].str.lower().unique())
    return sigs, gene_cols, drug_set


def load_all_datasets(handler: "ResponseHandler"):
    """Load all CTR-DB datasets with endpoint-aware harmonization."""
    from src.preprocessing.response_handler import ResponseHandler

    cat_path = CTRDB_DIR / "catalog.csv"
    pan_cat_path = CTRDB_DIR / "pan_cancer_catalog.csv"
    catalog = pd.read_csv(cat_path) if cat_path.exists() else pd.DataFrame()
    pan_catalog = pd.read_csv(pan_cat_path) if pan_cat_path.exists() else pd.DataFrame()

    expr_datasets = {}
    label_datasets = {}
    endpoint_families = {}

    for ds_dir in sorted(CTRDB_DIR.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("GSE"):
            continue
        geo_id = ds_dir.name
        expr_files = list(ds_dir.glob("*_expression.parquet"))
        label_file = ds_dir / "response_labels.parquet"
        if not expr_files or not label_file.exists():
            continue

        expr = pd.read_parquet(expr_files[0])
        labels = pd.read_parquet(label_file)["response"]
        common_samples = expr.index.intersection(labels.index)
        if len(common_samples) < 5:
            continue

        expr = expr.loc[common_samples]
        labels = labels.loc[common_samples]

        # Harmonize labels
        ef = handler.get_endpoint_family(geo_id)
        harmonized = handler.harmonize_labels(geo_id, labels, endpoint_family=ef)
        if harmonized is None:
            logger.info(f"Skipping {geo_id}: excluded by {handler.policy} policy")
            continue

        expr_datasets[geo_id] = expr
        label_datasets[geo_id] = harmonized
        endpoint_families[geo_id] = ef

    logger.info(f"Loaded {len(expr_datasets)} datasets after harmonization")
    return expr_datasets, label_datasets, endpoint_families, catalog, pan_catalog


def get_drug_for_dataset(geo_id, catalog, pan_catalog):
    for cat in [catalog, pan_catalog]:
        if cat.empty:
            continue
        row = cat[cat["geo_source"] == geo_id]
        if not row.empty:
            return str(row.iloc[0]["drug"])
    return ""


# ── Feature building ─────────────────────────────────────────────────────

def build_reversal_features(
    expr_datasets, label_datasets, endpoint_families,
    lincs_sigs, lincs_gene_cols, lincs_drug_set,
    catalog, pan_catalog, landmark_genes,
):
    """Build per-dataset reversal feature matrices."""
    # Compute per-drug mean gene signatures from LINCS
    drug_mean_gene = {}
    for drug, grp in lincs_sigs.groupby(lincs_sigs["pert_iname"].str.lower()):
        drug_mean_gene[drug] = grp[lincs_gene_cols].mean(axis=0)

    # Z-score patients within each dataset, restricted to landmark genes
    dataset_features = {}

    for geo_id in expr_datasets:
        expr = expr_datasets[geo_id]
        labels = label_datasets[geo_id]
        ef = endpoint_families[geo_id]

        available_genes = [g for g in landmark_genes if g in expr.columns]
        if len(available_genes) < 50:
            logger.warning(f"Skipping {geo_id}: only {len(available_genes)} landmark genes")
            continue

        # Z-score within dataset
        expr_lm = expr[available_genes].copy()
        cohort_mean = expr_lm.mean(axis=0)
        cohort_std = expr_lm.std(axis=0).replace(0, 1)
        expr_z = (expr_lm - cohort_mean) / cohort_std

        # Match drugs
        drug_str = get_drug_for_dataset(geo_id, catalog, pan_catalog)
        if not drug_str:
            continue
        components = parse_regimen_components(drug_str)
        matched_drugs = match_drugs_to_lincs(components, lincs_drug_set)
        if not matched_drugs:
            logger.info(f"  {geo_id}: no LINCS drugs matched for '{drug_str}'")
            continue

        # Compute drug signature (mean over matched drugs)
        drug_sig = np.zeros(len(available_genes), dtype=np.float64)
        n_d = 0
        for d in matched_drugs:
            dl = d.lower()
            if dl in drug_mean_gene:
                sig = drug_mean_gene[dl].reindex(available_genes).values.astype(np.float64)
                drug_sig += np.nan_to_num(sig, 0.0)
                n_d += 1
        if n_d > 0:
            drug_sig /= n_d
        else:
            continue

        # Reversal features: element-wise product
        X_rev = expr_z.values * drug_sig[np.newaxis, :]

        y = labels.values.astype(int)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        if n_pos < 3 or n_neg < 3:
            logger.info(f"Skipping {geo_id}: too few in one class (R={n_pos}, NR={n_neg})")
            continue

        dataset_features[geo_id] = {
            "X": X_rev,
            "y": y,
            "drug": drug_str,
            "endpoint_family": ef,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_genes": len(available_genes),
        }

    logger.info(f"Built features for {len(dataset_features)} datasets")
    return dataset_features


# ── LODO evaluation ──────────────────────────────────────────────────────

def run_lodo(dataset_features: dict) -> pd.DataFrame:
    """Run leave-one-dataset-out evaluation with endpoint-aware metrics."""
    all_geos = sorted(dataset_features.keys())
    if len(all_geos) < 2:
        logger.error("Need >= 2 datasets for LODO")
        return pd.DataFrame()

    # Align feature dimensions: use common gene count
    # All datasets may have different available genes, so we use the
    # minimum dimension approach (pad/truncate to match)
    dims = [dataset_features[g]["X"].shape[1] for g in all_geos]
    min_dim = min(dims)

    results = []

    for held_out in all_geos:
        train_geos = [g for g in all_geos if g != held_out]

        X_train_parts = []
        y_train_parts = []
        for tg in train_geos:
            x = dataset_features[tg]["X"][:, :min_dim]
            X_train_parts.append(x)
            y_train_parts.append(dataset_features[tg]["y"])

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        X_test = dataset_features[held_out]["X"][:, :min_dim]
        y_test = dataset_features[held_out]["y"]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_train_s = np.nan_to_num(X_train_s, 0.0)
        X_test_s = np.nan_to_num(X_test_s, 0.0)

        # L1-logistic regression
        clf = LogisticRegression(
            C=LODO_C, penalty="l1", solver="liblinear",
            max_iter=2000, random_state=42, class_weight="balanced",
        )
        try:
            clf.fit(X_train_s, y_train)
            y_prob = clf.predict_proba(X_test_s)
            if y_prob.shape[1] == 2:
                y_score = y_prob[:, 1]
            else:
                y_score = y_prob[:, 0]
        except Exception as e:
            logger.warning(f"  LODO {held_out}: fit failed: {e}")
            continue

        # Compute metrics
        ef = dataset_features[held_out]["endpoint_family"]
        try:
            auroc = roc_auc_score(y_test, y_score)
        except ValueError:
            auroc = 0.5
        try:
            auprc = average_precision_score(y_test, y_score)
        except ValueError:
            auprc = float("nan")

        result = {
            "held_out_dataset": held_out,
            "drug": dataset_features[held_out]["drug"],
            "endpoint_family": ef,
            "n_test": len(y_test),
            "n_test_pos": dataset_features[held_out]["n_pos"],
            "n_test_neg": dataset_features[held_out]["n_neg"],
            "n_train": len(y_train),
            "n_features": min_dim,
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
        }
        results.append(result)
        logger.info(
            f"  {held_out} [{ef}]: AUROC={auroc:.3f}, AUPRC={auprc:.3f} "
            f"(n_test={len(y_test)})"
        )

    return pd.DataFrame(results)


def aggregate_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate LODO results by endpoint family and overall."""
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Per-family aggregation
    family_results = []
    for ef, grp in results_df.groupby("endpoint_family"):
        family_results.append({
            "endpoint_family": ef,
            "n_datasets": len(grp),
            "mean_auroc": round(grp["auroc"].mean(), 4),
            "std_auroc": round(grp["auroc"].std(), 4),
            "median_auroc": round(grp["auroc"].median(), 4),
            "mean_auprc": round(grp["auprc"].mean(), 4),
            "std_auprc": round(grp["auprc"].std(), 4),
            "total_patients": int(grp["n_test"].sum()),
        })
    family_df = pd.DataFrame(family_results)

    # Overall summary
    summary = pd.DataFrame([{
        "metric": "mean_auroc",
        "overall": round(results_df["auroc"].mean(), 4),
        "n_datasets": len(results_df),
        "total_patients": int(results_df["n_test"].sum()),
    }, {
        "metric": "mean_auprc",
        "overall": round(results_df["auprc"].mean(), 4),
        "n_datasets": len(results_df),
        "total_patients": int(results_df["n_test"].sum()),
    }])

    # Add per-family columns to summary
    for _, row in family_df.iterrows():
        ef = row["endpoint_family"]
        summary.loc[summary["metric"] == "mean_auroc", ef] = row["mean_auroc"]
        summary.loc[summary["metric"] == "mean_auprc", ef] = row["mean_auprc"]

    return family_df, summary


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("LODO Evaluation with Native Endpoints")
    logger.info("=" * 70)

    # Load response handler
    handler = load_response_handler(policy="lenient")

    # Load data
    logger.info("Loading datasets ...")
    expr_datasets, label_datasets, endpoint_families, catalog, pan_catalog = \
        load_all_datasets(handler)

    if len(expr_datasets) < 2:
        logger.error("Need >= 2 datasets. Aborting.")
        return

    # Load landmark genes
    landmark_genes = load_landmark_genes()
    logger.info(f"Landmark genes: {len(landmark_genes)}")

    # Load LINCS
    logger.info("Loading LINCS signatures ...")
    result = load_lincs_signatures()
    if result[0] is None:
        logger.error("No LINCS signatures. Aborting.")
        return
    lincs_sigs, lincs_gene_cols, lincs_drug_set = result

    # Build features
    logger.info("Building reversal features ...")
    dataset_features = build_reversal_features(
        expr_datasets, label_datasets, endpoint_families,
        lincs_sigs, lincs_gene_cols, lincs_drug_set,
        catalog, pan_catalog, landmark_genes,
    )

    if len(dataset_features) < 2:
        logger.error("Not enough datasets with features for LODO. Aborting.")
        handler.flush_log()
        return

    # Run LODO
    logger.info("Running LODO evaluation ...")
    results_df = run_lodo(dataset_features)

    if results_df.empty:
        logger.error("No LODO results. Aborting.")
        handler.flush_log()
        return

    # Save per-dataset results
    results_path = RESULTS / "lodo_by_dataset.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    logger.info(f"Per-dataset results saved to {results_path}")

    # Aggregate
    family_df, summary_df = aggregate_results(results_df)

    family_path = RESULTS / "lodo_by_endpoint_family.tsv"
    family_df.to_csv(family_path, sep="\t", index=False)
    logger.info(f"Per-family results saved to {family_path}")

    summary_path = RESULTS / "lodo_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Summary saved to {summary_path}")

    # Flush transformation log
    handler.flush_log()

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("LODO RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total datasets evaluated: {len(results_df)}")
    logger.info(f"Overall mean AUROC: {results_df['auroc'].mean():.4f}")
    logger.info(f"Overall mean AUPRC: {results_df['auprc'].mean():.4f}")
    logger.info("\nPer endpoint family:")
    for _, row in family_df.iterrows():
        logger.info(
            f"  {row['endpoint_family']}: "
            f"AUROC={row['mean_auroc']:.4f} +/- {row['std_auroc']:.4f} "
            f"({row['n_datasets']} datasets, {row['total_patients']} patients)"
        )


if __name__ == "__main__":
    main()
