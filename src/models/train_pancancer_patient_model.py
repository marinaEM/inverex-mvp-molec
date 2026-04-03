"""
Pan-cancer patient drug-response model using ALL CTR-DB 2.0 data.

Goal: maximise training data across all cancer types so the model
transfers better to breast-cancer patients.

Steps
-----
1. Build a pan-cancer catalog by querying CTR-DB for every cancer type.
2. Download expression matrices from GEO for the top datasets.
3. Pool all patients into one training matrix (L1000 genes, z-scored).
4. Train LightGBM with leave-one-dataset-out (LODO) cross-validation.
5. Save results, model, and feature importances.
6. Compare pan-cancer vs breast-only on breast held-out datasets.
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import DATA_CACHE, DATA_RAW, RANDOM_SEED, RESULTS
from src.data_ingestion.ctrdb import (
    CTRDB_API_BASE,
    download_geo_expression,
    load_ctrdb_dataset,
    parse_geo_response_labels,
)
from src.data_ingestion.lincs import load_landmark_genes

logger = logging.getLogger(__name__)

# ── All cancer types in CTR-DB 2.0 ─────────────────────────────────────
CANCER_TYPES = [
    "Breast cancer",
    "Lung cancer",
    "Colorectal cancer",
    "Ovarian cancer",
    "Melanoma",
    "Leukemia",
    "Lymphoma",
    "Glioblastoma",
    "Pancreatic cancer",
    "Gastric cancer",
    "Liver cancer",
    "Bladder cancer",
    "Prostate cancer",
    "Head and neck cancer",
    "Esophageal cancer",
    "Cervical cancer",
    "Kidney cancer",
    "Endometrial cancer",
    "Sarcoma",
    "Myeloma",
    "Thyroid cancer",
    "Brain cancer",
    "Neuroblastoma",
    "Mesothelioma",
    "Cholangiocarcinoma",
    "Adrenal cortex cancer",
]

PAN_CANCER_DIR = DATA_RAW / "ctrdb"
CATALOG_PATH = DATA_RAW / "ctrdb" / "pan_cancer_catalog.csv"


# =====================================================================
# STEP 1: Build pan-cancer catalog
# =====================================================================

def _collect_dataset_ids(cancer_data) -> list[str]:
    """Extract CTR-DB dataset ID strings from searchCancerApi response."""
    ids = []
    if not isinstance(cancer_data, list):
        return ids
    for record in cancer_data:
        if isinstance(record, dict) and "CTR-DB dataset" in record:
            ds_list = record["CTR-DB dataset"]
            if isinstance(ds_list, list):
                ids.extend(str(d) for d in ds_list)
    seen = set()
    unique = []
    for d in ids:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _parse_geo_id(source_str: str) -> str:
    """Extract GSE accession from a Source field."""
    import re

    if not source_str:
        return ""
    match = re.search(r"(GSE\d+)", str(source_str))
    return match.group(1) if match else ""


def _fetch_dataset_metadata_batch(
    dataset_ids: list[str],
    timeout: int = 20,
    batch_pause: float = 0.15,
) -> list[dict]:
    """Fetch metadata for a list of CTR-DB dataset IDs via searchDatasetApi."""
    records = []
    for i, ds_id in enumerate(dataset_ids):
        try:
            resp = requests.post(
                f"{CTRDB_API_BASE}/searchDatasetApi",
                json={"search": ds_id},
                timeout=timeout,
            )
            resp.raise_for_status()
            api_data = resp.json()

            if isinstance(api_data, list) and api_data:
                item = api_data[0]
            elif isinstance(api_data, dict):
                item = api_data
            else:
                continue

            rec = {
                "dataset_id": item.get("CTR-DB ID", ds_id),
                "drug": item.get("Therapeutic regimen", "unknown"),
                "sample_size": int(item.get("Sample size", 0)),
                "n_responders": int(
                    item.get('Sample size of the "response" group', 0)
                ),
                "n_nonresponders": int(
                    item.get('Sample size of the "non-response" group', 0)
                ),
                "geo_source": _parse_geo_id(item.get("Source", "")),
                "platform": item.get("Platform", ""),
                "response_grouping": item.get(
                    "Original resposne grouping", ""
                ),
                "predefined_grouping": item.get(
                    "Predefined grouping criteria of response and non-response groups",
                    "",
                ),
                "data_type": item.get("Data type", ""),
                "dataset_type": item.get("Dataset type", ""),
            }
            records.append(rec)

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  Fetched metadata for {i + 1}/{len(dataset_ids)} datasets"
                )

        except Exception as exc:
            logger.debug(f"Failed to fetch {ds_id}: {exc}")
            continue

        time.sleep(batch_pause)

    logger.info(
        f"Fetched metadata for {len(records)}/{len(dataset_ids)} datasets"
    )
    return records


def build_pan_cancer_catalog(
    timeout: int = 20,
    force: bool = False,
) -> pd.DataFrame:
    """
    Query CTR-DB API for ALL cancer types and build a comprehensive catalog.

    Saves to data/raw/ctrdb/pan_cancer_catalog.csv.
    """
    if CATALOG_PATH.exists() and not force:
        logger.info(f"Loading existing pan-cancer catalog from {CATALOG_PATH}")
        return pd.read_csv(CATALOG_PATH)

    logger.info("=" * 70)
    logger.info("STEP 1: Building pan-cancer catalog from CTR-DB 2.0")
    logger.info("=" * 70)

    all_dataset_ids_by_cancer: dict[str, list[str]] = {}

    for cancer in CANCER_TYPES:
        try:
            resp = requests.post(
                f"{CTRDB_API_BASE}/searchCancerApi",
                json={"search": cancer},
                timeout=timeout,
            )
            resp.raise_for_status()
            cancer_data = resp.json()
            ds_ids = _collect_dataset_ids(cancer_data)
            all_dataset_ids_by_cancer[cancer] = ds_ids
            logger.info(f"  {cancer}: {len(ds_ids)} dataset IDs")
            time.sleep(0.3)
        except Exception as exc:
            logger.warning(f"  {cancer}: API failed ({exc})")
            all_dataset_ids_by_cancer[cancer] = []

    # Deduplicate across cancer types — keep first cancer assignment
    seen_ids = set()
    dedup_map: dict[str, str] = {}  # dataset_id -> cancer_type
    total_raw = 0
    for cancer, ids in all_dataset_ids_by_cancer.items():
        for ds_id in ids:
            total_raw += 1
            if ds_id not in seen_ids:
                seen_ids.add(ds_id)
                dedup_map[ds_id] = cancer

    all_unique_ids = list(dedup_map.keys())
    logger.info(
        f"\nTotal raw dataset IDs: {total_raw}, "
        f"unique: {len(all_unique_ids)}"
    )

    # Fetch metadata for all unique datasets
    logger.info("Fetching metadata for all datasets ...")
    records = _fetch_dataset_metadata_batch(all_unique_ids, timeout=timeout)

    # Attach cancer_type
    for rec in records:
        rec["cancer_type"] = dedup_map.get(rec["dataset_id"], "Unknown")

    catalog = pd.DataFrame(records)
    if catalog.empty:
        logger.error("No metadata fetched — catalog is empty.")
        return catalog

    # Log summary
    logger.info(f"\nPan-cancer catalog: {len(catalog)} datasets")
    for ct in catalog["cancer_type"].value_counts().head(20).items():
        logger.info(f"  {ct[0]}: {ct[1]} datasets")

    geo_mask = catalog["geo_source"].str.startswith("GSE", na=False)
    logger.info(
        f"Datasets with GEO source: {geo_mask.sum()}/{len(catalog)}"
    )

    # Save
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(CATALOG_PATH, index=False)
    logger.info(f"Catalog saved to {CATALOG_PATH}")

    return catalog


# =====================================================================
# STEP 2: Download expression from GEO
# =====================================================================

def download_pan_cancer_geo(
    catalog: pd.DataFrame,
    min_sample_size: int = 30,
    min_responders: int = 5,
    min_nonresponders: int = 5,
    max_datasets: int = 80,
    dest_dir: Path = PAN_CANCER_DIR,
) -> pd.DataFrame:
    """
    Download expression matrices from GEO for the top pan-cancer datasets.

    Filters to datasets with sufficient sample size and clear response labels.
    Reuses any already-downloaded breast datasets.
    """
    logger.info("=" * 70)
    logger.info("STEP 2: Downloading expression from GEO")
    logger.info("=" * 70)

    # Filter catalog
    mask = (
        catalog["geo_source"].str.startswith("GSE", na=False)
        & (catalog["sample_size"] >= min_sample_size)
        & (catalog["n_responders"] >= min_responders)
        & (catalog["n_nonresponders"] >= min_nonresponders)
    )
    eligible = catalog[mask].copy()
    logger.info(
        f"Eligible datasets (>={min_sample_size} samples, "
        f">={min_responders} responders, >={min_nonresponders} non-responders, "
        f"GEO source): {len(eligible)}"
    )

    # Deduplicate by GEO source — keep largest per GEO ID
    eligible = (
        eligible.sort_values("sample_size", ascending=False)
        .drop_duplicates(subset="geo_source", keep="first")
        .head(max_datasets)
        .reset_index(drop=True)
    )
    logger.info(f"Will attempt {len(eligible)} unique GEO datasets")

    # Track download status
    download_status = []
    n_success = 0
    n_skip_cached = 0
    n_fail = 0

    for idx, row in eligible.iterrows():
        geo_id = row["geo_source"]
        ds_dir = dest_dir / geo_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded with labels
        expr_files = list(ds_dir.glob("*_expression.parquet"))
        label_file = ds_dir / "response_labels.parquet"

        if expr_files and label_file.exists():
            logger.info(
                f"  [{idx+1}/{len(eligible)}] {geo_id} — already downloaded"
            )
            download_status.append("cached")
            n_skip_cached += 1
            continue

        logger.info(
            f"  [{idx+1}/{len(eligible)}] Downloading {geo_id} "
            f"({row['cancer_type']}, n={row['sample_size']}, "
            f"drug={row['drug'][:50]}) ..."
        )

        try:
            # Download expression
            expr = download_geo_expression(geo_id, ds_dir)
            if expr is None:
                logger.warning(f"    SKIP {geo_id}: no expression data")
                download_status.append("no_expression")
                n_fail += 1
                continue

            # Extract labels
            meta = row.to_dict()
            labels = parse_geo_response_labels(geo_id, ds_dir, dataset_meta=meta)
            if labels is None:
                logger.warning(f"    SKIP {geo_id}: no response labels")
                download_status.append("no_labels")
                n_fail += 1
                continue

            # Save labels
            labels.to_frame("response").to_parquet(
                ds_dir / "response_labels.parquet"
            )

            # Verify overlap
            common_samples = expr.index.intersection(labels.index)
            if len(common_samples) < 10:
                logger.warning(
                    f"    SKIP {geo_id}: only {len(common_samples)} "
                    f"overlapping samples"
                )
                download_status.append("low_overlap")
                n_fail += 1
                continue

            n_success += 1
            download_status.append("ok")
            logger.info(
                f"    OK {geo_id}: {len(common_samples)} samples "
                f"(R={int(labels[common_samples].sum())}, "
                f"NR={int((1 - labels[common_samples]).sum())})"
            )

        except Exception as exc:
            logger.error(f"    FAIL {geo_id}: {exc}")
            download_status.append(f"error: {exc}")
            n_fail += 1

        time.sleep(1)  # be polite to GEO servers

    eligible["download_status"] = download_status

    logger.info(
        f"\nDownload summary: "
        f"{n_success} new + {n_skip_cached} cached = "
        f"{n_success + n_skip_cached} available, "
        f"{n_fail} failed"
    )

    return eligible


# =====================================================================
# STEP 3: Build pan-cancer training matrix
# =====================================================================

def build_training_matrix(
    catalog: pd.DataFrame,
    data_dir: Path = PAN_CANCER_DIR,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """
    Build the pooled pan-cancer training matrix.

    For each downloaded dataset:
    1. Load expression + response labels.
    2. Restrict to L1000 landmark genes.
    3. Z-score normalize expression within each dataset.
    4. Create binary response labels.

    Returns:
        X_pooled: patients x genes
        y_pooled: binary response
        dataset_ids: dataset identifier per patient
        cancer_types: cancer type per patient
        common_genes: list of gene names
    """
    logger.info("=" * 70)
    logger.info("STEP 3: Building pan-cancer training matrix")
    logger.info("=" * 70)

    # Load landmark genes
    landmark_df = load_landmark_genes()
    landmark_genes = landmark_df["gene_symbol"].tolist()
    logger.info(f"Landmark gene set: {len(landmark_genes)} genes")

    # Build cancer_type lookup from catalog
    cancer_type_lookup = {}
    if "cancer_type" in catalog.columns:
        for _, row in catalog.iterrows():
            geo = row.get("geo_source", "")
            ct = row.get("cancer_type", "Unknown")
            if geo:
                cancer_type_lookup[geo] = ct

    # Load all downloaded datasets
    all_X = []
    all_y = []
    all_dataset = []
    all_cancer = []
    loaded_datasets = []

    for ds_dir in sorted(data_dir.iterdir()):
        if not ds_dir.is_dir():
            continue

        geo_id = ds_dir.name
        if not geo_id.startswith("GSE"):
            continue

        result = load_ctrdb_dataset(ds_dir)
        if result is None:
            continue

        expr, labels = result

        # Find available landmark genes
        available_genes = [g for g in landmark_genes if g in expr.columns]
        if len(available_genes) < 50:
            logger.warning(
                f"  Skipping {geo_id}: only {len(available_genes)} "
                f"landmark genes available"
            )
            continue

        # Z-score within dataset
        expr_lm = expr[available_genes].copy()
        cohort_mean = expr_lm.mean(axis=0)
        cohort_std = expr_lm.std(axis=0).replace(0, 1)
        expr_z = (expr_lm - cohort_mean) / cohort_std

        # Determine cancer type
        cancer_type = cancer_type_lookup.get(geo_id, "Unknown")

        n_resp = int(labels.sum())
        n_nonresp = len(labels) - n_resp
        logger.info(
            f"  Loaded {geo_id} ({cancer_type}): "
            f"{len(labels)} patients, {len(available_genes)} genes, "
            f"R={n_resp}, NR={n_nonresp}"
        )

        all_X.append(expr_z)
        all_y.append(labels)
        all_dataset.extend([geo_id] * len(labels))
        all_cancer.extend([cancer_type] * len(labels))
        loaded_datasets.append(geo_id)

    if not all_X:
        logger.error("No usable datasets after loading")
        return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), []

    # Use genes present in at least 80% of datasets (union approach)
    # This avoids losing features when a few datasets have limited genes.
    from collections import Counter

    gene_counts = Counter()
    for x in all_X:
        gene_counts.update(x.columns.tolist())

    min_dataset_presence = max(2, int(0.8 * len(all_X)))
    common_genes = sorted(
        g for g, c in gene_counts.items() if c >= min_dataset_presence
    )

    logger.info(
        f"\nGenes present in >= {min_dataset_presence}/{len(all_X)} datasets: "
        f"{len(common_genes)}"
    )

    if len(common_genes) < 30:
        logger.error(f"Too few common genes ({len(common_genes)})")
        return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), []

    # Align to common genes — fill missing genes with 0 (z-score mean)
    X_list = []
    for x in all_X:
        aligned = x.reindex(columns=common_genes, fill_value=0.0)
        X_list.append(aligned)

    X_pooled = pd.concat(X_list, axis=0).reset_index(drop=True)
    y_pooled = pd.concat(all_y, axis=0).reset_index(drop=True).astype(int)
    dataset_ids = pd.Series(all_dataset, name="dataset_id")
    cancer_types = pd.Series(all_cancer, name="cancer_type")

    # Fill any remaining NaN with 0 (shouldn't be many after z-scoring)
    X_pooled = X_pooled.fillna(0.0)

    logger.info(
        f"\nPooled training matrix: "
        f"{len(X_pooled)} patients x {len(common_genes)} genes"
    )
    logger.info(
        f"Responders: {y_pooled.sum()}, "
        f"Non-responders: {(1 - y_pooled).sum()}"
    )
    logger.info(f"Datasets: {dataset_ids.nunique()}")
    logger.info(f"Cancer types: {cancer_types.nunique()}")

    # Log per-cancer-type summary
    for ct in cancer_types.unique():
        mask = cancer_types == ct
        n = mask.sum()
        n_resp = y_pooled[mask].sum()
        logger.info(f"  {ct}: {n} patients (R={n_resp}, NR={n - n_resp})")

    return X_pooled, y_pooled, dataset_ids, cancer_types, common_genes


# =====================================================================
# STEP 4 & 5: Train pan-cancer LightGBM with LODO-CV
# =====================================================================

def train_pan_cancer_model(
    X_pooled: pd.DataFrame,
    y_pooled: pd.Series,
    dataset_ids: pd.Series,
    cancer_types: pd.Series,
    common_genes: list[str],
    results_path: Path = RESULTS / "pan_cancer_model_lodo_results.csv",
    model_path: Path = RESULTS / "pan_cancer_patient_model.joblib",
    importances_path: Path = RESULTS / "pan_cancer_feature_importances.csv",
) -> pd.DataFrame:
    """
    Train LightGBM with leave-one-dataset-out cross-validation.

    Uses conservative hyperparameters to prevent overfitting.
    """
    import joblib
    import lightgbm as lgb
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    logger.info("=" * 70)
    logger.info("STEP 4: Training pan-cancer LightGBM (LODO-CV)")
    logger.info("=" * 70)

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 300,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 10,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "class_weight": "balanced",
    }

    unique_datasets = dataset_ids.unique()
    fold_results = []

    for held_out in unique_datasets:
        test_mask = dataset_ids == held_out
        train_mask = ~test_mask

        X_train = X_pooled[train_mask].reset_index(drop=True)
        y_train = y_pooled[train_mask].reset_index(drop=True)
        X_test = X_pooled[test_mask].reset_index(drop=True)
        y_test = y_pooled[test_mask].reset_index(drop=True)

        # Skip if insufficient test data
        if len(y_test) < 10 or y_test.nunique() < 2:
            logger.info(f"  Skipping {held_out}: insufficient test data")
            continue

        n_test_resp = int(y_test.sum())
        n_test_nonresp = len(y_test) - n_test_resp
        if n_test_resp < 3 or n_test_nonresp < 3:
            logger.info(
                f"  Skipping {held_out}: too few in one class "
                f"(R={n_test_resp}, NR={n_test_nonresp})"
            )
            continue

        # Determine cancer type for this dataset
        ct = cancer_types[test_mask].iloc[0]

        # Train
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        sensitivity = (
            y_pred[y_test == 1].sum() / y_test.sum()
            if y_test.sum() > 0
            else 0.0
        )
        specificity = (
            (1 - y_pred[y_test == 0]).sum() / (1 - y_test).sum()
            if (1 - y_test).sum() > 0
            else 0.0
        )

        fold_result = {
            "held_out_dataset": held_out,
            "cancer_type": ct,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_test_responders": n_test_resp,
            "n_test_nonresponders": n_test_nonresp,
            "auc": round(auc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
        }
        fold_results.append(fold_result)

        logger.info(
            f"  {held_out} ({ct}): AUC={auc:.3f}, "
            f"BalAcc={bal_acc:.3f}, "
            f"Sens={float(sensitivity):.3f}, Spec={float(specificity):.3f} "
            f"(n={len(X_test)})"
        )

    if not fold_results:
        logger.error("No successful LODO-CV folds")
        return pd.DataFrame()

    results_df = pd.DataFrame(fold_results)

    # ── Summary statistics ──────────────────────────────────────────
    # Overall mean
    summary = {
        "held_out_dataset": "MEAN_ALL",
        "cancer_type": "ALL",
        "n_train": int(results_df["n_train"].mean()),
        "n_test": int(results_df["n_test"].mean()),
        "n_test_responders": int(results_df["n_test_responders"].mean()),
        "n_test_nonresponders": int(
            results_df["n_test_nonresponders"].mean()
        ),
        "auc": round(results_df["auc"].mean(), 4),
        "balanced_accuracy": round(
            results_df["balanced_accuracy"].mean(), 4
        ),
        "sensitivity": round(results_df["sensitivity"].mean(), 4),
        "specificity": round(results_df["specificity"].mean(), 4),
    }

    # Breast-only mean
    breast_mask = results_df["cancer_type"] == "Breast cancer"
    if breast_mask.sum() > 0:
        breast_results = results_df[breast_mask]
        breast_summary = {
            "held_out_dataset": "MEAN_BREAST",
            "cancer_type": "Breast cancer",
            "n_train": int(breast_results["n_train"].mean()),
            "n_test": int(breast_results["n_test"].mean()),
            "n_test_responders": int(
                breast_results["n_test_responders"].mean()
            ),
            "n_test_nonresponders": int(
                breast_results["n_test_nonresponders"].mean()
            ),
            "auc": round(breast_results["auc"].mean(), 4),
            "balanced_accuracy": round(
                breast_results["balanced_accuracy"].mean(), 4
            ),
            "sensitivity": round(breast_results["sensitivity"].mean(), 4),
            "specificity": round(breast_results["specificity"].mean(), 4),
        }
    else:
        breast_summary = None

    summaries = [summary]
    if breast_summary:
        summaries.append(breast_summary)

    results_df = pd.concat(
        [results_df, pd.DataFrame(summaries)], ignore_index=True
    )

    # Save LODO results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nLODO results saved to {results_path}")

    # ── Train final model on all data ───────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 5: Training final model on ALL data + saving")
    logger.info("=" * 70)

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_pooled, y_pooled)
    joblib.dump(final_model, model_path)
    logger.info(f"Final model saved to {model_path}")

    # Feature importances
    importances = (
        pd.DataFrame(
            {"gene": common_genes, "importance": final_model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(importances_path, index=False)
    logger.info(f"Feature importances saved to {importances_path}")
    logger.info(f"Top 20 genes:\n{importances.head(20).to_string()}")

    return results_df


# =====================================================================
# STEP 6: Compare pan-cancer vs breast-only on breast held-out data
# =====================================================================

def compare_with_breast_only(
    lodo_results: pd.DataFrame,
    breast_only_path: Path = RESULTS / "patient_model_lodo_results.csv",
) -> None:
    """
    Compare the pan-cancer model vs the breast-only model on breast
    cancer held-out datasets.
    """
    logger.info("=" * 70)
    logger.info("STEP 6: Comparing pan-cancer vs breast-only")
    logger.info("=" * 70)

    if not breast_only_path.exists():
        logger.warning(
            f"Breast-only results not found at {breast_only_path}. Skipping."
        )
        return

    breast_only = pd.read_csv(breast_only_path)

    # Get breast-cancer results from pan-cancer LODO
    pan_breast = lodo_results[
        (lodo_results["cancer_type"] == "Breast cancer")
        & ~lodo_results["held_out_dataset"].str.startswith("MEAN")
    ].copy()

    if pan_breast.empty:
        logger.warning("No breast cancer results in pan-cancer LODO")
        return

    # Get breast-only results (exclude MEAN row)
    breast_only_folds = breast_only[
        breast_only["held_out_dataset"] != "MEAN"
    ].copy()

    # Find common held-out datasets
    common_datasets = set(pan_breast["held_out_dataset"]) & set(
        breast_only_folds["held_out_dataset"]
    )

    if not common_datasets:
        logger.info(
            "No overlapping held-out datasets between pan-cancer and "
            "breast-only. Comparing aggregate metrics."
        )
        pan_mean_auc = pan_breast["auc"].mean()
        bo_mean_auc = breast_only_folds["auc"].mean()
        logger.info(f"\n  Pan-cancer (breast folds) mean AUC: {pan_mean_auc:.4f}")
        logger.info(f"  Breast-only mean AUC:                {bo_mean_auc:.4f}")
        diff = pan_mean_auc - bo_mean_auc
        logger.info(f"  Difference (pan - breast):           {diff:+.4f}")
        return

    logger.info(
        f"\nComparing on {len(common_datasets)} common held-out datasets:"
    )
    logger.info(f"{'Dataset':<20} {'Pan-cancer AUC':>15} {'Breast-only AUC':>16} {'Diff':>8}")
    logger.info("-" * 65)

    diffs = []
    for ds in sorted(common_datasets):
        pan_auc = pan_breast.loc[
            pan_breast["held_out_dataset"] == ds, "auc"
        ].values[0]
        bo_auc = breast_only_folds.loc[
            breast_only_folds["held_out_dataset"] == ds, "auc"
        ].values[0]
        diff = pan_auc - bo_auc
        diffs.append(diff)
        logger.info(f"  {ds:<20} {pan_auc:>12.4f}    {bo_auc:>12.4f}  {diff:>+8.4f}")

    mean_pan = pan_breast.loc[
        pan_breast["held_out_dataset"].isin(common_datasets), "auc"
    ].mean()
    mean_bo = breast_only_folds.loc[
        breast_only_folds["held_out_dataset"].isin(common_datasets), "auc"
    ].mean()
    mean_diff = mean_pan - mean_bo

    logger.info("-" * 65)
    logger.info(f"  {'MEAN':<20} {mean_pan:>12.4f}    {mean_bo:>12.4f}  {mean_diff:>+8.4f}")

    if mean_diff > 0:
        logger.info(
            f"\nPan-cancer model IMPROVES breast AUC by {mean_diff:+.4f}"
        )
    else:
        logger.info(
            f"\nPan-cancer model is lower by {mean_diff:.4f} — "
            f"breast-only model wins on these folds"
        )

    # Save comparison
    comp_path = RESULTS / "pan_vs_breast_comparison.csv"
    comp_rows = []
    for ds in sorted(common_datasets):
        pan_auc = pan_breast.loc[
            pan_breast["held_out_dataset"] == ds, "auc"
        ].values[0]
        bo_auc = breast_only_folds.loc[
            breast_only_folds["held_out_dataset"] == ds, "auc"
        ].values[0]
        comp_rows.append(
            {
                "dataset": ds,
                "pan_cancer_auc": pan_auc,
                "breast_only_auc": bo_auc,
                "difference": pan_auc - bo_auc,
            }
        )
    comp_rows.append(
        {
            "dataset": "MEAN",
            "pan_cancer_auc": mean_pan,
            "breast_only_auc": mean_bo,
            "difference": mean_diff,
        }
    )
    pd.DataFrame(comp_rows).to_csv(comp_path, index=False)
    logger.info(f"Comparison saved to {comp_path}")


# =====================================================================
# Main pipeline
# =====================================================================

def run_pan_cancer_pipeline(force_catalog: bool = False) -> None:
    """Run the complete pan-cancer model pipeline."""
    logger.info("=" * 70)
    logger.info("PAN-CANCER PATIENT DRUG RESPONSE MODEL")
    logger.info("=" * 70)

    # Step 1: Build catalog
    catalog = build_pan_cancer_catalog(force=force_catalog)
    if catalog.empty:
        logger.error("Empty catalog — aborting.")
        return

    # Step 2: Download from GEO
    download_results = download_pan_cancer_geo(catalog)

    # Step 3: Build training matrix
    X_pooled, y_pooled, dataset_ids, cancer_types, common_genes = (
        build_training_matrix(catalog)
    )
    if X_pooled.empty:
        logger.error("Empty training matrix — aborting.")
        return

    # Step 4 & 5: Train and save
    lodo_results = train_pan_cancer_model(
        X_pooled, y_pooled, dataset_ids, cancer_types, common_genes
    )
    if lodo_results.empty:
        logger.error("No LODO results — aborting.")
        return

    # Step 6: Compare with breast-only
    compare_with_breast_only(lodo_results)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    mean_row = lodo_results[lodo_results["held_out_dataset"] == "MEAN_ALL"]
    if not mean_row.empty:
        logger.info(
            f"Overall LODO AUC: {mean_row['auc'].values[0]:.4f}"
        )

    breast_row = lodo_results[
        lodo_results["held_out_dataset"] == "MEAN_BREAST"
    ]
    if not breast_row.empty:
        logger.info(
            f"Breast LODO AUC:  {breast_row['auc'].values[0]:.4f}"
        )

    logger.info(f"Total patients:   {len(X_pooled)}")
    logger.info(f"Total datasets:   {dataset_ids.nunique()}")
    logger.info(f"Gene features:    {len(common_genes)}")


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Pan-cancer patient drug-response model"
    )
    parser.add_argument(
        "--force-catalog",
        action="store_true",
        help="Re-download the pan-cancer catalog even if it exists",
    )
    args = parser.parse_args()

    run_pan_cancer_pipeline(force_catalog=args.force_catalog)
