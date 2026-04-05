"""
Treatment-class-specific models for INVEREX.

Core question: Is the AUC ceiling at 0.61 because one pooled model can't
capture treatment-specific biology, or because expression alone has a hard limit?

Steps:
  1. Classify datasets by treatment class
  2. Train class-specific LightGBMs with LODO
  3. Routing ensemble (weighted blend of pooled + class-specific)
  4. I-SPY2 arm-level evaluation
  5. Feature importance comparison
  6. Summary comparison table

Usage:
    pixi run python -m src.models.treatment_class_models
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
DATA_METADATA = ROOT / "data" / "metadata"
RESULTS = ROOT / "results"

for d in [DATA_METADATA, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── LightGBM params (as specified) ───────────────────────────────────────
LGBM_PARAMS = {
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
    "random_state": 42,
    "verbose": -1,
}

# ── Treatment class assignment for I-SPY2 arms ──────────────────────────
ISPY2_ARM_CLASS = {
    "Paclitaxel": "chemo",
    "Paclitaxel + Neratinib": "her2_targeted",
    "Paclitaxel + ABT 888 + Carboplatin": "dna_damage",
    "Paclitaxel + Pembrolizumab": "immunotherapy",
    "T-DM1 + Pertuzumab": "her2_targeted",
    "Paclitaxel + Ganitumab": "other_targeted",
    "Paclitaxel + Ganetespib": "other_targeted",
    "Paclitaxel + MK-2206": "targeted",
    "Paclitaxel + MK-2206 + Trastuzumab": "targeted",
    "Paclitaxel + Trastuzumab": "her2_targeted",
    "Paclitaxel + Pertuzumab + Trastuzumab": "her2_targeted",
    "Paclitaxel + AMG 386": "other_targeted",
    "Paclitaxel + AMG 386 + Trastuzumab": "other_targeted",
    "Paclitaxel + AMG-386": "other_targeted",
}


# ======================================================================
# STEP 1: Load and classify datasets
# ======================================================================

def load_landmark_genes() -> list:
    """Load 212 L1000 landmark gene symbols from geneinfo_beta_input.txt."""
    gi = pd.read_csv(DATA_CACHE / "geneinfo_beta_input.txt", sep="\t")
    genes = gi["gene_symbol"].dropna().unique().tolist()
    logger.info(f"Loaded {len(genes)} landmark genes")
    return genes


def _align_expr_labels_ctrdb(geo_id: str) -> tuple:
    """Load and align expression + labels for a CTR-DB dataset.
    CTR-DB: expression index = sample_id (GSM*), labels index = sample_id.
    """
    expr_path = DATA_RAW / "ctrdb" / geo_id / f"{geo_id}_expression.parquet"
    label_path = DATA_RAW / "ctrdb" / geo_id / "response_labels.parquet"
    if not expr_path.exists() or not label_path.exists():
        return None, None
    expr = pd.read_parquet(expr_path)
    labels = pd.read_parquet(label_path)
    if "response" not in labels.columns:
        return None, None
    # Labels index is sample_id for CTR-DB
    common = expr.index.intersection(labels.index)
    if len(common) < 10:
        return None, None
    return expr.loc[common], labels.loc[common, "response"]


def _align_expr_labels_ispy2() -> tuple:
    """Load I-SPY2 expression and labels, align by position."""
    expr = pd.read_parquet(DATA_RAW / "ispy2" / "GSE194040_expression.parquet")
    labels = pd.read_parquet(DATA_RAW / "ispy2" / "response_labels.parquet")
    # Expression index is internal IDs (like '756412'), labels have integer index
    # Both have 988 rows - align by position
    assert len(expr) == len(labels), f"I-SPY2 size mismatch: {len(expr)} vs {len(labels)}"
    # Reset both to positional alignment
    expr_reset = expr.reset_index(drop=True)
    labels_reset = labels.reset_index(drop=True)
    return expr_reset, labels_reset


def _align_expr_labels_brightness() -> tuple:
    """Load BrighTNess expression and labels, align by position."""
    expr = pd.read_parquet(DATA_RAW / "brightness" / "GSE164458_expression.parquet")
    labels = pd.read_parquet(DATA_RAW / "brightness" / "response_labels.parquet")
    # Expression index is internal IDs ('102001' etc), labels have integer index
    assert len(expr) == len(labels), f"BrighTNess size mismatch: {len(expr)} vs {len(labels)}"
    expr_reset = expr.reset_index(drop=True)
    labels_reset = labels.reset_index(drop=True)
    return expr_reset, labels_reset


def _align_expr_labels_durva() -> tuple:
    """Load durvalumab+olaparib expression and labels."""
    expr = pd.read_parquet(DATA_RAW / "durva_olap_breast" / "GSE173839_expression.parquet")
    labels = pd.read_parquet(DATA_RAW / "durva_olap_breast" / "response_labels.parquet")
    # Expression index is GSM IDs, labels have integer index with sample_id column
    # Align via sample_id
    labels_indexed = labels.set_index("sample_id")
    common = expr.index.intersection(labels_indexed.index)
    if len(common) < 10:
        return None, None
    labels_indexed = labels_indexed.loc[common]
    # char_pcr is the response (may be string), filter out -1
    labels_indexed["char_pcr"] = pd.to_numeric(labels_indexed["char_pcr"], errors="coerce")
    valid = labels_indexed["char_pcr"].isin([0, 1])
    labels_indexed = labels_indexed[valid]
    common = labels_indexed.index
    if len(common) < 10:
        return None, None
    return expr.loc[common], labels_indexed["char_pcr"].astype(int)


def _align_expr_labels_hoogstraat2() -> tuple:
    """Load hoogstraat_2 (GSE192341) expression and labels."""
    expr = pd.read_parquet(DATA_RAW / "hoogstraat_2" / "GSE192341_expression.parquet")
    labels = pd.read_parquet(DATA_RAW / "hoogstraat_2" / "response_labels.parquet")
    # Need to create response from char_pathological_complete_response_(pcr)
    pcr_col = "char_pathological_complete_response_(pcr)"
    if pcr_col not in labels.columns:
        return None, None
    labels["response"] = (labels[pcr_col] == "pCR").astype(int)
    # Align by position (expression has more rows than labels sometimes)
    # Expression idx is internal IDs, labels have integer index with sample_id
    # Use positional alignment: both ordered the same way
    n = min(len(expr), len(labels))
    expr_reset = expr.iloc[:n].reset_index(drop=True)
    labels_reset = labels.iloc[:n].reset_index(drop=True)
    return expr_reset, labels_reset["response"]


def get_ctrdb_treatment_class(geo_id: str) -> str:
    """Determine treatment class for a CTR-DB dataset from the catalog + inventory."""
    # First check inventory
    inv_path = DATA_METADATA / "all_datasets_inventory.tsv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path, sep="\t")
        match = inv[(inv["source"] == "ctrdb") & (inv["geo_accession"] == geo_id)]
        if len(match) > 0:
            tc = match.iloc[0]["treatment_class"]
            # Map inventory classes to our taxonomy
            class_map = {
                "chemotherapy": "chemo",
                "endocrine": "endocrine",
                "targeted": "her2_targeted",
            }
            return class_map.get(tc, tc)

    # Fall back to catalog drug name
    cat_path = DATA_RAW / "ctrdb" / "catalog.csv"
    if cat_path.exists():
        cat = pd.read_csv(cat_path)
        match = cat[cat["geo_source"] == geo_id]
        if len(match) > 0:
            drug = match.iloc[0]["drug"].lower()
            if any(d in drug for d in ["tamoxifen", "letrozole", "fulvestrant",
                                        "anastrozole", "exemestane"]):
                return "endocrine"
            if "trastuzumab" in drug and "chemo" not in drug:
                return "her2_targeted"
            if "mk-2206" in drug:
                return "targeted"
            # Default: chemo
            return "chemo"

    return "chemo"  # safe default for breast cancer neoadjuvant


def classify_and_load_all_datasets(landmark_genes: list) -> pd.DataFrame:
    """
    Load all datasets, assign treatment classes, restrict to landmark genes,
    per-dataset z-score, and return a metadata DataFrame.

    Returns a list of dicts with keys:
      dataset_id, treatment_class, n_patients, n_responders, source,
      expression (DataFrame), response (Series)
    """
    datasets = []

    # ── CTR-DB datasets (breast cancer only) ─────────────────────────
    # Only use the 18 datasets in the inventory (verified breast cancer)
    inv = pd.read_csv(DATA_METADATA / "all_datasets_inventory.tsv", sep="\t")
    ctrdb_geos = inv[inv["source"] == "ctrdb"]["geo_accession"].tolist()

    for geo_id in ctrdb_geos:
        expr, resp = _align_expr_labels_ctrdb(geo_id)
        if expr is None:
            logger.warning(f"Skipping CTR-DB {geo_id}: no valid data")
            continue
        tc = get_ctrdb_treatment_class(geo_id)
        datasets.append({
            "dataset_id": f"ctrdb_{geo_id}",
            "geo_accession": geo_id,
            "source": "ctrdb",
            "treatment_class": tc,
            "expression": expr,
            "response": resp,
        })
        logger.info(f"  CTR-DB {geo_id}: n={len(resp)}, class={tc}, "
                     f"resp_rate={resp.mean():.2f}")

    # ── I-SPY2: split by arm ─────────────────────────────────────────
    expr_ispy2, labels_ispy2 = _align_expr_labels_ispy2()
    for arm_name, arm_class in ISPY2_ARM_CLASS.items():
        mask = labels_ispy2["char_arm"] == arm_name
        if mask.sum() < 10:
            logger.warning(f"Skipping I-SPY2 arm '{arm_name}': only {mask.sum()} pts")
            continue
        arm_expr = expr_ispy2.loc[mask].reset_index(drop=True)
        arm_resp = labels_ispy2.loc[mask, "response"].reset_index(drop=True).astype(int)
        # Need both classes
        if arm_resp.nunique() < 2:
            logger.warning(f"Skipping I-SPY2 arm '{arm_name}': single class")
            continue
        safe_name = arm_name.replace(" ", "_").replace("+", "plus")
        datasets.append({
            "dataset_id": f"ispy2_{safe_name}",
            "geo_accession": "GSE194040",
            "source": "ispy2",
            "treatment_class": arm_class,
            "expression": arm_expr,
            "response": arm_resp,
        })
        logger.info(f"  I-SPY2 arm '{arm_name}': n={len(arm_resp)}, class={arm_class}, "
                     f"resp_rate={arm_resp.mean():.2f}")

    # ── BrighTNess ───────────────────────────────────────────────────
    expr_bright, labels_bright = _align_expr_labels_brightness()
    # BrighTNess as whole = dna_damage (PARP trial)
    resp_bright = labels_bright["response"].astype(int)
    if resp_bright.nunique() >= 2:
        datasets.append({
            "dataset_id": "brightness_GSE164458",
            "geo_accession": "GSE164458",
            "source": "brightness",
            "treatment_class": "dna_damage",
            "expression": expr_bright,
            "response": resp_bright,
        })
        logger.info(f"  BrighTNess: n={len(resp_bright)}, class=dna_damage, "
                     f"resp_rate={resp_bright.mean():.2f}")

    # ── Durvalumab + Olaparib ────────────────────────────────────────
    expr_durva, resp_durva = _align_expr_labels_durva()
    if expr_durva is not None and resp_durva.nunique() >= 2:
        datasets.append({
            "dataset_id": "durva_olap_GSE173839",
            "geo_accession": "GSE173839",
            "source": "durva_olap",
            "treatment_class": "immunotherapy",
            "expression": expr_durva,
            "response": resp_durva,
        })
        logger.info(f"  Durva+Olap: n={len(resp_durva)}, class=immunotherapy, "
                     f"resp_rate={resp_durva.mean():.2f}")

    # ── Hoogstraat_2 (neoadjuvant chemo) ─────────────────────────────
    expr_hoog, resp_hoog = _align_expr_labels_hoogstraat2()
    if expr_hoog is not None and resp_hoog.nunique() >= 2:
        datasets.append({
            "dataset_id": "hoogstraat2_GSE192341",
            "geo_accession": "GSE192341",
            "source": "hoogstraat_2",
            "treatment_class": "chemo",
            "expression": expr_hoog,
            "response": resp_hoog,
        })
        logger.info(f"  Hoogstraat2: n={len(resp_hoog)}, class=chemo, "
                     f"resp_rate={resp_hoog.mean():.2f}")

    # ── Restrict to landmark genes and per-dataset z-score ───────────
    logger.info(f"\nTotal datasets loaded: {len(datasets)}")
    for ds in datasets:
        expr = ds["expression"]
        available_genes = [g for g in landmark_genes if g in expr.columns]
        if len(available_genes) < 50:
            logger.warning(f"  {ds['dataset_id']}: only {len(available_genes)} landmark genes, skipping")
            ds["skip"] = True
            continue
        expr_lm = expr[available_genes].copy()
        # Per-dataset z-scoring (safe default, no leakage)
        expr_z = (expr_lm - expr_lm.mean()) / (expr_lm.std() + 1e-8)
        ds["expression"] = expr_z
        ds["n_genes"] = len(available_genes)
        ds["n_patients"] = len(ds["response"])
        ds["n_responders"] = int(ds["response"].sum())
        ds["skip"] = False

    # Filter out skipped
    datasets = [ds for ds in datasets if not ds.get("skip", False)]
    logger.info(f"Datasets after landmark gene filter: {len(datasets)}")

    return datasets


def save_treatment_class_splits(datasets: list) -> pd.DataFrame:
    """Save treatment_class_splits.tsv."""
    rows = []
    for ds in datasets:
        rows.append({
            "dataset_id": ds["dataset_id"],
            "geo_accession": ds["geo_accession"],
            "source": ds["source"],
            "treatment_class": ds["treatment_class"],
            "n_patients": ds["n_patients"],
            "n_responders": ds["n_responders"],
            "n_genes": ds["n_genes"],
        })
    df = pd.DataFrame(rows)
    outpath = DATA_METADATA / "treatment_class_splits.tsv"
    df.to_csv(outpath, sep="\t", index=False)
    logger.info(f"Saved treatment class splits to {outpath}")

    # Print summary
    logger.info("\n=== Treatment Class Summary ===")
    for tc, grp in df.groupby("treatment_class"):
        logger.info(f"  {tc}: {len(grp)} datasets, {grp['n_patients'].sum()} total patients")
    return df


# ======================================================================
# STEP 2: Train class-specific LightGBMs with LODO
# ======================================================================

def _make_common_features(datasets: list) -> list:
    """Find the intersection of gene columns across all datasets."""
    if not datasets:
        return []
    common = set(datasets[0]["expression"].columns)
    for ds in datasets[1:]:
        common = common.intersection(ds["expression"].columns)
    return sorted(common)


def _train_and_predict_lodo(
    train_datasets: list,
    test_dataset: dict,
    common_genes: list,
) -> tuple:
    """
    Train LightGBM on train_datasets, predict on test_dataset.
    Returns (y_true, y_pred_proba) for the test set.
    """
    # Build training matrix
    X_trains = []
    y_trains = []
    for ds in train_datasets:
        X_trains.append(ds["expression"][common_genes])
        y_trains.append(ds["response"])
    X_train = pd.concat(X_trains, axis=0, ignore_index=True)
    y_train = pd.concat(y_trains, axis=0, ignore_index=True)

    X_test = test_dataset["expression"][common_genes]
    y_test = test_dataset["response"].reset_index(drop=True)

    # Fill NaN with 0 (some genes may have NaN after z-scoring)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict_proba(X_test)[:, 1]

    return y_test.values, y_pred, model


def run_lodo_for_class(
    datasets: list,
    class_name: str,
    common_genes: list,
) -> list:
    """Run LODO within a treatment class. Returns per-dataset results."""
    results = []
    n_datasets = len(datasets)
    if n_datasets < 2:
        logger.warning(f"  Class '{class_name}': only {n_datasets} datasets, skipping LODO")
        return results

    for i, test_ds in enumerate(datasets):
        train_ds = [ds for j, ds in enumerate(datasets) if j != i]
        if len(train_ds) == 0:
            continue

        y_true, y_pred, model = _train_and_predict_lodo(train_ds, test_ds, common_genes)

        # Need both classes in test set
        if len(np.unique(y_true)) < 2:
            logger.warning(f"  Skipping {test_ds['dataset_id']}: single class in test")
            continue

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan

        n_train = sum(len(ds["response"]) for ds in train_ds)
        results.append({
            "dataset_id": test_ds["dataset_id"],
            "treatment_class": class_name,
            "model_type": "class_specific",
            "auc": auc,
            "n_test": len(y_true),
            "n_train": n_train,
            "n_train_datasets": len(train_ds),
            "resp_rate_test": float(y_true.mean()),
            "y_true": y_true,
            "y_pred": y_pred,
            "model": model,
        })
        logger.info(f"    LODO {test_ds['dataset_id']}: AUC={auc:.3f} "
                     f"(n_test={len(y_true)}, n_train={n_train})")

    return results


def run_pooled_lodo(all_datasets: list, common_genes: list) -> list:
    """Run LODO using ALL datasets pooled together."""
    results = []
    for i, test_ds in enumerate(all_datasets):
        train_ds = [ds for j, ds in enumerate(all_datasets) if j != i]
        if len(train_ds) == 0:
            continue

        y_true, y_pred, model = _train_and_predict_lodo(train_ds, test_ds, common_genes)

        if len(np.unique(y_true)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan

        n_train = sum(len(ds["response"]) for ds in train_ds)
        results.append({
            "dataset_id": test_ds["dataset_id"],
            "treatment_class": test_ds["treatment_class"],
            "model_type": "pooled",
            "auc": auc,
            "n_test": len(y_true),
            "n_train": n_train,
            "n_train_datasets": len(train_ds),
            "resp_rate_test": float(y_true.mean()),
            "y_true": y_true,
            "y_pred": y_pred,
            "model": model,
        })
        logger.info(f"    POOLED LODO {test_ds['dataset_id']}: AUC={auc:.3f}")

    return results


# ======================================================================
# STEP 3: Routing ensemble
# ======================================================================

def compute_routing_ensemble(
    pooled_results: list,
    class_results: list,
) -> list:
    """
    For each held-out dataset, blend pooled + class-specific predictions.
    Weight based on class training size:
      >1000 patients: 0.8 class + 0.2 pooled
      300-1000:       0.6 class + 0.4 pooled
      <300:           0.3 class + 0.7 pooled
    """
    # Index class results by dataset_id
    class_by_ds = {r["dataset_id"]: r for r in class_results}

    # Compute class training sizes
    class_train_sizes = {}
    for r in class_results:
        tc = r["treatment_class"]
        if tc not in class_train_sizes:
            class_train_sizes[tc] = r["n_train"]
        else:
            class_train_sizes[tc] = max(class_train_sizes[tc], r["n_train"])

    ensemble_results = []
    for pr in pooled_results:
        ds_id = pr["dataset_id"]
        tc = pr["treatment_class"]

        if ds_id not in class_by_ds:
            # No class-specific model available, use pooled only
            ensemble_results.append({
                "dataset_id": ds_id,
                "treatment_class": tc,
                "pooled_auc": pr["auc"],
                "class_auc": np.nan,
                "routed_auc": pr["auc"],
                "weight_class": 0.0,
                "weight_pooled": 1.0,
                "n_test": pr["n_test"],
            })
            continue

        cr = class_by_ds[ds_id]
        train_size = class_train_sizes.get(tc, 0)

        if train_size > 1000:
            w_class, w_pooled = 0.8, 0.2
        elif train_size >= 300:
            w_class, w_pooled = 0.6, 0.4
        else:
            w_class, w_pooled = 0.3, 0.7

        # Blend predictions
        y_true = pr["y_true"]
        y_blend = w_class * cr["y_pred"] + w_pooled * pr["y_pred"]

        try:
            routed_auc = roc_auc_score(y_true, y_blend)
        except ValueError:
            routed_auc = np.nan

        ensemble_results.append({
            "dataset_id": ds_id,
            "treatment_class": tc,
            "pooled_auc": pr["auc"],
            "class_auc": cr["auc"],
            "routed_auc": routed_auc,
            "weight_class": w_class,
            "weight_pooled": w_pooled,
            "n_test": pr["n_test"],
            "class_train_size": train_size,
        })

    return ensemble_results


# ======================================================================
# STEP 4: I-SPY2 arm-level evaluation
# ======================================================================

def evaluate_ispy2_arms(
    pooled_results: list,
    class_results: list,
    ensemble_results: list,
) -> pd.DataFrame:
    """Extract I-SPY2 arm-specific results."""
    rows = []
    for er in ensemble_results:
        if "ispy2_" in er["dataset_id"]:
            rows.append(er)

    if not rows:
        logger.warning("No I-SPY2 arm results found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ======================================================================
# STEP 5: Feature importance comparison
# ======================================================================

def extract_feature_importances(
    class_results_by_class: dict,
    common_genes: list,
) -> pd.DataFrame:
    """
    For each treatment class, train a model on ALL class data and
    extract top 20 features.
    """
    rows = []
    for tc, datasets in class_results_by_class.items():
        if len(datasets) < 2:
            continue
        # Train on all data in this class
        X_all = pd.concat([ds["expression"][common_genes] for ds in datasets],
                          axis=0, ignore_index=True).fillna(0)
        y_all = pd.concat([ds["response"] for ds in datasets],
                          axis=0, ignore_index=True)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_all, y_all)

        importances = pd.Series(model.feature_importances_, index=common_genes)
        top20 = importances.nlargest(20)

        for rank, (gene, imp) in enumerate(top20.items(), 1):
            rows.append({
                "treatment_class": tc,
                "rank": rank,
                "gene": gene,
                "importance": imp,
            })

    return pd.DataFrame(rows)


# ======================================================================
# STEP 6: Summary table
# ======================================================================

def build_summary_table(ensemble_results: list) -> pd.DataFrame:
    """Build the final comparison table by treatment class."""
    df = pd.DataFrame(ensemble_results)
    if df.empty:
        return df

    rows = []
    for tc, grp in df.groupby("treatment_class"):
        rows.append({
            "treatment_class": tc,
            "n_datasets": len(grp),
            "n_total_patients": grp["n_test"].sum(),
            "pooled_auc_mean": grp["pooled_auc"].mean(),
            "pooled_auc_median": grp["pooled_auc"].median(),
            "class_auc_mean": grp["class_auc"].mean(),
            "class_auc_median": grp["class_auc"].median(),
            "routed_auc_mean": grp["routed_auc"].mean(),
            "routed_auc_median": grp["routed_auc"].median(),
        })

    # Overall
    rows.append({
        "treatment_class": "OVERALL",
        "n_datasets": len(df),
        "n_total_patients": df["n_test"].sum(),
        "pooled_auc_mean": df["pooled_auc"].mean(),
        "pooled_auc_median": df["pooled_auc"].median(),
        "class_auc_mean": df["class_auc"].mean(),
        "class_auc_median": df["class_auc"].median(),
        "routed_auc_mean": df["routed_auc"].mean(),
        "routed_auc_median": df["routed_auc"].median(),
    })

    return pd.DataFrame(rows)


def check_for_leakage(results: list) -> list:
    """Flag any AUC > 0.70 for leakage investigation."""
    flagged = []
    for r in results:
        if r.get("auc", 0) > 0.70 or r.get("routed_auc", 0) > 0.70:
            flagged.append(r)
    return flagged


# ======================================================================
# MAIN
# ======================================================================

def main():
    logger.info("=" * 70)
    logger.info("INVEREX Treatment-Class-Specific Models")
    logger.info("=" * 70)

    # ── Step 1: Load and classify ────────────────────────────────────
    logger.info("\n>>> STEP 1: Classify datasets by treatment class")
    landmark_genes = load_landmark_genes()
    datasets = classify_and_load_all_datasets(landmark_genes)
    splits_df = save_treatment_class_splits(datasets)

    # ── Find common genes across all datasets ────────────────────────
    common_genes = _make_common_features(datasets)
    logger.info(f"\nCommon landmark genes across all datasets: {len(common_genes)}")
    if len(common_genes) < 50:
        logger.error("Too few common genes. Aborting.")
        return

    # ── Group by treatment class ─────────────────────────────────────
    class_groups = {}
    for ds in datasets:
        tc = ds["treatment_class"]
        class_groups.setdefault(tc, []).append(ds)

    logger.info("\nTreatment class groups:")
    for tc, ds_list in sorted(class_groups.items()):
        total_pts = sum(ds["n_patients"] for ds in ds_list)
        logger.info(f"  {tc}: {len(ds_list)} datasets, {total_pts} patients")

    # ── Step 2: Train class-specific LightGBMs with LODO ─────────────
    logger.info("\n>>> STEP 2: Train class-specific LightGBMs (LODO)")
    all_class_results = []
    for tc, ds_list in sorted(class_groups.items()):
        total_pts = sum(ds["n_patients"] for ds in ds_list)
        n_ds = len(ds_list)
        if n_ds < 3 or total_pts < 100:
            logger.info(f"  Skipping class '{tc}': {n_ds} datasets, {total_pts} pts "
                        f"(need >=3 datasets and >=100 patients)")
            continue
        logger.info(f"\n  --- Training class-specific model for: {tc} ---")
        results = run_lodo_for_class(ds_list, tc, common_genes)
        all_class_results.extend(results)

    # Also train pooled model for comparison
    logger.info("\n>>> STEP 2b: Train POOLED model (all datasets, LODO)")
    pooled_results = run_pooled_lodo(datasets, common_genes)

    # Save per-dataset results
    lodo_rows = []
    for r in all_class_results:
        lodo_rows.append({
            "dataset_id": r["dataset_id"],
            "treatment_class": r["treatment_class"],
            "model_type": "class_specific",
            "auc": r["auc"],
            "n_test": r["n_test"],
            "n_train": r["n_train"],
            "n_train_datasets": r["n_train_datasets"],
            "resp_rate_test": r["resp_rate_test"],
        })
    for r in pooled_results:
        lodo_rows.append({
            "dataset_id": r["dataset_id"],
            "treatment_class": r["treatment_class"],
            "model_type": "pooled",
            "auc": r["auc"],
            "n_test": r["n_test"],
            "n_train": r["n_train"],
            "n_train_datasets": r["n_train_datasets"],
            "resp_rate_test": r["resp_rate_test"],
        })

    lodo_df = pd.DataFrame(lodo_rows)
    lodo_path = RESULTS / "class_specific_lodo.tsv"
    lodo_df.to_csv(lodo_path, sep="\t", index=False)
    logger.info(f"\nSaved LODO results to {lodo_path}")

    # Print class-specific vs pooled comparison
    logger.info("\n=== Class-Specific vs Pooled AUC (per-dataset) ===")
    class_df = lodo_df[lodo_df["model_type"] == "class_specific"]
    pooled_df = lodo_df[lodo_df["model_type"] == "pooled"]
    if len(class_df) > 0:
        for _, cr in class_df.iterrows():
            pr_match = pooled_df[pooled_df["dataset_id"] == cr["dataset_id"]]
            pooled_auc_str = f"{pr_match.iloc[0]['auc']:.3f}" if len(pr_match) > 0 else "N/A"
            logger.info(f"  {cr['dataset_id']} ({cr['treatment_class']}): "
                        f"class={cr['auc']:.3f}, pooled={pooled_auc_str}")

    # ── Step 3: Routing ensemble ─────────────────────────────────────
    logger.info("\n>>> STEP 3: Routing ensemble")
    ensemble_results = compute_routing_ensemble(pooled_results, all_class_results)

    routing_df = pd.DataFrame(ensemble_results)
    routing_path = RESULTS / "routing_evaluation.tsv"
    routing_df.to_csv(routing_path, sep="\t", index=False)
    logger.info(f"Saved routing evaluation to {routing_path}")

    # ── Step 4: I-SPY2 arm-level evaluation ──────────────────────────
    logger.info("\n>>> STEP 4: I-SPY2 arm-level evaluation")
    ispy2_df = evaluate_ispy2_arms(pooled_results, all_class_results, ensemble_results)
    if not ispy2_df.empty:
        ispy2_path = RESULTS / "ispy2_arm_evaluation.tsv"
        ispy2_df.to_csv(ispy2_path, sep="\t", index=False)
        logger.info(f"Saved I-SPY2 arm evaluation to {ispy2_path}")
        logger.info("\n=== I-SPY2 Per-Arm Results ===")
        for _, row in ispy2_df.iterrows():
            logger.info(f"  {row['dataset_id']}: pooled={row['pooled_auc']:.3f}, "
                        f"class={row.get('class_auc', 'N/A')}, "
                        f"routed={row['routed_auc']:.3f}")
    else:
        logger.info("No I-SPY2 arm results to report.")
        # Create empty file
        pd.DataFrame(columns=["dataset_id", "treatment_class", "pooled_auc",
                               "class_auc", "routed_auc"]).to_csv(
            RESULTS / "ispy2_arm_evaluation.tsv", sep="\t", index=False)

    # ── Step 5: Feature importance comparison ────────────────────────
    logger.info("\n>>> STEP 5: Feature importance comparison")
    feat_df = extract_feature_importances(class_groups, common_genes)
    if not feat_df.empty:
        feat_path = RESULTS / "class_specific_feature_importances.tsv"
        feat_df.to_csv(feat_path, sep="\t", index=False)
        logger.info(f"Saved feature importances to {feat_path}")
        logger.info("\n=== Top 5 Features per Class ===")
        for tc in feat_df["treatment_class"].unique():
            top5 = feat_df[feat_df["treatment_class"] == tc].head(5)
            genes = ", ".join(f"{r['gene']}({r['importance']})" for _, r in top5.iterrows())
            logger.info(f"  {tc}: {genes}")
    else:
        pd.DataFrame(columns=["treatment_class", "rank", "gene", "importance"]).to_csv(
            RESULTS / "class_specific_feature_importances.tsv", sep="\t", index=False)

    # ── Step 6: Summary table ────────────────────────────────────────
    logger.info("\n>>> STEP 6: Summary comparison table")
    summary_df = build_summary_table(ensemble_results)
    if not summary_df.empty:
        summary_path = RESULTS / "treatment_class_architecture_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        logger.info(f"Saved summary to {summary_path}")
        logger.info("\n" + "=" * 80)
        logger.info("TREATMENT CLASS ARCHITECTURE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Class':<20} {'Pooled AUC':>12} {'Class AUC':>12} {'Routed AUC':>12} {'N_datasets':>12}")
        logger.info("-" * 68)
        for _, row in summary_df.iterrows():
            pooled = f"{row['pooled_auc_mean']:.3f}" if pd.notna(row['pooled_auc_mean']) else "N/A"
            cls = f"{row['class_auc_mean']:.3f}" if pd.notna(row['class_auc_mean']) else "N/A"
            routed = f"{row['routed_auc_mean']:.3f}" if pd.notna(row['routed_auc_mean']) else "N/A"
            logger.info(f"{row['treatment_class']:<20} {pooled:>12} {cls:>12} {routed:>12} {row['n_datasets']:>12}")
        logger.info("=" * 80)

    # ── Leakage check ────────────────────────────────────────────────
    logger.info("\n>>> LEAKAGE CHECK")
    all_results_for_check = all_class_results + pooled_results
    flagged = check_for_leakage(all_results_for_check)
    if flagged:
        logger.warning(f"WARNING: {len(flagged)} results with AUC > 0.70 detected!")
        for f in flagged:
            logger.warning(f"  {f['dataset_id']} ({f.get('model_type','?')}): "
                          f"AUC={f.get('auc', '?'):.3f}")
        logger.warning("Investigating potential leakage sources:")
        logger.warning("  - Per-dataset z-scoring used (safe)")
        logger.warning("  - No ComBat with response covariate (safe)")
        logger.warning("  - LODO train/test split is dataset-level (safe)")
        for f in flagged:
            n_test = f.get("n_test", 0)
            resp_rate = f.get("resp_rate_test", 0)
            logger.warning(f"  {f['dataset_id']}: n_test={n_test}, resp_rate={resp_rate:.2f}")
            if n_test < 30:
                logger.warning(f"    -> Small test set ({n_test}): AUC may be unreliable")
            if resp_rate < 0.1 or resp_rate > 0.9:
                logger.warning(f"    -> Extreme class imbalance: AUC may be inflated")
    else:
        logger.info("No AUC > 0.70 detected. No leakage concerns.")

    # ── Final answer to the core question ────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("ANSWER TO CORE QUESTION")
    logger.info("=" * 80)
    if not summary_df.empty:
        overall = summary_df[summary_df["treatment_class"] == "OVERALL"]
        if len(overall) > 0:
            pooled_overall = overall.iloc[0]["pooled_auc_mean"]
            class_overall = overall.iloc[0]["class_auc_mean"]
            routed_overall = overall.iloc[0]["routed_auc_mean"]

            delta_class = class_overall - pooled_overall if pd.notna(class_overall) else 0
            delta_routed = routed_overall - pooled_overall if pd.notna(routed_overall) else 0

            logger.info(f"Pooled LODO AUC:          {pooled_overall:.3f}")
            if pd.notna(class_overall):
                logger.info(f"Class-specific LODO AUC:  {class_overall:.3f} (delta: {delta_class:+.3f})")
            logger.info(f"Routed ensemble AUC:      {routed_overall:.3f} (delta: {delta_routed:+.3f})")

            if abs(delta_class) < 0.02 and abs(delta_routed) < 0.02:
                logger.info("\nCONCLUSION: Treatment-class stratification provides minimal improvement.")
                logger.info("The AUC ceiling is likely driven by expression-alone limitations,")
                logger.info("not by pooling across treatment classes.")
            elif delta_class > 0.03:
                logger.info("\nCONCLUSION: Class-specific models show meaningful improvement.")
                logger.info("The pooled model was indeed losing treatment-specific signal.")
            else:
                logger.info("\nCONCLUSION: Mixed results. Some classes benefit, others do not.")
                logger.info("Both treatment-specific biology and expression limits contribute.")

    logger.info("\n>>> ALL DONE.")


if __name__ == "__main__":
    main()
