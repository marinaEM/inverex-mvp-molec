"""
Multi-modal patient features ablation study.

Breaks the AUC 0.60 expression ceiling by adding mutation/CNV/clinical features.

Configs evaluated (all with full LODO):
  A: expression only (212 landmark genes)           -- baseline ~0.610
  B: expression + mutation gene status               -- +mutation
  C: expression + mutations + CNV/biomarkers         -- +CNV
  D: expression + mutations + CNV + clinical (ER/HER2/PR)  -- full multimodal

Also stratified by:
  - Treatment class
  - Whether held-out dataset has mutation data vs not

Usage:
    pixi run python -m src.models.multimodal_ablation
"""

import logging
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

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

for d in [DATA_CACHE, DATA_METADATA, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── LightGBM params (same as existing pipeline) ─────────────────────────
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

# ── Feature group definitions ────────────────────────────────────────────
MUTATION_GENE_COLS = [
    "TP53_mutated", "PIK3CA_mutated", "BRCA1_mutated", "BRCA2_mutated",
    "PTEN_mutated", "CDH1_mutated", "ESR1_mutated", "ERBB2_mutated",
]
PATHWAY_COLS = ["n_mutations_pi3k", "n_mutations_ddr"]
CNV_COLS = ["ERBB2_amplified"]
CLINICAL_COLS = ["ER_positive", "HER2_positive", "PR_positive"]

CONFIG_FEATURES = {
    "A_expr_only": [],
    "B_expr_mut": MUTATION_GENE_COLS + PATHWAY_COLS,
    "C_expr_mut_cnv": MUTATION_GENE_COLS + PATHWAY_COLS + CNV_COLS,
    "D_expr_mut_cnv_clin": MUTATION_GENE_COLS + PATHWAY_COLS + CNV_COLS + CLINICAL_COLS,
}


# ======================================================================
# Data loading (replicates treatment_class_models patterns)
# ======================================================================

def load_landmark_genes() -> list:
    """Load L1000 landmark gene symbols."""
    gi = pd.read_csv(DATA_CACHE / "geneinfo_beta_input.txt", sep="\t")
    genes = gi["gene_symbol"].dropna().unique().tolist()
    logger.info(f"Loaded {len(genes)} landmark genes")
    return genes


def _align_expr_labels_ctrdb(geo_id: str):
    expr_path = DATA_RAW / "ctrdb" / geo_id / f"{geo_id}_expression.parquet"
    label_path = DATA_RAW / "ctrdb" / geo_id / "response_labels.parquet"
    if not expr_path.exists() or not label_path.exists():
        return None, None
    expr = pd.read_parquet(expr_path)
    labels = pd.read_parquet(label_path)
    if "response" not in labels.columns:
        return None, None
    common = expr.index.intersection(labels.index)
    if len(common) < 10:
        return None, None
    return expr.loc[common], labels.loc[common, "response"]


def _align_expr_labels_ispy2():
    expr = pd.read_parquet(DATA_RAW / "ispy2" / "GSE194040_expression.parquet")
    labels = pd.read_parquet(DATA_RAW / "ispy2" / "response_labels.parquet")
    assert len(expr) == len(labels)
    return expr.reset_index(drop=True), labels.reset_index(drop=True)


def _align_expr_labels_brightness():
    expr = pd.read_parquet(DATA_RAW / "brightness" / "GSE164458_expression.parquet")
    labels = pd.read_parquet(DATA_RAW / "brightness" / "response_labels.parquet")
    assert len(expr) == len(labels)
    return expr.reset_index(drop=True), labels.reset_index(drop=True)


def _align_expr_labels_durva():
    expr = pd.read_parquet(DATA_RAW / "durva_olap_breast" / "GSE173839_expression.parquet")
    labels = pd.read_parquet(DATA_RAW / "durva_olap_breast" / "response_labels.parquet")
    labels_indexed = labels.set_index("sample_id")
    common = expr.index.intersection(labels_indexed.index)
    if len(common) < 10:
        return None, None
    labels_indexed = labels_indexed.loc[common]
    labels_indexed["char_pcr"] = pd.to_numeric(labels_indexed["char_pcr"], errors="coerce")
    valid = labels_indexed["char_pcr"].isin([0, 1])
    labels_indexed = labels_indexed[valid]
    common = labels_indexed.index
    if len(common) < 10:
        return None, None
    return expr.loc[common], labels_indexed["char_pcr"].astype(int)


def _align_expr_labels_hoogstraat2():
    expr = pd.read_parquet(DATA_RAW / "hoogstraat_2" / "GSE192341_expression.parquet")
    labels = pd.read_parquet(DATA_RAW / "hoogstraat_2" / "response_labels.parquet")
    pcr_col = "char_pathological_complete_response_(pcr)"
    if pcr_col not in labels.columns:
        return None, None
    labels["response"] = (labels[pcr_col] == "pCR").astype(int)
    n = min(len(expr), len(labels))
    return expr.iloc[:n].reset_index(drop=True), labels.iloc[:n]["response"].reset_index(drop=True)


def get_ctrdb_treatment_class(geo_id: str) -> str:
    inv_path = DATA_METADATA / "all_datasets_inventory.tsv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path, sep="\t")
        match = inv[(inv["source"] == "ctrdb") & (inv["geo_accession"] == geo_id)]
        if len(match) > 0:
            tc = match.iloc[0]["treatment_class"]
            class_map = {"chemotherapy": "chemo", "endocrine": "endocrine",
                         "targeted": "her2_targeted"}
            return class_map.get(tc, tc)
    return "chemo"


# ── I-SPY2 arm classification ───────────────────────────────────────────
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
}


# ======================================================================
# Mutation feature loader
# ======================================================================

def _extract_mutation_features_for_ctrdb(geo_id: str) -> pd.DataFrame:
    """Extract mutation features from GEO SOFT for a CTR-DB dataset."""
    from src.features.mutation_features import (
        _extract_from_geo_soft, _add_pathway_burdens, ALL_FEATURE_COLS,
    )
    soft_dir = DATA_RAW / "ctrdb" / geo_id
    df = _extract_from_geo_soft(geo_id, soft_dir)
    if len(df) > 0:
        df = _add_pathway_burdens(df)
    return df


def _extract_mutation_features_for_ispy2(arm_mask: pd.Series) -> pd.DataFrame:
    """Extract mutation features from I-SPY2 labels for a specific arm."""
    from src.features.mutation_features import (
        _extract_from_labels_parquet, _add_pathway_burdens, ALL_FEATURE_COLS,
    )
    labels = pd.read_parquet(DATA_RAW / "ispy2" / "response_labels.parquet")
    labels = labels[arm_mask.values]
    df = _extract_from_labels_parquet(labels, "ispy2")
    if len(df) > 0:
        df = _add_pathway_burdens(df)
    # Reset index to match positional expression alignment
    df = df.reset_index(drop=True)
    return df


def _extract_mutation_features_for_brightness() -> pd.DataFrame:
    """Extract mutation features for BrighTNess (all TNBC)."""
    from src.features.mutation_features import ALL_FEATURE_COLS
    labels = pd.read_parquet(DATA_RAW / "brightness" / "response_labels.parquet")
    n = len(labels)
    df = pd.DataFrame(index=range(n), columns=ALL_FEATURE_COLS, dtype=float)
    df["ER_positive"] = 0.0
    df["HER2_positive"] = 0.0
    df["PR_positive"] = 0.0
    df["ERBB2_amplified"] = 0.0
    return df


def _extract_mutation_features_for_durva() -> pd.DataFrame:
    """Extract mutation features for durva+olap dataset."""
    from src.features.mutation_features import (
        _extract_from_labels_parquet, _add_pathway_burdens, ALL_FEATURE_COLS,
    )
    labels = pd.read_parquet(DATA_RAW / "durva_olap_breast" / "response_labels.parquet")
    labels_indexed = labels.set_index("sample_id")
    labels_indexed["char_pcr"] = pd.to_numeric(labels_indexed["char_pcr"], errors="coerce")
    valid = labels_indexed["char_pcr"].isin([0, 1])
    labels_indexed = labels_indexed[valid]
    # Create a labels-like DF for extraction
    df = _extract_from_labels_parquet(labels, "durva_olap")
    if len(df) > 0:
        df = _add_pathway_burdens(df)
    # Align to valid samples
    if "sample_id" in labels.columns:
        valid_ids = labels_indexed.index
        if df.index.name == "sample_id" or (hasattr(df.index, 'name') and df.index.name == "sample_id"):
            df = df.loc[df.index.intersection(valid_ids)]
    return df


def _extract_mutation_features_for_hoogstraat2() -> pd.DataFrame:
    """Extract mutation features for Hoogstraat 2."""
    from src.features.mutation_features import (
        _extract_from_labels_parquet, _add_pathway_burdens, ALL_FEATURE_COLS,
    )
    labels = pd.read_parquet(DATA_RAW / "hoogstraat_2" / "response_labels.parquet")
    df = _extract_from_labels_parquet(labels, "hoogstraat_2")
    if len(df) > 0:
        df = _add_pathway_burdens(df)
    df = df.reset_index(drop=True)
    return df


# ======================================================================
# Load all datasets with mutation features
# ======================================================================

def load_all_datasets_with_features(landmark_genes: list) -> list:
    """
    Load all datasets, z-score expression on landmark genes, and attach
    mutation/CNV/clinical features.

    Returns a list of dicts:
        dataset_id, treatment_class, source,
        expression (DataFrame), response (Series),
        mutation_features (DataFrame), n_patients, n_genes,
        has_mutation_data (bool)
    """
    datasets = []
    inv = pd.read_csv(DATA_METADATA / "all_datasets_inventory.tsv", sep="\t")
    ctrdb_geos = inv[inv["source"] == "ctrdb"]["geo_accession"].tolist()

    # ── CTR-DB ──
    for geo_id in ctrdb_geos:
        expr, resp = _align_expr_labels_ctrdb(geo_id)
        if expr is None:
            continue
        tc = get_ctrdb_treatment_class(geo_id)
        mut_df = _extract_mutation_features_for_ctrdb(geo_id)

        # Align mutation features to expression samples
        common_samples = expr.index.intersection(mut_df.index)
        if len(common_samples) > 0:
            mut_aligned = mut_df.loc[common_samples].copy()
            expr_aligned = expr.loc[common_samples]
            resp_aligned = resp.loc[common_samples]
        else:
            # No mutation data available for this dataset
            mut_aligned = pd.DataFrame(
                np.nan, index=expr.index,
                columns=mut_df.columns if len(mut_df.columns) > 0 else
                    MUTATION_GENE_COLS + PATHWAY_COLS + CNV_COLS + CLINICAL_COLS,
            )
            expr_aligned = expr
            resp_aligned = resp

        datasets.append({
            "dataset_id": f"ctrdb_{geo_id}",
            "geo_accession": geo_id,
            "source": "ctrdb",
            "treatment_class": tc,
            "expression": expr_aligned,
            "response": resp_aligned,
            "mutation_features": mut_aligned,
        })
        logger.info(f"  CTR-DB {geo_id}: n={len(resp_aligned)}, class={tc}")

    # ── I-SPY2 arms ──
    expr_ispy2, labels_ispy2 = _align_expr_labels_ispy2()
    for arm_name, arm_class in ISPY2_ARM_CLASS.items():
        mask = labels_ispy2["char_arm"] == arm_name
        if mask.sum() < 10:
            continue
        arm_expr = expr_ispy2.loc[mask].reset_index(drop=True)
        arm_resp = labels_ispy2.loc[mask, "response"].reset_index(drop=True).astype(int)
        if arm_resp.nunique() < 2:
            continue

        mut_df = _extract_mutation_features_for_ispy2(mask)
        # Align by position (both reset to 0-based index)
        n = min(len(arm_expr), len(mut_df))
        mut_df = mut_df.iloc[:n].reset_index(drop=True)
        arm_expr = arm_expr.iloc[:n].reset_index(drop=True)
        arm_resp = arm_resp.iloc[:n].reset_index(drop=True)

        safe_name = arm_name.replace(" ", "_").replace("+", "plus")
        datasets.append({
            "dataset_id": f"ispy2_{safe_name}",
            "geo_accession": "GSE194040",
            "source": "ispy2",
            "treatment_class": arm_class,
            "expression": arm_expr,
            "response": arm_resp,
            "mutation_features": mut_df,
        })
        logger.info(f"  I-SPY2 {arm_name}: n={len(arm_resp)}, class={arm_class}")

    # ── BrighTNess ──
    expr_bright, labels_bright = _align_expr_labels_brightness()
    resp_bright = labels_bright["response"].astype(int)
    if resp_bright.nunique() >= 2:
        mut_df = _extract_mutation_features_for_brightness()
        n = min(len(expr_bright), len(mut_df))
        datasets.append({
            "dataset_id": "brightness_GSE164458",
            "geo_accession": "GSE164458",
            "source": "brightness",
            "treatment_class": "dna_damage",
            "expression": expr_bright.iloc[:n].reset_index(drop=True),
            "response": resp_bright.iloc[:n].reset_index(drop=True),
            "mutation_features": mut_df.iloc[:n].reset_index(drop=True),
        })
        logger.info(f"  BrighTNess: n={n}")

    # ── Durva+Olap ──
    expr_durva, resp_durva = _align_expr_labels_durva()
    if expr_durva is not None and resp_durva.nunique() >= 2:
        mut_df = _extract_mutation_features_for_durva()
        # Align by index
        common = expr_durva.index.intersection(mut_df.index)
        if len(common) >= 10:
            datasets.append({
                "dataset_id": "durva_olap_GSE173839",
                "geo_accession": "GSE173839",
                "source": "durva_olap",
                "treatment_class": "immunotherapy",
                "expression": expr_durva.loc[common],
                "response": resp_durva.loc[common],
                "mutation_features": mut_df.loc[common],
            })
            logger.info(f"  Durva+Olap: n={len(common)}")
        else:
            # Fall back to positional
            n = min(len(expr_durva), len(mut_df))
            datasets.append({
                "dataset_id": "durva_olap_GSE173839",
                "geo_accession": "GSE173839",
                "source": "durva_olap",
                "treatment_class": "immunotherapy",
                "expression": expr_durva.reset_index(drop=True).iloc[:n],
                "response": resp_durva.reset_index(drop=True).iloc[:n],
                "mutation_features": mut_df.reset_index(drop=True).iloc[:n],
            })
            logger.info(f"  Durva+Olap: n={n} (positional)")

    # ── Hoogstraat 2 ──
    expr_hoog, resp_hoog = _align_expr_labels_hoogstraat2()
    if expr_hoog is not None and resp_hoog.nunique() >= 2:
        mut_df = _extract_mutation_features_for_hoogstraat2()
        n = min(len(expr_hoog), len(mut_df))
        datasets.append({
            "dataset_id": "hoogstraat2_GSE192341",
            "geo_accession": "GSE192341",
            "source": "hoogstraat_2",
            "treatment_class": "chemo",
            "expression": expr_hoog.iloc[:n].reset_index(drop=True),
            "response": resp_hoog.iloc[:n].reset_index(drop=True),
            "mutation_features": mut_df.iloc[:n].reset_index(drop=True),
        })
        logger.info(f"  Hoogstraat2: n={n}")

    # ── Restrict to landmark genes and per-dataset z-score ──
    logger.info(f"\nTotal datasets loaded: {len(datasets)}")
    filtered = []
    for ds in datasets:
        expr = ds["expression"]
        available_genes = [g for g in landmark_genes if g in expr.columns]
        if len(available_genes) < 50:
            logger.warning(f"  {ds['dataset_id']}: only {len(available_genes)} "
                           f"landmark genes, skipping")
            continue
        expr_lm = expr[available_genes].copy()
        expr_z = (expr_lm - expr_lm.mean()) / (expr_lm.std() + 1e-8)
        ds["expression"] = expr_z
        ds["n_genes"] = len(available_genes)
        ds["n_patients"] = len(ds["response"])
        ds["n_responders"] = int(ds["response"].sum())

        # Check if this dataset has any mutation/clinical data
        mut = ds["mutation_features"]
        all_extra_cols = MUTATION_GENE_COLS + PATHWAY_COLS + CNV_COLS + CLINICAL_COLS
        existing_extra = [c for c in all_extra_cols if c in mut.columns]
        has_any = mut[existing_extra].notna().any().any() if existing_extra else False
        ds["has_mutation_data"] = bool(has_any)

        filtered.append(ds)

    logger.info(f"Datasets after filter: {len(filtered)}")
    n_with_mut = sum(1 for ds in filtered if ds["has_mutation_data"])
    logger.info(f"Datasets with mutation/clinical data: {n_with_mut}")

    return filtered


# ======================================================================
# LODO evaluation
# ======================================================================

def _make_common_genes(datasets: list) -> list:
    if not datasets:
        return []
    common = set(datasets[0]["expression"].columns)
    for ds in datasets[1:]:
        common = common.intersection(ds["expression"].columns)
    return sorted(common)


def _build_feature_matrix(
    ds: dict,
    common_genes: list,
    extra_feature_cols: list,
) -> pd.DataFrame:
    """
    Build the feature matrix for a dataset by concatenating
    expression genes + extra features (mutation/CNV/clinical).

    NaN is preserved for LightGBM.
    """
    X_expr = ds["expression"][common_genes].copy()

    if not extra_feature_cols:
        return X_expr

    mut = ds["mutation_features"]
    available = [c for c in extra_feature_cols if c in mut.columns]
    if available:
        X_extra = mut[available].copy().reset_index(drop=True)
        X_extra.index = X_expr.index
    else:
        X_extra = pd.DataFrame(
            np.nan, index=X_expr.index, columns=extra_feature_cols,
        )

    # Ensure all requested columns exist (fill missing with NaN)
    for col in extra_feature_cols:
        if col not in X_extra.columns:
            X_extra[col] = np.nan

    X_extra = X_extra[extra_feature_cols]  # enforce column order
    return pd.concat([X_expr, X_extra], axis=1)


def run_lodo_ablation(
    datasets: list,
    common_genes: list,
    config_name: str,
    extra_feature_cols: list,
) -> list:
    """
    Run full LODO for one configuration.
    Returns per-dataset result dicts.
    """
    results = []
    n_datasets = len(datasets)

    for i, test_ds in enumerate(datasets):
        train_ds_list = [ds for j, ds in enumerate(datasets) if j != i]
        if len(train_ds_list) == 0:
            continue

        # Build train matrix
        X_trains = []
        y_trains = []
        for ds in train_ds_list:
            X_trains.append(_build_feature_matrix(ds, common_genes, extra_feature_cols))
            y_trains.append(ds["response"])
        X_train = pd.concat(X_trains, axis=0, ignore_index=True)
        y_train = pd.concat(y_trains, axis=0, ignore_index=True)

        # Build test matrix
        X_test = _build_feature_matrix(test_ds, common_genes, extra_feature_cols)
        y_test = test_ds["response"].reset_index(drop=True)

        # Need both classes in test set
        if len(y_test) < 10 or y_test.nunique() < 2:
            continue
        n_resp = int(y_test.sum())
        n_nonresp = len(y_test) - n_resp
        if n_resp < 3 or n_nonresp < 3:
            continue

        # Train LightGBM -- NaN left as is (LightGBM native NaN support)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        sensitivity = float(y_pred[y_test == 1].sum() / y_test.sum()) if y_test.sum() > 0 else 0.0
        specificity = float((1 - y_pred[y_test == 0]).sum() / (1 - y_test).sum()) if (1 - y_test).sum() > 0 else 0.0

        results.append({
            "config": config_name,
            "held_out_dataset": test_ds["dataset_id"],
            "treatment_class": test_ds["treatment_class"],
            "source": test_ds["source"],
            "has_mutation_data": test_ds["has_mutation_data"],
            "n_test": len(y_test),
            "n_test_resp": n_resp,
            "n_train": len(X_train),
            "auc": round(auc, 4) if not np.isnan(auc) else np.nan,
            "balanced_accuracy": round(bal_acc, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "n_features": X_test.shape[1],
        })

        logger.info(f"  [{config_name}] {test_ds['dataset_id']}: "
                     f"AUC={auc:.3f}, BalAcc={bal_acc:.3f}")

    return results


# ======================================================================
# Leakage check
# ======================================================================

def check_leakage(results_df: pd.DataFrame) -> None:
    """If mean AUC > 0.70, log a leakage warning."""
    for config in results_df["config"].unique():
        cfg_df = results_df[results_df["config"] == config]
        mean_auc = cfg_df["auc"].mean()
        if mean_auc > 0.70:
            logger.warning(
                f"LEAKAGE CHECK: config {config} has mean AUC={mean_auc:.3f} > 0.70. "
                f"Investigate potential data leakage!"
            )
            # Check if clinical features correlate with response
            flagged = cfg_df[cfg_df["auc"] > 0.85]
            for _, row in flagged.iterrows():
                logger.warning(
                    f"  HIGH AUC={row['auc']:.3f} on {row['held_out_dataset']} "
                    f"(has_mutation_data={row['has_mutation_data']})"
                )


# ======================================================================
# Summary builders
# ======================================================================

def build_ablation_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-config summary."""
    rows = []
    for config in sorted(results_df["config"].unique()):
        cfg = results_df[results_df["config"] == config]
        rows.append({
            "config": config,
            "n_datasets": len(cfg),
            "mean_auc": round(cfg["auc"].mean(), 4),
            "median_auc": round(cfg["auc"].median(), 4),
            "std_auc": round(cfg["auc"].std(), 4),
            "mean_bal_acc": round(cfg["balanced_accuracy"].mean(), 4),
            "mean_sensitivity": round(cfg["sensitivity"].mean(), 4),
            "mean_specificity": round(cfg["specificity"].mean(), 4),
            "n_features": int(cfg["n_features"].iloc[0]) if len(cfg) > 0 else 0,
        })
    return pd.DataFrame(rows)


def build_treatment_class_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-config x treatment-class summary."""
    rows = []
    for config in sorted(results_df["config"].unique()):
        for tc in sorted(results_df["treatment_class"].unique()):
            subset = results_df[
                (results_df["config"] == config) &
                (results_df["treatment_class"] == tc)
            ]
            if len(subset) == 0:
                continue
            rows.append({
                "config": config,
                "treatment_class": tc,
                "n_datasets": len(subset),
                "n_patients": subset["n_test"].sum(),
                "mean_auc": round(subset["auc"].mean(), 4),
                "median_auc": round(subset["auc"].median(), 4),
                "std_auc": round(subset["auc"].std(), 4),
            })
    return pd.DataFrame(rows)


def build_mutation_data_stratification(results_df: pd.DataFrame) -> pd.DataFrame:
    """Stratify results by whether held-out dataset has mutation data."""
    rows = []
    for config in sorted(results_df["config"].unique()):
        for has_mut in [True, False]:
            subset = results_df[
                (results_df["config"] == config) &
                (results_df["has_mutation_data"] == has_mut)
            ]
            if len(subset) == 0:
                continue
            rows.append({
                "config": config,
                "has_mutation_data_in_test": has_mut,
                "n_datasets": len(subset),
                "mean_auc": round(subset["auc"].mean(), 4),
                "median_auc": round(subset["auc"].median(), 4),
            })
    return pd.DataFrame(rows)


# ======================================================================
# MAIN
# ======================================================================

def main():
    logger.info("=" * 70)
    logger.info("INVEREX Multi-Modal Ablation Study")
    logger.info("=" * 70)

    # ── Step 1: Build mutation features ──
    logger.info("\n>>> STEP 1: Build mutation / CNV / clinical features")
    from src.features.mutation_features import build_all_mutation_features
    features, availability = build_all_mutation_features()
    features.to_parquet(DATA_CACHE / "mutation_features.parquet", index=False)
    availability.to_csv(DATA_METADATA / "multimodal_availability.tsv", sep="\t", index=False)
    logger.info(f"Availability summary:\n{availability.to_string(index=False)}")

    # ── Step 2: Load all datasets ──
    logger.info("\n>>> STEP 2: Load datasets with multimodal features")
    landmark_genes = load_landmark_genes()
    datasets = load_all_datasets_with_features(landmark_genes)

    if len(datasets) < 2:
        logger.error("Need at least 2 datasets for LODO. Aborting.")
        return

    common_genes = _make_common_genes(datasets)
    logger.info(f"Common genes: {len(common_genes)}")

    # ── Step 3: Run LODO for each config ──
    logger.info("\n>>> STEP 3: Running LODO ablation for all configs")
    all_results = []

    for config_name, extra_cols in CONFIG_FEATURES.items():
        logger.info(f"\n--- Config: {config_name} "
                     f"(+{len(extra_cols)} extra features) ---")
        results = run_lodo_ablation(datasets, common_genes, config_name, extra_cols)
        all_results.extend(results)

        # Quick summary
        if results:
            aucs = [r["auc"] for r in results if not np.isnan(r.get("auc", np.nan))]
            logger.info(f"  {config_name}: mean AUC = {np.mean(aucs):.4f} "
                         f"(n={len(aucs)} folds)")

    results_df = pd.DataFrame(all_results)

    # ── Step 4: Leakage check ──
    logger.info("\n>>> STEP 4: Leakage check")
    check_leakage(results_df)

    # ── Step 5: Build summaries ──
    logger.info("\n>>> STEP 5: Build summaries")
    ablation_summary = build_ablation_summary(results_df)
    tc_summary = build_treatment_class_summary(results_df)
    mut_strat = build_mutation_data_stratification(results_df)

    # ── Step 6: Save results ──
    logger.info("\n>>> STEP 6: Saving results")

    results_df.to_csv(RESULTS / "multimodal_ablation.tsv", sep="\t", index=False)
    logger.info(f"  Saved per-dataset results: {RESULTS / 'multimodal_ablation.tsv'}")

    tc_summary.to_csv(RESULTS / "multimodal_by_treatment_class.tsv", sep="\t", index=False)
    logger.info(f"  Saved treatment class: {RESULTS / 'multimodal_by_treatment_class.tsv'}")

    # ── Print final summary ──
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n{ablation_summary.to_string(index=False)}")

    logger.info("\n" + "=" * 70)
    logger.info("BY TREATMENT CLASS")
    logger.info("=" * 70)
    logger.info(f"\n{tc_summary.to_string(index=False)}")

    logger.info("\n" + "=" * 70)
    logger.info("MUTATION DATA STRATIFICATION")
    logger.info("=" * 70)
    logger.info(f"\n{mut_strat.to_string(index=False)}")

    # ── Delta analysis ──
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG COMPARISON (delta vs baseline A_expr_only)")
    logger.info("=" * 70)
    baseline_auc = ablation_summary[
        ablation_summary["config"] == "A_expr_only"
    ]["mean_auc"].values[0]
    for _, row in ablation_summary.iterrows():
        delta = row["mean_auc"] - baseline_auc
        sign = "+" if delta >= 0 else ""
        logger.info(f"  {row['config']}: AUC={row['mean_auc']:.4f} "
                     f"({sign}{delta:.4f} vs baseline)")


if __name__ == "__main__":
    main()
