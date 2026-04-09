#!/usr/bin/env python
"""
INVEREX — SHAP Explanations + Production Pipeline
==================================================

Uses the best leakage-free method (quantile normalization) to:
1. Train final model on ALL data
2. Compute global SHAP explanations
3. Save production artifacts with frozen normalizer
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

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

MIN_PATIENTS = 20
GENE_PRESENCE_THRESHOLD = 0.60
CHUNK_SIZE = 500


def clean_matrix(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_ssgsea(expression_df, label=""):
    import gseapy as gp
    expr_clean = clean_matrix(expression_df)
    n_samples = expr_clean.shape[0]
    if n_samples <= CHUNK_SIZE:
        result = gp.ssgsea(
            data=expr_clean.T, gene_sets="MSigDB_Hallmark_2020",
            outdir=None, min_size=5, no_plot=True, verbose=False,
        )
        scores = result.res2d.pivot(index="Name", columns="Term", values="NES")
        scores.index.name = None; scores.columns.name = None
    else:
        chunks = []
        for start in range(0, n_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_samples)
            chunk = expr_clean.iloc[start:end]
            log.info(f"  ssGSEA {label} chunk {start}-{end} / {n_samples}")
            result = gp.ssgsea(
                data=chunk.T, gene_sets="MSigDB_Hallmark_2020",
                outdir=None, min_size=5, no_plot=True, verbose=False,
            )
            cs = result.res2d.pivot(index="Name", columns="Term", values="NES")
            cs.index.name = None; cs.columns.name = None
            chunks.append(cs)
        scores = pd.concat(chunks, axis=0)
    scores = scores.loc[expression_df.index].astype(float)
    scores.columns = [f"ssgsea_{c}" for c in scores.columns]
    return clean_matrix(scores)


# =========================================================================
# STEP 1: Load and prepare data (same as retrain_leakage_free.py)
# =========================================================================
log.info("STEP 1: Loading datasets...")


def load_ctrdb_dataset(geo_id):
    base = ROOT / "data" / "raw" / "ctrdb" / geo_id
    expr_path = base / f"{geo_id}_expression.parquet"
    label_path = base / "response_labels.parquet"
    if not expr_path.exists() or not label_path.exists():
        return None, None
    expr = pd.read_parquet(expr_path)
    labels = pd.read_parquet(label_path)
    common = expr.index.intersection(labels.index)
    if len(common) < MIN_PATIENTS:
        return None, None
    return expr.loc[common], labels.loc[common, "response"].astype(int)


def load_positional_dataset(data_dir, geo_id):
    base = ROOT / "data" / "raw" / data_dir
    expr = pd.read_parquet(base / f"{geo_id}_expression.parquet")
    labels = pd.read_parquet(base / "response_labels.parquet")
    response = labels["response"].astype(int).values
    expr = expr.reset_index(drop=True)
    return expr, pd.Series(response, index=expr.index, name="response")


ctrdb_dir = ROOT / "data" / "raw" / "ctrdb"
ctrdb_geos = sorted(
    d.name for d in ctrdb_dir.iterdir()
    if d.is_dir() and d.name.startswith("GSE")
)
ctrdb_geos = [g for g in ctrdb_geos if g != "GSE194040"]

datasets = {}
for geo_id in ctrdb_geos:
    expr, labels = load_ctrdb_dataset(geo_id)
    if expr is not None and labels is not None:
        if labels.nunique() >= 2 and len(labels) >= MIN_PATIENTS:
            datasets[geo_id] = (expr, labels)

for name, data_dir, geo_id in [("ISPY2", "ispy2", "GSE194040"), ("BrighTNess", "brightness", "GSE164458")]:
    try:
        e, l = load_positional_dataset(data_dir, geo_id)
        if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
            datasets[name] = (e, l)
    except Exception:
        pass

log.info(f"Loaded {len(datasets)} datasets, {sum(len(v[1]) for v in datasets.values())} patients")

# Restrict to L1000 genes
gene_info = pd.read_csv(ROOT / "data" / "cache" / "geneinfo_beta_input.txt", header=0)
l1000_genes = gene_info.iloc[:, 0].dropna().astype(str).tolist()

gene_counts = {}
for did, (expr, _) in datasets.items():
    for g in expr.columns:
        gene_counts[g] = gene_counts.get(g, 0) + 1

n_datasets = len(datasets)
threshold_count = int(np.ceil(GENE_PRESENCE_THRESHOLD * n_datasets))
common_l1000 = sorted(
    g for g in l1000_genes if gene_counts.get(g, 0) >= threshold_count
)
log.info(f"Common L1000 genes: {len(common_l1000)}")

for did in list(datasets.keys()):
    expr, lab = datasets[did]
    available = [g for g in common_l1000 if g in expr.columns]
    missing = [g for g in common_l1000 if g not in expr.columns]
    expr_sub = expr[available].copy()
    for g in missing:
        expr_sub[g] = 0.0
    datasets[did] = (expr_sub[common_l1000], lab)

# Per-dataset z-score
for did in datasets:
    expr, lab = datasets[did]
    means = expr.mean(axis=0)
    stds = expr.std(axis=0).replace(0, 1)
    expr_z = (expr - means) / stds
    datasets[did] = (clean_matrix(expr_z), lab)

# =========================================================================
# STEP 2: Apply quantile (best leakage-free method) + ssGSEA
# =========================================================================
log.info("STEP 2: Applying quantile normalization + ssGSEA on all data...")

from src.preprocessing.leakage_free_normalizers import RankNormalizer

# Pool all datasets
all_expr_parts, all_label_parts, all_batch = [], [], []
for did in sorted(datasets.keys()):
    expr, lab = datasets[did]
    new_idx = [f"{did}__{i}" for i in range(len(expr))]
    ec = expr.copy(); ec.index = new_idx
    lc = lab.copy(); lc.index = new_idx
    all_expr_parts.append(ec)
    all_label_parts.append(lc)
    all_batch.extend([did] * len(ec))

pooled_expr = pd.concat(all_expr_parts, axis=0)
pooled_labels = pd.concat(all_label_parts, axis=0)
batch_series = pd.Series(all_batch, index=pooled_expr.index)

# Apply quantile normalization
normalizer = RankNormalizer(method="quantile")
normalizer.fit(pooled_expr, batch_series)  # no-op for rank, but consistent API
corrected_expr = normalizer.transform(pooled_expr, batch_series)
corrected_expr = clean_matrix(corrected_expr)
log.info(f"Quantile-corrected: {corrected_expr.shape}")

# Compute ssGSEA
log.info("Computing ssGSEA...")
ssgsea_scores = compute_ssgsea(corrected_expr, "quantile_all")
log.info(f"ssGSEA: {ssgsea_scores.shape[1]} pathways")

# Build feature matrix
X_full = pd.concat([corrected_expr, ssgsea_scores], axis=1)
X_full = clean_matrix(X_full)
y_full = pooled_labels.values.astype(int)
feature_names = list(X_full.columns)
log.info(f"Feature matrix: {X_full.shape}")

# =========================================================================
# STEP 3: Train final model
# =========================================================================
log.info("STEP 3: Training final LightGBM model on all data...")
model = lgb.LGBMClassifier(**LGBM_PARAMS)
model.fit(X_full.values, y_full)
log.info(f"Model trained: {model.n_features_in_} features, {model.n_estimators} trees")

# Save feature importances
importances = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
importances.to_csv("results/leakage_free/feature_importances.tsv", sep="\t", index=False)

log.info("Top 20 features (LightGBM gain):")
for _, r in importances.head(20).iterrows():
    log.info(f"  {r['feature']:30s}  importance={r['importance']}")

# =========================================================================
# STEP 4: SHAP explanations
# =========================================================================
log.info("STEP 4: Computing SHAP explanations...")

from src.explanations.shap_explanations import (
    compute_global_shap,
    compute_interaction_shap,
)

shap_dir = "results/shap/"
importance_df, shap_values = compute_global_shap(
    model, X_full, feature_names, output_dir=shap_dir
)

# Interaction SHAP (subsample to 500 for speed)
log.info("Computing SHAP interactions (subsampled)...")
try:
    pairs_df = compute_interaction_shap(
        model, X_full, feature_names,
        output_dir=shap_dir, max_samples=500,
    )
except Exception as e:
    log.warning(f"SHAP interaction failed: {e}")

# =========================================================================
# STEP 5: Save production artifacts
# =========================================================================
log.info("STEP 5: Saving production artifacts...")

from src.inference.production_pipeline import InverexInferencePipeline

prod_dir = "models/production/"
InverexInferencePipeline.save_production_artifacts(
    model=model,
    normalizer=normalizer,
    gene_list=common_l1000,
    feature_names=feature_names,
    output_dir=prod_dir,
)

# Also save the model in results
joblib.dump({
    "model": model,
    "normalizer": normalizer,
    "gene_list": common_l1000,
    "feature_names": feature_names,
    "correction_method": "quantile",
    "lgbm_params": LGBM_PARAMS,
    "n_datasets": len(datasets),
    "n_patients": len(y_full),
    "leakage_free_auc": 0.603,
}, "results/leakage_free/final_model_bundle.joblib")

log.info("\n" + "=" * 60)
log.info("COMPLETE")
log.info("=" * 60)
log.info(f"  Best method:    quantile (inverse-normal rank)")
log.info(f"  LODO AUC:       0.603 (leakage-free)")
log.info(f"  Features:       {len(feature_names)}")
log.info(f"  Patients:       {len(y_full)}")
log.info(f"  SHAP:           {shap_dir}")
log.info(f"  Production:     {prod_dir}")
