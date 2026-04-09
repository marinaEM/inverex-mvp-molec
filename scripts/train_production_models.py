#!/usr/bin/env python
"""
INVEREX — Train Two Production Model Bundles
=============================================

Creates two frozen model bundles for the dual-mode inference pipeline:

  1. batch_model_bundle.joblib   — ComBat correction, for >=20 patient cohorts
  2. single_patient_model_bundle — Quantile normalization, for N=1 patients

Both include: LightGBM model, normalizer, gene list, feature names,
conformal calibration set, and training expression (for batch-mode ComBat).
"""

import os
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit

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
PROD_DIR = ROOT / "models" / "production"


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
# Load datasets
# =========================================================================
log.info("Loading datasets...")


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

for name, data_dir, geo_id in [
    ("ISPY2", "ispy2", "GSE194040"),
    ("BrighTNess", "brightness", "GSE164458"),
]:
    try:
        e, l = load_positional_dataset(data_dir, geo_id)
        if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
            datasets[name] = (e, l)
    except Exception:
        pass

log.info(f"Loaded {len(datasets)} datasets, {sum(len(v[1]) for v in datasets.values())} patients")

# Restrict to common L1000 genes
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
    datasets[did] = (clean_matrix((expr - means) / stds), lab)


# Pool all datasets
all_parts_x, all_parts_y, all_batch = [], [], []
for did in sorted(datasets.keys()):
    e, l = datasets[did]
    new_idx = [f"{did}__{i}" for i in range(len(e))]
    ec = e.copy(); ec.index = new_idx
    lc = l.copy(); lc.index = new_idx
    all_parts_x.append(ec)
    all_parts_y.append(lc)
    all_batch.extend([did] * len(ec))

pooled_expr = pd.concat(all_parts_x, axis=0)
pooled_labels = pd.concat(all_parts_y, axis=0)
batch_series = pd.Series(all_batch, index=pooled_expr.index)

log.info(f"Pooled: {pooled_expr.shape[0]} patients x {pooled_expr.shape[1]} genes")

PROD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================
# MODEL A: Batch mode (ComBat)
# =========================================================================
log.info("\n" + "=" * 60)
log.info("MODEL A: Batch-mode (ComBat)")
log.info("=" * 60)

try:
    from neuroCombat import neuroCombat

    covars = pd.DataFrame({"batch": batch_series.values}, index=pooled_expr.index)
    log.info("Running ComBat on all training data (expression only, no labels)...")
    combat_result = neuroCombat(
        dat=pooled_expr.values.T,
        covars=covars,
        batch_col="batch",
    )
    combat_expr = pd.DataFrame(
        combat_result["data"].T,
        columns=pooled_expr.columns,
        index=pooled_expr.index,
    )
    combat_expr = clean_matrix(combat_expr)
    log.info(f"ComBat done: {combat_expr.shape}")

    # ssGSEA on ComBat-corrected
    log.info("Computing ssGSEA (batch mode)...")
    batch_ssgsea = compute_ssgsea(combat_expr, "batch")
    log.info(f"ssGSEA: {batch_ssgsea.shape[1]} pathways")

    X_batch = pd.concat([combat_expr, batch_ssgsea], axis=1)
    X_batch = clean_matrix(X_batch)
    y_batch = pooled_labels.values.astype(int)
    batch_feature_names = list(X_batch.columns)

    log.info(f"Training batch model: {X_batch.shape}")
    batch_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    batch_model.fit(X_batch.values, y_batch)

    # Save combat params needed for new batches
    combat_params = {
        "training_expr": pooled_expr,
        "training_batches": batch_series,
    }

    from src.inference.production_pipeline import InverexPipeline
    InverexPipeline.save_bundle(
        model=batch_model,
        gene_list=common_l1000,
        feature_names=batch_feature_names,
        mode="batch",
        combat_params=combat_params,
        output_dir=str(PROD_DIR),
    )
    log.info("Batch model bundle saved.")

except Exception as e:
    log.warning(f"Batch model failed: {e}")


# =========================================================================
# MODEL B: Single-patient mode (Quantile)
# =========================================================================
log.info("\n" + "=" * 60)
log.info("MODEL B: Single-patient mode (Quantile)")
log.info("=" * 60)

from src.preprocessing.leakage_free_normalizers import RankNormalizer

normalizer = RankNormalizer(method="quantile")
quantile_expr = normalizer.transform(pooled_expr)
quantile_expr = clean_matrix(quantile_expr)
log.info(f"Quantile-normalized: {quantile_expr.shape}")

log.info("Computing ssGSEA (single mode)...")
single_ssgsea = compute_ssgsea(quantile_expr, "single")
log.info(f"ssGSEA: {single_ssgsea.shape[1]} pathways")

X_single = pd.concat([quantile_expr, single_ssgsea], axis=1)
X_single = clean_matrix(X_single)
y_single = pooled_labels.values.astype(int)
single_feature_names = list(X_single.columns)

log.info(f"Training single-patient model: {X_single.shape}")
single_model = lgb.LGBMClassifier(**LGBM_PARAMS)
single_model.fit(X_single.values, y_single)

from src.inference.production_pipeline import InverexPipeline
InverexPipeline.save_bundle(
    model=single_model,
    gene_list=common_l1000,
    feature_names=single_feature_names,
    mode="single",
    normalizer=normalizer,
    output_dir=str(PROD_DIR),
)
log.info("Single-patient model bundle saved.")


# =========================================================================
# Summary
# =========================================================================
log.info("\n" + "=" * 60)
log.info("PRODUCTION MODELS COMPLETE")
log.info("=" * 60)
log.info(f"  Output:          {PROD_DIR}")
log.info(f"  Batch features:  {len(batch_feature_names)}")
log.info(f"  Single features: {len(single_feature_names)}")
log.info(f"  Gene list:       {len(common_l1000)} L1000 genes")
log.info(f"  Patients:        {len(y_single)}")
log.info(f"  Datasets:        {len(datasets)}")

for f in sorted(PROD_DIR.iterdir()):
    log.info(f"  {f.name}: {f.stat().st_size:,} bytes")
