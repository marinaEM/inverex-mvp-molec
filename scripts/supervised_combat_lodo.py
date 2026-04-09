#!/usr/bin/env python
"""
INVEREX — Supervised ComBat LODO Evaluation (212 curated genes)
===============================================================

Tests whether using TRAINING-ONLY response labels as a ComBat covariate
can recover part of the 0.594 → 0.763 gap without leaking test labels.

For each LODO fold (held-out dataset H):
  1. Pool all datasets (T + H)
  2. Set H's response labels to NaN (hidden from ComBat)
  3. Run ComBat with T's labels as biological covariate
  4. Split corrected data, compute ssGSEA
  5. Train LightGBM on corrected T, predict on corrected H
  6. Evaluate AUC

Three strategies for handling H's missing labels:
  A. mean_imputation:  fill NaN with training response rate
  B. two_step:         fit ComBat on T only (with labels), apply to T+H
  C. no_labels:        ComBat without any covariate (baseline)

Also re-runs leaked ComBat (all labels) and quantile as baselines.
All on 212 curated genes.
"""

import os
import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "supervised_combat"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "lodo.log", mode="w"),
    ],
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


def quantile_normalize(expression_df):
    mat = expression_df.values.copy()
    n_samples, n_genes = mat.shape
    ranked = np.zeros_like(mat)
    for i in range(n_samples):
        ranked[i, :] = rankdata(mat[i, :], method="average")
    quantiles = (ranked - 0.5) / n_genes
    ranked = norm.ppf(np.clip(quantiles, 1e-7, 1 - 1e-7))
    return pd.DataFrame(ranked, index=expression_df.index, columns=expression_df.columns)


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

# =========================================================================
# Get 212 curated gene list (from v3 model)
# =========================================================================
feat_imp = pd.read_csv(ROOT / "results" / "full_retrain" / "feature_importances.tsv", sep="\t")
genes_212 = sorted([
    f.replace("gene_", "") for f in feat_imp["feature"].tolist()
    if f.startswith("gene_")
])
log.info(f"212 curated genes loaded")

# Restrict and z-score
for did in list(datasets.keys()):
    expr, lab = datasets[did]
    available = [g for g in genes_212 if g in expr.columns]
    missing = [g for g in genes_212 if g not in expr.columns]
    expr_sub = expr[available].copy()
    for g in missing:
        expr_sub[g] = 0.0
    expr_sub = expr_sub[genes_212]
    means = expr_sub.mean(axis=0)
    stds = expr_sub.std(axis=0).replace(0, 1)
    datasets[did] = (clean_matrix((expr_sub - means) / stds), lab)


# Pool all datasets
def pool_all():
    parts_x, parts_y, batch_list = [], [], []
    s2d = {}
    for did in sorted(datasets.keys()):
        e, l = datasets[did]
        idx = [f"{did}__{i}" for i in range(len(e))]
        ec = e.copy(); ec.index = idx
        lc = l.copy(); lc.index = idx
        parts_x.append(ec)
        parts_y.append(lc)
        batch_list.extend([did] * len(ec))
        for s in idx:
            s2d[s] = did
    X = pd.concat(parts_x, axis=0)
    y = pd.concat(parts_y, axis=0)
    b = pd.Series(batch_list, index=X.index)
    return X, y, b, s2d


pooled_expr, pooled_labels, batch_series, s2d = pool_all()
log.info(f"Pooled: {pooled_expr.shape}")


# =========================================================================
# ComBat helper functions
# =========================================================================
from neuroCombat import neuroCombat


def run_combat_with_covariate(expr, batches, covariate):
    """ComBat with response as continuous covariate."""
    covars = pd.DataFrame({
        "batch": batches.values,
        "response": covariate.values.astype(float),
    }, index=expr.index)
    result = neuroCombat(
        dat=expr.values.T,
        covars=covars,
        batch_col="batch",
        continuous_cols=["response"],
    )
    return clean_matrix(pd.DataFrame(
        result["data"].T, columns=expr.columns, index=expr.index,
    ))


def run_combat_no_covariate(expr, batches):
    """ComBat without any biological covariate."""
    covars = pd.DataFrame({"batch": batches.values}, index=expr.index)
    result = neuroCombat(
        dat=expr.values.T,
        covars=covars,
        batch_col="batch",
    )
    return clean_matrix(pd.DataFrame(
        result["data"].T, columns=expr.columns, index=expr.index,
    ))


def run_combat_two_step(all_expr, batches, response_labels):
    """
    Two-step supervised ComBat:
    1. Fit ComBat on TRAINING samples only (where response is not NaN), with labels
    2. For test samples, apply correction using parameters from step 1
       by running ComBat on all data but with training stats frozen.

    Implementation: run ComBat on training only first to get corrected training.
    Then for each test batch, standardize using that batch's own stats and
    map to the training grand mean/pooled variance.
    """
    train_mask = response_labels.notna()
    train_expr = all_expr.loc[train_mask]
    train_batches = batches.loc[train_mask]
    train_labels = response_labels.loc[train_mask]

    # Step 1: ComBat on training data with labels
    corrected_train = run_combat_with_covariate(train_expr, train_batches, train_labels)

    # Step 2: For test data, apply a frozen correction
    # Use the training ComBat output's grand mean + pooled std as reference
    grand_mean = corrected_train.mean(axis=0)
    pooled_std = corrected_train.std(axis=0).replace(0, 1)

    test_mask = ~train_mask
    test_expr = all_expr.loc[test_mask]
    test_batches = batches.loc[test_mask]

    corrected_test = test_expr.copy()
    for bid in test_batches.unique():
        bmask = test_batches == bid
        bdata = test_expr.loc[bmask]
        if len(bdata) > 1:
            bm = bdata.mean(axis=0)
            bs = bdata.std(axis=0).replace(0, 1)
            bs[bs < 1e-6] = 1.0
            standardized = (bdata - bm) / bs
            corrected_test.loc[bmask] = standardized * pooled_std + grand_mean
        else:
            corrected_test.loc[bmask] = bdata - bdata.mean(axis=1).values[0] + grand_mean.mean()

    corrected_test = clean_matrix(corrected_test)

    # Combine
    corrected_all = pd.concat([corrected_train, corrected_test], axis=0)
    corrected_all = corrected_all.loc[all_expr.index]
    return corrected_all


# =========================================================================
# LODO evaluation
# =========================================================================
log.info("=" * 60)
log.info("SUPERVISED COMBAT LODO (212 curated genes)")
log.info("=" * 60)

dataset_ids = sorted(set(s2d.values()))

configs = [
    ("leaked_all_labels",      "ComBat + all labels (leaked ceiling)"),
    ("supervised_mean_impute", "ComBat + training labels, test=mean"),
    ("supervised_two_step",    "ComBat on train only + frozen for test"),
    ("combat_no_labels",       "ComBat without labels"),
    ("quantile",               "Quantile normalization (honest)"),
]

all_results = []

for config_key, config_desc in configs:
    log.info(f"\n{'='*60}")
    log.info(f"{config_key}: {config_desc}")
    log.info(f"{'='*60}")

    fold_results = []
    t_start = time.time()

    for test_did in dataset_ids:
        train_samples = [s for s in pooled_expr.index if s2d[s] != test_did]
        test_samples = [s for s in pooled_expr.index if s2d[s] == test_did]

        train_y = pooled_labels.loc[train_samples].values.astype(int)
        test_y = pooled_labels.loc[test_samples].values.astype(int)

        if len(np.unique(test_y)) < 2 or len(np.unique(train_y)) < 2:
            continue

        try:
            if config_key == "leaked_all_labels":
                # ALL labels visible — the 0.763 scenario
                corrected = run_combat_with_covariate(
                    pooled_expr, batch_series, pooled_labels
                )
                train_corrected = corrected.loc[train_samples]
                test_corrected = corrected.loc[test_samples]

            elif config_key == "supervised_mean_impute":
                # Training labels real, test labels = training mean
                response = pooled_labels.copy().astype(float)
                train_mean = response.loc[train_samples].mean()
                response.loc[test_samples] = train_mean
                corrected = run_combat_with_covariate(
                    pooled_expr, batch_series, response
                )
                train_corrected = corrected.loc[train_samples]
                test_corrected = corrected.loc[test_samples]

            elif config_key == "supervised_two_step":
                # Fit ComBat on training only with labels, frozen apply to test
                response = pooled_labels.copy().astype(float)
                response.loc[test_samples] = np.nan
                corrected = run_combat_two_step(
                    pooled_expr, batch_series, response
                )
                train_corrected = corrected.loc[train_samples]
                test_corrected = corrected.loc[test_samples]

            elif config_key == "combat_no_labels":
                corrected = run_combat_no_covariate(pooled_expr, batch_series)
                train_corrected = corrected.loc[train_samples]
                test_corrected = corrected.loc[test_samples]

            elif config_key == "quantile":
                train_corrected = quantile_normalize(pooled_expr.loc[train_samples])
                test_corrected = quantile_normalize(pooled_expr.loc[test_samples])

        except Exception as e:
            log.warning(f"  {config_key}/{test_did}: correction failed: {e}")
            continue

        # ssGSEA
        try:
            train_ssgsea = compute_ssgsea(train_corrected, f"{config_key}_tr_{test_did}")
            test_ssgsea = compute_ssgsea(test_corrected, f"{config_key}_te_{test_did}")
            common_cols = sorted(set(train_ssgsea.columns) & set(test_ssgsea.columns))
            X_train = pd.concat([train_corrected, train_ssgsea[common_cols]], axis=1)
            X_test = pd.concat([test_corrected, test_ssgsea[common_cols]], axis=1)
        except Exception:
            X_train = train_corrected
            X_test = test_corrected

        X_train = clean_matrix(X_train).values
        X_test = clean_matrix(X_test).values

        mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
        mdl.fit(X_train, train_y)
        yp = mdl.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(test_y, yp)
        except ValueError:
            auc = np.nan

        fold_results.append({
            "config": config_key,
            "dataset_id": test_did,
            "n_test": len(test_y),
            "n_test_pos": int(test_y.sum()),
            "auc": auc,
        })
        log.info(f"  {config_key}/{test_did}: AUC={auc:.4f} (n={len(test_y)}, pos={int(test_y.sum())})")

    elapsed = time.time() - t_start
    fold_df = pd.DataFrame(fold_results)
    mean_auc = fold_df["auc"].mean()
    std_auc = fold_df["auc"].std()
    log.info(f"  → {config_key}: Mean AUC = {mean_auc:.4f} ± {std_auc:.4f} ({elapsed:.0f}s)")
    all_results.append(fold_df)


# =========================================================================
# Summary
# =========================================================================
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(OUT_DIR / "lodo_all_configs.tsv", sep="\t", index=False)

summary = (
    combined.groupby("config")["auc"]
    .agg(["mean", "std", "median", "min", "max", "count"])
    .sort_values("mean", ascending=False)
)
summary.to_csv(OUT_DIR / "summary.tsv", sep="\t")

log.info(f"\n{'='*60}")
log.info("SUPERVISED COMBAT — FULL COMPARISON (212 curated genes)")
log.info(f"{'='*60}")
for config, row in summary.iterrows():
    log.info(f"  {config:30s}  AUC = {row['mean']:.4f} ± {row['std']:.4f}")

# Gap recovery analysis
leaked_auc = summary.loc["leaked_all_labels", "mean"] if "leaked_all_labels" in summary.index else 0.763
quantile_auc = summary.loc["quantile", "mean"] if "quantile" in summary.index else 0.594
gap = leaked_auc - quantile_auc

log.info(f"\nGap analysis (leaked={leaked_auc:.3f}, quantile={quantile_auc:.3f}, gap={gap:.3f}):")
for config in ["supervised_mean_impute", "supervised_two_step", "combat_no_labels"]:
    if config in summary.index:
        auc = summary.loc[config, "mean"]
        recovery = (auc - quantile_auc) / gap * 100 if gap > 0 else 0
        log.info(f"  {config:30s}  AUC={auc:.4f}  recovery={recovery:.1f}%")

log.info(f"\nDone. Results in {OUT_DIR}/")
