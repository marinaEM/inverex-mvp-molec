#!/usr/bin/env python
"""
INVEREX — Batch-Mode ComBat LODO Validation
============================================

Validates that ComBat is legitimate when batch context is available.

For each LODO fold (held-out dataset H):
  1. Combine training batches T + held-out batch H into one matrix
  2. Run ComBat on (T + H) using ONLY expression (NO labels as covariate)
  3. Split corrected data back into T and H
  4. Compute ssGSEA on corrected data
  5. Train LightGBM on corrected T with labels
  6. Predict on corrected H
  7. Evaluate AUC

This is NOT the same as the leaked ComBat (which used labels as covariate
and ran once globally). Here ComBat is re-estimated per fold using only
expression features — the held-out batch provides its own batch statistics
for correction, which is exactly what happens in real deployment.

If batch-mode AUC ≈ 0.66 → ComBat is legitimate for batch deployment.
If batch-mode AUC ≈ 0.60 → Even batch-mode doesn't help. Use quantile only.
"""

import os
import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "batch_mode"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "retrain.log", mode="w"),
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


def run_combat_expression_only(expression_df, batch_labels):
    """Run neuroCombat using expression features only — NO labels."""
    from neuroCombat import neuroCombat

    covars = pd.DataFrame({"batch": batch_labels.values}, index=expression_df.index)

    combat_result = neuroCombat(
        dat=expression_df.values.T,
        covars=covars,
        batch_col="batch",
    )
    corrected = pd.DataFrame(
        combat_result["data"].T,
        columns=expression_df.columns,
        index=expression_df.index,
    )
    return clean_matrix(corrected)


# =========================================================================
# Load datasets (reuse from retrain_leakage_free.py)
# =========================================================================
log.info("=" * 70)
log.info("Loading datasets")
log.info("=" * 70)


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
treatment_map = {}

for geo_id in ctrdb_geos:
    expr, labels = load_ctrdb_dataset(geo_id)
    if expr is not None and labels is not None:
        if labels.nunique() >= 2 and len(labels) >= MIN_PATIENTS:
            datasets[geo_id] = (expr, labels)
            treatment_map[geo_id] = "chemo"

for name, data_dir, geo_id in [
    ("ISPY2", "ispy2", "GSE194040"),
    ("BrighTNess", "brightness", "GSE164458"),
]:
    try:
        e, l = load_positional_dataset(data_dir, geo_id)
        if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
            datasets[name] = (e, l)
            treatment_map[name] = "combination" if name == "ISPY2" else "parp"
            log.info(f"  {name}: {len(l)} patients")
    except Exception as exc:
        log.warning(f"  {name} failed: {exc}")

total_patients = sum(len(v[1]) for v in datasets.values())
log.info(f"Loaded {len(datasets)} datasets, {total_patients} total patients")

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
    log.info(f"  {did}: z-scored ({expr_z.shape})")


# =========================================================================
# Batch-mode ComBat LODO
# =========================================================================
log.info("=" * 70)
log.info("BATCH-MODE ComBat LODO (expression-only, re-estimated per fold)")
log.info("=" * 70)

dataset_ids = sorted(datasets.keys())
results = []
t_global = time.time()

for test_did in dataset_ids:
    t0 = time.time()
    train_dids = [d for d in dataset_ids if d != test_did]

    # Pool ALL datasets (train + test) for ComBat
    all_parts_x = []
    all_parts_y = []
    all_batch = []
    sample_to_dataset = {}

    for d in dataset_ids:
        e, l = datasets[d]
        new_idx = [f"{d}__{i}" for i in range(len(e))]
        ec = e.copy(); ec.index = new_idx
        lc = l.copy(); lc.index = new_idx
        all_parts_x.append(ec)
        all_parts_y.append(lc)
        all_batch.extend([d] * len(ec))
        for s in new_idx:
            sample_to_dataset[s] = d

    combined_expr = pd.concat(all_parts_x, axis=0)
    combined_labels = pd.concat(all_parts_y, axis=0)
    combined_batch = pd.Series(all_batch, index=combined_expr.index)

    # Run ComBat on combined (train + test) — expression only, NO labels
    try:
        corrected_combined = run_combat_expression_only(combined_expr, combined_batch)
    except Exception as exc:
        log.warning(f"  ComBat failed for fold {test_did}: {exc}")
        continue

    # Split corrected data back
    train_mask = [s for s in corrected_combined.index if sample_to_dataset[s] != test_did]
    test_mask = [s for s in corrected_combined.index if sample_to_dataset[s] == test_did]

    train_expr_corrected = corrected_combined.loc[train_mask]
    test_expr_corrected = corrected_combined.loc[test_mask]
    train_y = combined_labels.loc[train_mask].values.astype(int)
    test_y = combined_labels.loc[test_mask].values.astype(int)

    if len(np.unique(test_y)) < 2 or len(np.unique(train_y)) < 2:
        log.info(f"  batch_combat/{test_did}: skipped (single class)")
        continue

    # Compute ssGSEA on corrected data
    try:
        train_ssgsea = compute_ssgsea(train_expr_corrected, f"train_{test_did}")
        test_ssgsea = compute_ssgsea(test_expr_corrected, f"test_{test_did}")
        common_ssgsea_cols = sorted(set(train_ssgsea.columns) & set(test_ssgsea.columns))
        train_X = pd.concat([train_expr_corrected, train_ssgsea[common_ssgsea_cols]], axis=1)
        test_X = pd.concat([test_expr_corrected, test_ssgsea[common_ssgsea_cols]], axis=1)
    except Exception:
        train_X = train_expr_corrected
        test_X = test_expr_corrected

    # Train and predict
    train_X_clean = clean_matrix(train_X).values
    test_X_clean = clean_matrix(test_X).values

    mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
    mdl.fit(train_X_clean, train_y)
    yp = mdl.predict_proba(test_X_clean)[:, 1]

    try:
        auc = roc_auc_score(test_y, yp)
    except ValueError:
        auc = np.nan

    elapsed = time.time() - t0
    results.append({
        "correction_method": "batch_combat",
        "dataset_id": test_did,
        "n_train": len(train_y),
        "n_test": len(test_y),
        "n_test_pos": int(test_y.sum()),
        "n_features": train_X_clean.shape[1],
        "auc": auc,
        "treatment_class": treatment_map.get(test_did, "unknown"),
    })
    log.info(
        f"  batch_combat/{test_did}: AUC={auc:.4f} "
        f"(n={len(test_y)}, pos={int(test_y.sum())}, {elapsed:.0f}s)"
    )

# =========================================================================
# Summary
# =========================================================================
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "batch_mode_combat_lodo.tsv", sep="\t", index=False)

# Also copy to top-level results
results_df.to_csv(ROOT / "results" / "batch_mode_combat_lodo.tsv", sep="\t", index=False)

mean_auc = results_df["auc"].mean()
std_auc = results_df["auc"].std()
median_auc = results_df["auc"].median()

log.info(f"\n{'='*60}")
log.info(f"BATCH-MODE ComBat LODO RESULTS")
log.info(f"{'='*60}")
log.info(f"  Mean AUC:   {mean_auc:.4f} ± {std_auc:.4f}")
log.info(f"  Median AUC: {median_auc:.4f}")
log.info(f"  N folds:    {len(results_df)}")
log.info(f"  Time:       {time.time()-t_global:.0f}s")
log.info(f"")
log.info(f"  Compare to:")
log.info(f"    Leaked ComBat (global, labels as covariate): 0.666")
log.info(f"    Quantile (single-patient, leakage-free):     0.603")
log.info(f"    None (per-dataset z-score):                  0.588")

# Summary TSV
summary = pd.DataFrame([{
    "method": "batch_combat_per_fold",
    "mean_auc": mean_auc,
    "std_auc": std_auc,
    "median_auc": median_auc,
    "n_folds": len(results_df),
}])
summary.to_csv(OUT_DIR / "batch_mode_summary.tsv", sep="\t", index=False)

log.info(f"\nDone. Results in {OUT_DIR}/")
