#!/usr/bin/env python
"""
INVEREX — Leakage-Free Batch Correction Retrain
================================================

The E_full_combat config achieved 0.761 LODO AUC, but ComBat was applied to
ALL samples before the LODO split, leaking held-out batch statistics.

This script moves batch correction INSIDE the LODO loop:
  1. Split train/test by dataset
  2. Fit normalizer on training datasets ONLY
  3. Transform both train and test with frozen normalizer
  4. Compute ssGSEA on corrected data (within-fold)
  5. Train LightGBM, predict, evaluate

Tests six correction methods:
  - none:               per-dataset z-score only (existing baseline)
  - leaked_combat:      ComBat on ALL data, then LODO (reproduces 0.761)
  - reference_anchored: leakage-free reference-based normalization
  - rank:               within-sample rank normalization (no parameters)
  - quantile:           rank → inverse-normal transform
  - frozen_combat:      ComBat fitted on train only, frozen for test
"""

import os
import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "leakage_free"
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

# Same hyperparameters as v3 pipeline
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
    """Compute ssGSEA Hallmark pathway scores. Input: samples x genes."""
    import gseapy as gp

    expr_clean = clean_matrix(expression_df)
    n_samples = expr_clean.shape[0]

    if n_samples <= CHUNK_SIZE:
        result = gp.ssgsea(
            data=expr_clean.T,
            gene_sets="MSigDB_Hallmark_2020",
            outdir=None,
            min_size=5,
            no_plot=True,
            verbose=False,
        )
        scores = result.res2d.pivot(index="Name", columns="Term", values="NES")
        scores.index.name = None
        scores.columns.name = None
    else:
        chunks = []
        for start in range(0, n_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_samples)
            chunk = expr_clean.iloc[start:end]
            result = gp.ssgsea(
                data=chunk.T,
                gene_sets="MSigDB_Hallmark_2020",
                outdir=None,
                min_size=5,
                no_plot=True,
                verbose=False,
            )
            cs = result.res2d.pivot(index="Name", columns="Term", values="NES")
            cs.index.name = None
            cs.columns.name = None
            chunks.append(cs)
        scores = pd.concat(chunks, axis=0)

    scores = scores.loc[expression_df.index].astype(float)
    scores.columns = [f"ssgsea_{c}" for c in scores.columns]
    return clean_matrix(scores)


# =========================================================================
# STEP 1: Load all datasets (same as v3)
# =========================================================================
log.info("=" * 70)
log.info("STEP 1: Loading all datasets")
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
    assert len(expr) == len(labels)
    response = labels["response"].astype(int).values
    expr = expr.reset_index(drop=True)
    return expr, pd.Series(response, index=expr.index, name="response")


ctrdb_dir = ROOT / "data" / "raw" / "ctrdb"
ctrdb_geos = sorted(
    d.name
    for d in ctrdb_dir.iterdir()
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

try:
    e, l = load_positional_dataset("ispy2", "GSE194040")
    if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
        datasets["ISPY2"] = (e, l)
        treatment_map["ISPY2"] = "combination"
        log.info(f"  ISPY2: {len(l)} patients")
except Exception as exc:
    log.warning(f"  ISPY2 failed: {exc}")

try:
    e, l = load_positional_dataset("brightness", "GSE164458")
    if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
        datasets["BrighTNess"] = (e, l)
        treatment_map["BrighTNess"] = "parp"
        log.info(f"  BrighTNess: {len(l)} patients")
except Exception as exc:
    log.warning(f"  BrighTNess failed: {exc}")

total_patients = sum(len(v[1]) for v in datasets.values())
log.info(f"Loaded {len(datasets)} datasets, {total_patients} total patients")


# =========================================================================
# STEP 2: Restrict to L1000 landmark genes (same as v3)
# =========================================================================
log.info("=" * 70)
log.info("STEP 2: Restricting to L1000 landmark genes")
log.info("=" * 70)

gene_info = pd.read_csv(
    ROOT / "data" / "cache" / "geneinfo_beta_input.txt", header=0
)
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
log.info(f"L1000 genes present in >= {threshold_count}/{n_datasets} datasets: {len(common_l1000)}")

# Restrict each dataset to common L1000 genes
for did in list(datasets.keys()):
    expr, lab = datasets[did]
    available = [g for g in common_l1000 if g in expr.columns]
    missing = [g for g in common_l1000 if g not in expr.columns]
    expr_sub = expr[available].copy()
    for g in missing:
        expr_sub[g] = 0.0
    datasets[did] = (expr_sub[common_l1000], lab)

n_genes = len(common_l1000)
log.info(f"Gene matrix: {n_genes} features per patient")


# =========================================================================
# STEP 3: Per-dataset z-scoring (always applied first)
# =========================================================================
log.info("=" * 70)
log.info("STEP 3: Z-score normalization per dataset")
log.info("=" * 70)

for did in datasets:
    expr, lab = datasets[did]
    means = expr.mean(axis=0)
    stds = expr.std(axis=0).replace(0, 1)
    expr_z = (expr - means) / stds
    expr_z = clean_matrix(expr_z)
    datasets[did] = (expr_z, lab)
    log.info(f"  {did}: z-scored ({expr_z.shape})")


# =========================================================================
# STEP 4: Leakage-free LODO evaluation
# =========================================================================
log.info("=" * 70)
log.info("STEP 4: Leakage-free LODO evaluation")
log.info("=" * 70)

from src.preprocessing.leakage_free_normalizers import (
    ReferenceAnchoredNormalizer,
    RankNormalizer,
    FrozenComBat,
)

# Pre-compute the leaked ComBat correction (for baseline comparison)
# This replicates the v3 pipeline: ComBat on ALL data, then LODO
leaked_combat_data = None
try:
    from neuroCombat import neuroCombat

    # Pool all datasets
    all_expr_parts = []
    all_label_parts = []
    all_batch = []
    sample_to_dataset = {}

    for did in sorted(datasets.keys()):
        expr, lab = datasets[did]
        new_idx = [f"{did}__{i}" for i in range(len(expr))]
        ec = expr.copy()
        ec.index = new_idx
        lc = lab.copy()
        lc.index = new_idx
        all_expr_parts.append(ec)
        all_label_parts.append(lc)
        all_batch.extend([did] * len(ec))
        for s in new_idx:
            sample_to_dataset[s] = did

    pooled_expr = pd.concat(all_expr_parts, axis=0)
    pooled_labels = pd.concat(all_label_parts, axis=0)
    batch_series = pd.Series(all_batch, index=pooled_expr.index)

    covariates = pd.DataFrame(
        {
            "batch": batch_series.values,
            "response": pooled_labels.values.astype(float),
        },
        index=pooled_expr.index,
    )

    log.info("Computing leaked ComBat baseline (ALL data)...")
    t0 = time.time()
    combat_result = neuroCombat(
        dat=pooled_expr.values.T,
        covars=covariates,
        batch_col="batch",
        continuous_cols=["response"],
    )
    leaked_combat_expr = pd.DataFrame(
        combat_result["data"].T,
        columns=pooled_expr.columns,
        index=pooled_expr.index,
    )
    leaked_combat_expr = clean_matrix(leaked_combat_expr)
    log.info(f"  Leaked ComBat done in {time.time()-t0:.1f}s")

    # Compute ssGSEA on leaked ComBat data
    log.info("  Computing ssGSEA on leaked ComBat data...")
    leaked_ssgsea = compute_ssgsea(leaked_combat_expr, "leaked")
    log.info(f"  ssGSEA done: {leaked_ssgsea.shape[1]} pathways")

    leaked_combat_data = {
        "expr": leaked_combat_expr,
        "ssgsea": leaked_ssgsea,
        "labels": pooled_labels,
        "sample_to_dataset": sample_to_dataset,
    }
except Exception as e:
    log.warning(f"Leaked ComBat baseline failed: {e}")

# Also store the uncorrected (z-scored only) pooled data for the "none" baseline
none_pooled_expr = pd.concat(all_expr_parts, axis=0) if all_expr_parts else None
none_pooled_labels = pd.concat(all_label_parts, axis=0) if all_label_parts else None


def run_lodo_for_method(method_name):
    """Run full LODO with a specific correction method."""
    log.info(f"\n{'='*60}")
    log.info(f"LODO with correction: {method_name}")
    log.info(f"{'='*60}")

    dataset_ids = sorted(datasets.keys())
    results = []
    t_start = time.time()

    for test_did in dataset_ids:
        # --- Split into train/test datasets ---
        train_dids = [d for d in dataset_ids if d != test_did]
        test_expr, test_labels = datasets[test_did]

        # Handle the "leaked_combat" baseline specially
        if method_name == "leaked_combat":
            if leaked_combat_data is None:
                continue
            lcd = leaked_combat_data
            test_mask = [
                s for s in lcd["expr"].index
                if lcd["sample_to_dataset"][s] == test_did
            ]
            train_mask = [
                s for s in lcd["expr"].index
                if lcd["sample_to_dataset"][s] != test_did
            ]

            # Genes + ssGSEA (like E_full_combat)
            train_X = pd.concat(
                [lcd["expr"].loc[train_mask], lcd["ssgsea"].loc[train_mask]],
                axis=1,
            )
            test_X = pd.concat(
                [lcd["expr"].loc[test_mask], lcd["ssgsea"].loc[test_mask]],
                axis=1,
            )
            train_y = lcd["labels"].loc[train_mask].values.astype(int)
            test_y = lcd["labels"].loc[test_mask].values.astype(int)

        elif method_name == "none":
            # Per-dataset z-score only (already applied), just pool
            train_parts_x = []
            train_parts_y = []
            for d in train_dids:
                e, l = datasets[d]
                new_idx = [f"{d}__{i}" for i in range(len(e))]
                ec = e.copy(); ec.index = new_idx
                train_parts_x.append(ec)
                lc = l.copy(); lc.index = new_idx
                train_parts_y.append(lc)

            train_expr_pooled = pd.concat(train_parts_x, axis=0)
            train_labels_pooled = pd.concat(train_parts_y, axis=0)

            test_idx = [f"{test_did}__{i}" for i in range(len(test_expr))]
            test_expr_indexed = test_expr.copy()
            test_expr_indexed.index = test_idx

            # Compute ssGSEA separately for train and test
            try:
                train_ssgsea = compute_ssgsea(train_expr_pooled, f"none_train_{test_did}")
                test_ssgsea = compute_ssgsea(test_expr_indexed, f"none_test_{test_did}")
                # Align columns
                common_ssgsea_cols = sorted(
                    set(train_ssgsea.columns) & set(test_ssgsea.columns)
                )
                train_X = pd.concat(
                    [train_expr_pooled, train_ssgsea[common_ssgsea_cols]], axis=1
                )
                test_X = pd.concat(
                    [test_expr_indexed, test_ssgsea[common_ssgsea_cols]], axis=1
                )
            except Exception:
                train_X = train_expr_pooled
                test_X = test_expr_indexed

            train_y = train_labels_pooled.values.astype(int)
            test_y = test_labels.values.astype(int)

        else:
            # === LEAKAGE-FREE METHODS ===
            # 1. Pool train datasets
            train_parts_x = []
            train_parts_y = []
            train_batch = []
            for d in train_dids:
                e, l = datasets[d]
                new_idx = [f"{d}__{i}" for i in range(len(e))]
                ec = e.copy(); ec.index = new_idx
                train_parts_x.append(ec)
                lc = l.copy(); lc.index = new_idx
                train_parts_y.append(lc)
                train_batch.extend([d] * len(ec))

            train_expr_pooled = pd.concat(train_parts_x, axis=0)
            train_labels_pooled = pd.concat(train_parts_y, axis=0)
            train_batch_series = pd.Series(
                train_batch, index=train_expr_pooled.index
            )

            test_idx = [f"{test_did}__{i}" for i in range(len(test_expr))]
            test_expr_indexed = test_expr.copy()
            test_expr_indexed.index = test_idx
            test_batch_series = pd.Series(
                [test_did] * len(test_expr_indexed), index=test_expr_indexed.index
            )

            # 2. Instantiate and fit normalizer on TRAIN ONLY
            if method_name == "reference_anchored":
                normalizer = ReferenceAnchoredNormalizer()
            elif method_name == "rank":
                normalizer = RankNormalizer(method="rank")
            elif method_name == "quantile":
                normalizer = RankNormalizer(method="quantile")
            elif method_name == "frozen_combat":
                normalizer = FrozenComBat()
            else:
                raise ValueError(f"Unknown method: {method_name}")

            normalizer.fit(train_expr_pooled, train_batch_series)

            # 3. Transform BOTH using frozen parameters
            train_corrected = normalizer.transform(
                train_expr_pooled, train_batch_series
            )
            test_corrected = normalizer.transform(
                test_expr_indexed, test_batch_series
            )
            train_corrected = clean_matrix(train_corrected)
            test_corrected = clean_matrix(test_corrected)

            # 4. Compute ssGSEA on corrected data
            try:
                train_ssgsea = compute_ssgsea(
                    train_corrected, f"{method_name}_train_{test_did}"
                )
                test_ssgsea = compute_ssgsea(
                    test_corrected, f"{method_name}_test_{test_did}"
                )
                common_ssgsea_cols = sorted(
                    set(train_ssgsea.columns) & set(test_ssgsea.columns)
                )
                train_X = pd.concat(
                    [train_corrected, train_ssgsea[common_ssgsea_cols]], axis=1
                )
                test_X = pd.concat(
                    [test_corrected, test_ssgsea[common_ssgsea_cols]], axis=1
                )
            except Exception as exc:
                log.warning(f"  ssGSEA failed for {method_name}/{test_did}: {exc}")
                train_X = train_corrected
                test_X = test_corrected

            train_y = train_labels_pooled.values.astype(int)
            test_y = test_labels.values.astype(int)

        # --- Skip if single class ---
        if len(np.unique(test_y)) < 2 or len(np.unique(train_y)) < 2:
            log.info(f"  {method_name}/{test_did}: skipped (single class)")
            continue

        # --- Train and predict ---
        train_X_clean = clean_matrix(train_X).values
        test_X_clean = clean_matrix(test_X).values

        mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
        mdl.fit(train_X_clean, train_y)
        yp = mdl.predict_proba(test_X_clean)[:, 1]

        try:
            auc = roc_auc_score(test_y, yp)
        except ValueError:
            auc = np.nan

        n_features = train_X_clean.shape[1]
        results.append({
            "correction_method": method_name,
            "dataset_id": test_did,
            "n_train": len(train_y),
            "n_test": len(test_y),
            "n_test_pos": int(test_y.sum()),
            "n_features": n_features,
            "auc": auc,
            "treatment_class": treatment_map.get(test_did, "unknown"),
        })
        log.info(
            f"  {method_name}/{test_did}: AUC={auc:.4f} "
            f"(n={len(test_y)}, pos={int(test_y.sum())}, feat={n_features})"
        )

    elapsed = time.time() - t_start
    df = pd.DataFrame(results)
    mean_auc = df["auc"].mean() if len(df) > 0 else np.nan
    log.info(f"  → {method_name}: Mean AUC = {mean_auc:.4f} ({elapsed:.0f}s)")
    return df


# =========================================================================
# RUN ALL METHODS
# =========================================================================
METHODS = [
    "leaked_combat",       # Baseline: ComBat on ALL data (the 0.761)
    "none",                # Per-dataset z-score only (existing honest baseline)
    "reference_anchored",  # Leakage-free Method A
    "frozen_combat",       # Leakage-free Method C
    "rank",                # Leakage-free Method B (raw ranks)
    "quantile",            # Leakage-free Method B (quantile normalization)
]

all_results = []
for method in METHODS:
    result_df = run_lodo_for_method(method)
    all_results.append(result_df)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(
    OUT_DIR / "leakage_free_retrain_comparison.tsv", sep="\t", index=False
)
log.info(f"\nSaved per-dataset results: {OUT_DIR}/leakage_free_retrain_comparison.tsv")

# Summary table
summary = (
    combined.groupby("correction_method")["auc"]
    .agg(["mean", "std", "median", "min", "max", "count"])
    .sort_values("mean", ascending=False)
)
summary.to_csv(OUT_DIR / "leakage_free_retrain_summary.tsv", sep="\t")

log.info("\n" + "=" * 70)
log.info("SUMMARY: Leakage-free batch correction comparison")
log.info("=" * 70)
for method, row in summary.iterrows():
    log.info(
        f"  {method:25s}  AUC = {row['mean']:.4f} ± {row['std']:.4f}  "
        f"(median={row['median']:.4f}, n={int(row['count'])})"
    )

# Also copy to top-level results for easy access
summary.to_csv(
    ROOT / "results" / "leakage_free_retrain_summary.tsv", sep="\t"
)
combined.to_csv(
    ROOT / "results" / "leakage_free_retrain_comparison.tsv", sep="\t", index=False
)

log.info(f"\nDone. Results in {OUT_DIR}/")
