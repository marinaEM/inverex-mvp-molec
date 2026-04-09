#!/usr/bin/env python
"""
Agent 5: Stratified Modeling
============================

Test whether response signal is more predictable when stratified by:
  - Drug class
  - Cancer type
  - Technology platform

For each stratum: hold out one dataset at a time, train on the other
datasets WITHIN that stratum (proper LODO within stratum), evaluate.
"""

import os, sys, json, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from scripts.overnight._shared import (
    BENCH, load_baseline_data, run_lodo_loop, aggregate_run,
    write_run_metrics, append_to_master, setup_logger,
    DATASET_DRUG_MAP, DATASET_CANCER_MAP, PLATFORM_MAP,
    tune_threshold, evaluate_predictions,
)

AGENT = "agent5_stratified"
log = setup_logger(AGENT)
SEED = 42

LGBM_PARAMS = {
    "objective": "binary", "metric": "auc", "n_estimators": 300,
    "num_leaves": 31, "max_depth": 5, "min_child_samples": 10,
    "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 2.0, "random_state": SEED, "verbose": -1,
}

MIN_DATASETS_PER_STRATUM = 2
MIN_PATIENTS_PER_STRATUM = 100

log.info("=" * 70)
log.info("AGENT 5: Stratified Modeling")
log.info("=" * 70)

X = pd.read_parquet(BENCH / "configs" / "baseline_features.parquet")
data = load_baseline_data()
log.info(f"Features: {X.shape}")


def run_stratified_lodo(stratum_name, stratum_value, dataset_filter):
    """LODO within a stratum: only datasets matching the filter."""
    pooled_labels = data["pooled_labels"]
    s2d = data["s2d"]
    valid_datasets = [d for d in set(s2d.values()) if dataset_filter(d)]

    if len(valid_datasets) < MIN_DATASETS_PER_STRATUM:
        return None, f"only {len(valid_datasets)} datasets"

    valid_samples = [s for s in X.index if s2d[s] in valid_datasets]
    if len(valid_samples) < MIN_PATIENTS_PER_STRATUM:
        return None, f"only {len(valid_samples)} patients"

    X_stratum = X.loc[valid_samples]
    fold_results = []

    for holdout_id in valid_datasets:
        train_s = [s for s in valid_samples if s2d[s] != holdout_id]
        test_s = [s for s in valid_samples if s2d[s] == holdout_id]
        train_y = pooled_labels.loc[train_s].values.astype(int)
        test_y = pooled_labels.loc[test_s].values.astype(int)

        if len(np.unique(test_y)) < 2 or len(np.unique(train_y)) < 2:
            continue
        if len(test_y) < 5 or len(train_y) < 30:
            continue

        try:
            mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
            mdl.fit(X_stratum.loc[train_s].values, train_y)
            tr_pred = mdl.predict_proba(X_stratum.loc[train_s].values)[:, 1]
            threshold = tune_threshold(train_y, tr_pred)
            te_pred = mdl.predict_proba(X_stratum.loc[test_s].values)[:, 1]
            metrics = evaluate_predictions(test_y, te_pred, threshold)
        except Exception as e:
            continue

        fold_results.append({
            "stratum_name": stratum_name,
            "stratum_value": stratum_value,
            "holdout": holdout_id,
            "n_train_datasets": len(valid_datasets) - 1,
            **metrics,
        })

    return fold_results, None


# =========================================================================
# By drug class
# =========================================================================
log.info("\n=== Stratification: drug class ===")
all_drug_classes = sorted(set(DATASET_DRUG_MAP.values()))
log.info(f"All drug classes ({len(all_drug_classes)}): {all_drug_classes}")

drug_results = []
for drug in all_drug_classes:
    dataset_filter = lambda d, dr=drug: DATASET_DRUG_MAP.get(d) == dr
    fold_results, skip_reason = run_stratified_lodo("drug_class", drug, dataset_filter)
    if fold_results is None:
        n_ds = sum(1 for d in set(data["s2d"].values()) if DATASET_DRUG_MAP.get(d) == drug)
        log.info(f"  {drug}: SKIPPED ({skip_reason}, {n_ds} datasets)")
        continue
    summary = aggregate_run(fold_results)
    summary["stratum_name"] = "drug_class"
    summary["stratum_value"] = drug
    summary["n_datasets_in_stratum"] = sum(1 for d in set(data["s2d"].values()) if DATASET_DRUG_MAP.get(d) == drug)
    log.info(f"  {drug}: AUROC={summary.get('mean_auroc',0):.4f}  MCC={summary.get('mean_mcc',0):.4f}  n_folds={summary['n_folds']}")
    drug_results.append(summary)
    if fold_results:
        write_run_metrics(AGENT, f"drug_{drug}", fold_results, {"stratum": drug})

if drug_results:
    pd.DataFrame(drug_results).to_csv(BENCH / "raw_metrics" / f"{AGENT}_drug_summary.tsv", sep="\t", index=False)


# =========================================================================
# By cancer type
# =========================================================================
log.info("\n=== Stratification: cancer type ===")
all_cancers = sorted(set(DATASET_CANCER_MAP.values()))
log.info(f"All cancer types ({len(all_cancers)}): {all_cancers}")

cancer_results = []
for cancer in all_cancers:
    dataset_filter = lambda d, c=cancer: DATASET_CANCER_MAP.get(d) == c
    fold_results, skip_reason = run_stratified_lodo("cancer_type", cancer, dataset_filter)
    if fold_results is None:
        n_ds = sum(1 for d in set(data["s2d"].values()) if DATASET_CANCER_MAP.get(d) == cancer)
        log.info(f"  {cancer}: SKIPPED ({skip_reason}, {n_ds} datasets)")
        continue
    summary = aggregate_run(fold_results)
    summary["stratum_name"] = "cancer_type"
    summary["stratum_value"] = cancer
    summary["n_datasets_in_stratum"] = sum(1 for d in set(data["s2d"].values()) if DATASET_CANCER_MAP.get(d) == cancer)
    log.info(f"  {cancer}: AUROC={summary.get('mean_auroc',0):.4f}  MCC={summary.get('mean_mcc',0):.4f}  n_folds={summary['n_folds']}")
    cancer_results.append(summary)
    if fold_results:
        write_run_metrics(AGENT, f"cancer_{cancer}", fold_results, {"stratum": cancer})

if cancer_results:
    pd.DataFrame(cancer_results).to_csv(BENCH / "raw_metrics" / f"{AGENT}_cancer_summary.tsv", sep="\t", index=False)


# =========================================================================
# By technology
# =========================================================================
log.info("\n=== Stratification: technology platform ===")
all_techs = sorted(set(PLATFORM_MAP.values()) - {"targeted", "beadarray", "unknown"})

tech_results = []
for tech in all_techs:
    dataset_filter = lambda d, t=tech: PLATFORM_MAP.get(d) == t
    fold_results, skip_reason = run_stratified_lodo("technology", tech, dataset_filter)
    if fold_results is None:
        log.info(f"  {tech}: SKIPPED ({skip_reason})")
        continue
    summary = aggregate_run(fold_results)
    summary["stratum_name"] = "technology"
    summary["stratum_value"] = tech
    log.info(f"  {tech}: AUROC={summary.get('mean_auroc',0):.4f}  MCC={summary.get('mean_mcc',0):.4f}  n_folds={summary['n_folds']}")
    tech_results.append(summary)
    if fold_results:
        write_run_metrics(AGENT, f"tech_{tech}", fold_results, {"stratum": tech})

if tech_results:
    pd.DataFrame(tech_results).to_csv(BENCH / "raw_metrics" / f"{AGENT}_tech_summary.tsv", sep="\t", index=False)


# Combined summary
all_strata = drug_results + cancer_results + tech_results
if all_strata:
    df = pd.DataFrame(all_strata)
    df = df.sort_values("mean_auroc", ascending=False)
    df.to_csv(BENCH / "summaries" / "stratified_summary.csv", index=False)
    log.info("\n=== Top 10 strata by AUROC ===")
    for _, r in df.head(10).iterrows():
        log.info(f"  {r['stratum_name']}={r['stratum_value']}: AUROC={r.get('mean_auroc',0):.4f}  "
                 f"MCC={r.get('mean_mcc',0):.4f}  n_folds={r.get('n_folds',0)}")

log.info(f"\nDone.")
