#!/usr/bin/env python
"""
Agent 4: Pathway vs Gene Feature Comparison
============================================

Test whether pathway-level features are more informative than gene-level
or add complementary signal.

Configs:
  1. Genes only (918 rank)
  2. ssGSEA only (50 Hallmark pathways)
  3. singscore only (47 rank-based pathways)
  4. ssGSEA + singscore only
  5. Genes + ssGSEA
  6. Genes + singscore
  7. Genes + ssGSEA + singscore (full kitchen sink)
"""

import os, sys, time, warnings
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
    within_sample_rank_inv_norm, compute_singscore, compute_ssgsea, clean_matrix,
)

AGENT = "agent4_pathway"
log = setup_logger(AGENT)
SEED = 42

LGBM_PARAMS = {
    "objective": "binary", "metric": "auc", "n_estimators": 300,
    "num_leaves": 31, "max_depth": 5, "min_child_samples": 10,
    "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 2.0, "random_state": SEED, "verbose": -1,
}

log.info("=" * 70)
log.info("AGENT 4: Pathway vs Gene Features")
log.info("=" * 70)

data = load_baseline_data()
log.info(f"{len(data['datasets'])} datasets, {len(data['pooled_labels'])} patients")

# Build all feature sets
log.info("\nBuilding features...")
t0 = time.time()
rank_expr = within_sample_rank_inv_norm(data["pooled_expr"])
log.info(f"  Rank expression: {rank_expr.shape}")
singscore = compute_singscore(data["pooled_expr"], data["common_genes"])
log.info(f"  Singscore: {singscore.shape[1]} pathways")
ssgsea = compute_ssgsea(data["pooled_expr"])
log.info(f"  ssGSEA: {ssgsea.shape[1]} pathways")
log.info(f"Built in {time.time()-t0:.0f}s")

FEATURE_CONFIGS = {
    "genes_only_918":           [rank_expr],
    "ssgsea_only":              [ssgsea],
    "singscore_only":           [singscore],
    "ssgsea_plus_singscore":    [ssgsea, singscore],
    "genes_plus_ssgsea":        [rank_expr, ssgsea],
    "genes_plus_singscore":     [rank_expr, singscore],
    "genes_plus_all_pathways":  [rank_expr, ssgsea, singscore],
}

for name, parts in FEATURE_CONFIGS.items():
    X = clean_matrix(pd.concat(parts, axis=1))
    log.info(f"\n--- {name} ({X.shape[1]} features) ---")
    t0 = time.time()
    def factory():
        return lgb.LGBMClassifier(**LGBM_PARAMS)
    fold_results = run_lodo_loop(data, X, factory, name)
    summary = aggregate_run(fold_results)
    summary["n_features"] = X.shape[1]
    log.info(f"  AUROC={summary.get('mean_auroc',0):.4f}  AUPRC={summary.get('mean_auprc',0):.4f}  "
             f"MCC={summary.get('mean_mcc',0):.4f}  ({time.time()-t0:.0f}s)")
    write_run_metrics(AGENT, name, fold_results, {"feature_set": name, **LGBM_PARAMS})
    append_to_master(AGENT, name, summary, {"feature_set": name}, feature_set=name)


log.info(f"\nDone.")
