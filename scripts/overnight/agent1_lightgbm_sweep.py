#!/usr/bin/env python
"""
Agent 1: LightGBM Capacity Sweep
=================================

Optuna-based hyperparameter search to test whether the current depth-5
LightGBM is too conservative. 50 trials, LODO evaluation per trial.

Question: Does deeper or less constrained LightGBM beat the 0.602 baseline?
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scripts.overnight._shared import (
    BENCH, load_baseline_data, run_lodo_loop, aggregate_run,
    write_run_metrics, append_to_master, setup_logger,
)

AGENT = "agent1_lgbm_sweep"
log = setup_logger(AGENT)
N_TRIALS = 50
SEED = 42

log.info("=" * 70)
log.info("AGENT 1: LightGBM Capacity Sweep")
log.info("=" * 70)

log.info("Loading cached baseline features...")
X = pd.read_parquet(BENCH / "configs" / "baseline_features.parquet")
log.info(f"Features: {X.shape}")
data = load_baseline_data()
log.info(f"{len(data['datasets'])} datasets, {len(data['pooled_labels'])} patients")


# =========================================================================
# Baseline (current production config) for reference
# =========================================================================
log.info("\n=== Baseline (current production LightGBM config) ===")
BASELINE_PARAMS = {
    "objective": "binary", "metric": "auc", "n_estimators": 300,
    "num_leaves": 31, "max_depth": 5, "min_child_samples": 10,
    "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 2.0, "random_state": SEED, "verbose": -1,
}

def baseline_factory():
    return lgb.LGBMClassifier(**BASELINE_PARAMS)

t0 = time.time()
fold_results = run_lodo_loop(data, X, baseline_factory, "baseline", log_fn=log.info)
baseline_summary = aggregate_run(fold_results)
log.info(f"Baseline: AUROC={baseline_summary.get('mean_auroc', 0):.4f}  "
         f"AUPRC={baseline_summary.get('mean_auprc', 0):.4f}  "
         f"MCC={baseline_summary.get('mean_mcc', 0):.4f}  ({time.time()-t0:.0f}s)")
write_run_metrics(AGENT, "baseline", fold_results, BASELINE_PARAMS)
append_to_master(AGENT, "baseline", baseline_summary, BASELINE_PARAMS)
BASELINE_AUROC = baseline_summary.get("mean_auroc", 0.602)


# =========================================================================
# Optuna sweep
# =========================================================================
log.info(f"\n=== Optuna sweep ({N_TRIALS} trials) ===")

trial_results = []

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": SEED,
        "verbose": -1,
    }

    def factory():
        return lgb.LGBMClassifier(**params)

    fold_results = run_lodo_loop(data, X, factory, f"trial_{trial.number}")
    summary = aggregate_run(fold_results)
    auroc = summary.get("mean_auroc", float("nan"))

    trial_results.append({
        "trial": trial.number, "params": params, "fold_results": fold_results,
        "summary": summary,
    })
    log.info(f"Trial {trial.number:3d}: AUROC={auroc:.4f}  MCC={summary.get('mean_mcc',0):.4f}  "
             f"depth={params['max_depth']}  leaves={params['num_leaves']}  "
             f"lr={params['learning_rate']:.3f}  n_est={params['n_estimators']}")
    return auroc

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

# Save all trial results
all_trials_df = pd.DataFrame([{
    "trial": tr["trial"],
    **tr["params"],
    **tr["summary"],
} for tr in trial_results])
all_trials_df.to_csv(BENCH / "raw_metrics" / f"{AGENT}_all_trials.tsv", sep="\t", index=False)

# Best trial
best_trial = trial_results[max(range(len(trial_results)), key=lambda i: trial_results[i]["summary"].get("mean_auroc", 0))]
log.info(f"\n=== Best trial: {best_trial['trial']} ===")
log.info(f"AUROC: {best_trial['summary'].get('mean_auroc', 0):.4f}  "
         f"MCC: {best_trial['summary'].get('mean_mcc', 0):.4f}")
log.info(f"Params: {json.dumps(best_trial['params'], indent=2)}")

write_run_metrics(AGENT, "best_trial", best_trial["fold_results"], best_trial["params"])
append_to_master(AGENT, f"best_trial_{best_trial['trial']}", best_trial["summary"], best_trial["params"])

# Top 5 trials
top5 = sorted(trial_results, key=lambda t: t["summary"].get("mean_auroc", 0), reverse=True)[:5]
log.info(f"\n=== Top 5 trials ===")
for t in top5:
    s = t["summary"]
    log.info(f"  trial {t['trial']:3d}: AUROC={s.get('mean_auroc',0):.4f}  "
             f"MCC={s.get('mean_mcc',0):.4f}  depth={t['params']['max_depth']}")

# Compare best vs baseline
delta = best_trial["summary"].get("mean_auroc", 0) - BASELINE_AUROC
log.info(f"\nDelta vs baseline: {delta:+.4f} AUROC")
if abs(delta) < 0.005:
    log.info("Verdict: NO MEANINGFUL IMPROVEMENT — capacity is not the bottleneck")
elif delta > 0.01:
    log.info("Verdict: MEANINGFUL IMPROVEMENT — increased capacity helps")
elif delta > 0:
    log.info("Verdict: SUGGESTIVE GAIN — small improvement from sweep")
else:
    log.info("Verdict: Best sweep matches or underperforms baseline")

log.info(f"\nDone. Outputs in {BENCH}/")
