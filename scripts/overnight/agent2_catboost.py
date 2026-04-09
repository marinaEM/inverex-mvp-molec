#!/usr/bin/env python
"""
Agent 2: CatBoost Benchmark
============================

Test whether CatBoost beats LightGBM on identical features and splits.
30 Optuna trials.
"""

import os, sys, json, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scripts.overnight._shared import (
    BENCH, load_baseline_data, run_lodo_loop, aggregate_run,
    write_run_metrics, append_to_master, setup_logger,
)

AGENT = "agent2_catboost"
log = setup_logger(AGENT)
N_TRIALS = 30
SEED = 42

log.info("=" * 70)
log.info("AGENT 2: CatBoost Benchmark")
log.info("=" * 70)

X = pd.read_parquet(BENCH / "configs" / "baseline_features.parquet")
data = load_baseline_data()
log.info(f"Features: {X.shape}, {len(data['datasets'])} datasets")


# Default CatBoost
log.info("\n=== Default CatBoost ===")
DEFAULT_PARAMS = {
    "iterations": 500, "depth": 6, "learning_rate": 0.05,
    "l2_leaf_reg": 3, "random_seed": SEED, "verbose": False,
    "thread_count": -1, "loss_function": "Logloss",
}

def default_factory():
    return CatBoostClassifier(**DEFAULT_PARAMS)

t0 = time.time()
fold_results = run_lodo_loop(data, X, default_factory, "default", log_fn=log.info)
default_summary = aggregate_run(fold_results)
log.info(f"Default CatBoost: AUROC={default_summary.get('mean_auroc',0):.4f}  "
         f"MCC={default_summary.get('mean_mcc',0):.4f}  ({time.time()-t0:.0f}s)")
write_run_metrics(AGENT, "default", fold_results, DEFAULT_PARAMS)
append_to_master(AGENT, "default", default_summary, DEFAULT_PARAMS)


# Optuna sweep
log.info(f"\n=== Optuna sweep ({N_TRIALS} trials) ===")
trial_results = []

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "border_count": trial.suggest_int("border_count", 32, 254),
        "random_seed": SEED, "verbose": False, "thread_count": -1,
        "loss_function": "Logloss",
    }

    def factory():
        return CatBoostClassifier(**params)

    fold_results = run_lodo_loop(data, X, factory, f"trial_{trial.number}")
    summary = aggregate_run(fold_results)
    auroc = summary.get("mean_auroc", float("nan"))
    trial_results.append({"trial": trial.number, "params": params,
                          "fold_results": fold_results, "summary": summary})
    log.info(f"Trial {trial.number:3d}: AUROC={auroc:.4f}  MCC={summary.get('mean_mcc',0):.4f}  "
             f"depth={params['depth']}  lr={params['learning_rate']:.3f}  iter={params['iterations']}")
    return auroc

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

# Save all trials
all_trials_df = pd.DataFrame([{
    "trial": tr["trial"], **tr["params"], **tr["summary"],
} for tr in trial_results])
all_trials_df.to_csv(BENCH / "raw_metrics" / f"{AGENT}_all_trials.tsv", sep="\t", index=False)

best_trial = trial_results[max(range(len(trial_results)), key=lambda i: trial_results[i]["summary"].get("mean_auroc", 0))]
log.info(f"\n=== Best CatBoost trial: {best_trial['trial']} ===")
log.info(f"AUROC: {best_trial['summary'].get('mean_auroc', 0):.4f}  "
         f"MCC: {best_trial['summary'].get('mean_mcc', 0):.4f}")
write_run_metrics(AGENT, "best_trial", best_trial["fold_results"], best_trial["params"])
append_to_master(AGENT, f"best_trial_{best_trial['trial']}", best_trial["summary"], best_trial["params"])

log.info(f"\nDone.")
