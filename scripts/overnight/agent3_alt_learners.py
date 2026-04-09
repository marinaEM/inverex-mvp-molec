#!/usr/bin/env python
"""
Agent 3: Alternative Tabular Learners
======================================

Test whether radically different learners change the conclusion.
- Elastic net logistic regression
- Random Forest
- XGBoost
- (skipping xRFM for time)
"""

import os, sys, json, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from scripts.overnight._shared import (
    BENCH, load_baseline_data, run_lodo_loop, aggregate_run,
    write_run_metrics, append_to_master, setup_logger,
)

AGENT = "agent3_alt_learners"
log = setup_logger(AGENT)
SEED = 42

log.info("=" * 70)
log.info("AGENT 3: Alternative Tabular Learners")
log.info("=" * 70)

X = pd.read_parquet(BENCH / "configs" / "baseline_features.parquet")
data = load_baseline_data()
log.info(f"Features: {X.shape}")


# =========================================================================
# Elastic Net Logistic Regression
# =========================================================================
log.info("\n=== Elastic Net Logistic Regression ===")

class LRWrapper:
    def __init__(self, **kwargs):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(**kwargs)
    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self
    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

LR_CONFIGS = [
    {"penalty": "elasticnet", "C": 1.0, "l1_ratio": 0.5, "solver": "saga", "max_iter": 1000, "random_state": SEED},
    {"penalty": "elasticnet", "C": 0.1, "l1_ratio": 0.5, "solver": "saga", "max_iter": 1000, "random_state": SEED},
    {"penalty": "l2", "C": 1.0, "max_iter": 1000, "random_state": SEED},
    {"penalty": "l1", "C": 0.5, "solver": "saga", "max_iter": 1000, "random_state": SEED},
]

for i, params in enumerate(LR_CONFIGS):
    name = f"lr_{params.get('penalty','l2')}_C{params.get('C',1.0)}"
    if "l1_ratio" in params: name += f"_l1r{params['l1_ratio']}"
    log.info(f"\n--- {name} ---")
    t0 = time.time()
    def factory(p=params):
        return LRWrapper(**p)
    fold_results = run_lodo_loop(data, X, factory, name)
    summary = aggregate_run(fold_results)
    log.info(f"  AUROC={summary.get('mean_auroc',0):.4f}  MCC={summary.get('mean_mcc',0):.4f}  ({time.time()-t0:.0f}s)")
    write_run_metrics(AGENT, name, fold_results, params)
    append_to_master(AGENT, name, summary, params)


# =========================================================================
# Random Forest
# =========================================================================
log.info("\n=== Random Forest ===")
RF_CONFIGS = [
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "n_jobs": -1, "random_state": SEED},
    {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 10, "n_jobs": -1, "random_state": SEED},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt", "n_jobs": -1, "random_state": SEED},
]

for i, params in enumerate(RF_CONFIGS):
    name = f"rf_n{params['n_estimators']}_d{params.get('max_depth','none')}_leaf{params['min_samples_leaf']}"
    log.info(f"\n--- {name} ---")
    t0 = time.time()
    def factory(p=params):
        return RandomForestClassifier(**p)
    fold_results = run_lodo_loop(data, X, factory, name)
    summary = aggregate_run(fold_results)
    log.info(f"  AUROC={summary.get('mean_auroc',0):.4f}  MCC={summary.get('mean_mcc',0):.4f}  ({time.time()-t0:.0f}s)")
    write_run_metrics(AGENT, name, fold_results, params)
    append_to_master(AGENT, name, summary, params)


# =========================================================================
# XGBoost
# =========================================================================
log.info("\n=== XGBoost ===")
XGB_CONFIGS = [
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8,
     "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 2.0, "random_state": SEED,
     "n_jobs": -1, "eval_metric": "logloss", "verbosity": 0},
    {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.03, "subsample": 0.8,
     "colsample_bytree": 0.6, "reg_alpha": 0.5, "reg_lambda": 2.0, "random_state": SEED,
     "n_jobs": -1, "eval_metric": "logloss", "verbosity": 0},
    {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.7,
     "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 5.0, "random_state": SEED,
     "n_jobs": -1, "eval_metric": "logloss", "verbosity": 0},
]

for i, params in enumerate(XGB_CONFIGS):
    name = f"xgb_n{params['n_estimators']}_d{params['max_depth']}_lr{params['learning_rate']}"
    log.info(f"\n--- {name} ---")
    t0 = time.time()
    def factory(p=params):
        return xgb.XGBClassifier(**p)
    fold_results = run_lodo_loop(data, X, factory, name)
    summary = aggregate_run(fold_results)
    log.info(f"  AUROC={summary.get('mean_auroc',0):.4f}  MCC={summary.get('mean_mcc',0):.4f}  ({time.time()-t0:.0f}s)")
    write_run_metrics(AGENT, name, fold_results, params)
    append_to_master(AGENT, name, summary, params)


log.info(f"\nDone.")
