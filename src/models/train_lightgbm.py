"""
LightGBM drug-response model — training with Bayesian hyperparameter optimization.

Replicates scTherapy's training approach:
  - LightGBM regressor predicting percent inhibition
  - Bayesian optimization (via Optuna) with repeated 10-fold CV
  - Feature importances for interpretability

Improvements over scTherapy for breast-cancer MVP:
  - Breast-cancer-specific training data
  - Optional subtype indicator features
  - Full local model (no black-box API)
  - Feature importance analysis for rationale panel
"""
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from src.config import (
    DATA_PROCESSED,
    LIGHTGBM_DEFAULT_PARAMS,
    RANDOM_SEED,
    RESULTS,
)
from src.features.build_training_matrix import build_training_matrix

logger = logging.getLogger(__name__)


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    optimize_hparams: bool = True,
    n_optuna_trials: int = 30,
    cv_folds: int = 5,
    output_dir: Path = RESULTS,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM regressor for drug-response prediction.

    Args:
        X: Feature matrix (gene z-scores + ECFP4 + dose)
        y: Target (percent inhibition)
        optimize_hparams: Whether to run Bayesian HP optimization
        n_optuna_trials: Number of Optuna trials
        cv_folds: Number of CV folds
        output_dir: Where to save the trained model

    Returns:
        Trained LGBMRegressor
    """
    model_path = output_dir / "lightgbm_drug_model.joblib"
    metrics_path = output_dir / "lightgbm_metrics.json"

    if optimize_hparams:
        logger.info(f"Running Bayesian HP optimization ({n_optuna_trials} trials)...")
        best_params = _optimize_hyperparams(X, y, n_optuna_trials, cv_folds)
    else:
        best_params = LIGHTGBM_DEFAULT_PARAMS.copy()

    # Train final model on full data
    logger.info("Training final LightGBM model on full dataset...")
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X, y,
        callbacks=[lgb.log_evaluation(100)],
    )

    # Cross-validation score for reporting
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        lgb.LGBMRegressor(**best_params),
        X, y, cv=cv, scoring="neg_root_mean_squared_error"
    )
    rmse_mean = -cv_scores.mean()
    rmse_std = cv_scores.std()

    logger.info(f"CV RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}")

    # Feature importances
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Separate gene vs ECFP vs dose importances
    gene_imp = importances[
        ~importances["feature"].str.startswith("ecfp_")
        & (importances["feature"] != "log_dose_um")
    ]
    ecfp_imp = importances[importances["feature"].str.startswith("ecfp_")]
    dose_imp = importances[importances["feature"] == "log_dose_um"]

    logger.info(f"\nTop 20 gene features by importance:")
    for _, row in gene_imp.head(20).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']}")

    logger.info(f"\nDose feature importance: {dose_imp['importance'].sum()}")
    logger.info(f"Total ECFP importance: {ecfp_imp['importance'].sum()}")
    logger.info(f"Total gene importance: {gene_imp['importance'].sum()}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    importances.to_csv(output_dir / "feature_importances.csv", index=False)

    metrics = {
        "cv_rmse_mean": float(rmse_mean),
        "cv_rmse_std": float(rmse_std),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "best_params": {k: str(v) for k, v in best_params.items()},
        "top_gene_features": gene_imp.head(20).to_dict("records"),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model saved to {model_path}")
    return model


def _optimize_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    cv_folds: int = 5,
) -> dict:
    """
    Bayesian hyperparameter optimization using Optuna.

    Searches over key LightGBM parameters that affect
    the bias-variance tradeoff.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed. Using default hyperparameters.")
        return LIGHTGBM_DEFAULT_PARAMS.copy()

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "random_state": RANDOM_SEED,
            "verbose": -1,
        }

        model = lgb.LGBMRegressor(**params)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="neg_root_mean_squared_error"
        )
        return -scores.mean()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": RANDOM_SEED,
        "verbose": -1,
    })

    logger.info(f"Best trial RMSE: {study.best_value:.3f}")
    logger.info(f"Best params: {best}")
    return best


def load_trained_model(model_dir: Path = RESULTS) -> lgb.LGBMRegressor:
    """Load a previously trained model."""
    model_path = model_dir / "lightgbm_drug_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run training first.")
    return joblib.load(model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Building training matrix...")
    X, y, features = build_training_matrix(use_demo=True)

    logger.info("Training LightGBM...")
    model = train_lightgbm(
        X, y,
        optimize_hparams=False,  # Use defaults for quick demo
        output_dir=RESULTS,
    )
    print(f"\nModel trained with {model.n_features_} features")
