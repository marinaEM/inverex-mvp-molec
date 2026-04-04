"""
Compare LightGBM vs Shallow NN vs Ensemble on cell-line drug-response data.

Runs 5-fold cross-validation and reports RMSE and R-squared for each model.
Also saves per-fold results and a summary table.

Usage:
    pixi run python scripts/compare_nn_lightgbm.py
"""
import logging
import os
import sys
import time

# Must set before any torch/lightgbm import to avoid OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import LIGHTGBM_DEFAULT_PARAMS, RANDOM_SEED, RESULTS, DATA_PROCESSED
from src.models.interaction_nn import DrugResponseNN, train_nn, predict_nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("LightGBM vs Shallow NN vs Ensemble — 5-Fold CV Comparison")
    logger.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    X_path = DATA_PROCESSED / "training_matrix.parquet"
    y_path = DATA_PROCESSED / "training_target.parquet"

    if not X_path.exists() or not y_path.exists():
        logger.error(
            "Training data not found. Run the pipeline first to create "
            "data/processed/training_matrix.parquet and training_target.parquet"
        )
        sys.exit(1)

    X_df = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)

    X = X_df.values.astype(np.float32)
    y = y_df["pct_inhibition"].values.astype(np.float32)
    feature_names = list(X_df.columns)

    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}")

    # ── 5-Fold CV ────────────────────────────────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        logger.info(f"\n{'─'*50}")
        logger.info(f"Fold {fold_idx + 1}/5 — train={len(train_idx)}, test={len(test_idx)}")
        logger.info(f"{'─'*50}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── LightGBM ────────────────────────────────────────────
        logger.info("Training LightGBM...")
        t0 = time.time()

        lgbm_params = LIGHTGBM_DEFAULT_PARAMS.copy()
        lgbm_model = lgb.LGBMRegressor(**lgbm_params)
        lgbm_model.fit(
            X_train,
            y_train,
            feature_name=feature_names,
            callbacks=[lgb.log_evaluation(0)],  # suppress per-iteration logs
        )
        lgbm_preds = lgbm_model.predict(X_test)
        lgbm_time = time.time() - t0

        lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_preds))
        lgbm_r2 = r2_score(y_test, lgbm_preds)
        logger.info(f"  LightGBM — RMSE={lgbm_rmse:.3f}, R2={lgbm_r2:.3f} ({lgbm_time:.1f}s)")

        # ── Shallow NN ──────────────────────────────────────────
        logger.info("Training Shallow NN...")
        t0 = time.time()

        # Standardize features for NN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Split training into train/val for early stopping (80/20)
        n_train = len(X_train_scaled)
        rng = np.random.RandomState(RANDOM_SEED + fold_idx)
        val_mask = rng.rand(n_train) < 0.2
        train_mask = ~val_mask

        nn_model = train_nn(
            X_train=X_train_scaled[train_mask],
            y_train=y_train[train_mask],
            X_val=X_train_scaled[val_mask],
            y_val=y_train[val_mask],
            hidden_dims=[256, 128],
            dropout=0.3,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            max_epochs=200,
            patience=15,
            device="cpu",
        )

        nn_preds = predict_nn(nn_model, X_test_scaled)
        nn_time = time.time() - t0

        nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
        nn_r2 = r2_score(y_test, nn_preds)
        logger.info(f"  Shallow NN — RMSE={nn_rmse:.3f}, R2={nn_r2:.3f} ({nn_time:.1f}s)")

        # ── Ensemble (average predictions) ──────────────────────
        ens_preds = (lgbm_preds + nn_preds) / 2.0
        ens_rmse = np.sqrt(mean_squared_error(y_test, ens_preds))
        ens_r2 = r2_score(y_test, ens_preds)
        logger.info(f"  Ensemble  — RMSE={ens_rmse:.3f}, R2={ens_r2:.3f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "lgbm_rmse": round(lgbm_rmse, 4),
            "lgbm_r2": round(lgbm_r2, 4),
            "lgbm_time_s": round(lgbm_time, 1),
            "nn_rmse": round(nn_rmse, 4),
            "nn_r2": round(nn_r2, 4),
            "nn_time_s": round(nn_time, 1),
            "ensemble_rmse": round(ens_rmse, 4),
            "ensemble_r2": round(ens_r2, 4),
        })

    # ── Save per-fold results ────────────────────────────────────────
    results_df = pd.DataFrame(fold_results)
    comparison_path = RESULTS / "nn_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    logger.info(f"\nPer-fold results saved to {comparison_path}")

    # ── Summary ──────────────────────────────────────────────────────
    summary_rows = []
    for model_name, rmse_col, r2_col in [
        ("LightGBM", "lgbm_rmse", "lgbm_r2"),
        ("ShallowNN", "nn_rmse", "nn_r2"),
        ("Ensemble", "ensemble_rmse", "ensemble_r2"),
    ]:
        summary_rows.append({
            "model": model_name,
            "rmse_mean": round(results_df[rmse_col].mean(), 4),
            "rmse_std": round(results_df[rmse_col].std(), 4),
            "r2_mean": round(results_df[r2_col].mean(), 4),
            "r2_std": round(results_df[r2_col].std(), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS / "nn_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    # ── Print final summary ──────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY — 5-Fold Cross-Validation")
    logger.info("=" * 70)
    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['model']:12s}  RMSE = {row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f}  "
            f"R2 = {row['r2_mean']:.4f} +/- {row['r2_std']:.4f}"
        )
    logger.info("=" * 70)

    # ── Print per-fold table ─────────────────────────────────────────
    logger.info("\nPer-fold details:")
    logger.info(results_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    main()
