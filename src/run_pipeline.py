"""
INVEREX MVP — Full pipeline orchestrator.

Usage:
    python -m src.run_pipeline --demo          # Demo mode (synthetic data)
    python -m src.run_pipeline --real           # Real data (requires downloads)
    python -m src.run_pipeline --optimize       # With HP optimization
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_CACHE, DATA_PROCESSED, RESULTS
from src.features.build_training_matrix import build_training_matrix
from src.models.train_lightgbm import train_lightgbm

logger = logging.getLogger(__name__)


def run_pipeline(
    use_demo: bool = True,
    optimize_hparams: bool = False,
    n_optuna_trials: int = 30,
):
    """Run the full INVEREX data-matching and training pipeline."""

    logger.info("=" * 70)
    logger.info("INVEREX MVP — Drug-Response Prediction Pipeline")
    logger.info(f"  Mode: {'DEMO (synthetic data)' if use_demo else 'REAL (downloaded data)'}")
    logger.info(f"  HP optimization: {optimize_hparams}")
    logger.info("=" * 70)

    # ── Phase 1: Build training matrix ─────────────────────────────
    logger.info("\n📊 Phase 1: Building training matrix...")
    logger.info("  Matching LINCS L1000 × PharmacoDB × PubChem for breast cancer")

    X, y, feature_names = build_training_matrix(
        use_demo=use_demo,
        cache_dir=DATA_CACHE,
        output_dir=DATA_PROCESSED,
    )

    logger.info(f"\n  ✓ Training matrix ready:")
    logger.info(f"    Samples:  {X.shape[0]:,}")
    logger.info(f"    Features: {X.shape[1]:,}")

    gene_feats = [f for f in feature_names
                  if not f.startswith("ecfp_") and f != "log_dose_um"]
    ecfp_feats = [f for f in feature_names if f.startswith("ecfp_")]
    logger.info(f"    • Gene z-scores: {len(gene_feats)}")
    logger.info(f"    • ECFP4 bits:    {len(ecfp_feats)}")
    logger.info(f"    • Dose feature:  1")
    logger.info(f"    Target range: [{y.min():.1f}%, {y.max():.1f}%]")
    logger.info(f"    Target mean:  {y.mean():.1f}% ± {y.std():.1f}%")

    # ── Phase 2: Train LightGBM ────────────────────────────────────
    logger.info("\n🌲 Phase 2: Training LightGBM drug-response model...")

    model = train_lightgbm(
        X, y,
        optimize_hparams=optimize_hparams,
        n_optuna_trials=n_optuna_trials,
        output_dir=RESULTS,
    )

    logger.info(f"\n  ✓ Model trained:")
    logger.info(f"    Trees: {model.n_estimators}")
    logger.info(f"    Features: {model.n_features_}")

    # ── Phase 3: Quick sanity check ────────────────────────────────
    logger.info("\n🔍 Phase 3: Sanity checks...")

    # Predict on training data (should be reasonably good)
    train_preds = model.predict(X)
    train_rmse = np.sqrt(np.mean((train_preds - y.values) ** 2))
    train_corr = np.corrcoef(train_preds, y.values)[0, 1]

    logger.info(f"  Training RMSE: {train_rmse:.3f}")
    logger.info(f"  Training correlation: {train_corr:.3f}")

    # Feature importance summary
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    top_genes = imp[
        ~imp["feature"].str.startswith("ecfp_")
        & (imp["feature"] != "log_dose_um")
    ].head(10)

    logger.info("\n  Top 10 gene features:")
    for _, row in top_genes.iterrows():
        logger.info(f"    {row['feature']:12s}  importance={row['importance']:.0f}")

    dose_imp = imp[imp["feature"] == "log_dose_um"]["importance"].values
    if len(dose_imp) > 0:
        logger.info(f"    {'log_dose_um':12s}  importance={dose_imp[0]:.0f}")

    # ── Summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("✅ Pipeline complete!")
    logger.info(f"   Model saved to: {RESULTS / 'lightgbm_drug_model.joblib'}")
    logger.info(f"   Metrics saved to: {RESULTS / 'lightgbm_metrics.json'}")
    logger.info(f"   Feature importances: {RESULTS / 'feature_importances.csv'}")
    logger.info("=" * 70)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INVEREX MVP Pipeline")
    parser.add_argument(
        "--demo", action="store_true", default=True,
        help="Use demo/synthetic data (default)"
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Use real downloaded data"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run Bayesian HP optimization"
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of Optuna trials (default: 30)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    use_demo = not args.real
    run_pipeline(
        use_demo=use_demo,
        optimize_hparams=args.optimize,
        n_optuna_trials=args.trials,
    )
