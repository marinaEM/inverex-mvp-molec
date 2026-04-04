"""
Split conformal prediction for drug-response models (Step 6).

Provides calibrated uncertainty quantification for both regression
(cell-line pct_inhibition) and classification (patient response)
predictions.

Method: Split Conformal Prediction
-----------------------------------
1. Split training data into proper-training (70%) and calibration (30%).
2. Train the model on the proper-training set.
3. Compute nonconformity scores on the calibration set:
   - Regression: |y_true - y_pred|
   - Classification: 1 - predicted probability for the true class
4. For new predictions, construct:
   - Regression: prediction intervals [pred - q, pred + q]
     where q = quantile(scores, (1-alpha)(1+1/n_cal))
   - Classification: conformal p-values per class

Reports coverage rate, mean interval width, fraction of confident
predictions.

Entry point: ``run_conformal_experiment()`` or
``pixi run python -m src.models.conformal_prediction``
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Avoid OpenMP conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED, RESULTS, RANDOM_SEED

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Conformal prediction for regression
# ======================================================================

class ConformalRegressor:
    """
    Split conformal prediction wrapper for regression models.

    Parameters
    ----------
    alpha : float
        Significance level (1 - desired coverage). Default 0.1 for 90%.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.q_hat: Optional[float] = None
        self.n_calibration: int = 0

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Calibrate using residuals from a calibration set.

        Parameters
        ----------
        y_true : array of true values on calibration set.
        y_pred : array of predictions on calibration set.

        Returns
        -------
        q_hat : the conformal quantile threshold.
        """
        self.calibration_scores = np.abs(y_true - y_pred)
        self.n_calibration = len(self.calibration_scores)

        # Quantile with finite-sample correction
        level = (1 - self.alpha) * (1 + 1 / self.n_calibration)
        level = min(level, 1.0)  # cap at 1.0
        self.q_hat = float(np.quantile(self.calibration_scores, level))

        logger.info(
            "Calibrated conformal regressor: alpha=%.2f, n_cal=%d, "
            "q_hat=%.4f, score_mean=%.4f, score_median=%.4f",
            self.alpha,
            self.n_calibration,
            self.q_hat,
            float(np.mean(self.calibration_scores)),
            float(np.median(self.calibration_scores)),
        )
        return self.q_hat

    def predict_intervals(
        self,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct prediction intervals.

        Parameters
        ----------
        y_pred : point predictions for new data.

        Returns
        -------
        lower : lower bounds of prediction intervals.
        upper : upper bounds of prediction intervals.
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() first.")

        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        return lower, upper

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """
        Evaluate coverage and interval width on test data.

        Parameters
        ----------
        y_true : true values on test set.
        y_pred : predictions on test set.

        Returns
        -------
        dict with coverage_rate, mean_interval_width, median_interval_width,
        fraction_confident (interval width < median response range).
        """
        lower, upper = self.predict_intervals(y_pred)
        covered = (y_true >= lower) & (y_true <= upper)
        coverage_rate = float(np.mean(covered))

        widths = upper - lower
        mean_width = float(np.mean(widths))
        median_width = float(np.median(widths))

        # Fraction of "confident" predictions:
        # intervals narrower than half the response range
        response_range = float(np.ptp(y_true))
        confident_threshold = response_range / 2
        fraction_confident = float(np.mean(widths < confident_threshold))

        return {
            "alpha": self.alpha,
            "target_coverage": 1 - self.alpha,
            "coverage_rate": round(coverage_rate, 4),
            "n_test": len(y_true),
            "n_calibration": self.n_calibration,
            "q_hat": round(self.q_hat, 4),
            "mean_interval_width": round(mean_width, 4),
            "median_interval_width": round(median_width, 4),
            "fraction_confident": round(fraction_confident, 4),
            "response_range": round(response_range, 4),
        }


# ======================================================================
# 2. Conformal prediction for classification
# ======================================================================

class ConformalClassifier:
    """
    Split conformal prediction wrapper for binary classification.

    Uses the conformal p-value approach: for a new sample, include
    class c in the prediction set if p_c > alpha.

    Parameters
    ----------
    alpha : float
        Significance level. Default 0.1 for 90% coverage.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.calibration_scores: Optional[dict[int, np.ndarray]] = None
        self.n_calibration: int = 0

    def calibrate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ):
        """
        Calibrate using predicted probabilities on calibration set.

        Parameters
        ----------
        y_true : binary labels (0/1).
        y_proba : predicted probabilities for class 1.
        """
        self.calibration_scores = {}
        # Nonconformity score: 1 - predicted probability of true class
        scores_0 = 1 - (1 - y_proba[y_true == 0])  # = y_proba for class-0 samples
        scores_1 = 1 - y_proba[y_true == 1]

        self.calibration_scores[0] = scores_0
        self.calibration_scores[1] = scores_1
        self.n_calibration = len(y_true)

        logger.info(
            "Calibrated conformal classifier: alpha=%.2f, n_cal=%d, "
            "n_class0=%d, n_class1=%d",
            self.alpha, self.n_calibration,
            len(scores_0), len(scores_1),
        )

    def predict_sets(
        self,
        y_proba: np.ndarray,
    ) -> list[set[int]]:
        """
        Construct conformal prediction sets.

        Parameters
        ----------
        y_proba : predicted probabilities for class 1.

        Returns
        -------
        list of sets, one per sample, containing predicted classes.
        """
        if self.calibration_scores is None:
            raise ValueError("Must call calibrate() first.")

        prediction_sets = []
        for p1 in y_proba:
            pset = set()
            # p-value for class 0
            score_if_0 = p1  # nonconformity: 1 - P(true class) = 1 - (1-p1) = p1
            n0 = len(self.calibration_scores[0])
            if n0 > 0:
                pval_0 = float(np.mean(self.calibration_scores[0] >= score_if_0))
            else:
                pval_0 = 1.0
            if pval_0 > self.alpha:
                pset.add(0)

            # p-value for class 1
            score_if_1 = 1 - p1  # nonconformity: 1 - P(true class) = 1 - p1
            n1 = len(self.calibration_scores[1])
            if n1 > 0:
                pval_1 = float(np.mean(self.calibration_scores[1] >= score_if_1))
            else:
                pval_1 = 1.0
            if pval_1 > self.alpha:
                pset.add(1)

            prediction_sets.append(pset)

        return prediction_sets

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict:
        """
        Evaluate coverage and efficiency on test data.

        Parameters
        ----------
        y_true : binary labels (0/1).
        y_proba : predicted probabilities for class 1.

        Returns
        -------
        dict with coverage, avg_set_size, fraction_singleton, fraction_empty.
        """
        pred_sets = self.predict_sets(y_proba)

        covered = sum(1 for yt, ps in zip(y_true, pred_sets) if yt in ps)
        coverage_rate = covered / len(y_true)

        set_sizes = [len(ps) for ps in pred_sets]
        avg_set_size = float(np.mean(set_sizes))
        fraction_singleton = float(np.mean([s == 1 for s in set_sizes]))
        fraction_empty = float(np.mean([s == 0 for s in set_sizes]))
        fraction_both = float(np.mean([s == 2 for s in set_sizes]))

        return {
            "alpha": self.alpha,
            "target_coverage": 1 - self.alpha,
            "coverage_rate": round(coverage_rate, 4),
            "n_test": len(y_true),
            "n_calibration": self.n_calibration,
            "avg_set_size": round(avg_set_size, 4),
            "fraction_singleton": round(fraction_singleton, 4),
            "fraction_empty": round(fraction_empty, 4),
            "fraction_both_classes": round(fraction_both, 4),
        }


# ======================================================================
# 3. Run conformal experiment on cell-line regression model
# ======================================================================

def run_regression_conformal(
    alphas: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Conformal prediction on the cell-line NN model.

    1. Load training matrix.
    2. Split 70/30 into proper-train / calibration.
    3. Train NN on proper-train.
    4. Calibrate conformal wrapper on calibration set.
    5. Evaluate on calibration set (held-out from training).

    Returns DataFrame of metrics for each alpha level.
    """
    import torch
    from src.models.interaction_nn import DrugResponseNN, train_nn, predict_nn

    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20]

    logger.info("Loading training data ...")
    X_df = pd.read_parquet(DATA_PROCESSED / "training_matrix.parquet")
    y_df = pd.read_parquet(DATA_PROCESSED / "training_target.parquet")
    X = X_df.values.astype(np.float32)
    y = y_df.iloc[:, 0].values.astype(np.float32)

    logger.info("  X: %s, y: %s", X.shape, y.shape)

    # 70/30 split
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED,
    )
    logger.info(
        "  Train: %d samples, Calibration: %d samples",
        len(X_train), len(X_cal),
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)

    # Train NN
    logger.info("Training NN on proper-training set ...")
    # Use a portion of training as validation for early stopping
    n_val = max(int(len(X_train_s) * 0.15), 20)
    X_tr = X_train_s[:-n_val]
    y_tr = y_train[:-n_val]
    X_va = X_train_s[-n_val:]
    y_va = y_train[-n_val:]

    model = train_nn(
        X_tr, y_tr, X_va, y_va,
        hidden_dims=[256, 128],
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        max_epochs=200,
        patience=15,
    )

    # Predictions on calibration set
    y_cal_pred = predict_nn(model, X_cal_s)
    cal_rmse = float(np.sqrt(np.mean((y_cal_pred - y_cal) ** 2)))
    logger.info("Calibration set RMSE: %.4f", cal_rmse)

    # Further split calibration into calibration + test for evaluation
    # (so coverage is evaluated on truly unseen data)
    n_cal_proper = int(len(X_cal_s) * 0.5)
    X_cal_proper = X_cal_s[:n_cal_proper]
    y_cal_proper = y_cal[:n_cal_proper]
    y_cal_pred_proper = y_cal_pred[:n_cal_proper]

    X_test = X_cal_s[n_cal_proper:]
    y_test = y_cal[n_cal_proper:]
    y_test_pred = y_cal_pred[n_cal_proper:]

    results = []
    for alpha in alphas:
        logger.info("\n--- alpha = %.2f ---", alpha)
        cf = ConformalRegressor(alpha=alpha)
        cf.calibrate(y_cal_proper, y_cal_pred_proper)
        metrics = cf.evaluate(y_test, y_test_pred)
        metrics["model"] = "regression_nn"
        metrics["split"] = "test"
        results.append(metrics)
        logger.info(
            "  Coverage: %.4f (target: %.2f), "
            "Width: %.4f, Confident: %.4f",
            metrics["coverage_rate"],
            1 - alpha,
            metrics["mean_interval_width"],
            metrics["fraction_confident"],
        )

    return pd.DataFrame(results)


# ======================================================================
# 4. Run conformal experiment on patient classification model
# ======================================================================

def run_classification_conformal(
    alphas: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Conformal prediction on the patient L1-logistic classification model.

    Uses pooled CTR-DB data with a 70/30 train/calibrate split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning

    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20]

    logger.info("Loading CTR-DB data for classification conformal ...")

    # Load datasets (reuse the stacking ensemble loader)
    ctrdb_dir = DATA_RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    from src.data_ingestion.lincs import load_landmark_genes
    from src.data_ingestion.ctrdb import load_all_breast_ctrdb

    landmark_df = load_landmark_genes()
    landmark_genes = landmark_df["gene_symbol"].tolist()

    datasets = load_all_breast_ctrdb()
    if len(datasets) < 2:
        logger.warning("Not enough CTR-DB datasets for classification conformal")
        return pd.DataFrame()

    # Pool all datasets
    all_X = []
    all_y = []
    for geo_id, (expr, labels) in datasets.items():
        available_genes = [g for g in landmark_genes if g in expr.columns]
        if len(available_genes) < 50:
            continue
        expr_lm = expr[available_genes]
        cohort_mean = expr_lm.mean(axis=0)
        cohort_std = expr_lm.std(axis=0).replace(0, 1)
        expr_z = (expr_lm - cohort_mean) / cohort_std
        all_X.append(expr_z)
        all_y.append(labels)

    if not all_X:
        logger.warning("No usable CTR-DB datasets")
        return pd.DataFrame()

    # Common genes
    common_genes = set(all_X[0].columns)
    for x in all_X[1:]:
        common_genes &= set(x.columns)
    common_genes = sorted(common_genes)

    X_list = [x[common_genes] for x in all_X]
    X_pooled = pd.concat(X_list, axis=0).reset_index(drop=True)
    y_pooled = pd.concat(all_y, axis=0).reset_index(drop=True).astype(int)

    X = X_pooled.values.astype(np.float64)
    y = y_pooled.values

    # Handle NaN values (from gene coverage differences across datasets)
    X = np.nan_to_num(X, nan=0.0)

    logger.info(
        "Pooled: %d patients, %d genes, %d responders",
        len(X), len(common_genes), int(y.sum()),
    )

    # 70/30 split
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y,
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0)
    X_cal_s = np.nan_to_num(X_cal_s, nan=0.0)

    # Train L1-logistic
    logger.info("Training L1-logistic on proper-training set ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(
            C=0.05, solver="liblinear", max_iter=2000, random_state=42,
        )
        clf.fit(X_train_s, y_train)

    # Predictions on calibration set
    y_cal_proba = clf.predict_proba(X_cal_s)[:, 1]

    # Split calibration into proper-cal + test
    n_cal_proper = int(len(X_cal_s) * 0.5)
    y_cal_proper_labels = y_cal[:n_cal_proper]
    y_cal_proper_proba = y_cal_proba[:n_cal_proper]
    y_test = y_cal[n_cal_proper:]
    y_test_proba = y_cal_proba[n_cal_proper:]

    results = []
    for alpha in alphas:
        logger.info("\n--- Classification alpha = %.2f ---", alpha)
        cf = ConformalClassifier(alpha=alpha)
        cf.calibrate(y_cal_proper_labels, y_cal_proper_proba)
        metrics = cf.evaluate(y_test, y_test_proba)
        metrics["model"] = "classification_l1_logistic"
        metrics["split"] = "test"
        results.append(metrics)
        logger.info(
            "  Coverage: %.4f (target: %.2f), "
            "Avg set size: %.4f, Singleton: %.4f, Empty: %.4f",
            metrics["coverage_rate"],
            1 - alpha,
            metrics["avg_set_size"],
            metrics["fraction_singleton"],
            metrics["fraction_empty"],
        )

    return pd.DataFrame(results)


# ======================================================================
# 5. Main experiment
# ======================================================================

def run_conformal_experiment():
    """
    End-to-end conformal prediction experiment.

    1. Regression conformal on cell-line NN model.
    2. Classification conformal on patient L1-logistic model.
    3. Save combined metrics.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.time()

    logger.info("=" * 60)
    logger.info("STEP 6: CONFORMAL PREDICTION")
    logger.info("=" * 60)

    alphas = [0.05, 0.10, 0.15, 0.20]

    # ---- Regression ----
    logger.info("\n" + "-" * 60)
    logger.info("Regression conformal prediction (cell-line NN)")
    logger.info("-" * 60)
    reg_results = run_regression_conformal(alphas=alphas)

    # ---- Classification ----
    logger.info("\n" + "-" * 60)
    logger.info("Classification conformal prediction (patient L1-logistic)")
    logger.info("-" * 60)
    try:
        cls_results = run_classification_conformal(alphas=alphas)
    except Exception as e:
        logger.warning("Classification conformal failed: %s", e)
        cls_results = pd.DataFrame()

    # ---- Combine and save ----
    all_results = pd.concat([reg_results, cls_results], ignore_index=True)

    logger.info("\n" + "=" * 60)
    logger.info("CONFORMAL PREDICTION RESULTS")
    logger.info("=" * 60)

    for _, row in all_results.iterrows():
        if row["model"] == "regression_nn":
            logger.info(
                "  Regression alpha=%.2f: coverage=%.4f (target=%.2f), "
                "width=%.4f, confident=%.4f",
                row["alpha"],
                row["coverage_rate"],
                row["target_coverage"],
                row.get("mean_interval_width", 0),
                row.get("fraction_confident", 0),
            )
        else:
            logger.info(
                "  Classification alpha=%.2f: coverage=%.4f (target=%.2f), "
                "set_size=%.4f, singleton=%.4f",
                row["alpha"],
                row["coverage_rate"],
                row["target_coverage"],
                row.get("avg_set_size", 0),
                row.get("fraction_singleton", 0),
            )

    # Save
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "conformal_prediction_metrics.tsv"
    all_results.to_csv(out_path, sep="\t", index=False)
    logger.info("\nResults saved to %s", out_path)

    elapsed = time.time() - t_start
    logger.info("Total time: %.0fs", elapsed)

    return all_results


if __name__ == "__main__":
    run_conformal_experiment()
