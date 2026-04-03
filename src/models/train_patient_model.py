"""
Train a LightGBM drug-response model directly on pooled CTR-DB
patient data using leave-one-dataset-out cross-validation (LODO-CV).

Unlike the cell-line model (trained on LINCS x GDSC2), this model
is trained on patient expression + clinical response labels from
multiple GEO breast-cancer datasets.

Each patient is represented by:
  - Gene z-scores on L1000 landmark genes (computed within their dataset)
  - Binary response label (1 = responder / pCR, 0 = non-responder / RD)

Evaluation: for each held-out dataset, train on all others and predict
on the held-out set.  Report AUC, balanced accuracy, and sensitivity.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_CACHE, DATA_RAW, RANDOM_SEED, RESULTS
from src.data_ingestion.ctrdb import load_all_breast_ctrdb
from src.data_ingestion.lincs import load_landmark_genes

logger = logging.getLogger(__name__)


def train_patient_model_lodo(
    data_dir: Path = DATA_RAW / "ctrdb",
    output_path: Path = RESULTS / "patient_model_lodo_results.csv",
    model_output: Path = RESULTS / "patient_lightgbm_model.joblib",
) -> pd.DataFrame:
    """
    Train and evaluate a patient drug-response model using
    leave-one-dataset-out cross-validation.

    Steps:
      1. Load all CTR-DB datasets.
      2. For each dataset, compute gene z-scores on L1000 genes.
      3. Pool data across datasets with dataset labels.
      4. LODO-CV: hold out one dataset, train on the rest.
      5. Report per-fold and aggregate metrics.
      6. Train a final model on all data and save.

    Returns a DataFrame with per-fold metrics.
    """
    import joblib
    import lightgbm as lgb
    from sklearn.metrics import (
        balanced_accuracy_score,
        roc_auc_score,
    )

    # Load all datasets
    datasets = load_all_breast_ctrdb(data_dir)
    if len(datasets) < 2:
        logger.error(
            f"Need at least 2 datasets for LODO-CV, got {len(datasets)}"
        )
        return pd.DataFrame()

    # Load landmark genes
    landmark_df = load_landmark_genes()
    landmark_genes = landmark_df["gene_symbol"].tolist()

    # Build pooled dataset
    all_X = []
    all_y = []
    all_dataset_ids = []

    for geo_id, (expr, labels) in datasets.items():
        # Find available landmark genes
        available_genes = [g for g in landmark_genes if g in expr.columns]
        if len(available_genes) < 50:
            logger.warning(
                f"Skipping {geo_id}: only {len(available_genes)} "
                f"landmark genes"
            )
            continue

        # Z-score within dataset
        expr_lm = expr[available_genes]
        cohort_mean = expr_lm.mean(axis=0)
        cohort_std = expr_lm.std(axis=0).replace(0, 1)
        expr_z = (expr_lm - cohort_mean) / cohort_std

        # Use a fixed set of genes (the intersection across datasets)
        all_X.append(expr_z)
        all_y.append(labels)
        all_dataset_ids.extend([geo_id] * len(labels))

    if not all_X:
        logger.error("No usable datasets after filtering")
        return pd.DataFrame()

    # Find common genes across all datasets
    common_genes = set(all_X[0].columns)
    for x in all_X[1:]:
        common_genes &= set(x.columns)
    common_genes = sorted(common_genes)

    logger.info(f"Common genes across {len(all_X)} datasets: {len(common_genes)}")

    if len(common_genes) < 50:
        logger.error(f"Too few common genes ({len(common_genes)})")
        return pd.DataFrame()

    # Align to common genes
    X_list = [x[common_genes] for x in all_X]
    X_pooled = pd.concat(X_list, axis=0).reset_index(drop=True)
    y_pooled = pd.concat(all_y, axis=0).reset_index(drop=True).astype(int)
    dataset_ids = pd.Series(all_dataset_ids, name="dataset_id")

    logger.info(
        f"Pooled dataset: {len(X_pooled)} patients, {len(common_genes)} genes, "
        f"{y_pooled.sum()} responders / {(1 - y_pooled).sum()} non-responders, "
        f"{dataset_ids.nunique()} datasets"
    )

    # Conservative hyperparameters for small datasets
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_child_samples": 10,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "class_weight": "balanced",
    }

    # LODO-CV
    unique_datasets = dataset_ids.unique()
    fold_results = []

    for held_out in unique_datasets:
        # Split
        test_mask = dataset_ids == held_out
        train_mask = ~test_mask

        X_train = X_pooled[train_mask].reset_index(drop=True)
        y_train = y_pooled[train_mask].reset_index(drop=True)
        X_test = X_pooled[test_mask].reset_index(drop=True)
        y_test = y_pooled[test_mask].reset_index(drop=True)

        if len(y_test) < 10 or y_test.nunique() < 2:
            logger.info(f"Skipping {held_out}: insufficient test data")
            continue

        n_test_resp = int(y_test.sum())
        n_test_nonresp = len(y_test) - n_test_resp
        if n_test_resp < 3 or n_test_nonresp < 3:
            logger.info(
                f"Skipping {held_out}: too few in one class "
                f"(R={n_test_resp}, NR={n_test_nonresp})"
            )
            continue

        # Train
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        sensitivity = (
            y_pred[y_test == 1].sum() / y_test.sum()
            if y_test.sum() > 0
            else 0.0
        )
        specificity = (
            (1 - y_pred[y_test == 0]).sum() / (1 - y_test).sum()
            if (1 - y_test).sum() > 0
            else 0.0
        )

        fold_result = {
            "held_out_dataset": held_out,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_test_responders": n_test_resp,
            "n_test_nonresponders": n_test_nonresp,
            "auc": round(auc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
        }
        fold_results.append(fold_result)

        logger.info(
            f"  {held_out}: AUC={auc:.3f}, BalAcc={bal_acc:.3f}, "
            f"Sens={sensitivity:.3f}, Spec={specificity:.3f}"
        )

    if not fold_results:
        logger.error("No successful LODO-CV folds")
        return pd.DataFrame()

    results_df = pd.DataFrame(fold_results)

    # Add summary row
    summary = {
        "held_out_dataset": "MEAN",
        "n_train": int(results_df["n_train"].mean()),
        "n_test": int(results_df["n_test"].mean()),
        "n_test_responders": int(results_df["n_test_responders"].mean()),
        "n_test_nonresponders": int(results_df["n_test_nonresponders"].mean()),
        "auc": round(results_df["auc"].mean(), 4),
        "balanced_accuracy": round(results_df["balanced_accuracy"].mean(), 4),
        "sensitivity": round(results_df["sensitivity"].mean(), 4),
        "specificity": round(results_df["specificity"].mean(), 4),
    }
    results_df = pd.concat(
        [results_df, pd.DataFrame([summary])], ignore_index=True
    )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nLODO-CV results saved to {output_path}")

    # Train final model on all data
    logger.info("Training final patient model on all data ...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_pooled, y_pooled)
    joblib.dump(final_model, model_output)
    logger.info(f"Patient model saved to {model_output}")

    # Feature importances
    importances = pd.DataFrame({
        "gene": common_genes,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_path = RESULTS / "patient_model_feature_importances.csv"
    importances.to_csv(imp_path, index=False)
    logger.info(f"Feature importances saved to {imp_path}")
    logger.info(f"Top 10 genes:\n{importances.head(10).to_string()}")

    return results_df


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    results = train_patient_model_lodo()
    if len(results) > 0:
        print("\n" + "=" * 70)
        print("PATIENT MODEL LODO-CV RESULTS")
        print("=" * 70)
        print(results.to_string(index=False))
    else:
        print("\nNo results. Download CTR-DB datasets first.")
