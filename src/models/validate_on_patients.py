"""
Validate the cell-line-trained drug-response model on real patient data.

For each CTR-DB breast-cancer dataset:
  1. Load expression matrix + binary response labels.
  2. Compute patient disease signatures on L1000 landmark genes
     (z-score relative to the cohort mean).
  3. Predict drug response with the trained LightGBM model.
  4. Compare predicted scores between responders and non-responders.
  5. Compute: AUC, Wilcoxon rank-sum p-value, mean score difference.

The key hypothesis: if the cell-line model captures biology relevant
to real patients, predicted inhibition scores should be higher for
responders (pCR) than non-responders (RD).
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.config import DATA_CACHE, DATA_RAW, RESULTS, ECFP_NBITS
from src.data_ingestion.ctrdb import load_all_breast_ctrdb
from src.data_ingestion.lincs import load_landmark_genes

logger = logging.getLogger(__name__)


def validate_on_ctrdb_patients(
    model_path: Path = RESULTS / "lightgbm_drug_model.joblib",
    fp_path: Path = DATA_CACHE / "drug_fingerprints.parquet",
    data_dir: Path = DATA_RAW / "ctrdb",
    output_path: Path = RESULTS / "ctrdb_validation_results.csv",
) -> pd.DataFrame:
    """
    Run validation on all available CTR-DB patient datasets.

    For each dataset, computes a single aggregate prediction score
    per patient (using available drug fingerprints), then evaluates
    discrimination between responders and non-responders.

    Returns a DataFrame with one row per dataset and columns:
        geo_id, drug, n_patients, n_responders, n_nonresponders,
        auc, wilcoxon_pvalue, mean_score_responders,
        mean_score_nonresponders, mean_diff, cohens_d
    """
    # Load model
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return pd.DataFrame()

    model = joblib.load(model_path)
    model_features = model.feature_name_
    gene_features = [
        f for f in model_features
        if not f.startswith("ecfp_") and f != "log_dose_um"
    ]
    ecfp_features = [f for f in model_features if f.startswith("ecfp_")]
    logger.info(
        f"Loaded model: {len(gene_features)} genes, "
        f"{len(ecfp_features)} ECFP bits, 1 dose feature"
    )

    # Load drug fingerprints
    if fp_path.exists():
        drug_fps = pd.read_parquet(fp_path)
        logger.info(f"Loaded {len(drug_fps)} drug fingerprints")
    else:
        logger.warning("No drug fingerprints; using random ECFP for validation")
        drug_fps = None

    # Load landmark genes
    landmark_df = load_landmark_genes()
    landmark_genes = landmark_df["gene_symbol"].tolist()

    # Load datasets
    datasets = load_all_breast_ctrdb(data_dir)
    if not datasets:
        logger.error("No CTR-DB datasets loaded. Run download first.")
        return pd.DataFrame()

    # Load catalog for drug info
    catalog_path = data_dir / "catalog.csv"
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        geo_to_drug = dict(zip(catalog["geo_source"], catalog["drug"]))
    else:
        geo_to_drug = {}

    # Validate each dataset
    results = []
    for geo_id, (expr, labels) in datasets.items():
        logger.info(f"\nValidating {geo_id} ...")
        drug_name = geo_to_drug.get(geo_id, "unknown")

        result = _validate_single_dataset(
            geo_id=geo_id,
            drug_name=drug_name,
            expr=expr,
            labels=labels,
            model=model,
            model_features=model_features,
            gene_features=gene_features,
            ecfp_features=ecfp_features,
            landmark_genes=landmark_genes,
            drug_fps=drug_fps,
        )

        if result is not None:
            results.append(result)

    if not results:
        logger.warning("No successful validations")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nValidation results saved to {output_path}")

    # Summary
    sig_count = (results_df["wilcoxon_pvalue"] < 0.05).sum()
    logger.info(
        f"\nSummary: {len(results_df)} datasets validated, "
        f"{sig_count} with significant separation (p<0.05)"
    )
    logger.info(f"Mean AUC across datasets: {results_df['auc'].mean():.3f}")

    return results_df


def _validate_single_dataset(
    geo_id: str,
    drug_name: str,
    expr: pd.DataFrame,
    labels: pd.Series,
    model,
    model_features: list[str],
    gene_features: list[str],
    ecfp_features: list[str],
    landmark_genes: list[str],
    drug_fps: Optional[pd.DataFrame],
) -> Optional[dict]:
    """Validate on a single dataset. Returns a result dict or None."""
    n_patients = len(labels)
    n_resp = int(labels.sum())
    n_nonresp = n_patients - n_resp

    if n_resp < 3 or n_nonresp < 3:
        logger.warning(
            f"Skipping {geo_id}: too few in one class "
            f"(R={n_resp}, NR={n_nonresp})"
        )
        return None

    # Compute patient signatures on landmark genes
    # z-score each gene across the cohort
    available_genes = [g for g in gene_features if g in expr.columns]
    if len(available_genes) < 20:
        # Try landmark genes instead
        available_genes = [g for g in landmark_genes if g in expr.columns]
        if len(available_genes) < 20:
            logger.warning(
                f"Skipping {geo_id}: only {len(available_genes)} "
                f"model genes in expression data"
            )
            return None

    logger.info(
        f"{geo_id}: {n_patients} patients, {n_resp} R / {n_nonresp} NR, "
        f"{len(available_genes)} genes available"
    )

    # Z-score expression across the cohort
    expr_subset = expr[available_genes].copy()
    cohort_mean = expr_subset.mean(axis=0)
    cohort_std = expr_subset.std(axis=0).replace(0, 1)
    expr_z = (expr_subset - cohort_mean) / cohort_std

    # Build feature matrix for prediction
    # Use the gene z-scores as patient "disease signature",
    # a representative drug fingerprint, and a standard dose
    scores = _predict_patient_scores(
        expr_z=expr_z,
        model=model,
        model_features=model_features,
        gene_features=gene_features,
        ecfp_features=ecfp_features,
        drug_fps=drug_fps,
    )

    if scores is None or len(scores) == 0:
        return None

    # Align scores with labels
    common = scores.index.intersection(labels.index)
    scores = scores.loc[common]
    labels_aligned = labels.loc[common]

    if len(common) < 10:
        logger.warning(f"{geo_id}: only {len(common)} samples after alignment")
        return None

    resp_scores = scores[labels_aligned == 1]
    nonresp_scores = scores[labels_aligned == 0]

    # Compute metrics
    try:
        auc = roc_auc_score(labels_aligned, scores)
    except ValueError:
        auc = 0.5

    try:
        stat, wilcoxon_p = stats.mannwhitneyu(
            resp_scores, nonresp_scores, alternative="two-sided"
        )
    except ValueError:
        wilcoxon_p = 1.0

    mean_resp = resp_scores.mean()
    mean_nonresp = nonresp_scores.mean()
    mean_diff = mean_resp - mean_nonresp

    # Cohen's d
    pooled_std = np.sqrt(
        (resp_scores.std() ** 2 + nonresp_scores.std() ** 2) / 2
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    logger.info(
        f"  AUC={auc:.3f}, p={wilcoxon_p:.4f}, "
        f"mean_R={mean_resp:.2f}, mean_NR={mean_nonresp:.2f}, "
        f"d={cohens_d:.3f}"
    )

    return {
        "geo_id": geo_id,
        "drug": drug_name,
        "n_patients": len(common),
        "n_responders": int(labels_aligned.sum()),
        "n_nonresponders": int((1 - labels_aligned).sum()),
        "auc": round(auc, 4),
        "wilcoxon_pvalue": round(wilcoxon_p, 6),
        "mean_score_responders": round(mean_resp, 4),
        "mean_score_nonresponders": round(mean_nonresp, 4),
        "mean_diff": round(mean_diff, 4),
        "cohens_d": round(cohens_d, 4),
        "n_genes_used": len([g for g in gene_features if g in expr.columns]),
    }


def _predict_patient_scores(
    expr_z: pd.DataFrame,
    model,
    model_features: list[str],
    gene_features: list[str],
    ecfp_features: list[str],
    drug_fps: Optional[pd.DataFrame],
) -> Optional[pd.Series]:
    """
    Compute a single aggregate predicted score per patient.

    Approach: for each patient, build the feature vector using their
    gene z-scores plus an average drug fingerprint (averaged over all
    drugs) at a standard dose.  This gives us one "general drug
    sensitivity" score per patient.

    If drug fingerprints are available, uses the mean ECFP across
    all drugs. Otherwise, uses zeros (neutral fingerprint).
    """
    n_patients = len(expr_z)

    # Gene features: align to model's expected genes
    gene_matrix = np.zeros((n_patients, len(gene_features)), dtype=np.float32)
    for j, gene in enumerate(gene_features):
        if gene in expr_z.columns:
            gene_matrix[:, j] = expr_z[gene].values.astype(np.float32)

    # ECFP features: use mean fingerprint across available drugs
    if drug_fps is not None:
        ecfp_cols = [c for c in drug_fps.columns if c.startswith("ecfp_")]
        if ecfp_cols:
            mean_fp = drug_fps[ecfp_cols].mean(axis=0).values
            ecfp_matrix = np.tile(mean_fp, (n_patients, 1)).astype(np.float32)
        else:
            ecfp_matrix = np.zeros(
                (n_patients, len(ecfp_features)), dtype=np.float32
            )
    else:
        ecfp_matrix = np.zeros(
            (n_patients, len(ecfp_features)), dtype=np.float32
        )

    # Align ECFP to model feature order
    ecfp_aligned = np.zeros((n_patients, len(ecfp_features)), dtype=np.float32)
    if drug_fps is not None:
        ecfp_cols = [c for c in drug_fps.columns if c.startswith("ecfp_")]
        for j, feat in enumerate(ecfp_features):
            if feat in ecfp_cols:
                idx = ecfp_cols.index(feat)
                ecfp_aligned[:, j] = ecfp_matrix[:, idx]

    # Dose: use log(1 + 1.0) as a standard dose
    dose_col = np.full((n_patients, 1), np.log1p(1.0), dtype=np.float32)

    # Combine
    X = np.concatenate([gene_matrix, ecfp_aligned, dose_col], axis=1)
    X_df = pd.DataFrame(X, columns=model_features, index=expr_z.index)

    # Predict
    try:
        predictions = model.predict(X_df)
        scores = pd.Series(predictions, index=expr_z.index, name="predicted_score")
        return scores
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        return None


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    results = validate_on_ctrdb_patients()
    if len(results) > 0:
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print(
            results[
                ["geo_id", "drug", "n_patients", "auc",
                 "wilcoxon_pvalue", "cohens_d"]
            ].to_string(index=False)
        )
    else:
        print("\nNo validation results. Download CTR-DB datasets first.")
