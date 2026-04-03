"""
Patient-level drug-ranking inference.

Takes a TCGA-BRCA patient's molecular signature and produces a ranked
list of drugs with predicted inhibition scores and interpretability info.

This mirrors scTherapy's inference: substitute the patient's DEGs
(vs. normal/centroid) for the drug-perturbation expression signature,
then predict which drugs at which doses would produce the highest inhibition.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DATA_CACHE,
    DATA_PROCESSED,
    ECFP_NBITS,
    RESULTS,
)
from src.data_ingestion.lincs import load_landmark_genes
from src.data_ingestion.tcga import (
    build_patient_cohort,
    compute_patient_signature,
    load_tcga_expression,
)
from src.ranking.personalized_ranker import PersonalizedDrugRanker

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment-dependent optional dependency
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent optional dependency
    lgb = Any

# Standard doses to evaluate (µM), matching common LINCS concentrations
EVAL_DOSES = [0.04, 0.12, 0.37, 1.11, 3.33, 10.0]


def predict_drugs_for_patient(
    sample_id: str,
    model: Any,
    drug_fingerprints: pd.DataFrame,
    expression: Optional[pd.DataFrame] = None,
    cohort: Optional[pd.DataFrame] = None,
    landmark_genes: Optional[list[str]] = None,
    patient_signature: Optional[pd.Series] = None,
    doses_um: list[float] = EVAL_DOSES,
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Predict drug response rankings for a single TCGA-BRCA patient.

    Workflow (mirroring scTherapy):
      1. Compute patient disease signature (DEGs vs. reference)
      2. For each drug × dose, construct feature vector:
         [patient gene z-scores] + [drug ECFP4] + [log_dose]
      3. Predict percent inhibition with trained LightGBM
      4. For each drug, select the best dose (highest predicted inhibition)
      5. Return ranked drug list

    Args:
        sample_id: TCGA sample identifier
        model: Trained LGBMRegressor
        drug_fingerprints: DataFrame with compound_name + ecfp_* columns
        expression: Expression matrix (optional if patient_signature provided)
        cohort: Patient cohort metadata (optional if patient_signature provided)
        landmark_genes: List of L1000 gene symbols
        patient_signature: Pre-computed signature (overrides expression/cohort)
        doses_um: List of doses to evaluate
        top_k: Number of top drugs to return

    Returns:
        DataFrame with columns:
            drug_name, best_dose_um, predicted_inhibition, confidence,
            + top contributing gene features
    """
    # ── Step 1: Get patient signature ──────────────────────────────
    if patient_signature is None:
        if expression is None or cohort is None or landmark_genes is None:
            raise ValueError(
                "Must provide either patient_signature or "
                "(expression + cohort + landmark_genes)"
            )
        patient_signature = compute_patient_signature(
            sample_id, expression, cohort, landmark_genes,
            method="subtype_centroid",
        )

    # ── Step 2: Identify feature columns from model ────────────────
    model_features = model.feature_name_
    gene_features = [
        f for f in model_features
        if not f.startswith("ecfp_") and f != "log_dose_um"
    ]
    ecfp_features = [f for f in model_features if f.startswith("ecfp_")]

    # Align patient signature to model's gene features
    patient_gene_values = []
    for gene in gene_features:
        if gene in patient_signature.index:
            patient_gene_values.append(float(patient_signature[gene]))
        else:
            patient_gene_values.append(0.0)  # Missing gene → no change
    patient_gene_array = np.array(patient_gene_values, dtype=np.float32)

    # ── Step 3: Build prediction matrix (drugs × doses) ────────────
    compounds = drug_fingerprints["compound_name"].unique()
    ecfp_cols = [c for c in drug_fingerprints.columns if c.startswith("ecfp_")]

    rows = []
    meta_rows = []

    for compound in compounds:
        fp_row = drug_fingerprints[
            drug_fingerprints["compound_name"] == compound
        ].iloc[0]
        fp_values = fp_row[ecfp_cols].values.astype(np.int8)

        # Align ECFP features to model's expected columns
        ecfp_aligned = np.zeros(len(ecfp_features), dtype=np.int8)
        for i, feat in enumerate(ecfp_features):
            if feat in ecfp_cols:
                idx = ecfp_cols.index(feat)
                ecfp_aligned[i] = fp_values[idx]

        for dose in doses_um:
            log_dose = np.float32(np.log1p(dose))
            feature_row = np.concatenate([
                patient_gene_array, ecfp_aligned, [log_dose]
            ])
            rows.append(feature_row)
            meta_rows.append({
                "drug_name": compound,
                "dose_um": dose,
            })

    X_pred = pd.DataFrame(rows, columns=model_features)
    meta = pd.DataFrame(meta_rows)

    # ── Step 4: Predict ────────────────────────────────────────────
    logger.info(
        f"Predicting {len(X_pred)} drug×dose combinations for {sample_id}..."
    )
    predictions = model.predict(X_pred)
    meta["predicted_inhibition"] = predictions.clip(0, 100)

    # ── Step 5: Select best dose per drug ──────────────────────────
    best_per_drug = (
        meta
        .sort_values("predicted_inhibition", ascending=False)
        .groupby("drug_name")
        .first()
        .reset_index()
        .rename(columns={"dose_um": "best_dose_um"})
        .sort_values("predicted_inhibition", ascending=False)
    )

    # ── Step 6: Add interpretability info ──────────────────────────
    # Get feature importances from model
    if hasattr(model, "feature_importances_"):
        feat_imp = dict(zip(model_features, model.feature_importances_))
    else:
        feat_imp = {}

    # For each drug, identify the top contributing gene features
    # (based on model importance × patient signature magnitude)
    contributing_genes = []
    for _, row in best_per_drug.iterrows():
        gene_contributions = {}
        for gene, val in zip(gene_features, patient_gene_array):
            imp = feat_imp.get(gene, 0)
            # Contribution = importance × abs(patient z-score)
            gene_contributions[gene] = imp * abs(val)

        top_genes = sorted(
            gene_contributions.items(), key=lambda x: x[1], reverse=True
        )[:5]
        contributing_genes.append(
            "; ".join([f"{g}({v:.0f})" for g, v in top_genes if v > 0])
        )

    best_per_drug["top_contributing_genes"] = contributing_genes

    # Add confidence tier based on prediction magnitude
    best_per_drug["confidence"] = pd.cut(
        best_per_drug["predicted_inhibition"],
        bins=[-1, 30, 50, 70, 101],
        labels=["low", "moderate", "high", "very_high"],
    )

    return best_per_drug.head(top_k).reset_index(drop=True)


def predict_all_patients(
    sample_ids: list[str],
    model_path: Path = RESULTS / "lightgbm_drug_model.joblib",
    fp_path: Path = DATA_CACHE / "drug_fingerprints.parquet",
    output_dir: Path = RESULTS,
    top_k: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Run drug predictions for multiple TCGA-BRCA patients.

    Returns dict: sample_id → ranked drug DataFrame
    """
    # Load model
    model = joblib.load(model_path)
    logger.info(f"Loaded model with {model.n_features_} features")

    # Load drug fingerprints
    drug_fps = pd.read_parquet(fp_path)
    logger.info(f"Loaded {len(drug_fps)} drug fingerprints")

    # Load TCGA data
    expression = load_tcga_expression()
    cohort = build_patient_cohort()
    gene_df = load_landmark_genes()
    landmark_genes = gene_df["gene_symbol"].tolist()

    results = {}
    for sid in sample_ids:
        if sid not in expression.index:
            logger.warning(f"Sample {sid} not found in expression data. Skipping.")
            continue
        try:
            rankings = predict_drugs_for_patient(
                sample_id=sid,
                model=model,
                drug_fingerprints=drug_fps,
                expression=expression,
                cohort=cohort,
                landmark_genes=landmark_genes,
                top_k=top_k,
            )
            results[sid] = rankings
            logger.info(
                f"  {sid}: top drug = {rankings.iloc[0]['drug_name']} "
                f"({rankings.iloc[0]['predicted_inhibition']:.1f}% @ "
                f"{rankings.iloc[0]['best_dose_um']} µM)"
            )
        except Exception as e:
            logger.error(f"Failed for {sid}: {e}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    for sid, df in results.items():
        safe_id = sid.replace("/", "_")
        df.to_csv(output_dir / f"drug_rankings_{safe_id}.csv", index=False)

    return results


def predict_personalized_drugs_for_patient(
    sample_id: str,
    ranker: PersonalizedDrugRanker,
    expression: Optional[pd.DataFrame] = None,
    cohort: Optional[pd.DataFrame] = None,
    landmark_genes: Optional[list[str]] = None,
    top_k: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the personalized composite ranker for a single TCGA-BRCA patient.

    Returns:
        ranking_df, patient_summary
    """
    if expression is None:
        expression = load_tcga_expression()
    if cohort is None:
        cohort = build_patient_cohort()

    rankings, patient_summary = ranker.rank_patient(
        sample_id=sample_id,
        expression=expression,
        cohort=cohort,
        landmark_genes=landmark_genes,
        top_k=top_k,
    )
    return rankings, patient_summary


def predict_all_patients_personalized(
    sample_ids: list[str],
    ranker: Optional[PersonalizedDrugRanker] = None,
    output_dir: Path = RESULTS,
    top_k: int = 20,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """
    Run personalized drug rankings for multiple TCGA-BRCA patients.

    Returns:
        dict(sample_id -> ranking_df), list(patient_summary)
    """
    if ranker is None:
        ranker = PersonalizedDrugRanker.from_project_artifacts()

    expression = load_tcga_expression()
    cohort = build_patient_cohort()
    landmark_genes = load_landmark_genes()["gene_symbol"].tolist()

    results = {}
    summaries = []
    for sid in sample_ids:
        if sid not in expression.index:
            logger.warning(f"Sample {sid} not found in expression data. Skipping.")
            continue
        try:
            rankings, patient_summary = predict_personalized_drugs_for_patient(
                sample_id=sid,
                ranker=ranker,
                expression=expression,
                cohort=cohort,
                landmark_genes=landmark_genes,
                top_k=top_k,
            )
            results[sid] = rankings
            summaries.append(patient_summary)
            if not rankings.empty:
                logger.info(
                    f"  {sid}: top drug = {rankings.iloc[0]['drug_name']} "
                    f"(final_score={rankings.iloc[0]['final_score']:.3f})"
                )
        except Exception as exc:
            logger.error(f"Failed personalized ranking for {sid}: {exc}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for sid, df in results.items():
        safe_id = sid.replace("/", "_")
        df.to_csv(output_dir / f"drug_rankings_{safe_id}.csv", index=False)

    return results, summaries


def compute_reversal_score(
    patient_signature: pd.Series,
    drug_signature: pd.Series,
) -> float:
    """
    Classical connectivity-map reversal score.

    Anti-correlation between patient disease signature and drug
    perturbation signature. Higher (more negative correlation)
    means the drug better "reverses" the disease state.

    Returns score in [-1, 1] where -1 = perfect reversal.
    """
    # Align genes
    common = patient_signature.index.intersection(drug_signature.index)
    if len(common) < 10:
        return 0.0

    p = patient_signature[common].values
    d = drug_signature[common].values

    # Remove NaN/inf
    valid = np.isfinite(p) & np.isfinite(d)
    if valid.sum() < 10:
        return 0.0

    # Pearson correlation (we want anti-correlation → negate)
    corr = np.corrcoef(p[valid], d[valid])[0, 1]
    return -corr  # Higher = better reversal


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Patient prediction module loaded.")
