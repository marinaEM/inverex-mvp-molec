"""
INVEREX Dual-Mode Production Inference Pipeline.

Mode A — Batch (>=20 patients from same platform):
    ComBat correction → LightGBM → SHAP → Conformal → Report
    Expected AUC: ~0.65 (if batch-mode validation confirms)

Mode B — Single patient:
    Quantile normalization → LightGBM → SHAP → Conformal → Report
    Expected AUC: ~0.60

Auto-detects mode from input size, or caller can force a mode.

Usage:
    pipeline = InverexPipeline("models/production/")

    # Auto-detect: >=20 patients → batch, otherwise single
    reports = pipeline.predict(expression_df)

    # Force single-patient mode
    reports = pipeline.predict(expression_df, force_mode="single")

    # Generate readable reports
    for r in reports:
        pipeline.generate_report(r)
"""

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500


def _clean_matrix(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def _compute_ssgsea(expression_df):
    """Compute ssGSEA Hallmark pathway scores."""
    import gseapy as gp

    expr_clean = _clean_matrix(expression_df)
    n_samples = expr_clean.shape[0]

    if n_samples <= CHUNK_SIZE:
        result = gp.ssgsea(
            data=expr_clean.T, gene_sets="MSigDB_Hallmark_2020",
            outdir=None, min_size=5, no_plot=True, verbose=False,
        )
        scores = result.res2d.pivot(index="Name", columns="Term", values="NES")
        scores.index.name = None
        scores.columns.name = None
    else:
        chunks = []
        for start in range(0, n_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_samples)
            chunk = expr_clean.iloc[start:end]
            result = gp.ssgsea(
                data=chunk.T, gene_sets="MSigDB_Hallmark_2020",
                outdir=None, min_size=5, no_plot=True, verbose=False,
            )
            cs = result.res2d.pivot(index="Name", columns="Term", values="NES")
            cs.index.name = None
            cs.columns.name = None
            chunks.append(cs)
        scores = pd.concat(chunks, axis=0)

    scores = scores.loc[expression_df.index].astype(float)
    scores.columns = [f"ssgsea_{c}" for c in scores.columns]
    return _clean_matrix(scores)


def _run_combat(expression_df, batch_labels):
    """Run neuroCombat using expression only (no labels)."""
    from neuroCombat import neuroCombat

    covars = pd.DataFrame({"batch": batch_labels.values}, index=expression_df.index)
    combat_result = neuroCombat(
        dat=expression_df.values.T,
        covars=covars,
        batch_col="batch",
    )
    corrected = pd.DataFrame(
        combat_result["data"].T,
        columns=expression_df.columns,
        index=expression_df.index,
    )
    return _clean_matrix(corrected)


class InverexPipeline:
    """
    Dual-mode production inference pipeline.
    All model parameters are FROZEN from training.
    """

    BATCH_THRESHOLD = 20

    def __init__(self, model_dir: str = "models/production/"):
        model_dir = Path(model_dir)

        # Load model bundles
        batch_path = model_dir / "batch_model_bundle.joblib"
        single_path = model_dir / "single_patient_model_bundle.joblib"

        self.has_batch = batch_path.exists()
        self.has_single = single_path.exists()

        if self.has_batch:
            self.batch_bundle = joblib.load(batch_path)
            self.batch_explainer = shap.TreeExplainer(self.batch_bundle["model"])
            logger.info("Loaded batch-mode model")

        if self.has_single:
            self.single_bundle = joblib.load(single_path)
            self.single_explainer = shap.TreeExplainer(self.single_bundle["model"])
            logger.info("Loaded single-patient model")

        if not self.has_batch and not self.has_single:
            raise FileNotFoundError(
                f"No model bundles found in {model_dir}. "
                "Need batch_model_bundle.joblib or single_patient_model_bundle.joblib"
            )

    def predict(
        self,
        expression_data: pd.DataFrame,
        sample_ids: list[str] = None,
        force_mode: str = None,
        platform_id: str = "unknown",
    ) -> list[dict]:
        """
        Run inference for one or more patients.

        Args:
            expression_data: DataFrame (samples x genes), raw expression.
            sample_ids: patient identifiers. Default: patient_0, patient_1, ...
            force_mode: "batch" or "single" to override auto-detection.
            platform_id: platform identifier for ComBat batch label.

        Returns:
            List of patient report dicts.
        """
        n_patients = len(expression_data)

        if force_mode:
            mode = force_mode
        elif n_patients >= self.BATCH_THRESHOLD and self.has_batch:
            mode = "batch"
        else:
            mode = "single"

        if mode == "batch" and not self.has_batch:
            logger.warning("Batch mode requested but no batch model loaded. Falling back to single.")
            mode = "single"
        if mode == "single" and not self.has_single:
            logger.warning("Single mode requested but no single model loaded. Falling back to batch.")
            mode = "batch"

        logger.info(f"Inference mode: {mode} ({n_patients} patients)")

        if mode == "batch":
            return self._predict_batch(expression_data, sample_ids, platform_id)
        else:
            return self._predict_single(expression_data, sample_ids)

    def _predict_batch(self, expression_data, sample_ids, platform_id):
        """Batch mode: ComBat-correct incoming cohort against training stats."""
        bundle = self.batch_bundle
        explainer = self.batch_explainer
        gene_list = bundle["gene_list"]
        feature_names = bundle["feature_names"]

        # Align to gene list
        expr = expression_data.reindex(columns=gene_list, fill_value=0.0)

        # Per-dataset z-score the incoming batch
        means = expr.mean(axis=0)
        stds = expr.std(axis=0).replace(0, 1)
        expr_z = _clean_matrix((expr - means) / stds)

        # ComBat: combine training data + incoming batch
        training_expr = bundle["combat_params"]["training_expr"]
        training_batches = bundle["combat_params"]["training_batches"]

        combined_expr = pd.concat([training_expr, expr_z], axis=0)
        incoming_batch = pd.Series(
            [f"incoming_{platform_id}"] * len(expr_z),
            index=expr_z.index,
        )
        combined_batches = pd.concat([training_batches, incoming_batch])

        try:
            corrected = _run_combat(combined_expr, combined_batches)
            corrected_incoming = corrected.loc[expr_z.index]
        except Exception as e:
            logger.warning(f"ComBat failed: {e}. Using z-scored data.")
            corrected_incoming = expr_z

        return self._score_patients(
            corrected_incoming, sample_ids, bundle, explainer, mode="batch"
        )

    def _predict_single(self, expression_data, sample_ids):
        """Single-patient mode: quantile normalize within-sample."""
        bundle = self.single_bundle
        explainer = self.single_explainer
        gene_list = bundle["gene_list"]

        expr = expression_data.reindex(columns=gene_list, fill_value=0.0)

        # Per-sample z-score first (matches training preprocessing)
        means = expr.mean(axis=0)
        stds = expr.std(axis=0).replace(0, 1)
        expr_z = _clean_matrix((expr - means) / stds)

        # Quantile normalization
        normalizer = bundle["normalizer"]
        corrected = normalizer.transform(expr_z)
        corrected = _clean_matrix(corrected)

        return self._score_patients(
            corrected, sample_ids, bundle, explainer, mode="single"
        )

    def _score_patients(self, corrected_expr, sample_ids, bundle, explainer, mode):
        """Build features, predict, explain for each patient."""
        model = bundle["model"]
        feature_names = bundle["feature_names"]
        conformal = bundle.get("conformal")

        if sample_ids is None:
            sample_ids = [f"patient_{i}" for i in range(len(corrected_expr))]

        # Compute ssGSEA
        n_gene_features = len(bundle["gene_list"])
        n_ssgsea_expected = len(feature_names) - n_gene_features
        try:
            ssgsea = _compute_ssgsea(corrected_expr)
            # Align ssGSEA columns to training feature order
            ssgsea_cols = [f for f in feature_names if f.startswith("ssgsea_")]
            for col in ssgsea_cols:
                if col not in ssgsea.columns:
                    ssgsea[col] = 0.0
            ssgsea = ssgsea.reindex(columns=ssgsea_cols, fill_value=0.0)
            X_full = pd.concat([corrected_expr, ssgsea], axis=1)
        except Exception as e:
            logger.warning(f"ssGSEA failed: {e}. Using genes only.")
            X_full = corrected_expr.copy()
            # Pad with zeros for missing ssGSEA features
            for col in feature_names:
                if col not in X_full.columns:
                    X_full[col] = 0.0

        # Align to training feature order
        X_full = X_full.reindex(columns=feature_names, fill_value=0.0)
        X_clean = _clean_matrix(X_full).values

        # Predict all patients at once
        pred_probs = model.predict_proba(X_clean)[:, 1]

        # SHAP for all patients
        sv = explainer.shap_values(X_clean)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]

        # Build reports
        reports = []
        for i in range(len(corrected_expr)):
            patient_id = sample_ids[i] if i < len(sample_ids) else f"patient_{i}"
            pred = float(pred_probs[i])
            shap_vals = sv[i]

            feat_df = pd.DataFrame({
                "feature": feature_names,
                "shap_value": shap_vals,
            }).sort_values("shap_value", key=abs, ascending=False)

            top_pos = (
                feat_df[feat_df["shap_value"] > 0]
                .head(5)[["feature", "shap_value"]]
                .to_dict("records")
            )
            top_neg = (
                feat_df[feat_df["shap_value"] < 0]
                .head(3)[["feature", "shap_value"]]
                .to_dict("records")
            )

            # Conformal confidence
            confidence = "N/A"
            if conformal is not None:
                try:
                    conf_result = conformal.predict_with_confidence(
                        X_clean[i : i + 1]
                    )
                    is_confident = conf_result.get("is_confident", False)
                    if isinstance(is_confident, np.ndarray):
                        is_confident = is_confident[0]
                    confidence = "High" if is_confident and pred > 0.5 else "Moderate"
                except Exception:
                    pass

            reports.append({
                "patient_id": patient_id,
                "inference_mode": mode,
                "predicted_response_prob": round(pred, 4),
                "confidence_tier": confidence,
                "top_positive_features": top_pos,
                "top_negative_features": top_neg,
                "timestamp": datetime.now().isoformat(),
            })

        return reports

    def generate_report(
        self, patient_report: dict, output_dir: str = "results/patient_reports/"
    ) -> tuple[str, str]:
        """Generate human-readable + JSON reports for a patient."""
        os.makedirs(output_dir, exist_ok=True)
        patient_id = patient_report["patient_id"]
        mode = patient_report["inference_mode"]
        pred = patient_report["predicted_response_prob"]
        conf = patient_report["confidence_tier"]

        mode_desc = (
            "ComBat batch correction" if mode == "batch"
            else "Quantile normalization"
        )

        lines = [
            "INVEREX Patient Report",
            "=" * 60,
            f"Patient:    {patient_id}",
            f"Mode:       {mode} ({mode_desc})",
            f"Response:   {pred*100:.1f}% predicted probability",
            f"Confidence: {conf}",
            f"Generated:  {patient_report['timestamp']}",
            "=" * 60,
            "",
        ]

        # Top features driving this prediction
        if patient_report["top_positive_features"]:
            lines.append("Features pushing TOWARD response:")
            for f in patient_report["top_positive_features"][:5]:
                lines.append(f"  {f['feature']:30s}  SHAP +{f['shap_value']:.4f}")
            lines.append("")

        if patient_report["top_negative_features"]:
            lines.append("Features pushing AGAINST response:")
            for f in patient_report["top_negative_features"][:3]:
                lines.append(f"  {f['feature']:30s}  SHAP {f['shap_value']:.4f}")
            lines.append("")

        lines.extend([
            "-" * 60,
            "This report is for research purposes only.",
            "Not intended for clinical decision-making.",
            "-" * 60,
            "Powered by Inverex",
        ])

        report_text = "\n".join(lines)

        safe_id = patient_id.replace("/", "_").replace(" ", "_")
        txt_path = f"{output_dir}/{safe_id}_report.txt"
        json_path = f"{output_dir}/{safe_id}_report.json"

        with open(txt_path, "w") as fh:
            fh.write(report_text)
        with open(json_path, "w") as fh:
            json.dump(patient_report, fh, indent=2, default=str)

        return txt_path, json_path

    @staticmethod
    def save_bundle(
        model, gene_list, feature_names, mode,
        normalizer=None, combat_params=None, conformal=None,
        output_dir="models/production/",
    ):
        """Save a model bundle for production."""
        os.makedirs(output_dir, exist_ok=True)
        filename = (
            "batch_model_bundle.joblib" if mode == "batch"
            else "single_patient_model_bundle.joblib"
        )
        bundle = {
            "model": model,
            "gene_list": gene_list,
            "feature_names": feature_names,
            "mode": mode,
        }
        if normalizer is not None:
            bundle["normalizer"] = normalizer
        if combat_params is not None:
            bundle["combat_params"] = combat_params
        if conformal is not None:
            bundle["conformal"] = conformal

        joblib.dump(bundle, f"{output_dir}/{filename}")
        logger.info(f"Saved {mode} model bundle to {output_dir}/{filename}")
