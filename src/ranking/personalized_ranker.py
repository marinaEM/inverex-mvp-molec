"""
Personalized ranking layer for breast-cancer drug prioritization.

This module keeps the existing LightGBM model as an auxiliary ranking prior,
but makes the final score explicit and auditable:

- RNA reversal against LINCS signatures
- mutation / pathway relevance
- subtype / tissue context
- clinical actionability
- optional ML prior
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_CACHE, DATA_PROCESSED, RESULTS
from src.data_ingestion.lincs import load_landmark_genes
from src.data_ingestion.tcga import build_patient_cohort, compute_patient_signature, load_tcga_expression
from src.ranking.drug_metadata import DEFAULT_METADATA_PATH, ensure_drug_metadata, load_drug_metadata

DEFAULT_CONFIG_PATH = DATA_PROCESSED / "personalized_ranking_config.json"
EVAL_DOSES = [0.04, 0.12, 0.37, 1.11, 3.33, 10.0]

L1000_META_COLUMNS = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
CELL_LINE_CONTEXT = {
    "BT474": "Her2",
    "HCC1937": "Basal",
    "HS578T": "Basal",
    "MCF7": "Luminal",
    "MDAMB231": "Basal",
    "MDAMB468": "Basal",
    "SKBR3": "Her2",
    "T47D": "Luminal",
    "ZR751": "Luminal",
}

PATHWAY_GENE_SETS = {
    "her2_egfr": ["ERBB2", "EGFR", "ERBB3", "FGFR1", "GRB7"],
    "pi3k_mtor": ["PIK3CA", "AKT1", "MTOR", "PTEN", "RHEB", "EIF4EBP1", "RPS6KB1"],
    "proliferation": ["MKI67", "TOP2A", "AURKA", "AURKB", "PLK1", "CCNB1", "CDK1", "CDC20", "BIRC5", "UBE2C", "TYMS", "TUBB"],
    "er_luminal": ["ESR1", "PGR", "GATA3", "FOXA1", "CCND1", "BCL2"],
    "dna_damage": ["BRCA1", "BRCA2", "PARP1", "PARP2", "RAD51", "ATM", "ATR", "CHEK1", "CHEK2"],
    "immune_inflammation": ["CCL2", "CXCL8", "IL1B", "IL6", "NFKB1", "RELA", "STAT3"],
}

DEFAULT_RANKING_CONFIG = {
    "weights": {
        "rna_score": 0.35,
        "mutation_pathway_score": 0.22,
        "subtype_context_score": 0.18,
        "clinical_actionability_score": 0.17,
        "ml_score": 0.08,
    },
    "rna_top_k_signatures": 3,
    "rna_top_k_genes": 5,
    "component_clip": 1.0,
}

try:  # pragma: no cover - environment-dependent optional dependency
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent optional dependency
    lgb = Any


@dataclass
class PatientProfile:
    sample_id: str
    subtype: str
    subtype_group: str
    er_status: str
    pr_status: str
    her2_status: str
    signature: pd.Series
    pathway_scores: dict[str, float]
    top_up_genes: list[str]
    top_down_genes: list[str]
    mutation_flags: dict[str, int]
    er_positive: bool
    her2_active: bool
    pi3k_axis_altered: bool
    brca_loss: bool
    luminal_marker_altered: bool
    tp53_mutated: bool
    high_proliferation: bool


def _clip(value: float, limit: float = 1.0) -> float:
    return float(np.clip(value, -limit, limit))


def _truthy_status(value: Any, positive_terms: set[str]) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in positive_terms


def _normalize_to_unit_interval(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    ranks = series.rank(method="average", pct=True)
    return (2 * ranks - 1).astype(float)


def _infer_subtype_group(subtype: str) -> str:
    subtype_lower = str(subtype).strip().lower()
    if subtype_lower.startswith("lum"):
        return "Luminal"
    if subtype_lower.startswith("her2"):
        return "Her2"
    if "basal" in subtype_lower:
        return "Basal"
    if subtype_lower == "normal":
        return "Normal"
    return "Unknown"


def _format_gene_list(genes: list[str]) -> str:
    return ", ".join(genes) if genes else "None"


def compute_reversal_score(patient_signature: pd.Series, drug_signature: pd.Series) -> float:
    """Connectivity-style anti-correlation score. Higher values mean stronger reversal."""
    common = patient_signature.index.intersection(drug_signature.index)
    if len(common) < 10:
        return 0.0

    try:
        patient_values = patient_signature[common].astype(float).values
        drug_values = drug_signature[common].astype(float).values
    except (TypeError, ValueError):
        return 0.0
    valid = np.isfinite(patient_values) & np.isfinite(drug_values)
    if valid.sum() < 10:
        return 0.0

    corr = np.corrcoef(patient_values[valid], drug_values[valid])[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(-corr)


def _compute_pathway_scores(signature: pd.Series) -> dict[str, float]:
    scores: dict[str, float] = {}
    for pathway, genes in PATHWAY_GENE_SETS.items():
        available = [gene for gene in genes if gene in signature.index]
        if not available:
            scores[pathway] = 0.0
            continue
        scores[pathway] = float(signature[available].mean())
    return scores


def load_personalized_ranking_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load ranking config from JSON, writing defaults if the file does not exist."""
    if config_path.exists():
        with open(config_path) as handle:
            return json.load(handle)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as handle:
        json.dump(DEFAULT_RANKING_CONFIG, handle, indent=2)
    return json.loads(json.dumps(DEFAULT_RANKING_CONFIG))


class PersonalizedDrugRanker:
    """Composite personalized ranker for TCGA-BRCA patients."""

    def __init__(
        self,
        drug_fingerprints: pd.DataFrame,
        lincs_signatures: pd.DataFrame,
        model: Any | None = None,
        pancancer_model: Any | None = None,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.drug_fingerprints = drug_fingerprints.copy()
        self.candidate_drugs = sorted(self.drug_fingerprints["compound_name"].dropna().unique().tolist())
        self.metadata_path = metadata_path
        self.config = load_personalized_ranking_config(config_path)
        self.weights = self.config["weights"]
        self.model = model
        self.pancancer_model = pancancer_model

        ensure_drug_metadata(self.candidate_drugs, metadata_path)
        metadata = load_drug_metadata(self.candidate_drugs, metadata_path)
        self.drug_metadata = metadata.set_index("drug_name")

        subset = lincs_signatures[lincs_signatures["pert_iname"].isin(self.candidate_drugs)].copy()
        self.lincs_signatures = subset
        self.lincs_gene_cols = [column for column in subset.columns if column not in L1000_META_COLUMNS]

    @classmethod
    def from_project_artifacts(
        cls,
        model_path: Path = RESULTS / "lightgbm_drug_model.joblib",
        pancancer_model_path: Path = RESULTS / "pan_cancer_patient_model.joblib",
        fp_path: Path = DATA_CACHE / "drug_fingerprints.parquet",
        lincs_path: Path = DATA_CACHE / "breast_l1000_signatures.parquet",
        metadata_path: Path = DEFAULT_METADATA_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> "PersonalizedDrugRanker":
        """Instantiate the ranker from cached project artifacts."""
        drug_fingerprints = pd.read_parquet(fp_path)
        lincs_signatures = pd.read_parquet(lincs_path)
        try:
            model = joblib.load(model_path) if model_path.exists() else None
        except Exception:
            model = None
        try:
            pancancer_model = joblib.load(pancancer_model_path) if pancancer_model_path.exists() else None
        except Exception:
            pancancer_model = None
        return cls(
            drug_fingerprints=drug_fingerprints,
            lincs_signatures=lincs_signatures,
            model=model,
            pancancer_model=pancancer_model,
            metadata_path=metadata_path,
            config_path=config_path,
        )

    def build_patient_profile(
        self,
        sample_id: str,
        expression: pd.DataFrame,
        cohort: pd.DataFrame,
        landmark_genes: list[str] | None = None,
    ) -> PatientProfile:
        """Build a patient summary used by the composite ranker."""
        if landmark_genes is None:
            landmark_genes = load_landmark_genes()["gene_symbol"].tolist()

        signature = compute_patient_signature(
            sample_id=sample_id,
            expression=expression,
            cohort=cohort,
            landmark_genes=landmark_genes,
            method="subtype_centroid",
        )
        signature = signature.sort_index()
        patient_row = cohort.loc[sample_id]

        subtype = patient_row.get("pam50_subtype", "Unknown")
        if pd.isna(subtype) or str(subtype).strip() == "":
            if _truthy_status(patient_row.get("her2_status"), {"positive"}):
                subtype = "Her2"
            elif _truthy_status(patient_row.get("er_status"), {"positive"}):
                subtype = "Luminal"
            else:
                subtype = "Basal"

        er_positive = _truthy_status(patient_row.get("er_status"), {"positive"})
        her2_active = any(
            [
                _truthy_status(patient_row.get("her2_status"), {"positive"}),
                int(patient_row.get("ERBB2_amp", 0) or 0) == 1,
                int(patient_row.get("mut_ERBB2", 0) or 0) == 1,
                str(subtype).strip().lower() == "her2",
            ]
        )

        mutation_flags = {
            "ERBB2_amp": int(patient_row.get("ERBB2_amp", 0) or 0),
            "mut_TP53": int(patient_row.get("mut_TP53", 0) or 0),
            "mut_PIK3CA": int(patient_row.get("mut_PIK3CA", 0) or 0),
            "mut_ERBB2": int(patient_row.get("mut_ERBB2", 0) or 0),
            "mut_ESR1": int(patient_row.get("mut_ESR1", 0) or 0),
            "mut_GATA3": int(patient_row.get("mut_GATA3", 0) or 0),
            "mut_MAP3K1": int(patient_row.get("mut_MAP3K1", 0) or 0),
            "mut_AKT1": int(patient_row.get("mut_AKT1", 0) or 0),
            "mut_PTEN": int(patient_row.get("mut_PTEN", 0) or 0),
            "mut_BRCA1": int(patient_row.get("mut_BRCA1", 0) or 0),
            "mut_BRCA2": int(patient_row.get("mut_BRCA2", 0) or 0),
        }

        pathway_scores = _compute_pathway_scores(signature)
        top_up_genes = signature.sort_values(ascending=False).head(5).index.tolist()
        top_down_genes = signature.sort_values().head(5).index.tolist()

        return PatientProfile(
            sample_id=sample_id,
            subtype=str(subtype),
            subtype_group=_infer_subtype_group(str(subtype)),
            er_status=str(patient_row.get("er_status", "Unknown")),
            pr_status=str(patient_row.get("pr_status", "Unknown")),
            her2_status=str(patient_row.get("her2_status", "Unknown")),
            signature=signature,
            pathway_scores=pathway_scores,
            top_up_genes=top_up_genes,
            top_down_genes=top_down_genes,
            mutation_flags=mutation_flags,
            er_positive=er_positive,
            her2_active=her2_active,
            pi3k_axis_altered=any(
                [
                    mutation_flags["mut_PIK3CA"] == 1,
                    mutation_flags["mut_AKT1"] == 1,
                    mutation_flags["mut_PTEN"] == 1,
                ]
            ),
            brca_loss=any(
                [
                    mutation_flags["mut_BRCA1"] == 1,
                    mutation_flags["mut_BRCA2"] == 1,
                ]
            ),
            luminal_marker_altered=any(
                [
                    mutation_flags["mut_ESR1"] == 1,
                    mutation_flags["mut_GATA3"] == 1,
                    mutation_flags["mut_MAP3K1"] == 1,
                ]
            ),
            tp53_mutated=mutation_flags["mut_TP53"] == 1,
            high_proliferation=pathway_scores["proliferation"] >= 0.75,
        )

    def summarize_patient(self, profile: PatientProfile) -> dict[str, Any]:
        """Return a compact patient molecular summary."""
        active_mutations = [
            flag.replace("mut_", "")
            for flag, value in profile.mutation_flags.items()
            if value == 1 and flag.startswith("mut_")
        ]
        if profile.mutation_flags.get("ERBB2_amp", 0) == 1:
            active_mutations.append("ERBB2_amp")

        return {
            "patient_id": profile.sample_id,
            "pam50_subtype": profile.subtype,
            "er_status": profile.er_status,
            "pr_status": profile.pr_status,
            "her2_status": profile.her2_status,
            "active_mutations": active_mutations,
            "top_up_genes": profile.top_up_genes,
            "top_down_genes": profile.top_down_genes,
            "pathway_scores": profile.pathway_scores,
        }

    def _compute_treatability_score(self, profile: PatientProfile) -> dict[str, Any]:
        """Compute a patient-level treatability score using the pan-cancer model.

        This is NOT drug-specific — it estimates how likely this patient is
        to respond to treatment in general, based on expression patterns
        learned from 3,730 patients across 11 cancer types.

        Returns dict with probability, label, and note.
        """
        if self.pancancer_model is None:
            return {"probability": None, "label": "unavailable", "note": "Pan-cancer model not loaded"}

        try:
            model_features = self.pancancer_model.feature_name_
            patient_values = np.array(
                [float(profile.signature.get(gene, 0.0)) for gene in model_features],
                dtype=np.float32,
            ).reshape(1, -1)
            prob = float(self.pancancer_model.predict_proba(patient_values)[0, 1])

            if prob >= 0.65:
                label = "high"
            elif prob >= 0.45:
                label = "moderate"
            else:
                label = "low"

            return {
                "probability": round(prob, 3),
                "label": label,
                "note": f"Pan-cancer response probability {prob:.1%} ({label}) — "
                        f"based on expression patterns from 3,730 patients across 11 cancer types.",
            }
        except Exception as exc:
            return {"probability": None, "label": "error", "note": f"Pan-cancer score failed: {exc}"}

    def _predict_ml_component(self, profile: PatientProfile) -> pd.DataFrame:
        """Score the optional LightGBM component for each drug.

        If the serialized model is unavailable, attempts to load a cached
        legacy ranking CSV (from the original LightGBM-only pipeline) and
        normalizes its predicted_inhibition into a [−1, 1] score.  This is
        an explicit fallback — the ML weight (default 8 %) keeps its
        influence bounded.
        """
        import logging as _log
        _logger = _log.getLogger(__name__)

        if self.model is None:
            # Try legacy CSV fallback (only files without final_score are legacy)
            fallback_path = RESULTS / f"drug_rankings_{profile.sample_id.replace('/', '_')}.csv"
            if fallback_path.exists():
                fallback = pd.read_csv(fallback_path)
                if "predicted_inhibition" in fallback.columns and "final_score" not in fallback.columns:
                    fallback = fallback[fallback["drug_name"].isin(self.candidate_drugs)].copy()
                    if not fallback.empty:
                        _logger.info(f"ML component: using legacy CSV fallback for {profile.sample_id}")
                        fallback["ml_score"] = _normalize_to_unit_interval(fallback["predicted_inhibition"]).values
                        if "best_dose_um" not in fallback.columns and "dose_um" in fallback.columns:
                            fallback = fallback.rename(columns={"dose_um": "best_dose_um"})
                        return fallback[["drug_name", "best_dose_um", "predicted_inhibition", "ml_score"]]

            _logger.warning(f"ML component: no model and no legacy CSV for {profile.sample_id}; ML score = 0")
            return pd.DataFrame(columns=["drug_name", "best_dose_um", "predicted_inhibition", "ml_score"])

        model_features = self.model.feature_name_
        gene_features = [feature for feature in model_features if not feature.startswith("ecfp_") and feature != "log_dose_um"]
        ecfp_features = [feature for feature in model_features if feature.startswith("ecfp_")]

        patient_gene_array = np.array(
            [float(profile.signature.get(gene, 0.0)) for gene in gene_features],
            dtype=np.float32,
        )

        ecfp_cols = [column for column in self.drug_fingerprints.columns if column.startswith("ecfp_")]
        fp_lookup = self.drug_fingerprints.set_index("compound_name")[ecfp_cols]

        rows = []
        meta_rows = []
        for drug_name in self.candidate_drugs:
            fp_values = fp_lookup.loc[drug_name].values.astype(np.int8)
            ecfp_aligned = np.zeros(len(ecfp_features), dtype=np.int8)
            for index, feature in enumerate(ecfp_features):
                if feature in ecfp_cols:
                    ecfp_aligned[index] = fp_values[ecfp_cols.index(feature)]

            for dose in EVAL_DOSES:
                feature_row = np.concatenate([patient_gene_array, ecfp_aligned, [np.log1p(dose)]])
                rows.append(feature_row)
                meta_rows.append({"drug_name": drug_name, "dose_um": dose})

        prediction_matrix = pd.DataFrame(rows, columns=model_features)
        predicted = self.model.predict(prediction_matrix).clip(0, 100)
        meta = pd.DataFrame(meta_rows)
        meta["predicted_inhibition"] = predicted

        best = (
            meta.sort_values("predicted_inhibition", ascending=False)
            .groupby("drug_name")
            .first()
            .reset_index()
            .rename(columns={"dose_um": "best_dose_um"})
        )
        best["ml_score"] = _normalize_to_unit_interval(best["predicted_inhibition"]).values
        return best

    def _score_rna_component(self, profile: PatientProfile, drug_name: str) -> dict[str, Any]:
        """Score transcriptomic reversal using the top LINCS signatures for a drug."""
        subset = self.lincs_signatures[self.lincs_signatures["pert_iname"] == drug_name]
        if subset.empty:
            return {
                "rna_score": 0.0,
                "rna_rationale": "No LINCS breast-cancer signature available for this compound in the cached library.",
                "top_reversed_genes": [],
                "top_reversed_programs": [],
            }

        signature_values = []
        top_gene_count = int(self.config.get("rna_top_k_genes", 5))
        for _, row in subset.iterrows():
            drug_signature = row[self.lincs_gene_cols]
            reversal = compute_reversal_score(profile.signature, drug_signature)

            common = profile.signature.index.intersection(drug_signature.index)
            try:
                patient_common = profile.signature[common].astype(float)
                drug_common = drug_signature[common].astype(float)
            except (TypeError, ValueError):
                patient_common = pd.Series(dtype=float)
                drug_common = pd.Series(dtype=float)
            contribution = (-patient_common * drug_common).sort_values(ascending=False)
            positive = contribution[contribution > 0]
            top_genes = positive.head(top_gene_count).index.tolist()

            pathway_supports = []
            for pathway, genes in PATHWAY_GENE_SETS.items():
                available = [gene for gene in genes if gene in common]
                if not available:
                    continue
                patient_pathway = float(patient_common[available].mean())
                drug_pathway = float(drug_common[available].mean())
                support = -patient_pathway * drug_pathway
                pathway_supports.append((pathway, support))

            signature_values.append(
                {
                    "reversal": float(reversal),
                    "cell_id": row.get("cell_id", ""),
                    "dose_um": float(row.get("dose_um", np.nan)),
                    "top_genes": top_genes,
                    "top_programs": [pathway for pathway, support in sorted(pathway_supports, key=lambda item: item[1], reverse=True)[:2] if support > 0],
                }
            )

        top_k = max(1, int(self.config.get("rna_top_k_signatures", 3)))
        top_matches = sorted(signature_values, key=lambda item: item["reversal"], reverse=True)[:top_k]
        rna_score = _clip(float(np.mean([match["reversal"] for match in top_matches])), self.config.get("component_clip", 1.0))
        best_match = top_matches[0]

        rationale = (
            f"Top LINCS reversal came from {best_match['cell_id']} signatures "
            f"(best reversal {best_match['reversal']:.2f}). "
            f"Reversed genes: {_format_gene_list(best_match['top_genes']) or 'None'}."
        )
        if best_match["top_programs"]:
            rationale += f" Reversed programs: {_format_gene_list(best_match['top_programs'])}."

        return {
            "rna_score": rna_score,
            "rna_rationale": rationale,
            "top_reversed_genes": best_match["top_genes"],
            "top_reversed_programs": best_match["top_programs"],
        }

    def _score_mutation_pathway_component(self, profile: PatientProfile, metadata: pd.Series) -> tuple[float, list[str]]:
        """Score mutation / pathway relevance based on a small curated mapping."""
        score = 0.0
        reasons: list[str] = []

        if profile.her2_active and metadata["her2_relevance"] > 0:
            bonus = 0.75 * float(metadata["her2_relevance"]) / 3.0
            score += bonus
            reasons.append("ERBB2/HER2-active disease aligns with HER2-pathway relevance.")
        elif not profile.her2_active and metadata["her2_relevance"] >= 2:
            score -= 0.1
            reasons.append("HER2-directed rationale is weaker without HER2 activation.")

        if profile.pi3k_axis_altered and metadata["pi3k_mtor_relevance"] > 0:
            bonus = 0.6 * float(metadata["pi3k_mtor_relevance"]) / 3.0
            score += bonus
            reasons.append("PIK3CA/AKT/PTEN alteration supports PI3K-AKT-mTOR targeting.")

        if profile.brca_loss and metadata["parp_relevance"] > 0:
            bonus = 0.7 * float(metadata["parp_relevance"]) / 3.0
            score += bonus
            reasons.append("BRCA1/2 alteration supports DNA-damage/PARP sensitivity logic.")

        if profile.luminal_marker_altered and (metadata["endocrine_relevance"] > 0 or metadata["cdk46_relevance"] > 0):
            bonus = 0.35 * float(max(metadata["endocrine_relevance"], metadata["cdk46_relevance"])) / 3.0
            score += bonus
            reasons.append("Luminal-pathway alteration supports endocrine / CDK4/6 relevance.")

        if profile.tp53_mutated and profile.high_proliferation and max(metadata["microtubule_relevance"], metadata["dna_damage_relevance"]) > 0:
            score += 0.12
            reasons.append("TP53 mutation is treated only as weak support for a proliferative cytotoxic phenotype.")

        return _clip(score, self.config.get("component_clip", 1.0)), reasons

    def _score_subtype_context_component(self, profile: PatientProfile, metadata: pd.Series) -> tuple[float, list[str]]:
        """Score subtype and tissue-context plausibility."""
        score = 0.0
        reasons: list[str] = []

        if profile.subtype_group == "Her2":
            if metadata["her2_relevance"] > 0:
                score += 0.7 * float(metadata["her2_relevance"]) / 3.0
                reasons.append("HER2 subtype favors HER2-pathway-directed agents.")
            if metadata["hsp90_relevance"] > 0:
                score += 0.2 * float(metadata["hsp90_relevance"]) / 3.0
                reasons.append("HER2 subtype gives modest weight to HSP90 client-protein logic.")
            if metadata["microtubule_relevance"] > 0:
                score += 0.15 * float(metadata["microtubule_relevance"]) / 3.0
                reasons.append("Taxane/vinca classes remain clinically plausible in HER2-positive breast cancer.")

        elif profile.subtype_group == "Luminal":
            if metadata["endocrine_relevance"] > 0:
                score += 0.55 * float(metadata["endocrine_relevance"]) / 3.0
                reasons.append("Luminal context favors endocrine therapy.")
            if metadata["cdk46_relevance"] > 0:
                score += 0.45 * float(metadata["cdk46_relevance"]) / 3.0
                reasons.append("Luminal context favors CDK4/6 pathway relevance.")
            if metadata["luminal_relevance"] > 0:
                score += 0.25 * float(metadata["luminal_relevance"]) / 3.0

        elif profile.subtype_group == "Basal":
            if metadata["tnbc_relevance"] > 0:
                score += 0.4 * float(metadata["tnbc_relevance"]) / 3.0
                reasons.append("Basal/TNBC context favors TNBC-relevant agents.")
            if metadata["dna_damage_relevance"] > 0:
                score += 0.25 * float(metadata["dna_damage_relevance"]) / 3.0
                reasons.append("Basal/TNBC context modestly favors DNA-damage classes.")
            if metadata["microtubule_relevance"] > 0:
                score += 0.2 * float(metadata["microtubule_relevance"]) / 3.0

        if profile.er_positive and metadata["endocrine_relevance"] > 0:
            score += 0.25 * float(metadata["endocrine_relevance"]) / 3.0
            reasons.append("ER-positive status supports endocrine relevance.")
        elif not profile.er_positive and metadata["endocrine_relevance"] > 0:
            score -= 0.3 * float(metadata["endocrine_relevance"]) / 3.0
            reasons.append("ER-negative status penalizes endocrine-only logic.")

        return _clip(score, self.config.get("component_clip", 1.0)), reasons

    def _score_clinical_component(self, metadata: pd.Series) -> tuple[float, bool, list[str]]:
        """Score clinical actionability and decide whether the drug should be excluded."""
        score = float(metadata["status_score"])
        reasons = [f"Clinical status: {metadata['clinical_status']}."]
        excluded = bool(int(metadata["tool_compound"]) == 1)

        score += 0.12 * (float(metadata["breast_cancer_relevance"]) - 1.0)
        if int(metadata["standard_of_care_breast"]) == 1:
            score += 0.1
            reasons.append("Standard breast-cancer class.")

        if float(metadata["breast_cancer_relevance"]) == 0 and metadata["clinical_status"] not in {"approved_breast", "approved_oncology"}:
            score -= 0.2
            reasons.append("No strong breast-cancer-specific actionability signal.")

        if int(metadata["trial_mappable"]) == 0:
            score -= 0.1
            reasons.append("Not readily mappable to breast-cancer trial interventions.")

        if excluded:
            reasons.append("Flagged as a research/tool compound and excluded from final recommendations.")
            score = -1.0

        return _clip(score, self.config.get("component_clip", 1.0)), excluded, reasons

    def _determine_evidence_tier(
        self,
        profile: PatientProfile,
        metadata: pd.Series,
        excluded: bool,
    ) -> str:
        """Assign an evidence tier based on ALL patient biomarkers.

        The mapping table connects each patient molecular flag to the
        drug-metadata relevance column that would constitute a
        biomarker-drug match.  Any single match at relevance >= 2
        qualifies the drug for Tier 1.
        """
        if excluded:
            return "Exclude"

        # ── Biomarker → drug-relevance mapping ─────────────────────
        # Each entry: (patient_flag_is_active, metadata_relevance_col)
        biomarker_drug_pairs: list[tuple[bool, str]] = [
            # Receptor / amplification status
            (profile.her2_active,           "her2_relevance"),
            (profile.er_positive,           "endocrine_relevance"),
            (profile.er_positive,           "luminal_relevance"),
            # Pathway alterations
            (profile.pi3k_axis_altered,     "pi3k_mtor_relevance"),
            (profile.brca_loss,             "parp_relevance"),
            (profile.brca_loss,             "dna_damage_relevance"),
            # TP53 → DNA-damage / cell-cycle agents
            (profile.tp53_mutated,          "dna_damage_relevance"),
            # High proliferation → CDK4/6 inhibitors, microtubule agents
            (profile.high_proliferation,    "cdk46_relevance"),
            (profile.high_proliferation,    "microtubule_relevance"),
            # Luminal marker alterations (ESR1/GATA3/MAP3K1 muts)
            (profile.luminal_marker_altered, "endocrine_relevance"),
            # Triple-negative / basal context
            (profile.subtype_group == "Basal", "tnbc_relevance"),
        ]

        biomarker_match = any(
            active and metadata.get(rel_col, 0) >= 2
            for active, rel_col in biomarker_drug_pairs
        )

        if biomarker_match:
            return "Tier 1"
        if int(metadata["standard_of_care_breast"]) == 1:
            return "Tier 1"
        if metadata["clinical_status"] in {"approved_breast", "approved_oncology", "investigational_breast"} and metadata["breast_cancer_relevance"] >= 1:
            return "Tier 2"
        return "Tier 3"

    def rank_patient(
        self,
        sample_id: str,
        expression: pd.DataFrame,
        cohort: pd.DataFrame,
        landmark_genes: list[str] | None = None,
        top_k: int = 30,
        include_excluded: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Rank drugs for one TCGA-BRCA patient and return rankings plus summary."""
        profile = self.build_patient_profile(sample_id, expression, cohort, landmark_genes)
        treatability = self._compute_treatability_score(profile)
        ml_component = self._predict_ml_component(profile)
        ml_lookup = ml_component.set_index("drug_name").to_dict("index") if not ml_component.empty else {}

        rows = []
        for drug_name in self.candidate_drugs:
            if drug_name not in self.drug_metadata.index:
                continue
            metadata = self.drug_metadata.loc[drug_name]
            rna = self._score_rna_component(profile, drug_name)
            mutation_score, mutation_reasons = self._score_mutation_pathway_component(profile, metadata)
            context_score, context_reasons = self._score_subtype_context_component(profile, metadata)
            clinical_score, excluded_flag, clinical_reasons = self._score_clinical_component(metadata)

            ml_entry = ml_lookup.get(drug_name, {})
            ml_score = float(ml_entry.get("ml_score", 0.0))
            predicted_inhibition = float(ml_entry.get("predicted_inhibition", np.nan))
            best_dose_um = float(ml_entry.get("best_dose_um", np.nan))

            final_score = (
                self.weights["rna_score"] * rna["rna_score"]
                + self.weights["mutation_pathway_score"] * mutation_score
                + self.weights["subtype_context_score"] * context_score
                + self.weights["clinical_actionability_score"] * clinical_score
                + self.weights["ml_score"] * ml_score
            )
            if excluded_flag:
                final_score -= 1.0

            evidence_tier = self._determine_evidence_tier(profile, metadata, excluded_flag)

            rationale_short_parts = [
                f"RNA {rna['rna_score']:.2f}",
                f"mutation/pathway {mutation_score:.2f}",
                f"context {context_score:.2f}",
                f"clinical {clinical_score:.2f}",
            ]
            if not np.isnan(predicted_inhibition):
                rationale_short_parts.append(f"ML prior {predicted_inhibition:.1f}%")

            confidence_notes = [
                "LightGBM is used only as an auxiliary ranking prior from breast cell-line inhibition, not as a validated patient response predictor.",
            ]
            if treatability["probability"] is not None:
                confidence_notes.append(treatability["note"])
            if evidence_tier == "Tier 3":
                confidence_notes.append("Evidence is mainly computational or preclinical.")
            if excluded_flag:
                confidence_notes.append("Excluded from recommendation view due to tool-compound status.")
            if metadata["clinical_status"] in {"investigational_breast", "investigational_oncology"}:
                confidence_notes.append("Clinical relevance remains trial-level or investigational.")

            rows.append(
                {
                    "patient_id": sample_id,
                    "drug": drug_name,
                    "drug_name": drug_name,
                    "final_score": final_score,
                    "rna_score": rna["rna_score"],
                    "mutation_pathway_score": mutation_score,
                    "subtype_context_score": context_score,
                    "clinical_actionability_score": clinical_score,
                    "ml_score": ml_score,
                    "predicted_inhibition": predicted_inhibition,
                    "best_dose_um": best_dose_um,
                    "rna_rationale": rna["rna_rationale"],
                    "mutation_rationale": " ".join(mutation_reasons) if mutation_reasons else "No mutation-linked bonus applied.",
                    "context_rationale": " ".join(context_reasons) if context_reasons else "No additional subtype-context rule applied.",
                    "clinical_rationale": " ".join(clinical_reasons),
                    "rationale_short": "; ".join(rationale_short_parts),
                    "rationale_long": " ".join(
                        [
                            rna["rna_rationale"],
                            " ".join(mutation_reasons) if mutation_reasons else "",
                            " ".join(context_reasons) if context_reasons else "",
                            " ".join(clinical_reasons),
                        ]
                    ).strip(),
                    "top_contributing_genes": "; ".join(rna["top_reversed_genes"]),
                    "top_reversed_programs": "; ".join(rna["top_reversed_programs"]),
                    "evidence_tier": evidence_tier,
                    "excluded_flag": excluded_flag,
                    "confidence_notes": " ".join(confidence_notes),
                    "treatability_prob": treatability["probability"],
                    "treatability_label": treatability["label"],
                    "trial_query_term": metadata["trial_query_term"],
                }
            )

        ranking = pd.DataFrame(rows)
        ranking = ranking.sort_values(
            ["excluded_flag", "final_score", "clinical_actionability_score", "rna_score"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

        non_excluded = ranking[~ranking["excluded_flag"]].copy()
        if not non_excluded.empty:
            percentile = non_excluded["final_score"].rank(method="average", pct=True)
            base_confidence = pd.cut(
                percentile,
                bins=[0.0, 0.4, 0.75, 0.9, 1.0],
                labels=["low", "moderate", "high", "very_high"],
                include_lowest=True,
            ).astype(str)

            # Boost confidence when pan-cancer treatability is high AND drug is Tier 1
            _conf_order = ["low", "moderate", "high", "very_high"]
            def _boost(row_idx):
                conf = base_confidence.iloc[row_idx]
                tier = non_excluded.iloc[row_idx]["evidence_tier"]
                treat = treatability["label"]
                if treat == "high" and tier == "Tier 1" and conf in ("high", "moderate"):
                    pos = _conf_order.index(conf)
                    return _conf_order[min(pos + 1, 3)]
                return conf

            non_excluded["confidence"] = [_boost(i) for i in range(len(non_excluded))]
            ranking = ranking.drop(columns=["confidence"], errors="ignore").merge(
                non_excluded[["drug_name", "confidence"]],
                on="drug_name",
                how="left",
            )
        ranking["confidence"] = ranking["confidence"].fillna("low")

        if not include_excluded:
            ranking = ranking[~ranking["excluded_flag"]].copy()

        ranking = ranking.head(top_k).reset_index(drop=True)
        summary = self.summarize_patient(profile)
        summary["treatability"] = treatability
        return ranking, summary


def rank_tcga_patient(
    sample_id: str,
    top_k: int = 30,
    ranker: PersonalizedDrugRanker | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience wrapper for ranking one TCGA patient from cached artifacts."""
    if ranker is None:
        ranker = PersonalizedDrugRanker.from_project_artifacts()

    expression = load_tcga_expression()
    cohort = build_patient_cohort()
    landmark_genes = load_landmark_genes()["gene_symbol"].tolist()
    return ranker.rank_patient(
        sample_id=sample_id,
        expression=expression,
        cohort=cohort,
        landmark_genes=landmark_genes,
        top_k=top_k,
    )
