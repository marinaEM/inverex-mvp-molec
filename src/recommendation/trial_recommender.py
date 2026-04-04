"""
TrialRecommender -- end-to-end drug recommendation engine for clinical trial matching.

Handles three scenarios:
  A. **Known drug** -- in LINCS + GDSC2 training data (e.g. paclitaxel).
     All models fire: reversal, pathway, cell-line model, drug-target features,
     treatability, clinical metadata.  Highest confidence.

  B. **LINCS-only drug** -- has a LINCS L1000 signature but no GDSC2
     dose-response label.  Reversal + pathway perturbation scores work;
     ChemBERTa embedding finds similar drugs with training data.
     Moderate confidence.

  C. **New drug** -- only SMILES structure + known targets; not in LINCS or GDSC2.
     ChemBERTa embedding -> k-nearest neighbours in training drug space ->
     transfer their predictions.  Target vulnerability + analog reversal +
     treatability.  Low confidence (explicitly flagged).

Entry point::

    recommender = TrialRecommender.from_project_artifacts()
    results = recommender.recommend_for_patient(
        "TCGA-A2-A04W-01", candidate_drugs, expression, cohort
    )
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import DATA_CACHE, DATA_PROCESSED, RESULTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence rules per scenario
# ---------------------------------------------------------------------------
CONFIDENCE_RULES = {
    "known": {
        "base_confidence": "high",
        "conditions": {
            "very_high": "evidence_tier == 'Tier 1' AND treatability_label == 'high'",
            "high": "evidence_tier in ('Tier 1', 'Tier 2')",
            "moderate": "evidence_tier == 'Tier 3'",
        },
    },
    "lincs_only": {
        "base_confidence": "moderate",
        "conditions": {
            "moderate": "reversal_score > 0.3 AND target_vulnerability > 0.5",
            "low": "otherwise",
        },
    },
    "new": {
        "base_confidence": "low",
        "conditions": {
            "moderate": "nearest_similarity > 0.85 AND n_analogs >= 3 AND target_vulnerability > 0.5",
            "low": "otherwise",
            "very_low": "nearest_similarity < 0.7 OR n_analogs < 2",
        },
    },
}

# Target-to-pathway mapping (curated for breast cancer)
TARGET_PATHWAY_MAP: dict[str, list[str]] = {
    "ERBB2": ["her2_egfr"],
    "EGFR": ["her2_egfr"],
    "ERBB3": ["her2_egfr"],
    "FGFR1": ["her2_egfr"],
    "PIK3CA": ["pi3k_mtor"],
    "AKT1": ["pi3k_mtor"],
    "MTOR": ["pi3k_mtor"],
    "PTEN": ["pi3k_mtor"],
    "CDK4": ["proliferation"],
    "CDK6": ["proliferation"],
    "CDK1": ["proliferation"],
    "CDK2": ["proliferation"],
    "AURKA": ["proliferation"],
    "AURKB": ["proliferation"],
    "PLK1": ["proliferation"],
    "MKI67": ["proliferation"],
    "CCNB1": ["proliferation"],
    "CCND1": ["er_luminal", "proliferation"],
    "ESR1": ["er_luminal"],
    "PGR": ["er_luminal"],
    "GATA3": ["er_luminal"],
    "FOXA1": ["er_luminal"],
    "BCL2": ["er_luminal"],
    "BRCA1": ["dna_damage"],
    "BRCA2": ["dna_damage"],
    "PARP1": ["dna_damage"],
    "PARP2": ["dna_damage"],
    "RAD51": ["dna_damage"],
    "ATM": ["dna_damage"],
    "ATR": ["dna_damage"],
    "CHEK1": ["dna_damage"],
    "CHEK2": ["dna_damage"],
    "NFKB1": ["immune_inflammation"],
    "RELA": ["immune_inflammation"],
    "STAT3": ["immune_inflammation"],
    "IL6": ["immune_inflammation"],
    "TOP2A": ["proliferation"],
    "TUBB": ["proliferation"],
    "BIRC5": ["proliferation"],
}


class TrialRecommender:
    """
    End-to-end drug recommendation engine for clinical trial matching.
    Handles known drugs, LINCS-only drugs, and completely new drugs.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        ranker: Any | None = None,
        chemberta_embeddings: pd.DataFrame | None = None,
        smiles_cache: pd.DataFrame | None = None,
        drug_fingerprints: pd.DataFrame | None = None,
        lincs_signatures: pd.DataFrame | None = None,
        drug_targets: dict[str, list[str]] | None = None,
        pancancer_model: Any | None = None,
        drug_metadata: pd.DataFrame | None = None,
        known_drug_names: set[str] | None = None,
        lincs_drug_names: set[str] | None = None,
    ) -> None:
        self.ranker = ranker
        self.chemberta_embeddings = chemberta_embeddings
        self.smiles_cache = smiles_cache
        self.drug_fingerprints = drug_fingerprints
        self.lincs_signatures = lincs_signatures
        self.drug_targets = drug_targets or {}
        self.pancancer_model = pancancer_model
        self.drug_metadata = drug_metadata
        self.known_drug_names = known_drug_names or set()
        self.lincs_drug_names = lincs_drug_names or set()

        # Build embedding lookup matrix (compound_name -> numpy vector)
        self._emb_matrix: np.ndarray | None = None
        self._emb_names: list[str] = []
        if self.chemberta_embeddings is not None and not self.chemberta_embeddings.empty:
            emb_cols = [c for c in self.chemberta_embeddings.columns if c.startswith("chemberta_")]
            self._emb_matrix = self.chemberta_embeddings[emb_cols].values.astype(np.float32)
            self._emb_names = self.chemberta_embeddings["compound_name"].tolist()

        # LINCS meta columns (to separate gene columns)
        self._lincs_meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
        if self.lincs_signatures is not None:
            self._lincs_gene_cols = [
                c for c in self.lincs_signatures.columns if c not in self._lincs_meta_cols
            ]
        else:
            self._lincs_gene_cols = []

        logger.info(
            "TrialRecommender initialised: %d known drugs, %d LINCS drugs, "
            "%d ChemBERTa embeddings",
            len(self.known_drug_names),
            len(self.lincs_drug_names),
            len(self._emb_names),
        )

    # --------------------------------------------------------- factory method
    @classmethod
    def from_project_artifacts(
        cls,
        cache_dir: Path = DATA_CACHE,
        processed_dir: Path = DATA_PROCESSED,
        results_dir: Path = RESULTS,
    ) -> "TrialRecommender":
        """Load everything from cached project files."""
        import joblib

        logger.info("Loading TrialRecommender from project artifacts ...")

        # ---- Personalized ranker (optional) ----------------------------------
        ranker = None
        try:
            from src.ranking.personalized_ranker import PersonalizedDrugRanker

            ranker = PersonalizedDrugRanker.from_project_artifacts()
            logger.info("  Loaded PersonalizedDrugRanker")
        except Exception as exc:
            logger.warning("  Could not load PersonalizedDrugRanker: %s", exc)

        # ---- ChemBERTa embeddings -------------------------------------------
        chemberta_embeddings = None
        emb_path = cache_dir / "chemberta_embeddings.parquet"
        if emb_path.exists():
            chemberta_embeddings = pd.read_parquet(emb_path)
            logger.info("  Loaded ChemBERTa embeddings: %s", chemberta_embeddings.shape)
        else:
            logger.warning("  ChemBERTa embeddings not found at %s", emb_path)

        # ---- SMILES cache ---------------------------------------------------
        smiles_cache = None
        smiles_path = cache_dir / "compound_smiles_cache.parquet"
        if smiles_path.exists():
            smiles_cache = pd.read_parquet(smiles_path)
            logger.info("  Loaded SMILES cache: %d compounds", len(smiles_cache))
        else:
            logger.warning("  SMILES cache not found at %s", smiles_path)

        # ---- Drug fingerprints ----------------------------------------------
        drug_fingerprints = None
        fp_path = cache_dir / "drug_fingerprints.parquet"
        if fp_path.exists():
            drug_fingerprints = pd.read_parquet(fp_path)
            logger.info("  Loaded drug fingerprints: %s", drug_fingerprints.shape)
        else:
            logger.warning("  Drug fingerprints not found at %s", fp_path)

        # ---- LINCS signatures -----------------------------------------------
        lincs_signatures = None
        lincs_path = cache_dir / "breast_l1000_signatures.parquet"
        if lincs_path.exists():
            lincs_signatures = pd.read_parquet(lincs_path)
            logger.info("  Loaded LINCS signatures: %s", lincs_signatures.shape)
        else:
            logger.warning("  LINCS signatures not found at %s", lincs_path)

        # ---- Drug targets from GDSC2 ----------------------------------------
        drug_targets: dict[str, list[str]] = {}
        try:
            from src.features.drug_target_interactions import parse_drug_targets

            drug_targets = parse_drug_targets()
            logger.info("  Parsed drug targets: %d drugs", len(drug_targets))
        except Exception as exc:
            logger.warning("  Could not parse drug targets: %s", exc)

        # ---- Pan-cancer model ------------------------------------------------
        pancancer_model = None
        pc_path = results_dir / "pan_cancer_patient_model.joblib"
        if pc_path.exists():
            try:
                pancancer_model = joblib.load(pc_path)
                logger.info("  Loaded pan-cancer patient model")
            except Exception as exc:
                logger.warning("  Could not load pan-cancer model: %s", exc)

        # ---- Drug metadata ---------------------------------------------------
        drug_metadata = None
        meta_path = processed_dir / "drug_metadata_breast.tsv"
        if meta_path.exists():
            drug_metadata = pd.read_csv(meta_path, sep="\t")
            logger.info("  Loaded drug metadata: %d drugs", len(drug_metadata))
        else:
            logger.warning("  Drug metadata not found at %s", meta_path)

        # ---- Compute known / LINCS drug sets ---------------------------------
        known_drug_names: set[str] = set()
        if drug_fingerprints is not None:
            known_drug_names = set(drug_fingerprints["compound_name"].str.lower().unique())

        lincs_drug_names: set[str] = set()
        if lincs_signatures is not None:
            lincs_drug_names = set(lincs_signatures["pert_iname"].str.lower().unique())

        return cls(
            ranker=ranker,
            chemberta_embeddings=chemberta_embeddings,
            smiles_cache=smiles_cache,
            drug_fingerprints=drug_fingerprints,
            lincs_signatures=lincs_signatures,
            drug_targets=drug_targets,
            pancancer_model=pancancer_model,
            drug_metadata=drug_metadata,
            known_drug_names=known_drug_names,
            lincs_drug_names=lincs_drug_names,
        )

    # --------------------------------------------------------- classify drug
    def classify_drug(
        self,
        drug_name: str | None = None,
        smiles: str | None = None,
        targets: list[str] | None = None,
    ) -> str:
        """
        Determine which scenario this drug falls into.

        Returns
        -------
        str
            ``"known"`` | ``"lincs_only"`` | ``"new"``
        """
        name_lower = drug_name.strip().lower() if drug_name else ""

        if name_lower in self.known_drug_names:
            return "known"

        if name_lower in self.lincs_drug_names:
            return "lincs_only"

        return "new"

    # -------------------------------------------------- main recommendation
    def recommend_for_patient(
        self,
        sample_id: str,
        candidate_drugs: list[dict],
        expression: pd.DataFrame | None = None,
        cohort: pd.DataFrame | None = None,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """
        Rank candidate drugs for a patient, handling mixed known/new drugs.

        Parameters
        ----------
        sample_id : str
            TCGA sample ID (e.g. ``"TCGA-A2-A04W-01"``).
        candidate_drugs : list[dict]
            Each dict can have: ``name``, ``smiles``, ``targets``, ``drug_class``.
        expression : pd.DataFrame, optional
            Patient expression matrix (samples x genes). Loaded from TCGA if None.
        cohort : pd.DataFrame, optional
            Patient cohort metadata. Loaded from TCGA if None.
        top_k : int
            Maximum number of drugs to return.

        Returns
        -------
        pd.DataFrame
            Columns: drug_name, scenario, score, confidence,
            confidence_interval_lower, confidence_interval_upper,
            evidence_tier, rationale, nearest_analogs,
            recommended_trial_design, trial_matches
        """
        # ---- Load data if not provided --------------------------------------
        if expression is None or cohort is None:
            try:
                from src.data_ingestion.tcga import build_patient_cohort, load_tcga_expression

                if expression is None:
                    expression = load_tcga_expression()
                if cohort is None:
                    cohort = build_patient_cohort()
            except Exception as exc:
                logger.error("Could not load TCGA data: %s", exc)
                raise

        # ---- Build patient profile ------------------------------------------
        patient_profile = self._build_patient_profile(sample_id, expression, cohort)

        # ---- Score each candidate -------------------------------------------
        results: list[dict[str, Any]] = []
        for drug_info in candidate_drugs:
            drug_name = drug_info.get("name", "unknown")
            scenario = self.classify_drug(
                drug_name=drug_name,
                smiles=drug_info.get("smiles"),
                targets=drug_info.get("targets"),
            )
            logger.info("Scoring %s (scenario=%s)", drug_name, scenario)

            try:
                if scenario == "known":
                    row = self._score_known_drug(patient_profile, drug_name)
                elif scenario == "lincs_only":
                    row = self._score_lincs_drug(patient_profile, drug_name, drug_info)
                else:
                    row = self._score_new_drug(patient_profile, drug_info)
            except Exception as exc:
                logger.error("Error scoring %s: %s", drug_name, exc, exc_info=True)
                row = self._fallback_row(drug_name, scenario, str(exc))

            row["drug_name"] = drug_name
            row["scenario"] = scenario
            results.append(row)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # ---- Sort by score descending ---------------------------------------
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df = df.head(top_k).reset_index(drop=True)

        # ---- Ensure output columns exist ------------------------------------
        for col in [
            "drug_name", "scenario", "score", "confidence",
            "confidence_interval_lower", "confidence_interval_upper",
            "evidence_tier", "rationale", "nearest_analogs",
            "recommended_trial_design", "trial_matches",
        ]:
            if col not in df.columns:
                df[col] = None

        return df

    # ===================================================================
    # Scenario A: Known drug (in LINCS + GDSC2)
    # ===================================================================
    def _score_known_drug(
        self,
        patient_profile: Any,
        drug_name: str,
    ) -> dict[str, Any]:
        """Full scoring pipeline for Scenario A drugs."""
        result: dict[str, Any] = {}

        # -- Use the personalised ranker if available -------------------------
        ranker_score = 0.0
        rna_score = 0.0
        mutation_score = 0.0
        context_score = 0.0
        clinical_score = 0.0
        evidence_tier = "Tier 3"
        treatability_label = "unavailable"
        rationale_parts: list[str] = []

        if self.ranker is not None:
            try:
                rna_info = self.ranker._score_rna_component(patient_profile, drug_name)
                rna_score = rna_info.get("rna_score", 0.0)
                rationale_parts.append(rna_info.get("rna_rationale", ""))

                # Mutation / pathway
                if drug_name in self.ranker.drug_metadata.index:
                    metadata = self.ranker.drug_metadata.loc[drug_name]
                    mutation_score, mut_reasons = self.ranker._score_mutation_pathway_component(
                        patient_profile, metadata
                    )
                    context_score, ctx_reasons = self.ranker._score_subtype_context_component(
                        patient_profile, metadata
                    )
                    clinical_score, excluded, clin_reasons = self.ranker._score_clinical_component(
                        metadata
                    )
                    evidence_tier = self.ranker._determine_evidence_tier(
                        patient_profile, metadata, excluded
                    )
                    rationale_parts.extend(mut_reasons)
                    rationale_parts.extend(ctx_reasons)
                    rationale_parts.extend(clin_reasons)

                # Weighted composite
                w = self.ranker.weights
                ranker_score = (
                    w["rna_score"] * rna_score
                    + w["mutation_pathway_score"] * mutation_score
                    + w["subtype_context_score"] * context_score
                    + w["clinical_actionability_score"] * clinical_score
                )
            except Exception as exc:
                logger.warning("Ranker scoring failed for %s: %s", drug_name, exc)
                rationale_parts.append(f"Ranker scoring partially failed: {exc}")

        # -- Treatability (pan-cancer model) ----------------------------------
        treatability = self._compute_treatability(patient_profile)
        treatability_label = treatability.get("label", "unavailable")
        if treatability.get("note"):
            rationale_parts.append(treatability["note"])

        # -- Drug targets if available ----------------------------------------
        targets = self.drug_targets.get(drug_name.lower(), [])
        target_vuln = self.compute_target_vulnerability(patient_profile, targets) if targets else {}
        if target_vuln:
            rationale_parts.append(target_vuln.get("rationale", ""))

        # -- Composite score --------------------------------------------------
        score = ranker_score
        if target_vuln:
            # Blend in a small target vulnerability bonus
            score += 0.05 * target_vuln.get("vulnerability_score", 0.0)

        # -- Confidence -------------------------------------------------------
        confidence = self._determine_confidence(
            scenario="known",
            evidence_tier=evidence_tier,
            treatability_label=treatability_label,
            score=score,
        )

        # -- Confidence interval (heuristic for known drugs) ------------------
        ci_half = 0.08 if confidence in ("high", "very_high") else 0.15
        ci_lower = max(score - ci_half, -1.0)
        ci_upper = min(score + ci_half, 1.0)

        # -- Trial matches ----------------------------------------------------
        trials = self._safe_match_trials(drug_name)

        result.update(
            {
                "score": round(float(score), 4),
                "confidence": confidence,
                "confidence_interval_lower": round(ci_lower, 4),
                "confidence_interval_upper": round(ci_upper, 4),
                "evidence_tier": evidence_tier,
                "rationale": " ".join([p for p in rationale_parts if p]).strip(),
                "nearest_analogs": [],
                "recommended_trial_design": self._recommend_trial_design("known", drug_name),
                "trial_matches": trials,
                "rna_score": round(rna_score, 4),
                "treatability_label": treatability_label,
                "treatability_prob": treatability.get("probability"),
                "target_vulnerability": target_vuln.get("vulnerability_score", None),
            }
        )
        return result

    # ===================================================================
    # Scenario B: LINCS-only drug
    # ===================================================================
    def _score_lincs_drug(
        self,
        patient_profile: Any,
        drug_name: str,
        drug_info: dict | None = None,
    ) -> dict[str, Any]:
        """Reversal + pathway + analog scoring for Scenario B drugs."""
        rationale_parts: list[str] = []

        # -- Reversal score from LINCS ----------------------------------------
        reversal_score = self._compute_reversal_from_lincs(patient_profile, drug_name)
        rationale_parts.append(
            f"LINCS reversal score: {reversal_score:.3f}."
        )

        # -- Target vulnerability ---------------------------------------------
        targets = (drug_info or {}).get("targets", [])
        if not targets:
            targets = self.drug_targets.get(drug_name.lower(), [])
        target_vuln = self.compute_target_vulnerability(patient_profile, targets) if targets else {}
        vuln_score = target_vuln.get("vulnerability_score", 0.0)
        if target_vuln.get("rationale"):
            rationale_parts.append(target_vuln["rationale"])

        # -- Find nearest known analogs via ChemBERTa ------------------------
        smiles = (drug_info or {}).get("smiles", "")
        if not smiles and self.smiles_cache is not None:
            match = self.smiles_cache[self.smiles_cache["compound_name"].str.lower() == drug_name.lower()]
            if not match.empty:
                smiles = match.iloc[0]["smiles"]

        nearest_analogs: list[tuple[str, float]] = []
        analog_scores: list[float] = []
        if smiles and self._emb_matrix is not None:
            nearest_analogs = self.find_nearest_drugs(smiles, k=5)
            # Transfer scores from known analogs
            for analog_name, sim in nearest_analogs:
                if analog_name.lower() in self.known_drug_names:
                    analog_scores.append(sim)
            if nearest_analogs:
                top_names = [n for n, _ in nearest_analogs[:3]]
                rationale_parts.append(
                    f"Nearest known analogs (ChemBERTa): {', '.join(top_names)}."
                )

        # -- Treatability -----------------------------------------------------
        treatability = self._compute_treatability(patient_profile)
        if treatability.get("note"):
            rationale_parts.append(treatability["note"])

        # -- Composite score --------------------------------------------------
        # Weighted blend: reversal dominates, with target and analog bonuses
        score = 0.50 * reversal_score + 0.25 * vuln_score
        if analog_scores:
            score += 0.25 * np.mean(analog_scores)
        else:
            score += 0.0

        # -- Confidence -------------------------------------------------------
        confidence = self._determine_confidence(
            scenario="lincs_only",
            reversal_score=reversal_score,
            target_vulnerability=vuln_score,
        )

        # -- Confidence interval (wider for LINCS-only) -----------------------
        ci_half = 0.20
        ci_lower = max(score - ci_half, -1.0)
        ci_upper = min(score + ci_half, 1.0)

        # -- Trials -----------------------------------------------------------
        trials = self._safe_match_trials(drug_name, drug_info)

        return {
            "score": round(float(score), 4),
            "confidence": confidence,
            "confidence_interval_lower": round(ci_lower, 4),
            "confidence_interval_upper": round(ci_upper, 4),
            "evidence_tier": "Tier 3",
            "rationale": " ".join([p for p in rationale_parts if p]).strip(),
            "nearest_analogs": [n for n, _ in nearest_analogs[:5]],
            "recommended_trial_design": self._recommend_trial_design("lincs_only", drug_name),
            "trial_matches": trials,
            "rna_score": round(reversal_score, 4),
            "treatability_label": treatability.get("label", "unavailable"),
            "treatability_prob": treatability.get("probability"),
            "target_vulnerability": round(vuln_score, 4),
        }

    # ===================================================================
    # Scenario C: New drug
    # ===================================================================
    def _score_new_drug(
        self,
        patient_profile: Any,
        drug_info: dict,
    ) -> dict[str, Any]:
        """
        Analog-based scoring for Scenario C drugs.

        1. ChemBERTa embedding -> k-nearest neighbours in training drug space
        2. Target vulnerability -> are targets active in this patient?
        3. Analog reversal -> average reversal scores of similar known drugs
        4. Treatability -> general response probability
        5. Explicit low confidence flag
        """
        drug_name = drug_info.get("name", "unknown")
        smiles = drug_info.get("smiles", "")
        targets = drug_info.get("targets", [])
        rationale_parts: list[str] = []

        # ---- 1. ChemBERTa nearest neighbours --------------------------------
        nearest_analogs: list[tuple[str, float]] = []
        best_similarity = 0.0
        n_known_analogs = 0
        if smiles and self._emb_matrix is not None:
            nearest_analogs = self.find_nearest_drugs(smiles, k=5)
            if nearest_analogs:
                best_similarity = nearest_analogs[0][1]
                n_known_analogs = sum(
                    1 for name, _ in nearest_analogs if name.lower() in self.known_drug_names
                )
                top_names = [f"{n} ({s:.2f})" for n, s in nearest_analogs[:3]]
                rationale_parts.append(
                    f"Nearest analogs (ChemBERTa cosine): {', '.join(top_names)}."
                )
        else:
            rationale_parts.append(
                "No SMILES provided or ChemBERTa embeddings unavailable; "
                "analog search skipped."
            )

        # ---- 2. Target vulnerability ----------------------------------------
        target_vuln = self.compute_target_vulnerability(patient_profile, targets) if targets else {}
        vuln_score = target_vuln.get("vulnerability_score", 0.0)
        if target_vuln.get("rationale"):
            rationale_parts.append(target_vuln["rationale"])

        # ---- 3. Analog reversal -- average reversal of nearest known drugs ---
        analog_reversal = 0.0
        analog_reversal_details: list[str] = []
        for analog_name, sim in nearest_analogs:
            if analog_name.lower() in self.known_drug_names or analog_name.lower() in self.lincs_drug_names:
                rev = self._compute_reversal_from_lincs(patient_profile, analog_name)
                if rev != 0.0:
                    analog_reversal_details.append(f"{analog_name}: {rev:.3f}")
                    analog_reversal += rev * sim  # weight by similarity
        if analog_reversal_details:
            total_sim = sum(s for _, s in nearest_analogs if s > 0)
            if total_sim > 0:
                analog_reversal /= total_sim
            rationale_parts.append(
                f"Analog reversal (similarity-weighted): {analog_reversal:.3f} "
                f"from [{', '.join(analog_reversal_details[:3])}]."
            )

        # ---- 4. Treatability ------------------------------------------------
        treatability = self._compute_treatability(patient_profile)
        treat_prob = treatability.get("probability") or 0.5
        if treatability.get("note"):
            rationale_parts.append(treatability["note"])

        # ---- 5. Composite score (analog-transferred) ------------------------
        # Score is built from available evidence, weighted lower than known
        score = (
            0.35 * analog_reversal
            + 0.30 * vuln_score
            + 0.20 * best_similarity
            + 0.15 * treat_prob
        )

        # ---- Confidence (explicitly low for new drugs) ----------------------
        confidence = self._determine_confidence(
            scenario="new",
            nearest_similarity=best_similarity,
            n_analogs=n_known_analogs,
            target_vulnerability=vuln_score,
        )

        # -- Wide confidence interval for new drugs ---------------------------
        ci_half = 0.30
        ci_lower = max(score - ci_half, -1.0)
        ci_upper = min(score + ci_half, 1.0)

        # Low-confidence note
        rationale_parts.append(
            "NOTE: This drug is NOT in our training data. All scores are "
            "transferred from structural analogs. Confidence is LOW."
        )

        # -- Trials (search by name + targets) --------------------------------
        trials = self._safe_match_trials(drug_name, drug_info)

        return {
            "score": round(float(score), 4),
            "confidence": confidence,
            "confidence_interval_lower": round(ci_lower, 4),
            "confidence_interval_upper": round(ci_upper, 4),
            "evidence_tier": "Tier 3 (analog-based)",
            "rationale": " ".join([p for p in rationale_parts if p]).strip(),
            "nearest_analogs": [n for n, _ in nearest_analogs[:5]],
            "recommended_trial_design": self._recommend_trial_design(
                "new", drug_name, drug_info
            ),
            "trial_matches": trials,
            "rna_score": round(analog_reversal, 4),
            "treatability_label": treatability.get("label", "unavailable"),
            "treatability_prob": treatability.get("probability"),
            "target_vulnerability": round(vuln_score, 4),
        }

    # ===================================================================
    # ChemBERTa nearest-neighbour search
    # ===================================================================
    def find_nearest_drugs(
        self,
        smiles_or_embedding: str | np.ndarray,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find the k most similar drugs in our training set using ChemBERTa
        cosine similarity.

        Parameters
        ----------
        smiles_or_embedding
            Either a SMILES string (will be embedded with ChemBERTa) or a
            pre-computed embedding vector.
        k : int
            Number of neighbours to return.

        Returns
        -------
        list of (drug_name, similarity_score) tuples, sorted descending.
        """
        if self._emb_matrix is None or len(self._emb_names) == 0:
            logger.warning("No ChemBERTa embeddings available for neighbour search")
            return []

        # Get query embedding
        if isinstance(smiles_or_embedding, str):
            query_emb = self._embed_smiles(smiles_or_embedding)
            if query_emb is None:
                return []
        else:
            query_emb = np.asarray(smiles_or_embedding, dtype=np.float32)

        # Vectorised cosine similarity
        norms = np.linalg.norm(self._emb_matrix, axis=1)
        query_norm = np.linalg.norm(query_emb)
        if query_norm < 1e-12:
            logger.warning("Query embedding has near-zero norm")
            return []

        cosine_sims = self._emb_matrix @ query_emb / (norms * query_norm + 1e-8)

        # Top k
        top_indices = np.argsort(cosine_sims)[::-1][:k]
        results = [
            (self._emb_names[i], float(cosine_sims[i]))
            for i in top_indices
        ]
        return results

    # ===================================================================
    # Target vulnerability scoring
    # ===================================================================
    def compute_target_vulnerability(
        self,
        patient_profile: Any,
        targets: list[str],
    ) -> dict[str, Any]:
        """
        Assess whether drug targets are active/dysregulated in this patient.

        Uses expression z-scores from the patient signature and pathway
        activity scores from the patient profile.

        Returns
        -------
        dict with vulnerability_score, target_expression, mean_target_z,
        n_targets_found, n_targets_dysregulated, pathway_activity, rationale.
        """
        if not targets:
            return {
                "vulnerability_score": 0.0,
                "target_expression": {},
                "mean_target_z": 0.0,
                "n_targets_found": 0,
                "n_targets_dysregulated": 0,
                "pathway_activity": {},
                "rationale": "No drug targets provided.",
            }

        # 1. Direct target expression from patient signature
        signature = patient_profile.signature
        target_expression: dict[str, float] = {}
        for gene in targets:
            if gene in signature.index:
                target_expression[gene] = float(signature[gene])

        mean_target_z = float(np.mean(list(target_expression.values()))) if target_expression else 0.0
        n_dysregulated = sum(1 for v in target_expression.values() if abs(v) > 2)

        # 2. Map targets to pathways and get pathway activity
        target_pathways = self._map_targets_to_pathways(targets)
        pathway_activity: dict[str, float] = {}
        for pw in target_pathways:
            pathway_activity[pw] = patient_profile.pathway_scores.get(pw, 0.0)

        # 3. Overall vulnerability score
        vulnerability = 0.0
        if mean_target_z > 1.0:  # targets are overexpressed
            vulnerability += 0.4
        if n_dysregulated > 0:
            vulnerability += 0.3
        if any(v > 0.5 for v in pathway_activity.values()):
            vulnerability += 0.3

        return {
            "vulnerability_score": min(vulnerability, 1.0),
            "target_expression": target_expression,
            "mean_target_z": round(mean_target_z, 4),
            "n_targets_found": len(target_expression),
            "n_targets_dysregulated": n_dysregulated,
            "pathway_activity": pathway_activity,
            "rationale": self._build_target_rationale(target_expression, pathway_activity, targets),
        }

    # ===================================================================
    # Trial matching
    # ===================================================================
    def match_trials(
        self,
        drug_name: str,
        drug_info: dict | None = None,
    ) -> list[dict]:
        """
        Query ClinicalTrials.gov for matching trials.

        Uses ``trial_query_term`` from metadata if available, otherwise
        uses ``drug_name`` directly. For new drugs, also searches by
        target / pathway.
        """
        from src.trials.match_trials import search_breast_trials

        # Determine query term
        query_term = drug_name
        if self.drug_metadata is not None:
            match = self.drug_metadata[
                self.drug_metadata["drug_name"].str.lower() == drug_name.lower()
            ]
            if not match.empty:
                qt = match.iloc[0].get("trial_query_term")
                if pd.notna(qt) and str(qt).strip():
                    query_term = str(qt).strip()

        trials = search_breast_trials(query_term)

        # For new drugs, also search by mechanism / target
        if drug_info and drug_info.get("targets"):
            for target in drug_info["targets"][:3]:
                try:
                    target_trials = search_breast_trials(f"{target} inhibitor")
                    trials.extend(target_trials)
                except Exception as exc:
                    logger.warning("Trial search for target %s failed: %s", target, exc)

        # Deduplicate by NCT ID
        seen: set[str] = set()
        unique: list[dict] = []
        for t in trials:
            nct = t.get("nct_id", "")
            if nct and nct not in seen:
                seen.add(nct)
                unique.append(t)

        return unique

    # ===================================================================
    # Report generation
    # ===================================================================
    def generate_report(
        self,
        patient_id: str,
        recommendations_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate a structured report suitable for display in the Streamlit
        app and JSON export.

        Includes: patient summary, ranked drugs with rationale, trial matches,
        confidence assessment, and disclaimers.
        """
        report: dict[str, Any] = {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": (
                "RESEARCH DEMO ONLY. Not a clinical treatment recommendation. "
                "This system is for exploratory analysis and must not be used "
                "for medical decision-making."
            ),
            "n_drugs_evaluated": len(recommendations_df),
            "scenario_counts": {},
            "drugs": [],
        }

        if "scenario" in recommendations_df.columns:
            report["scenario_counts"] = recommendations_df["scenario"].value_counts().to_dict()

        for idx, row in recommendations_df.iterrows():
            drug_report: dict[str, Any] = {
                "rank": int(idx) + 1,
                "drug_name": row.get("drug_name", "unknown"),
                "scenario": row.get("scenario", "unknown"),
                "score": _safe_round(row.get("score")),
                "confidence": row.get("confidence", "unknown"),
                "confidence_interval": [
                    _safe_round(row.get("confidence_interval_lower")),
                    _safe_round(row.get("confidence_interval_upper")),
                ],
                "evidence_tier": row.get("evidence_tier", "N/A"),
                "rationale": row.get("rationale", ""),
                "trial_matches": _safe_jsonify(row.get("trial_matches", [])),
            }

            if row.get("scenario") == "new":
                analogs = row.get("nearest_analogs", [])
                drug_report["nearest_analogs"] = analogs if isinstance(analogs, list) else []
                first_analog = analogs[0] if analogs else "unknown"
                drug_report["note"] = (
                    f"This drug is not in our training data. Predictions are "
                    f"based on structural similarity to {first_analog} and "
                    f"target pathway activity. Confidence is low."
                )
            elif row.get("scenario") == "lincs_only":
                analogs = row.get("nearest_analogs", [])
                drug_report["nearest_analogs"] = analogs if isinstance(analogs, list) else []
                drug_report["note"] = (
                    "This drug has LINCS L1000 signatures but no GDSC2 "
                    "dose-response data. No cell-line model prediction is "
                    "available. Confidence is moderate."
                )

            report["drugs"].append(drug_report)

        return report

    # ===================================================================
    # Private helpers
    # ===================================================================

    def _build_patient_profile(self, sample_id, expression, cohort):
        """Build a PatientProfile using the ranker or a minimal fallback."""
        if self.ranker is not None:
            try:
                return self.ranker.build_patient_profile(sample_id, expression, cohort)
            except Exception as exc:
                logger.warning("Ranker profile build failed: %s; using fallback", exc)

        # Minimal fallback: build a lightweight profile
        from src.ranking.personalized_ranker import PatientProfile, _compute_pathway_scores

        if sample_id not in cohort.index:
            raise ValueError(f"Sample {sample_id} not found in cohort index")

        patient_row = cohort.loc[sample_id]
        # Build signature (z-scored expression for landmark genes)
        if sample_id in expression.index:
            patient_expr = expression.loc[sample_id]
            mean = expression.mean()
            std = expression.std().replace(0, 1)
            signature = (patient_expr - mean) / std
        else:
            signature = pd.Series(dtype=float)

        subtype = str(patient_row.get("pam50_subtype", "Unknown"))
        pathway_scores = _compute_pathway_scores(signature)

        return PatientProfile(
            sample_id=sample_id,
            subtype=subtype,
            subtype_group=_infer_subtype_group_fallback(subtype),
            er_status=str(patient_row.get("er_status", "Unknown")),
            pr_status=str(patient_row.get("pr_status", "Unknown")),
            her2_status=str(patient_row.get("her2_status", "Unknown")),
            signature=signature,
            pathway_scores=pathway_scores,
            top_up_genes=signature.sort_values(ascending=False).head(5).index.tolist() if not signature.empty else [],
            top_down_genes=signature.sort_values().head(5).index.tolist() if not signature.empty else [],
            mutation_flags={k: int(patient_row.get(k, 0) or 0) for k in [
                "ERBB2_amp", "mut_TP53", "mut_PIK3CA", "mut_ERBB2",
                "mut_ESR1", "mut_GATA3", "mut_MAP3K1", "mut_AKT1",
                "mut_PTEN", "mut_BRCA1", "mut_BRCA2",
            ]},
            er_positive=str(patient_row.get("er_status", "")).strip().lower() == "positive",
            her2_active=str(patient_row.get("her2_status", "")).strip().lower() == "positive",
            pi3k_axis_altered=any([
                int(patient_row.get("mut_PIK3CA", 0) or 0) == 1,
                int(patient_row.get("mut_AKT1", 0) or 0) == 1,
                int(patient_row.get("mut_PTEN", 0) or 0) == 1,
            ]),
            brca_loss=any([
                int(patient_row.get("mut_BRCA1", 0) or 0) == 1,
                int(patient_row.get("mut_BRCA2", 0) or 0) == 1,
            ]),
            luminal_marker_altered=any([
                int(patient_row.get("mut_ESR1", 0) or 0) == 1,
                int(patient_row.get("mut_GATA3", 0) or 0) == 1,
                int(patient_row.get("mut_MAP3K1", 0) or 0) == 1,
            ]),
            tp53_mutated=int(patient_row.get("mut_TP53", 0) or 0) == 1,
            high_proliferation=pathway_scores.get("proliferation", 0) >= 0.75,
        )

    def _compute_reversal_from_lincs(self, patient_profile, drug_name: str) -> float:
        """Compute average reversal score across top LINCS signatures for a drug."""
        if self.lincs_signatures is None:
            return 0.0

        from src.ranking.personalized_ranker import compute_reversal_score

        subset = self.lincs_signatures[
            self.lincs_signatures["pert_iname"].str.lower() == drug_name.lower()
        ]
        if subset.empty:
            return 0.0

        reversals = []
        for _, row in subset.iterrows():
            drug_sig = row[self._lincs_gene_cols]
            rev = compute_reversal_score(patient_profile.signature, drug_sig)
            reversals.append(rev)

        if not reversals:
            return 0.0

        # Average of top 3
        reversals.sort(reverse=True)
        top_k = min(3, len(reversals))
        return float(np.mean(reversals[:top_k]))

    def _compute_treatability(self, patient_profile) -> dict[str, Any]:
        """Compute patient-level treatability using the pan-cancer model."""
        if self.pancancer_model is None:
            return {"probability": None, "label": "unavailable", "note": "Pan-cancer model not loaded."}

        try:
            model_features = self.pancancer_model.feature_name_
            patient_values = np.array(
                [float(patient_profile.signature.get(gene, 0.0)) for gene in model_features],
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
                "note": (
                    f"Pan-cancer response probability {prob:.1%} ({label}) -- "
                    f"based on expression patterns from 3,730 patients across 11 cancer types."
                ),
            }
        except Exception as exc:
            logger.warning("Treatability computation failed: %s", exc)
            return {"probability": None, "label": "error", "note": f"Pan-cancer score failed: {exc}"}

    def _embed_smiles(self, smiles: str) -> np.ndarray | None:
        """Embed a single SMILES string using ChemBERTa."""
        try:
            from src.features.chemical_embeddings import get_chemberta_embeddings

            embeddings = get_chemberta_embeddings([smiles])
            return embeddings[0]
        except Exception as exc:
            logger.warning("ChemBERTa embedding failed for SMILES: %s", exc)
            return None

    def _map_targets_to_pathways(self, targets: list[str]) -> list[str]:
        """Map drug target genes to pathway names from the curated mapping."""
        pathways: set[str] = set()
        for gene in targets:
            gene_upper = gene.upper()
            if gene_upper in TARGET_PATHWAY_MAP:
                pathways.update(TARGET_PATHWAY_MAP[gene_upper])
        return sorted(pathways)

    def _build_target_rationale(
        self,
        target_expression: dict[str, float],
        pathway_activity: dict[str, float],
        targets: list[str],
    ) -> str:
        """Build a human-readable rationale for target vulnerability."""
        parts: list[str] = []
        n_found = len(target_expression)
        n_total = len(targets)

        if n_found == 0:
            return f"None of the {n_total} drug targets found in patient expression profile."

        parts.append(f"{n_found}/{n_total} drug targets found in expression profile.")

        overexpressed = [g for g, z in target_expression.items() if z > 2]
        underexpressed = [g for g, z in target_expression.items() if z < -2]
        if overexpressed:
            parts.append(f"Overexpressed (z > 2): {', '.join(overexpressed)}.")
        if underexpressed:
            parts.append(f"Underexpressed (z < -2): {', '.join(underexpressed)}.")

        active_pathways = [pw for pw, score in pathway_activity.items() if score > 0.5]
        if active_pathways:
            parts.append(f"Active pathways: {', '.join(active_pathways)}.")

        return " ".join(parts)

    def _determine_confidence(self, scenario: str, **kwargs) -> str:
        """Determine confidence level based on scenario-specific rules."""
        if scenario == "known":
            evidence_tier = kwargs.get("evidence_tier", "Tier 3")
            treat_label = kwargs.get("treatability_label", "unavailable")
            if evidence_tier == "Tier 1" and treat_label == "high":
                return "very_high"
            if evidence_tier in ("Tier 1", "Tier 2"):
                return "high"
            return "moderate"

        elif scenario == "lincs_only":
            rev = kwargs.get("reversal_score", 0.0)
            vuln = kwargs.get("target_vulnerability", 0.0)
            if rev > 0.3 and vuln > 0.5:
                return "moderate"
            return "low"

        elif scenario == "new":
            sim = kwargs.get("nearest_similarity", 0.0)
            n_analogs = kwargs.get("n_analogs", 0)
            vuln = kwargs.get("target_vulnerability", 0.0)
            if sim < 0.7 or n_analogs < 2:
                return "very_low"
            if sim > 0.85 and n_analogs >= 3 and vuln > 0.5:
                return "moderate"
            return "low"

        return "low"

    def _recommend_trial_design(
        self,
        scenario: str,
        drug_name: str,
        drug_info: dict | None = None,
    ) -> str:
        """Generate a recommended trial design note based on scenario."""
        if scenario == "known":
            return (
                f"Standard phase II/III trial design appropriate for {drug_name}. "
                f"Biomarker-driven patient selection recommended."
            )
        elif scenario == "lincs_only":
            return (
                f"Phase I/II trial recommended for {drug_name} with dose-escalation "
                f"and pharmacogenomic monitoring. Correlative LINCS signature "
                f"analysis suggested."
            )
        else:
            targets_str = ", ".join((drug_info or {}).get("targets", [])[:3]) or "unknown"
            return (
                f"Phase I dose-finding trial recommended for {drug_name}. "
                f"Target engagement assay for {targets_str} required. "
                f"Basket/umbrella trial design with molecular pre-screening suggested."
            )

    def _safe_match_trials(self, drug_name: str, drug_info: dict | None = None) -> list[dict]:
        """Trial matching with error handling."""
        try:
            return self.match_trials(drug_name, drug_info)
        except Exception as exc:
            logger.warning("Trial matching failed for %s: %s", drug_name, exc)
            return []

    def _fallback_row(self, drug_name: str, scenario: str, error_msg: str) -> dict[str, Any]:
        """Generate a minimal result row when scoring fails."""
        return {
            "score": 0.0,
            "confidence": "very_low",
            "confidence_interval_lower": -0.3,
            "confidence_interval_upper": 0.3,
            "evidence_tier": "N/A",
            "rationale": f"Scoring failed: {error_msg}",
            "nearest_analogs": [],
            "recommended_trial_design": "Unable to generate recommendation due to scoring error.",
            "trial_matches": [],
            "rna_score": 0.0,
            "treatability_label": "unavailable",
            "treatability_prob": None,
            "target_vulnerability": None,
        }


# ===================================================================
# Module-level helpers
# ===================================================================

def _infer_subtype_group_fallback(subtype: str) -> str:
    """Infer broad subtype group from PAM50 subtype string."""
    s = str(subtype).strip().lower()
    if s.startswith("lum"):
        return "Luminal"
    if s.startswith("her2"):
        return "Her2"
    if "basal" in s:
        return "Basal"
    if s == "normal":
        return "Normal"
    return "Unknown"


def _safe_round(val: Any, digits: int = 4) -> Any:
    """Round a value safely, returning None for non-numeric."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return val


def _safe_jsonify(val: Any) -> Any:
    """Make a value JSON-serialisable."""
    if isinstance(val, (list, dict, str, int, float, bool, type(None))):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return str(val)
