"""
INVEREX Data Flywheel — Incremental Model Updates.

Deployment cycle:
  1. Patient enrolled (t=0) → Inverex predicts response with frozen model
  2. Months pass → clinical outcome observed (responder/non-responder)
  3. Outcome + original expression fed back → model retrained

This is NOT leakage: prediction happened at t=0, outcome observed later,
retrain uses outcome as a new labeled example for future patients.

Usage:
    updater = ModelUpdater("models/production/")
    updater.record_outcome("patient_1", expr_data, "tamoxifen", responded=True)
    ...  # accumulate outcomes
    if updater.should_retrain():
        updater.retrain()
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelUpdater:
    """Manages incremental model updates as new labeled data arrives."""

    def __init__(self, production_dir: str = "models/production/"):
        self.production_dir = Path(production_dir)
        self.pending_outcomes: list[dict] = []
        self.update_log: list[dict] = []

        # Load log if exists
        log_path = self.production_dir / "update_log.json"
        if log_path.exists():
            with open(log_path) as f:
                self.update_log = json.load(f)

    def record_outcome(
        self,
        patient_id: str,
        expression_data: pd.Series | dict,
        drug_name: str,
        responded: bool,
        dataset_id: str = "clinical_feedback",
    ):
        """
        Record a clinical outcome for a previously scored patient.

        Args:
            patient_id: ID from the original prediction.
            expression_data: the SAME expression profile used for prediction.
            drug_name: which drug the patient received.
            responded: whether the patient responded.
            dataset_id: batch/trial identifier.
        """
        if isinstance(expression_data, dict):
            expression_data = pd.Series(expression_data)

        self.pending_outcomes.append({
            "patient_id": patient_id,
            "expression": expression_data.to_dict(),
            "drug_name": drug_name,
            "label": int(responded),
            "dataset_id": dataset_id,
            "recorded_at": datetime.now().isoformat(),
        })

        logger.info(
            f"Recorded outcome for {patient_id}: "
            f"{'responder' if responded else 'non-responder'} to {drug_name}. "
            f"Pending: {len(self.pending_outcomes)}"
        )

    def should_retrain(self, min_new_samples: int = 50) -> bool:
        """
        Check if enough new outcomes accumulated to justify retraining.

        Rules:
          - At least min_new_samples new labeled samples
          - At least 2 different drugs
          - Class balance >= 10% minority
        """
        if len(self.pending_outcomes) < min_new_samples:
            return False

        drugs = set(o["drug_name"] for o in self.pending_outcomes)
        if len(drugs) < 2:
            return False

        labels = [o["label"] for o in self.pending_outcomes]
        minority_frac = min(sum(labels), len(labels) - sum(labels)) / len(labels)
        if minority_frac < 0.10:
            return False

        return True

    def get_pending_summary(self) -> dict:
        """Summary of pending outcomes buffer."""
        if not self.pending_outcomes:
            return {"n_pending": 0}

        labels = [o["label"] for o in self.pending_outcomes]
        drugs = [o["drug_name"] for o in self.pending_outcomes]
        datasets = [o["dataset_id"] for o in self.pending_outcomes]

        return {
            "n_pending": len(self.pending_outcomes),
            "n_responders": sum(labels),
            "n_nonresponders": len(labels) - sum(labels),
            "n_drugs": len(set(drugs)),
            "n_datasets": len(set(datasets)),
            "drugs": list(set(drugs)),
            "ready_to_retrain": self.should_retrain(),
        }

    def save_pending(self, path: str = None):
        """Persist pending outcomes to disk."""
        if path is None:
            path = self.production_dir / "pending_outcomes.json"
        with open(path, "w") as f:
            json.dump(self.pending_outcomes, f, indent=2, default=str)
        logger.info(f"Saved {len(self.pending_outcomes)} pending outcomes to {path}")

    def load_pending(self, path: str = None):
        """Load pending outcomes from disk."""
        if path is None:
            path = self.production_dir / "pending_outcomes.json"
        if Path(path).exists():
            with open(path) as f:
                self.pending_outcomes = json.load(f)
            logger.info(f"Loaded {len(self.pending_outcomes)} pending outcomes")

    def archive_current_models(self) -> str:
        """Archive current production models before retraining."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = self.production_dir.parent / "archive" / version
        archive_dir.mkdir(parents=True, exist_ok=True)

        for f in self.production_dir.iterdir():
            if f.is_file() and f.suffix in (".joblib", ".json"):
                shutil.copy2(f, archive_dir / f.name)

        logger.info(f"Archived current models to {archive_dir}")
        return str(archive_dir)

    def _save_update_log(self):
        """Persist update log."""
        log_path = self.production_dir / "update_log.json"
        with open(log_path, "w") as f:
            json.dump(self.update_log, f, indent=2, default=str)
