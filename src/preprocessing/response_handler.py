"""
Response endpoint handler for CTR-DB datasets.

Provides configurable harmonization of heterogeneous clinical response
endpoints (pCR/RD, RECIST, survival-derived, pharmacodynamic) into a
unified framework while preserving native label granularity.

Harmonization policies
----------------------
- **strict**: Only clean binary conversions are used (pCR vs RD, CR+PR vs PD).
  SD is excluded.  Survival endpoints require explicit time cutoff.
- **lenient**: Broader binarization (CR+PR vs SD+PD).  Survival endpoints
  use event/no-event as-is.
- **native_only**: No harmonization; return original labels as-is.

All transformations are logged to ``results/label_transformation_log.tsv``.
"""
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# ── Endpoint family definitions ──────────────────────────────────────────

PATHOLOGIC_POS = {"pcr", "rcb-0", "rcb-i", "rcb-0/i", "pathologic complete"}
PATHOLOGIC_NEG = {"rd", "residual disease", "npcr", "ncr", "non-pcr", "rcb-ii", "rcb-iii", "rcb-ii/iii"}

RECIST_POS_STRICT = {"cr", "pr"}
RECIST_NEG_STRICT = {"pd"}
RECIST_EXCLUDED_STRICT = {"sd"}
RECIST_POS_LENIENT = {"cr", "pr"}
RECIST_NEG_LENIENT = {"sd", "pd"}

SURVIVAL_POS = {"no relapse", "no recurrence", "long-term responder", "long-her"}
SURVIVAL_NEG = {"relapse", "recurrence", "control"}

PHARMA_POS = {"responder", "sensitive", "response", "high response"}
PHARMA_NEG = {"nonresponder", "non-responder", "resistant", "acquired-resistant", "low response", "non-response"}


class ResponseHandler:
    """
    Parse and harmonize response labels for CTR-DB datasets.

    Parameters
    ----------
    metadata_path : Path
        Path to the endpoint metadata TSV (built by Phase 2.1).
    policy : str
        One of 'strict', 'lenient', 'native_only'.
    log_path : Path
        Where to write transformation log.
    """

    def __init__(
        self,
        metadata_path: Path = DATA_RAW / ".." / ".." / "data" / "metadata" / "ctrdb_endpoint_metadata.tsv",
        policy: str = "lenient",
        log_path: Path = RESULTS / "label_transformation_log.tsv",
    ):
        self.policy = policy
        self.log_path = log_path
        self._log_rows: list[dict] = []

        # Load endpoint metadata
        if metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path, sep="\t")
        else:
            self.metadata = pd.DataFrame()
            logger.warning(f"Endpoint metadata not found at {metadata_path}")

    # ── Public API ───────────────────────────────────────────────────────

    def parse_native_labels(
        self,
        geo_id: str,
        labels: pd.Series,
    ) -> pd.Series:
        """
        Return labels as stored (binary 0/1).  These are already binarized
        at download time.  This method exists for interface consistency and
        logging.
        """
        n_pos = int(labels.sum())
        n_neg = int((1 - labels).sum())
        self._log(geo_id, "parse_native", "binary_as_stored", n_pos, n_neg, "identity")
        return labels

    def harmonize_labels(
        self,
        geo_id: str,
        labels: pd.Series,
        endpoint_family: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """
        Apply harmonization policy to convert native labels to analysis-ready labels.

        For the current codebase, labels are already binary (0/1) from ingestion.
        This method applies policy-based filtering:
        - strict: excludes datasets with ambiguous binarization
        - lenient: uses all datasets as-is
        - native_only: returns labels unchanged

        Returns None if the dataset should be excluded under the current policy.
        """
        if endpoint_family is None:
            endpoint_family = self.get_endpoint_family(geo_id)

        if self.policy == "native_only":
            self._log(geo_id, "harmonize", endpoint_family,
                      int(labels.sum()), int((1 - labels).sum()), "native_only")
            return labels

        if self.policy == "strict":
            return self._harmonize_strict(geo_id, labels, endpoint_family)

        if self.policy == "lenient":
            return self._harmonize_lenient(geo_id, labels, endpoint_family)

        raise ValueError(f"Unknown policy: {self.policy}")

    def get_endpoint_family(self, geo_id: str) -> str:
        """
        Look up the endpoint family for a dataset from metadata.

        Returns one of: pathologic_response, radiographic, survival,
        pharmacodynamic, unknown.
        """
        if self.metadata.empty:
            return "unknown"

        row = self.metadata[self.metadata["dataset_id"] == geo_id]
        if row.empty:
            return "unknown"

        return str(row.iloc[0].get("endpoint_family", "unknown"))

    def get_endpoint_info(self, geo_id: str) -> dict:
        """Return full endpoint metadata dict for a dataset."""
        if self.metadata.empty:
            return {"endpoint_family": "unknown"}

        row = self.metadata[self.metadata["dataset_id"] == geo_id]
        if row.empty:
            return {"endpoint_family": "unknown"}

        return row.iloc[0].to_dict()

    def get_appropriate_metric(self, endpoint_family: str) -> str:
        """
        Return the recommended primary metric for an endpoint family.

        - pathologic_response, radiographic, pharmacodynamic -> 'auroc'
        - survival -> 'concordance'
        - continuous -> 'spearman'
        """
        if endpoint_family in ("pathologic_response", "radiographic", "pharmacodynamic"):
            return "auroc"
        elif endpoint_family == "survival":
            return "concordance"
        elif endpoint_family == "continuous":
            return "spearman"
        return "auroc"

    def flush_log(self) -> None:
        """Write accumulated log entries to disk."""
        if not self._log_rows:
            return

        df = pd.DataFrame(self._log_rows)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.log_path.exists():
            existing = pd.read_csv(self.log_path, sep="\t")
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(self.log_path, sep="\t", index=False)
        logger.info(f"Label transformation log written to {self.log_path} ({len(df)} entries)")
        self._log_rows = []

    # ── Private methods ──────────────────────────────────────────────────

    def _harmonize_strict(
        self, geo_id: str, labels: pd.Series, endpoint_family: str,
    ) -> Optional[pd.Series]:
        """
        Strict policy:
        - pathologic_response: allow (clean binary)
        - radiographic: allow only if predefined_grouping excludes SD,
          otherwise exclude dataset
        - survival: flag as derived, exclude
        - pharmacodynamic: allow
        - unknown: exclude
        """
        if endpoint_family == "pathologic_response":
            self._log(geo_id, "harmonize_strict", endpoint_family,
                      int(labels.sum()), int((1 - labels).sum()), "pass_through")
            return labels

        if endpoint_family == "radiographic":
            info = self.get_endpoint_info(geo_id)
            predef = str(info.get("predefined_grouping", "")).lower()
            # Under strict: only allow if SD is NOT included in the response group
            if "sd" in predef and "non_response" not in predef.split("sd")[0]:
                self._log(geo_id, "harmonize_strict", endpoint_family,
                          int(labels.sum()), int((1 - labels).sum()),
                          "sd_ambiguous_excluded")
                return None
            self._log(geo_id, "harmonize_strict", endpoint_family,
                      int(labels.sum()), int((1 - labels).sum()), "pass_through")
            return labels

        if endpoint_family == "pharmacodynamic":
            self._log(geo_id, "harmonize_strict", endpoint_family,
                      int(labels.sum()), int((1 - labels).sum()), "pass_through")
            return labels

        if endpoint_family == "survival":
            self._log(geo_id, "harmonize_strict", endpoint_family,
                      int(labels.sum()), int((1 - labels).sum()),
                      "survival_excluded_no_cutoff")
            return None

        # unknown
        self._log(geo_id, "harmonize_strict", endpoint_family,
                  int(labels.sum()), int((1 - labels).sum()), "unknown_excluded")
        return None

    def _harmonize_lenient(
        self, geo_id: str, labels: pd.Series, endpoint_family: str,
    ) -> Optional[pd.Series]:
        """
        Lenient policy:
        - pathologic_response: allow
        - radiographic: allow (CR+PR vs SD+PD)
        - survival: allow (event vs no-event)
        - pharmacodynamic: allow
        - unknown: allow with warning
        """
        if endpoint_family == "unknown":
            logger.warning(f"{geo_id}: unknown endpoint family, using labels as-is (lenient)")

        self._log(geo_id, "harmonize_lenient", endpoint_family,
                  int(labels.sum()), int((1 - labels).sum()), "pass_through")
        return labels

    def _log(
        self,
        geo_id: str,
        operation: str,
        endpoint_family: str,
        n_pos: int,
        n_neg: int,
        action: str,
    ) -> None:
        """Append a log entry."""
        self._log_rows.append({
            "dataset_id": geo_id,
            "operation": operation,
            "endpoint_family": endpoint_family,
            "n_responders": n_pos,
            "n_nonresponders": n_neg,
            "action": action,
            "policy": self.policy,
        })


# ── Convenience functions ────────────────────────────────────────────────

def load_response_handler(
    policy: str = "lenient",
    metadata_path: Optional[Path] = None,
) -> ResponseHandler:
    """
    Factory function to load a ResponseHandler with the endpoint metadata.
    """
    if metadata_path is None:
        from src.config import ROOT
        metadata_path = ROOT / "data" / "metadata" / "ctrdb_endpoint_metadata.tsv"

    return ResponseHandler(
        metadata_path=metadata_path,
        policy=policy,
    )


def classify_endpoint_family(response_grouping: str, predefined_grouping: str) -> str:
    """
    Classify a dataset's endpoint family from catalog metadata strings.

    Returns one of: pathologic_response, radiographic, survival,
    pharmacodynamic, unknown.
    """
    rl = str(response_grouping).lower()
    pl = str(predefined_grouping).lower()

    if any(k in rl or k in pl for k in ["pcr", "rcb", "pathologic"]):
        return "pathologic_response"
    if any(k in rl or k in pl for k in ["cr:", "pr:", "sd:", "pd:", "cr and pr"]):
        return "radiographic"
    if any(k in rl or k in pl for k in ["relapse", "recurrence", "survival"]):
        return "survival"
    if any(k in rl or k in pl for k in ["sensitive", "resistant", "responder", "nonresponder", "long-her"]):
        return "pharmacodynamic"
    if any(k in rl or k in pl for k in ["complete response", "incomplete response"]):
        return "pathologic_response"

    return "unknown"
