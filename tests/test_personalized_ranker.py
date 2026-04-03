import unittest

import pandas as pd

from src.ranking.drug_metadata import build_drug_metadata
from src.ranking.personalized_ranker import compute_reversal_score


class DrugMetadataTests(unittest.TestCase):
    def test_tool_compounds_are_flagged(self) -> None:
        metadata = build_drug_metadata(["MG-132", "lapatinib"]).set_index("drug_name")

        self.assertEqual(int(metadata.loc["MG-132", "tool_compound"]), 1)
        self.assertEqual(metadata.loc["lapatinib", "clinical_status"], "approved_breast")
        self.assertEqual(int(metadata.loc["lapatinib", "her2_relevance"]), 3)


class ReversalScoreTests(unittest.TestCase):
    def test_opposing_signatures_score_highly(self) -> None:
        patient = pd.Series(
            {f"GENE_{idx}": float(idx) for idx in range(1, 13)},
            dtype=float,
        )
        drug = -patient

        score = compute_reversal_score(patient, drug)
        self.assertGreater(score, 0.95)

    def test_matching_signatures_score_poorly(self) -> None:
        patient = pd.Series(
            {f"GENE_{idx}": float(idx) for idx in range(1, 13)},
            dtype=float,
        )
        drug = patient.copy()

        score = compute_reversal_score(patient, drug)
        self.assertLess(score, -0.95)


if __name__ == "__main__":
    unittest.main()
