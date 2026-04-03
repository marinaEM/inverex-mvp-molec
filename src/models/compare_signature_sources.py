"""
Compare cell-line (LINCS) vs. patient-derived (CDS-DB) drug signatures
for predicting patient drug response.

For drugs where we have both:
  - LINCS cell-line perturbation signatures (from L1000)
  - CDS-DB patient perturbation signatures (from ex-vivo treated biopsies)

We compute reversal scores against patient disease signatures and compare
AUC for predicting clinical response.

This helps answer: do patient-derived signatures better predict patient
outcomes than cell-line signatures?
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.config import DATA_CACHE, DATA_RAW, RESULTS
from src.data_ingestion.cdsdb import load_breast_perturbation_signatures
from src.data_ingestion.ctrdb import load_all_breast_ctrdb
from src.data_ingestion.lincs import (
    build_breast_signature_matrix,
    load_landmark_genes,
)
from src.models.predict_patients import compute_reversal_score

logger = logging.getLogger(__name__)

# Drug name mappings (LINCS name -> canonical name)
DRUG_NAME_MAP = {
    "letrozole": "letrozole",
    "anastrozole": "anastrozole",
    "tamoxifen": "tamoxifen",
    "trastuzumab": "trastuzumab",
    "lapatinib": "lapatinib",
    "paclitaxel": "paclitaxel",
    "docetaxel": "docetaxel",
    "doxorubicin": "doxorubicin",
    "cyclophosphamide": "cyclophosphamide",
    "fluorouracil": "fluorouracil",
    "5-fluorouracil": "fluorouracil",
}


def compare_signature_sources(
    ctrdb_dir: Path = DATA_RAW / "ctrdb",
    cdsdb_dir: Path = DATA_RAW / "cdsdb",
    output_path: Path = RESULTS / "lincs_vs_cdsdb_comparison.csv",
) -> pd.DataFrame:
    """
    Compare LINCS cell-line vs CDS-DB patient drug signatures for
    predicting clinical response.

    Steps:
      1. Load LINCS breast-cancer signatures.
      2. Load CDS-DB patient perturbation signatures.
      3. For each drug present in both sources, and for each CTR-DB
         validation dataset involving that drug:
         - Compute reversal scores using LINCS signatures.
         - Compute reversal scores using CDS-DB signatures.
         - Compare AUC for responder/non-responder discrimination.

    Returns a DataFrame with columns:
        drug, geo_id, source, n_patients, auc, wilcoxon_p, mean_diff
    """
    # Load landmark genes
    landmark_df = load_landmark_genes()
    landmark_genes = set(landmark_df["gene_symbol"].tolist())

    # Load LINCS signatures
    lincs_sigs = _load_lincs_drug_signatures(landmark_genes)
    logger.info(
        f"LINCS: {len(lincs_sigs)} drug signatures loaded"
    )

    # Load CDS-DB signatures
    cdsdb_sigs = _load_cdsdb_drug_signatures(cdsdb_dir, landmark_genes)
    logger.info(
        f"CDS-DB: {len(cdsdb_sigs)} drug signatures loaded"
    )

    # Find overlapping drugs
    lincs_drugs = set(lincs_sigs.keys())
    cdsdb_drugs = set(cdsdb_sigs.keys())
    common_drugs = lincs_drugs & cdsdb_drugs

    logger.info(
        f"LINCS drugs: {len(lincs_drugs)}, CDS-DB drugs: {len(cdsdb_drugs)}, "
        f"Common: {len(common_drugs)}"
    )

    if not common_drugs:
        logger.warning(
            "No overlapping drugs between LINCS and CDS-DB. "
            "Will compare LINCS-only predictions on available datasets."
        )
        # Even without common drugs, we can evaluate LINCS signatures
        # against CTR-DB datasets where drugs overlap with LINCS
        common_drugs = lincs_drugs  # evaluate LINCS on all its drugs

    # Load CTR-DB validation datasets
    ctrdb_datasets = load_all_breast_ctrdb(ctrdb_dir)
    if not ctrdb_datasets:
        logger.error("No CTR-DB datasets loaded")
        return pd.DataFrame()

    # Load catalog for drug mapping
    catalog_path = ctrdb_dir / "catalog.csv"
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        geo_to_drug = {}
        for _, row in catalog.iterrows():
            drug_name = _normalize_drug_name(str(row["drug"]))
            geo_to_drug[row["geo_source"]] = drug_name
    else:
        geo_to_drug = {}

    # Compare
    results = []
    for geo_id, (expr, labels) in ctrdb_datasets.items():
        dataset_drug = geo_to_drug.get(geo_id, "")

        # Compute patient disease signatures (z-score across cohort)
        available_landmark = [g for g in landmark_genes if g in expr.columns]
        if len(available_landmark) < 50:
            continue

        expr_z = _compute_cohort_zscore(expr, available_landmark)

        # Evaluate LINCS signatures
        for drug_key, lincs_sig in lincs_sigs.items():
            if drug_key not in dataset_drug and dataset_drug not in drug_key:
                continue  # Only evaluate relevant drugs

            result = _evaluate_signature(
                geo_id=geo_id,
                drug=drug_key,
                source="LINCS",
                drug_signature=lincs_sig,
                patient_expr_z=expr_z,
                labels=labels,
            )
            if result:
                results.append(result)

        # Evaluate CDS-DB signatures
        for drug_key, cdsdb_sig in cdsdb_sigs.items():
            if drug_key not in dataset_drug and dataset_drug not in drug_key:
                continue

            result = _evaluate_signature(
                geo_id=geo_id,
                drug=drug_key,
                source="CDS-DB",
                drug_signature=cdsdb_sig,
                patient_expr_z=expr_z,
                labels=labels,
            )
            if result:
                results.append(result)

    if not results:
        logger.warning(
            "No drug-dataset matches found for comparison. "
            "This is expected if downloaded datasets have different drugs."
        )
        # Generate a summary comparison with whatever we have
        return _generate_available_comparison(
            lincs_sigs, cdsdb_sigs, ctrdb_datasets, geo_to_drug, landmark_genes
        )

    results_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Comparison saved to {output_path}")
    return results_df


def _normalize_drug_name(name: str) -> str:
    """Normalize drug name to lowercase canonical form."""
    name = name.lower().strip()
    # Extract individual drug names from combinations
    for sep in ["+", "/", " and ", ",", "("]:
        name = name.split(sep)[0].strip()
    return DRUG_NAME_MAP.get(name, name)


def _load_lincs_drug_signatures(
    landmark_genes: set[str],
) -> dict[str, pd.Series]:
    """
    Load LINCS breast-cancer drug signatures, averaged per drug.

    Returns dict: drug_name -> Series(gene_symbol -> z-score)
    """
    try:
        sigs = build_breast_signature_matrix()
        if sigs.empty:
            logger.warning("No LINCS signatures available")
            return {}

        gene_cols = [
            c for c in sigs.columns
            if c not in {"sig_id", "pert_id", "pert_iname", "cell_id",
                         "dose_um", "pert_idose"}
            and c in landmark_genes
        ]

        if not gene_cols:
            logger.warning("No landmark genes in LINCS signatures")
            return {}

        # Average per drug
        drug_sigs = {}
        for drug, group in sigs.groupby("pert_iname"):
            mean_sig = group[gene_cols].mean()
            drug_sigs[drug.lower()] = mean_sig

        return drug_sigs

    except Exception as exc:
        logger.warning(f"Could not load LINCS signatures: {exc}")
        return {}


def _load_cdsdb_drug_signatures(
    cdsdb_dir: Path,
    landmark_genes: set[str],
) -> dict[str, pd.Series]:
    """
    Load CDS-DB patient perturbation signatures, averaged per drug.

    Returns dict: drug_name -> Series(gene_symbol -> mean_log_fc)
    """
    sigs = load_breast_perturbation_signatures(cdsdb_dir)
    if sigs.empty:
        return {}

    drug_sigs = {}
    for drug, group in sigs.groupby("drug"):
        if "gene_symbol" not in group.columns or "log_fc" not in group.columns:
            continue

        # Filter to landmark genes and average across patients
        group_lm = group[group["gene_symbol"].isin(landmark_genes)]
        if group_lm.empty:
            continue

        mean_sig = group_lm.groupby("gene_symbol")["log_fc"].mean()
        drug_sigs[drug.lower()] = mean_sig

    return drug_sigs


def _compute_cohort_zscore(
    expr: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    """Z-score expression across the cohort for the given genes."""
    subset = expr[genes].copy()
    mean = subset.mean(axis=0)
    std = subset.std(axis=0).replace(0, 1)
    return (subset - mean) / std


def _evaluate_signature(
    geo_id: str,
    drug: str,
    source: str,
    drug_signature: pd.Series,
    patient_expr_z: pd.DataFrame,
    labels: pd.Series,
) -> Optional[dict]:
    """
    Evaluate a drug signature's ability to discriminate responders
    from non-responders using reversal scores.
    """
    common = patient_expr_z.index.intersection(labels.index)
    if len(common) < 10:
        return None

    expr_aligned = patient_expr_z.loc[common]
    labels_aligned = labels.loc[common]

    n_resp = int(labels_aligned.sum())
    n_nonresp = len(labels_aligned) - n_resp
    if n_resp < 3 or n_nonresp < 3:
        return None

    # Compute reversal score for each patient
    reversal_scores = []
    for sample_id in common:
        patient_sig = expr_aligned.loc[sample_id]
        rev_score = compute_reversal_score(patient_sig, drug_signature)
        reversal_scores.append(rev_score)

    scores = pd.Series(reversal_scores, index=common)

    resp_scores = scores[labels_aligned == 1]
    nonresp_scores = scores[labels_aligned == 0]

    try:
        auc = roc_auc_score(labels_aligned, scores)
    except ValueError:
        auc = 0.5

    try:
        _, wilcoxon_p = stats.mannwhitneyu(
            resp_scores, nonresp_scores, alternative="two-sided"
        )
    except ValueError:
        wilcoxon_p = 1.0

    return {
        "drug": drug,
        "geo_id": geo_id,
        "source": source,
        "n_patients": len(common),
        "n_responders": n_resp,
        "n_nonresponders": n_nonresp,
        "auc": round(auc, 4),
        "wilcoxon_p": round(wilcoxon_p, 6),
        "mean_score_responders": round(resp_scores.mean(), 4),
        "mean_score_nonresponders": round(nonresp_scores.mean(), 4),
        "mean_diff": round(resp_scores.mean() - nonresp_scores.mean(), 4),
    }


def _generate_available_comparison(
    lincs_sigs: dict,
    cdsdb_sigs: dict,
    ctrdb_datasets: dict,
    geo_to_drug: dict,
    landmark_genes: set[str],
) -> pd.DataFrame:
    """
    When no drug-dataset matches exist, generate a summary of what
    signatures and datasets are available for future comparison.
    """
    rows = []
    rows.append({
        "drug": "SUMMARY",
        "geo_id": "N/A",
        "source": "LINCS",
        "n_patients": 0,
        "n_responders": 0,
        "n_nonresponders": 0,
        "auc": np.nan,
        "wilcoxon_p": np.nan,
        "mean_score_responders": np.nan,
        "mean_score_nonresponders": np.nan,
        "mean_diff": np.nan,
        "note": f"{len(lincs_sigs)} LINCS drugs available: {', '.join(list(lincs_sigs.keys())[:10])}",
    })
    rows.append({
        "drug": "SUMMARY",
        "geo_id": "N/A",
        "source": "CDS-DB",
        "n_patients": 0,
        "n_responders": 0,
        "n_nonresponders": 0,
        "auc": np.nan,
        "wilcoxon_p": np.nan,
        "mean_score_responders": np.nan,
        "mean_score_nonresponders": np.nan,
        "mean_diff": np.nan,
        "note": f"{len(cdsdb_sigs)} CDS-DB drugs available: {', '.join(list(cdsdb_sigs.keys())[:10])}",
    })
    for geo_id, drug in geo_to_drug.items():
        rows.append({
            "drug": drug,
            "geo_id": geo_id,
            "source": "CTR-DB dataset",
            "n_patients": len(ctrdb_datasets.get(geo_id, (pd.DataFrame(), pd.Series()))[1]),
            "n_responders": 0,
            "n_nonresponders": 0,
            "auc": np.nan,
            "wilcoxon_p": np.nan,
            "mean_score_responders": np.nan,
            "mean_score_nonresponders": np.nan,
            "mean_diff": np.nan,
            "note": "Available for validation",
        })

    df = pd.DataFrame(rows)
    output_path = RESULTS / "lincs_vs_cdsdb_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    results = compare_signature_sources()
    if len(results) > 0:
        print("\n" + "=" * 70)
        print("LINCS vs CDS-DB COMPARISON")
        print("=" * 70)
        print(results.to_string(index=False))
