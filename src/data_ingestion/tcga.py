"""
TCGA-BRCA data ingestion via UCSC Xena.

Strategy:
    UCSC Xena provides easy programmatic access to TCGA processed data.
    We pull:
      - HiSeqV2 gene expression (log2(x+1) RSEM normalized)
      - Clinical metadata (PAM50 subtype, stage, ER/PR/HER2, etc.)
      - Somatic mutations (curated)
      - Gene-level copy number (GISTIC2 thresholded)
"""
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_CACHE,
    XENA_HUB,
    TCGA_EXPRESSION_DATASET,
    TCGA_CLINICAL_DATASET,
    TCGA_MUTATION_DATASET,
    TCGA_CNV_DATASET,
    PAM50_SUBTYPES,
)
from src.data_ingestion.utils import download_file

logger = logging.getLogger(__name__)


def _xena_url(dataset: str) -> str:
    """Construct UCSC Xena download URL for a dataset.

    Tries .gz first; some datasets are only available uncompressed.
    """
    return f"{XENA_HUB}/{dataset}.gz"


def _xena_url_fallback(dataset: str) -> str:
    """Fallback URL without .gz suffix (some Xena datasets lack .gz)."""
    return f"{XENA_HUB}/{dataset}"


def load_tcga_expression(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """
    Load TCGA-BRCA RNA-seq expression matrix.

    Returns DataFrame: rows = samples, columns = genes.
    Values are log2(RSEM + 1) normalized counts.
    """
    cache_path = cache_dir / "tcga_brca_expression.parquet"
    if cache_path.exists():
        logger.info("Loading cached TCGA-BRCA expression...")
        return pd.read_parquet(cache_path)

    gz_path = cache_dir / "tcga_brca_expression.tsv.gz"
    if not gz_path.exists():
        url = _xena_url(TCGA_EXPRESSION_DATASET)
        logger.info("Downloading TCGA-BRCA expression (~100 MB)...")
        download_file(url, gz_path)

    logger.info("Parsing TCGA-BRCA expression matrix...")
    df = pd.read_csv(gz_path, sep="\t", index_col=0)
    # Xena format: rows = genes, columns = samples → transpose
    df = df.T
    df.index.name = "sample_id"

    logger.info(f"Expression matrix: {df.shape[0]} samples × {df.shape[1]} genes")
    df.to_parquet(cache_path)
    return df


def load_tcga_clinical(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """
    Load TCGA-BRCA clinical metadata.

    Key columns: sampleID, PAM50Call_RNAseq, ER/PR/HER2 status,
    pathologic_stage, age_at_diagnosis, etc.
    """
    cache_path = cache_dir / "tcga_brca_clinical.parquet"
    if cache_path.exists():
        logger.info("Loading cached TCGA-BRCA clinical...")
        return pd.read_parquet(cache_path)

    gz_path = cache_dir / "tcga_brca_clinical.tsv.gz"
    raw_path = cache_dir / "tcga_brca_clinical.tsv"
    if not gz_path.exists() and not raw_path.exists():
        url = _xena_url(TCGA_CLINICAL_DATASET)
        logger.info("Downloading TCGA-BRCA clinical metadata...")
        try:
            download_file(url, gz_path)
        except Exception:
            logger.info("Trying uncompressed URL fallback...")
            url = _xena_url_fallback(TCGA_CLINICAL_DATASET)
            download_file(url, raw_path)

    logger.info("Parsing TCGA-BRCA clinical data...")
    read_path = gz_path if gz_path.exists() else raw_path
    df = pd.read_csv(read_path, sep="\t", index_col=0, low_memory=False)
    df.index.name = "sample_id"

    # Standardize PAM50 subtype column name — prefer PAM50Call_RNAseq (text labels)
    if "PAM50Call_RNAseq" in df.columns:
        df["pam50_subtype"] = df["PAM50Call_RNAseq"]
    else:
        pam50_cols = [c for c in df.columns if "pam50" in c.lower()]
        if pam50_cols:
            df["pam50_subtype"] = df[pam50_cols[0]]

    logger.info(f"Clinical data: {df.shape[0]} samples, {df.shape[1]} fields")
    df.to_parquet(cache_path)
    return df


def load_tcga_mutations(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """
    Load TCGA-BRCA somatic mutation calls.

    Returns long-format DataFrame with sample_id, gene, variant info.
    """
    cache_path = cache_dir / "tcga_brca_mutations.parquet"
    if cache_path.exists():
        logger.info("Loading cached TCGA-BRCA mutations...")
        return pd.read_parquet(cache_path)

    gz_path = cache_dir / "tcga_brca_mutations.tsv.gz"
    raw_path = cache_dir / "tcga_brca_mutations.tsv"
    if not gz_path.exists() and not raw_path.exists():
        url = _xena_url(TCGA_MUTATION_DATASET)
        logger.info("Downloading TCGA-BRCA mutations...")
        try:
            download_file(url, gz_path)
        except Exception:
            try:
                logger.info("Trying uncompressed URL fallback...")
                url = _xena_url_fallback(TCGA_MUTATION_DATASET)
                download_file(url, raw_path)
            except Exception as e:
                logger.warning(f"Could not download mutation data from Xena: {e}")

    if gz_path.exists() or raw_path.exists():
        logger.info("Parsing TCGA-BRCA mutations...")
        read_path = gz_path if gz_path.exists() else raw_path
        try:
            df = pd.read_csv(read_path, sep="\t", low_memory=False)
            df.to_parquet(cache_path, index=False)
            logger.info(f"Mutation data: {df.shape[0]} records")
            return df
        except Exception as e:
            logger.warning(f"Could not parse mutation data: {e}")

    # Fallback: fetch key mutation data from cBioPortal API
    logger.info("Trying cBioPortal API for TCGA-BRCA mutations...")
    return _fetch_mutations_cbio(cache_dir)


def _fetch_mutations_cbio(cache_dir: Path) -> pd.DataFrame:
    """Fetch key breast-cancer gene mutations from cBioPortal public API."""
    import requests as _requests

    cache_path = cache_dir / "tcga_brca_mutations.parquet"
    key_genes = [
        "TP53", "PIK3CA", "ERBB2", "GATA3", "CDH1", "MAP3K1",
        "PTEN", "AKT1", "ESR1", "BRCA1", "BRCA2",
    ]
    base = "https://www.cbioportal.org/api"
    study = "brca_tcga"

    all_muts = []
    for gene in key_genes:
        try:
            # Get entrez gene id
            resp = _requests.get(
                f"{base}/genes/{gene}", timeout=15,
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                continue
            entrez_id = resp.json().get("entrezGeneId")

            # Fetch mutations for this gene in TCGA-BRCA
            resp = _requests.get(
                f"{base}/molecular-profiles/{study}_mutations/mutations",
                params={
                    "entrezGeneId": entrez_id,
                    "sampleListId": f"{study}_all",
                },
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code == 200:
                muts = resp.json()
                for m in muts:
                    all_muts.append({
                        "sample": m.get("sampleId", ""),
                        "gene": gene,
                        "mutationType": m.get("mutationType", ""),
                        "proteinChange": m.get("proteinChange", ""),
                    })
                logger.info(f"  {gene}: {len(muts)} mutations")
        except Exception as e:
            logger.warning(f"  {gene}: cBioPortal fetch failed ({e})")

    if all_muts:
        df = pd.DataFrame(all_muts)
        df.to_parquet(cache_path, index=False)
        logger.info(f"cBioPortal mutations: {len(df)} records across {df['gene'].nunique()} genes")
        return df

    logger.warning("Could not fetch mutations from any source.")
    return pd.DataFrame()


def load_tcga_cnv(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """
    Load TCGA-BRCA gene-level copy number (GISTIC2 thresholded).

    Values: -2 (deep del), -1 (shallow del), 0 (neutral),
            1 (low gain), 2 (high amplification)
    """
    cache_path = cache_dir / "tcga_brca_cnv.parquet"
    if cache_path.exists():
        logger.info("Loading cached TCGA-BRCA CNV...")
        return pd.read_parquet(cache_path)

    gz_path = cache_dir / "tcga_brca_cnv.tsv.gz"
    raw_path = cache_dir / "tcga_brca_cnv.tsv"
    if not gz_path.exists() and not raw_path.exists():
        url = _xena_url(TCGA_CNV_DATASET)
        logger.info("Downloading TCGA-BRCA CNV data...")
        try:
            download_file(url, gz_path)
        except Exception:
            try:
                logger.info("Trying uncompressed URL fallback...")
                url = _xena_url_fallback(TCGA_CNV_DATASET)
                download_file(url, raw_path)
            except Exception as e:
                logger.warning(f"Could not download CNV data: {e}")
                return pd.DataFrame()

    logger.info("Parsing TCGA-BRCA CNV matrix...")
    read_path = gz_path if gz_path.exists() else raw_path
    try:
        df = pd.read_csv(read_path, sep="\t", index_col=0)
        df = df.T
        df.index.name = "sample_id"
        df.to_parquet(cache_path)
        logger.info(f"CNV matrix: {df.shape[0]} samples × {df.shape[1]} genes")
        return df
    except Exception as e:
        logger.warning(f"Could not parse CNV data: {e}")
        return pd.DataFrame()


def build_patient_cohort(
    cache_dir: Path = DATA_CACHE,
    require_expression: bool = True,
    require_subtype: bool = False,
) -> pd.DataFrame:
    """
    Build a clean TCGA-BRCA patient cohort with harmonized IDs.

    Returns a metadata DataFrame indexed by sample_id with:
      - pam50_subtype
      - er_status, pr_status, her2_status
      - key mutation flags (TP53, PIK3CA, ERBB2, GATA3, CDH1, MAP3K1)
      - ERBB2 amplification flag from CNV
    """
    cache_path = cache_dir / "tcga_brca_cohort.parquet"
    if cache_path.exists():
        logger.info("Loading cached TCGA-BRCA cohort...")
        return pd.read_parquet(cache_path)

    # Load expression to get sample IDs
    expr = load_tcga_expression(cache_dir)
    sample_ids = set(expr.index)

    # Load clinical
    clinical = load_tcga_clinical(cache_dir)
    clinical_ids = set(clinical.index)

    # Intersect
    common = sorted(sample_ids & clinical_ids)
    logger.info(f"Samples with expression + clinical: {len(common)}")

    cohort = clinical.loc[clinical.index.isin(common)].copy()

    # Extract ER/PR/HER2 status — use nature2012 columns (best coverage)
    status_map = {
        "er_status": "ER_Status_nature2012",
        "pr_status": "PR_Status_nature2012",
        "her2_status": "HER2_Final_Status_nature2012",
    }
    for out_col, src_col in status_map.items():
        if src_col in cohort.columns:
            cohort[out_col] = cohort[src_col]
        else:
            matches = [c for c in cohort.columns if out_col.replace("_", "") in c.lower().replace("_", "")]
            if matches:
                cohort[out_col] = cohort[matches[0]]

    # Add mutation flags for key breast cancer genes
    mutations = load_tcga_mutations(cache_dir)
    key_genes = ["TP53", "PIK3CA", "ERBB2", "GATA3", "CDH1", "MAP3K1",
                 "PTEN", "AKT1", "ESR1", "BRCA1", "BRCA2"]

    if len(mutations) > 0:
        # Identify sample ID column
        sample_col = None
        for c in ["sample", "sampleID", "#sample"]:
            if c in mutations.columns:
                sample_col = c
                break

        gene_col = None
        for c in ["gene", "Hugo_Symbol"]:
            if c in mutations.columns:
                gene_col = c
                break

        if sample_col and gene_col:
            for gene in key_genes:
                mutated_samples = set(
                    mutations.loc[mutations[gene_col] == gene, sample_col]
                )
                cohort[f"mut_{gene}"] = cohort.index.isin(mutated_samples).astype(int)

    # Add ERBB2 amplification from CNV
    cnv = load_tcga_cnv(cache_dir)
    if len(cnv) > 0 and "ERBB2" in cnv.columns:
        erbb2_amp = cnv["ERBB2"] >= 2
        cohort["ERBB2_amp"] = cohort.index.map(
            lambda x: int(erbb2_amp.get(x, False))
        )

    if require_subtype and "pam50_subtype" in cohort.columns:
        cohort = cohort[cohort["pam50_subtype"].isin(PAM50_SUBTYPES)]
        logger.info(f"Filtered to samples with PAM50 subtype: {len(cohort)}")

    cohort.to_parquet(cache_path)
    logger.info(f"Built patient cohort: {len(cohort)} patients")
    return cohort


def compute_patient_signature(
    sample_id: str,
    expression: pd.DataFrame,
    cohort: pd.DataFrame,
    landmark_genes: list[str],
    method: str = "subtype_centroid",
) -> pd.Series:
    """
    Compute a patient's disease signature (log2FC or z-score vs. reference).

    This is the key step that produces the input for the LightGBM model,
    analogous to scTherapy's DEGs between malignant and normal cells.

    Args:
        sample_id: TCGA sample identifier
        expression: Full expression matrix (samples × genes)
        cohort: Patient cohort metadata (with pam50_subtype)
        landmark_genes: List of L1000 landmark gene symbols
        method: "cohort_centroid" or "subtype_centroid"

    Returns:
        Series indexed by gene symbol with log2FC values
    """
    # Filter to landmark genes present in expression data
    available_genes = [g for g in landmark_genes if g in expression.columns]

    patient_expr = expression.loc[sample_id, available_genes]

    if method == "subtype_centroid" and "pam50_subtype" in cohort.columns:
        subtype = cohort.loc[sample_id, "pam50_subtype"]
        if pd.notna(subtype):
            # Use same-subtype samples as reference
            subtype_samples = cohort[
                (cohort["pam50_subtype"] == subtype)
                & (cohort.index != sample_id)
            ].index
            if len(subtype_samples) >= 10:
                ref_expr = expression.loc[
                    expression.index.isin(subtype_samples), available_genes
                ]
                centroid = ref_expr.mean(axis=0)
                std = ref_expr.std(axis=0).replace(0, 1)
                z_scores = (patient_expr - centroid) / std
                return z_scores

    # Fallback: cohort centroid (all other BRCA samples)
    other_samples = [s for s in expression.index if s != sample_id]
    ref_expr = expression.loc[other_samples, available_genes]
    centroid = ref_expr.mean(axis=0)
    std = ref_expr.std(axis=0).replace(0, 1)
    z_scores = (patient_expr - centroid) / std
    return z_scores
