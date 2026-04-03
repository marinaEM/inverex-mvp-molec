"""
CDS-DB (Cancer Drug Sensitivity DataBase) data ingestion.

CDS-DB (http://cdsdb.ncpsb.org.cn/) hosts patient-derived drug
perturbation signatures: gene expression changes measured in patient
tumour cells after ex-vivo drug treatment.  These are distinct from
the cell-line-derived signatures in LINCS, making them valuable for
comparing cell-line vs. patient-derived drug models.

The database provides:
  - Per-drug, per-patient perturbation gene-expression signatures
  - Cancer type annotations
  - Drug metadata

Approach:
  1. Attempt to query the CDS-DB web interface / API for breast-cancer
     drug perturbation signatures.
  2. If the site is unreachable, use a curated fallback list of GEO
     datasets known to contain patient-derived drug perturbation data.
  3. Download and parse the expression data from GEO.

Note: CDS-DB does not have a documented REST API; we scrape the
website for structured data where possible.
"""
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import DATA_CACHE, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

CDSDB_URL = "http://cdsdb.ncpsb.org.cn/"

# Fallback: known GEO datasets with patient-derived drug perturbation
# signatures in breast cancer.  These datasets measure gene expression
# changes in patient tumour biopsies before/after drug treatment.
FALLBACK_PERTURBATION_DATASETS = [
    {
        "geo_id": "GSE87455",
        "drug": "Letrozole",
        "cancer": "Breast cancer (ER+)",
        "n_patients": 58,
        "description": (
            "Pre/post-treatment biopsies from ER+ breast cancer "
            "patients treated with letrozole.  Paired design allows "
            "computing per-patient drug perturbation signatures."
        ),
        "data_type": "Microarray",
    },
    {
        "geo_id": "GSE20181",
        "drug": "Letrozole",
        "cancer": "Breast cancer (ER+)",
        "n_patients": 55,
        "description": (
            "Pre/post aromatase inhibitor treatment in ER+ breast "
            "cancer. Expression measured on paired biopsies."
        ),
        "data_type": "Microarray",
    },
    {
        "geo_id": "GSE33658",
        "drug": "Anastrozole",
        "cancer": "Breast cancer (ER+)",
        "n_patients": 81,
        "description": (
            "Pre/post-anastrozole paired biopsies in postmenopausal "
            "women with ER+ early breast cancer."
        ),
        "data_type": "Microarray",
    },
    {
        "geo_id": "GSE55374",
        "drug": "Trastuzumab + Lapatinib",
        "cancer": "Breast cancer (HER2+)",
        "n_patients": 51,
        "description": (
            "Pre/post anti-HER2 therapy in HER2+ breast cancer. "
            "Paired biopsies for perturbation signature."
        ),
        "data_type": "Microarray",
    },
]


def fetch_breast_cdsdb_catalog(
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Query CDS-DB for breast-cancer drug perturbation datasets.

    If the CDS-DB site is unreachable, returns a curated fallback
    catalog of GEO datasets with paired pre/post-treatment
    expression data in breast cancer.

    Returns a DataFrame with columns:
        geo_id, drug, cancer, n_patients, description, data_type
    """
    logger.info("Fetching breast-cancer perturbation catalog from CDS-DB ...")

    try:
        resp = requests.get(CDSDB_URL, timeout=timeout)
        resp.raise_for_status()

        # Try to parse the HTML for structured data
        catalog = _scrape_cdsdb_breast_catalog(resp.text)
        if catalog is not None and len(catalog) > 0:
            logger.info(f"CDS-DB: found {len(catalog)} breast cancer entries")
            return catalog

    except Exception as exc:
        logger.warning(f"CDS-DB unreachable ({exc}); using fallback catalog.")

    return _cdsdb_fallback_catalog()


def download_all_breast_cdsdb(
    max_datasets: int = 10,
    dest_dir: Path = DATA_RAW / "cdsdb",
) -> pd.DataFrame:
    """
    Download breast-cancer perturbation datasets from GEO.

    For each dataset with paired pre/post-treatment samples,
    downloads expression and computes per-patient perturbation
    signatures (post - pre).

    Returns the catalog with a ``downloaded`` column.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    catalog = fetch_breast_cdsdb_catalog()
    catalog = catalog.head(max_datasets)

    logger.info(f"Will download {len(catalog)} perturbation datasets")

    downloaded = []
    for _, row in catalog.iterrows():
        geo_id = row["geo_id"]
        ds_dir = dest_dir / geo_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        success = False
        try:
            from src.data_ingestion.ctrdb import download_geo_expression
            expr = download_geo_expression(geo_id, ds_dir)
            if expr is not None:
                success = True
                logger.info(
                    f"Downloaded {geo_id}: {expr.shape[0]} samples x "
                    f"{expr.shape[1]} genes"
                )

                # Try to compute perturbation signatures from paired data
                pert_sigs = _compute_perturbation_signatures(geo_id, ds_dir, expr)
                if pert_sigs is not None:
                    pert_sigs.to_parquet(ds_dir / "perturbation_signatures.parquet")
                    logger.info(
                        f"  Computed {pert_sigs.shape[0]} perturbation signatures"
                    )
            else:
                logger.warning(f"No expression from {geo_id}")
        except Exception as exc:
            logger.error(f"Failed {geo_id}: {exc}")

        downloaded.append(success)

    catalog["downloaded"] = downloaded
    catalog.to_csv(dest_dir / "catalog.csv", index=False)
    return catalog


def load_breast_perturbation_signatures(
    data_dir: Path = DATA_RAW / "cdsdb",
) -> pd.DataFrame:
    """
    Load all computed perturbation signatures.

    Returns a DataFrame with columns:
        patient_id, drug, gene_symbol, log_fc
    """
    if not data_dir.exists():
        logger.warning(f"CDS-DB data directory not found: {data_dir}")
        return pd.DataFrame()

    all_sigs = []
    catalog_path = data_dir / "catalog.csv"
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
    else:
        catalog = fetch_breast_cdsdb_catalog()

    for _, row in catalog.iterrows():
        geo_id = row["geo_id"]
        sig_path = data_dir / geo_id / "perturbation_signatures.parquet"
        if sig_path.exists():
            sigs = pd.read_parquet(sig_path)
            sigs["drug"] = row.get("drug", "unknown")
            sigs["geo_id"] = geo_id
            all_sigs.append(sigs)

    if all_sigs:
        result = pd.concat(all_sigs, ignore_index=True)
        logger.info(
            f"Loaded perturbation signatures: {result.shape[0]} rows, "
            f"{result['drug'].nunique()} drugs, "
            f"{result['patient_id'].nunique() if 'patient_id' in result.columns else '?'} patients"
        )
        return result
    else:
        logger.warning("No perturbation signatures found")
        return pd.DataFrame()


# ── Internal helpers ──────────────────────────────────────────────────

def _cdsdb_fallback_catalog() -> pd.DataFrame:
    """Return fallback catalog of patient perturbation datasets."""
    return pd.DataFrame(FALLBACK_PERTURBATION_DATASETS)


def _scrape_cdsdb_breast_catalog(html: str) -> Optional[pd.DataFrame]:
    """
    Try to extract breast-cancer dataset info from CDS-DB HTML.
    Returns a DataFrame or None if parsing fails.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Look for tables or structured data about datasets
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]

            # Check if this table has relevant columns
            has_drug = any("drug" in h for h in headers)
            has_cancer = any("cancer" in h for h in headers)

            if has_drug or has_cancer:
                data_rows = []
                for row in rows[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all("td")]
                    if cells:
                        data_rows.append(cells)

                if data_rows:
                    df = pd.DataFrame(data_rows, columns=headers[:len(data_rows[0])])
                    # Filter to breast cancer
                    cancer_col = [c for c in df.columns if "cancer" in c]
                    if cancer_col:
                        df = df[
                            df[cancer_col[0]].str.contains(
                                "breast", case=False, na=False
                            )
                        ]
                    if len(df) > 0:
                        return _normalize_cdsdb_catalog(df)

    except Exception as exc:
        logger.debug(f"CDS-DB HTML parsing failed: {exc}")

    return None


def _normalize_cdsdb_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize scraped CDS-DB catalog to standard columns."""
    result = pd.DataFrame()
    for src, dst in [
        ("geo", "geo_id"),
        ("drug", "drug"),
        ("cancer", "cancer"),
        ("sample", "n_patients"),
    ]:
        matching = [c for c in df.columns if src in c.lower()]
        if matching:
            result[dst] = df[matching[0]]

    if "geo_id" not in result.columns:
        return pd.DataFrame()

    if "n_patients" in result.columns:
        result["n_patients"] = pd.to_numeric(result["n_patients"], errors="coerce")

    result["description"] = ""
    result["data_type"] = "unknown"
    return result


def _compute_perturbation_signatures(
    geo_id: str,
    ds_dir: Path,
    expr_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Compute per-patient drug perturbation signatures from paired
    pre/post-treatment samples.

    Looks for sample metadata indicating treatment timepoint,
    pairs pre-treatment with post-treatment samples from the same
    patient, and computes log fold-change.

    Returns a long-form DataFrame with columns:
        patient_id, gene_symbol, log_fc
    """
    import GEOparse

    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(ds_dir), silent=True)
    except Exception as exc:
        logger.warning(f"Could not load {geo_id} for perturbation parsing: {exc}")
        return None

    # Parse sample metadata to find pre/post pairs
    sample_info = []
    for gsm_name, gsm in gse.gsms.items():
        info = {"sample_id": gsm_name}
        chars = gsm.metadata.get("characteristics_ch1", [])
        if isinstance(chars, list):
            for item in chars:
                item_str = str(item).strip()
                if ":" in item_str:
                    key, val = item_str.split(":", 1)
                    info[key.strip().lower()] = val.strip()

        title = gsm.metadata.get("title", [""])[0] if gsm.metadata.get("title") else ""
        info["title"] = title.lower()

        sample_info.append(info)

    meta = pd.DataFrame(sample_info).set_index("sample_id")

    # Identify time-point / treatment columns
    pre_keywords = ["pre", "baseline", "before", "day 0", "untreated", "pre-treatment"]
    post_keywords = ["post", "on-treatment", "after", "day 14", "treated", "post-treatment"]

    pairs = _find_pre_post_pairs(meta, pre_keywords, post_keywords)
    if not pairs:
        logger.info(f"{geo_id}: could not identify pre/post treatment pairs")
        return None

    logger.info(f"{geo_id}: found {len(pairs)} pre/post pairs")

    # Compute fold-changes
    rows = []
    for patient_id, (pre_sample, post_sample) in pairs.items():
        if pre_sample not in expr_df.index or post_sample not in expr_df.index:
            continue

        pre_vals = expr_df.loc[pre_sample]
        post_vals = expr_df.loc[post_sample]

        # Log fold-change (already log-scale for most microarrays)
        lfc = post_vals - pre_vals

        for gene, fc_val in lfc.items():
            if pd.notna(fc_val):
                rows.append({
                    "patient_id": patient_id,
                    "gene_symbol": gene,
                    "log_fc": float(fc_val),
                })

    if rows:
        return pd.DataFrame(rows)
    return None


def _find_pre_post_pairs(
    meta: pd.DataFrame,
    pre_keywords: list[str],
    post_keywords: list[str],
) -> dict[str, tuple[str, str]]:
    """
    Identify pre/post treatment sample pairs from metadata.

    Returns dict: patient_id -> (pre_sample_id, post_sample_id)
    """
    pairs = {}

    # Strategy 1: look for patient ID + timepoint columns
    patient_col = None
    time_col = None

    for col in meta.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ["patient", "subject", "individual", "donor"]):
            patient_col = col
        if any(k in col_lower for k in ["time", "treatment", "visit", "timepoint"]):
            time_col = col

    if patient_col and time_col:
        for pid in meta[patient_col].unique():
            patient_samples = meta[meta[patient_col] == pid]
            pre_samples = []
            post_samples = []
            for sid, row in patient_samples.iterrows():
                val = str(row[time_col]).lower()
                if any(k in val for k in pre_keywords):
                    pre_samples.append(sid)
                elif any(k in val for k in post_keywords):
                    post_samples.append(sid)
            if pre_samples and post_samples:
                pairs[str(pid)] = (pre_samples[0], post_samples[0])
        if pairs:
            return pairs

    # Strategy 2: use sample titles to find pre/post
    for col in ["title"] + list(meta.columns):
        values = meta[col].astype(str).str.lower() if col in meta.columns else pd.Series()
        pre_mask = values.apply(lambda v: any(k in v for k in pre_keywords))
        post_mask = values.apply(lambda v: any(k in v for k in post_keywords))

        if pre_mask.sum() > 0 and post_mask.sum() > 0:
            # Try to match by patient ID in sample names
            pre_ids = meta.index[pre_mask]
            post_ids = meta.index[post_mask]

            # Look for numeric patient IDs in the titles/values
            for pre_sid in pre_ids:
                pre_title = str(values.get(pre_sid, ""))
                # Extract patient number
                nums = re.findall(r"(\d+)", pre_title)
                for num in nums:
                    for post_sid in post_ids:
                        post_title = str(values.get(post_sid, ""))
                        if num in post_title and pre_sid != post_sid:
                            if num not in pairs:
                                pairs[num] = (pre_sid, post_sid)
            if pairs:
                return pairs

    return pairs


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("CDS-DB breast cancer perturbation catalog")
    logger.info("=" * 60)

    catalog = fetch_breast_cdsdb_catalog()
    print(f"\nCatalog: {len(catalog)} datasets")
    print(catalog.to_string())

    out_dir = DATA_RAW / "cdsdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "catalog.csv", index=False)
    print(f"\nSaved to {out_dir / 'catalog.csv'}")
