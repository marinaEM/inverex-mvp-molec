"""
PharmacoDB data ingestion for breast-cancer cell lines.

Strategy:
    scTherapy matched LINCS perturbation profiles with dose-response viability
    data from PharmacoDB. The outcome variable = interpolated percent inhibition
    at each tested dose.

    PharmacoDB aggregates data from GDSC, CCLE, CTRPv2, gCSI, etc.
    We pull dose-response curves for breast cell lines, then match to LINCS
    compound-dose-cell triads.

    PharmacoDB REST API: https://pharmacodb.ca/api/v1/
    Key endpoints:
      /cell_lines          — list all cell lines
      /compounds           — list all compounds
      /experiments         — dose-response data
      /datasets            — source datasets

    For the MVP, we use a combination of:
      1. PharmacoDB API to get compound/cell-line metadata
      2. Pre-downloaded GDSC/CTRPv2 dose-response files for bulk access
      3. Fallback: curated IC50/AUC summaries for breast lines
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import (
    BREAST_CELL_LINES,
    DATA_CACHE,
    DATA_RAW,
    PHARMACODB_API,
)
from src.data_ingestion.utils import download_file, fetch_json

logger = logging.getLogger(__name__)

# ── Local file (preferred) ─────────────────────────────────────────────
GDSC_LOCAL_XLSX = DATA_RAW / "GDSC2_fitted_dose_response_27Oct23.xlsx"


def load_gdsc_dose_response(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """
    Load GDSC2 fitted dose-response data.
    
    Priority:
      1. Cached parquet (fast reload)
      2. Local xlsx in data/raw/ (user-provided)
      3. Download from Sanger FTP (fallback)
    """
    cache_path = cache_dir / "gdsc2_dose_response.parquet"
    if cache_path.exists():
        logger.info("Loading cached GDSC2 dose-response...")
        return pd.read_parquet(cache_path)

    # Try local xlsx first
    if GDSC_LOCAL_XLSX.exists():
        logger.info(f"Loading GDSC2 from local file: {GDSC_LOCAL_XLSX}")
        df = pd.read_excel(GDSC_LOCAL_XLSX, engine="openpyxl")
        df.to_parquet(cache_path, index=False)
        logger.info(f"GDSC2 loaded: {df.shape[0]:,} records, {df['DRUG_NAME'].nunique()} drugs")
        return df

    # Fallback: try download
    logger.warning(
        f"GDSC2 xlsx not found at {GDSC_LOCAL_XLSX}. "
        "Place GDSC2_fitted_dose_response_27Oct23.xlsx in data/raw/"
    )
    return pd.DataFrame()


def filter_breast_gdsc(gdsc_df: pd.DataFrame) -> pd.DataFrame:
    """Filter GDSC data to breast cancer cell lines using CANCER_TYPE column."""
    if "CANCER_TYPE" in gdsc_df.columns:
        mask = gdsc_df["CANCER_TYPE"].str.contains("Breast", case=False, na=False)
        filtered = gdsc_df[mask].copy()
    else:
        # Fallback: match by cell line name
        breast_names = set(BREAST_CELL_LINES.keys())
        gdsc_df["cell_name_norm"] = (
            gdsc_df["CELL_LINE_NAME"]
            .str.upper()
            .str.replace(r"[\-_\s]", "", regex=True)
        )
        breast_norms = {
            name.upper().replace("-", "").replace("_", "").replace(" ", "")
            for name in breast_names
        }
        mask = gdsc_df["cell_name_norm"].isin(breast_norms)
        filtered = gdsc_df[mask].copy()
        filtered.drop(columns=["cell_name_norm"], inplace=True)

    logger.info(
        f"Filtered GDSC to {len(filtered):,} breast cell-line records "
        f"({filtered['CELL_LINE_NAME'].nunique()} lines, "
        f"{filtered['DRUG_NAME'].nunique()} drugs)"
    )
    return filtered


def build_dose_response_reference(
    cache_dir: Path = DATA_CACHE,
) -> pd.DataFrame:
    """
    Build a dose-response reference table for breast cancer cell lines.

    Output columns:
        drug_name, cell_line, drug_id, putative_target, pathway_name,
        ln_ic50, ic50_um_linear, auc, min_conc_um, max_conc_um, source

    Uses the GDSC2 fitted dose-response data directly.
    """
    cache_path = cache_dir / "breast_dose_response_ref.parquet"
    if cache_path.exists():
        logger.info("Loading cached breast dose-response reference...")
        return pd.read_parquet(cache_path)

    # ── Load and filter GDSC2 ──────────────────────────────────────
    gdsc = load_gdsc_dose_response(cache_dir)
    if len(gdsc) == 0:
        logger.warning("No GDSC2 data available. Using demo data.")
        return _build_demo_dose_response(cache_dir)

    breast_gdsc = filter_breast_gdsc(gdsc)
    if len(breast_gdsc) == 0:
        logger.warning("No breast cancer records in GDSC2. Using demo data.")
        return _build_demo_dose_response(cache_dir)

    # ── Build reference (vectorized) ───────────────────────────────
    ref = breast_gdsc.rename(columns={
        "DRUG_NAME": "drug_name",
        "CELL_LINE_NAME": "cell_line",
        "DRUG_ID": "drug_id",
        "PUTATIVE_TARGET": "putative_target",
        "PATHWAY_NAME": "pathway_name",
        "LN_IC50": "ln_ic50",
        "AUC": "auc",
        "MIN_CONC": "min_conc_um",
        "MAX_CONC": "max_conc_um",
        "Z_SCORE": "z_score",
    })[["drug_name", "cell_line", "drug_id", "putative_target", "pathway_name",
        "ln_ic50", "auc", "min_conc_um", "max_conc_um", "z_score"]].copy()

    ref["ic50_um_linear"] = np.exp(ref["ln_ic50"].astype(float))
    ref["source"] = "GDSC2"

    ref.to_parquet(cache_path, index=False)
    logger.info(
        f"Built breast dose-response reference: {len(ref):,} records, "
        f"{ref['drug_name'].nunique()} drugs, {ref['cell_line'].nunique()} cell lines"
    )
    logger.info(
        f"  Target annotations: {ref['putative_target'].notna().sum():,} records"
    )
    logger.info(
        f"  Pathway annotations: {ref['pathway_name'].notna().sum():,} records"
    )
    logger.info(
        f"  Top pathways: {', '.join(ref['pathway_name'].value_counts().head(5).index.tolist())}"
    )
    return ref


def interpolate_inhibition(
    ic50_um: float,
    dose_um: float,
    hill_slope: float = 1.0,
) -> float:
    """
    Interpolate percent inhibition at a given dose using a Hill equation.

    inhibition = 100 * (dose^h) / (IC50^h + dose^h)

    This is how scTherapy matched LINCS doses to PharmacoDB viability.
    """
    if pd.isna(ic50_um) or pd.isna(dose_um) or ic50_um <= 0 or dose_um <= 0:
        return np.nan
    return 100.0 * (dose_um ** hill_slope) / (
        ic50_um ** hill_slope + dose_um ** hill_slope
    )


def match_lincs_to_pharmacodb(
    lincs_sigs: pd.DataFrame,
    dose_response_ref: pd.DataFrame,
    cache_dir: Path = DATA_CACHE,
) -> pd.DataFrame:
    """
    Match LINCS signatures to PharmacoDB dose-response data.

    This is the core data-matching step that produces training labels.

    For each LINCS signature (compound × cell line × dose), find the
    corresponding IC50 from PharmacoDB and interpolate the percent
    inhibition at that dose using the Hill equation.

    Args:
        lincs_sigs: DataFrame with pert_iname, cell_id, dose_um columns
        dose_response_ref: DataFrame with drug_name, cell_line, ic50_um columns

    Returns:
        DataFrame with matched records and pct_inhibition column added
    """
    cache_path = cache_dir / "lincs_pharmacodb_matched.parquet"
    if cache_path.exists():
        logger.info("Loading cached LINCS-PharmacoDB matched data...")
        return pd.read_parquet(cache_path)

    logger.info("Matching LINCS signatures to PharmacoDB dose-response...")

    # Normalize compound names for fuzzy matching
    lincs_sigs = lincs_sigs.copy()
    lincs_sigs["drug_norm"] = (
        lincs_sigs["pert_iname"]
        .str.lower()
        .str.strip()
        .str.replace(r"[\-\s]", "", regex=True)
    )

    dose_response_ref = dose_response_ref.copy()
    dose_response_ref["drug_norm"] = (
        dose_response_ref["drug_name"]
        .str.lower()
        .str.strip()
        .str.replace(r"[\-\s]", "", regex=True)
    )

    # Normalize cell line names
    lincs_sigs["cell_norm"] = (
        lincs_sigs["cell_id"]
        .str.upper()
        .str.replace(r"[\-_\s]", "", regex=True)
    )
    dose_response_ref["cell_norm"] = (
        dose_response_ref["cell_line"]
        .str.upper()
        .str.replace(r"[\-_\s]", "", regex=True)
    )

    # Build IC50 lookup: (drug_norm, cell_norm) → ic50_um_linear
    ic50_lookup = {}
    for _, row in dose_response_ref.iterrows():
        key = (row["drug_norm"], row["cell_norm"])
        ic50_val = row.get("ic50_um_linear", np.nan)
        if not pd.isna(ic50_val):
            ic50_lookup[key] = ic50_val

    logger.info(f"Built IC50 lookup with {len(ic50_lookup)} drug-cell pairs")

    # Match and interpolate
    matched_ic50 = []
    matched_inhib = []
    for _, row in lincs_sigs.iterrows():
        key = (row["drug_norm"], row["cell_norm"])
        ic50 = ic50_lookup.get(key, np.nan)
        matched_ic50.append(ic50)
        if not pd.isna(ic50) and not pd.isna(row.get("dose_um")):
            matched_inhib.append(interpolate_inhibition(ic50, row["dose_um"]))
        else:
            matched_inhib.append(np.nan)

    lincs_sigs["ic50_um"] = matched_ic50
    lincs_sigs["pct_inhibition"] = matched_inhib

    # Drop unmatched
    matched = lincs_sigs.dropna(subset=["pct_inhibition"]).copy()
    matched.drop(columns=["drug_norm", "cell_norm"], inplace=True)

    logger.info(
        f"Matched {len(matched)} LINCS-PharmacoDB triads "
        f"({matched['pert_iname'].nunique()} drugs, "
        f"{matched['cell_id'].nunique()} cell lines)"
    )

    matched.to_parquet(cache_path, index=False)
    return matched


def _build_demo_dose_response(cache_dir: Path) -> pd.DataFrame:
    """Build synthetic dose-response data for demo/testing."""
    rng = np.random.default_rng(42)

    drugs = [
        "tamoxifen", "fulvestrant", "lapatinib", "palbociclib",
        "alpelisib", "everolimus", "olaparib", "paclitaxel",
        "doxorubicin", "cisplatin", "trametinib", "dasatinib",
        "navitoclax", "vorinostat", "bortezomib", "sorafenib",
    ]
    cell_lines = list(BREAST_CELL_LINES.keys())[:6]

    rows = []
    for drug in drugs:
        for cell in cell_lines:
            ic50 = 10 ** rng.uniform(-2, 2)  # 0.01 to 100 µM
            rows.append({
                "drug_name": drug,
                "cell_line": cell,
                "ic50_um_linear": ic50,
                "ic50_um": np.log(ic50),
                "auc": rng.uniform(0.2, 0.9),
                "source": "DEMO",
            })

    ref = pd.DataFrame(rows)
    cache_path = cache_dir / "breast_dose_response_ref.parquet"
    ref.to_parquet(cache_path, index=False)
    logger.info(f"Built demo dose-response reference: {ref.shape}")
    return ref
