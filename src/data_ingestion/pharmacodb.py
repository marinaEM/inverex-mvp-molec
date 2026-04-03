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
import time
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


# ---------------------------------------------------------------------------
# Drug synonym table: maps normalized GDSC name -> normalized LINCS pert_iname
# ---------------------------------------------------------------------------
# These are curated mappings between GDSC2 drug names and their LINCS L1000
# equivalents.  GDSC tends to use INN/generic names while LINCS often uses
# research codes (and vice-versa).  The keys and values are the *original*
# GDSC / LINCS names (before normalization) so the mapping is human-readable;
# the helper function _build_drug_alias_map() normalizes both sides.

_DRUG_SYNONYM_TABLE: dict[str, str] = {
    # ── prefix-stripped generics ──────────────────────────────────────
    "5-Fluorouracil":       "fluorouracil",
    "5-azacytidine":        "azacitidine",
    # ── INN generic <-> research code ─────────────────────────────────
    "Rapamycin":            "sirolimus",
    "Pictilisib":           "GDC-0941",
    "Dactolisib":           "NVP-BEZ235",
    "Luminespib":           "NVP-AUY922",
    "Daporinad":            "FK-866",
    "Obatoclax Mesylate":   "obatoclax",
    "Nutlin-3a (-)":        "nutlin-3",
    # ── AZD / AstraZeneca compounds ───────────────────────────────────
    "Sapitinib":            "AZD8931",
    "Osimertinib":          "AZD9291",
    "AZD5363":              "capivasertib",
    "Savolitinib":          "AZD6094",
    # ── GSK compounds ─────────────────────────────────────────────────
    "Afuresertib":          "GSK2110183",
    "Uprosertib":           "GSK2141795",
    # ── BET / bromodomain inhibitors ──────────────────────────────────
    "I-BET-762":            "GSK525762",
    "OTX015":               "birabresib",
    "JQ1":                  "JQ-1",
    # ── MEK / ERK inhibitors ──────────────────────────────────────────
    "Refametinib":          "RDEA119",
    "Ulixertinib":          "BVD-523",
    # ── CDK inhibitors ────────────────────────────────────────────────
    "Dinaciclib":           "SCH-727965",
    "Ribociclib":           "LEE011",
    # ── PARP inhibitors ───────────────────────────────────────────────
    "Niraparib":            "MK-4827",
    "Talazoparib":          "BMN-673",
    # ── PI3K / AKT / mTOR ────────────────────────────────────────────
    "Alpelisib":            "BYL719",
    "Taselisib":            "GDC-0032",
    "Ipatasertib":          "GDC-0068",
    "AZD2014":              "vistusertib",
    # ── Hedgehog pathway ──────────────────────────────────────────────
    "Vismodegib":           "GDC-0449",
    # ── BCL-2 family ──────────────────────────────────────────────────
    "Venetoclax":           "ABT-199",
    "Navitoclax":           "ABT-263",
    # ── Epigenetic modulators ─────────────────────────────────────────
    "Romidepsin":           "FK228",
    "Vorinostat":           "SAHA",
    "EPZ004777":            "EPZ-004777",
    "EPZ5676":              "EPZ-5676",
    "SGC0946":              "SGC-0946",
    "GSK-LSD1":             "GSK-LSD1-2HCl",
    "GSK343":               "GSK-343",
    "RVX-208":              "RVX208",
    # ── Wnt pathway ───────────────────────────────────────────────────
    "LGK974":               "LGK-974",
    "Wnt-C59":              "Wnt-C59",
    # ── DNA damage / ATR / CHK ────────────────────────────────────────
    "AZD6738":              "ceralasertib",
    "VE-822":               "VE-822",
    "VE821":                "VE-821",
    "MK-8776":              "SCH-900776",
    # ── Other kinase inhibitors ───────────────────────────────────────
    "Dabrafenib":           "GSK2118436",
    "AZD4547":              "AZD-4547",
    "AZD3759":              "AZD-3759",
    "AZD1208":              "AZD-1208",
    "Entospletinib":        "GS-9973",
    "PRT062607":            "PRT-062607",
    "LCL161":               "LCL-161",
    # ── Proteasome / IAP ──────────────────────────────────────────────
    "Sepantronium bromide": "YM-155",
    # ── Misc research codes ───────────────────────────────────────────
    "SB505124":             "SB-505124",
    "CZC24832":             "CZC-24832",
    "WZ4003":               "WZ-4003",
    "CCT007093":            "CCT-007093",
    "EHT-1864":             "EHT1864",
    "PF-4708671":           "PF-4708671",
    "LY2109761":            "LY-2109761",
    "SCH772984":            "SCH-772984",
    "GSK269962A":           "GSK-269962A",
    "GSK2606414":           "GSK-2606414",
    "GSK2578215A":          "GSK-2578215A",
    "AGI-5198":             "AGI5198",
    "AGI-6780":             "AGI6780",
    "AMG-319":              "AMG319",
    "NVP-ADW742":           "NVP-ADW-742",
    "A-366":                "A366",
    "AT13148":              "AT-13148",
    "AZD5582":              "AZD-5582",
    "AZD5991":              "AZD-5991",
    "AZD5153":              "AZD-5153",
    "AZD8186":              "AZD-8186",
    "AZD1332":              "AZD-1332",
    "GDC0810":              "GDC-0810",
    "ML323":                "ML-323",
    "Bromosporine":         "bromosporine",
    "IOX2":                 "IOX-2",
    "Oxaliplatin":          "oxaliplatin",
    "Bleomycin":            "bleomycin",
    "Bleomycin (50 uM)":    "bleomycin",
    "Nelarabine":           "nelarabine",
    "Zoledronate":          "zoledronic-acid",
    "Podophyllotoxin bromide": "podophyllotoxin",
    "Pyridostatin":         "pyridostatin",
    "WEHI-539":             "WEHI539",
    "WIKI4":                "WIKI-4",
    "IWP-2":                "IWP2",
    "PFI-1":                "PFI1",
    "PFI3":                 "PFI-3",
    "Wee1 Inhibitor":       "MK-1775",
    "JNK Inhibitor VIII":   "JNK-IN-8",
    "SL0101":               "SL-0101",
    "N-acetyl cysteine":    "acetylcysteine",
    "alpha-lipoic acid":    "thioctic-acid",
    "ascorbate (vitamin C)": "ascorbic-acid",
    # Note: Dihydrorotenone is a reduced form of rotenone -- related but
    # chemically distinct, so not mapped here.  PubChem bridge may match it.
    "GNE-317":              "GNE317",
    "GSK591":               "GSK-591",
    "GSK2801":              "GSK-2801",
    "GSK2830371":           "GSK-2830371",
    "I-BRD9":               "IBRD9",
    "OF-1":                 "OF1",
    "SGC-CBP30":            "SGC-CBP-30",
    "Avagacestat":          "BMS-708163",
    "AZ960":                "AZ-960",
    "AZ6102":               "AZ-6102",
}


# ---------------------------------------------------------------------------
# Cell-line synonym table
# ---------------------------------------------------------------------------
# Maps alternative/non-standard cell-line names to a canonical normalized form.
# Normalization = uppercase + strip hyphens/underscores/spaces.
# Both keys and values should be in normalized form.
_CELL_LINE_ALIAS_TABLE: dict[str, str] = {
    # GDSC uses "SK-BR-3" which normalises to "SKBR3" — same as LINCS.
    # These are extra edge-cases where even after normalisation the names
    # still differ.
    "HS578T":       "HS578T",       # identity — just to be safe
    "ZR751":        "ZR751",
    "ZR7530":       "ZR7530",
    "MDAMB175VII":  "MDAMB175VII",
}


def _normalize_drug(name: str) -> str:
    """Lowercase, strip whitespace/hyphens/spaces for drug matching."""
    return name.lower().strip().replace("-", "").replace(" ", "")


def _normalize_cell(name: str) -> str:
    """Uppercase, strip hyphens/underscores/spaces for cell-line matching."""
    import re
    return re.sub(r"[\-_\s]", "", name.upper())


# ---------------------------------------------------------------------------
# Drug alias map builder
# ---------------------------------------------------------------------------

def _build_drug_alias_map(
    gdsc_drug_names: list[str],
    lincs_drug_names: list[str],
    cache_dir: Path = DATA_CACHE,
    use_pubchem_bridge: bool = True,
) -> dict[str, str]:
    """
    Build a mapping from normalized GDSC drug names to normalized LINCS
    drug names, using:
      1. A hardcoded synonym table (_DRUG_SYNONYM_TABLE)
      2. PubChem CID bridging via the LINCS pertinfo (which already stores CIDs)

    Returns:
        dict mapping norm(gdsc_name) -> norm(lincs_name)
    """
    alias_map: dict[str, str] = {}

    # Normalised sets for quick lookup
    lincs_norm_set = {_normalize_drug(n) for n in lincs_drug_names}
    lincs_norm_to_orig = {}
    for n in lincs_drug_names:
        lincs_norm_to_orig.setdefault(_normalize_drug(n), n)

    # ── 1. Hardcoded synonym table ────────────────────────────────────
    for gdsc_orig, lincs_orig in _DRUG_SYNONYM_TABLE.items():
        gn = _normalize_drug(gdsc_orig)
        ln = _normalize_drug(lincs_orig)
        if ln in lincs_norm_set:
            alias_map[gn] = ln
        # Also try the LINCS name directly (unnormalized match)
        elif lincs_orig in lincs_drug_names:
            alias_map[gn] = _normalize_drug(lincs_orig)

    n_hardcoded = len(alias_map)
    logger.info(f"Drug alias map: {n_hardcoded} from hardcoded synonym table")

    # ── 2. PubChem CID bridge ─────────────────────────────────────────
    if use_pubchem_bridge:
        n_cid = _apply_pubchem_cid_bridge(
            alias_map, gdsc_drug_names, lincs_drug_names, lincs_norm_set, cache_dir
        )
        logger.info(f"Drug alias map: {n_cid} additional from PubChem CID bridge")

    logger.info(f"Drug alias map total: {len(alias_map)} entries")
    return alias_map


def _apply_pubchem_cid_bridge(
    alias_map: dict[str, str],
    gdsc_drug_names: list[str],
    lincs_drug_names: list[str],
    lincs_norm_set: set[str],
    cache_dir: Path,
) -> int:
    """
    For GDSC drugs not yet matched, look up their PubChem CID and try to
    match against LINCS drugs via the CID stored in LINCS pertinfo.

    Mutates alias_map in-place and returns the number of new matches added.
    """
    # Figure out which GDSC drugs are still unmatched after hardcoded + norm
    already_matched = set(alias_map.keys()) | lincs_norm_set
    unmatched_gdsc = [
        n for n in gdsc_drug_names
        if _normalize_drug(n) not in already_matched
    ]
    if not unmatched_gdsc:
        return 0

    # Load LINCS pertinfo which has pubchem_cid
    try:
        pertinfo_path = cache_dir / "GSE92742_pert_info.parquet"
        if not pertinfo_path.exists():
            logger.debug("No pertinfo cache for PubChem CID bridge")
            return 0
        pertinfo = pd.read_parquet(pertinfo_path)
    except Exception:
        return 0

    # Build CID -> LINCS pert_iname lookup (only for drugs in our breast set)
    lincs_name_set = set(lincs_drug_names)
    cid_to_lincs: dict[str, str] = {}
    for _, row in pertinfo.iterrows():
        name = row.get("pert_iname", "")
        cid = str(row.get("pubchem_cid", "")).strip()
        if name in lincs_name_set and cid and cid != "nan" and cid != "-666":
            # First-come wins (most drugs have a unique CID)
            cid_to_lincs.setdefault(cid, name)

    # For each unmatched GDSC drug, query PubChem for its CID
    pubchem_rest = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    n_added = 0

    # Cap at 100 lookups to keep runtime reasonable
    candidates = [
        n for n in unmatched_gdsc
        if not n.strip().isdigit() and _normalize_drug(n) not in alias_map
    ][:100]

    if candidates:
        logger.info(
            f"PubChem CID bridge: querying {len(candidates)} unmatched GDSC drugs..."
        )

    for drug_name in candidates:
        gn = _normalize_drug(drug_name)
        if gn in alias_map:
            continue

        try:
            from urllib.parse import quote
            url = (
                f"{pubchem_rest}/compound/name/"
                f"{quote(drug_name, safe='')}/cids/JSON"
            )
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                cid_list = (
                    data.get("IdentifierList", {}).get("CID", [])
                )
                for cid_int in cid_list:
                    cid = str(cid_int)
                    if cid in cid_to_lincs:
                        lincs_name = cid_to_lincs[cid]
                        alias_map[gn] = _normalize_drug(lincs_name)
                        n_added += 1
                        logger.info(
                            f"  PubChem CID bridge: {drug_name} -> "
                            f"{lincs_name} (CID {cid})"
                        )
                        break
            time.sleep(0.22)  # PubChem rate limit ~5 req/s
        except Exception:
            continue

    return n_added


# ---------------------------------------------------------------------------
# Cell-line alias map builder
# ---------------------------------------------------------------------------

def _build_cell_line_alias_map(
    gdsc_cell_names: list[str],
    lincs_cell_names: list[str],
) -> dict[str, str]:
    """
    Build a mapping from normalized GDSC cell-line names to normalized
    LINCS cell_id values.

    After standard normalization (uppercase, strip hyphens/underscores/spaces),
    most names should already match.  This function handles the remaining
    edge-cases via the _CELL_LINE_ALIAS_TABLE.

    Returns:
        dict mapping norm(gdsc_name) -> norm(lincs_name)
    """
    alias_map: dict[str, str] = {}
    lincs_norm_set = {_normalize_cell(n) for n in lincs_cell_names}

    for gdsc_orig, lincs_norm in _CELL_LINE_ALIAS_TABLE.items():
        gn = _normalize_cell(gdsc_orig)
        if lincs_norm in lincs_norm_set:
            alias_map[gn] = lincs_norm

    logger.info(f"Cell-line alias map: {len(alias_map)} entries")
    return alias_map

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

    For each LINCS signature (compound x cell line x dose), find the
    corresponding IC50 from PharmacoDB and interpolate the percent
    inhibition at that dose using the Hill equation.

    Matching strategy:
      1. Normalize both drug and cell-line names (lowercase / uppercase,
         strip hyphens, underscores, spaces).
      2. Apply a curated drug-synonym alias map (INN generics <-> research
         codes) so that e.g. GDSC "Rapamycin" matches LINCS "sirolimus".
      3. Use PubChem CID as a bridge for remaining unmatched drugs.
      4. Apply a cell-line alias map for edge-cases.
      5. Join on (drug_norm, cell_norm) to look up IC50 values.

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

    # ── 1. Normalize compound names ───────────────────────────────────
    lincs_sigs = lincs_sigs.copy()
    lincs_sigs["drug_norm"] = lincs_sigs["pert_iname"].apply(_normalize_drug)

    dose_response_ref = dose_response_ref.copy()
    dose_response_ref["drug_norm"] = dose_response_ref["drug_name"].apply(
        _normalize_drug
    )

    # ── 2. Normalize cell-line names ──────────────────────────────────
    lincs_sigs["cell_norm"] = lincs_sigs["cell_id"].apply(_normalize_cell)
    dose_response_ref["cell_norm"] = dose_response_ref["cell_line"].apply(
        _normalize_cell
    )

    # ── 3. Build drug alias map (GDSC norm -> LINCS norm) ─────────────
    gdsc_drugs = dose_response_ref["drug_name"].unique().tolist()
    lincs_drugs = lincs_sigs["pert_iname"].unique().tolist()
    drug_alias = _build_drug_alias_map(
        gdsc_drugs, lincs_drugs, cache_dir=cache_dir
    )

    # Apply: remap GDSC drug_norm using alias map so it matches LINCS
    # We remap the dose_response_ref side: for each GDSC drug, if it has
    # an alias that points to a LINCS norm, use that instead.
    dose_response_ref["drug_norm"] = dose_response_ref["drug_norm"].map(
        lambda dn: drug_alias.get(dn, dn)
    )

    # Log drug overlap stats
    gdsc_norms = set(dose_response_ref["drug_norm"].unique())
    lincs_norms = set(lincs_sigs["drug_norm"].unique())
    drug_overlap = gdsc_norms & lincs_norms
    logger.info(
        f"Drug overlap after alias mapping: {len(drug_overlap)}/{len(gdsc_norms)} "
        f"GDSC drugs match LINCS ({100*len(drug_overlap)/max(len(gdsc_norms),1):.0f}%)"
    )

    # ── 4. Build cell-line alias map ──────────────────────────────────
    gdsc_cells = dose_response_ref["cell_line"].unique().tolist()
    lincs_cells = lincs_sigs["cell_id"].unique().tolist()
    cell_alias = _build_cell_line_alias_map(gdsc_cells, lincs_cells)

    # Apply cell alias to GDSC side
    if cell_alias:
        dose_response_ref["cell_norm"] = dose_response_ref["cell_norm"].map(
            lambda cn: cell_alias.get(cn, cn)
        )

    # Log cell overlap stats
    gdsc_cell_norms = set(dose_response_ref["cell_norm"].unique())
    lincs_cell_norms = set(lincs_sigs["cell_norm"].unique())
    cell_overlap = gdsc_cell_norms & lincs_cell_norms
    logger.info(
        f"Cell-line overlap: {len(cell_overlap)} GDSC lines match LINCS "
        f"({sorted(cell_overlap)})"
    )

    # ── 5. Build IC50 lookup: (drug_norm, cell_norm) -> ic50_um ───────
    ic50_lookup: dict[tuple[str, str], float] = {}
    for _, row in dose_response_ref.iterrows():
        key = (row["drug_norm"], row["cell_norm"])
        ic50_val = row.get("ic50_um_linear", np.nan)
        if not pd.isna(ic50_val):
            ic50_lookup[key] = ic50_val

    logger.info(f"Built IC50 lookup with {len(ic50_lookup)} drug-cell pairs")

    # ── 6. Match and interpolate ──────────────────────────────────────
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

    n_drugs_matched = matched["pert_iname"].nunique()
    n_cells_matched = matched["cell_id"].nunique()
    logger.info(
        f"Matched {len(matched):,} LINCS-PharmacoDB triads "
        f"({n_drugs_matched} drugs, {n_cells_matched} cell lines)"
    )

    if len(matched) == 0:
        logger.warning(
            "No matches found. Check that GDSC2 and LINCS data are loaded "
            "correctly and contain overlapping drugs/cell-lines."
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
