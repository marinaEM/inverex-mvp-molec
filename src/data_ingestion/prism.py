"""
PRISM (Profiling Relative Inhibition Simultaneously in Mixtures) data ingestion.

PRISM is from the Broad Institute's DepMap project. It screens ~8,000 compounds
across ~900 cancer cell lines using a barcoded multiplexed assay. The primary
readout is log-fold-change viability.

This module:
  1. Downloads PRISM secondary screen data (treatment info, dose-response,
     collapsed viability).
  2. Filters to breast cancer cell lines.
  3. Builds a training matrix analogous to GDSC2 matching but using PRISM
     viability data.
  4. Compares PRISM vs GDSC2 drug/cell-line coverage.
  5. Integrates PRISM data into the existing INVEREX training pipeline.

Data source: DepMap portal / figshare
  - Treatment info: drug metadata (name, dose, MOA, target)
  - Dose-response: curve fits (IC50, AUC, EC50 per drug-cell-line)
  - Collapsed viability: log-fold-change matrix (drugs x cell lines)
"""
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    BREAST_CELL_IDS_LINCS,
    BREAST_CELL_LINES,
    DATA_CACHE,
    DATA_RAW,
    ECFP_NBITS,
    RESULTS,
)
from src.data_ingestion.utils import download_file

logger = logging.getLogger(__name__)

# ── PRISM download URLs (figshare / DepMap) ─────────────────────────────
PRISM_TREATMENT_INFO_URL = (
    "https://ndownloader.figshare.com/files/20237739"
)
PRISM_DOSE_RESPONSE_URL = (
    "https://ndownloader.figshare.com/files/20237715"
)
PRISM_VIABILITY_URL = (
    "https://ndownloader.figshare.com/files/20237718"
)

# Filenames we save locally
TREATMENT_INFO_FILE = "secondary-screen-replicate-treatment-info.csv"
DOSE_RESPONSE_FILE = "secondary-screen-dose-response-curve-parameters.csv"
VIABILITY_FILE = "secondary-screen-replicate-collapsed-logfold-change.csv"

# DepMap cell line info URL (for mapping depmap_id to cell line names)
DEPMAP_CELL_LINE_URL = (
    "https://ndownloader.figshare.com/files/35020903"
)
DEPMAP_CELL_LINE_FILE = "sample_info.csv"


# ── Breast cancer cell-line identifiers for PRISM matching ──────────────
# DepMap uses ACH-XXXXX IDs. We'll also match by stripped_cell_line_name.
# These are the canonical breast cancer cell line names to look for.
BREAST_CELL_LINE_NAMES = set(BREAST_CELL_LINES.keys()) | {
    "MCF-7", "T-47D", "MDA-MB-231", "BT-474", "SK-BR-3",
    "Hs 578T", "MDA-MB-468", "HCC1937", "ZR-75-1", "MDA-MB-436",
    "CAL-51", "HCC1806", "BT-549", "MDA-MB-157",
    "HCC1954", "HCC1143", "HCC1187", "HCC1395", "HCC1599",
    "HCC2157", "HCC2218", "HCC38", "BT-20", "MDA-MB-453",
    "CAMA-1", "EFM-19", "MDA-MB-361", "UACC-812", "ZR-75-30",
    "SUM-149PT", "SUM-159PT", "SUM-185PE", "SUM-190PT",
    "AU565", "EFM-192A", "KPL-1", "MFM-223",
}


def _normalize_cell(name: str) -> str:
    """Uppercase, strip hyphens/underscores/spaces for cell-line matching."""
    return re.sub(r"[\-_\s]", "", name.upper())


def _normalize_drug(name: str) -> str:
    """Lowercase, strip whitespace/hyphens/spaces for drug matching."""
    return name.lower().strip().replace("-", "").replace(" ", "")


# ── Download functions ──────────────────────────────────────────────────

def download_prism_data(dest_dir: Optional[Path] = None) -> dict[str, Path]:
    """
    Download PRISM secondary screen data from DepMap/figshare.

    Downloads:
      - Treatment info (drug metadata: name, dose, MOA, target)
      - Dose-response curve parameters (IC50, AUC per drug-cell-line)
      - Collapsed viability (logfold-change matrix)

    Returns dict mapping filename -> local path.
    """
    if dest_dir is None:
        dest_dir = DATA_RAW / "prism"
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Treatment info (~2 MB)
    ti_path = dest_dir / TREATMENT_INFO_FILE
    if not ti_path.exists():
        logger.info("Downloading PRISM treatment info...")
        download_file(PRISM_TREATMENT_INFO_URL, ti_path, timeout=120)
    else:
        logger.info(f"PRISM treatment info already exists: {ti_path}")
    files["treatment_info"] = ti_path

    # Dose-response curve parameters (~50 MB)
    dr_path = dest_dir / DOSE_RESPONSE_FILE
    if not dr_path.exists():
        logger.info("Downloading PRISM dose-response curve parameters...")
        download_file(PRISM_DOSE_RESPONSE_URL, dr_path, timeout=300)
    else:
        logger.info(f"PRISM dose-response already exists: {dr_path}")
    files["dose_response"] = dr_path

    # Collapsed viability logfold-change (~400 MB)
    via_path = dest_dir / VIABILITY_FILE
    if not via_path.exists():
        logger.info("Downloading PRISM collapsed viability (this may take a while)...")
        download_file(PRISM_VIABILITY_URL, via_path, timeout=600)
    else:
        logger.info(f"PRISM viability already exists: {via_path}")
    files["viability"] = via_path

    logger.info(f"PRISM data downloaded to {dest_dir}")
    for k, v in files.items():
        size_mb = v.stat().st_size / 1e6 if v.exists() else 0
        logger.info(f"  {k}: {v.name} ({size_mb:.1f} MB)")

    return files


def download_depmap_cell_info(dest_dir: Optional[Path] = None) -> Path:
    """Download DepMap sample_info.csv for cell line annotation."""
    if dest_dir is None:
        dest_dir = DATA_RAW / "prism"
    dest_dir.mkdir(parents=True, exist_ok=True)

    path = dest_dir / DEPMAP_CELL_LINE_FILE
    if not path.exists():
        logger.info("Downloading DepMap cell line info...")
        download_file(DEPMAP_CELL_LINE_URL, path, timeout=120)
    return path


# ── Loading functions ───────────────────────────────────────────────────

def load_prism_treatment_info(
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load PRISM drug metadata: name, dose, MOA, target, clinical phase.

    Note: figshare URLs can redirect to different file contents than the
    filename suggests.  We auto-detect which file is the treatment metadata
    (small, ~4K-5K rows of drug info) vs the dose-response curves
    (large, ~700K rows with depmap_id, auc, ic50).

    Returns DataFrame with drug metadata columns.
    """
    if data_dir is None:
        data_dir = DATA_RAW / "prism"

    # Try each downloaded file and pick the one that looks like drug metadata
    # (has 'dose' column but not 'depmap_id', or is the smaller file)
    candidates = [
        data_dir / TREATMENT_INFO_FILE,
        data_dir / DOSE_RESPONSE_FILE,
        data_dir / VIABILITY_FILE,
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False, nrows=5)
            # The treatment info file has columns like:
            # column_name, broad_id, name, dose, screen_id, moa, target, ...
            # but does NOT have depmap_id or auc at the top level
            if "dose" in df.columns and "depmap_id" not in df.columns:
                df = pd.read_csv(path, low_memory=False)
                logger.info(
                    f"Loaded PRISM treatment info from {path.name}: {len(df)} rows, "
                    f"{df['name'].nunique() if 'name' in df.columns else '?'} unique drugs"
                )
                return df
        except Exception:
            continue

    # Fallback: if we have the large dose-response file, extract drug metadata from it
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False, nrows=5)
            if "depmap_id" in df.columns and "name" in df.columns:
                logger.info(f"Extracting treatment info from dose-response file {path.name}...")
                df_full = pd.read_csv(path, low_memory=False)
                # Extract unique drug metadata
                drug_cols = [c for c in df_full.columns if c in [
                    "broad_id", "name", "moa", "target", "disease.area",
                    "indication", "smiles", "phase", "screen_id",
                ]]
                if drug_cols:
                    ti = df_full[drug_cols].drop_duplicates(subset=["name"] if "name" in drug_cols else None)
                    logger.info(
                        f"Extracted PRISM treatment info: {len(ti)} rows, "
                        f"{ti['name'].nunique() if 'name' in ti.columns else '?'} unique drugs"
                    )
                    return ti
        except Exception:
            continue

    raise FileNotFoundError(
        f"PRISM treatment info not found in {data_dir}. Run download_prism_data() first."
    )


def load_prism_dose_response(
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load PRISM dose-response curve fits (IC50, AUC, EC50).

    Auto-detects the correct file (the one with depmap_id, auc, ic50 columns).

    Columns: depmap_id, ccle_name, screen_id, name, auc, ic50, ec50, ...
    """
    if data_dir is None:
        data_dir = DATA_RAW / "prism"

    candidates = [
        data_dir / TREATMENT_INFO_FILE,
        data_dir / DOSE_RESPONSE_FILE,
        data_dir / VIABILITY_FILE,
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False, nrows=5)
            # The dose-response file has depmap_id and auc columns
            if "depmap_id" in df.columns and "auc" in df.columns:
                logger.info(f"Loading PRISM dose-response from {path.name}...")
                df = pd.read_csv(path, low_memory=False)
                logger.info(
                    f"Loaded PRISM dose-response: {len(df)} rows, "
                    f"{df['depmap_id'].nunique()} cell lines, "
                    f"{df['name'].nunique() if 'name' in df.columns else '?'} drugs"
                )
                return df
        except Exception:
            continue

    raise FileNotFoundError(
        f"PRISM dose-response not found in {data_dir}. Run download_prism_data() first."
    )


def load_prism_viability(
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load PRISM collapsed viability log-fold-change matrix.

    Rows are treatments (drug::dose::screen), columns are DepMap cell line IDs.
    Values are log2 fold-change viability.
    """
    if data_dir is None:
        data_dir = DATA_RAW / "prism"
    path = data_dir / VIABILITY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"PRISM viability not found at {path}. Run download_prism_data() first."
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info(
        f"Loaded PRISM viability: {df.shape[0]} treatments x {df.shape[1]-1} cell lines"
    )
    return df


def load_depmap_cell_info(
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load DepMap cell line annotations (depmap_id -> cell_line_name, cancer type)."""
    if data_dir is None:
        data_dir = DATA_RAW / "prism"

    path = data_dir / DEPMAP_CELL_LINE_FILE
    if not path.exists():
        path = download_depmap_cell_info(data_dir)

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded DepMap cell info: {len(df)} cell lines")
    return df


# ── Filtering functions ─────────────────────────────────────────────────

def identify_breast_cell_lines(
    cell_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify breast cancer cell lines in DepMap annotations.

    Uses primary_disease / lineage columns to find breast lines,
    then cross-references with known breast cancer cell line names.

    Returns filtered DataFrame with breast cancer lines only.
    """
    # Try different column names for cancer type
    breast_mask = pd.Series(False, index=cell_info.index)

    for col in ["primary_disease", "lineage", "disease", "Subtype",
                 "primary_or_metastasis"]:
        if col in cell_info.columns:
            breast_mask |= cell_info[col].str.contains(
                "breast", case=False, na=False
            )

    # Also match by stripped_cell_line_name
    if "stripped_cell_line_name" in cell_info.columns:
        known_norms = {_normalize_cell(n) for n in BREAST_CELL_LINE_NAMES}
        breast_mask |= cell_info["stripped_cell_line_name"].apply(
            lambda x: _normalize_cell(str(x)) in known_norms if pd.notna(x) else False
        )

    # Also try cell_line_name
    if "cell_line_name" in cell_info.columns:
        known_norms = {_normalize_cell(n) for n in BREAST_CELL_LINE_NAMES}
        breast_mask |= cell_info["cell_line_name"].apply(
            lambda x: _normalize_cell(str(x)) in known_norms if pd.notna(x) else False
        )

    breast = cell_info[breast_mask].copy()
    logger.info(
        f"Identified {len(breast)} breast cancer cell lines in DepMap"
    )
    return breast


def filter_breast_prism(
    prism_dr: pd.DataFrame,
    breast_depmap_ids: set[str],
) -> pd.DataFrame:
    """
    Filter PRISM dose-response data to breast cancer cell lines.

    Args:
        prism_dr: PRISM dose-response DataFrame with depmap_id column
        breast_depmap_ids: set of DepMap IDs for breast cancer lines

    Returns filtered DataFrame.
    """
    if "depmap_id" not in prism_dr.columns:
        logger.warning("No depmap_id column in PRISM dose-response. Cannot filter.")
        return prism_dr

    filtered = prism_dr[prism_dr["depmap_id"].isin(breast_depmap_ids)].copy()
    logger.info(
        f"Filtered PRISM to breast cancer: {len(filtered)} records "
        f"({filtered['depmap_id'].nunique()} cell lines)"
    )
    return filtered


# ── LINCS matching ──────────────────────────────────────────────────────

def _build_prism_to_lincs_drug_map(
    prism_drug_names: list[str],
    lincs_drug_names: list[str],
) -> dict[str, str]:
    """
    Build a mapping from normalized PRISM drug names to normalized LINCS names.
    Uses direct normalization matching.
    """
    lincs_norm_set = {_normalize_drug(n) for n in lincs_drug_names}
    mapping = {}
    for pname in prism_drug_names:
        pn = _normalize_drug(pname)
        if pn in lincs_norm_set:
            mapping[pn] = pn
    return mapping


def _build_prism_to_lincs_cell_map(
    prism_cell_names: list[str],
    lincs_cell_names: list[str],
) -> dict[str, str]:
    """
    Build a mapping from normalized PRISM cell names to LINCS cell_id values.
    """
    lincs_norm_set = {_normalize_cell(n) for n in lincs_cell_names}
    mapping = {}
    for cname in prism_cell_names:
        cn = _normalize_cell(cname)
        if cn in lincs_norm_set:
            mapping[cn] = cn
    return mapping


def build_prism_training_matrix(
    breast_prism_dr: pd.DataFrame,
    lincs_sigs: pd.DataFrame,
    drug_fps: pd.DataFrame,
    cell_info: pd.DataFrame,
    cache_dir: Path = DATA_CACHE,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build training matrix from PRISM data, analogous to GDSC2 matching.

    For each (drug, cell_line) in PRISM with a matching LINCS signature:
    1. Get the gene z-scores from LINCS
    2. Get drug fingerprints (ECFP4)
    3. Get PRISM AUC/IC50 -> compute pct_inhibition
    4. Assemble: [gene z-scores | drug features | dose] -> response

    Returns (X, y) where X is the feature matrix and y is pct_inhibition.
    """
    cache_x = cache_dir / "prism_training_matrix.parquet"
    cache_y = cache_dir / "prism_training_target.parquet"
    if cache_x.exists() and cache_y.exists():
        logger.info("Loading cached PRISM training matrix...")
        X = pd.read_parquet(cache_x)
        y = pd.read_parquet(cache_y).squeeze()
        return X, y

    logger.info("Building PRISM training matrix...")

    # Build cell-name mapping. PRISM dose-response may have ccle_name
    # (e.g. "MCF7_BREAST") which we can use directly.
    # Also build depmap_id -> stripped_name mapping from cell_info as fallback.
    depmap_to_name = {}
    if "ccle_name" in breast_prism_dr.columns:
        # Extract cell name from CCLE name (format: CELLNAME_TISSUE)
        for _, row in breast_prism_dr.drop_duplicates("depmap_id").iterrows():
            ccle = str(row.get("ccle_name", ""))
            depmap_id = str(row.get("depmap_id", ""))
            if ccle and "_" in ccle:
                cell_name = ccle.split("_")[0]
                depmap_to_name[depmap_id] = cell_name

    # Also use cell_info if available
    name_col = None
    for col in ["stripped_cell_line_name", "cell_line_name", "CCLE_Name"]:
        if col in cell_info.columns:
            name_col = col
            break
    if name_col:
        id_col = "DepMap_ID" if "DepMap_ID" in cell_info.columns else "depmap_id"
        if id_col in cell_info.columns:
            for _, row in cell_info.iterrows():
                did = str(row.get(id_col, ""))
                cname = str(row.get(name_col, ""))
                if did and cname and did not in depmap_to_name:
                    depmap_to_name[did] = cname

    logger.info(f"Cell name mapping: {len(depmap_to_name)} entries")

    # Determine which column has the drug name in PRISM
    drug_col = "name" if "name" in breast_prism_dr.columns else "broad_id"
    depmap_col = "depmap_id" if "depmap_id" in breast_prism_dr.columns else None
    if depmap_col is None:
        logger.error("No depmap_id in PRISM dose-response data")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Normalize LINCS data
    lincs_drugs = lincs_sigs["pert_iname"].unique().tolist()
    lincs_cells = lincs_sigs["cell_id"].unique().tolist()
    lincs_drug_norms = {_normalize_drug(d): d for d in lincs_drugs}
    lincs_cell_norms = {_normalize_cell(c): c for c in lincs_cells}

    # Log what we're matching against
    logger.info(f"LINCS drugs: {len(lincs_drugs)}, LINCS cells: {lincs_cells}")

    # Identify gene columns in LINCS signatures
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
    gene_cols = [c for c in lincs_sigs.columns if c not in meta_cols]

    # Build an averaged LINCS signature lookup: (drug_norm, cell_norm) -> gene z-scores
    # Also build a drug-only lookup for cases where exact cell line is not in LINCS
    # but drug is (use average across all cell lines for that drug)
    logger.info("Building averaged LINCS signature lookup...")
    lincs_sigs_copy = lincs_sigs.copy()
    lincs_sigs_copy["drug_norm"] = lincs_sigs_copy["pert_iname"].apply(_normalize_drug)
    lincs_sigs_copy["cell_norm"] = lincs_sigs_copy["cell_id"].apply(_normalize_cell)

    # Average across doses/replicates for each (drug, cell_line)
    avg_sigs = lincs_sigs_copy.groupby(["drug_norm", "cell_norm"])[gene_cols].mean()
    avg_sig_dict = {idx: row.values for idx, row in avg_sigs.iterrows()}

    # Also build drug-only averages (across all cell lines)
    drug_avg_sigs = lincs_sigs_copy.groupby("drug_norm")[gene_cols].mean()
    drug_avg_dict = {idx: row.values for idx, row in drug_avg_sigs.iterrows()}

    logger.info(
        f"LINCS averaged signatures: {len(avg_sig_dict)} (drug, cell) pairs, "
        f"{len(drug_avg_dict)} drug-only averages"
    )

    # Drug fingerprint lookup
    ecfp_cols = [c for c in drug_fps.columns if c.startswith("ecfp_")]
    fp_lookup = {}
    if "compound_name" in drug_fps.columns:
        for _, row in drug_fps.iterrows():
            fp_lookup[_normalize_drug(row["compound_name"])] = row[ecfp_cols].values

    # Check PRISM drug overlap with LINCS
    prism_drugs_norm = set(breast_prism_dr[drug_col].dropna().apply(_normalize_drug).unique())
    lincs_drugs_norm = set(lincs_drug_norms.keys())
    drug_overlap = prism_drugs_norm & lincs_drugs_norm
    logger.info(
        f"PRISM breast drugs: {len(prism_drugs_norm)}, "
        f"overlapping with LINCS: {len(drug_overlap)}"
    )
    if drug_overlap:
        logger.info(f"  Sample overlapping drugs: {sorted(list(drug_overlap))[:20]}")

    # Check cell line overlap
    prism_cells_norm = set()
    for did in breast_prism_dr[depmap_col].dropna().unique():
        cname = depmap_to_name.get(str(did), "")
        if cname:
            prism_cells_norm.add(_normalize_cell(cname))
    cell_overlap = prism_cells_norm & set(lincs_cell_norms.keys())
    logger.info(
        f"PRISM breast cell lines (named): {len(prism_cells_norm)}, "
        f"overlapping with LINCS: {len(cell_overlap)} -> {sorted(cell_overlap)}"
    )

    # Iterate PRISM dose-response rows and match to LINCS
    matched_rows = []
    matched_targets = []
    no_lincs_match = 0
    no_lincs_drug = 0
    no_fp_match = 0
    no_cell_name = 0
    used_drug_avg = 0

    for _, prow in breast_prism_dr.iterrows():
        drug_name = str(prow.get(drug_col, ""))
        depmap_id = str(prow.get(depmap_col, ""))

        # Get cell line name
        cell_name = depmap_to_name.get(depmap_id, "")
        if not cell_name:
            no_cell_name += 1
            continue

        drug_n = _normalize_drug(drug_name)
        cell_n = _normalize_cell(cell_name)

        # Look up LINCS signature: first try exact (drug, cell), then drug-only
        sig_key = (drug_n, cell_n)
        gene_z = None
        if sig_key in avg_sig_dict:
            gene_z = avg_sig_dict[sig_key]
        elif drug_n in drug_avg_dict:
            gene_z = drug_avg_dict[drug_n]
            used_drug_avg += 1
        else:
            if drug_n not in lincs_drugs_norm:
                no_lincs_drug += 1
            else:
                no_lincs_match += 1
            continue

        # Get fingerprint
        fp = fp_lookup.get(drug_n)
        if fp is None:
            no_fp_match += 1
            # Use zero fingerprint as fallback
            fp = np.zeros(len(ecfp_cols), dtype=np.int8)

        # Get response value: convert PRISM AUC to pct_inhibition
        # PRISM AUC is area under the dose-response curve (0-1 scale typically,
        # but can exceed 1 for proliferative effects)
        # Higher AUC = more viability = less inhibition
        # pct_inhibition = (1 - AUC) * 100
        auc_val = prow.get("auc", np.nan)
        ic50_val = prow.get("ic50", np.nan)

        if pd.notna(auc_val):
            auc_f = float(auc_val)
            # PRISM AUC can be > 1 (proliferative) or negative
            # Clip to reasonable range
            pct_inhibition = (1.0 - auc_f) * 100.0
            pct_inhibition = np.clip(pct_inhibition, 0, 100)
        elif pd.notna(ic50_val):
            # Use IC50 with a standard dose of 2.5 uM (median PRISM dose)
            ic50_um = float(ic50_val)
            if ic50_um > 0:
                dose_um = 2.5
                pct_inhibition = 100.0 * (dose_um / (ic50_um + dose_um))
                pct_inhibition = np.clip(pct_inhibition, 0, 100)
            else:
                continue
        else:
            continue

        # Standard dose (use EC50 if available, else IC50, else 2.5 uM)
        dose_um = 2.5
        ec50_val = prow.get("ec50", np.nan)
        if pd.notna(ec50_val) and float(ec50_val) > 0:
            dose_um = min(float(ec50_val), 100.0)  # cap at 100 uM
        elif pd.notna(ic50_val) and float(ic50_val) > 0:
            dose_um = min(float(ic50_val), 100.0)
        dose_um = max(dose_um, 0.001)

        # Build row: gene_z + ecfp + log_dose
        row_features = np.concatenate([
            gene_z.astype(np.float32),
            fp.astype(np.int8),
            [np.float32(np.log1p(dose_um))],
        ])
        matched_rows.append(row_features)
        matched_targets.append(float(pct_inhibition))

    logger.info(
        f"PRISM matching stats: "
        f"{len(matched_rows)} matched, "
        f"{used_drug_avg} used drug-avg signature, "
        f"{no_lincs_drug} no LINCS drug, "
        f"{no_lincs_match} drug in LINCS but no matching cell, "
        f"{no_fp_match} no fingerprint (used zeros), "
        f"{no_cell_name} no cell name mapping"
    )

    if len(matched_rows) == 0:
        logger.warning("No PRISM-LINCS matches found.")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Assemble into DataFrames
    feature_names = list(gene_cols) + list(ecfp_cols) + ["log_dose_um"]
    X = pd.DataFrame(
        np.array(matched_rows, dtype=np.float32),
        columns=feature_names,
    )
    y = pd.Series(matched_targets, dtype=np.float32, name="pct_inhibition")

    # Cache
    X.to_parquet(cache_x, index=False)
    pd.DataFrame({"pct_inhibition": y}).to_parquet(cache_y, index=False)

    logger.info(f"PRISM training matrix: {X.shape[0]} samples x {X.shape[1]} features")
    return X, y


def build_prism_viability_training(
    viability_df: pd.DataFrame,
    treatment_info: pd.DataFrame,
    lincs_sigs: pd.DataFrame,
    drug_fps: pd.DataFrame,
    breast_depmap_ids: set[str],
    cell_info: pd.DataFrame,
    cache_dir: Path = DATA_CACHE,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Alternative approach: use the collapsed viability (logfold-change) matrix
    directly to build training data.

    logfold-change -> pct_inhibition: pct_inhibition = (1 - 2^logFC) * 100

    This can give more data points than dose-response curves.
    """
    cache_x = cache_dir / "prism_viability_training_matrix.parquet"
    cache_y = cache_dir / "prism_viability_training_target.parquet"
    if cache_x.exists() and cache_y.exists():
        logger.info("Loading cached PRISM viability training matrix...")
        X = pd.read_parquet(cache_x)
        y = pd.read_parquet(cache_y).squeeze()
        return X, y

    logger.info("Building PRISM viability-based training matrix...")

    # The viability file has rows = treatments, columns = depmap_ids
    # First column is usually something like "row_name" or treatment identifier
    id_col = viability_df.columns[0]

    # Filter to breast cell lines
    breast_cols = [c for c in viability_df.columns[1:] if c in breast_depmap_ids]
    if not breast_cols:
        logger.warning("No breast cancer cell lines found in viability matrix columns")
        return pd.DataFrame(), pd.Series(dtype=float)

    logger.info(f"Breast cell lines in viability matrix: {len(breast_cols)}")

    # Build depmap_id -> cell_line_name
    depmap_to_name = {}
    name_col = None
    for col in ["stripped_cell_line_name", "cell_line_name"]:
        if col in cell_info.columns:
            name_col = col
            break
    id_key = "DepMap_ID" if "DepMap_ID" in cell_info.columns else "depmap_id"
    if name_col and id_key in cell_info.columns:
        depmap_to_name = dict(zip(cell_info[id_key], cell_info[name_col]))

    # Build treatment_info lookup: column_name -> (drug_name, dose)
    treatment_lookup = {}
    if "column_name" in treatment_info.columns:
        for _, row in treatment_info.drop_duplicates("column_name").iterrows():
            treatment_lookup[row["column_name"]] = {
                "name": row.get("name", ""),
                "dose": row.get("dose", np.nan),
                "moa": row.get("moa", ""),
                "target": row.get("target", ""),
            }

    # LINCS data prep
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
    gene_cols = [c for c in lincs_sigs.columns if c not in meta_cols]

    lincs_copy = lincs_sigs.copy()
    lincs_copy["drug_norm"] = lincs_copy["pert_iname"].apply(_normalize_drug)
    lincs_copy["cell_norm"] = lincs_copy["cell_id"].apply(_normalize_cell)
    avg_sigs = lincs_copy.groupby(["drug_norm", "cell_norm"])[gene_cols].mean()
    avg_sig_dict = {idx: row.values for idx, row in avg_sigs.iterrows()}

    ecfp_cols = [c for c in drug_fps.columns if c.startswith("ecfp_")]
    fp_lookup = {}
    if "compound_name" in drug_fps.columns:
        for _, row in drug_fps.iterrows():
            fp_lookup[_normalize_drug(row["compound_name"])] = row[ecfp_cols].values

    matched_rows = []
    matched_targets = []
    n_matched = 0
    n_no_lincs = 0

    for _, vrow in viability_df.iterrows():
        treatment_id = str(vrow[id_col])
        tinfo = treatment_lookup.get(treatment_id, None)
        if tinfo is None:
            continue

        drug_name = str(tinfo["name"])
        dose_um = float(tinfo["dose"]) if pd.notna(tinfo.get("dose")) else 2.5
        drug_n = _normalize_drug(drug_name)

        for depmap_id in breast_cols:
            logfc = vrow.get(depmap_id, np.nan)
            if pd.isna(logfc):
                continue

            cell_name = depmap_to_name.get(depmap_id, "")
            if not cell_name:
                continue
            cell_n = _normalize_cell(cell_name)

            sig_key = (drug_n, cell_n)
            if sig_key not in avg_sig_dict:
                n_no_lincs += 1
                continue

            gene_z = avg_sig_dict[sig_key]
            fp = fp_lookup.get(drug_n, np.zeros(len(ecfp_cols), dtype=np.int8))

            # Convert logFC to pct_inhibition
            pct_inhibition = (1.0 - 2.0 ** float(logfc)) * 100.0
            pct_inhibition = np.clip(pct_inhibition, 0, 100)

            row_features = np.concatenate([
                gene_z.astype(np.float32),
                fp.astype(np.int8),
                [np.float32(np.log1p(dose_um))],
            ])
            matched_rows.append(row_features)
            matched_targets.append(float(pct_inhibition))
            n_matched += 1

    logger.info(
        f"PRISM viability matching: {n_matched} matched, {n_no_lincs} no LINCS sig"
    )

    if not matched_rows:
        return pd.DataFrame(), pd.Series(dtype=float)

    feature_names = list(gene_cols) + list(ecfp_cols) + ["log_dose_um"]
    X = pd.DataFrame(np.array(matched_rows, dtype=np.float32), columns=feature_names)
    y = pd.Series(matched_targets, dtype=np.float32, name="pct_inhibition")

    X.to_parquet(cache_x, index=False)
    pd.DataFrame({"pct_inhibition": y}).to_parquet(cache_y, index=False)
    logger.info(f"PRISM viability training matrix: {X.shape[0]} x {X.shape[1]}")
    return X, y


# ── Comparison/analysis functions ───────────────────────────────────────

def compare_prism_gdsc2_overlap(
    prism_treatment_info: pd.DataFrame,
    gdsc_dose_response: pd.DataFrame,
    lincs_sigs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare drug/cell-line overlap between PRISM and GDSC2.

    Returns a summary DataFrame with:
        - Total drugs in each source
        - Overlapping drugs
        - NEW drugs from PRISM
        - Of new drugs: how many match LINCS
        - Drug category breakdown
    """
    # Get drug names from each source
    prism_drug_col = "name" if "name" in prism_treatment_info.columns else "broad_id"
    prism_drugs = set(
        prism_treatment_info[prism_drug_col]
        .dropna()
        .unique()
    )

    gdsc_drugs = set(
        gdsc_dose_response["DRUG_NAME"].dropna().unique()
    ) if "DRUG_NAME" in gdsc_dose_response.columns else set()

    lincs_drugs = set(
        lincs_sigs["pert_iname"].dropna().unique()
    )

    # Normalize for comparison
    prism_norm = {_normalize_drug(d): d for d in prism_drugs}
    gdsc_norm = {_normalize_drug(d): d for d in gdsc_drugs}
    lincs_norm = {_normalize_drug(d): d for d in lincs_drugs}

    prism_set = set(prism_norm.keys())
    gdsc_set = set(gdsc_norm.keys())
    lincs_set = set(lincs_norm.keys())

    overlap_prism_gdsc = prism_set & gdsc_set
    new_in_prism = prism_set - gdsc_set
    new_in_prism_and_lincs = new_in_prism & lincs_set

    # Categorize drugs using PRISM MOA annotations
    moa_col = "moa" if "moa" in prism_treatment_info.columns else None
    target_col = "target" if "target" in prism_treatment_info.columns else None

    categories = {
        "chemotherapy": [],
        "targeted_therapy": [],
        "endocrine_therapy": [],
        "immunotherapy": [],
        "epigenetic": [],
        "other": [],
    }

    endocrine_keywords = [
        "estrogen", "progesterone", "androgen", "aromatase", "serm",
        "serd", "er agonist", "er antagonist", "hormone",
        "tamoxifen", "letrozole", "anastrozole", "fulvestrant",
        "exemestane", "toremifene", "raloxifene",
    ]
    chemo_keywords = [
        "dna damage", "topoisomerase", "tubulin", "microtubule",
        "alkylating", "antimetabolite", "mitotic", "nucleoside",
        "platinum", "anthracycline",
    ]
    targeted_keywords = [
        "kinase", "inhibitor", "receptor", "tyrosine", "mapk",
        "pi3k", "akt", "mtor", "cdk", "parp", "vegf", "egfr",
        "her2", "erbb", "braf", "mek", "alk", "ret", "fgfr",
        "bcl", "hdac", "proteasome", "jak", "stat",
    ]
    immuno_keywords = [
        "immune", "pd-1", "pd-l1", "ctla", "checkpoint",
    ]
    epigenetic_keywords = [
        "hdac", "dnmt", "histone", "methyltransferase",
        "bromodomain", "bet", "ezh", "lsd1",
    ]

    if moa_col or target_col:
        drug_to_moa = {}
        for _, row in prism_treatment_info.drop_duplicates(prism_drug_col).iterrows():
            dname = _normalize_drug(str(row.get(prism_drug_col, "")))
            moa_str = str(row.get(moa_col, "")).lower() if moa_col else ""
            target_str = str(row.get(target_col, "")).lower() if target_col else ""
            combined = moa_str + " " + target_str + " " + str(row.get(prism_drug_col, "")).lower()
            drug_to_moa[dname] = combined

        for dname in new_in_prism:
            desc = drug_to_moa.get(dname, "")
            categorized = False
            if any(kw in desc for kw in endocrine_keywords):
                categories["endocrine_therapy"].append(dname)
                categorized = True
            if any(kw in desc for kw in immuno_keywords):
                categories["immunotherapy"].append(dname)
                categorized = True
            if any(kw in desc for kw in epigenetic_keywords):
                categories["epigenetic"].append(dname)
                categorized = True
            if any(kw in desc for kw in chemo_keywords):
                categories["chemotherapy"].append(dname)
                categorized = True
            if any(kw in desc for kw in targeted_keywords):
                categories["targeted_therapy"].append(dname)
                categorized = True
            if not categorized:
                categories["other"].append(dname)

    # Build summary
    summary_rows = [
        {"metric": "Total drugs in PRISM", "value": len(prism_drugs)},
        {"metric": "Total drugs in GDSC2", "value": len(gdsc_drugs)},
        {"metric": "Total drugs in LINCS", "value": len(lincs_drugs)},
        {"metric": "PRISM-GDSC2 overlap", "value": len(overlap_prism_gdsc)},
        {"metric": "NEW drugs in PRISM (not in GDSC2)", "value": len(new_in_prism)},
        {"metric": "NEW PRISM drugs also in LINCS", "value": len(new_in_prism_and_lincs)},
        {"metric": "PRISM-LINCS total overlap", "value": len(prism_set & lincs_set)},
    ]
    for cat, drugs_list in categories.items():
        summary_rows.append({
            "metric": f"NEW PRISM drugs - {cat}",
            "value": len(drugs_list),
        })

    summary = pd.DataFrame(summary_rows)
    logger.info("\n" + "=" * 60)
    logger.info("PRISM vs GDSC2 Drug Coverage Comparison")
    logger.info("=" * 60)
    for _, row in summary.iterrows():
        logger.info(f"  {row['metric']}: {row['value']}")
    logger.info("=" * 60)

    return summary


def build_drug_coverage_table(
    prism_treatment_info: pd.DataFrame,
    gdsc_dose_response: pd.DataFrame,
    lincs_sigs: pd.DataFrame,
    gdsc2_n_samples: int = 719,
    prism_n_samples: int = 0,
) -> pd.DataFrame:
    """
    Build a drug coverage comparison table:
        Category             GDSC2 only   GDSC2+PRISM
        Chemotherapy drugs   X            X+Y
        Targeted therapy     X            X+Y
        Endocrine therapy    X            X+Y
        Total samples        719          719+prism
    """
    prism_drug_col = "name" if "name" in prism_treatment_info.columns else "broad_id"
    moa_col = "moa" if "moa" in prism_treatment_info.columns else None
    target_col = "target" if "target" in prism_treatment_info.columns else None

    # Get GDSC2 drug categorization from PATHWAY_NAME
    gdsc_categories = {"chemotherapy": set(), "targeted_therapy": set(),
                       "endocrine_therapy": set(), "other": set()}

    if "PATHWAY_NAME" in gdsc_dose_response.columns:
        for _, row in gdsc_dose_response.drop_duplicates("DRUG_NAME").iterrows():
            drug = row["DRUG_NAME"]
            pathway = str(row.get("PATHWAY_NAME", "")).lower()
            target = str(row.get("PUTATIVE_TARGET", "")).lower()

            if any(kw in pathway + " " + target for kw in [
                "dna replication", "apoptosis regulation", "mitosis",
                "cell cycle", "genome integrity",
            ]):
                gdsc_categories["chemotherapy"].add(drug)
            elif any(kw in (drug.lower() + " " + target) for kw in [
                "estrogen", "progesterone", "aromatase", "tamoxifen",
                "fulvestrant", "letrozole",
            ]):
                gdsc_categories["endocrine_therapy"].add(drug)
            elif any(kw in pathway + " " + target for kw in [
                "kinase", "rtk", "pi3k", "mapk", "erbb", "igf",
                "akt", "mtor", "wnt", "hedgehog", "notch",
                "chromatin", "metabolism",
            ]):
                gdsc_categories["targeted_therapy"].add(drug)
            else:
                gdsc_categories["other"].add(drug)

    # Get PRISM drug categorization from MOA
    prism_categories = {"chemotherapy": set(), "targeted_therapy": set(),
                        "endocrine_therapy": set(), "other": set()}

    if moa_col:
        for _, row in prism_treatment_info.drop_duplicates(prism_drug_col).iterrows():
            drug = str(row.get(prism_drug_col, ""))
            moa = str(row.get(moa_col, "")).lower()
            tgt = str(row.get(target_col, "")).lower() if target_col else ""
            combined = moa + " " + tgt + " " + drug.lower()

            if any(kw in combined for kw in [
                "estrogen", "progesterone", "aromatase", "serm", "serd",
                "hormone", "androgen receptor",
            ]):
                prism_categories["endocrine_therapy"].add(drug)
            elif any(kw in combined for kw in [
                "dna damage", "topoisomerase", "tubulin", "alkylating",
                "antimetabolite", "nucleoside",
            ]):
                prism_categories["chemotherapy"].add(drug)
            elif any(kw in combined for kw in [
                "kinase", "inhibitor", "receptor", "pi3k", "akt", "mtor",
                "cdk", "parp", "vegf", "egfr", "erbb", "braf", "mek",
                "hdac", "proteasome", "jak",
            ]):
                prism_categories["targeted_therapy"].add(drug)
            else:
                prism_categories["other"].add(drug)

    table_rows = []
    for cat in ["chemotherapy", "targeted_therapy", "endocrine_therapy", "other"]:
        gdsc_count = len(gdsc_categories.get(cat, set()))
        prism_count = len(prism_categories.get(cat, set()))
        combined = len(gdsc_categories.get(cat, set()) | prism_categories.get(cat, set()))
        table_rows.append({
            "Category": cat.replace("_", " ").title(),
            "GDSC2_only": gdsc_count,
            "PRISM_only": prism_count,
            "GDSC2_PRISM_combined": combined,
        })

    table_rows.append({
        "Category": "Total training samples",
        "GDSC2_only": gdsc2_n_samples,
        "PRISM_only": prism_n_samples,
        "GDSC2_PRISM_combined": gdsc2_n_samples + prism_n_samples,
    })

    return pd.DataFrame(table_rows)
