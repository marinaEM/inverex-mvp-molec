"""
PRISM PAN-CANCER integration for the INVEREX pipeline.

The existing prism.py filters to breast-cancer cell lines only. This module
operates on ALL cell lines across ALL cancer types, giving the model exposure
to targeted therapies, endocrine agents, kinase inhibitors, and diverse biology
beyond chemotherapy.

The breast-specific signal is extracted later by the personalized ranker's
context layer -- the base model should learn pan-cancer drug-biology relationships.

Pipeline:
    1. Download PRISM secondary screen + DepMap cell line annotations.
    2. Load and parse all PRISM files (treatment info, dose-response, logFC).
    3. Analyze coverage vs GDSC2 and LINCS.
    4. Match PRISM drugs to LINCS signatures (ALL cell lines).
    5. Build pan-cancer training matrix: gene z-scores + viability targets.
    6. Retrain LightGBM: GDSC2-only vs GDSC2+PRISM, 5-fold CV.
    7. Validate on CTR-DB patients with LODO cross-validation.
    8. Drug category analysis (chemo bias reduction).
    9. Retrain treatability model with PRISM-expanded predictions.
"""
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score

from src.config import (
    DATA_CACHE,
    DATA_RAW,
    ECFP_NBITS,
    LIGHTGBM_DEFAULT_PARAMS,
    RANDOM_SEED,
    RESULTS,
)
from src.data_ingestion.utils import download_file

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
PRISM_DIR = DATA_RAW / "prism"

# ── PRISM download URLs (try in order) ─────────────────────────────────
PRISM_URLS = {
    "treatment_info": [
        "https://ndownloader.figshare.com/files/20237739",
        "https://figshare.com/ndownloader/files/20237739",
    ],
    "dose_response": [
        "https://ndownloader.figshare.com/files/20237715",
        "https://figshare.com/ndownloader/files/20237715",
    ],
    "logfold_change": [
        "https://ndownloader.figshare.com/files/20237718",
        "https://figshare.com/ndownloader/files/20237718",
    ],
    "cell_line_info": [
        "https://ndownloader.figshare.com/files/35020903",
        "https://depmap.org/portal/download/api/download?file_name=Model.csv",
    ],
}

PRISM_FILENAMES = {
    "treatment_info": "secondary-screen-replicate-treatment-info.csv",
    "dose_response": "secondary-screen-dose-response-curve-parameters.csv",
    "logfold_change": "secondary-screen-replicate-collapsed-logfold-change.csv",
    "cell_line_info": "Model.csv",
}


# =====================================================================
# Utility helpers
# =====================================================================

def _normalize_drug_name(name: str) -> str:
    """Normalize drug name: lowercase, strip hyphens/spaces/parens."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    for ch in ["-", " ", "(", ")", ".", ","]:
        s = s.replace(ch, "")
    return s


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find first matching column (case-insensitive)."""
    col_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in col_lower:
            return col_lower[cand.lower()]
    return None


def _find_name_column(df: pd.DataFrame) -> Optional[str]:
    """Find the drug name column in a PRISM DataFrame."""
    for c in ["name", "Name", "compound_name", "pert_iname"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


# =====================================================================
# 1. Download PRISM data
# =====================================================================

def _try_download(urls: list[str], dest: Path, timeout: int = 120) -> bool:
    """Try downloading from a list of URLs. Return True on success."""
    for url in urls:
        try:
            logger.info(f"  Trying {url[:80]}...")
            download_file(url, dest, timeout=timeout)
            if dest.exists() and dest.stat().st_size > 1000:
                logger.info(f"  OK: {dest.name} ({dest.stat().st_size:,} bytes)")
                return True
        except Exception as e:
            logger.warning(f"  Failed: {e}")
    return False


def download_prism_data(
    dest_dir: Path = PRISM_DIR, force: bool = False
) -> dict[str, Path]:
    """Download all PRISM secondary screen files + DepMap cell line info."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    logger.info("=" * 70)
    logger.info("PRISM PAN-CANCER DATA DOWNLOAD")
    logger.info("=" * 70)

    for data_type, urls in PRISM_URLS.items():
        filename = PRISM_FILENAMES[data_type]
        dest = dest_dir / filename

        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1e6
            logger.info(f"  {data_type}: already downloaded ({size_mb:.1f} MB)")
            downloaded[data_type] = dest
            continue

        logger.info(f"Downloading {data_type}...")
        if _try_download(urls, dest):
            downloaded[data_type] = dest
        else:
            logger.error(f"  FAILED to download {data_type}")

    logger.info(f"\nDownloaded {len(downloaded)}/{len(PRISM_URLS)} files")
    return downloaded


# =====================================================================
# 2. Load PRISM data
# =====================================================================

def load_prism_treatment_info(prism_dir: Path = PRISM_DIR) -> pd.DataFrame:
    """Load PRISM treatment info (drug metadata)."""
    cache_path = prism_dir / "pancancer_treatment_info.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Auto-detect: treatment info has 'dose' but not 'depmap_id'
    for fname in [PRISM_FILENAMES["treatment_info"],
                  PRISM_FILENAMES["dose_response"],
                  PRISM_FILENAMES["logfold_change"]]:
        path = prism_dir / fname
        if not path.exists():
            continue
        try:
            peek = pd.read_csv(path, low_memory=False, nrows=5)
            if "dose" in peek.columns and "depmap_id" not in peek.columns:
                df = pd.read_csv(path, low_memory=False)
                logger.info(
                    f"Loaded treatment info from {fname}: {len(df)} rows, "
                    f"{df['name'].nunique() if 'name' in df.columns else '?'} drugs"
                )
                df.to_parquet(cache_path, index=False)
                return df
        except Exception:
            continue

    # Fallback: extract from dose-response
    for fname in [PRISM_FILENAMES["dose_response"],
                  PRISM_FILENAMES["treatment_info"]]:
        path = prism_dir / fname
        if not path.exists():
            continue
        try:
            peek = pd.read_csv(path, low_memory=False, nrows=5)
            if "depmap_id" in peek.columns and "name" in peek.columns:
                df = pd.read_csv(path, low_memory=False)
                drug_cols = [c for c in df.columns if c in [
                    "broad_id", "name", "moa", "target", "disease.area",
                    "indication", "smiles", "phase", "screen_id",
                ]]
                if drug_cols:
                    ti = df[drug_cols].drop_duplicates(
                        subset=["name"] if "name" in drug_cols else None
                    )
                    logger.info(f"Extracted treatment info: {len(ti)} rows")
                    ti.to_parquet(cache_path, index=False)
                    return ti
        except Exception:
            continue

    logger.warning(f"PRISM treatment info not found in {prism_dir}")
    return pd.DataFrame()


def load_prism_dose_response(prism_dir: Path = PRISM_DIR) -> pd.DataFrame:
    """Load PRISM dose-response (IC50, AUC). ALL cell lines."""
    cache_path = prism_dir / "pancancer_dose_response.parquet"
    if cache_path.exists():
        logger.info("Loading cached pan-cancer dose-response...")
        return pd.read_parquet(cache_path)

    for fname in [PRISM_FILENAMES["dose_response"],
                  PRISM_FILENAMES["treatment_info"],
                  PRISM_FILENAMES["logfold_change"]]:
        path = prism_dir / fname
        if not path.exists():
            continue
        try:
            peek = pd.read_csv(path, low_memory=False, nrows=5)
            if "depmap_id" in peek.columns and "auc" in peek.columns:
                logger.info(f"Loading dose-response from {fname}...")
                df = pd.read_csv(path, low_memory=False)
                logger.info(
                    f"Dose-response: {len(df)} rows, "
                    f"{df['depmap_id'].nunique()} cell lines, "
                    f"{df['name'].nunique() if 'name' in df.columns else '?'} drugs"
                )
                df.to_parquet(cache_path, index=False)
                return df
        except Exception:
            continue

    logger.warning(f"PRISM dose-response not found in {prism_dir}")
    return pd.DataFrame()


def load_prism_cell_info(prism_dir: Path = PRISM_DIR) -> pd.DataFrame:
    """Load DepMap cell line info (depmap_id -> cancer type)."""
    cache_path = prism_dir / "pancancer_cell_info.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Try Model.csv first, then sample_info.csv
    for fname in [PRISM_FILENAMES["cell_line_info"], "sample_info.csv"]:
        path = prism_dir / fname
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            logger.info(f"Cell line info: {len(df)} lines from {fname}")
            df.to_parquet(cache_path, index=False)
            return df

    logger.warning(f"Cell line info not found in {prism_dir}")
    return pd.DataFrame()


# =====================================================================
# 3. Coverage analysis
# =====================================================================

def analyze_prism_coverage(prism_dir: Path = PRISM_DIR) -> pd.DataFrame:
    """
    Comprehensive coverage report: PRISM drugs/cell-lines/cancer-types,
    overlap with GDSC2 and LINCS.
    """
    logger.info("=" * 70)
    logger.info("PRISM PAN-CANCER COVERAGE ANALYSIS")
    logger.info("=" * 70)

    treatment = load_prism_treatment_info(prism_dir)
    dose_resp = load_prism_dose_response(prism_dir)
    cell_info = load_prism_cell_info(prism_dir)

    rows = []

    # -- PRISM statistics --
    prism_drug_names = set()
    name_col = _find_name_column(treatment) if not treatment.empty else None
    if name_col and not treatment.empty:
        prism_drug_names = set(treatment[name_col].dropna().unique())
        rows.append({"metric": "PRISM total drugs (treatment_info)", "value": len(prism_drug_names)})
        named_drugs = [d for d in prism_drug_names
                       if not str(d).startswith("BRD-")]
        rows.append({"metric": "PRISM drugs with common names", "value": len(named_drugs)})
        logger.info(f"PRISM drugs: {len(prism_drug_names)} total, {len(named_drugs)} named")

    if not dose_resp.empty:
        rows.append({"metric": "PRISM dose-response rows", "value": len(dose_resp)})
        if "depmap_id" in dose_resp.columns:
            n_cells = dose_resp["depmap_id"].nunique()
            rows.append({"metric": "PRISM cell lines (dose-response)", "value": n_cells})
        if "name" in dose_resp.columns:
            n_drugs_dr = dose_resp["name"].nunique()
            rows.append({"metric": "PRISM drugs (dose-response)", "value": n_drugs_dr})
        logger.info(f"PRISM dose-response: {len(dose_resp)} rows")

    # -- Cancer types --
    if not cell_info.empty:
        lineage_col = _find_col(cell_info, [
            "OncotreeLineage", "primary_disease", "lineage",
            "OncotreePrimaryDisease",
        ])
        if lineage_col:
            cancer_types = cell_info[lineage_col].dropna().unique()
            rows.append({"metric": "Cancer types (DepMap)", "value": len(cancer_types)})
            logger.info(f"Cancer types: {len(cancer_types)}")
            for ct in sorted(cancer_types)[:15]:
                n = (cell_info[lineage_col] == ct).sum()
                logger.info(f"  {ct}: {n} cell lines")

    # -- GDSC2 overlap --
    gdsc2_path = DATA_CACHE / "gdsc2_dose_response.parquet"
    gdsc2_drugs = set()
    gdsc2_cancer_types = set()
    if gdsc2_path.exists():
        gdsc2 = pd.read_parquet(gdsc2_path)
        gdsc2_drugs = set(gdsc2["DRUG_NAME"].unique())
        gdsc2_cancer_types = set(gdsc2["CANCER_TYPE"].unique())
        rows.append({"metric": "GDSC2 drugs", "value": len(gdsc2_drugs)})
        rows.append({"metric": "GDSC2 cancer types", "value": len(gdsc2_cancer_types)})

        # Normalized overlap
        gdsc2_norm = set(_normalize_drug_name(d) for d in gdsc2_drugs)
        prism_norm = set(_normalize_drug_name(d) for d in prism_drug_names)
        overlap = prism_norm & gdsc2_norm
        rows.append({"metric": "PRISM-GDSC2 drug overlap", "value": len(overlap)})
        rows.append({"metric": "PRISM-only drugs (not in GDSC2)", "value": len(prism_norm - gdsc2_norm)})
        logger.info(f"GDSC2 drugs: {len(gdsc2_drugs)}, overlap: {len(overlap)}")

    # -- LINCS overlap --
    siginfo_path = DATA_CACHE / "GSE92742_sig_info.parquet"
    if siginfo_path.exists():
        siginfo = pd.read_parquet(siginfo_path)
        lincs_drugs = set(
            _normalize_drug_name(d)
            for d in siginfo[siginfo["pert_type"] == "trt_cp"]["pert_iname"].unique()
        )
        prism_norm = set(_normalize_drug_name(d) for d in prism_drug_names)
        lincs_overlap = prism_norm & lincs_drugs
        rows.append({"metric": "LINCS drugs (all)", "value": len(lincs_drugs)})
        rows.append({"metric": "PRISM-LINCS drug overlap", "value": len(lincs_overlap)})
        logger.info(f"LINCS drugs: {len(lincs_drugs)}, PRISM overlap: {len(lincs_overlap)}")
        if lincs_overlap:
            logger.info(f"  Sample overlap: {sorted(lincs_overlap)[:15]}")

    # -- LINCS signature files overlap --
    for sig_file, label in [
        ("all_cellline_drug_signatures.parquet", "all-cellline sigs"),
        ("breast_l1000_signatures.parquet", "breast sigs"),
    ]:
        p = DATA_CACHE / sig_file
        if p.exists():
            sigs = pd.read_parquet(p)
            sig_drugs = set(_normalize_drug_name(d) for d in sigs["pert_iname"].unique())
            prism_norm = set(_normalize_drug_name(d) for d in prism_drug_names)
            ol = prism_norm & sig_drugs
            rows.append({"metric": f"PRISM vs {label} drug overlap", "value": len(ol)})
            rows.append({"metric": f"{label} total drugs", "value": len(sig_drugs)})
            rows.append({"metric": f"{label} total sigs", "value": len(sigs)})
            rows.append({"metric": f"{label} cell lines", "value": sigs["cell_id"].nunique()})
            logger.info(f"{label}: {len(sigs)} sigs, {len(sig_drugs)} drugs, overlap={len(ol)}")

    coverage = pd.DataFrame(rows)
    save_path = RESULTS / "prism_integration_analysis.csv"
    coverage.to_csv(save_path, index=False)
    logger.info(f"\nCoverage saved to {save_path}")
    return coverage


# =====================================================================
# 4. Match PRISM drugs to LINCS signatures (ALL cell lines)
# =====================================================================

def match_prism_to_lincs(
    prism_dir: Path = PRISM_DIR,
    cache_dir: Path = DATA_CACHE,
) -> pd.DataFrame:
    """
    Match PRISM drugs to LINCS L1000 signatures using ALL cell lines.
    Returns LINCS signatures for drugs that overlap with PRISM.
    """
    logger.info("=" * 70)
    logger.info("MATCHING PRISM DRUGS TO LINCS (PAN-CANCER, ALL CELL LINES)")
    logger.info("=" * 70)

    cache_path = cache_dir / "prism_pancancer_lincs_matched.parquet"
    if cache_path.exists():
        logger.info("Loading cached pan-cancer PRISM-LINCS matches...")
        return pd.read_parquet(cache_path)

    # Load PRISM treatment info for drug names
    treatment = load_prism_treatment_info(prism_dir)
    if treatment.empty:
        logger.error("No PRISM treatment info")
        return pd.DataFrame()

    name_col = _find_name_column(treatment)
    if name_col is None:
        logger.error("No drug name column in PRISM treatment info")
        return pd.DataFrame()

    # Build normalized PRISM drug name set (exclude BRD-only names)
    prism_norm_to_orig = {}
    for name in treatment[name_col].dropna().unique():
        norm = _normalize_drug_name(name)
        if norm and not norm.startswith("brd"):
            prism_norm_to_orig[norm] = str(name)
    logger.info(f"PRISM named drugs: {len(prism_norm_to_orig)}")

    # Load ALL LINCS signatures (combine all-cellline + breast)
    all_sigs = _load_all_lincs_signatures(cache_dir)
    if all_sigs.empty:
        logger.error("No LINCS signatures available")
        return pd.DataFrame()

    # Build LINCS drug name set
    lincs_norm_to_orig = {}
    for name in all_sigs["pert_iname"].dropna().unique():
        norm = _normalize_drug_name(name)
        if norm:
            lincs_norm_to_orig[norm] = str(name)
    logger.info(f"LINCS drugs (all cell lines): {len(lincs_norm_to_orig)}")

    # Find overlap
    overlap = set(prism_norm_to_orig.keys()) & set(lincs_norm_to_orig.keys())
    logger.info(f"Drug name overlap: {len(overlap)}")
    if overlap:
        logger.info(f"  Sample: {sorted(overlap)[:20]}")

    if not overlap:
        logger.warning("No overlap between PRISM and LINCS drug names")
        return pd.DataFrame()

    # Filter LINCS to matched drugs
    matched_lincs_names = [lincs_norm_to_orig[n] for n in overlap]
    matched = all_sigs[all_sigs["pert_iname"].isin(matched_lincs_names)].copy()
    matched["prism_drug_norm"] = matched["pert_iname"].apply(_normalize_drug_name)

    logger.info(
        f"Matched LINCS sigs: {len(matched)} "
        f"({matched['pert_iname'].nunique()} drugs, "
        f"{matched['cell_id'].nunique()} cell lines)"
    )

    matched.to_parquet(cache_path, index=False)
    return matched


def _load_all_lincs_signatures(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """Load and combine ALL available LINCS signatures."""
    frames = []

    all_path = cache_dir / "all_cellline_drug_signatures.parquet"
    if all_path.exists():
        df = pd.read_parquet(all_path)
        logger.info(f"  all-cellline sigs: {df.shape} ({df['cell_id'].nunique()} cells)")
        frames.append(df)

    breast_path = cache_dir / "breast_l1000_signatures.parquet"
    if breast_path.exists():
        df = pd.read_parquet(breast_path)
        logger.info(f"  breast sigs: {df.shape} ({df['cell_id'].nunique()} cells)")
        frames.append(df)

    if not frames:
        logger.warning("No LINCS signature files found in cache")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset="sig_id", keep="first")
    logger.info(
        f"  Combined: {combined.shape} "
        f"({combined['pert_iname'].nunique()} drugs, "
        f"{combined['cell_id'].nunique()} cell lines)"
    )
    return combined


# =====================================================================
# 5. Build pan-cancer training matrix
# =====================================================================

def build_prism_training_matrix(
    prism_dir: Path = PRISM_DIR,
    cache_dir: Path = DATA_CACHE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build pan-cancer training matrix from PRISM + LINCS.

    For ALL matched (drug, cell_line):
      - Features: gene z-scores from LINCS + log_dose
      - Target: pct_inhibition from PRISM AUC/IC50 or logFC
      - Metadata: drug name, cell line, cancer type (NOT used as feature)

    Returns (X, y, metadata).
    """
    prism_cache = cache_dir / "prism_pancancer_X.parquet"
    target_cache = cache_dir / "prism_pancancer_y.parquet"
    meta_cache = cache_dir / "prism_pancancer_meta.parquet"

    if prism_cache.exists() and target_cache.exists() and meta_cache.exists():
        logger.info("Loading cached pan-cancer PRISM training matrix...")
        X = pd.read_parquet(prism_cache)
        y = pd.read_parquet(target_cache).squeeze()
        meta = pd.read_parquet(meta_cache)
        logger.info(f"  Loaded: {X.shape[0]} samples, {meta['drug'].nunique()} drugs, "
                     f"{meta['cancer_type'].nunique()} cancer types")
        return X, y, meta

    logger.info("=" * 70)
    logger.info("BUILDING PAN-CANCER PRISM TRAINING MATRIX")
    logger.info("=" * 70)

    # -- Get matched LINCS signatures --
    matched_sigs = match_prism_to_lincs(prism_dir, cache_dir)
    if matched_sigs.empty:
        logger.error("No PRISM-LINCS matches")
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    # -- Load PRISM dose-response for viability targets --
    dose_resp = load_prism_dose_response(prism_dir)

    # -- Build viability lookup --
    viability = _build_viability_lookup(dose_resp, prism_dir)

    if not viability:
        logger.warning("No PRISM viability data. Falling back to GDSC2 matching.")
        return _build_fallback_gdsc2_matrix(matched_sigs, cache_dir)

    # -- Load cell line info for cancer type annotation --
    cell_info = load_prism_cell_info(prism_dir)
    cell_cancer = _build_cell_cancer_map(cell_info)

    # Also map LINCS cell lines to GDSC2 cancer types
    lincs_cancer = _build_lincs_cell_cancer_map(cache_dir)

    # -- Identify gene columns --
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id",
                 "pert_idose", "dose_um", "prism_drug_norm"}
    gene_cols = sorted(c for c in matched_sigs.columns if c not in meta_cols)

    # -- Build averaged signatures per (drug, cell) --
    logger.info("Building averaged LINCS signatures per (drug, cell)...")
    sigs_copy = matched_sigs.copy()
    sigs_copy["_drug_norm"] = sigs_copy["pert_iname"].apply(_normalize_drug_name)
    sigs_copy["_cell_norm"] = sigs_copy["cell_id"].apply(
        lambda x: re.sub(r"[\-_\s]", "", str(x).upper())
    )
    avg_sigs = sigs_copy.groupby(["_drug_norm", "_cell_norm"])[gene_cols].mean()
    avg_sig_dict = {idx: row.values.astype(np.float32) for idx, row in avg_sigs.iterrows()}
    logger.info(f"  Averaged signatures: {len(avg_sig_dict)} (drug, cell) pairs")

    # Also build drug-averaged signatures (across all cell lines)
    drug_avg = sigs_copy.groupby("_drug_norm")[gene_cols].mean()
    drug_avg_dict = {idx: row.values.astype(np.float32) for idx, row in drug_avg.iterrows()}
    logger.info(f"  Drug-averaged signatures: {len(drug_avg_dict)} drugs")

    # -- Match viability to signatures --
    logger.info("Matching viability targets to LINCS signatures...")

    rows_X = []
    rows_y = []
    rows_meta = []
    match_stats = Counter()

    for (drug_norm, prism_cell_id), pct_inh in viability.items():
        # Try exact (drug, cell) match
        cell_upper = re.sub(r"[\-_\s]", "", prism_cell_id.upper())
        key = (drug_norm, cell_upper)
        if key in avg_sig_dict:
            gene_z = avg_sig_dict[key]
            match_stats["exact_match"] += 1
        elif drug_norm in drug_avg_dict:
            gene_z = drug_avg_dict[drug_norm]
            match_stats["drug_avg_match"] += 1
        else:
            match_stats["no_match"] += 1
            continue

        # Determine cancer type
        cancer_type = cell_cancer.get(prism_cell_id, "Unknown")
        if cancer_type == "Unknown":
            cancer_type = lincs_cancer.get(cell_upper, "Unknown")

        rows_X.append(np.append(gene_z, np.log1p(np.float32(1.0))))  # default dose
        rows_y.append(pct_inh)
        rows_meta.append({
            "drug": drug_norm,
            "prism_cell": prism_cell_id,
            "cancer_type": cancer_type,
            "dose_um": 1.0,
        })

        if len(rows_X) >= 150000:
            logger.info("  Reached 150k sample cap")
            break

    logger.info(f"  Match stats: {dict(match_stats)}")

    if not rows_X:
        logger.warning("No matches. Falling back to GDSC2.")
        return _build_fallback_gdsc2_matrix(matched_sigs, cache_dir)

    # -- Assemble --
    feature_names = gene_cols + ["log_dose_um"]
    X = pd.DataFrame(rows_X, columns=feature_names, dtype=np.float32)
    y = pd.Series(rows_y, name="pct_inhibition", dtype=np.float32)
    meta = pd.DataFrame(rows_meta)

    # Clean
    valid = y.notna() & np.isfinite(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    meta = meta.loc[valid].reset_index(drop=True)

    logger.info(f"\nPRISM training matrix: {X.shape[0]} samples x {X.shape[1]} features")
    logger.info(f"  Drugs: {meta['drug'].nunique()}")
    logger.info(f"  PRISM cell lines: {meta['prism_cell'].nunique()}")
    logger.info(f"  Cancer types: {meta['cancer_type'].nunique()}")
    logger.info(f"  Target: mean={y.mean():.1f}%, std={y.std():.1f}%, "
                f"range=[{y.min():.1f}, {y.max():.1f}]")

    for ct, cnt in meta["cancer_type"].value_counts().head(10).items():
        logger.info(f"    {ct}: {cnt}")

    # Cache
    X.to_parquet(prism_cache, index=False)
    pd.DataFrame({"pct_inhibition": y}).to_parquet(target_cache, index=False)
    meta.to_parquet(meta_cache, index=False)

    return X, y, meta


def _build_viability_lookup(
    dose_resp: pd.DataFrame,
    prism_dir: Path,
) -> dict[tuple[str, str], float]:
    """
    Build (norm_drug, depmap_id) -> pct_inhibition lookup from dose-response.
    Uses AUC: pct_inhibition = (1 - AUC) * 100, clipped to [0, 100].
    """
    lookup = {}

    if dose_resp.empty:
        return lookup

    logger.info("Building viability lookup from PRISM dose-response...")

    # Identify columns
    drug_col = _find_col(dose_resp, ["name", "broad_id", "compound"])
    cell_col = _find_col(dose_resp, ["depmap_id", "DepMap_ID"])
    auc_col = _find_col(dose_resp, ["auc", "AUC"])
    ic50_col = _find_col(dose_resp, ["ic50", "IC50", "ec50", "EC50"])

    if not drug_col or not cell_col:
        logger.warning(f"Cannot identify drug/cell columns. Available: {list(dose_resp.columns)}")
        return lookup

    # If drug names are BRD IDs, resolve via treatment info
    broad_to_name = {}
    treatment = load_prism_treatment_info(prism_dir)
    if not treatment.empty:
        broad_col_ti = _find_col(treatment, ["broad_id", "BRD_ID"])
        name_col_ti = _find_name_column(treatment)
        if broad_col_ti and name_col_ti:
            for _, row in treatment.iterrows():
                bid = str(row.get(broad_col_ti, ""))
                nm = str(row.get(name_col_ti, ""))
                if bid and nm and nm != "nan":
                    broad_to_name[bid] = nm

    count = 0
    for _, row in dose_resp.iterrows():
        drug_raw = str(row[drug_col])
        cell_raw = str(row[cell_col])

        # Resolve name
        drug_name = broad_to_name.get(drug_raw, drug_raw)
        drug_norm = _normalize_drug_name(drug_name)
        if not drug_norm or drug_norm.startswith("brd"):
            continue

        # Get AUC -> pct_inhibition
        pct = np.nan
        if auc_col and pd.notna(row.get(auc_col)):
            auc_val = float(row[auc_col])
            pct = (1.0 - auc_val) * 100.0
        elif ic50_col and pd.notna(row.get(ic50_col)):
            ic50_val = float(row[ic50_col])
            if ic50_val > 0:
                pct = 100.0 * (1.0 / (1.0 + ic50_val))  # Hill at conc=1uM

        if not np.isnan(pct):
            pct = float(np.clip(pct, 0, 100))
            lookup[(drug_norm, cell_raw)] = pct
            count += 1

    logger.info(f"  Viability lookup: {count} entries, "
                f"{len(set(k[0] for k in lookup))} drugs")
    return lookup


def _build_cell_cancer_map(cell_info: pd.DataFrame) -> dict[str, str]:
    """Map DepMap cell ID -> cancer type."""
    if cell_info.empty:
        return {}
    id_col = _find_col(cell_info, ["ModelID", "DepMap_ID", "depmap_id"])
    cancer_col = _find_col(cell_info, [
        "OncotreeLineage", "OncotreePrimaryDisease",
        "primary_disease", "lineage",
    ])
    if not id_col or not cancer_col:
        return {}
    result = {}
    for _, row in cell_info.iterrows():
        cid = str(row[id_col])
        ct = str(row[cancer_col])
        if cid and ct and ct != "nan":
            result[cid] = ct
    logger.info(f"  Cell-cancer map: {len(result)} entries")
    return result


def _build_lincs_cell_cancer_map(cache_dir: Path) -> dict[str, str]:
    """Map LINCS cell_id (uppercased) to cancer type from GDSC2."""
    gdsc2_path = cache_dir / "gdsc2_dose_response.parquet"
    if not gdsc2_path.exists():
        return {}
    gdsc2 = pd.read_parquet(gdsc2_path)
    result = {}
    for _, row in gdsc2.drop_duplicates("CELL_LINE_NAME").iterrows():
        cell = re.sub(r"[\-_\s]", "", str(row["CELL_LINE_NAME"]).upper())
        ct = str(row.get("CANCER_TYPE", "Unknown"))
        result[cell] = ct
    return result


def _build_fallback_gdsc2_matrix(
    matched_sigs: pd.DataFrame,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Fallback: use GDSC2 dose-response with all matched LINCS sigs.
    Pan-cancer: uses ALL cancer types from GDSC2, not just breast.
    """
    logger.info("Building GDSC2 pan-cancer fallback training matrix...")

    gdsc2_path = cache_dir / "gdsc2_dose_response.parquet"
    if not gdsc2_path.exists():
        logger.error("GDSC2 not found")
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    gdsc2 = pd.read_parquet(gdsc2_path)
    logger.info(f"GDSC2: {len(gdsc2)} rows, {gdsc2['DRUG_NAME'].nunique()} drugs, "
                f"{gdsc2['CANCER_TYPE'].nunique()} cancer types")

    # Build lookup: (norm_drug, norm_cell) -> (auc, cancer_type)
    gdsc2_lookup = {}
    for _, row in gdsc2.iterrows():
        dn = _normalize_drug_name(row["DRUG_NAME"])
        cn = re.sub(r"[\-_\s]", "", str(row["CELL_LINE_NAME"]).upper())
        auc = row.get("AUC", np.nan)
        ct = row.get("CANCER_TYPE", "Unknown")
        if dn and not pd.isna(auc):
            pct = float(np.clip((1.0 - auc) * 100.0, 0, 100))
            gdsc2_lookup[(dn, cn)] = (pct, ct)

    # Also build drug-average lookup
    drug_avg = {}
    drug_ct = {}
    for (dn, cn), (pct, ct) in gdsc2_lookup.items():
        drug_avg.setdefault(dn, []).append(pct)
        drug_ct.setdefault(dn, Counter())[ct] += 1

    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id",
                 "pert_idose", "dose_um", "prism_drug_norm"}
    gene_cols = sorted(c for c in matched_sigs.columns if c not in meta_cols)

    rows_X = []
    rows_y = []
    rows_meta = []

    for _, sig in matched_sigs.iterrows():
        dn = _normalize_drug_name(sig["pert_iname"])
        cn = re.sub(r"[\-_\s]", "", str(sig.get("cell_id", "")).upper())
        dose = sig.get("dose_um", 1.0)
        if pd.isna(dose):
            dose = 1.0

        key = (dn, cn)
        if key in gdsc2_lookup:
            pct, ct = gdsc2_lookup[key]
        elif dn in drug_avg:
            pct = float(np.mean(drug_avg[dn]))
            ct = drug_ct[dn].most_common(1)[0][0]
        else:
            continue

        gene_z = sig[gene_cols].values.astype(np.float32)
        rows_X.append(np.append(gene_z, np.log1p(np.float32(dose))))
        rows_y.append(pct)
        rows_meta.append({
            "drug": sig["pert_iname"],
            "prism_cell": cn,
            "cancer_type": ct,
            "dose_um": dose,
        })

    if not rows_X:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    feature_names = gene_cols + ["log_dose_um"]
    X = pd.DataFrame(rows_X, columns=feature_names, dtype=np.float32)
    y = pd.Series(rows_y, name="pct_inhibition", dtype=np.float32)
    meta = pd.DataFrame(rows_meta)

    valid = y.notna() & np.isfinite(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    meta = meta.loc[valid].reset_index(drop=True)

    logger.info(f"GDSC2 fallback: {X.shape[0]} samples, {meta['drug'].nunique()} drugs, "
                f"{meta['cancer_type'].nunique()} cancer types")

    # Cache
    X.to_parquet(cache_dir / "prism_pancancer_X.parquet", index=False)
    pd.DataFrame({"pct_inhibition": y}).to_parquet(
        cache_dir / "prism_pancancer_y.parquet", index=False
    )
    meta.to_parquet(cache_dir / "prism_pancancer_meta.parquet", index=False)

    return X, y, meta


# =====================================================================
# 6. Load GDSC2-only baseline training data
# =====================================================================

def _load_gdsc2_only_training(cache_dir: Path = DATA_CACHE):
    """Load existing GDSC2-breast training data as baseline."""
    processed = cache_dir.parent / "processed"
    X_path = processed / "training_matrix.parquet"
    y_path = processed / "training_target.parquet"

    if X_path.exists() and y_path.exists():
        X = pd.read_parquet(X_path)
        y = pd.read_parquet(y_path).squeeze()
        logger.info(f"GDSC2-only baseline: {X.shape}")
        return X, y

    # Build from scratch using breast dose-response + LINCS
    logger.info("Building GDSC2-only baseline from breast dose-response...")
    dr_path = cache_dir / "breast_dose_response_ref.parquet"
    sigs_path = cache_dir / "breast_l1000_signatures.parquet"

    if not dr_path.exists() or not sigs_path.exists():
        logger.warning("Cannot build GDSC2-only baseline (missing files)")
        return pd.DataFrame(), pd.Series(dtype=float)

    dose_ref = pd.read_parquet(dr_path)
    sigs = pd.read_parquet(sigs_path)

    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
    gene_cols = sorted(c for c in sigs.columns if c not in meta_cols)

    # DR lookup
    dr_lookup = {}
    for _, row in dose_ref.iterrows():
        dn = _normalize_drug_name(row["drug_name"])
        cn = re.sub(r"[\-_\s]", "", str(row["cell_line"]).upper())
        auc = row.get("auc", np.nan)
        if dn and not pd.isna(auc):
            dr_lookup[(dn, cn)] = float(np.clip((1.0 - auc) * 100.0, 0, 100))

    # Drug-average
    drug_avg = {}
    for (dn, _), pct in dr_lookup.items():
        drug_avg.setdefault(dn, []).append(pct)

    rows_X = []
    rows_y = []

    for _, sig in sigs.iterrows():
        dn = _normalize_drug_name(sig["pert_iname"])
        cn = re.sub(r"[\-_\s]", "", str(sig.get("cell_id", "")).upper())
        dose = sig.get("dose_um", 1.0)
        if pd.isna(dose):
            dose = 1.0

        key = (dn, cn)
        if key in dr_lookup:
            pct = dr_lookup[key]
        elif dn in drug_avg:
            pct = float(np.mean(drug_avg[dn]))
        else:
            continue

        gene_z = sig[gene_cols].values.astype(np.float32)
        rows_X.append(np.append(gene_z, np.log1p(np.float32(dose))))
        rows_y.append(pct)

    if not rows_X:
        return pd.DataFrame(), pd.Series(dtype=float)

    feature_names = gene_cols + ["log_dose_um"]
    X = pd.DataFrame(rows_X, columns=feature_names, dtype=np.float32)
    y = pd.Series(rows_y, name="pct_inhibition", dtype=np.float32)

    valid = y.notna() & np.isfinite(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    logger.info(f"GDSC2-only: {X.shape}")
    return X, y


# =====================================================================
# 7. Retrain pan-cancer models and compare
# =====================================================================

def retrain_pancancer_models(
    prism_dir: Path = PRISM_DIR,
    cache_dir: Path = DATA_CACHE,
    results_dir: Path = RESULTS,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Compare: GDSC2-only vs GDSC2+PRISM vs PRISM-only.
    Train LightGBM with 5-fold CV, report RMSE.
    """
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON: GDSC2-only vs GDSC2+PRISM (pan-cancer)")
    logger.info("=" * 70)

    results = []

    # -- Model A: GDSC2-only --
    logger.info("\n--- Model A: GDSC2-only (breast baseline) ---")
    X_gdsc, y_gdsc = _load_gdsc2_only_training(cache_dir)

    if not X_gdsc.empty and len(X_gdsc) >= 20:
        rmse_gdsc, model_gdsc = _train_and_evaluate(X_gdsc, y_gdsc, cv_folds, "GDSC2-only")
        results.append({
            "model": "GDSC2-only (breast)",
            "n_samples": len(X_gdsc),
            "n_features": X_gdsc.shape[1],
            "cv_rmse": round(rmse_gdsc, 4),
            "cancer_types": "1 (breast)",
        })
        joblib.dump(model_gdsc, results_dir / "prism_lightgbm_gdsc2_only.joblib")
    else:
        logger.warning("GDSC2-only data insufficient")

    # -- Model B: PRISM pan-cancer --
    logger.info("\n--- Model B: PRISM pan-cancer ---")
    X_prism, y_prism, meta_prism = build_prism_training_matrix(prism_dir, cache_dir)

    if not X_prism.empty and len(X_prism) >= 20:
        rmse_prism, model_prism = _train_and_evaluate(X_prism, y_prism, cv_folds, "PRISM")
        results.append({
            "model": "PRISM (pan-cancer)",
            "n_samples": len(X_prism),
            "n_features": X_prism.shape[1],
            "cv_rmse": round(rmse_prism, 4),
            "cancer_types": f"{meta_prism['cancer_type'].nunique()} (pan-cancer)",
        })
        joblib.dump(model_prism, results_dir / "prism_lightgbm_prism_only.joblib")

    # -- Model C: Combined GDSC2 + PRISM --
    if not X_gdsc.empty and not X_prism.empty:
        logger.info("\n--- Model C: GDSC2 + PRISM combined ---")
        common_feats = sorted(set(X_gdsc.columns) & set(X_prism.columns))
        if len(common_feats) >= 100:
            X_combined = pd.concat(
                [X_gdsc[common_feats], X_prism[common_feats]],
                axis=0, ignore_index=True
            )
            y_combined = pd.concat([y_gdsc, y_prism], axis=0, ignore_index=True)
            logger.info(f"Combined: {X_combined.shape[0]} = {len(X_gdsc)} + {len(X_prism)}")

            rmse_comb, model_comb = _train_and_evaluate(
                X_combined, y_combined, cv_folds, "GDSC2+PRISM"
            )
            n_ct = meta_prism["cancer_type"].nunique() if not meta_prism.empty else "?"
            results.append({
                "model": "GDSC2+PRISM (combined)",
                "n_samples": len(X_combined),
                "n_features": X_combined.shape[1],
                "cv_rmse": round(rmse_comb, 4),
                "cancer_types": f"{n_ct}+ (pan-cancer)",
            })
            joblib.dump(model_comb, results_dir / "prism_lightgbm_combined.joblib")
        else:
            logger.warning(f"Only {len(common_feats)} common features; skipping combined model")

    comparison = pd.DataFrame(results)
    save_path = results_dir / "prism_pancancer_model_comparison.csv"
    comparison.to_csv(save_path, index=False)
    logger.info(f"\nComparison saved to {save_path}")
    logger.info(f"\n{comparison.to_string(index=False)}")

    return comparison


def _train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    label: str = "",
) -> tuple[float, lgb.LGBMRegressor]:
    """Train LightGBM with CV, return (RMSE, trained model)."""
    params = LIGHTGBM_DEFAULT_PARAMS.copy()
    model = lgb.LGBMRegressor(**params)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        lgb.LGBMRegressor(**params), X, y,
        cv=cv, scoring="neg_root_mean_squared_error"
    )
    rmse = -scores.mean()
    rmse_std = scores.std()
    logger.info(f"  [{label}] CV RMSE: {rmse:.4f} +/- {rmse_std:.4f} (n={len(X)})")

    model.fit(X, y)
    return rmse, model


# =====================================================================
# 8. Test on CTR-DB patients (LODO)
# =====================================================================

def test_on_ctrdb_patients(
    prism_dir: Path = PRISM_DIR,
    cache_dir: Path = DATA_CACHE,
    results_dir: Path = RESULTS,
) -> pd.DataFrame:
    """
    Run PRISM-expanded model on ALL CTR-DB datasets with LODO CV.
    Compare to existing pan-cancer model.
    """
    logger.info("=" * 70)
    logger.info("CTR-DB LODO VALIDATION (PRISM-EXPANDED)")
    logger.info("=" * 70)

    ctrdb_dir = DATA_RAW / "ctrdb"
    if not ctrdb_dir.exists():
        logger.error("No CTR-DB directory")
        return pd.DataFrame()

    catalog_path = ctrdb_dir / "pan_cancer_catalog.csv"
    if not catalog_path.exists():
        logger.warning("No pan-cancer catalog")
        return pd.DataFrame()

    catalog = pd.read_csv(catalog_path)
    cancer_type_lookup = {}
    for _, row in catalog.iterrows():
        geo = row.get("geo_source", "")
        ct = row.get("cancer_type", "Unknown")
        if geo:
            cancer_type_lookup[geo] = ct

    # Load landmark genes
    from src.data_ingestion.lincs import load_landmark_genes
    landmark_df = load_landmark_genes()
    landmark_genes = landmark_df["gene_symbol"].tolist()

    # Load all CTR-DB datasets
    all_X = []
    all_y = []
    all_dataset = []
    all_cancer = []

    for ds_dir in sorted(ctrdb_dir.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("GSE"):
            continue
        geo_id = ds_dir.name

        try:
            from src.data_ingestion.ctrdb import load_ctrdb_dataset
            result = load_ctrdb_dataset(ds_dir)
            if result is None:
                continue
            expr, labels = result
        except Exception:
            continue

        avail = [g for g in landmark_genes if g in expr.columns]
        if len(avail) < 50:
            continue

        expr_lm = expr[avail].copy()
        mu = expr_lm.mean(axis=0)
        sd = expr_lm.std(axis=0).replace(0, 1)
        expr_z = (expr_lm - mu) / sd

        ct = cancer_type_lookup.get(geo_id, "Unknown")
        all_X.append(expr_z)
        all_y.append(labels)
        all_dataset.extend([geo_id] * len(labels))
        all_cancer.extend([ct] * len(labels))

    if not all_X:
        logger.error("No usable CTR-DB datasets")
        return pd.DataFrame()

    # Common genes
    gene_counts = Counter()
    for x in all_X:
        gene_counts.update(x.columns.tolist())
    min_presence = max(2, int(0.8 * len(all_X)))
    common_genes = sorted(g for g, c in gene_counts.items() if c >= min_presence)
    logger.info(f"Common genes: {len(common_genes)} across {len(all_X)} datasets")

    X_list = [x.reindex(columns=common_genes, fill_value=0.0) for x in all_X]
    X_pooled = pd.concat(X_list, axis=0).reset_index(drop=True).fillna(0.0)
    y_pooled = pd.concat(all_y, axis=0).reset_index(drop=True).astype(int)
    dataset_ids = pd.Series(all_dataset, name="dataset_id")
    cancer_types = pd.Series(all_cancer, name="cancer_type")

    logger.info(f"Pooled: {len(X_pooled)} patients, {dataset_ids.nunique()} datasets")

    # LODO
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 300,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 10,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "class_weight": "balanced",
    }

    fold_results = []
    for held_out in dataset_ids.unique():
        test_mask = dataset_ids == held_out
        train_mask = ~test_mask
        X_tr = X_pooled[train_mask].reset_index(drop=True)
        y_tr = y_pooled[train_mask].reset_index(drop=True)
        X_te = X_pooled[test_mask].reset_index(drop=True)
        y_te = y_pooled[test_mask].reset_index(drop=True)

        if len(y_te) < 10 or y_te.nunique() < 2:
            continue
        n_r = int(y_te.sum())
        n_nr = len(y_te) - n_r
        if n_r < 3 or n_nr < 3:
            continue

        ct = cancer_types[test_mask].iloc[0]
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(X_tr, y_tr)
        yp = mdl.predict_proba(X_te)[:, 1]
        ypred = (yp >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_te, yp)
        except ValueError:
            auc = 0.5
        bacc = balanced_accuracy_score(y_te, ypred)
        sens = ypred[y_te == 1].sum() / max(y_te.sum(), 1)
        spec = (1 - ypred[y_te == 0]).sum() / max((1 - y_te).sum(), 1)

        fold_results.append({
            "held_out_dataset": held_out,
            "cancer_type": ct,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_test_responders": n_r,
            "n_test_nonresponders": n_nr,
            "auc": round(auc, 4),
            "balanced_accuracy": round(bacc, 4),
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "model": "PRISM-expanded",
        })
        logger.info(f"  {held_out} ({ct}): AUC={auc:.3f}, BalAcc={bacc:.3f} (n={len(X_te)})")

    if not fold_results:
        logger.error("No LODO folds completed")
        return pd.DataFrame()

    results_df = pd.DataFrame(fold_results)

    # Summary rows
    mean_all = {
        "held_out_dataset": "MEAN_ALL", "cancer_type": "ALL",
        "n_train": int(results_df["n_train"].mean()),
        "n_test": int(results_df["n_test"].mean()),
        "n_test_responders": int(results_df["n_test_responders"].mean()),
        "n_test_nonresponders": int(results_df["n_test_nonresponders"].mean()),
        "auc": round(results_df["auc"].mean(), 4),
        "balanced_accuracy": round(results_df["balanced_accuracy"].mean(), 4),
        "sensitivity": round(results_df["sensitivity"].mean(), 4),
        "specificity": round(results_df["specificity"].mean(), 4),
        "model": "PRISM-expanded",
    }
    results_df = pd.concat([results_df, pd.DataFrame([mean_all])], ignore_index=True)

    breast_folds = results_df[
        (results_df["cancer_type"] == "Breast cancer")
        & (~results_df["held_out_dataset"].str.startswith("MEAN"))
    ]
    if not breast_folds.empty:
        mean_breast = {
            "held_out_dataset": "MEAN_BREAST", "cancer_type": "Breast cancer",
            "n_train": int(breast_folds["n_train"].mean()),
            "n_test": int(breast_folds["n_test"].mean()),
            "n_test_responders": int(breast_folds["n_test_responders"].mean()),
            "n_test_nonresponders": int(breast_folds["n_test_nonresponders"].mean()),
            "auc": round(breast_folds["auc"].mean(), 4),
            "balanced_accuracy": round(breast_folds["balanced_accuracy"].mean(), 4),
            "sensitivity": round(breast_folds["sensitivity"].mean(), 4),
            "specificity": round(breast_folds["specificity"].mean(), 4),
            "model": "PRISM-expanded",
        }
        results_df = pd.concat([results_df, pd.DataFrame([mean_breast])], ignore_index=True)

    save_path = results_dir / "prism_ctrdb_lodo_results.csv"
    results_df.to_csv(save_path, index=False)
    logger.info(f"\nLODO results saved to {save_path}")

    # Compare with existing
    _compare_with_existing_lodo(results_df, results_dir)
    return results_df


def _compare_with_existing_lodo(prism_results: pd.DataFrame, results_dir: Path):
    """Compare PRISM LODO vs existing pan-cancer LODO."""
    existing_path = results_dir / "pan_cancer_model_lodo_results.csv"
    if not existing_path.exists():
        logger.info("No existing LODO results for comparison")
        return

    existing = pd.read_csv(existing_path)
    e_folds = existing[~existing["held_out_dataset"].str.startswith("MEAN")]
    p_folds = prism_results[~prism_results["held_out_dataset"].str.startswith("MEAN")]
    common = set(e_folds["held_out_dataset"]) & set(p_folds["held_out_dataset"])

    if not common:
        logger.info("No overlapping datasets for comparison")
        # Compare means
        e_mean = e_folds["auc"].mean() if not e_folds.empty else 0
        p_mean = p_folds["auc"].mean() if not p_folds.empty else 0
        logger.info(f"  Existing mean AUC: {e_mean:.4f}")
        logger.info(f"  PRISM mean AUC:    {p_mean:.4f}")
        return

    logger.info(f"\nComparison on {len(common)} common datasets:")
    logger.info(f"{'Dataset':<20} {'Existing':>10} {'PRISM':>10} {'Diff':>8}")
    logger.info("-" * 52)

    for ds in sorted(common):
        e_auc = e_folds.loc[e_folds["held_out_dataset"] == ds, "auc"].values[0]
        p_auc = p_folds.loc[p_folds["held_out_dataset"] == ds, "auc"].values[0]
        logger.info(f"  {ds:<20} {e_auc:>8.4f}   {p_auc:>8.4f} {p_auc-e_auc:>+8.4f}")

    e_m = e_folds.loc[e_folds["held_out_dataset"].isin(common), "auc"].mean()
    p_m = p_folds.loc[p_folds["held_out_dataset"].isin(common), "auc"].mean()
    logger.info("-" * 52)
    logger.info(f"  {'MEAN':<20} {e_m:>8.4f}   {p_m:>8.4f} {p_m-e_m:>+8.4f}")


# =====================================================================
# 9. Drug category analysis
# =====================================================================

def analyze_drug_categories(
    prism_dir: Path = PRISM_DIR,
    cache_dir: Path = DATA_CACHE,
    results_dir: Path = RESULTS,
) -> pd.DataFrame:
    """
    Drug category breakdown: GDSC2-only vs GDSC2+PRISM.
    Measures whether PRISM reduces chemo bias.
    """
    logger.info("=" * 70)
    logger.info("DRUG CATEGORY ANALYSIS (chemo bias check)")
    logger.info("=" * 70)

    CATEGORIES = {
        "Chemotherapy": [
            "camptothecin", "cisplatin", "carboplatin", "oxaliplatin",
            "doxorubicin", "epirubicin", "paclitaxel", "docetaxel",
            "gemcitabine", "fluorouracil", "5fu", "methotrexate",
            "cyclophosphamide", "vincristine", "vinblastine", "etoposide",
            "irinotecan", "topotecan", "cytarabine", "bleomycin",
            "capecitabine", "eribulin", "vinorelbine", "pemetrexed",
            "temozolomide", "mitomycin",
        ],
        "Kinase inhibitors": [
            "imatinib", "dasatinib", "nilotinib", "bosutinib",
            "gefitinib", "erlotinib", "afatinib", "osimertinib",
            "lapatinib", "neratinib", "sorafenib", "sunitinib",
            "pazopanib", "axitinib", "cabozantinib", "lenvatinib",
            "vemurafenib", "dabrafenib", "trametinib", "cobimetinib",
            "selumetinib", "crizotinib", "alectinib", "ruxolitinib",
            "ibrutinib", "palbociclib", "ribociclib", "abemaciclib",
            "staurosporine", "midostaurin", "regorafenib",
        ],
        "Endocrine therapy": [
            "tamoxifen", "fulvestrant", "letrozole", "anastrozole",
            "exemestane", "toremifene", "raloxifene", "megestrol",
            "enzalutamide", "abiraterone", "bicalutamide",
        ],
        "HDAC inhibitors": [
            "vorinostat", "panobinostat", "romidepsin", "belinostat",
            "entinostat", "tucidinostat", "trichostatin", "valproic",
        ],
        "mTOR/PI3K/AKT": [
            "everolimus", "temsirolimus", "rapamycin", "sirolimus",
            "alpelisib", "buparlisib", "copanlisib", "idelalisib",
            "pictilisib", "wortmannin",
        ],
        "PARP inhibitors": [
            "olaparib", "niraparib", "rucaparib", "talazoparib", "veliparib",
        ],
        "BCL-2 family": [
            "venetoclax", "navitoclax", "obatoclax",
        ],
    }

    def _categorize(drug_set):
        cats = {}
        categorized = set()
        for cat, kws in CATEGORIES.items():
            matched = set()
            for d in drug_set:
                dl = str(d).lower()
                if any(kw in dl for kw in kws):
                    matched.add(d)
                    categorized.add(d)
            cats[cat] = len(matched)
        cats["Other targeted"] = len(drug_set - categorized)
        return cats

    # GDSC2 drugs
    gdsc2_path = cache_dir / "gdsc2_dose_response.parquet"
    gdsc2_drugs = set()
    gdsc2_ct = set()
    if gdsc2_path.exists():
        g = pd.read_parquet(gdsc2_path)
        gdsc2_drugs = set(g["DRUG_NAME"].unique())
        gdsc2_ct = set(g["CANCER_TYPE"].unique())

    # PRISM drugs
    treatment = load_prism_treatment_info(prism_dir)
    prism_drugs = set()
    if not treatment.empty:
        nc = _find_name_column(treatment)
        if nc:
            prism_drugs = set(treatment[nc].dropna().unique())

    # PRISM metadata for cancer types
    meta_path = cache_dir / "prism_pancancer_meta.parquet"
    prism_ct = set()
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        prism_ct = set(meta["cancer_type"].unique()) - {"Unknown"}

    combined = gdsc2_drugs | prism_drugs
    gdsc2_cats = _categorize(gdsc2_drugs)
    combined_cats = _categorize(combined)

    rows = []
    for cat in list(CATEGORIES.keys()) + ["Other targeted"]:
        rows.append({
            "Category": cat,
            "GDSC2_only": gdsc2_cats.get(cat, 0),
            "GDSC2_PRISM": combined_cats.get(cat, 0),
            "PRISM_added": combined_cats.get(cat, 0) - gdsc2_cats.get(cat, 0),
        })

    rows.append({"Category": "Total drugs", "GDSC2_only": len(gdsc2_drugs),
                  "GDSC2_PRISM": len(combined), "PRISM_added": len(combined) - len(gdsc2_drugs)})
    rows.append({"Category": "Cancer types", "GDSC2_only": len(gdsc2_ct),
                  "GDSC2_PRISM": len(gdsc2_ct | prism_ct),
                  "PRISM_added": len(prism_ct - gdsc2_ct)})

    # Training rows
    X_gdsc, _ = _load_gdsc2_only_training(cache_dir)
    n_prism = 0
    pc = cache_dir / "prism_pancancer_X.parquet"
    if pc.exists():
        n_prism = len(pd.read_parquet(pc))
    n_gdsc = len(X_gdsc) if not X_gdsc.empty else 0
    rows.append({"Category": "Total training rows", "GDSC2_only": n_gdsc,
                  "GDSC2_PRISM": n_gdsc + n_prism, "PRISM_added": n_prism})

    cat_df = pd.DataFrame(rows)
    save_path = results_dir / "prism_drug_category_analysis.csv"
    cat_df.to_csv(save_path, index=False)
    logger.info(f"\n{cat_df.to_string(index=False)}")
    logger.info(f"\nSaved to {save_path}")
    return cat_df


# =====================================================================
# 10. Retrain treatability model
# =====================================================================

def retrain_treatability_model(
    prism_dir: Path = PRISM_DIR,
    cache_dir: Path = DATA_CACHE,
    results_dir: Path = RESULTS,
) -> pd.DataFrame:
    """
    Retrain patient treatability with PRISM-expanded training.
    Compares with existing model to check chemo bias reduction.
    """
    logger.info("=" * 70)
    logger.info("RETRAIN TREATABILITY MODEL (PRISM-expanded)")
    logger.info("=" * 70)

    results = test_on_ctrdb_patients(prism_dir, cache_dir, results_dir)

    if not results.empty:
        existing_path = results_dir / "pan_cancer_model_lodo_results.csv"
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            e_mean = existing[existing["held_out_dataset"] == "MEAN_ALL"]
            p_mean = results[results["held_out_dataset"] == "MEAN_ALL"]
            if not e_mean.empty and not p_mean.empty:
                logger.info(f"\nExisting LODO AUC: {e_mean['auc'].values[0]:.4f}")
                logger.info(f"PRISM-expanded AUC: {p_mean['auc'].values[0]:.4f}")
                logger.info(f"Diff: {p_mean['auc'].values[0] - e_mean['auc'].values[0]:+.4f}")

    return results


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_prism_pancancer_pipeline(
    skip_download: bool = False,
    force_download: bool = False,
) -> dict:
    """
    Complete PRISM pan-cancer integration pipeline.

    Steps:
        1. Download PRISM + DepMap data
        2. Analyze coverage (vs GDSC2, LINCS)
        3. Match PRISM drugs to LINCS signatures
        4. Build pan-cancer training matrix
        5. Retrain models (GDSC2-only vs GDSC2+PRISM)
        6. LODO on CTR-DB patients
        7. Drug category analysis
        8. Retrain treatability model

    Returns dict of all results.
    """
    logger.info("=" * 70)
    logger.info("PRISM PAN-CANCER INTEGRATION PIPELINE")
    logger.info("=" * 70)

    output = {}

    # 1. Download
    if not skip_download:
        output["downloaded"] = download_prism_data(force=force_download)

    # 2. Coverage
    output["coverage"] = analyze_prism_coverage()

    # 3. PRISM-LINCS matching
    output["matched_sigs"] = match_prism_to_lincs()

    # 4. Training matrix
    X, y, meta = build_prism_training_matrix()
    output["n_training_samples"] = len(X)
    output["n_drugs"] = meta["drug"].nunique() if not meta.empty else 0
    output["n_cancer_types"] = meta["cancer_type"].nunique() if not meta.empty else 0

    # 5. Model comparison
    output["model_comparison"] = retrain_pancancer_models()

    # 6. CTR-DB LODO
    output["ctrdb_lodo"] = test_on_ctrdb_patients()

    # 7. Drug categories
    output["drug_categories"] = analyze_drug_categories()

    # 8. Summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    for fname in [
        "prism_integration_analysis.csv",
        "prism_pancancer_model_comparison.csv",
        "prism_ctrdb_lodo_results.csv",
        "prism_drug_category_analysis.csv",
    ]:
        p = RESULTS / fname
        if p.exists():
            logger.info(f"  {p}")

    if not output.get("model_comparison", pd.DataFrame()).empty:
        logger.info(f"\nModel comparison:\n{output['model_comparison'].to_string(index=False)}")

    if not output.get("drug_categories", pd.DataFrame()).empty:
        logger.info(f"\nDrug categories:\n{output['drug_categories'].to_string(index=False)}")

    return output


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="PRISM pan-cancer integration")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    run_prism_pancancer_pipeline(
        skip_download=args.skip_download,
        force_download=args.force_download,
    )
