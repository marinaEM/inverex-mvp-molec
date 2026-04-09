#!/usr/bin/env python3
"""
Download additional breast cancer treatment response datasets found
via GEO search to further reduce chemotherapy bias.

Focuses on:
- Immunotherapy + PARP (GSE173839: durvalumab+olaparib+paclitaxel)
- Endocrine response (GSE59515: letrozole response prediction)
- Endocrine resistance (GSE111563: dormant/acquired resistant ER+)
- Endocrine pre/post (GSE145325: ER+ letrozole RNA-seq)
- Immunotherapy (GSE157284: PD-L1 in TNBC)
- Endocrine pre/post (GSE240671: neoadjuvant letrozole RNA-seq)

Usage:
    pixi run python scripts/download_additional_datasets.py
"""

import os
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import GEOparse

warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/marinaesteban-medina/Desktop/INVEREX/inverex-mvp")
RAW_DIR = BASE_DIR / "data" / "raw"
CTRDB_DIR = RAW_DIR / "ctrdb"

existing_ctrdb = sorted([
    d.split("/")[-1]
    for d in [str(p) for p in CTRDB_DIR.iterdir()]
    if os.path.isdir(d) and "GSE" in d.split("/")[-1]
])


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


ADDITIONAL_DATASETS = [
    {
        "name": "durva_olap_breast",
        "geo_id": "GSE173839",
        "description": "Durvalumab+olaparib+paclitaxel HER2- stage II/III breast",
        "treatment_class": "immunotherapy+parp",
        "expected_n": 105,
    },
    {
        "name": "endocrine_response_letrozole",
        "geo_id": "GSE59515",
        "description": "Endocrine therapy response prediction (letrozole)",
        "treatment_class": "endocrine",
        "expected_n": 75,
    },
    {
        "name": "endocrine_resistance_er",
        "geo_id": "GSE111563",
        "description": "Dormant/acquired resistant ER+ breast tumors",
        "treatment_class": "endocrine",
        "expected_n": 101,
    },
    {
        "name": "letrozole_rnaseq",
        "geo_id": "GSE145325",
        "description": "ER+ breast tumor RNA-seq treated with letrozole",
        "treatment_class": "endocrine",
        "expected_n": 58,
    },
    {
        "name": "pdl1_tnbc",
        "geo_id": "GSE157284",
        "description": "PD-L1 expression in TNBC (immunotherapy biomarker)",
        "treatment_class": "immunotherapy",
        "expected_n": 82,
    },
    {
        "name": "neoadj_letrozole_rnaseq",
        "geo_id": "GSE240671",
        "description": "Pre/post neoadjuvant letrozole RNA-seq",
        "treatment_class": "endocrine",
        "expected_n": 122,
    },
]

results = {}

for ds in ADDITIONAL_DATASETS:
    geo_id = ds["geo_id"]
    name = ds["name"]
    dest_dir = RAW_DIR / name

    log(f"\n{'='*70}")
    log(f"Downloading {geo_id} ({name}): {ds['description']}")
    log(f"{'='*70}")

    # Check CTR-DB overlap
    if geo_id in existing_ctrdb:
        log(f"  SKIP: {geo_id} already in CTR-DB")
        results[geo_id] = {"status": "already_in_ctrdb"}
        continue

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    soft_files = list(dest_dir.glob(f"{geo_id}*.soft.gz"))
    if soft_files:
        log(f"  SOFT file already exists: {soft_files[0].name}")
        try:
            gse = GEOparse.get_GEO(filepath=str(soft_files[0]))
        except Exception as e:
            log(f"  ERROR parsing: {e}")
            results[geo_id] = {"status": "failed", "error": str(e)}
            continue
    else:
        log(f"  Downloading from GEO...")
        try:
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(dest_dir), silent=True)
        except Exception as e:
            log(f"  DOWNLOAD FAILED: {e}")
            results[geo_id] = {"status": "failed", "error": str(e)}
            continue

    log(f"  Title: {gse.metadata.get('title', ['?'])[0][:100]}")
    log(f"  Platforms: {list(gse.gpls.keys())}")
    log(f"  Samples: {len(gse.gsms)}")

    # Extract phenotype
    pheno_rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {"sample_id": gsm_name}
        for key, val in gsm.metadata.items():
            if isinstance(val, list):
                val = val[0] if len(val) == 1 else "; ".join(val)
            row[key] = val
        chars = gsm.metadata.get("characteristics_ch1", [])
        for c in chars:
            if ":" in c:
                k, v = c.split(":", 1)
                row[f"char_{k.strip().lower().replace(' ', '_')}"] = v.strip()
        pheno_rows.append(row)

    pheno_df = pd.DataFrame(pheno_rows)

    # Identify response columns
    response_cols = []
    for col in pheno_df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in [
            "pcr", "response", "rcb", "residual", "ki67",
            "pathologic", "clinical_response", "treatment_arm",
            "arm", "drug", "regimen", "therapy", "subtype",
            "er_status", "her2_status", "pr_status",
            "receptor", "tnbc", "pdl1", "pd-l1", "resistant",
            "sensitive", "dormant", "relapse", "recurrence",
        ]):
            response_cols.append(col)

    log(f"  Response/treatment columns: {response_cols}")
    for col in response_cols[:10]:
        vals = pheno_df[col].dropna().unique()
        if len(vals) <= 20:
            log(f"    {col}: {list(vals)}")
        else:
            log(f"    {col}: {len(vals)} unique values")

    # Also show char_ columns
    char_cols = [c for c in pheno_df.columns if c.startswith("char_")]
    log(f"  Characteristic columns: {char_cols}")
    for col in char_cols:
        vals = pheno_df[col].dropna().unique()
        if len(vals) <= 15:
            log(f"    {col}: {list(vals)}")

    # Try to extract expression
    expression_extracted = False
    expr_df = None

    first_gsm = list(gse.gsms.values())[0]
    if first_gsm.table is not None and len(first_gsm.table) > 0:
        log(f"  Extracting expression from GSM tables...")
        try:
            expr_dict = {}
            for gsm_name, gsm in gse.gsms.items():
                if gsm.table is not None and len(gsm.table) > 0:
                    tbl = gsm.table.copy()
                    if "ID_REF" in tbl.columns and "VALUE" in tbl.columns:
                        tbl = tbl.set_index("ID_REF")["VALUE"]
                        expr_dict[gsm_name] = pd.to_numeric(tbl, errors="coerce")

            if expr_dict:
                expr_df = pd.DataFrame(expr_dict).T
                expr_df.index.name = "sample_id"
                expr_df = expr_df.dropna(axis=1, how="all")
                log(f"  Expression: {expr_df.shape[0]} samples x {expr_df.shape[1]} features")
                expression_extracted = True
        except Exception as e:
            log(f"  ERROR extracting expression: {e}")
    else:
        log(f"  No expression in GSM tables")
        supp = gse.metadata.get("supplementary_file", [])
        log(f"  Supplementary files: {supp[:3]}")

    # Save outputs
    if expression_extracted and expr_df is not None:
        expr_path = dest_dir / f"{geo_id}_expression.parquet"
        expr_df.to_parquet(expr_path)
        log(f"  Saved expression: {expr_path}")

    pheno_path = dest_dir / "response_labels.parquet"
    pheno_df.to_parquet(pheno_path, index=False)
    log(f"  Saved phenotype: {pheno_path}")

    has_expression = expression_extracted
    has_response = len(response_cols) > 0

    if has_expression and has_response:
        status = "downloaded"
    elif has_expression:
        status = "downloaded_no_response"
    elif has_response:
        status = "downloaded_no_expression"
    else:
        status = "downloaded_metadata_only"

    results[geo_id] = {
        "status": status,
        "n_patients": len(pheno_df),
        "has_expression": has_expression,
        "has_response": has_response,
        "expression_shape": expr_df.shape if expr_df is not None else None,
        "response_cols": response_cols,
    }

    log(f"  RESULT: {status} | {len(pheno_df)} samples | "
        f"expr={has_expression} | response={has_response}")


# ═══════════════════════════════════════════════════════════════════
# Download supplementary expression for RNA-seq datasets
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Downloading supplementary expression matrices for RNA-seq datasets")
log("=" * 70)

import urllib.request


def download_ftp(url, dest_path):
    if dest_path.exists():
        log(f"  Already downloaded: {dest_path.name}")
        return True
    log(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, str(dest_path))
        log(f"  Saved: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        log(f"  DOWNLOAD FAILED: {e}")
        return False


# GSE173839 supplementary
log("\n--- GSE173839 supplementary ---")
d173 = RAW_DIR / "durva_olap_breast"
# Check what supplementary files exist
import GEOparse as gp
try:
    soft_files = list(d173.glob("*.soft.gz"))
    if soft_files:
        gse = gp.get_GEO(filepath=str(soft_files[0]))
        supps = gse.metadata.get("supplementary_file", [])
        log(f"  Supplementary files: {supps}")
        for url in supps:
            fname = url.split("/")[-1]
            if any(kw in fname.lower() for kw in ["expression", "count", "tpm", "fpkm", "processed"]):
                dest = d173 / fname
                download_ftp(url, dest)
except Exception as e:
    log(f"  Error: {e}")

# GSE145325 supplementary
log("\n--- GSE145325 supplementary ---")
d145 = RAW_DIR / "letrozole_rnaseq"
try:
    soft_files = list(d145.glob("*.soft.gz"))
    if soft_files:
        gse = gp.get_GEO(filepath=str(soft_files[0]))
        supps = gse.metadata.get("supplementary_file", [])
        log(f"  Supplementary files: {supps}")
        for url in supps:
            fname = url.split("/")[-1]
            if any(kw in fname.lower() for kw in ["expression", "count", "tpm", "fpkm",
                                                     "processed", "normalized", "gene"]):
                dest = d145 / fname
                download_ftp(url, dest)
except Exception as e:
    log(f"  Error: {e}")

# GSE240671 supplementary
log("\n--- GSE240671 supplementary ---")
d240 = RAW_DIR / "neoadj_letrozole_rnaseq"
try:
    soft_files = list(d240.glob("*.soft.gz"))
    if soft_files:
        gse = gp.get_GEO(filepath=str(soft_files[0]))
        supps = gse.metadata.get("supplementary_file", [])
        log(f"  Supplementary files: {supps}")
        for url in supps:
            fname = url.split("/")[-1]
            if any(kw in fname.lower() for kw in ["expression", "count", "tpm", "fpkm",
                                                     "processed", "normalized", "gene"]):
                dest = d240 / fname
                download_ftp(url, dest)
except Exception as e:
    log(f"  Error: {e}")


log("\n" + "=" * 70)
log("SUMMARY OF ADDITIONAL DOWNLOADS")
log("=" * 70)

for geo_id, r in results.items():
    log(f"  {geo_id}: {r['status']} "
        f"(n={r.get('n_patients', '?')}, "
        f"expr={r.get('has_expression', '?')}, "
        f"resp={r.get('has_response', '?')})")

log("\nDONE")
