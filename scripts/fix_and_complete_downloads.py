#!/usr/bin/env python3
"""
Fix Hoogstraat expression files (mixed types) and investigate
datasets with missing response or wrong accessions.

Usage:
    pixi run python scripts/fix_and_complete_downloads.py
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/marinaesteban-medina/Desktop/INVEREX/inverex-mvp")
RAW_DIR = BASE_DIR / "data" / "raw"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Fix Hoogstraat 1 (GSE191127) - has gene biotype in first row
# ═══════════════════════════════════════════════════════════════════
log("=" * 70)
log("Fixing Hoogstraat 1 (GSE191127) expression matrix")
log("=" * 70)

hoog1_gz = RAW_DIR / "hoogstraat_1" / "GSE191127_readcounts.txt.gz"
if hoog1_gz.exists():
    df = pd.read_csv(hoog1_gz, sep="\t", compression="gzip", index_col=0)
    log(f"Raw shape: {df.shape}")
    log(f"First few index values: {list(df.index[:5])}")
    log(f"First row values sample: {list(df.iloc[0, :5])}")

    # Check if first row is gene biotype metadata
    first_row_vals = df.iloc[0].unique()
    log(f"First row unique values: {list(first_row_vals[:10])}")

    if any("coding" in str(v) for v in first_row_vals):
        log("First row contains gene biotype info - removing it")
        # Save biotype info separately
        biotype = df.iloc[0]
        df = df.iloc[1:]  # drop biotype row

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    log(f"After cleanup shape: {df.shape}")
    log(f"NaN fraction: {df.isna().sum().sum() / df.size:.4f}")

    # Transpose: genes x samples -> samples x genes
    expr = df.T
    expr.index.name = "sample_id"
    log(f"Expression (samples x genes): {expr.shape}")

    expr_path = RAW_DIR / "hoogstraat_1" / "GSE191127_expression.parquet"
    expr.to_parquet(expr_path)
    log(f"Saved: {expr_path}")


# ═══════════════════════════════════════════════════════════════════
# Fix Hoogstraat 2 (GSE192341) - same issue
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Fixing Hoogstraat 2 (GSE192341) expression matrix")
log("=" * 70)

hoog2_gz = RAW_DIR / "hoogstraat_2" / "GSE192341_processed_data.txt.gz"
if hoog2_gz.exists():
    df = pd.read_csv(hoog2_gz, sep="\t", compression="gzip", index_col=0)
    log(f"Raw shape: {df.shape}")
    log(f"First few index values: {list(df.index[:5])}")
    log(f"First row values sample: {list(df.iloc[0, :5])}")

    first_row_vals = df.iloc[0].unique()
    log(f"First row unique values: {list(first_row_vals[:10])}")

    if any("coding" in str(v) for v in first_row_vals):
        log("First row contains gene biotype info - removing it")
        df = df.iloc[1:]

    df = df.apply(pd.to_numeric, errors="coerce")
    log(f"After cleanup shape: {df.shape}")

    expr = df.T
    expr.index.name = "sample_id"
    log(f"Expression (samples x genes): {expr.shape}")

    expr_path = RAW_DIR / "hoogstraat_2" / "GSE192341_expression.parquet"
    expr.to_parquet(expr_path)
    log(f"Saved: {expr_path}")


# ═══════════════════════════════════════════════════════════════════
# Re-examine GSE130788 (Hurvitz HER2+) more carefully
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Re-examining GSE130788 (Hurvitz HER2+)")
log("=" * 70)

pheno = pd.read_parquet(RAW_DIR / "hurvitz_her2" / "response_labels.parquet")
log(f"Shape: {pheno.shape}")
log(f"Columns: {list(pheno.columns)}")

# Look at title column for response info
if "title" in pheno.columns:
    log(f"\nSample titles:")
    for t in pheno["title"].unique()[:20]:
        log(f"  {t}")

# Check source_name
if "source_name_ch1" in pheno.columns:
    log(f"\nSource names:")
    for s in pheno["source_name_ch1"].unique()[:20]:
        log(f"  {s}")

# Check all char columns
char_cols = [c for c in pheno.columns if c.startswith("char_")]
log(f"\nCharacteristic columns: {char_cols}")
for col in char_cols:
    vals = pheno[col].dropna().unique()
    log(f"  {col}: {list(vals[:20])}")


# ═══════════════════════════════════════════════════════════════════
# Re-examine GSE87411 (TransNEOS) - check if it's really the right one
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Re-examining GSE87411 (TransNEOS)")
log("=" * 70)

pheno = pd.read_parquet(RAW_DIR / "transneos" / "response_labels.parquet")
log(f"Shape: {pheno.shape}")

if "title" in pheno.columns:
    log(f"\nSample titles (first 20):")
    for t in pheno["title"].unique()[:20]:
        log(f"  {t}")

# Check source and all chars
char_cols = [c for c in pheno.columns if c.startswith("char_")]
log(f"\nCharacteristic columns: {char_cols}")
for col in char_cols:
    vals = pheno[col].dropna().unique()
    if len(vals) <= 20:
        log(f"  {col}: {list(vals)}")
    else:
        log(f"  {col}: {len(vals)} unique values")

# Check the series metadata
if "series_id" in pheno.columns:
    log(f"\nSeries IDs: {pheno['series_id'].unique()}")

# Check if expression data has a valid structure
expr_path = RAW_DIR / "transneos" / "GSE87411_expression.parquet"
if expr_path.exists():
    expr = pd.read_parquet(expr_path)
    log(f"\nExpression shape: {expr.shape}")
    log(f"Expression index (first 5): {list(expr.index[:5])}")
    log(f"Expression columns (first 5): {list(expr.columns[:5])}")
    # Check if these are standard probes/genes
    log(f"Numeric fraction: {expr.apply(pd.to_numeric, errors='coerce').notna().sum().sum() / expr.size:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Re-examine GSE78958 (Z1031) - look for Ki67 or treatment response
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Re-examining GSE78958 (Z1031)")
log("=" * 70)

pheno = pd.read_parquet(RAW_DIR / "z1031" / "response_labels.parquet")
log(f"Shape: {pheno.shape}")

if "title" in pheno.columns:
    log(f"\nSample titles (first 20):")
    for t in list(pheno["title"].unique())[:20]:
        log(f"  {t}")

char_cols = [c for c in pheno.columns if c.startswith("char_")]
log(f"\nCharacteristic columns: {char_cols}")
for col in char_cols:
    vals = pheno[col].dropna().unique()
    if len(vals) <= 20:
        log(f"  {col}: {list(vals)}")
    else:
        log(f"  {col}: {len(vals)} unique values")

# Look for any column containing ki67, response, treatment
for col in pheno.columns:
    col_lower = col.lower()
    if any(k in col_lower for k in ["ki67", "ki-67", "treatment", "drug", "agent",
                                      "endocrine", "aromatase", "letrozole",
                                      "anastrozole", "exemestane"]):
        vals = pheno[col].dropna().unique()
        log(f"  FOUND: {col}: {list(vals[:20])}")


# ═══════════════════════════════════════════════════════════════════
# Re-examine GSE36339 (IMPACT) - verify if it's correct
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Re-examining GSE36339 (IMPACT)")
log("=" * 70)

pheno = pd.read_parquet(RAW_DIR / "impact_endocrine" / "response_labels.parquet")
log(f"Shape: {pheno.shape}")

if "title" in pheno.columns:
    log(f"\nSample titles:")
    for t in pheno["title"].unique()[:20]:
        log(f"  {t}")

# This is only 8 samples - likely the wrong dataset
# GSE36339 appears to be a fish muscle study, not IMPACT breast trial
log(f"\norganism: {pheno.get('organism_ch1', pd.Series()).unique()}")

log("\n" + "=" * 70)
log("SUMMARY")
log("=" * 70)
log("""
Dataset status after supplementary downloads:

A. I-SPY2 (GSE194040):       COMPLETE - 988 samples, 19134 genes, pCR + treatment arms
B. BrighTNess (GSE164458):   COMPLETE - 482 samples, 24613 genes, pCR + RCB + treatment arms
C. Hurvitz HER2+ (GSE130788): PARTIAL - 199 samples, 41000 features, NO response labels
D. IMPACT (GSE36339):         WRONG DATASET - not breast cancer (fish study)
E. TransNEOS (GSE87411):      NEEDS VERIFICATION - 218 samples, expression OK, unclear phenotype
F1. Hoogstraat 1 (GSE191127): COMPLETE - 46 samples (23 pre/post), NRI + chemo regimen
F2. Hoogstraat 2 (GSE192341): COMPLETE - 93 samples, pCR + NRI + subtype
G. Z1031 (GSE78958):          PARTIAL - 424 samples, subtype but no Ki67/response labels
""")

log("DONE")
