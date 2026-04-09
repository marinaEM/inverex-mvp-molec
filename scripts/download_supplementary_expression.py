#!/usr/bin/env python3
"""
Download supplementary expression matrices for datasets where
expression data is not embedded in GSM tables.

Also re-examine phenotype data for response columns.

Usage:
    pixi run python scripts/download_supplementary_expression.py
"""

import os
import gzip
import urllib.request
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


def download_ftp(url, dest_path):
    """Download a file from FTP URL."""
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


def read_tsv_gz(path, **kwargs):
    """Read a gzipped TSV file."""
    return pd.read_csv(path, sep="\t", compression="gzip", **kwargs)


# ═══════════════════════════════════════════════════════════════════
# A. I-SPY2 (GSE194040) - gene-level expression matrix
# ═══════════════════════════════════════════════════════════════════
log("=" * 70)
log("A. I-SPY2 (GSE194040) - supplementary expression matrix")
log("=" * 70)

ispy2_dir = RAW_DIR / "ispy2"
ispy2_dir.mkdir(parents=True, exist_ok=True)

# Gene-level expression (most useful)
ispy2_url = ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194040/suppl/"
             "GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt.gz")
ispy2_gz = ispy2_dir / "GSE194040_gene_level_expression.txt.gz"
if download_ftp(ispy2_url, ispy2_gz):
    try:
        df = read_tsv_gz(ispy2_gz, index_col=0)
        log(f"  Raw shape: {df.shape}")
        # This is genes x samples; transpose to samples x genes
        if df.shape[0] > df.shape[1]:
            # genes are rows, samples are columns
            expr = df.T
        else:
            expr = df
        expr.index.name = "sample_id"
        log(f"  Expression shape (samples x genes): {expr.shape}")

        # Save as parquet
        expr_path = ispy2_dir / "GSE194040_expression.parquet"
        expr.to_parquet(expr_path)
        log(f"  Saved: {expr_path}")
    except Exception as e:
        log(f"  ERROR parsing expression: {e}")
        import traceback; traceback.print_exc()

# Now examine phenotype more carefully
log("\n  Examining I-SPY2 phenotype data...")
pheno = pd.read_parquet(ispy2_dir / "response_labels.parquet")
log(f"  Phenotype shape: {pheno.shape}")

# Key response/treatment columns
for col in pheno.columns:
    if any(kw in col.lower() for kw in ["pcr", "arm", "hr", "her2", "subtype",
                                          "treatment", "response", "rcb"]):
        vals = pheno[col].dropna().unique()
        log(f"  {col}: {list(vals[:15])} (n_unique={len(vals)})")

# Parse treatment arms
if "char_arm" in pheno.columns:
    log(f"\n  Treatment arm distribution:")
    arm_counts = pheno["char_arm"].value_counts()
    for arm, n in arm_counts.items():
        log(f"    {arm}: {n}")

if "char_pcr" in pheno.columns:
    log(f"\n  pCR distribution:")
    pcr_counts = pheno["char_pcr"].value_counts()
    for pcr, n in pcr_counts.items():
        log(f"    {pcr}: {n}")


# ═══════════════════════════════════════════════════════════════════
# B. BrighTNess (GSE164458) - RNA-seq expression matrix
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("B. BrighTNess (GSE164458) - supplementary expression matrix")
log("=" * 70)

bright_dir = RAW_DIR / "brightness"
bright_dir.mkdir(parents=True, exist_ok=True)

bright_url = ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164458/suppl/"
              "GSE164458_BrighTNess_RNAseq_log2_Processed_ASTOR.txt.gz")
bright_gz = bright_dir / "GSE164458_expression_rnaseq.txt.gz"
if download_ftp(bright_url, bright_gz):
    try:
        df = read_tsv_gz(bright_gz, index_col=0)
        log(f"  Raw shape: {df.shape}")
        # Determine orientation
        if df.shape[0] > df.shape[1]:
            expr = df.T
        else:
            expr = df
        expr.index.name = "sample_id"
        log(f"  Expression shape (samples x genes): {expr.shape}")
        expr_path = bright_dir / "GSE164458_expression.parquet"
        expr.to_parquet(expr_path)
        log(f"  Saved: {expr_path}")
    except Exception as e:
        log(f"  ERROR parsing expression: {e}")
        import traceback; traceback.print_exc()

# Phenotype
log("\n  Examining BrighTNess phenotype data...")
pheno = pd.read_parquet(bright_dir / "response_labels.parquet")
for col in pheno.columns:
    if any(kw in col.lower() for kw in ["pcr", "arm", "rcb", "pathologic",
                                          "response", "treatment", "residual",
                                          "planned"]):
        vals = pheno[col].dropna().unique()
        log(f"  {col}: {list(vals[:20])} (n_unique={len(vals)})")


# ═══════════════════════════════════════════════════════════════════
# F1. Hoogstraat 1 (GSE191127) - read counts
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("F1. Hoogstraat 1 (GSE191127) - supplementary expression matrix")
log("=" * 70)

hoog1_dir = RAW_DIR / "hoogstraat_1"
hoog1_dir.mkdir(parents=True, exist_ok=True)

hoog1_url = ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE191nnn/GSE191127/suppl/"
             "GSE191127_readcounts_prepost.txt.gz")
hoog1_gz = hoog1_dir / "GSE191127_readcounts.txt.gz"
if download_ftp(hoog1_url, hoog1_gz):
    try:
        df = read_tsv_gz(hoog1_gz, index_col=0)
        log(f"  Raw shape: {df.shape}")
        if df.shape[0] > df.shape[1]:
            expr = df.T
        else:
            expr = df
        expr.index.name = "sample_id"
        log(f"  Expression shape (samples x genes): {expr.shape}")
        expr_path = hoog1_dir / "GSE191127_expression.parquet"
        expr.to_parquet(expr_path)
        log(f"  Saved: {expr_path}")
    except Exception as e:
        log(f"  ERROR parsing expression: {e}")
        import traceback; traceback.print_exc()

# Phenotype
log("\n  Examining Hoogstraat 1 phenotype data...")
pheno = pd.read_parquet(hoog1_dir / "response_labels.parquet")
for col in pheno.columns:
    if any(kw in col.lower() for kw in ["chemo", "response", "nri", "subtype",
                                          "therapy", "treatment", "pcr"]):
        vals = pheno[col].dropna().unique()
        log(f"  {col}: {list(vals[:20])} (n_unique={len(vals)})")


# ═══════════════════════════════════════════════════════════════════
# F2. Hoogstraat 2 (GSE192341) - processed data
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("F2. Hoogstraat 2 (GSE192341) - supplementary expression matrix")
log("=" * 70)

hoog2_dir = RAW_DIR / "hoogstraat_2"
hoog2_dir.mkdir(parents=True, exist_ok=True)

hoog2_url = ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE192nnn/GSE192341/suppl/"
             "GSE192341_processed_data.txt.gz")
hoog2_gz = hoog2_dir / "GSE192341_processed_data.txt.gz"
if download_ftp(hoog2_url, hoog2_gz):
    try:
        df = read_tsv_gz(hoog2_gz, index_col=0)
        log(f"  Raw shape: {df.shape}")
        if df.shape[0] > df.shape[1]:
            expr = df.T
        else:
            expr = df
        expr.index.name = "sample_id"
        log(f"  Expression shape (samples x genes): {expr.shape}")
        expr_path = hoog2_dir / "GSE192341_expression.parquet"
        expr.to_parquet(expr_path)
        log(f"  Saved: {expr_path}")
    except Exception as e:
        log(f"  ERROR parsing expression: {e}")
        import traceback; traceback.print_exc()

# Phenotype
log("\n  Examining Hoogstraat 2 phenotype data...")
pheno = pd.read_parquet(hoog2_dir / "response_labels.parquet")
for col in pheno.columns:
    if any(kw in col.lower() for kw in ["pcr", "response", "nri", "subtype",
                                          "therapy", "treatment", "chemo"]):
        vals = pheno[col].dropna().unique()
        log(f"  {col}: {list(vals[:20])} (n_unique={len(vals)})")


# ═══════════════════════════════════════════════════════════════════
# Re-examine datasets that appeared to have no response labels
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Re-examining phenotype data for 'no response' datasets")
log("=" * 70)

# C. Hurvitz HER2+ (GSE130788)
log("\n--- GSE130788 (hurvitz_her2) ---")
pheno = pd.read_parquet(RAW_DIR / "hurvitz_her2" / "response_labels.parquet")
char_cols = [c for c in pheno.columns if c.startswith("char_")]
for col in char_cols:
    vals = pheno[col].dropna().unique()
    log(f"  {col}: {list(vals[:15])} (n_unique={len(vals)})")

# D. IMPACT endocrine (GSE36339)
log("\n--- GSE36339 (impact_endocrine) ---")
pheno = pd.read_parquet(RAW_DIR / "impact_endocrine" / "response_labels.parquet")
char_cols = [c for c in pheno.columns if c.startswith("char_")]
for col in char_cols:
    vals = pheno[col].dropna().unique()
    log(f"  {col}: {list(vals[:15])} (n_unique={len(vals)})")
# Also show all columns
log(f"  All columns: {list(pheno.columns)}")

# E. TransNEOS (GSE87411)
log("\n--- GSE87411 (transneos) ---")
pheno = pd.read_parquet(RAW_DIR / "transneos" / "response_labels.parquet")
char_cols = [c for c in pheno.columns if c.startswith("char_")]
for col in char_cols:
    vals = pheno[col].dropna().unique()
    if len(vals) <= 15:
        log(f"  {col}: {list(vals)}")
    else:
        log(f"  {col}: {len(vals)} unique values (sample: {list(vals[:5])})")

# G. Z1031 (GSE78958)
log("\n--- GSE78958 (z1031) ---")
pheno = pd.read_parquet(RAW_DIR / "z1031" / "response_labels.parquet")
char_cols = [c for c in pheno.columns if c.startswith("char_")]
for col in char_cols:
    vals = pheno[col].dropna().unique()
    if len(vals) <= 15:
        log(f"  {col}: {list(vals)}")
    else:
        log(f"  {col}: {len(vals)} unique values (sample: {list(vals[:5])})")


log("\n" + "=" * 70)
log("DONE - supplementary expression downloads")
log("=" * 70)
