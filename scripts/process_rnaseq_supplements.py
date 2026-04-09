#!/usr/bin/env python3
"""
Process supplementary RNA-seq count matrices for datasets where
expression was not in GSM tables.

Also re-check GSE145325 phenotype (char_group has responder/nonresponder).

Usage:
    pixi run python scripts/process_rnaseq_supplements.py
"""

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
# GSE145325 - letrozole RNA-seq raw counts
# ═══════════════════════════════════════════════════════════════════
log("=" * 70)
log("Processing GSE145325 (letrozole_rnaseq) raw counts")
log("=" * 70)

gz_path = RAW_DIR / "letrozole_rnaseq" / "GSE145325_rawcounts_valencia.csv.gz"
if gz_path.exists():
    df = pd.read_csv(gz_path, compression="gzip", index_col=0)
    log(f"Raw shape: {df.shape}")
    log(f"Index (first 5): {list(df.index[:5])}")
    log(f"Columns (first 5): {list(df.columns[:5])}")

    # Genes as rows, samples as columns -> transpose
    if df.shape[0] > df.shape[1]:
        expr = df.T
    else:
        expr = df
    expr.index.name = "sample_id"

    # Convert to numeric
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=1, how="all")
    # Remove duplicate column names
    expr = expr.loc[:, ~expr.columns.duplicated()]
    log(f"Expression (samples x genes): {expr.shape}")

    expr_path = RAW_DIR / "letrozole_rnaseq" / "GSE145325_expression.parquet"
    expr.to_parquet(expr_path)
    log(f"Saved: {expr_path}")

    # Re-check phenotype - char_group has response info
    pheno = pd.read_parquet(RAW_DIR / "letrozole_rnaseq" / "response_labels.parquet")
    char_cols = [c for c in pheno.columns if c.startswith("char_")]
    log(f"\nPhenotype char columns: {char_cols}")
    for col in char_cols:
        vals = pheno[col].dropna().unique()
        log(f"  {col}: {list(vals[:20])}")
else:
    log(f"File not found: {gz_path}")


# ═══════════════════════════════════════════════════════════════════
# GSE240671 - neoadjuvant chemotherapy RNA-seq raw counts
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Processing GSE240671 (neoadj_letrozole_rnaseq) raw counts")
log("=" * 70)

gz_path = RAW_DIR / "neoadj_letrozole_rnaseq" / "GSE240671_raw_count_all_libraries.csv.gz"
if gz_path.exists():
    df = pd.read_csv(gz_path, compression="gzip", index_col=0)
    log(f"Raw shape: {df.shape}")
    log(f"Index (first 5): {list(df.index[:5])}")
    log(f"Columns (first 5): {list(df.columns[:5])}")

    if df.shape[0] > df.shape[1]:
        expr = df.T
    else:
        expr = df
    expr.index.name = "sample_id"

    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=1, how="all")
    expr = expr.loc[:, ~expr.columns.duplicated()]
    log(f"Expression (samples x genes): {expr.shape}")

    expr_path = RAW_DIR / "neoadj_letrozole_rnaseq" / "GSE240671_expression.parquet"
    expr.to_parquet(expr_path)
    log(f"Saved: {expr_path}")

    pheno = pd.read_parquet(RAW_DIR / "neoadj_letrozole_rnaseq" / "response_labels.parquet")
    log(f"\nPhenotype RCB distribution:")
    if "char_rcb_category" in pheno.columns:
        log(f"  {pheno['char_rcb_category'].value_counts().to_dict()}")
    if "char_nac_category" in pheno.columns:
        log(f"  NAC type: {pheno['char_nac_category'].value_counts().to_dict()}")
    if "char_hormonotherapy_type_category" in pheno.columns:
        log(f"  Hormono: {pheno['char_hormonotherapy_type_category'].value_counts().to_dict()}")
else:
    log(f"File not found: {gz_path}")


log("\nDONE")
