#!/usr/bin/env python3
"""
Fix GSE240671 count matrix parsing (tab-delimited within CSV).

Usage:
    pixi run python scripts/fix_gse240671.py
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


gz_path = RAW_DIR / "neoadj_letrozole_rnaseq" / "GSE240671_raw_count_all_libraries.csv.gz"

log("Parsing GSE240671 with tab separator...")
# Try tab separator
df = pd.read_csv(gz_path, compression="gzip", sep="\t", index_col=0)
log(f"Shape with tab sep: {df.shape}")
log(f"Index (first 3): {list(df.index[:3])}")
log(f"Columns (first 5): {list(df.columns[:5])}")

if df.shape[1] > 1:
    # Check for metadata columns (chr, start, end, ensembl_id, etc.)
    # Drop non-numeric columns
    log(f"Column dtypes: {df.dtypes.value_counts().to_dict()}")

    # The data may have columns: Gene, Chr, Coords, EnsemblID, then sample columns
    # Check first few columns
    log(f"First 5 columns data types and samples:")
    for col in df.columns[:8]:
        log(f"  {col}: dtype={df[col].dtype}, sample={df[col].iloc[0]}")

    # Find where numeric sample columns start
    numeric_start = None
    for i, col in enumerate(df.columns):
        try:
            pd.to_numeric(df[col], errors="raise")
            numeric_start = i
            break
        except (ValueError, TypeError):
            continue

    if numeric_start is not None:
        log(f"Numeric columns start at index {numeric_start}: {df.columns[numeric_start]}")
        # Keep only numeric columns (these are samples)
        sample_cols = df.columns[numeric_start:]
        expr_raw = df[sample_cols]
        expr_raw = expr_raw.apply(pd.to_numeric, errors="coerce")
        log(f"Expression (genes x samples): {expr_raw.shape}")

        # Transpose to samples x genes
        expr = expr_raw.T
        expr.index.name = "sample_id"
        expr = expr.loc[:, ~expr.columns.duplicated()]
        log(f"Expression (samples x genes): {expr.shape}")

        expr_path = RAW_DIR / "neoadj_letrozole_rnaseq" / "GSE240671_expression.parquet"
        expr.to_parquet(expr_path)
        log(f"Saved: {expr_path}")
    else:
        log("ERROR: Could not find numeric columns")
else:
    log("Tab separator didn't work either")
    # Maybe semi-colon or other delimiter
    log("Trying auto-detect...")
    import gzip
    with gzip.open(gz_path, "rt") as f:
        for i, line in enumerate(f):
            if i < 3:
                log(f"  Line {i}: {line[:200]}")
            else:
                break

log("DONE")
