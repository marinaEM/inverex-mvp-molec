"""
Gene Universe Discovery
=======================
Scans ALL expression datasets (CTR-DB, I-SPY2, BrighTNess, TCGA-BRCA) and
identifies genes present in >= 50% of datasets.  Keeps the top 2,000 by
prevalence (breaking ties alphabetically).

Outputs
-------
gene_universe : list[str]   – ordered list of 2,000 gene symbols
gene2idx      : dict         – gene symbol → integer index (0 = [CLS], 1…2000 = genes)
"""

from __future__ import annotations

import os, json, logging
from pathlib import Path
from typing import Optional

import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # inverex-mvp
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results" / "foundation"


def _load_dataset_genes(path: Path) -> set[str]:
    """Return gene symbols (columns) from a parquet expression file."""
    df = pd.read_parquet(path, columns=None)
    # Columns are gene symbols; index is sample_id
    genes = set(df.columns.tolist())
    # Remove metadata-like columns
    genes -= {"sample_id", "Sample", "sample"}
    return genes


def discover_all_datasets() -> list[tuple[str, Path]]:
    """Return (dataset_name, expression_path) for every available dataset."""
    datasets: list[tuple[str, Path]] = []

    # CTR-DB
    ctrdb = DATA_RAW / "ctrdb"
    if ctrdb.exists():
        for gse_dir in sorted(ctrdb.iterdir()):
            if not gse_dir.is_dir():
                continue
            expr_files = list(gse_dir.glob("*_expression.parquet"))
            if expr_files:
                datasets.append((gse_dir.name, expr_files[0]))

    # I-SPY2
    ispy2_expr = DATA_RAW / "ispy2" / "GSE194040_expression.parquet"
    if ispy2_expr.exists():
        datasets.append(("ISPY2", ispy2_expr))

    # BrighTNess
    brightness_expr = DATA_RAW / "brightness" / "GSE164458_expression.parquet"
    if brightness_expr.exists():
        datasets.append(("BrighTNess", brightness_expr))

    # TCGA-BRCA
    tcga_expr = DATA_CACHE / "tcga_brca_expression.parquet"
    if tcga_expr.exists():
        datasets.append(("TCGA_BRCA", tcga_expr))

    logger.info("Found %d expression datasets", len(datasets))
    return datasets


def _load_landmark_genes() -> list[str]:
    """Load landmark genes from geneinfo_beta_input.txt."""
    path = DATA_CACHE / "geneinfo_beta_input.txt"
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t")
    return [g for g in df["gene_symbol"].tolist() if isinstance(g, str) and len(g) > 0]


def build_gene_universe(
    min_prevalence: float = 0.50,
    max_genes: int = 2_000,
    cache_path: Optional[Path] = None,
) -> tuple[list[str], dict[str, int]]:
    """
    Build the gene universe: genes present in >= `min_prevalence` fraction
    of all datasets, capped at `max_genes`.

    Landmark genes that pass the prevalence threshold are always included
    (prioritised over non-landmark genes to ensure ESR1, ERBB2, MKI67, etc.
    are available for subtype inference and biological interpretation).

    Returns (gene_list, gene2idx) where gene2idx[gene] is 1-based
    (index 0 is reserved for [CLS]).
    """
    if cache_path is None:
        cache_path = RESULTS / "gene_universe.json"

    if cache_path.exists():
        logger.info("Loading cached gene universe from %s", cache_path)
        with open(cache_path) as f:
            data = json.load(f)
        gene_list = data["genes"]
        gene2idx = {g: i + 1 for i, g in enumerate(gene_list)}
        gene2idx["[CLS]"] = 0
        return gene_list, gene2idx

    datasets = discover_all_datasets()
    n_datasets = len(datasets)
    logger.info("Scanning %d datasets for gene prevalence …", n_datasets)

    gene_counts: dict[str, int] = {}
    for name, path in datasets:
        genes = _load_dataset_genes(path)
        for g in genes:
            gene_counts[g] = gene_counts.get(g, 0) + 1

    threshold = int(min_prevalence * n_datasets)
    logger.info(
        "Prevalence threshold: present in >= %d / %d datasets", threshold, n_datasets
    )

    # All genes passing the threshold
    eligible = {g: c for g, c in gene_counts.items() if c >= threshold}

    # Priority 1: landmark genes that pass threshold
    landmark = _load_landmark_genes()
    priority_genes = [g for g in landmark if g in eligible]
    priority_set = set(priority_genes)

    # Priority 2: remaining eligible genes by count desc, then alphabetical
    remaining = sorted(
        [g for g in eligible if g not in priority_set],
        key=lambda g: (-eligible[g], g),
    )

    gene_list = priority_genes + remaining
    gene_list = gene_list[:max_genes]

    logger.info(
        "Gene universe: %d eligible, %d landmark priority, kept top %d",
        len(eligible), len(priority_genes), len(gene_list),
    )

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"genes": gene_list, "n_datasets": n_datasets, "threshold": threshold}, f)

    gene2idx = {g: i + 1 for i, g in enumerate(gene_list)}
    gene2idx["[CLS]"] = 0
    return gene_list, gene2idx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    genes, g2i = build_gene_universe()
    print(f"Gene universe: {len(genes)} genes")
    print(f"First 20: {genes[:20]}")
    print(f"Last  20: {genes[-20:]}")
