"""
Gene Universe Discovery (Full)
==============================
Scans ALL expression datasets (CTR-DB 36+, I-SPY2, BrighTNess, TCGA-BRCA) and
identifies genes present in >= 40% of datasets.  No cap -- keeps all eligible
genes (~5,000-8,000 expected).

Outputs
-------
gene_universe : list[str]   -- ordered list of gene symbols
gene2idx      : dict         -- gene symbol -> integer index (0 = [CLS], 1..G = genes)
dataset_info  : dict         -- per-dataset metadata (n_samples, n_genes, platform)
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
RESULTS = ROOT / "results" / "agent_c_full"


def _load_dataset_genes(path: Path) -> set[str]:
    """Return gene symbols (columns) from a parquet expression file."""
    df = pd.read_parquet(path, columns=None)
    genes = set(df.columns.tolist())
    # Remove metadata-like columns
    genes -= {"sample_id", "Sample", "sample", "index", "Unnamed: 0"}
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


def discover_labeled_datasets() -> list[tuple[str, Path, Path]]:
    """Find datasets that have both expression + response labels."""
    labeled = []

    # CTR-DB
    ctrdb = DATA_RAW / "ctrdb"
    if ctrdb.exists():
        for gse_dir in sorted(ctrdb.iterdir()):
            if not gse_dir.is_dir():
                continue
            expr_files = list(gse_dir.glob("*_expression.parquet"))
            resp_file = gse_dir / "response_labels.parquet"
            if expr_files and resp_file.exists():
                labeled.append((gse_dir.name, expr_files[0], resp_file))

    # I-SPY2
    ispy2_expr = DATA_RAW / "ispy2" / "GSE194040_expression.parquet"
    ispy2_resp = DATA_RAW / "ispy2" / "response_labels.parquet"
    if ispy2_expr.exists() and ispy2_resp.exists():
        labeled.append(("ISPY2", ispy2_expr, ispy2_resp))

    # BrighTNess
    br_expr = DATA_RAW / "brightness" / "GSE164458_expression.parquet"
    br_resp = DATA_RAW / "brightness" / "response_labels.parquet"
    if br_expr.exists() and br_resp.exists():
        labeled.append(("BrighTNess", br_expr, br_resp))

    return labeled


def build_gene_universe(
    min_prevalence: float = 0.40,
    max_genes: int = 1_000,
    cache_path: Optional[Path] = None,
) -> tuple[list[str], dict[str, int], dict]:
    """
    Build the FULL gene universe: genes present in >= `min_prevalence` fraction
    of all datasets, no hard cap (max_genes is a safety limit).

    Returns (gene_list, gene2idx, info_dict) where gene2idx[gene] is 1-based
    (index 0 is reserved for [CLS]).
    """
    if cache_path is None:
        cache_path = RESULTS / "gene_universe_full.json"

    if cache_path.exists():
        logger.info("Loading cached gene universe from %s", cache_path)
        with open(cache_path) as f:
            data = json.load(f)
        gene_list = data["genes"]
        gene2idx = {g: i + 1 for i, g in enumerate(gene_list)}
        gene2idx["[CLS]"] = 0
        info = {
            "n_genes": len(gene_list),
            "n_datasets": data["n_datasets"],
            "threshold": data["threshold"],
            "dataset_info": data.get("dataset_info", {}),
        }
        return gene_list, gene2idx, info

    datasets = discover_all_datasets()
    n_datasets = len(datasets)
    logger.info("Scanning %d datasets for gene prevalence ...", n_datasets)

    gene_counts: dict[str, int] = {}
    dataset_info: dict[str, dict] = {}
    for name, path in datasets:
        try:
            genes = _load_dataset_genes(path)
            df = pd.read_parquet(path)
            dataset_info[name] = {
                "n_samples": len(df),
                "n_genes": len(genes),
                "path": str(path),
            }
            for g in genes:
                gene_counts[g] = gene_counts.get(g, 0) + 1
            logger.info("  %s: %d samples, %d genes", name, len(df), len(genes))
        except Exception as e:
            logger.warning("  Failed to load %s: %s", name, e)

    threshold = max(1, int(min_prevalence * n_datasets))
    logger.info(
        "Prevalence threshold: present in >= %d / %d datasets (%.0f%%)",
        threshold, n_datasets, min_prevalence * 100,
    )

    # Filter and sort: by count descending, then alphabetical
    eligible = {g: c for g, c in gene_counts.items() if c >= threshold}

    # Priority genes: ensure these are always included if eligible
    PRIORITY_GENES = [
        "ESR1", "PGR", "ERBB2", "EGFR", "MKI67", "TP53", "PIK3CA",
        "AKT1", "MTOR", "PTEN", "CDH1", "GATA3", "MAP3K1", "BRCA1",
        "BRCA2", "RB1", "CCND1", "CCNE1", "CDK4", "CDK6",
        "AURKA", "AURKB", "TOP2A", "TYMS", "BCL2", "BIRC5",
        "GRB7", "FOXA1", "FOXC1", "KRT5", "KRT14", "KRT17",
        "VIM", "SNAI1", "SNAI2", "ZEB1", "TWIST1", "CDH2",
        "MYC", "CTNNB1", "NOTCH1", "JAK1", "JAK2", "STAT3",
        "AR", "VEGFA", "KDR", "PDCD1", "CD274", "CTLA4",
    ]
    priority_set = {g for g in PRIORITY_GENES if g in eligible}

    sorted_genes = sorted(eligible.keys(), key=lambda g: (-eligible[g], g))
    # Place priority genes first, then fill remaining
    gene_list = sorted([g for g in priority_set], key=lambda g: (-eligible[g], g))
    remaining = [g for g in sorted_genes if g not in priority_set]
    gene_list = gene_list + remaining[:max(max_genes - len(gene_list), 0)]

    logger.info(
        "Gene universe: %d eligible genes (kept %d)", len(eligible), len(gene_list)
    )

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {
                "genes": gene_list,
                "n_datasets": n_datasets,
                "threshold": threshold,
                "min_prevalence": min_prevalence,
                "n_eligible": len(eligible),
                "dataset_info": dataset_info,
            },
            f,
            indent=2,
        )

    gene2idx = {g: i + 1 for i, g in enumerate(gene_list)}
    gene2idx["[CLS]"] = 0
    info = {
        "n_genes": len(gene_list),
        "n_datasets": n_datasets,
        "threshold": threshold,
        "dataset_info": dataset_info,
    }
    return gene_list, gene2idx, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    genes, g2i, info = build_gene_universe()
    print(f"Gene universe: {len(genes)} genes from {info['n_datasets']} datasets")
    print(f"Threshold: present in >= {info['threshold']} datasets")
    print(f"First 30: {genes[:30]}")
    print(f"Last  10: {genes[-10:]}")
