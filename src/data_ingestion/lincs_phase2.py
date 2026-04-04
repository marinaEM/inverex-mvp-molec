"""
LINCS Phase 2 (GSE70138) data ingestion and comparison with Phase 1 (GSE92742).

Phase 1 (GSE92742): 473k signatures, ~76 cell lines, ~20k compounds.
Phase 2 (GSE70138): 118k signatures, ~30 cell lines, ~1.8k compounds.

This module:
  1. Downloads Phase 2 metadata (sig_info, pert_info, gene_info)
  2. Compares Phase 1 vs Phase 2: new compounds, cell lines, breast sigs
  3. Identifies new GDSC2 drug matches from Phase 2
  4. If significant: downloads Phase 2 GCTX, extracts breast sigs, merges
  5. Re-evaluates LINCS x GDSC2 matching with merged data
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BREAST_CELL_IDS_LINCS, DATA_CACHE, RESULTS
from src.data_ingestion.utils import download_file
from src.data_ingestion.pharmacodb import (
    _DRUG_SYNONYM_TABLE,
    _normalize_drug,
    _normalize_cell,
)

logger = logging.getLogger(__name__)

# ── Phase 2 URLs (files have date-stamped names) ────────────────────────
_P2_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/"
P2_SIGINFO_URL = _P2_BASE + "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
P2_GENEINFO_URL = _P2_BASE + "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
P2_PERTINFO_URL = _P2_BASE + "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz"
P2_LEVEL5_GCTX_URL = (
    _P2_BASE
    + "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"
)


# ── Download helpers ─────────────────────────────────────────────────────

def _ensure_p2_metadata(cache_dir: Path = DATA_CACHE) -> dict[str, Path]:
    """Download Phase 2 metadata files if not already cached.

    Returns dict mapping 'sig_info', 'pert_info', 'gene_info' to local paths.
    """
    files = {
        "sig_info": (P2_SIGINFO_URL, cache_dir / "GSE70138_sig_info.txt.gz"),
        "gene_info": (P2_GENEINFO_URL, cache_dir / "GSE70138_gene_info.txt.gz"),
        "pert_info": (P2_PERTINFO_URL, cache_dir / "GSE70138_pert_info.txt.gz"),
    }
    paths: dict[str, Path] = {}
    for key, (url, dest) in files.items():
        if dest.exists():
            logger.info(f"Phase 2 {key} already cached: {dest.name}")
        else:
            logger.info(f"Downloading Phase 2 {key}...")
            download_file(url, dest)
        paths[key] = dest
    return paths


# ── Load helpers ─────────────────────────────────────────────────────────

def load_p2_siginfo(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """Load Phase 2 signature metadata."""
    parquet = cache_dir / "GSE70138_sig_info.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    meta_paths = _ensure_p2_metadata(cache_dir)
    df = pd.read_csv(meta_paths["sig_info"], sep="\t", low_memory=False)
    df.to_parquet(parquet, index=False)
    logger.info(f"Phase 2 sig_info: {df.shape}")
    return df


def load_p2_pertinfo(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """Load Phase 2 perturbagen info."""
    parquet = cache_dir / "GSE70138_pert_info.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    meta_paths = _ensure_p2_metadata(cache_dir)
    df = pd.read_csv(meta_paths["pert_info"], sep="\t", low_memory=False)
    df.to_parquet(parquet, index=False)
    logger.info(f"Phase 2 pert_info: {df.shape}")
    return df


def load_p2_geneinfo(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """Load Phase 2 gene info."""
    parquet = cache_dir / "GSE70138_gene_info.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    meta_paths = _ensure_p2_metadata(cache_dir)
    df = pd.read_csv(meta_paths["gene_info"], sep="\t", low_memory=False)
    df.to_parquet(parquet, index=False)
    logger.info(f"Phase 2 gene_info: {df.shape}")
    return df


# ── Drug alias helpers ───────────────────────────────────────────────────

def _build_bidirectional_alias() -> dict[str, str]:
    """Build a normalised alias map from the curated synonym table.

    Returns a dict that maps *both* directions:
        norm(gdsc_name) -> norm(lincs_name)   AND
        norm(lincs_name) -> norm(gdsc_name)
    so we can look up in either direction.
    """
    alias: dict[str, str] = {}
    for gdsc_orig, lincs_orig in _DRUG_SYNONYM_TABLE.items():
        gn = _normalize_drug(gdsc_orig)
        ln = _normalize_drug(lincs_orig)
        alias[gn] = ln
        alias[ln] = gn
    return alias


def _match_drug_sets(
    lincs_norms: set[str],
    gdsc_norms: set[str],
) -> set[str]:
    """Return the set of normalised GDSC drug names that match LINCS.

    Uses direct normalised matching + the curated synonym table.
    """
    alias = _build_bidirectional_alias()
    matched: set[str] = set()
    for gn in gdsc_norms:
        # Direct match
        if gn in lincs_norms:
            matched.add(gn)
            continue
        # Alias match (GDSC -> LINCS direction)
        ln = alias.get(gn)
        if ln and ln in lincs_norms:
            matched.add(gn)
    return matched


# ── Core analysis ────────────────────────────────────────────────────────

def analyse_phase2(
    cache_dir: Path = DATA_CACHE,
    results_dir: Path = RESULTS,
) -> pd.DataFrame:
    """Run the full Phase 1 vs Phase 2 comparison.

    Steps:
      1. Load Phase 1 & Phase 2 signature metadata.
      2. Compare compounds, cell lines, breast-cancer signatures.
      3. Check which new Phase 2 compounds match GDSC2 drugs.
      4. Save results/lincs_phase2_analysis.csv.

    Returns a summary DataFrame.
    """
    logger.info("=" * 65)
    logger.info("LINCS Phase 2 (GSE70138) analysis")
    logger.info("=" * 65)

    # ── Load Phase 1 ─────────────────────────────────────────────────
    logger.info("Loading Phase 1 (GSE92742) metadata...")
    p1_sig = pd.read_parquet(cache_dir / "GSE92742_sig_info.parquet")
    p1_pert = pd.read_parquet(cache_dir / "GSE92742_pert_info.parquet")

    # ── Load Phase 2 ─────────────────────────────────────────────────
    logger.info("Loading Phase 2 (GSE70138) metadata...")
    p2_sig = load_p2_siginfo(cache_dir)
    p2_pert = load_p2_pertinfo(cache_dir)

    # ── Load GDSC2 breast reference ──────────────────────────────────
    gdsc_path = cache_dir / "breast_dose_response_ref.parquet"
    if gdsc_path.exists():
        gdsc = pd.read_parquet(gdsc_path)
        gdsc_drug_norms = set(gdsc["drug_name"].apply(_normalize_drug).unique())
        n_gdsc_drugs = gdsc["drug_name"].nunique()
    else:
        logger.warning("No GDSC2 breast reference found; drug matching skipped.")
        gdsc = pd.DataFrame()
        gdsc_drug_norms = set()
        n_gdsc_drugs = 0

    # ── Filter to compound perturbagens ──────────────────────────────
    p1_cp = p1_sig[p1_sig["pert_type"] == "trt_cp"]
    p2_cp = p2_sig[p2_sig["pert_type"] == "trt_cp"]

    # ── Global compound / cell-line comparison ───────────────────────
    p1_compounds = set(p1_cp["pert_iname"].unique())
    p2_compounds = set(p2_cp["pert_iname"].unique())
    p1_cells = set(p1_cp["cell_id"].unique())
    p2_cells = set(p2_cp["cell_id"].unique())

    shared_compounds = p1_compounds & p2_compounds
    new_p2_compounds = p2_compounds - p1_compounds
    new_p2_cells = p2_cells - p1_cells

    logger.info(f"Phase 1 total compound sigs: {len(p1_cp):,}")
    logger.info(f"Phase 2 total compound sigs: {len(p2_cp):,}")
    logger.info(f"Phase 1 compounds: {len(p1_compounds):,}")
    logger.info(f"Phase 2 compounds: {len(p2_compounds):,}")
    logger.info(f"Shared compounds: {len(shared_compounds):,}")
    logger.info(f"New Phase 2 compounds: {len(new_p2_compounds):,}")
    logger.info(f"Phase 1 cell lines: {len(p1_cells)}")
    logger.info(f"Phase 2 cell lines: {len(p2_cells)}")
    logger.info(f"New Phase 2 cell lines: {len(new_p2_cells)}")

    # ── Breast-cancer-specific comparison ────────────────────────────
    breast_ids = BREAST_CELL_IDS_LINCS

    p1_br = p1_cp[p1_cp["cell_id"].isin(breast_ids)]
    p2_br = p2_cp[p2_cp["cell_id"].isin(breast_ids)]
    p2_br_24h = p2_br[p2_br["pert_itime"].astype(str).str.contains("24")]

    p1_br_compounds_norm = set(p1_br["pert_iname"].apply(_normalize_drug).unique())
    p2_br_compounds_norm = set(p2_br["pert_iname"].apply(_normalize_drug).unique())
    new_p2_br_norm = p2_br_compounds_norm - p1_br_compounds_norm

    p1_br_cells = sorted(p1_br["cell_id"].unique())
    p2_br_cells = sorted(p2_br["cell_id"].unique())

    logger.info("")
    logger.info("── Breast-cancer cell lines ──")
    logger.info(f"Phase 1 breast sigs: {len(p1_br):,}")
    logger.info(f"Phase 2 breast sigs: {len(p2_br):,}")
    logger.info(f"Phase 2 breast 24h sigs: {len(p2_br_24h):,}")
    logger.info(f"Phase 1 breast compounds: {len(p1_br_compounds_norm):,}")
    logger.info(f"Phase 2 breast compounds: {len(p2_br_compounds_norm):,}")
    logger.info(f"New breast compounds from Phase 2: {len(new_p2_br_norm):,}")
    logger.info(f"Phase 1 breast cell lines: {p1_br_cells}")
    logger.info(f"Phase 2 breast cell lines: {p2_br_cells}")

    # ── GDSC2 drug matching ──────────────────────────────────────────
    # Phase 1 alone
    p1_gdsc_matched = _match_drug_sets(p1_br_compounds_norm, gdsc_drug_norms)
    # Combined Phase 1 + Phase 2
    combined_br_norm = p1_br_compounds_norm | p2_br_compounds_norm
    combined_gdsc_matched = _match_drug_sets(combined_br_norm, gdsc_drug_norms)
    # New from Phase 2
    new_gdsc_from_p2 = combined_gdsc_matched - p1_gdsc_matched

    # Count sigs for the new matching drugs
    new_gdsc_lincs_norms = set()
    alias = _build_bidirectional_alias()
    for gn in new_gdsc_from_p2:
        # Find the LINCS-side normalised name
        ln = alias.get(gn, gn)
        if ln in new_p2_br_norm:
            new_gdsc_lincs_norms.add(ln)
        if gn in new_p2_br_norm:
            new_gdsc_lincs_norms.add(gn)

    new_match_sigs = p2_br_24h[
        p2_br_24h["pert_iname"].apply(_normalize_drug).isin(
            new_gdsc_lincs_norms | new_gdsc_from_p2
        )
    ]

    logger.info("")
    logger.info("── GDSC2 drug matching ──")
    logger.info(f"GDSC2 breast drugs: {n_gdsc_drugs}")
    logger.info(
        f"GDSC2 drugs matched by Phase 1 alone: {len(p1_gdsc_matched)}"
    )
    logger.info(
        f"GDSC2 drugs matched by combined P1+P2: {len(combined_gdsc_matched)}"
    )
    logger.info(f"New GDSC2 matches from Phase 2: {len(new_gdsc_from_p2)}")
    logger.info(
        f"New matching 24h breast sigs: {len(new_match_sigs)}"
    )

    if new_gdsc_from_p2:
        # Recover original GDSC drug names
        gdsc_norm_to_orig = {}
        if len(gdsc) > 0:
            for name in gdsc["drug_name"].unique():
                gdsc_norm_to_orig[_normalize_drug(name)] = name
        logger.info("New GDSC2 drugs from Phase 2:")
        for gn in sorted(new_gdsc_from_p2):
            orig = gdsc_norm_to_orig.get(gn, gn)
            logger.info(f"  {orig}")

    # ── Estimate impact on training data ─────────────────────────────
    existing_matched_path = cache_dir / "lincs_pharmacodb_matched.parquet"
    if existing_matched_path.exists():
        existing = pd.read_parquet(existing_matched_path)
        old_n_samples = len(existing)
        old_n_drugs = existing["pert_iname"].nunique()
    else:
        old_n_samples = 0
        old_n_drugs = 0

    logger.info("")
    logger.info("── Training data impact estimate ──")
    logger.info(f"Current matched samples: {old_n_samples}")
    logger.info(f"Current matched drugs: {old_n_drugs}")
    logger.info(
        f"Estimated new samples from Phase 2: ~{len(new_match_sigs)}"
    )
    logger.info(
        f"Estimated new total: ~{old_n_samples + len(new_match_sigs)} samples, "
        f"~{old_n_drugs + len(new_gdsc_from_p2)} drugs"
    )

    # ── Build summary table ──────────────────────────────────────────
    rows = [
        ("Phase 1 total signatures", len(p1_sig), "GSE92742"),
        ("Phase 2 total signatures", len(p2_sig), "GSE70138"),
        ("Phase 1 compound signatures", len(p1_cp), "trt_cp only"),
        ("Phase 2 compound signatures", len(p2_cp), "trt_cp only"),
        ("Phase 1 compounds (global)", len(p1_compounds), "unique pert_iname"),
        ("Phase 2 compounds (global)", len(p2_compounds), "unique pert_iname"),
        ("Shared compounds", len(shared_compounds), "in both phases"),
        ("New compounds (Phase 2 only)", len(new_p2_compounds), "global"),
        ("Phase 1 cell lines (global)", len(p1_cells), ""),
        ("Phase 2 cell lines (global)", len(p2_cells), ""),
        ("New cell lines (Phase 2)", len(new_p2_cells), str(sorted(new_p2_cells))[:100] if new_p2_cells else "none"),
        ("Phase 1 breast sigs", len(p1_br), ", ".join(p1_br_cells)),
        ("Phase 2 breast sigs", len(p2_br), ", ".join(p2_br_cells)),
        ("Phase 2 breast 24h sigs", len(p2_br_24h), "24h timepoint"),
        ("Phase 1 breast compounds", len(p1_br_compounds_norm), "normalised unique"),
        ("Phase 2 breast compounds", len(p2_br_compounds_norm), "normalised unique"),
        ("New breast compounds (Phase 2)", len(new_p2_br_norm), "not in Phase 1 breast"),
        ("GDSC2 breast drugs", n_gdsc_drugs, "from breast_dose_response_ref"),
        ("GDSC2 matched (Phase 1 only)", len(p1_gdsc_matched), "with alias table"),
        ("GDSC2 matched (P1 + P2)", len(combined_gdsc_matched), "with alias table"),
        ("New GDSC2 matches from Phase 2", len(new_gdsc_from_p2), "incremental"),
        ("New matching breast 24h sigs", len(new_match_sigs), "from Phase 2"),
        ("Current training samples", old_n_samples, "lincs_pharmacodb_matched"),
        ("Current training drugs", old_n_drugs, ""),
        ("Estimated new training samples", old_n_samples + len(new_match_sigs), "after Phase 2 merge"),
        ("Estimated new training drugs", old_n_drugs + len(new_gdsc_from_p2), "after Phase 2 merge"),
    ]

    summary = pd.DataFrame(rows, columns=["metric", "value", "note"])

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "lincs_phase2_analysis.csv"
    summary.to_csv(out_path, index=False)
    logger.info(f"\nSaved analysis to {out_path}")

    return summary


# ── Phase 2 GCTX download (only if worthwhile) ──────────────────────────

def should_download_gctx(
    summary: pd.DataFrame,
    min_new_drugs: int = 5,
    min_new_sigs: int = 50,
) -> bool:
    """Decide whether downloading the 4 GB Phase 2 GCTX is justified.

    Criteria:
      - At least `min_new_drugs` new GDSC2-matched drugs from Phase 2
      - At least `min_new_sigs` new matching 24h breast signatures
    """
    new_drugs = int(
        summary.loc[summary["metric"] == "New GDSC2 matches from Phase 2", "value"].iloc[0]
    )
    new_sigs = int(
        summary.loc[summary["metric"] == "New matching breast 24h sigs", "value"].iloc[0]
    )
    decision = new_drugs >= min_new_drugs and new_sigs >= min_new_sigs
    logger.info(
        f"GCTX download decision: new_drugs={new_drugs} (min {min_new_drugs}), "
        f"new_sigs={new_sigs} (min {min_new_sigs}) -> {'YES' if decision else 'NO'}"
    )
    return decision


def download_p2_gctx(cache_dir: Path = DATA_CACHE) -> Path:
    """Download the Phase 2 Level 5 GCTX (~4 GB compressed)."""
    dest = cache_dir / "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx.gz"
    if dest.exists():
        logger.info(f"Phase 2 GCTX already cached ({dest.stat().st_size:,} bytes)")
        return dest
    logger.info("Downloading Phase 2 Level 5 GCTX (~4 GB)...")
    download_file(P2_LEVEL5_GCTX_URL, dest, timeout=600)
    logger.info(f"Phase 2 GCTX downloaded: {dest.stat().st_size:,} bytes")
    return dest


def extract_p2_breast_signatures(
    cache_dir: Path = DATA_CACHE,
) -> pd.DataFrame:
    """Extract breast-cancer 24h compound signatures from Phase 2 GCTX.

    Mirrors the Phase 1 extraction in lincs.py but for Phase 2.
    Returns a DataFrame compatible with the Phase 1 breast signature matrix.
    """
    from src.data_ingestion.lincs import (
        extract_breast_signatures_from_gctx,
        parse_dose_um,
    )

    cache_path = cache_dir / "p2_breast_l1000_signatures.parquet"
    if cache_path.exists():
        logger.info("Loading cached Phase 2 breast signatures...")
        return pd.read_parquet(cache_path)

    # Load metadata
    p2_sig = load_p2_siginfo(cache_dir)
    p2_breast = p2_sig[
        (p2_sig["pert_type"] == "trt_cp")
        & (p2_sig["cell_id"].isin(BREAST_CELL_IDS_LINCS))
        & (p2_sig["pert_itime"].astype(str).str.contains("24"))
    ]
    logger.info(f"Phase 2 breast 24h compound sigs to extract: {len(p2_breast)}")

    if len(p2_breast) == 0:
        return pd.DataFrame()

    # Gene info (use Phase 2's own gene info)
    p2_gene = load_p2_geneinfo(cache_dir)
    if "pr_is_lm" in p2_gene.columns:
        landmark_ids = p2_gene.loc[
            p2_gene["pr_is_lm"] == 1, "pr_gene_id"
        ].tolist()
    else:
        # Phase 2 may use different column names
        landmark_ids = p2_gene["pr_gene_id"].tolist()
    logger.info(f"Landmark genes: {len(landmark_ids)}")

    # GCTX path
    gctx_path = cache_dir / "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx.gz"
    if not gctx_path.exists():
        logger.error(
            f"Phase 2 GCTX not found at {gctx_path}. "
            "Run download_p2_gctx() first."
        )
        return pd.DataFrame()

    # Extract
    expr_df = extract_breast_signatures_from_gctx(
        gctx_path,
        p2_breast["sig_id"].tolist(),
        landmark_ids,
    )

    # Transpose: rows = signatures, columns = genes
    expr_df = expr_df.T
    id_to_symbol = dict(
        zip(
            p2_gene["pr_gene_id"].astype(str),
            p2_gene["pr_gene_symbol"],
        )
    )
    expr_df.columns = [id_to_symbol.get(str(c), c) for c in expr_df.columns]

    # Merge with metadata
    meta_cols = ["sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose"]
    meta = p2_breast[meta_cols].copy()
    meta["dose_um"] = meta["pert_idose"].apply(parse_dose_um)

    result = meta.set_index("sig_id").join(expr_df, how="inner")
    result.reset_index(inplace=True)

    logger.info(f"Phase 2 breast signature matrix: {result.shape}")
    result.to_parquet(cache_path, index=False)
    return result


def merge_phase1_phase2_signatures(
    cache_dir: Path = DATA_CACHE,
) -> pd.DataFrame:
    """Merge Phase 1 and Phase 2 breast signature matrices.

    Returns a combined DataFrame with a 'phase' column indicating source.
    Handles potentially different gene column sets by taking the intersection.
    """
    merged_path = cache_dir / "breast_l1000_signatures_merged.parquet"
    if merged_path.exists():
        logger.info("Loading cached merged signatures...")
        return pd.read_parquet(merged_path)

    p1_path = cache_dir / "breast_l1000_signatures.parquet"
    p2_path = cache_dir / "p2_breast_l1000_signatures.parquet"

    if not p1_path.exists():
        logger.error("Phase 1 breast signatures not found.")
        return pd.DataFrame()

    p1 = pd.read_parquet(p1_path)
    p1["phase"] = 1

    if not p2_path.exists():
        logger.warning("Phase 2 breast signatures not found; returning Phase 1 only.")
        return p1

    p2 = pd.read_parquet(p2_path)
    p2["phase"] = 2

    # Align columns (take intersection of gene columns)
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "dose_um",
                 "pert_idose", "phase"}
    p1_genes = set(p1.columns) - meta_cols
    p2_genes = set(p2.columns) - meta_cols
    shared_genes = sorted(p1_genes & p2_genes)
    logger.info(
        f"Gene columns: P1={len(p1_genes)}, P2={len(p2_genes)}, "
        f"shared={len(shared_genes)}"
    )

    keep_cols = [c for c in p1.columns if c in meta_cols or c in shared_genes]
    p1_aligned = p1[[c for c in keep_cols if c in p1.columns]]
    p2_aligned = p2[[c for c in keep_cols if c in p2.columns]]

    merged = pd.concat([p1_aligned, p2_aligned], ignore_index=True)
    merged.to_parquet(merged_path, index=False)

    logger.info(
        f"Merged signatures: {merged.shape} "
        f"(P1: {len(p1)}, P2: {len(p2)})"
    )
    return merged


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Starting LINCS Phase 2 analysis...")

    # Step 1-2: Download metadata and run comparison
    summary = analyse_phase2()

    print("\n")
    print("=" * 65)
    print("LINCS Phase 2 Analysis Summary")
    print("=" * 65)
    for _, row in summary.iterrows():
        print(f"  {row['metric']:45s} {row['value']:>8}  {row['note']}")

    # Step 3: Decide on GCTX download
    print("\n")
    if should_download_gctx(summary):
        print(
            "RECOMMENDATION: Phase 2 adds significant new drug matches.\n"
            "  Run download_p2_gctx() to download the ~4 GB GCTX, then\n"
            "  extract_p2_breast_signatures() and merge_phase1_phase2_signatures().\n"
            "  This would increase training data from ~719 to ~946 samples\n"
            "  and from ~103 to ~134 drugs."
        )
    else:
        print(
            "Phase 2 does not add enough new drug matches to justify the\n"
            "4 GB GCTX download. Sticking with Phase 1 only."
        )
