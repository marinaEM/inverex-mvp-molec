"""
LINCS L1000 data ingestion for breast-cancer cell lines.

Strategy:
    scTherapy used LINCS 2020 (Level 5 MODZ signatures) matched to PharmacoDB.
    We replicate this but filter to breast-cancer cell lines only.

    Level 5 = replicate-collapsed z-scores (MODZ) per compound-dose-cell-time.
    The full Level 5 dataset is ~40 GB (GSE92742 + GSE70138).

    For the MVP we use cmapPy to read the .gctx files, or alternatively
    download pre-processed subsets from clue.io.

    Fallback: We provide a script to build a compact breast-only reference
    from the publicly available signature metadata + UCSC drug-gene matrices.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import (
    BREAST_CELL_IDS_LINCS,
    DATA_CACHE,
    DATA_RAW,
    N_LANDMARK_GENES,
)
from src.data_ingestion.utils import download_file as _download_file

logger = logging.getLogger(__name__)

# ── URLs ──────────────────────────────────────────────────────────────
# Gene info file (defines the 978 landmark genes)
GENE_INFO_URL = (
    "https://raw.githubusercontent.com/kris-nader/scTherapy/main/geneinfo_beta_input.txt"
)

# Signature metadata from GSE92742 (Level 5)
# These are small TSV files that describe each signature.
SIGINFO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz"
GENEINFO_LINCS_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz"
PERTINFO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz"
INSTINFO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_inst_info.txt.gz"

# Level 5 GCTX (large — ~7 GB compressed)
LEVEL5_GCTX_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"


def load_landmark_genes(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """Load the ~978 L1000 landmark gene list (used by scTherapy)."""
    cache_path = cache_dir / "geneinfo_beta_input.txt"
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")

    try:
        logger.info("Downloading L1000 landmark gene list from scTherapy repo...")
        resp = requests.get(GENE_INFO_URL, timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return pd.read_csv(cache_path, sep="\t")
    except Exception as e:
        logger.warning(f"Could not download landmark gene list: {e}")
        logger.info("Using built-in landmark gene set (top breast-cancer-relevant genes)")
        return _builtin_landmark_genes(cache_dir)


def _builtin_landmark_genes(cache_dir: Path) -> pd.DataFrame:
    """
    Fallback: a curated set of gene symbols when the download fails.
    These are well-known genes from the L1000 landmark set that are
    important in breast cancer biology and drug response.
    In production, the full 978-gene list should be downloaded.
    """
    # Top ~200 breast-cancer-relevant landmark genes (subset of L1000 978)
    genes = [
        "ESR1", "PGR", "ERBB2", "EGFR", "MKI67", "TP53", "PIK3CA", "AKT1",
        "MTOR", "PTEN", "CDH1", "GATA3", "MAP3K1", "BRCA1", "BRCA2", "RB1",
        "CCND1", "CCNE1", "CDK4", "CDK6", "MYC", "FGFR1", "MDM2", "KRAS",
        "BRAF", "MAP2K1", "MAPK1", "MAPK3", "JAK2", "STAT3", "BCL2", "MCL1",
        "BAX", "CASP3", "CASP8", "VEGFA", "KDR", "PDGFRA", "KIT", "MET",
        "ALK", "ROS1", "RET", "NTRK1", "ABL1", "SRC", "FYN", "YES1",
        "AURKA", "AURKB", "PLK1", "TTK", "BUB1", "CDC20", "BIRC5", "TOP2A",
        "TYMS", "DHFR", "RRM1", "RRM2", "TUBB", "TUBA1A", "ABCB1", "ABCG2",
        "HDAC1", "HDAC2", "HDAC3", "HDAC6", "DNMT1", "DNMT3A", "EZH2",
        "BRD4", "CREBBP", "EP300", "SIRT1", "KAT2A", "KDM1A", "KDM5A",
        "NOTCH1", "NOTCH2", "HES1", "JAG1", "DLL1", "WNT1", "CTNNB1",
        "APC", "GSK3B", "AXIN1", "FZD1", "LRP5", "SHH", "SMO", "GLI1",
        "TGFB1", "SMAD2", "SMAD3", "SMAD4", "TGFBR1", "TGFBR2", "BMP2",
        "IL6", "IL1B", "TNF", "NFKB1", "RELA", "IKBKB", "TLR4", "MYD88",
        "CXCL8", "CCL2", "CXCR4", "CCR5", "CD274", "PDCD1", "CTLA4",
        "CD8A", "CD4", "FOXP3", "IFNG", "PRF1", "GZMA", "GZMB",
        "HIF1A", "EPAS1", "VHL", "LDHA", "PKM", "HK2", "G6PD", "FASN",
        "ACACA", "SCD", "HMGCR", "SQLE", "SREBF1", "PPARG", "PPARA",
        "ATM", "ATR", "CHEK1", "CHEK2", "WEE1", "PARP1", "PARP2",
        "RAD51", "XRCC1", "MLH1", "MSH2", "PMS2", "ERCC1", "XPC",
        "CDKN1A", "CDKN2A", "CDKN1B", "CDK2", "CDK1", "CCNA2", "CCNB1",
        "E2F1", "CDC25A", "CDC25C", "PLK4", "NEK2", "CENPE", "KIF11",
        "HSP90AA1", "HSP90AB1", "HSPA1A", "HSPA5", "HSPA8", "HSPB1",
        "PSMA1", "PSMB5", "UBE2C", "UBB", "USP7", "MDM4", "XIAP",
        "BCL2L1", "BCL2L11", "BID", "BAK1", "BBC3", "PMAIP1",
        "FOS", "JUN", "EGR1", "MYB", "SOX2", "POU5F1", "NANOG", "KLF4",
        "GAPDH", "ACTB", "B2M", "RPL13A", "HPRT1", "TBP", "GUSB",
        "FOXA1", "AR", "NR3C1", "VDR", "RXRA", "RARA", "NR1I2",
        "CYP3A4", "CYP1A2", "CYP2D6", "UGT1A1", "SLCO1B1", "SLC22A1",
    ]
    df = pd.DataFrame({"gene_symbol": genes})
    cache_path = cache_dir / "geneinfo_beta_input.txt"
    df.to_csv(cache_path, sep="\t", index=False)
    return df


def load_lincs_siginfo(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """
    Download and cache LINCS signature metadata.
    Returns a DataFrame with columns including:
        sig_id, pert_id, pert_iname, pert_type, cell_id,
        pert_idose, pert_itime, distil_id, ...
    """
    cache_path = cache_dir / "GSE92742_sig_info.parquet"
    if cache_path.exists():
        logger.info("Loading cached LINCS siginfo...")
        return pd.read_parquet(cache_path)

    gz_path = cache_dir / "GSE92742_sig_info.txt.gz"
    if not gz_path.exists():
        logger.info("Downloading LINCS siginfo (~15 MB)...")
        _download_file(SIGINFO_URL, gz_path)

    logger.info("Parsing LINCS siginfo...")
    df = pd.read_csv(gz_path, sep="\t", low_memory=False)
    df.to_parquet(cache_path, index=False)
    return df


def load_lincs_pertinfo(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """Download and cache LINCS perturbagen info (compound names, targets, etc.)."""
    cache_path = cache_dir / "GSE92742_pert_info.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    gz_path = cache_dir / "GSE92742_pert_info.txt.gz"
    if not gz_path.exists():
        logger.info("Downloading LINCS pertinfo...")
        _download_file(PERTINFO_URL, gz_path)

    df = pd.read_csv(gz_path, sep="\t", low_memory=False)
    df.to_parquet(cache_path, index=False)
    return df


def load_lincs_geneinfo(cache_dir: Path = DATA_CACHE) -> pd.DataFrame:
    """Download LINCS gene info (maps pr_gene_id → gene_symbol, landmark status)."""
    cache_path = cache_dir / "GSE92742_gene_info.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    gz_path = cache_dir / "GSE92742_gene_info.txt.gz"
    if not gz_path.exists():
        logger.info("Downloading LINCS gene info...")
        _download_file(GENEINFO_LINCS_URL, gz_path)

    df = pd.read_csv(gz_path, sep="\t", low_memory=False)
    df.to_parquet(cache_path, index=False)
    return df


def filter_breast_signatures(siginfo: pd.DataFrame) -> pd.DataFrame:
    """
    Filter LINCS signatures to:
      - breast cancer cell lines
      - compound perturbagens (pert_type == 'trt_cp')
      - 24h time point (most common, matches scTherapy)
      - quality-passing signatures
    """
    mask = (
        siginfo["cell_id"].isin(BREAST_CELL_IDS_LINCS)
        & (siginfo["pert_type"] == "trt_cp")
        & (siginfo["pert_itime"].astype(str).str.contains("24"))
    )
    # If there's a qc_pass column, filter on it
    if "qc_pass" in siginfo.columns:
        mask = mask & (siginfo["qc_pass"] == 1)

    filtered = siginfo[mask].copy()
    logger.info(
        f"Filtered to {len(filtered)} breast-cancer signatures "
        f"({filtered['cell_id'].nunique()} cell lines, "
        f"{filtered['pert_iname'].nunique()} compounds)"
    )
    return filtered


def parse_dose_um(dose_str) -> Optional[float]:
    """
    Parse LINCS dose strings like '10 µM', '0.1 um', '3.33 uM' → float in µM.
    Returns None if unparseable.
    """
    if pd.isna(dose_str):
        return None
    s = str(dose_str).strip().lower()
    # Remove units
    for unit in ["µm", "um", "μm"]:
        s = s.replace(unit, "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_breast_signatures_from_gctx(
    gctx_path: Path,
    breast_sig_ids: list[str],
    landmark_gene_ids: list[int],
) -> pd.DataFrame:
    """
    Extract a subset of Level 5 signatures from the GCTX file.

    Uses cmapPy for efficient sliced reading.
    Returns DataFrame: rows = landmark genes, columns = sig_ids.
    """
    try:
        from cmapPy.pandasGEXpress import parse as gctx_parse

        logger.info(
            f"Extracting {len(breast_sig_ids)} signatures × "
            f"{len(landmark_gene_ids)} genes from GCTX..."
        )
        gctoo = gctx_parse.parse(
            str(gctx_path),
            cid=breast_sig_ids,
            rid=[str(g) for g in landmark_gene_ids],
        )
        return gctoo.data_df

    except ImportError:
        logger.error(
            "cmapPy not installed. Install with: pip install cmapPy"
        )
        raise


def build_breast_signature_matrix(
    cache_dir: Path = DATA_CACHE,
    gctx_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    High-level function: build or load a cached breast-cancer-only
    signature matrix (genes × signatures) with metadata.

    If the full GCTX is available, extracts the breast subset.
    Otherwise, loads from a pre-built cache.

    Returns:
        DataFrame with columns:
            sig_id, pert_id, pert_iname, cell_id, dose_um,
            + one column per landmark gene (z-score)
    """
    cache_path = cache_dir / "breast_l1000_signatures.parquet"
    if cache_path.exists():
        logger.info("Loading cached breast L1000 signature matrix...")
        return pd.read_parquet(cache_path)

    # Load metadata
    siginfo = load_lincs_siginfo(cache_dir)
    breast_sigs = filter_breast_signatures(siginfo)

    if gctx_path is not None and gctx_path.exists():
        # Extract from full GCTX
        geneinfo = load_lincs_geneinfo(cache_dir)
        landmark_ids = geneinfo.loc[
            geneinfo["pr_is_lm"] == 1, "pr_gene_id"
        ].tolist()

        expr_df = extract_breast_signatures_from_gctx(
            gctx_path, breast_sigs["sig_id"].tolist(), landmark_ids
        )

        # Transpose: rows = signatures, columns = genes
        expr_df = expr_df.T
        # Map gene IDs to symbols
        id_to_symbol = dict(
            zip(geneinfo["pr_gene_id"].astype(str), geneinfo["pr_gene_symbol"])
        )
        expr_df.columns = [id_to_symbol.get(str(c), c) for c in expr_df.columns]

        # Merge with metadata
        meta_cols = ["sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose"]
        meta = breast_sigs[meta_cols].copy()
        meta["dose_um"] = meta["pert_idose"].apply(parse_dose_um)

        result = meta.set_index("sig_id").join(expr_df, how="inner")
        result.reset_index(inplace=True)

        logger.info(f"Built breast signature matrix: {result.shape}")
        result.to_parquet(cache_path, index=False)
        return result

    else:
        logger.warning(
            "Full GCTX file not found. To build the signature matrix, either:\n"
            "  1. Download GSE92742_Broad_LINCS_Level5_COMPZ.MODZ*.gctx.gz\n"
            "     and pass its path as gctx_path, or\n"
            "  2. Use the synthetic/demo signature builder (see build_demo_signatures)."
        )
        return pd.DataFrame()


def build_demo_signatures(
    n_compounds: int = 200,
    cache_dir: Path = DATA_CACHE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a DEMO signature matrix for development/testing.

    Uses the real metadata (siginfo + pertinfo) to get realistic compound
    names, cell lines, and doses, but generates synthetic z-scores.

    This allows the full pipeline to run end-to-end without the 7 GB GCTX.
    The synthetic scores are drawn from N(0, 1) — the model trained on these
    won't be meaningful, but the pipeline mechanics are testable.

    For a real run, replace with build_breast_signature_matrix().
    """
    cache_path = cache_dir / "breast_l1000_demo_signatures.parquet"
    if cache_path.exists():
        logger.info("Loading cached demo signatures...")
        return pd.read_parquet(cache_path)

    rng = np.random.default_rng(seed)

    # Try to load real metadata; fall back to fully synthetic if unavailable
    try:
        siginfo = load_lincs_siginfo(cache_dir)
        breast_sigs = filter_breast_signatures(siginfo)
    except Exception as e:
        logger.warning(f"Could not load LINCS siginfo ({e}). Using fully synthetic demo.")
        return _create_fully_synthetic_demo(n_compounds, cache_dir, seed)

    if len(breast_sigs) == 0:
        logger.warning("No siginfo available. Creating fully synthetic demo data.")
        return _create_fully_synthetic_demo(n_compounds, cache_dir, seed)

    # Sample a subset of compounds
    compounds = breast_sigs["pert_iname"].unique()
    if len(compounds) > n_compounds:
        compounds = rng.choice(compounds, n_compounds, replace=False)

    subset = breast_sigs[breast_sigs["pert_iname"].isin(compounds)].copy()
    subset["dose_um"] = subset["pert_idose"].apply(parse_dose_um)

    # Load landmark genes
    gene_df = load_landmark_genes(cache_dir)
    gene_symbols = gene_df["gene_symbol"].tolist()

    # Generate synthetic z-scores
    n_sigs = len(subset)
    n_genes = len(gene_symbols)
    z_scores = rng.standard_normal((n_sigs, n_genes)).astype(np.float32)

    result = subset[["sig_id", "pert_id", "pert_iname", "cell_id", "dose_um"]].reset_index(drop=True)
    gene_df_scores = pd.DataFrame(z_scores, columns=gene_symbols)
    result = pd.concat([result, gene_df_scores], axis=1)

    logger.info(f"Built demo signature matrix: {result.shape}")
    result.to_parquet(cache_path, index=False)
    return result


def _create_fully_synthetic_demo(
    n_compounds: int, cache_dir: Path, seed: int
) -> pd.DataFrame:
    """Fallback: fully synthetic data when no LINCS metadata is available."""
    rng = np.random.default_rng(seed)

    # Use a list of known breast-cancer-relevant drugs
    drug_names = [
        "tamoxifen", "fulvestrant", "letrozole", "anastrozole",
        "trastuzumab-dm1", "lapatinib", "neratinib", "tucatinib",
        "palbociclib", "ribociclib", "abemaciclib", "alpelisib",
        "everolimus", "olaparib", "talazoparib", "capecitabine",
        "doxorubicin", "paclitaxel", "docetaxel", "cisplatin",
        "carboplatin", "gemcitabine", "eribulin", "vinorelbine",
        "pembrolizumab", "atezolizumab", "sacituzumab-govitecan",
        "ado-trastuzumab-emtansine", "pertuzumab", "dasatinib",
        "sorafenib", "vorinostat", "panobinostat", "bortezomib",
        "navitoclax", "venetoclax", "trametinib", "selumetinib",
        "idelalisib", "copanlisib", "ipatasertib", "capivasertib",
        "gedatolisib", "pictilisib", "buparlisib", "taselisib",
        "dinaciclib", "flavopiridol", "entinostat", "tucidinostat",
    ]
    drug_names = drug_names[:n_compounds]

    cell_lines = BREAST_CELL_IDS_LINCS
    doses = [0.04, 0.12, 0.37, 1.11, 3.33, 10.0]

    rows = []
    for drug in drug_names:
        for cell in cell_lines:
            for dose in doses:
                rows.append({
                    "sig_id": f"SYN_{drug}_{cell}_{dose}",
                    "pert_id": f"BRD-SYN-{drug[:8]}",
                    "pert_iname": drug,
                    "cell_id": cell,
                    "dose_um": dose,
                })

    meta = pd.DataFrame(rows)

    # Load landmark genes for column names
    try:
        gene_df = load_landmark_genes(cache_dir)
        gene_symbols = gene_df["gene_symbol"].tolist()
    except Exception:
        gene_symbols = [f"GENE_{i}" for i in range(N_LANDMARK_GENES)]

    z_scores = rng.standard_normal((len(meta), len(gene_symbols))).astype(np.float32)
    gene_data = pd.DataFrame(z_scores, columns=gene_symbols)
    result = pd.concat([meta, gene_data], axis=1)

    cache_path = cache_dir / "breast_l1000_demo_signatures.parquet"
    result.to_parquet(cache_path, index=False)
    logger.info(f"Built fully synthetic demo signatures: {result.shape}")
    return result
