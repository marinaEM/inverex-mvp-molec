"""
Build the training matrix for the LightGBM drug-response model.

This replicates scTherapy's feature schema:
    Features = [gene fold-changes (978)] + [ECFP4 fingerprint (1024)] + [log_dose (1)]
    Target   = percent inhibition (continuous, from PharmacoDB)

Each row represents one (drug × dose × cell-line) triad from LINCS, matched
to a PharmacoDB viability measurement.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_CACHE, DATA_PROCESSED, ECFP_NBITS
from src.data_ingestion.lincs import (
    build_breast_signature_matrix,
    build_demo_signatures,
    load_landmark_genes,
)
from src.data_ingestion.pharmacodb import (
    build_dose_response_reference,
    match_lincs_to_pharmacodb,
)
from src.data_ingestion.pubchem import (
    build_demo_fingerprints,
    build_fingerprint_matrix,
)

logger = logging.getLogger(__name__)


def build_training_matrix(
    use_demo: bool = False,
    cache_dir: Path = DATA_CACHE,
    output_dir: Path = DATA_PROCESSED,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build the full training matrix for LightGBM.

    Returns:
        X: Feature DataFrame (gene z-scores + ECFP4 bits + log_dose)
        y: Target Series (percent inhibition)
        feature_names: List of feature column names
    """
    output_path = output_dir / "training_matrix.parquet"
    target_path = output_dir / "training_target.parquet"

    if output_path.exists() and target_path.exists():
        logger.info("Loading cached training matrix...")
        X = pd.read_parquet(output_path)
        y = pd.read_parquet(target_path).squeeze()
        return X, y, list(X.columns)

    # ── Step 1: Load/build LINCS signatures ────────────────────────
    logger.info("Step 1: Loading LINCS L1000 signatures...")
    if use_demo:
        logger.info("  Demo mode → building fully synthetic training matrix")
        return _build_demo_training_matrix(cache_dir, output_dir)
    
    lincs_sigs = build_breast_signature_matrix(cache_dir=cache_dir)
    if len(lincs_sigs) == 0:
        logger.warning("No real LINCS data available. Falling back to demo.")
        lincs_sigs = build_demo_signatures(n_compounds=100, cache_dir=cache_dir)

    # ── Step 2: Load/build dose-response reference ─────────────────
    logger.info("Step 2: Loading dose-response reference...")
    dose_ref = build_dose_response_reference(cache_dir=cache_dir)

    # ── Step 3: Match LINCS ↔ PharmacoDB ───────────────────────────
    logger.info("Step 3: Matching LINCS to PharmacoDB...")
    matched = match_lincs_to_pharmacodb(lincs_sigs, dose_ref, cache_dir)

    if len(matched) == 0:
        logger.warning("No LINCS-PharmacoDB matches. Using demo data pipeline.")
        return _build_demo_training_matrix(cache_dir, output_dir)

    # ── Step 4: Add drug fingerprints ──────────────────────────────
    logger.info("Step 4: Computing drug fingerprints...")
    compounds = matched["pert_iname"].unique().tolist()

    try:
        fp_df = build_fingerprint_matrix(compounds, cache_dir)
    except (ImportError, Exception) as e:
        logger.warning(f"Fingerprint computation failed ({e}). Using demo fingerprints.")
        fp_df = build_demo_fingerprints(compounds, cache_dir)

    # ── Step 5: Assemble feature matrix ────────────────────────────
    logger.info("Step 5: Assembling training matrix...")

    # Identify gene columns (everything that's not metadata)
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "dose_um",
                 "ic50_um", "pct_inhibition", "pert_idose"}
    gene_cols = [c for c in matched.columns if c not in meta_cols]

    # Gene features
    X_genes = matched[gene_cols].astype(np.float32)

    # Fingerprint features — merge on compound name
    ecfp_cols = [c for c in fp_df.columns if c.startswith("ecfp_")]
    fp_lookup = fp_df.set_index("compound_name")[ecfp_cols]

    X_ecfp = matched["pert_iname"].map(
        lambda name: fp_lookup.loc[name].values if name in fp_lookup.index else None
    )
    # Expand to columns
    valid_mask = X_ecfp.notna()
    ecfp_array = np.zeros((len(matched), len(ecfp_cols)), dtype=np.int8)
    for i, val in enumerate(X_ecfp):
        if val is not None:
            ecfp_array[i] = val
    X_ecfp_df = pd.DataFrame(ecfp_array, columns=ecfp_cols, index=matched.index)

    # Dose feature (log-transformed)
    X_dose = np.log1p(matched["dose_um"].fillna(1.0).astype(np.float32))
    X_dose_df = pd.DataFrame({"log_dose_um": X_dose}, index=matched.index)

    # Combine all features
    X = pd.concat([X_genes, X_ecfp_df, X_dose_df], axis=1)
    y = matched["pct_inhibition"].astype(np.float32)

    # Drop rows with missing target
    valid = y.notna() & np.isfinite(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    feature_names = list(X.columns)

    logger.info(
        f"Training matrix: {X.shape[0]} samples × {X.shape[1]} features "
        f"(genes: {len(gene_cols)}, ECFP: {len(ecfp_cols)}, dose: 1)"
    )
    logger.info(
        f"Target stats: mean={y.mean():.1f}%, std={y.std():.1f}%, "
        f"range=[{y.min():.1f}, {y.max():.1f}]"
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(output_path, index=False)
    pd.DataFrame({"pct_inhibition": y}).to_parquet(target_path, index=False)

    return X, y, feature_names


def _build_demo_training_matrix(
    cache_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Fully synthetic training matrix for demo/testing.
    Allows the complete pipeline to run without real data downloads.
    """
    rng = np.random.default_rng(42)
    n_samples = 5000

    # Gene features
    gene_df = load_landmark_genes(cache_dir)
    gene_symbols = gene_df["gene_symbol"].tolist() if len(gene_df) > 0 else [
        f"GENE_{i}" for i in range(978)
    ]
    X_genes = pd.DataFrame(
        rng.standard_normal((n_samples, len(gene_symbols))).astype(np.float32),
        columns=gene_symbols,
    )

    # ECFP features
    ecfp_cols = [f"ecfp_{i}" for i in range(ECFP_NBITS)]
    X_ecfp = pd.DataFrame(
        (rng.random((n_samples, ECFP_NBITS)) < 0.1).astype(np.int8),
        columns=ecfp_cols,
    )

    # Dose
    doses = rng.choice([0.04, 0.12, 0.37, 1.11, 3.33, 10.0], n_samples)
    X_dose = pd.DataFrame({"log_dose_um": np.log1p(doses).astype(np.float32)})

    X = pd.concat([X_genes, X_ecfp, X_dose], axis=1)

    # Synthetic target: loosely correlated with dose and a few gene features
    y = (
        30 + 20 * np.log1p(doses) / np.log1p(10)
        + 5 * X_genes.iloc[:, 0].values
        - 3 * X_genes.iloc[:, 1].values
        + rng.normal(0, 10, n_samples)
    ).clip(0, 100).astype(np.float32)
    y = pd.Series(y, name="pct_inhibition")

    output_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(output_dir / "training_matrix.parquet", index=False)
    pd.DataFrame({"pct_inhibition": y}).to_parquet(
        output_dir / "training_target.parquet", index=False
    )

    logger.info(f"Built demo training matrix: {X.shape}")
    return X, y, list(X.columns)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X, y, features = build_training_matrix(use_demo=True)
    print(f"\nTraining matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature groups:")
    gene_feats = [f for f in features if not f.startswith("ecfp_") and f != "log_dose_um"]
    ecfp_feats = [f for f in features if f.startswith("ecfp_")]
    print(f"  Gene z-scores: {len(gene_feats)}")
    print(f"  ECFP4 bits:    {len(ecfp_feats)}")
    print(f"  Dose:          1")
