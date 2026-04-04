"""
Chemical language model embeddings for the INVEREX pipeline.

Extracts ChemBERTa embeddings from SMILES strings to complement ECFP4
fingerprints as drug features. ChemBERTa captures contextual chemical
semantics that bit-vector fingerprints may miss.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Prevent OMP duplicate-library crash on macOS (conda torch + system libomp)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import DATA_CACHE, DATA_PROCESSED, LIGHTGBM_DEFAULT_PARAMS, RANDOM_SEED

logger = logging.getLogger(__name__)

# ── Default model ────────────────────────────────────────────────────────
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"


# ── Core embedding extraction ────────────────────────────────────────────
def get_chemberta_embeddings(
    smiles_list: list[str],
    model_name: str = CHEMBERTA_MODEL,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Extract mean-pooled ChemBERTa embeddings for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings (one per compound).
    model_name : str
        Hugging Face model identifier.
    batch_size : int
        Number of SMILES to tokenise at once.

    Returns
    -------
    np.ndarray
        Shape (n_compounds, hidden_dim).  Hidden dim is 384 for the 77M model.
    """
    logger.info(
        f"Loading ChemBERTa model: {model_name} for {len(smiles_list)} SMILES"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = model(**inputs)
            # Mean-pool over token dimension → (batch, hidden_dim)
            emb = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(emb)
            if (start // batch_size) % 5 == 0:
                logger.info(
                    f"  Embedded {min(start + batch_size, len(smiles_list))}"
                    f"/{len(smiles_list)} SMILES"
                )

    return np.vstack(embeddings)


# ── Build & cache embedding parquet ──────────────────────────────────────
def build_chemberta_embeddings(
    cache_dir: Path = DATA_CACHE,
    model_name: str = CHEMBERTA_MODEL,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame of ChemBERTa embeddings for every compound in the
    SMILES cache, and persist to ``cache_dir/chemberta_embeddings.parquet``.

    Returns
    -------
    pd.DataFrame
        Columns: compound_name, chemberta_0 … chemberta_{d-1}
    """
    out_path = cache_dir / "chemberta_embeddings.parquet"

    if out_path.exists() and not force:
        logger.info(f"Loading cached ChemBERTa embeddings from {out_path}")
        return pd.read_parquet(out_path)

    smiles_path = cache_dir / "compound_smiles_cache.parquet"
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"SMILES cache not found at {smiles_path}. "
            "Run the fingerprint pipeline first."
        )

    smiles_df = pd.read_parquet(smiles_path)
    logger.info(
        f"Extracting ChemBERTa embeddings for {len(smiles_df)} compounds …"
    )

    emb_array = get_chemberta_embeddings(
        smiles_df["smiles"].tolist(), model_name=model_name
    )
    hidden_dim = emb_array.shape[1]
    emb_cols = [f"chemberta_{i}" for i in range(hidden_dim)]

    emb_df = pd.DataFrame(emb_array, columns=emb_cols)
    emb_df.insert(0, "compound_name", smiles_df["compound_name"].values)

    emb_df.to_parquet(out_path, index=False)
    logger.info(
        f"Saved ChemBERTa embeddings ({emb_array.shape}) → {out_path}"
    )
    return emb_df


# ── Cross-validated evaluation ───────────────────────────────────────────
def evaluate_feature_sets(
    cache_dir: Path = DATA_CACHE,
    processed_dir: Path = DATA_PROCESSED,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Compare ECFP-only, ChemBERTa-only, and ECFP+ChemBERTa features via
    5-fold CV RMSE on the training matrix (LightGBM with default params).

    Returns
    -------
    pd.DataFrame
        One row per configuration with columns: feature_set, n_features,
        cv_rmse_mean, cv_rmse_std.
    """
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import cross_val_score

    # ── Load base data ────────────────────────────────────────────────
    X_full = pd.read_parquet(processed_dir / "training_matrix.parquet")
    y = pd.read_parquet(processed_dir / "training_target.parquet").squeeze()

    # Identify column groups
    ecfp_cols = [c for c in X_full.columns if c.startswith("ecfp_")]
    gene_cols = [
        c
        for c in X_full.columns
        if not c.startswith("ecfp_") and c != "log_dose_um"
    ]
    dose_col = ["log_dose_um"]

    logger.info(
        f"Base matrix: {X_full.shape[0]} samples, "
        f"{len(gene_cols)} gene cols, {len(ecfp_cols)} ecfp cols, 1 dose col"
    )

    # ── Load ChemBERTa embeddings ─────────────────────────────────────
    emb_df = build_chemberta_embeddings(cache_dir=cache_dir)
    chemberta_cols = [c for c in emb_df.columns if c.startswith("chemberta_")]
    logger.info(
        f"ChemBERTa embeddings: {len(emb_df)} compounds × "
        f"{len(chemberta_cols)} dims"
    )

    # ── Map embeddings to training rows ──────────────────────────────
    # We need the compound name for each training row.
    # The matched parquet preserves the same row order as training_matrix.
    matched_path = cache_dir / "lincs_pharmacodb_matched.parquet"
    matched = pd.read_parquet(matched_path)
    compound_names = matched["pert_iname"].values

    emb_lookup = emb_df.set_index("compound_name")[chemberta_cols]
    chemberta_matrix = np.zeros(
        (len(X_full), len(chemberta_cols)), dtype=np.float32
    )
    for i, name in enumerate(compound_names):
        if name in emb_lookup.index:
            chemberta_matrix[i] = emb_lookup.loc[name].values

    chemberta_df = pd.DataFrame(
        chemberta_matrix, columns=chemberta_cols, index=X_full.index
    )

    # ── Define feature configurations ────────────────────────────────
    configs = {
        "ECFP only": gene_cols + ecfp_cols + dose_col,
        "ChemBERTa only": gene_cols + dose_col,  # placeholder — replaced below
        "ECFP + ChemBERTa": gene_cols + ecfp_cols + dose_col,  # placeholder
    }

    # Build actual X matrices
    X_ecfp = X_full[gene_cols + ecfp_cols + dose_col]

    X_chemberta = pd.concat(
        [X_full[gene_cols], chemberta_df, X_full[dose_col]], axis=1
    )

    X_combined = pd.concat(
        [X_full[gene_cols + ecfp_cols], chemberta_df, X_full[dose_col]],
        axis=1,
    )

    feature_sets = {
        "ECFP only": X_ecfp,
        "ChemBERTa only": X_chemberta,
        "ECFP + ChemBERTa": X_combined,
    }

    # ── Run CV ────────────────────────────────────────────────────────
    results = []
    for name, X_feat in feature_sets.items():
        logger.info(
            f"Evaluating '{name}': {X_feat.shape[1]} features …"
        )

        lgbm = LGBMRegressor(**LIGHTGBM_DEFAULT_PARAMS)

        scores = cross_val_score(
            lgbm,
            X_feat,
            y,
            cv=n_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        rmse_values = -scores  # flip sign
        mean_rmse = rmse_values.mean()
        std_rmse = rmse_values.std()

        logger.info(
            f"  {name}: CV RMSE = {mean_rmse:.4f} +/- {std_rmse:.4f}"
        )
        results.append(
            {
                "feature_set": name,
                "n_features": X_feat.shape[1],
                "cv_rmse_mean": round(mean_rmse, 4),
                "cv_rmse_std": round(std_rmse, 4),
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


# ── Main entry-point ─────────────────────────────────────────────────────
def main() -> None:
    """Run embedding extraction and comparative evaluation end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("INVEREX — ChemBERTa chemical embedding evaluation")
    logger.info("=" * 60)

    # Step 1: build/cache embeddings
    emb_df = build_chemberta_embeddings()
    logger.info(
        f"Embedding matrix: {emb_df.shape[0]} compounds × "
        f"{emb_df.shape[1] - 1} dims"
    )

    # Step 2: comparative evaluation
    results_df = evaluate_feature_sets()

    # Step 3: save comparison results
    from src.config import RESULTS

    out_csv = RESULTS / "chemical_embeddings_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Results saved → {out_csv}")

    print("\n" + "=" * 60)
    print("Chemical Embedding Comparison — 5-fold CV RMSE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
