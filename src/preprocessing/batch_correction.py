"""
Cross-platform batch correction for CTR-DB patient datasets.

The 38 CTR-DB datasets come from different microarray platforms (Affymetrix
GPL96/570/571, Agilent, Illumina) and RNA-seq.  Platform-specific biases
hurt leave-one-dataset-out (LODO) cross-validation of reversal-score models.

This module implements four batch-correction methods, all operating on the
shared L1000 landmark genes:

    1. **per_dataset_zscore** -- z-score each gene within its source dataset,
       then pool.  Simplest; the existing pipeline already does this.

    2. **quantile_norm** -- force every dataset to share the same gene-value
       distribution via rank-based quantile normalization (Bolstad 2003).

    3. **rank_norm** -- replace expression values with within-sample ranks,
       then z-score across the pooled matrix.

    4. **combat** -- parametric ComBat batch correction (Johnson 2007) via
       the ``neuroCombat`` package.  Estimates and removes additive and
       multiplicative batch effects per gene.

Each method takes a dict of {geo_id: expression_df} and returns a single
pooled DataFrame (patients x genes) plus a Series of dataset labels.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _restrict_to_common_genes(
    datasets: dict[str, pd.DataFrame],
    landmark_genes: list[str],
    min_dataset_frac: float = 0.80,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Restrict all datasets to landmark genes present in >= min_dataset_frac
    of datasets.  Returns restricted dicts and sorted gene list.
    """
    from collections import Counter

    gene_presence = Counter()
    for geo_id, expr in datasets.items():
        available = set(expr.columns) & set(landmark_genes)
        gene_presence.update(available)

    n_ds = len(datasets)
    threshold = max(2, int(min_dataset_frac * n_ds))
    common_genes = sorted(g for g, c in gene_presence.items() if c >= threshold)
    logger.info(
        f"Common landmark genes (present in >= {threshold}/{n_ds} datasets): "
        f"{len(common_genes)}"
    )

    restricted = {}
    for geo_id, expr in datasets.items():
        available = [g for g in common_genes if g in expr.columns]
        df = expr[available].copy()
        # Fill missing common genes with NaN using reindex (avoids fragmentation)
        df = df.reindex(columns=common_genes)
        restricted[geo_id] = df

    return restricted, common_genes


def _pool_datasets(
    datasets: dict[str, pd.DataFrame],
    labels: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Concatenate expression datasets and labels.
    Returns (X_pooled, y_pooled, dataset_ids).
    """
    X_parts = []
    y_parts = []
    ds_parts = []

    for geo_id, expr in sorted(datasets.items()):
        if geo_id not in labels:
            continue
        lab = labels[geo_id]
        common_samples = expr.index.intersection(lab.index)
        if len(common_samples) < 5:
            continue
        X_parts.append(expr.loc[common_samples])
        y_parts.append(lab.loc[common_samples])
        ds_parts.extend([geo_id] * len(common_samples))

    if not X_parts:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=str)

    X = pd.concat(X_parts, axis=0).reset_index(drop=True)
    y = pd.concat(y_parts, axis=0).reset_index(drop=True).astype(int)
    ds = pd.Series(ds_parts, name="dataset_id").reset_index(drop=True)

    X = X.fillna(0.0)
    return X, y, ds


# ---------------------------------------------------------------------------
# Method 1: Per-dataset z-scoring (baseline)
# ---------------------------------------------------------------------------

def per_dataset_zscore(
    datasets: dict[str, pd.DataFrame],
    labels: dict[str, pd.Series],
    landmark_genes: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Z-score each gene within each dataset, then pool.

    This is the simplest approach and matches what the existing pipeline does.
    """
    restricted, common_genes = _restrict_to_common_genes(datasets, landmark_genes)

    corrected = {}
    for geo_id, expr in restricted.items():
        mean = expr.mean(axis=0)
        std = expr.std(axis=0).replace(0, 1)
        corrected[geo_id] = (expr - mean) / std

    X, y, ds = _pool_datasets(corrected, labels)
    X = X.fillna(0.0)
    logger.info(f"[per_dataset_zscore] Pooled: {X.shape[0]} patients x {X.shape[1]} genes")
    return X, y, ds, common_genes


# ---------------------------------------------------------------------------
# Method 2: Quantile normalization
# ---------------------------------------------------------------------------

def quantile_norm(
    datasets: dict[str, pd.DataFrame],
    labels: dict[str, pd.Series],
    landmark_genes: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Quantile normalization: force all samples (across all datasets) to have
    the same gene-value distribution.

    Steps:
    1. Pool all datasets.
    2. For each gene, replace values with their ranks within the gene column.
    3. Replace each rank with the mean value at that rank across all genes.
    4. Z-score the result.
    """
    restricted, common_genes = _restrict_to_common_genes(datasets, landmark_genes)
    X_raw, y, ds = _pool_datasets(restricted, labels)

    if X_raw.empty:
        return X_raw, y, ds, common_genes

    mat = X_raw.values.copy()
    n_samples, n_genes = mat.shape

    # Step 1: rank each column (gene) independently
    ranked = np.zeros_like(mat)
    for j in range(n_genes):
        ranked[:, j] = rankdata(mat[:, j], method="average")

    # Step 2: sort each column to get sorted values
    sorted_vals = np.sort(mat, axis=0)

    # Step 3: compute the mean value at each rank position across all genes
    rank_means = sorted_vals.mean(axis=1)  # shape (n_samples,)

    # Step 4: map ranks to rank means
    # For rank r, the value is rank_means[int(r) - 1] (fractional ranks use interpolation)
    qn_mat = np.zeros_like(mat)
    for j in range(n_genes):
        # Convert ranks to 0-indexed integers (handle fractional ranks)
        ranks_0 = ranked[:, j] - 1  # to 0-indexed
        floor_r = np.floor(ranks_0).astype(int)
        ceil_r = np.ceil(ranks_0).astype(int)
        floor_r = np.clip(floor_r, 0, n_samples - 1)
        ceil_r = np.clip(ceil_r, 0, n_samples - 1)
        frac = ranks_0 - np.floor(ranks_0)
        qn_mat[:, j] = (1 - frac) * rank_means[floor_r] + frac * rank_means[ceil_r]

    # Step 5: z-score the result
    qn_mean = qn_mat.mean(axis=0)
    qn_std = qn_mat.std(axis=0)
    qn_std[qn_std == 0] = 1
    qn_mat = (qn_mat - qn_mean) / qn_std

    X_qn = pd.DataFrame(qn_mat, columns=common_genes)
    logger.info(f"[quantile_norm] Pooled: {X_qn.shape[0]} patients x {X_qn.shape[1]} genes")
    return X_qn, y, ds, common_genes


# ---------------------------------------------------------------------------
# Method 3: Rank-based normalization
# ---------------------------------------------------------------------------

def rank_norm(
    datasets: dict[str, pd.DataFrame],
    labels: dict[str, pd.Series],
    landmark_genes: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Rank-based normalization: within each sample, replace expression values
    with their rank among genes, then z-score across the pooled matrix.

    This makes every sample's marginal distribution identical, removing
    platform-specific distributional shifts.
    """
    restricted, common_genes = _restrict_to_common_genes(datasets, landmark_genes)
    X_raw, y, ds = _pool_datasets(restricted, labels)

    if X_raw.empty:
        return X_raw, y, ds, common_genes

    mat = X_raw.values.copy()
    n_samples, n_genes = mat.shape

    # Within-sample ranking: for each sample, rank genes
    ranked = np.zeros_like(mat)
    for i in range(n_samples):
        ranked[i, :] = rankdata(mat[i, :], method="average")

    # Normalize ranks to [0, 1] range
    ranked = (ranked - 1) / (n_genes - 1) if n_genes > 1 else ranked

    # Z-score across samples (per gene)
    rk_mean = ranked.mean(axis=0)
    rk_std = ranked.std(axis=0)
    rk_std[rk_std == 0] = 1
    ranked_z = (ranked - rk_mean) / rk_std

    X_rk = pd.DataFrame(ranked_z, columns=common_genes)
    logger.info(f"[rank_norm] Pooled: {X_rk.shape[0]} patients x {X_rk.shape[1]} genes")
    return X_rk, y, ds, common_genes


# ---------------------------------------------------------------------------
# Method 4: ComBat
# ---------------------------------------------------------------------------

def combat_correction(
    datasets: dict[str, pd.DataFrame],
    labels: dict[str, pd.Series],
    landmark_genes: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    ComBat batch correction (Johnson et al., 2007) via neuroCombat.

    Estimates dataset-specific additive (location) and multiplicative (scale)
    batch effects per gene, then removes them using empirical Bayes shrinkage.
    """
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        logger.warning("neuroCombat not installed; falling back to per_dataset_zscore")
        return per_dataset_zscore(datasets, labels, landmark_genes)

    restricted, common_genes = _restrict_to_common_genes(datasets, landmark_genes)
    X_raw, y, ds = _pool_datasets(restricted, labels)

    if X_raw.empty:
        return X_raw, y, ds, common_genes

    # neuroCombat expects: data = genes x samples, batch = array of batch labels
    data_mat = X_raw.values.T  # (n_genes, n_samples)

    # Encode batch labels as integers
    batch_labels = ds.values
    unique_batches = np.unique(batch_labels)

    # Filter: ComBat needs at least 2 batches with > 1 sample each
    batch_sizes = pd.Series(batch_labels).value_counts()
    valid_batches = batch_sizes[batch_sizes > 1].index.tolist()
    if len(valid_batches) < 2:
        logger.warning("ComBat: fewer than 2 valid batches; falling back")
        return per_dataset_zscore(datasets, labels, landmark_genes)

    # Keep only samples from valid batches
    valid_mask = np.isin(batch_labels, valid_batches)
    data_mat_valid = data_mat[:, valid_mask]
    batch_valid = batch_labels[valid_mask]
    y_valid = y[valid_mask].reset_index(drop=True)
    ds_valid = ds[valid_mask].reset_index(drop=True)

    # Replace NaN/Inf
    data_mat_valid = np.nan_to_num(data_mat_valid, nan=0.0, posinf=0.0, neginf=0.0)

    # Check for zero-variance genes (ComBat will fail on them)
    gene_var = data_mat_valid.var(axis=1)
    nonzero_var_mask = gene_var > 1e-10
    if nonzero_var_mask.sum() < 10:
        logger.warning("ComBat: too few non-zero-variance genes; falling back")
        return per_dataset_zscore(datasets, labels, landmark_genes)

    # Subset to non-zero variance genes
    data_subset = data_mat_valid[nonzero_var_mask]
    gene_subset = [common_genes[i] for i in range(len(common_genes)) if nonzero_var_mask[i]]

    logger.info(
        f"[combat] Running neuroCombat on {data_subset.shape[1]} samples, "
        f"{data_subset.shape[0]} genes, {len(valid_batches)} batches"
    )

    try:
        # neuroCombat expects: dat = genes x samples (pd.DataFrame),
        # covars = pd.DataFrame with a batch column, batch_col = column name
        covars = pd.DataFrame({"batch": batch_valid})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combat_result = neuroCombat(
                dat=pd.DataFrame(data_subset),
                covars=covars,
                batch_col="batch",
                eb=True,
            )
        corrected_mat = combat_result["data"]  # genes x samples
    except Exception as exc:
        logger.error(f"ComBat failed: {exc}; falling back to per_dataset_zscore")
        return per_dataset_zscore(datasets, labels, landmark_genes)

    # Transpose back to samples x genes
    X_combat = pd.DataFrame(
        corrected_mat.T,
        columns=gene_subset,
    )

    # Re-add zero-variance genes as zeros
    for g in common_genes:
        if g not in X_combat.columns:
            X_combat[g] = 0.0
    X_combat = X_combat[common_genes]

    # Z-score the corrected data
    cb_mean = X_combat.mean(axis=0)
    cb_std = X_combat.std(axis=0).replace(0, 1)
    X_combat = (X_combat - cb_mean) / cb_std
    X_combat = X_combat.fillna(0.0)

    logger.info(f"[combat] Corrected: {X_combat.shape[0]} patients x {X_combat.shape[1]} genes")
    return X_combat, y_valid, ds_valid, common_genes


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BATCH_METHODS = {
    "per_dataset_zscore": per_dataset_zscore,
    "quantile_norm": quantile_norm,
    "rank_norm": rank_norm,
    "combat": combat_correction,
}


def apply_batch_correction(
    method_name: str,
    datasets: dict[str, pd.DataFrame],
    labels: dict[str, pd.Series],
    landmark_genes: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Apply a named batch correction method.

    Returns (X_corrected, y_labels, dataset_ids, gene_list).
    """
    if method_name not in BATCH_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Available: {list(BATCH_METHODS.keys())}"
        )
    return BATCH_METHODS[method_name](datasets, labels, landmark_genes)
