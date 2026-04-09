"""
Leakage-free batch correction normalizers for LODO evaluation.

The previous pipeline (retrain_full_combined_v3.py) applied ComBat to ALL
samples before the LODO split, leaking held-out batch statistics into the
correction.  These normalizers enforce a strict fit(train)/transform(any)
interface so the held-out dataset is never used for parameter estimation.

Three methods:
    A. ReferenceAnchoredNormalizer — learns grand mean + pooled std from
       training batches; maps new batches to this reference distribution.
    B. RankNormalizer — within-sample gene ranking (no parameters, no leakage
       by construction).  Optional inverse-normal transform.
    C. FrozenComBat — standard ComBat fitted on training batches only; new
       batches are corrected with frozen grand mean + pooled variance.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Method A: Reference-Anchored Normalization
# ---------------------------------------------------------------------------

class ReferenceAnchoredNormalizer:
    """
    Learns per-gene reference statistics (mean, std) from training batches.
    Applies correction to new samples using ONLY these frozen statistics.

    Training:
        For each gene, compute the grand mean and pooled within-batch std
        across all training batches.  This becomes the reference distribution.

    Transform:
        For each batch, standardize to zero mean / unit variance within-batch,
        then rescale to the reference distribution.
    """

    def __init__(self):
        self.reference_mean = None
        self.reference_std = None
        self.is_fitted = False

    def fit(self, expression_df: pd.DataFrame, batch_labels: pd.Series):
        """Compute reference distribution from training data ONLY."""
        self.reference_mean = expression_df.mean(axis=0)

        batches = pd.Series(batch_labels.values, index=expression_df.index)
        within_batch_vars = []
        within_batch_ns = []
        for batch_id in batches.unique():
            batch_data = expression_df.loc[batches == batch_id]
            if len(batch_data) > 1:
                within_batch_vars.append(
                    batch_data.var(axis=0) * (len(batch_data) - 1)
                )
                within_batch_ns.append(len(batch_data) - 1)

        if within_batch_ns:
            pooled_var = sum(within_batch_vars) / sum(within_batch_ns)
            self.reference_std = np.sqrt(pooled_var)
        else:
            self.reference_std = expression_df.std(axis=0)

        self.reference_std = self.reference_std.replace(0, 1.0)
        self.reference_std[self.reference_std < 1e-6] = 1.0
        self.is_fitted = True

    def transform(self, expression_df: pd.DataFrame, batch_labels: pd.Series):
        """Correct new data using the FROZEN reference."""
        assert self.is_fitted, "Must fit() before transform()"

        corrected = expression_df.copy()
        batches = pd.Series(batch_labels.values, index=expression_df.index)

        for batch_id in batches.unique():
            mask = batches == batch_id
            batch_data = expression_df.loc[mask]

            if len(batch_data) > 1:
                batch_mean = batch_data.mean(axis=0)
                batch_std = batch_data.std(axis=0)
                batch_std = batch_std.replace(0, 1.0)
                batch_std[batch_std < 1e-6] = 1.0

                standardized = (batch_data - batch_mean) / batch_std
                corrected.loc[mask] = (
                    standardized * self.reference_std + self.reference_mean
                )
            else:
                # Single sample — center to reference mean
                sample_mean = batch_data.iloc[0].mean()
                ref_mean_scalar = self.reference_mean.mean()
                corrected.loc[mask] = batch_data - sample_mean + ref_mean_scalar

        return corrected


# ---------------------------------------------------------------------------
# Method B: Rank Normalization
# ---------------------------------------------------------------------------

class RankNormalizer:
    """
    Converts each sample independently to within-sample gene ranks.
    No parameters to fit — completely batch-invariant by construction.

    Methods:
        "rank" — raw ranks (scaled to [0, 1])
        "quantile" — inverse-normal transform of ranks
    """

    def __init__(self, method: str = "rank"):
        assert method in ("rank", "quantile"), f"Unknown method: {method}"
        self.method = method

    def fit(self, expression_df: pd.DataFrame, batch_labels: pd.Series):
        """No-op. Rank normalization has no parameters."""
        pass

    def transform(self, expression_df: pd.DataFrame, batch_labels: pd.Series = None):
        """Transform each sample independently."""
        mat = expression_df.values.copy()
        n_samples, n_genes = mat.shape

        ranked = np.zeros_like(mat)
        for i in range(n_samples):
            ranked[i, :] = rankdata(mat[i, :], method="average")

        if self.method == "rank":
            # Scale to [0, 1]
            if n_genes > 1:
                ranked = (ranked - 1) / (n_genes - 1)
        elif self.method == "quantile":
            # Map ranks to quantiles of standard normal
            quantiles = (ranked - 0.5) / n_genes
            ranked = norm.ppf(np.clip(quantiles, 1e-7, 1 - 1e-7))

        return pd.DataFrame(ranked, index=expression_df.index, columns=expression_df.columns)


# ---------------------------------------------------------------------------
# Method C: Frozen ComBat
# ---------------------------------------------------------------------------

class FrozenComBat:
    """
    Standard ComBat location-scale adjustment, but:
    - fit() only uses training batches
    - transform() applies frozen grand mean + pooled variance to any data

    Test batches get corrected using only training-derived statistics.
    """

    def __init__(self):
        self.grand_mean = None
        self.pooled_var = None
        self.is_fitted = False

    def fit(self, expression_df: pd.DataFrame, batch_labels: pd.Series):
        """Estimate correction parameters from training data ONLY."""
        self.grand_mean = expression_df.mean(axis=0)

        batches = pd.Series(batch_labels.values, index=expression_df.index)
        residuals = expression_df.copy()
        for batch_id in batches.unique():
            mask = batches == batch_id
            batch_mean = expression_df.loc[mask].mean(axis=0)
            residuals.loc[mask] = expression_df.loc[mask] - batch_mean

        self.pooled_var = residuals.var(axis=0)
        self.pooled_var = self.pooled_var.replace(0, 1.0)
        self.pooled_var[self.pooled_var < 1e-6] = 1.0
        self.is_fitted = True

    def transform(self, expression_df: pd.DataFrame, batch_labels: pd.Series):
        """Apply frozen correction to any data."""
        assert self.is_fitted, "Must fit() before transform()"

        corrected = expression_df.copy()
        batches = pd.Series(batch_labels.values, index=expression_df.index)

        for batch_id in batches.unique():
            mask = batches == batch_id
            batch_data = expression_df.loc[mask]

            if len(batch_data) > 1:
                batch_mean = batch_data.mean(axis=0)
                batch_var = batch_data.var(axis=0)
                batch_var = batch_var.replace(0, 1.0)
                batch_var[batch_var < 1e-6] = 1.0

                standardized = (batch_data - batch_mean) / np.sqrt(batch_var)
                corrected.loc[mask] = (
                    standardized * np.sqrt(self.pooled_var) + self.grand_mean
                )
            else:
                # Single sample: center to grand mean
                corrected.loc[mask] = (
                    batch_data
                    - batch_data.mean(axis=1).values[0]
                    + self.grand_mean.mean()
                )

        return corrected
