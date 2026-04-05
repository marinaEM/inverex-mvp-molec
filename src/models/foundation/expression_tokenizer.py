"""
Expression Tokenizer
====================
Converts a raw expression vector into model inputs:

1. **Rank encoding**: within-sample rank of each gene (0–1 normalized)
2. **Magnitude encoding**: log2(expr+1) value, z-scored per gene across
   the pretraining corpus, then clipped to [-3, 3] and binned into 64 bins.

Produces for each sample:
    gene_ids  : int   tensor [n_genes]  – gene indices (1-based)
    ranks     : float tensor [n_genes]  – normalized ranks
    mag_bins  : int   tensor [n_genes]  – magnitude bin indices
    expr_vals : float tensor [n_genes]  – raw log2(x+1) values

Note: [CLS] is handled internally by the Perceiver-style encoder
(as a learnable latent token), NOT prepended here.
"""

from __future__ import annotations

import os, logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # inverex-mvp
RESULTS = ROOT / "results" / "foundation"

N_MAG_BINS = 64
MAG_CLIP = 3.0


class ExpressionTokenizer:
    """Tokenizes raw expression into rank + magnitude representations."""

    def __init__(
        self,
        gene_list: list[str],
        gene2idx: dict[str, int],
        gene_means: Optional[np.ndarray] = None,
        gene_stds: Optional[np.ndarray] = None,
    ):
        self.gene_list = gene_list
        self.gene2idx = gene2idx
        self.n_genes = len(gene_list)
        self.gene_means = gene_means  # shape [n_genes]
        self.gene_stds = gene_stds    # shape [n_genes]

        # Precompute gene_list set for fast lookup
        self._gene_set = set(gene_list)

    def fit_stats(self, expression_dfs: list[pd.DataFrame]) -> None:
        """Compute per-gene mean/std from a list of expression DataFrames."""
        # Build index map once
        gene_idx_map = {g: i for i, g in enumerate(self.gene_list)}

        all_vals = []
        for df in expression_dfs:
            common = [g for g in df.columns if g in gene_idx_map]
            if not common:
                continue
            sub = df[common].values.astype(np.float32)
            sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
            # log2(x+1) transform
            sub = np.log2(np.maximum(sub, 0) + 1)
            # Pad missing genes with NaN
            full = np.full((sub.shape[0], self.n_genes), np.nan, dtype=np.float32)
            idxs = [gene_idx_map[g] for g in common]
            full[:, idxs] = sub
            all_vals.append(full)

        stacked = np.concatenate(all_vals, axis=0)  # [N_total, n_genes]
        self.gene_means = np.nanmean(stacked, axis=0)
        self.gene_stds = np.nanstd(stacked, axis=0)
        self.gene_stds[self.gene_stds < 1e-6] = 1.0  # avoid division by zero

        logger.info(
            "Fitted stats on %d samples, %d genes", stacked.shape[0], self.n_genes
        )

    def save_stats(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = RESULTS / "tokenizer_stats.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            gene_means=self.gene_means,
            gene_stds=self.gene_stds,
            gene_list=np.array(self.gene_list),
        )

    def load_stats(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = RESULTS / "tokenizer_stats.npz"
        data = np.load(path, allow_pickle=True)
        self.gene_means = data["gene_means"]
        self.gene_stds = data["gene_stds"]

    def tokenize_sample(
        self, expr_series: pd.Series
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize a single sample (pd.Series with gene symbols as index).

        Returns dict with keys: gene_ids, ranks, mag_bins, expr_vals
        All have shape [n_genes] (no [CLS] — encoder handles that internally).
        """
        n = self.n_genes

        # Extract values for our gene universe
        raw = np.zeros(n, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        for i, g in enumerate(self.gene_list):
            if g in expr_series.index:
                val = float(expr_series[g])
                if np.isfinite(val) and val > 0:
                    raw[i] = val
                    mask[i] = True

        # Log2 transform
        log_vals = np.log2(raw + 1)

        # Rank encoding (among present genes)
        ranks = np.zeros(n, dtype=np.float32)
        if mask.sum() > 0:
            present_vals = log_vals[mask]
            order = np.argsort(np.argsort(present_vals)).astype(np.float32)
            order /= max(mask.sum() - 1, 1)
            ranks[mask] = order

        # Magnitude binning
        if self.gene_means is not None:
            z = (log_vals - self.gene_means) / self.gene_stds
            z = np.clip(z, -MAG_CLIP, MAG_CLIP)
            # Map [-3, 3] → [0, N_MAG_BINS-1]
            bins = ((z + MAG_CLIP) / (2 * MAG_CLIP) * (N_MAG_BINS - 1)).astype(int)
            bins = np.clip(bins, 0, N_MAG_BINS - 1)
        else:
            bins = np.zeros(n, dtype=int)

        # Gene ids are 1-based (0 reserved for [CLS] in embedding table)
        gene_ids = np.arange(1, n + 1)

        return {
            "gene_ids": torch.tensor(gene_ids, dtype=torch.long),
            "ranks": torch.tensor(ranks, dtype=torch.float32),
            "mag_bins": torch.tensor(bins, dtype=torch.long),
            "expr_vals": torch.tensor(log_vals, dtype=torch.float32),
        }

    def tokenize_batch(self, expr_df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Tokenize a DataFrame of samples (rows = samples, cols = genes)."""
        samples = []
        for idx in range(len(expr_df)):
            row = expr_df.iloc[idx]
            samples.append(self.tokenize_sample(row))

        return {
            key: torch.stack([s[key] for s in samples])
            for key in samples[0].keys()
        }

    def tokenize_batch_fast(self, expr_df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """
        Vectorized batch tokenization — much faster than per-sample loop.
        """
        n = self.n_genes
        B = len(expr_df)

        # Build matrix of gene values
        available = [g for g in self.gene_list if g in expr_df.columns]
        avail_idx = [self.gene_list.index(g) for g in available]

        raw = np.zeros((B, n), dtype=np.float32)
        if available:
            vals = expr_df[available].values.astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            vals = np.maximum(vals, 0.0)
            raw[:, avail_idx] = vals

        # Log2 transform
        log_vals = np.log2(raw + 1)

        # Ranks (per sample, among non-zero genes)
        ranks = np.zeros((B, n), dtype=np.float32)
        mask = raw > 0
        for b in range(B):
            m = mask[b]
            if m.sum() > 0:
                order = np.argsort(np.argsort(log_vals[b, m])).astype(np.float32)
                order /= max(m.sum() - 1, 1)
                ranks[b, m] = order

        # Magnitude bins
        if self.gene_means is not None:
            z = (log_vals - self.gene_means[None, :]) / self.gene_stds[None, :]
            z = np.clip(z, -MAG_CLIP, MAG_CLIP)
            bins = ((z + MAG_CLIP) / (2 * MAG_CLIP) * (N_MAG_BINS - 1)).astype(int)
            bins = np.clip(bins, 0, N_MAG_BINS - 1)
        else:
            bins = np.zeros((B, n), dtype=int)

        gene_ids = np.tile(np.arange(1, n + 1), (B, 1))

        return {
            "gene_ids": torch.tensor(gene_ids, dtype=torch.long),
            "ranks": torch.tensor(ranks, dtype=torch.float32),
            "mag_bins": torch.tensor(bins, dtype=torch.long),
            "expr_vals": torch.tensor(log_vals, dtype=torch.float32),
        }
