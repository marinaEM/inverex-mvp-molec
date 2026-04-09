"""
Expression Tokenizer (Full) -- Three Input Encodings for Ablation
=================================================================
Converts raw expression vectors into model inputs with THREE encoding modes:

1. **Raw**: normalized log2(expr+1) values, z-scored per gene
2. **Rank**: within-sample percentile ranks (0-1)
3. **Hybrid**: rank + binned magnitude + gene-presence flag

All modes produce:
    gene_ids    : int   tensor [n_genes+1]  -- gene indices (0=[CLS], 1..G)
    values      : float tensor [n_genes+1]  -- main continuous feature
    mag_bins    : int   tensor [n_genes+1]  -- magnitude bin (hybrid only, else 0)
    presence    : float tensor [n_genes+1]  -- 1.0 if gene present, 0.0 if missing
    expr_raw    : float tensor [n_genes+1]  -- raw log2(x+1) values (always available)
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
RESULTS = ROOT / "results" / "foundation_full"

N_MAG_BINS = 64
MAG_CLIP = 3.0


class ExpressionTokenizer:
    """Tokenizes raw expression with multiple encoding strategies for ablation."""

    def __init__(
        self,
        gene_list: list[str],
        gene2idx: dict[str, int],
        encoding: str = "hybrid",  # "raw", "rank", or "hybrid"
        gene_means: Optional[np.ndarray] = None,
        gene_stds: Optional[np.ndarray] = None,
    ):
        self.gene_list = gene_list
        self.gene2idx = gene2idx
        self.n_genes = len(gene_list)
        self.encoding = encoding
        self.gene_means = gene_means
        self.gene_stds = gene_stds

        # Pre-build a fast lookup dict from gene name to position
        self._gene_to_pos = {g: i for i, g in enumerate(gene_list)}

    def fit_stats(self, expression_dfs: list[pd.DataFrame]) -> None:
        """Compute per-gene mean/std from a list of expression DataFrames."""
        # Collect values gene-by-gene using vectorized approach
        gene_set = set(self.gene_list)
        all_vals = []
        for df in expression_dfs:
            common = [g for g in self.gene_list if g in df.columns]
            if not common:
                continue
            sub = df[common].values.astype(np.float32)
            # log2(x+1) transform
            sub = np.log2(np.maximum(sub, 0) + 1)
            # Pad missing genes with NaN
            full = np.full((sub.shape[0], self.n_genes), np.nan, dtype=np.float32)
            idxs = [self._gene_to_pos[g] for g in common]
            full[:, idxs] = sub
            all_vals.append(full)

        stacked = np.concatenate(all_vals, axis=0)
        self.gene_means = np.nanmean(stacked, axis=0)
        self.gene_stds = np.nanstd(stacked, axis=0)
        self.gene_stds[self.gene_stds < 1e-6] = 1.0

        logger.info(
            "Fitted stats on %d samples, %d genes", stacked.shape[0], self.n_genes
        )

    def save_stats(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = RESULTS / "tokenizer_stats_full.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            gene_means=self.gene_means,
            gene_stds=self.gene_stds,
            gene_list=np.array(self.gene_list),
        )

    def load_stats(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = RESULTS / "tokenizer_stats_full.npz"
        data = np.load(path, allow_pickle=True)
        self.gene_means = data["gene_means"]
        self.gene_stds = data["gene_stds"]

    def tokenize_dataframe(
        self, expr_df: pd.DataFrame
    ) -> dict[str, torch.Tensor]:
        """
        Vectorized tokenization for an entire DataFrame (rows=samples, cols=genes).
        Much faster than per-sample tokenization.

        Returns dict with keys: gene_ids, values, mag_bins, presence, expr_raw
        All have shape [N, n_genes + 1] (first position = [CLS]).
        """
        N = len(expr_df)
        n = self.n_genes

        # Find overlapping genes
        common = [g for g in self.gene_list if g in expr_df.columns]
        common_idxs = np.array([self._gene_to_pos[g] for g in common])

        # Extract raw expression for common genes
        raw = np.zeros((N, n), dtype=np.float32)
        presence = np.zeros((N, n), dtype=np.float32)
        if common:
            raw[:, common_idxs] = expr_df[common].values.astype(np.float32)
            presence[:, common_idxs] = 1.0

        # Log2 transform
        log_vals = np.log2(np.maximum(raw, 0) + 1)

        # Build values based on encoding
        if self.encoding == "raw":
            # Z-scored log expression
            if self.gene_means is not None:
                values = (log_vals - self.gene_means[np.newaxis, :]) / self.gene_stds[np.newaxis, :]
                values = np.clip(values, -MAG_CLIP, MAG_CLIP)
            else:
                values = log_vals
            mag_bins_arr = np.zeros((N, n), dtype=np.int64)

        elif self.encoding == "rank":
            # Within-sample percentile ranks
            values = np.zeros((N, n), dtype=np.float32)
            for i in range(N):
                present_mask = presence[i] > 0
                if present_mask.sum() > 1:
                    present_vals = log_vals[i, present_mask]
                    order = np.argsort(np.argsort(present_vals)).astype(np.float32)
                    order /= max(present_mask.sum() - 1, 1)
                    values[i, present_mask] = order
            mag_bins_arr = np.zeros((N, n), dtype=np.int64)

        elif self.encoding == "hybrid":
            # Ranks as primary value
            values = np.zeros((N, n), dtype=np.float32)
            for i in range(N):
                present_mask = presence[i] > 0
                if present_mask.sum() > 1:
                    present_vals = log_vals[i, present_mask]
                    order = np.argsort(np.argsort(present_vals)).astype(np.float32)
                    order /= max(present_mask.sum() - 1, 1)
                    values[i, present_mask] = order

            # Magnitude bins
            if self.gene_means is not None:
                z = (log_vals - self.gene_means[np.newaxis, :]) / self.gene_stds[np.newaxis, :]
                z = np.clip(z, -MAG_CLIP, MAG_CLIP)
                mag_bins_arr = ((z + MAG_CLIP) / (2 * MAG_CLIP) * (N_MAG_BINS - 1)).astype(np.int64)
                mag_bins_arr = np.clip(mag_bins_arr, 0, N_MAG_BINS - 1)
            else:
                mag_bins_arr = np.zeros((N, n), dtype=np.int64)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        # Prepend [CLS] token (position 0)
        gene_ids = np.zeros((N, n + 1), dtype=np.int64)
        gene_ids[:, 1:] = np.arange(1, n + 1)[np.newaxis, :]

        values_full = np.zeros((N, n + 1), dtype=np.float32)
        values_full[:, 1:] = values

        mag_bins_full = np.zeros((N, n + 1), dtype=np.int64)
        mag_bins_full[:, 1:] = mag_bins_arr

        presence_full = np.zeros((N, n + 1), dtype=np.float32)
        presence_full[:, 0] = 1.0  # CLS is always "present"
        presence_full[:, 1:] = presence

        expr_raw_full = np.zeros((N, n + 1), dtype=np.float32)
        expr_raw_full[:, 1:] = log_vals

        return {
            "gene_ids": torch.tensor(gene_ids, dtype=torch.long),
            "values": torch.tensor(values_full, dtype=torch.float32),
            "mag_bins": torch.tensor(mag_bins_full, dtype=torch.long),
            "presence": torch.tensor(presence_full, dtype=torch.float32),
            "expr_raw": torch.tensor(expr_raw_full, dtype=torch.float32),
        }

    def tokenize_sample(self, expr_series: pd.Series) -> dict[str, torch.Tensor]:
        """Tokenize a single sample (pd.Series with gene symbols as index)."""
        # Convert to single-row DataFrame and use vectorized method
        df = pd.DataFrame([expr_series])
        batch = self.tokenize_dataframe(df)
        return {k: v[0] for k, v in batch.items()}
