"""
Expression Encoder
==================
Gene-aware encoder that produces a [CLS] patient embedding.

Architecture (CPU-optimized)
-----------------------------
1. Per-gene feature: gene_embedding + rank_proj + mag_embedding
   (all three summed, then LayerNorm) → [B, G, d_model]

2. Multi-Head Attention Pooling: K learnable query tokens attend to gene
   features using efficient batched matrix multiply.

3. Latent MLP: the K pooled tokens are concatenated and passed through
   an MLP to produce the final d_model-dimensional embedding.

This avoids the O(G^2) self-attention bottleneck.  The only attention is
K queries × G keys, which for K=8, G=2000 is very small.

Config (CPU-feasible)
---------------------
    d_model    = 128
    n_pool_heads = 8 (attention pooling queries)
    n_layers   = 2 (MLP layers after pooling)
    n_genes    = 2000
    n_mag_bins = 64
"""

from __future__ import annotations

import os, math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class EncoderConfig:
    n_genes: int = 2_000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    n_mag_bins: int = 64
    n_latents: int = 16     # kept for config compat
    n_pool_heads: int = 8   # attention pooling heads
    dropout: float = 0.1


class AttentionPooling(nn.Module):
    """
    Multi-head attention pooling: K learnable queries attend to G gene features.
    Output: [B, K, d_model] where K = n_queries.
    """

    def __init__(self, d_model: int, n_queries: int = 8, n_heads: int = 4):
        super().__init__()
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_feats: torch.Tensor) -> torch.Tensor:
        """
        gene_feats: [B, G, d]
        returns: [B, K, d]
        """
        B, G, d = gene_feats.shape
        H = self.n_heads
        dh = self.d_head
        K = self.n_queries

        # Project keys and values
        keys = self.k_proj(gene_feats).view(B, G, H, dh).transpose(1, 2)    # [B,H,G,dh]
        values = self.v_proj(gene_feats).view(B, G, H, dh).transpose(1, 2)  # [B,H,G,dh]

        # Expand queries
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, d]
        q = q.view(B, K, H, dh).transpose(1, 2)          # [B, H, K, dh]

        # Scaled dot-product attention
        scale = math.sqrt(dh)
        attn = torch.matmul(q, keys.transpose(-2, -1)) / scale  # [B, H, K, G]
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.matmul(attn, values)  # [B, H, K, dh]
        out = out.transpose(1, 2).contiguous().view(B, K, d)  # [B, K, d]
        out = self.out_proj(out)
        out = self.norm(out + self.queries.unsqueeze(0))

        return out


class ExpressionEncoder(nn.Module):
    """
    Gene-aware expression encoder with attention pooling.

    Input: per-gene features (gene_id, rank, magnitude_bin)
    Output: [CLS] patient embedding (d_model-dim)
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Gene embedding: 0=[CLS]/unused, 1..n_genes
        self.gene_embedding = nn.Embedding(cfg.n_genes + 1, cfg.d_model)

        # Rank projection: scalar → d_model
        self.rank_proj = nn.Sequential(
            nn.Linear(1, cfg.d_model),
            nn.GELU(),
        )

        # Magnitude bin embedding
        self.mag_embedding = nn.Embedding(cfg.n_mag_bins + 1, cfg.d_model)

        # Gene-level normalization
        self.gene_norm = nn.LayerNorm(cfg.d_model)

        # Attention pooling: G genes → K latent tokens
        self.pool = AttentionPooling(cfg.d_model, n_queries=cfg.n_pool_heads, n_heads=cfg.n_heads)

        # MLP tower: K*d → d → d (produces final embedding)
        pool_dim = cfg.n_pool_heads * cfg.d_model
        layers = []
        for i in range(cfg.n_layers):
            d_in = pool_dim if i == 0 else cfg.d_model
            layers.extend([
                nn.Linear(d_in, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ])
        self.mlp = nn.Sequential(*layers)

        # Final norm
        self.cls_norm = nn.LayerNorm(cfg.d_model)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        gene_ids: torch.Tensor,    # [B, G] long
        ranks: torch.Tensor,       # [B, G] float
        mag_bins: torch.Tensor,    # [B, G] long
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        cls_emb : [B, d_model] – patient embedding
        latents : [B, n_pool_heads, d_model] – attention-pooled latents
        """
        # Per-gene features
        gene_emb = self.gene_embedding(gene_ids)        # [B, G, d]
        rank_emb = self.rank_proj(ranks.unsqueeze(-1))  # [B, G, d]
        mag_emb = self.mag_embedding(mag_bins)          # [B, G, d]

        gene_feats = self.gene_norm(gene_emb + rank_emb + mag_emb)  # [B, G, d]

        # Attention pooling: [B, G, d] → [B, K, d]
        latents = self.pool(gene_feats)  # [B, K, d]

        # Flatten and MLP
        B = gene_ids.shape[0]
        pooled = latents.view(B, -1)  # [B, K*d]
        cls_emb = self.mlp(pooled)    # [B, d]
        cls_emb = self.cls_norm(cls_emb)

        return cls_emb, latents

    def get_embedding_dim(self) -> int:
        return self.cfg.d_model
