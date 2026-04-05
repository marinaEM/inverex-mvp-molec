"""
Pretraining Objectives
======================
Three self-supervised objectives for the expression foundation model:

1. **Masked Gene Prediction (MGP)**
   - Mask 15% of gene values (set ranks to 0) before encoding.
   - From the [CLS] embedding, predict the original rank of each masked gene.
   - Loss: MSE on predicted vs. true rank.

2. **Pathway Activity Prediction (PAP)**
   - From the [CLS] embedding, predict ssGSEA Hallmark pathway scores.
   - Loss: MSE on predicted vs. computed pathway scores.

3. **Subtype Prediction (SUB)**
   - From the [CLS] embedding, predict PAM50 subtype
     (inferred from ESR1/ERBB2/MKI67 expression quartiles).
   - Loss: cross-entropy.

The Perceiver-style encoder outputs latent tokens (not per-gene outputs),
so MGP uses the full latent representation to decode masked gene ranks.
"""

from __future__ import annotations

import os, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # inverex-mvp
RESULTS = ROOT / "results" / "foundation"

# ---------------------------------------------------------------------------
# Pathway score computation
# ---------------------------------------------------------------------------

def compute_pathway_scores_simple(
    expr_df: pd.DataFrame,
    gene_list: list[str],
) -> tuple[np.ndarray | None, list[str]]:
    """
    Compute simplified pathway scores using gene-set mean rank expression.

    Returns (scores [n_samples, n_pathways], pathway_names) or (None, []).
    """
    try:
        import gseapy
        try:
            hallmark = gseapy.get_library("MSigDB_Hallmark_2020", organism="Human")
        except Exception:
            try:
                hallmark = gseapy.get_library("MSigDB_Hallmark_2020")
            except Exception as e:
                logger.warning("Could not load Hallmark gene sets: %s", e)
                return None, []
    except Exception as e:
        logger.warning("gseapy not available: %s", e)
        return None, []

    available = [g for g in gene_list if g in expr_df.columns]
    if len(available) < 100:
        logger.warning("Too few genes (%d) for pathway scoring", len(available))
        return None, []

    sub = expr_df[available].copy()
    # Fill NaN before ranking to avoid propagation
    sub = sub.fillna(0.0)
    ranked = sub.rank(axis=1, pct=True)

    pathway_names = []
    scores_list = []
    for pw_name, pw_genes in hallmark.items():
        overlap = [g for g in pw_genes if g in available]
        if len(overlap) >= 3:
            s = ranked[overlap].mean(axis=1).values.astype(np.float32)
            s = np.nan_to_num(s, nan=0.5)  # replace any remaining NaN with 0.5 (median rank)
            scores_list.append(s)
            pathway_names.append(pw_name)

    if not scores_list:
        return None, []

    scores = np.column_stack(scores_list)
    logger.info("Computed %d pathway scores for %d samples", len(pathway_names), len(expr_df))
    return scores, pathway_names


def infer_pam50_subtype(
    expr_df: pd.DataFrame,
) -> np.ndarray | None:
    """
    Infer simplified PAM50-like subtype from ESR1, ERBB2, MKI67 expression.

    Classes:
        0 = Luminal A  (ESR1-high, ERBB2-low, MKI67-low)
        1 = Luminal B  (ESR1-high, ERBB2-low, MKI67-high)
        2 = HER2+      (ERBB2-high)
        3 = Basal       (ESR1-low, ERBB2-low)
    """
    needed = ["ESR1", "ERBB2", "MKI67"]
    if not all(g in expr_df.columns for g in needed):
        logger.warning("Missing ESR1/ERBB2/MKI67 for subtype inference")
        return None

    esr1_raw = np.nan_to_num(expr_df["ESR1"].values.astype(float), nan=0.0)
    erbb2_raw = np.nan_to_num(expr_df["ERBB2"].values.astype(float), nan=0.0)
    mki67_raw = np.nan_to_num(expr_df["MKI67"].values.astype(float), nan=0.0)
    esr1 = np.log2(np.maximum(esr1_raw, 0) + 1)
    erbb2 = np.log2(np.maximum(erbb2_raw, 0) + 1)
    mki67 = np.log2(np.maximum(mki67_raw, 0) + 1)

    esr1_med = np.median(esr1)
    erbb2_q75 = np.percentile(erbb2, 75)
    mki67_med = np.median(mki67)

    subtypes = np.zeros(len(expr_df), dtype=np.int64)
    for i in range(len(expr_df)):
        if erbb2[i] > erbb2_q75:
            subtypes[i] = 2  # HER2+
        elif esr1[i] > esr1_med:
            if mki67[i] > mki67_med:
                subtypes[i] = 1  # Luminal B
            else:
                subtypes[i] = 0  # Luminal A
        else:
            subtypes[i] = 3  # Basal

    logger.info(
        "Inferred subtypes: LumA=%d, LumB=%d, HER2=%d, Basal=%d",
        (subtypes == 0).sum(), (subtypes == 1).sum(),
        (subtypes == 2).sum(), (subtypes == 3).sum(),
    )
    return subtypes


# ---------------------------------------------------------------------------
# Objective heads
# ---------------------------------------------------------------------------

class MaskedGenePredictionHead(nn.Module):
    """
    Predict rank of masked genes from [CLS] + gene identity.

    For each masked gene, we concatenate the [CLS] embedding with the
    gene embedding and predict the rank.
    """

    def __init__(self, d_model: int, gene_embedding: nn.Embedding):
        super().__init__()
        self.gene_embedding = gene_embedding  # shared with encoder
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        cls_emb: torch.Tensor,      # [B, d]
        masked_gene_ids: torch.Tensor,  # [B, n_masked] long
        true_ranks: torch.Tensor,       # [B, n_masked] float
    ) -> torch.Tensor:
        """Returns MSE loss on masked positions."""
        B, n_masked = masked_gene_ids.shape
        d = cls_emb.shape[1]

        # Gene embeddings for masked genes
        gene_embs = self.gene_embedding(masked_gene_ids)  # [B, n_masked, d]

        # Expand [CLS] to match
        cls_exp = cls_emb.unsqueeze(1).expand(-1, n_masked, -1)  # [B, n_masked, d]

        # Concatenate and predict
        combined = torch.cat([cls_exp, gene_embs], dim=-1)  # [B, n_masked, 2d]
        pred = self.proj(combined).squeeze(-1)  # [B, n_masked]

        loss = F.mse_loss(pred, true_ranks)
        return loss


class PathwayActivityHead(nn.Module):
    """Predict pathway scores from [CLS] embedding."""

    def __init__(self, d_model: int, n_pathways: int):
        super().__init__()
        self.n_pathways = n_pathways
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_pathways),
        )

    def forward(
        self, cls_emb: torch.Tensor, true_scores: torch.Tensor
    ) -> torch.Tensor:
        pred = self.proj(cls_emb)  # [B, n_pathways]
        loss = F.mse_loss(pred, true_scores)
        return loss


class SubtypePredictionHead(nn.Module):
    """Predict PAM50 subtype from [CLS] embedding."""

    def __init__(self, d_model: int, n_classes: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self, cls_emb: torch.Tensor, true_labels: torch.Tensor
    ) -> torch.Tensor:
        logits = self.proj(cls_emb)
        loss = F.cross_entropy(logits, true_labels)
        return loss


# ---------------------------------------------------------------------------
# Combined pretraining model
# ---------------------------------------------------------------------------

class FoundationPretrainModel(nn.Module):
    """Wraps the encoder + all pretraining heads."""

    def __init__(self, encoder, n_pathways: int, mask_ratio: float = 0.15):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d = encoder.get_embedding_dim()

        self.mgp_head = MaskedGenePredictionHead(d, encoder.gene_embedding)
        self.pap_head = PathwayActivityHead(d, n_pathways) if n_pathways > 0 else None
        self.sub_head = SubtypePredictionHead(d, n_classes=4)

    def create_mask(
        self, gene_ids: torch.Tensor, ranks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask `mask_ratio` of gene positions.

        Returns:
            masked_ranks  : ranks with masked positions zeroed
            mask_indices  : [B, n_masked] positional indices (0-based into L)
            masked_gene_ids : [B, n_masked] gene ids at masked positions
            true_ranks    : [B, n_masked] original ranks at masked positions
        """
        B, L = gene_ids.shape
        n_masked = max(1, int(L * self.mask_ratio))

        mask_indices = torch.zeros(B, n_masked, dtype=torch.long, device=gene_ids.device)
        for b in range(B):
            perm = torch.randperm(L, device=gene_ids.device)[:n_masked]
            mask_indices[b] = perm

        # Gather true ranks and gene ids at masked positions
        true_ranks = torch.gather(ranks, 1, mask_indices)
        masked_gene_ids = torch.gather(gene_ids, 1, mask_indices)

        # Zero out masked positions in ranks
        masked_ranks = ranks.clone()
        masked_ranks.scatter_(1, mask_indices, 0.0)

        return masked_ranks, mask_indices, masked_gene_ids, true_ranks

    def forward(
        self,
        gene_ids: torch.Tensor,         # [B, G]
        ranks: torch.Tensor,            # [B, G]
        mag_bins: torch.Tensor,         # [B, G]
        pathway_scores: torch.Tensor | None = None,
        subtypes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with all objectives."""
        # Create mask
        masked_ranks, mask_indices, masked_gene_ids, true_ranks = self.create_mask(
            gene_ids, ranks
        )

        # Encode with masked ranks
        cls_emb, latents = self.encoder(gene_ids, masked_ranks, mag_bins)

        losses = {}

        # MGP loss: predict masked gene ranks from [CLS] + gene identity
        losses["mgp"] = self.mgp_head(cls_emb, masked_gene_ids, true_ranks)

        # Pathway loss
        if self.pap_head is not None and pathway_scores is not None:
            losses["pap"] = self.pap_head(cls_emb, pathway_scores)
        else:
            losses["pap"] = torch.tensor(0.0, device=cls_emb.device)

        # Subtype loss
        if subtypes is not None:
            valid = subtypes >= 0
            if valid.any():
                losses["sub"] = self.sub_head(cls_emb[valid], subtypes[valid])
            else:
                losses["sub"] = torch.tensor(0.0, device=cls_emb.device)
        else:
            losses["sub"] = torch.tensor(0.0, device=cls_emb.device)

        # Total loss
        losses["total"] = losses["mgp"] + 0.5 * losses["pap"] + 0.5 * losses["sub"]
        losses["cls_emb"] = cls_emb

        return losses
