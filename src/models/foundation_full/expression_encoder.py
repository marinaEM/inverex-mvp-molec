"""
Expression Encoder (Full) -- Gene-Aware Transformer
====================================================
Two configurations:

- **medium**: d_model=256, n_layers=4, n_heads=4, ~2M params
- **full**  : d_model=384, n_layers=6, n_heads=8, ~5M params

Architecture:
    gene_embedding(gene_id) + value_projection(value) + [mag_embedding(bin)] + [presence_emb]
    --> LayerNorm --> TransformerEncoder (pre-norm, GELU)
    --> [CLS] output = patient embedding

Includes:
- Gradient reversal layer for domain adversarial training
- Configurable input encoding support (raw, rank, hybrid)
- Pathway-wise masking support
"""

from __future__ import annotations

import os, math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for domain adversarial training)
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIGS = {
    "medium": {
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "dropout": 0.1,
    },
    "full": {
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 8,
        "dropout": 0.1,
    },
}


@dataclass
class EncoderConfig:
    n_genes: int = 1_000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    n_mag_bins: int = 64
    dropout: float = 0.1
    use_mag_embedding: bool = True   # True for hybrid encoding
    use_presence_flag: bool = True   # True for hybrid encoding

    @classmethod
    def from_preset(cls, preset: str, n_genes: int, encoding: str = "hybrid") -> "EncoderConfig":
        """Create config from preset name ('medium' or 'full')."""
        c = CONFIGS[preset]
        use_mag = encoding == "hybrid"
        use_pres = encoding == "hybrid"
        return cls(
            n_genes=n_genes,
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            n_layers=c["n_layers"],
            dropout=c["dropout"],
            use_mag_embedding=use_mag,
            use_presence_flag=use_pres,
        )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ExpressionEncoder(nn.Module):
    """Gene-aware transformer encoder -> [CLS] patient embedding."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Gene embedding: index 0 = [CLS], 1..n_genes = genes
        self.gene_embedding = nn.Embedding(cfg.n_genes + 1, cfg.d_model)

        # Value projection: scalar value -> d_model
        self.value_proj = nn.Sequential(
            nn.Linear(1, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Magnitude bin embedding (hybrid only)
        if cfg.use_mag_embedding:
            self.mag_embedding = nn.Embedding(cfg.n_mag_bins + 1, cfg.d_model)
        else:
            self.mag_embedding = None

        # Presence flag projection (hybrid only)
        if cfg.use_presence_flag:
            self.presence_proj = nn.Linear(1, cfg.d_model)
        else:
            self.presence_proj = None

        # Input normalization
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.input_drop = nn.Dropout(cfg.dropout)

        # Transformer encoder (pre-norm for stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )

        # Final [CLS] normalization
        self.cls_norm = nn.LayerNorm(cfg.d_model)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        gene_ids: torch.Tensor,       # [B, L] long
        values: torch.Tensor,          # [B, L] float -- main continuous input
        mag_bins: Optional[torch.Tensor] = None,   # [B, L] long
        presence: Optional[torch.Tensor] = None,    # [B, L] float
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B, L] bool True=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        cls_emb   : [B, d_model]     -- patient embedding from [CLS]
        all_embs  : [B, L, d_model]  -- all position embeddings
        """
        # Build input representation
        gene_emb = self.gene_embedding(gene_ids)           # [B, L, d]
        val_emb = self.value_proj(values.unsqueeze(-1))    # [B, L, d]

        x = gene_emb + val_emb

        if self.mag_embedding is not None and mag_bins is not None:
            x = x + self.mag_embedding(mag_bins)

        if self.presence_proj is not None and presence is not None:
            x = x + self.presence_proj(presence.unsqueeze(-1))

        x = self.input_norm(x)
        x = self.input_drop(x)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Extract [CLS] embedding (position 0)
        cls_emb = self.cls_norm(x[:, 0, :])

        return cls_emb, x

    def get_embedding_dim(self) -> int:
        return self.cfg.d_model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Pretraining objective heads
# ---------------------------------------------------------------------------

class MaskedGenePredictionHead(nn.Module):
    """Predict value of masked gene from its transformer output."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        all_embs: torch.Tensor,      # [B, L, d]
        mask_indices: torch.Tensor,   # [B, n_masked]
        true_values: torch.Tensor,    # [B, n_masked]
    ) -> torch.Tensor:
        B, L, d = all_embs.shape
        idx = mask_indices.unsqueeze(-1).expand(-1, -1, d)
        masked_embs = torch.gather(all_embs, 1, idx)
        pred = self.proj(masked_embs).squeeze(-1)
        loss = F.mse_loss(pred, true_values)
        return loss


class PathwayActivityHead(nn.Module):
    """Predict pathway activity scores from [CLS]."""

    def __init__(self, d_model: int, n_pathways: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_pathways),
        )

    def forward(self, cls_emb: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        pred = self.proj(cls_emb)
        loss = F.mse_loss(pred, true_scores)
        return loss


class SubtypePredictionHead(nn.Module):
    """Predict PAM50 subtype from [CLS]."""

    def __init__(self, d_model: int, n_classes: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, cls_emb: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        logits = self.proj(cls_emb)
        loss = F.cross_entropy(logits, true_labels, ignore_index=-1)
        return loss


class MutationProxyHead(nn.Module):
    """Predict mutation status (TP53, PIK3CA, ERBB2) from [CLS] where available."""

    def __init__(self, d_model: int, n_mutations: int = 3):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 4, 1),
            )
            for _ in range(n_mutations)
        ])

    def forward(
        self,
        cls_emb: torch.Tensor,        # [B, d]
        mutation_labels: torch.Tensor,  # [B, n_mut] -- 0/1 or -1 for unknown
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=cls_emb.device)
        n_valid = 0
        for i, head in enumerate(self.heads):
            labels = mutation_labels[:, i]
            valid = labels >= 0
            if valid.sum() == 0:
                continue
            pred = head(cls_emb[valid]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, labels[valid].float())
            total_loss = total_loss + loss
            n_valid += 1
        if n_valid > 0:
            total_loss = total_loss / n_valid
        return total_loss


class DomainAdversarialHead(nn.Module):
    """Predict dataset/platform identity from [CLS] via gradient reversal."""

    def __init__(self, d_model: int, n_domains: int, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_domains),
        )

    def set_alpha(self, alpha: float):
        self.grl.set_alpha(alpha)

    def forward(self, cls_emb: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        reversed_emb = self.grl(cls_emb)
        logits = self.classifier(reversed_emb)
        loss = F.cross_entropy(logits, domain_labels)
        return loss


# ---------------------------------------------------------------------------
# Combined Pretraining Model
# ---------------------------------------------------------------------------

class FoundationPretrainModel(nn.Module):
    """
    Wraps the encoder + all pretraining objective heads.

    Configurable objectives via `objectives` dict:
        mgp: Masked Gene Prediction
        pap: Pathway Activity Prediction
        sub: Subtype Prediction
        mut: Mutation-proxy Prediction
        dav: Domain Adversarial
    """

    def __init__(
        self,
        encoder: ExpressionEncoder,
        objectives: dict,
        mask_ratio: float = 0.15,
        pathway_mask_prob: float = 0.20,
        pathway_gene_sets: Optional[dict[str, list[int]]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.pathway_mask_prob = pathway_mask_prob
        self.pathway_gene_sets = pathway_gene_sets  # {pathway_name: [gene_positions]}
        d = encoder.get_embedding_dim()

        # Build objective heads based on config
        self.obj_config = objectives
        self.mgp_head = MaskedGenePredictionHead(d) if objectives.get("mgp", True) else None
        self.pap_head = (
            PathwayActivityHead(d, objectives["pap_n_pathways"])
            if objectives.get("pap", False) and objectives.get("pap_n_pathways", 0) > 0
            else None
        )
        self.sub_head = SubtypePredictionHead(d, n_classes=4) if objectives.get("sub", False) else None
        self.mut_head = MutationProxyHead(d, n_mutations=3) if objectives.get("mut", False) else None
        self.dav_head = (
            DomainAdversarialHead(d, objectives["dav_n_domains"])
            if objectives.get("dav", False) and objectives.get("dav_n_domains", 0) > 0
            else None
        )

        # Objective weights
        self.weights = {
            "mgp": objectives.get("mgp_weight", 1.0),
            "pap": objectives.get("pap_weight", 0.5),
            "sub": objectives.get("sub_weight", 0.5),
            "mut": objectives.get("mut_weight", 0.3),
            "dav": objectives.get("dav_weight", 0.2),
        }

    def create_mask(
        self, gene_ids: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask `mask_ratio` of gene positions (not [CLS]).
        Optionally mask entire pathways with `pathway_mask_prob`.
        """
        B, L = gene_ids.shape
        n_genes = L - 1
        n_masked = max(1, int(n_genes * self.mask_ratio))

        mask_indices = torch.zeros(B, n_masked, dtype=torch.long, device=gene_ids.device)

        for b in range(B):
            # Decide whether to do pathway masking
            if (self.pathway_gene_sets is not None and
                    len(self.pathway_gene_sets) > 0 and
                    torch.rand(1).item() < self.pathway_mask_prob):
                # Pick a random pathway and mask its genes
                pw_names = list(self.pathway_gene_sets.keys())
                pw_name = pw_names[torch.randint(len(pw_names), (1,)).item()]
                pw_genes = self.pathway_gene_sets[pw_name]
                # Pathway genes (1-indexed positions)
                pw_positions = torch.tensor(pw_genes, device=gene_ids.device)
                if len(pw_positions) >= n_masked:
                    perm = torch.randperm(len(pw_positions))[:n_masked]
                    mask_indices[b] = pw_positions[perm]
                else:
                    # Fill with pathway genes + random others
                    remaining = n_masked - len(pw_positions)
                    all_positions = torch.arange(1, n_genes + 1, device=gene_ids.device)
                    # Exclude pathway genes from random selection
                    pw_set = set(pw_genes)
                    other = torch.tensor(
                        [p.item() for p in all_positions if p.item() not in pw_set],
                        device=gene_ids.device,
                    )
                    if len(other) >= remaining:
                        extra = other[torch.randperm(len(other))[:remaining]]
                    else:
                        extra = other
                        remaining = len(extra)
                    combined = torch.cat([pw_positions, extra[:remaining]])
                    if len(combined) < n_masked:
                        # Pad with random
                        pad = torch.randperm(n_genes, device=gene_ids.device)[:n_masked - len(combined)] + 1
                        combined = torch.cat([combined, pad])
                    mask_indices[b] = combined[:n_masked]
            else:
                perm = torch.randperm(n_genes, device=gene_ids.device)[:n_masked]
                mask_indices[b] = perm + 1  # +1 for [CLS] offset

        # Gather true values at masked positions
        true_values = torch.gather(values, 1, mask_indices)

        # Zero out masked positions in values
        masked_values = values.clone()
        masked_values.scatter_(1, mask_indices, 0.0)

        return masked_values, mask_indices, true_values

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        mag_bins: Optional[torch.Tensor] = None,
        presence: Optional[torch.Tensor] = None,
        pathway_scores: Optional[torch.Tensor] = None,
        subtypes: Optional[torch.Tensor] = None,
        mutation_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward with all objectives. Returns dict of losses + cls_emb."""

        # Create mask on values
        masked_values, mask_indices, true_values = self.create_mask(gene_ids, values)

        # Encode
        cls_emb, all_embs = self.encoder(
            gene_ids, masked_values, mag_bins=mag_bins, presence=presence
        )

        losses = {}

        # MGP
        if self.mgp_head is not None:
            losses["mgp"] = self.mgp_head(all_embs, mask_indices, true_values)
        else:
            losses["mgp"] = torch.tensor(0.0, device=cls_emb.device)

        # PAP
        if self.pap_head is not None and pathway_scores is not None:
            losses["pap"] = self.pap_head(cls_emb, pathway_scores)
        else:
            losses["pap"] = torch.tensor(0.0, device=cls_emb.device)

        # SUB
        if self.sub_head is not None and subtypes is not None:
            valid = subtypes >= 0
            if valid.sum() > 0:
                losses["sub"] = self.sub_head(cls_emb[valid], subtypes[valid])
            else:
                losses["sub"] = torch.tensor(0.0, device=cls_emb.device)
        else:
            losses["sub"] = torch.tensor(0.0, device=cls_emb.device)

        # MUT
        if self.mut_head is not None and mutation_labels is not None:
            losses["mut"] = self.mut_head(cls_emb, mutation_labels)
        else:
            losses["mut"] = torch.tensor(0.0, device=cls_emb.device)

        # DAV
        if self.dav_head is not None and domain_labels is not None:
            losses["dav"] = self.dav_head(cls_emb, domain_labels)
        else:
            losses["dav"] = torch.tensor(0.0, device=cls_emb.device)

        # Total weighted loss
        total = torch.tensor(0.0, device=cls_emb.device)
        for key in ["mgp", "pap", "sub", "mut", "dav"]:
            total = total + self.weights[key] * losses[key]
        losses["total"] = total
        losses["cls_emb"] = cls_emb

        return losses
