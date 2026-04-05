"""
Foundation Model Pretraining
============================
Self-supervised pretraining on ALL expression datasets using
Perceiver-style gene expression encoder.

Settings (CPU-feasible):
    - 2,000-gene universe, Perceiver with 16 latent tokens
    - d_model=128, n_heads=4, n_layers=2
    - batch_size=32, lr=1e-4, AdamW
    - 30 epochs, 15% gene masking
    - Three objectives: MGP + PAP + SUB
"""

from __future__ import annotations

import os, sys, time, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[3]  # inverex-mvp
sys.path.insert(0, str(ROOT / "src"))

from models.foundation.gene_universe import build_gene_universe, discover_all_datasets
from models.foundation.expression_tokenizer import ExpressionTokenizer
from models.foundation.expression_encoder import ExpressionEncoder, EncoderConfig
from models.foundation.pretrain_objectives import (
    FoundationPretrainModel,
    compute_pathway_scores_simple,
    infer_pam50_subtype,
)

logger = logging.getLogger(__name__)
RESULTS = ROOT / "results" / "foundation"


# ---------------------------------------------------------------------------
# Dataset — pre-tokenizes everything into tensors for speed
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """In-memory dataset of pre-tokenized expression + auxiliary labels."""

    def __init__(
        self,
        tokenizer: ExpressionTokenizer,
        expr_dfs: list[pd.DataFrame],
        dataset_names: list[str],
        gene_list: list[str],
    ):
        self.gene_list = gene_list
        n = tokenizer.n_genes

        logger.info("Building pretraining dataset from %d sources …", len(expr_dfs))

        # Collect all tokens and labels
        all_gene_ids = []
        all_ranks = []
        all_mag_bins = []
        all_pathway = []
        all_subtypes = []

        # First pass: compute pathway scores and subtypes per dataset
        pathway_arrays = []
        subtype_arrays = []
        pw_dim = 0
        for name, df in zip(dataset_names, expr_dfs):
            pw, pw_names = compute_pathway_scores_simple(df, gene_list)
            if pw is not None:
                pathway_arrays.append(pw)
                pw_dim = max(pw_dim, pw.shape[1])
            else:
                pathway_arrays.append(None)

            st = infer_pam50_subtype(df)
            subtype_arrays.append(st)

        self.n_pathways = pw_dim
        logger.info("Pathway dimension: %d", pw_dim)

        # Second pass: tokenize (vectorized) and collect
        for i, (name, df) in enumerate(zip(dataset_names, expr_dfs)):
            tokens = tokenizer.tokenize_batch_fast(df)
            all_gene_ids.append(tokens["gene_ids"])
            all_ranks.append(tokens["ranks"])
            all_mag_bins.append(tokens["mag_bins"])

            B = len(df)
            pw = pathway_arrays[i]
            if pw is not None:
                pw_padded = np.zeros((B, pw_dim), dtype=np.float32)
                pw_padded[:, :pw.shape[1]] = pw
                all_pathway.append(torch.tensor(pw_padded))
            else:
                all_pathway.append(torch.zeros(B, max(pw_dim, 1)))

            st = subtype_arrays[i]
            if st is not None:
                all_subtypes.append(torch.tensor(st, dtype=torch.long))
            else:
                all_subtypes.append(torch.full((B,), -1, dtype=torch.long))

            logger.info("  %s: %d samples tokenized", name, B)

        # Stack everything
        self.gene_ids = torch.cat(all_gene_ids, dim=0)
        self.ranks = torch.cat(all_ranks, dim=0)
        self.mag_bins = torch.cat(all_mag_bins, dim=0)
        self.pathway_scores = torch.cat(all_pathway, dim=0)
        self.subtypes = torch.cat(all_subtypes, dim=0)

        logger.info("Total pretraining samples: %d", len(self))

    def __len__(self):
        return self.gene_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "gene_ids": self.gene_ids[idx],
            "ranks": self.ranks[idx],
            "mag_bins": self.mag_bins[idx],
            "pathway_scores": self.pathway_scores[idx],
            "subtypes": self.subtypes[idx],
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pretrain(
    n_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    mask_ratio: float = 0.15,
    save_model: bool = True,
) -> tuple[FoundationPretrainModel, ExpressionTokenizer, list[str], dict[str, int]]:
    """
    Full pretraining pipeline:
      1. Build gene universe
      2. Load all data + tokenize
      3. Train Perceiver-style model
      4. Save artifacts
    """
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Step 1: Gene universe
    logger.info("=" * 60)
    logger.info("STEP 1: Building gene universe")
    logger.info("=" * 60)
    gene_list, gene2idx = build_gene_universe()
    n_genes = len(gene_list)
    logger.info("Gene universe: %d genes", n_genes)

    # Step 2: Load all expression data
    logger.info("=" * 60)
    logger.info("STEP 2: Loading expression data")
    logger.info("=" * 60)
    datasets = discover_all_datasets()
    expr_dfs = []
    dataset_names = []
    gene_set = set(gene_list)
    for name, path in datasets:
        try:
            df = pd.read_parquet(path)
            gene_cols = [c for c in df.columns if c in gene_set]
            if len(gene_cols) < 50:
                logger.warning("  %s: only %d gene overlap, skipping", name, len(gene_cols))
                continue
            expr_dfs.append(df)
            dataset_names.append(name)
            logger.info("  %s: %d samples, %d/%d genes", name, len(df), len(gene_cols), n_genes)
        except Exception as e:
            logger.warning("  Failed to load %s: %s", name, e)

    # Step 3: Fit tokenizer
    logger.info("=" * 60)
    logger.info("STEP 3: Fitting tokenizer statistics")
    logger.info("=" * 60)
    tokenizer = ExpressionTokenizer(gene_list, gene2idx)
    tokenizer.fit_stats(expr_dfs)
    tokenizer.save_stats()

    # Step 4: Build dataset
    logger.info("=" * 60)
    logger.info("STEP 4: Building pretraining dataset")
    logger.info("=" * 60)
    dataset = PretrainDataset(tokenizer, expr_dfs, dataset_names, gene_list)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    # Step 5: Build model
    logger.info("=" * 60)
    logger.info("STEP 5: Building Perceiver-style encoder")
    logger.info("=" * 60)
    cfg = EncoderConfig(
        n_genes=n_genes, d_model=128, n_heads=4, n_layers=2,
        n_latents=16, dropout=0.1,
    )
    encoder = ExpressionEncoder(cfg)
    model = FoundationPretrainModel(
        encoder, n_pathways=dataset.n_pathways, mask_ratio=mask_ratio
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")
    logger.info("Architecture: Perceiver with %d latent tokens", cfg.n_latents)

    # Step 6: Train
    logger.info("=" * 60)
    logger.info("STEP 6: Pretraining (%d epochs, %d batches/epoch)",
               n_epochs, len(dataloader))
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = []
    model.train()
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_losses = {"mgp": 0.0, "pap": 0.0, "sub": 0.0, "total": 0.0}
        n_batches = 0

        for batch in dataloader:
            gene_ids = batch["gene_ids"]
            ranks = batch["ranks"]
            mag_bins = batch["mag_bins"]
            pathway_scores = batch["pathway_scores"]
            subtypes = batch["subtypes"]

            # Guard against NaN in inputs
            if torch.isnan(ranks).any() or torch.isnan(mag_bins.float()).any():
                continue

            out = model(
                gene_ids, ranks, mag_bins,
                pathway_scores=pathway_scores if dataset.n_pathways > 0 else None,
                subtypes=subtypes,
            )

            loss = out["total"]

            # Skip NaN/Inf losses
            if not torch.isfinite(loss):
                logger.warning("  NaN/Inf loss at batch %d, skipping", n_batches)
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            # Check for NaN gradients
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    break
            if has_nan_grad:
                logger.warning("  NaN grad at batch %d, skipping", n_batches)
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in ["mgp", "pap", "sub", "total"]:
                epoch_losses[k] += out[k].item()
            n_batches += 1

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %2d/%d | total=%.4f mgp=%.4f pap=%.4f sub=%.4f | %.0fs elapsed",
            epoch, n_epochs,
            epoch_losses["total"], epoch_losses["mgp"],
            epoch_losses["pap"], epoch_losses["sub"],
            elapsed,
        )

        history.append({
            "epoch": epoch,
            "loss_total": epoch_losses["total"],
            "loss_mgp": epoch_losses["mgp"],
            "loss_pap": epoch_losses["pap"],
            "loss_sub": epoch_losses["sub"],
            "elapsed_s": round(elapsed, 1),
        })

    # Step 7: Save
    logger.info("=" * 60)
    logger.info("STEP 7: Saving artifacts")
    logger.info("=" * 60)

    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS / "pretrain_history.csv", index=False)
    logger.info("Saved pretrain_history.csv")

    if save_model:
        torch.save(model.state_dict(), RESULTS / "pretrain_model.pt")
        torch.save(encoder.state_dict(), RESULTS / "encoder.pt")
        logger.info("Saved model checkpoints")

        config = {
            "n_genes": n_genes,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "n_latents": cfg.n_latents,
            "n_pathways": dataset.n_pathways,
            "n_params": n_params,
            "n_samples": len(dataset),
            "n_datasets": len(dataset_names),
            "dataset_names": dataset_names,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "mask_ratio": mask_ratio,
        }
        with open(RESULTS / "pretrain_config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Saved pretrain_config.json")

    total_time = time.time() - t0
    logger.info("Pretraining complete in %.1f minutes", total_time / 60)

    return model, tokenizer, gene_list, gene2idx


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS / "pretrain.log"),
        ],
    )
    pretrain()
