"""
Foundation Model Pretraining (Full)
===============================================
Multi-objective pretraining with domain adversarial training on ALL expression data.

Objectives (modular, toggled via config):
    A. Masked gene prediction (reconstruct held-out gene values)
    B. Pathway activity prediction (predict mean-rank Hallmark from [CLS])
    C. Subtype prediction (predict inferred PAM50 from [CLS])
    D. Mutation-proxy prediction (TP53/PIK3CA/ERBB2 where available)
    E. Domain adversarial (gradient reversal against dataset identity)

Settings:
    - 5K-8K gene universe (>=40% prevalence)
    - medium: d_model=256, n_layers=4, n_heads=4 (~2M params)
    - batch_size=32, lr=1e-4, AdamW, cosine schedule with warmup
    - 15% gene masking, 20% pathway masking probability
    - Checkpoint every 10 epochs
"""

from __future__ import annotations

import os, sys, time, logging, json, math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models.foundation_full.gene_universe import (
    build_gene_universe,
    discover_all_datasets,
)
from models.foundation_full.expression_tokenizer import ExpressionTokenizer
from models.foundation_full.expression_encoder import (
    ExpressionEncoder,
    EncoderConfig,
    FoundationPretrainModel,
)

logger = logging.getLogger(__name__)

RESULTS = ROOT / "results" / "foundation_full"
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"


# ---------------------------------------------------------------------------
# Hallmark pathway gene sets (simple approach)
# ---------------------------------------------------------------------------

def load_hallmark_gene_sets() -> dict[str, list[str]]:
    """Load MSigDB Hallmark 2020 gene sets via gseapy."""
    try:
        import gseapy
        try:
            lib = gseapy.get_library("MSigDB_Hallmark_2020", organism="Human")
        except Exception:
            lib = gseapy.get_library("MSigDB_Hallmark_2020")
        return lib
    except Exception as e:
        logger.warning("Could not load Hallmark gene sets: %s", e)
        return {}


def compute_pathway_scores(
    expr_df: pd.DataFrame, gene_list: list[str], hallmark: dict
) -> np.ndarray | None:
    """Compute mean-rank pathway scores. Returns [N, n_pathways] or None."""
    available = [g for g in gene_list if g in expr_df.columns]
    if len(available) < 100:
        return None

    sub = expr_df[available].copy()
    ranked = sub.rank(axis=1, pct=True)

    pathway_names = []
    scores_list = []
    for pw_name, pw_genes in hallmark.items():
        overlap = [g for g in pw_genes if g in available]
        if len(overlap) >= 3:
            pw_score = ranked[overlap].mean(axis=1).values.astype(np.float32)
            scores_list.append(pw_score)
            pathway_names.append(pw_name)

    if not scores_list:
        return None

    scores = np.column_stack(scores_list)
    return scores


def infer_pam50_subtype(expr_df: pd.DataFrame) -> np.ndarray:
    """
    Infer simplified PAM50-like subtype from ESR1, ERBB2, MKI67.
    0=LumA, 1=LumB, 2=HER2+, 3=Basal, -1=unknown
    """
    needed = ["ESR1", "ERBB2", "MKI67"]
    if not all(g in expr_df.columns for g in needed):
        return np.full(len(expr_df), -1, dtype=np.int64)

    esr1 = np.log2(expr_df["ESR1"].values.astype(float) + 1)
    erbb2 = np.log2(expr_df["ERBB2"].values.astype(float) + 1)
    mki67 = np.log2(expr_df["MKI67"].values.astype(float) + 1)

    esr1_med = np.median(esr1)
    erbb2_q75 = np.percentile(erbb2, 75)
    mki67_med = np.median(mki67)

    subtypes = np.zeros(len(expr_df), dtype=np.int64)
    for i in range(len(expr_df)):
        if erbb2[i] > erbb2_q75:
            subtypes[i] = 2
        elif esr1[i] > esr1_med:
            subtypes[i] = 1 if mki67[i] > mki67_med else 0
        else:
            subtypes[i] = 3

    return subtypes


def load_mutation_labels(expr_df: pd.DataFrame, dataset_name: str) -> np.ndarray:
    """
    Load TP53/PIK3CA/ERBB2 mutation status. Returns [N, 3] with -1 for unknown.
    Only available for TCGA_BRCA.
    """
    labels = np.full((len(expr_df), 3), -1, dtype=np.float32)

    if dataset_name != "TCGA_BRCA":
        return labels

    mut_path = DATA_CACHE / "tcga_brca_mutations.parquet"
    if not mut_path.exists():
        return labels

    mut = pd.read_parquet(mut_path)
    genes_of_interest = ["TP53", "PIK3CA", "ERBB2"]

    # Build per-sample mutation dict
    mutated_samples = {}
    for _, row in mut.iterrows():
        sample = row["sample"]
        gene = row["gene"]
        if gene in genes_of_interest:
            if sample not in mutated_samples:
                mutated_samples[sample] = set()
            mutated_samples[sample].add(gene)

    # TCGA samples in expr_df
    for i, sample_id in enumerate(expr_df.index):
        # All TCGA samples: default to 0 (wild-type) if we have mutation data
        for j, gene in enumerate(genes_of_interest):
            labels[i, j] = 0  # assume WT
        if sample_id in mutated_samples:
            for j, gene in enumerate(genes_of_interest):
                if gene in mutated_samples[sample_id]:
                    labels[i, j] = 1

    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """In-memory dataset of pre-tokenized expression + auxiliary labels."""

    def __init__(
        self,
        tokenizer: ExpressionTokenizer,
        expr_dfs: list[pd.DataFrame],
        dataset_names: list[str],
        gene_list: list[str],
        hallmark: dict,
    ):
        self.gene_list = gene_list

        logger.info("Building pretraining dataset from %d sources ...", len(expr_dfs))

        # Tokenize all datasets
        all_tokens = []
        all_pathway_scores = []
        all_subtypes = []
        all_mutations = []
        all_domain_labels = []
        self.n_pathways = 0
        self.n_domains = len(expr_dfs)
        self.dataset_names = dataset_names

        for di, (name, df) in enumerate(zip(dataset_names, expr_dfs)):
            n = len(df)
            logger.info("  Tokenizing %s (%d samples) ...", name, n)

            # Tokenize
            tokens = tokenizer.tokenize_dataframe(df)
            all_tokens.append(tokens)

            # Pathway scores
            pw = compute_pathway_scores(df, gene_list, hallmark)
            if pw is not None:
                all_pathway_scores.append(pw)
                self.n_pathways = max(self.n_pathways, pw.shape[1])
            else:
                all_pathway_scores.append(None)

            # Subtypes
            all_subtypes.append(infer_pam50_subtype(df))

            # Mutations
            all_mutations.append(load_mutation_labels(df, name))

            # Domain labels
            all_domain_labels.append(np.full(n, di, dtype=np.int64))

        # Pad pathway scores to consistent dimension
        for i in range(len(all_pathway_scores)):
            n = len(expr_dfs[i])
            if all_pathway_scores[i] is None:
                all_pathway_scores[i] = np.zeros((n, self.n_pathways), dtype=np.float32)
            elif all_pathway_scores[i].shape[1] < self.n_pathways:
                padded = np.zeros((n, self.n_pathways), dtype=np.float32)
                padded[:, :all_pathway_scores[i].shape[1]] = all_pathway_scores[i]
                all_pathway_scores[i] = padded

        # Concatenate everything
        self.gene_ids = torch.cat([t["gene_ids"] for t in all_tokens], dim=0)
        self.values = torch.cat([t["values"] for t in all_tokens], dim=0)
        self.mag_bins = torch.cat([t["mag_bins"] for t in all_tokens], dim=0)
        self.presence = torch.cat([t["presence"] for t in all_tokens], dim=0)
        self.expr_raw = torch.cat([t["expr_raw"] for t in all_tokens], dim=0)

        self.pathway_scores = torch.tensor(
            np.concatenate(all_pathway_scores, axis=0), dtype=torch.float32
        )
        self.subtypes = torch.tensor(
            np.concatenate(all_subtypes, axis=0), dtype=torch.long
        )
        self.mutations = torch.tensor(
            np.concatenate(all_mutations, axis=0), dtype=torch.float32
        )
        self.domain_labels = torch.tensor(
            np.concatenate(all_domain_labels, axis=0), dtype=torch.long
        )

        self.n_samples = self.gene_ids.shape[0]
        logger.info(
            "Total pretraining samples: %d | pathways: %d | domains: %d",
            self.n_samples, self.n_pathways, self.n_domains,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "gene_ids": self.gene_ids[idx],
            "values": self.values[idx],
            "mag_bins": self.mag_bins[idx],
            "presence": self.presence[idx],
            "expr_raw": self.expr_raw[idx],
            "pathway_scores": self.pathway_scores[idx],
            "subtypes": self.subtypes[idx],
            "mutations": self.mutations[idx],
            "domain_labels": self.domain_labels[idx],
        }


def collate_fn(batch):
    """Stack all fields."""
    return {
        key: torch.stack([s[key] for s in batch])
        for key in batch[0].keys()
    }


# ---------------------------------------------------------------------------
# Cosine schedule with warmup
# ---------------------------------------------------------------------------

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr_scale = self.step_count / max(self.warmup_steps, 1)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lr_scale

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Pathway gene position mapping
# ---------------------------------------------------------------------------

def build_pathway_gene_positions(
    hallmark: dict, gene_list: list[str]
) -> dict[str, list[int]]:
    """Map pathway names to 1-indexed gene positions in the gene universe."""
    gene_to_pos = {g: i + 1 for i, g in enumerate(gene_list)}
    pw_positions = {}
    for pw_name, pw_genes in hallmark.items():
        positions = [gene_to_pos[g] for g in pw_genes if g in gene_to_pos]
        if len(positions) >= 3:
            pw_positions[pw_name] = positions
    return pw_positions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def pretrain(
    model_size: str = "medium",
    encoding: str = "hybrid",
    objectives_config: Optional[dict] = None,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    mask_ratio: float = 0.15,
    pathway_mask_prob: float = 0.20,
    save_tag: str = "",
    max_time_hours: float = 2.0,
    resume_from: Optional[Path] = None,
) -> tuple:
    """
    Run full pretraining pipeline.

    Returns (model, tokenizer, gene_list, gene2idx, history_df)
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"_{save_tag}" if save_tag else ""

    # Step 1: Gene universe
    logger.info("=" * 70)
    logger.info("STEP 1: Building FULL gene universe (>=40%% prevalence)")
    logger.info("=" * 70)
    gene_list, gene2idx, universe_info = build_gene_universe()
    n_genes = len(gene_list)
    logger.info("Gene universe: %d genes from %d datasets", n_genes, universe_info["n_datasets"])

    # Step 2: Load data
    logger.info("=" * 70)
    logger.info("STEP 2: Loading ALL expression data")
    logger.info("=" * 70)
    datasets = discover_all_datasets()
    expr_dfs = []
    dataset_names = []
    for name, path in datasets:
        try:
            df = pd.read_parquet(path)
            gene_cols = [c for c in df.columns if c in set(gene_list)]
            if len(gene_cols) < 50:
                logger.warning("  %s: only %d genes overlap, skipping", name, len(gene_cols))
                continue
            expr_dfs.append(df)
            dataset_names.append(name)
            logger.info("  Loaded %s: %d samples, %d/%d genes overlap",
                       name, len(df), len(gene_cols), n_genes)
        except Exception as e:
            logger.warning("  Failed to load %s: %s", name, e)

    total_samples = sum(len(df) for df in expr_dfs)
    logger.info("Total samples across %d datasets: %d", len(expr_dfs), total_samples)

    # Step 3: Load Hallmark gene sets
    logger.info("=" * 70)
    logger.info("STEP 3: Loading pathway gene sets")
    logger.info("=" * 70)
    hallmark = load_hallmark_gene_sets()
    logger.info("Loaded %d Hallmark pathways", len(hallmark))
    pw_positions = build_pathway_gene_positions(hallmark, gene_list)
    logger.info("Mapped %d pathways to gene positions", len(pw_positions))

    # Step 4: Fit tokenizer
    logger.info("=" * 70)
    logger.info("STEP 4: Fitting tokenizer (encoding=%s)", encoding)
    logger.info("=" * 70)
    tokenizer = ExpressionTokenizer(gene_list, gene2idx, encoding=encoding)
    tokenizer.fit_stats(expr_dfs)
    tokenizer.save_stats()

    # Step 5: Build dataset
    logger.info("=" * 70)
    logger.info("STEP 5: Building pretraining dataset")
    logger.info("=" * 70)
    dataset = PretrainDataset(tokenizer, expr_dfs, dataset_names, gene_list, hallmark)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # Step 6: Build model
    logger.info("=" * 70)
    logger.info("STEP 6: Building %s model (encoding=%s)", model_size, encoding)
    logger.info("=" * 70)

    cfg = EncoderConfig.from_preset(model_size, n_genes=n_genes, encoding=encoding)
    encoder = ExpressionEncoder(cfg)

    # Default objectives config
    if objectives_config is None:
        objectives_config = {
            "mgp": True, "mgp_weight": 1.0,
            "pap": True, "pap_weight": 0.5,
            "pap_n_pathways": dataset.n_pathways,
            "sub": True, "sub_weight": 0.5,
            "mut": True, "mut_weight": 0.3,
            "dav": True, "dav_weight": 0.2,
            "dav_n_domains": dataset.n_domains,
        }
    else:
        # Ensure n_pathways and n_domains are set
        objectives_config.setdefault("pap_n_pathways", dataset.n_pathways)
        objectives_config.setdefault("dav_n_domains", dataset.n_domains)

    model = FoundationPretrainModel(
        encoder,
        objectives=objectives_config,
        mask_ratio=mask_ratio,
        pathway_mask_prob=pathway_mask_prob,
        pathway_gene_sets=pw_positions if pathway_mask_prob > 0 else None,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")
    logger.info("Encoder parameters: %s", f"{encoder.count_parameters():,}")

    # Step 7: Setup optimizer
    logger.info("=" * 70)
    logger.info("STEP 7: Setting up training")
    logger.info("=" * 70)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    for pg in optimizer.param_groups:
        pg["initial_lr"] = lr

    steps_per_epoch = len(dataloader)
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = min(steps_per_epoch * 3, total_steps // 10)  # 3 epochs or 10%
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    logger.info("Steps/epoch: %d | Total steps: %d | Warmup: %d",
               steps_per_epoch, total_steps, warmup_steps)

    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_from is not None and resume_from.exists():
        logger.info("Resuming from checkpoint: %s", resume_from)
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        scheduler.step_count = ckpt.get("step_count", 0)
        logger.info("Resumed from epoch %d", start_epoch - 1)

    # Step 8: Train
    logger.info("=" * 70)
    logger.info("STEP 8: Pretraining (%d epochs, starting from %d)", n_epochs, start_epoch)
    logger.info("=" * 70)

    history = []
    model.train()
    t0 = time.time()
    max_time_s = max_time_hours * 3600

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_losses = {k: 0.0 for k in ["mgp", "pap", "sub", "mut", "dav", "total"]}
        n_batches = 0

        # Ramp up domain adversarial alpha over training
        if model.dav_head is not None:
            progress = (epoch - 1) / max(n_epochs - 1, 1)
            alpha = 2.0 / (1.0 + math.exp(-10 * progress)) - 1.0  # 0 -> 1
            model.dav_head.set_alpha(alpha)

        for batch in dataloader:
            gene_ids = batch["gene_ids"]
            values = batch["values"]
            mag_bins = batch["mag_bins"]
            presence = batch["presence"]
            pathway_scores = batch["pathway_scores"]
            subtypes = batch["subtypes"]
            mutations = batch["mutations"]
            domain_labels = batch["domain_labels"]

            out = model(
                gene_ids, values,
                mag_bins=mag_bins if cfg.use_mag_embedding else None,
                presence=presence if cfg.use_presence_flag else None,
                pathway_scores=pathway_scores if dataset.n_pathways > 0 else None,
                subtypes=subtypes,
                mutation_labels=mutations,
                domain_labels=domain_labels,
            )

            loss = out["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            for k in ["mgp", "pap", "sub", "mut", "dav", "total"]:
                epoch_losses[k] += out[k].item()
            n_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        elapsed = time.time() - t0
        current_lr = scheduler.get_lr()
        logger.info(
            "Epoch %3d/%d | total=%.4f mgp=%.4f pap=%.4f sub=%.4f mut=%.4f dav=%.4f | lr=%.2e | %.0fs",
            epoch, n_epochs,
            epoch_losses["total"], epoch_losses["mgp"],
            epoch_losses["pap"], epoch_losses["sub"],
            epoch_losses["mut"], epoch_losses["dav"],
            current_lr, elapsed,
        )

        history.append({
            "epoch": epoch,
            **{f"loss_{k}": epoch_losses[k] for k in ["total", "mgp", "pap", "sub", "mut", "dav"]},
            "lr": current_lr,
            "elapsed_s": elapsed,
        })

        # Checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs:
            ckpt_path = RESULTS / f"checkpoint{tag}_ep{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step_count": scheduler.step_count,
                "config": {
                    "model_size": model_size,
                    "encoding": encoding,
                    "n_genes": n_genes,
                    "objectives": objectives_config,
                },
            }, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        # Time limit check
        if elapsed > max_time_s:
            logger.warning(
                "Time limit (%.1f hours) reached at epoch %d. Stopping early.",
                max_time_hours, epoch,
            )
            break

    # Save final artifacts
    logger.info("=" * 70)
    logger.info("STEP 9: Saving artifacts")
    logger.info("=" * 70)

    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS / f"pretrain_history{tag}.csv", index=False)

    # Save encoder weights
    torch.save(encoder.state_dict(), RESULTS / f"encoder{tag}.pt")
    torch.save(model.state_dict(), RESULTS / f"pretrain_model{tag}.pt")

    # Save config
    config_dict = {
        "model_size": model_size,
        "encoding": encoding,
        "n_genes": n_genes,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "n_mag_bins": cfg.n_mag_bins,
        "use_mag_embedding": cfg.use_mag_embedding,
        "use_presence_flag": cfg.use_presence_flag,
        "n_pathways": dataset.n_pathways,
        "n_domains": dataset.n_domains,
        "n_params": n_params,
        "n_encoder_params": encoder.count_parameters(),
        "n_samples": dataset.n_samples,
        "n_datasets": len(dataset_names),
        "dataset_names": dataset_names,
        "n_epochs": epoch,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "mask_ratio": mask_ratio,
        "pathway_mask_prob": pathway_mask_prob,
        "objectives": objectives_config,
        "training_time_s": elapsed,
    }
    with open(RESULTS / f"pretrain_config{tag}.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    total_time = time.time() - t0
    logger.info("Pretraining complete in %.1f minutes (%d epochs)", total_time / 60, epoch)

    return model, tokenizer, gene_list, gene2idx, history_df


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS / "pretrain_full.log"),
        ],
    )
    pretrain(model_size="medium", encoding="hybrid", n_epochs=50)
