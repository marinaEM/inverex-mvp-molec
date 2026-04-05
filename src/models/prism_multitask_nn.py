"""
Multi-task neural network for drug-response prediction on PRISM pan-cancer data.

Trained on ~150k samples across ~375 drugs. Groups drugs by mechanism of action
(MOA) into pathway groups, then trains a multi-task architecture:

  Shared encoder:   Input -> Linear(512) -> ReLU -> Dropout(0.3)
                           -> Linear(256) -> ReLU -> Dropout(0.2)
  Pathway heads:    per pathway_group -> Linear(256, 128) -> ReLU -> Dropout(0.15)
  Drug output:      per drug -> Linear(128, 1)

Also trains a single-task NN baseline for comparison.

Entry point: ``pixi run python -m src.models.prism_multitask_nn``
"""

import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Avoid OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CACHE, DATA_RAW, RESULTS, RANDOM_SEED

logger = logging.getLogger(__name__)

PRISM_DIR = DATA_RAW / "prism"
RESULTS_DIR = RESULTS / "prism_multitask"


# ======================================================================
# 1. Pathway group construction from PRISM MOA annotations
# ======================================================================

# Keyword-based MOA -> pathway group mapping
MOA_KEYWORD_MAP = {
    "kinase_inhibitor": [
        "kinase inhibitor", "tyrosine kinase", "CDK inhibitor",
        "EGFR inhibitor", "VEGFR inhibitor", "BRAF inhibitor",
        "MEK inhibitor", "JAK inhibitor", "FLT3 inhibitor",
        "ALK inhibitor", "PI3K inhibitor", "mTOR inhibitor",
        "AKT inhibitor", "SRC inhibitor", "ABL inhibitor",
        "aurora kinase", "polo-like kinase", "RAF inhibitor",
        "ERK inhibitor", "MAP kinase", "receptor tyrosine",
    ],
    "dna_damage": [
        "topoisomerase", "DNA alkylating", "DNA synthesis",
        "DNA damage", "DNA intercalat", "PARP inhibitor",
        "ATR inhibitor", "ATM inhibitor", "CHK inhibitor",
        "Wee1 inhibitor", "nucleoside analog", "antimetabolite",
        "thymidylate synthase",
    ],
    "hormone": [
        "estrogen", "androgen", "glucocorticoid", "progesterone",
        "hormone", "aromatase", "SERM", "SERD",
        "mineralocorticoid", "thyroid",
    ],
    "epigenetic": [
        "HDAC inhibitor", "histone deacetylase", "BET inhibitor",
        "bromodomain", "EZH2 inhibitor", "DNMT inhibitor",
        "methyltransferase", "demethylase", "histone",
        "sirtuin", "epigenetic",
    ],
    "cytotoxic": [
        "tubulin", "microtubule", "taxane", "vinca",
        "proteasome", "protein synthesis", "ribosom",
        "mitotic", "kinesin", "anti-mitotic", "cytotoxic",
    ],
    "immune": [
        "immunostimulant", "immunosuppressant", "immune",
        "TNF", "interleukin", "interferon", "NFkB",
        "NF-kB", "NF-kappaB", "toll-like",
        "PD-1", "PD-L1", "CTLA-4",
    ],
    "apoptosis": [
        "BCL", "apoptosis", "IAP", "caspase",
        "survivin", "MCL1", "death receptor",
    ],
    "metabolism": [
        "HMG-CoA", "statin", "AMPK", "fatty acid",
        "glycolysis", "glutaminase", "IDH", "metabol",
        "lipase", "cholesterol", "oxidative phosphorylation",
    ],
    "signaling": [
        "Hedgehog", "Notch", "Wnt", "WNT",
        "TGF", "BMP", "SMAD", "beta-catenin",
        "smoothened", "gamma-secretase",
    ],
    "hsp_chaperone": [
        "HSP90", "HSP70", "heat shock", "chaperone",
    ],
}


def _normalize_drug_name(name: str) -> str:
    """Normalize drug name for matching."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    for ch in ["-", " ", "(", ")", ".", ","]:
        s = s.replace(ch, "")
    return s


def build_pathway_groups(
    meta: pd.DataFrame,
    prism_dir: Path = PRISM_DIR,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Build drug -> pathway_group mapping from PRISM treatment info MOA.

    Returns dict with:
      - drug_to_group: {norm_drug_name: pathway_group}
      - group_stats: {group: count}
    """
    logger.info("Building pathway groups from PRISM MOA annotations...")

    # Load treatment info
    ti_path = prism_dir / "pancancer_treatment_info.parquet"
    if ti_path.exists():
        ti = pd.read_parquet(ti_path)
    else:
        logger.warning("No treatment info found, using default grouping")
        ti = pd.DataFrame()

    # Build drug -> MOA lookup from treatment info
    drug_moa = {}
    drug_target = {}
    if not ti.empty and "moa" in ti.columns:
        for _, row in ti.drop_duplicates("name").iterrows():
            name_norm = _normalize_drug_name(row.get("name", ""))
            if name_norm:
                moa = str(row.get("moa", "")).lower() if pd.notna(row.get("moa")) else ""
                target = str(row.get("target", "")).lower() if pd.notna(row.get("target")) else ""
                drug_moa[name_norm] = moa
                drug_target[name_norm] = target

    logger.info(f"  Drugs with MOA annotation: {sum(1 for v in drug_moa.values() if v)}")
    logger.info(f"  Drugs with target annotation: {sum(1 for v in drug_target.values() if v)}")

    # Map each drug to a pathway group
    drug_to_group = {}
    unique_drugs = meta["drug"].unique()

    for drug in unique_drugs:
        drug_norm = _normalize_drug_name(drug)
        moa = drug_moa.get(drug_norm, "")
        target = drug_target.get(drug_norm, "")
        combined = f"{moa} {target}".lower()

        assigned = False
        for group, keywords in MOA_KEYWORD_MAP.items():
            for kw in keywords:
                if kw.lower() in combined:
                    drug_to_group[drug] = group
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            drug_to_group[drug] = "other"

    # Count samples per group
    meta_groups = meta["drug"].map(drug_to_group)
    group_stats = meta_groups.value_counts().to_dict()

    logger.info(f"  Pathway groups ({len(set(drug_to_group.values()))}):")
    for grp, cnt in sorted(group_stats.items(), key=lambda x: -x[1]):
        n_drugs = sum(1 for v in drug_to_group.values() if v == grp)
        logger.info(f"    {grp:20s}: {cnt:6d} samples, {n_drugs:3d} drugs")

    result = {
        "drug_to_group": drug_to_group,
        "group_stats": group_stats,
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved to {save_path}")

    return result


# ======================================================================
# 2. PRISM Multi-Task Neural Network
# ======================================================================

class PRISMMultiTaskNN(nn.Module):
    """
    Multi-task NN with deeper shared encoder for PRISM pan-cancer data.

    Architecture:
      Shared encoder:   Input -> Linear(512) -> ReLU -> Dropout(0.3)
                               -> Linear(256) -> ReLU -> Dropout(0.2)
      Pathway heads:    per group -> Linear(256, 128) -> ReLU -> Dropout(0.15)
      Drug outputs:     per drug -> Linear(128, 1)
    """

    def __init__(
        self,
        input_dim: int,
        pathway_names: list,
        drug_names: list,
        drug_to_pathway: dict,
    ):
        super().__init__()
        self.pathway_names = list(pathway_names)
        self.drug_names = list(drug_names)
        self.drug_to_pathway = drug_to_pathway

        # Index mappings
        self.pathway_idx = {pw: i for i, pw in enumerate(self.pathway_names)}
        self.drug_idx = {d: i for i, d in enumerate(self.drug_names)}

        # Shared encoder (deeper than GDSC2 version)
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Pathway-specific intermediate layers
        self.pathway_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.15),
            )
            for _ in self.pathway_names
        ])

        # Drug-specific output heads
        self.drug_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in self.drug_names
        ])

        # Pre-compute drug -> pathway index
        self._drug_to_pw_idx = {}
        for d in self.drug_names:
            pw = drug_to_pathway.get(d, self.pathway_names[0])
            if pw not in self.pathway_idx:
                pw = self.pathway_names[0]
            self._drug_to_pw_idx[d] = self.pathway_idx[pw]

    def forward(
        self,
        x: torch.Tensor,
        drug_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass routing each sample through its pathway head."""
        h = self.shared_encoder(x)  # (batch, 256)

        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, device=x.device)

        # Group samples by pathway for efficiency
        pw_groups: dict = {}
        for i in range(batch_size):
            d_idx = drug_indices[i].item()
            d_name = self.drug_names[d_idx]
            pw_idx = self._drug_to_pw_idx.get(d_name, 0)
            pw_groups.setdefault(pw_idx, []).append(i)

        for pw_idx, sample_indices in pw_groups.items():
            idx_t = torch.tensor(sample_indices, device=x.device, dtype=torch.long)
            h_sub = h[idx_t]
            pw_out = self.pathway_layers[pw_idx](h_sub)  # (n, 128)

            # Each sample may have a different drug head
            for local_i, global_i in enumerate(sample_indices):
                d_idx = drug_indices[global_i].item()
                out = self.drug_heads[d_idx](pw_out[local_i:local_i + 1])
                outputs[global_i] = out.squeeze()

        return outputs

    def get_drug_embeddings(
        self,
        x: torch.Tensor,
        drug_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get 128-dim pathway-specific embeddings for each sample."""
        h = self.shared_encoder(x)

        batch_size = x.size(0)
        embeddings = torch.zeros(batch_size, 128, device=x.device)

        pw_groups: dict = {}
        for i in range(batch_size):
            d_idx = drug_indices[i].item()
            d_name = self.drug_names[d_idx]
            pw_idx = self._drug_to_pw_idx.get(d_name, 0)
            pw_groups.setdefault(pw_idx, []).append(i)

        for pw_idx, sample_indices in pw_groups.items():
            idx_t = torch.tensor(sample_indices, device=x.device, dtype=torch.long)
            h_sub = h[idx_t]
            pw_out = self.pathway_layers[pw_idx](h_sub)
            for local_i, global_i in enumerate(sample_indices):
                embeddings[global_i] = pw_out[local_i]

        return embeddings


# ======================================================================
# 3. Single-Task NN (for comparison)
# ======================================================================

class PRISMSingleTaskNN(nn.Module):
    """
    Single-task NN with same total capacity as multi-task.

    Architecture:
      Input -> Linear(512) -> ReLU -> Dropout(0.3)
            -> Linear(256) -> ReLU -> Dropout(0.2)
            -> Linear(128) -> ReLU -> Dropout(0.15)
            -> Linear(1)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ======================================================================
# 4. Training utilities
# ======================================================================

def train_multitask(
    X_train: np.ndarray,
    y_train: np.ndarray,
    drug_idx_train: np.ndarray,
    model: PRISMMultiTaskNN,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    drug_idx_val: Optional[np.ndarray] = None,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> tuple:
    """Train multi-task NN. Returns (model, training_log)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(drug_idx_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    has_val = X_val is not None
    if has_val:
        # For large val sets, use batched evaluation
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(drug_idx_val, dtype=torch.long),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    training_log = []

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_b, y_b, d_b in train_loader:
            X_b, y_b, d_b = X_b.to(device), y_b.to(device), d_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b, d_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validation (batched)
        val_loss = None
        if has_val:
            model.eval()
            val_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for X_vb, y_vb, d_vb in val_loader:
                    X_vb, y_vb, d_vb = X_vb.to(device), y_vb.to(device), d_vb.to(device)
                    pred_v = model(X_vb, d_vb)
                    val_sum += criterion(pred_v, y_vb).item() * len(y_vb)
                    val_n += len(y_vb)
            val_loss = val_sum / max(val_n, 1)

            monitor = val_loss
        else:
            monitor = avg_train

        elapsed = time.time() - t0
        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "elapsed_s": round(elapsed, 1),
        })

        if monitor < best_val_loss:
            best_val_loss = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epochs_no_improve >= patience:
            vl_str = f", val={val_loss:.4f}" if val_loss is not None else ""
            logger.info(
                f"  Epoch {epoch+1}/{max_epochs}: train={avg_train:.4f}{vl_str} "
                f"[{elapsed:.1f}s] (no_improve={epochs_no_improve})"
            )

        if epochs_no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (best={best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.cpu()
    return model, training_log


def train_singletask(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
    input_dim: Optional[int] = None,
) -> tuple:
    """Train single-task NN. Returns (model, training_log)."""
    if input_dim is None:
        input_dim = X_train.shape[1]
    model = PRISMSingleTaskNN(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    has_val = X_val is not None
    if has_val:
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    training_log = []

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        val_loss = None
        if has_val:
            model.eval()
            val_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for X_vb, y_vb in val_loader:
                    X_vb, y_vb = X_vb.to(device), y_vb.to(device)
                    pred_v = model(X_vb)
                    val_sum += criterion(pred_v, y_vb).item() * len(y_vb)
                    val_n += len(y_vb)
            val_loss = val_sum / max(val_n, 1)
            monitor = val_loss
        else:
            monitor = avg_train

        elapsed = time.time() - t0
        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "elapsed_s": round(elapsed, 1),
        })

        if monitor < best_val_loss:
            best_val_loss = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epochs_no_improve >= patience:
            vl_str = f", val={val_loss:.4f}" if val_loss is not None else ""
            logger.info(
                f"  Epoch {epoch+1}/{max_epochs}: train={avg_train:.4f}{vl_str} "
                f"[{elapsed:.1f}s] (no_improve={epochs_no_improve})"
            )

        if epochs_no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (best={best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.cpu()
    return model, training_log


# ======================================================================
# 5. Prediction + Evaluation
# ======================================================================

def predict_multitask(
    model: PRISMMultiTaskNN,
    X: np.ndarray,
    drug_indices: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Batched prediction for multi-task model."""
    model.eval()
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(drug_indices, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_b, d_b in loader:
            p = model(X_b, d_b)
            preds.append(p.numpy())
    return np.concatenate(preds)


def predict_singletask(
    model: PRISMSingleTaskNN,
    X: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Batched prediction for single-task model."""
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (X_b,) in loader:
            p = model(X_b)
            preds.append(p.numpy())
    return np.concatenate(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R2."""
    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "n_samples": len(y_true),
    }


# ======================================================================
# 6. Extract drug embeddings
# ======================================================================

def extract_drug_embeddings(
    model: PRISMMultiTaskNN,
    X: np.ndarray,
    drug_indices: np.ndarray,
    drug_names: list,
    meta: pd.DataFrame,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Extract 128-dim pathway-specific embeddings per drug.
    Average across all samples for each drug.
    """
    logger.info("Extracting drug embeddings...")
    model.eval()

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(drug_indices, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for X_b, d_b in loader:
            emb = model.get_drug_embeddings(X_b, d_b)
            all_embeddings.append(emb.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (n_samples, 128)

    # Average by drug
    unique_drugs = sorted(set(meta["drug"].values))
    drug_idx_map = {d: i for i, d in enumerate(drug_names)}

    rows = []
    for drug in unique_drugs:
        mask = meta["drug"].values == drug
        if mask.sum() > 0:
            avg_emb = all_embeddings[mask].mean(axis=0)
            row = {"drug": drug}
            for j in range(128):
                row[f"emb_{j:03d}"] = float(avg_emb[j])
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"  Drug embeddings: {len(df)} drugs x 128 dims")
    return df


# ======================================================================
# 7. Main experiment
# ======================================================================

def run_prism_multitask_experiment():
    """
    Full pipeline:
      B1. Load PRISM pan-cancer data
      B2. Build pathway groups from MOA
      B3. Train multi-task NN
      B4. Train single-task NN
      B5. Compare
      B6. Extract drug embeddings
      B7. Save everything
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PRISM PAN-CANCER MULTI-TASK NN EXPERIMENT")
    logger.info("=" * 70)

    # ==================================================================
    # B1. Load PRISM training data
    # ==================================================================
    logger.info("\n--- B1: Loading PRISM pan-cancer data ---")

    X_path = DATA_CACHE / "prism_pancancer_X.parquet"
    y_path = DATA_CACHE / "prism_pancancer_y.parquet"
    meta_path = DATA_CACHE / "prism_pancancer_meta.parquet"

    if not (X_path.exists() and y_path.exists() and meta_path.exists()):
        logger.info("Cached data not found. Building from raw PRISM data...")
        from src.data_ingestion.prism_pancancer import build_prism_training_matrix
        X_df, y_series, meta = build_prism_training_matrix()
    else:
        logger.info("Loading cached PRISM data...")
        X_df = pd.read_parquet(X_path)
        y_series = pd.read_parquet(y_path).squeeze()
        meta = pd.read_parquet(meta_path)

    logger.info(f"  X: {X_df.shape}, y: {len(y_series)}, meta: {meta.shape}")
    logger.info(f"  Drugs: {meta['drug'].nunique()}, Cancer types: {meta['cancer_type'].nunique()}")
    logger.info(f"  Target stats: mean={y_series.mean():.2f}, std={y_series.std():.2f}, "
                f"range=[{y_series.min():.2f}, {y_series.max():.2f}]")

    X = X_df.values.astype(np.float32)
    y = y_series.values.astype(np.float32)

    # ==================================================================
    # B2. Build pathway groups
    # ==================================================================
    logger.info("\n--- B2: Building pathway groups from PRISM MOA ---")

    pw_save = DATA_CACHE / "prism_pathway_groups.json"
    pw_info = build_pathway_groups(meta, prism_dir=PRISM_DIR, save_path=pw_save)
    drug_to_group = pw_info["drug_to_group"]

    # ==================================================================
    # Prepare indices
    # ==================================================================
    unique_drugs = sorted(meta["drug"].unique())
    unique_groups = sorted(set(drug_to_group.values()))
    drug_idx_map = {d: i for i, d in enumerate(unique_drugs)}
    drug_indices = meta["drug"].map(drug_idx_map).values.astype(np.int64)

    logger.info(f"  Unique drugs: {len(unique_drugs)}")
    logger.info(f"  Pathway groups: {len(unique_groups)}: {unique_groups}")

    # ==================================================================
    # Train/Val split (90/10, stratified by drug)
    # ==================================================================
    logger.info("\n--- Train/Val split (90/10, stratified by drug) ---")

    # Stratify by drug to ensure all drugs appear in both sets
    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=meta["drug"].values,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    y_train = y[train_idx]
    y_val = y[val_idx]
    d_train = drug_indices[train_idx]
    d_val = drug_indices[val_idx]

    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    logger.info(f"  Train drugs: {len(set(meta['drug'].iloc[train_idx]))}")
    logger.info(f"  Val drugs: {len(set(meta['drug'].iloc[val_idx]))}")

    # ==================================================================
    # B3. Train Multi-Task NN
    # ==================================================================
    logger.info("\n--- B3: Training Multi-Task NN ---")
    logger.info(f"  Architecture: {X.shape[1]} -> 512 -> 256 -> [pathway: 128] -> 1")
    logger.info(f"  Pathway groups: {len(unique_groups)}")
    logger.info(f"  Drug heads: {len(unique_drugs)}")

    mt_model = PRISMMultiTaskNN(
        input_dim=X.shape[1],
        pathway_names=unique_groups,
        drug_names=unique_drugs,
        drug_to_pathway=drug_to_group,
    )

    n_params_mt = sum(p.numel() for p in mt_model.parameters())
    logger.info(f"  Parameters: {n_params_mt:,}")

    mt_model, mt_log = train_multitask(
        X_train, y_train, d_train, mt_model,
        X_val=X_val, y_val=y_val, drug_idx_val=d_val,
        lr=5e-4, weight_decay=1e-4,
        batch_size=512, max_epochs=100, patience=10,
    )

    # Evaluate multi-task
    mt_preds_val = predict_multitask(mt_model, X_val, d_val)
    mt_metrics = compute_metrics(y_val, mt_preds_val)
    logger.info(f"  Multi-task val: RMSE={mt_metrics['rmse']}, R2={mt_metrics['r2']}")

    # ==================================================================
    # B4. Train Single-Task NN
    # ==================================================================
    logger.info("\n--- B4: Training Single-Task NN ---")
    logger.info(f"  Architecture: {X.shape[1]} -> 512 -> 256 -> 128 -> 1")

    st_model, st_log = train_singletask(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        lr=5e-4, weight_decay=1e-4,
        batch_size=512, max_epochs=100, patience=10,
        input_dim=X.shape[1],
    )

    n_params_st = sum(p.numel() for p in st_model.parameters())
    logger.info(f"  Parameters: {n_params_st:,}")

    # Evaluate single-task
    st_preds_val = predict_singletask(st_model, X_val)
    st_metrics = compute_metrics(y_val, st_preds_val)
    logger.info(f"  Single-task val: RMSE={st_metrics['rmse']}, R2={st_metrics['r2']}")

    # ==================================================================
    # B5. Compare + Per-pathway analysis
    # ==================================================================
    logger.info("\n--- B5: Comparison ---")
    logger.info("=" * 60)
    logger.info(f"  {'Model':<30s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'N':>8s}")
    logger.info(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    logger.info(f"  {'Single-task NN (GDSC2, known)':<30s} {'14.48':>8s} {'--':>8s} {'--':>8s} {'719':>8s}")
    logger.info(f"  {'Multi-task NN (GDSC2, known)':<30s} {'15.25':>8s} {'--':>8s} {'--':>8s} {'719':>8s}")

    for label, metrics in [
        ("Single-task NN (PRISM)", st_metrics),
        ("Multi-task NN (PRISM)", mt_metrics),
    ]:
        logger.info(
            f"  {label:<30s} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} "
            f"{metrics['r2']:>8.4f} {metrics['n_samples']:>8d}"
        )

    # Per-pathway-group analysis
    logger.info("\n  Per-pathway-group RMSE:")
    logger.info(f"  {'Group':<25s} {'MT RMSE':>8s} {'ST RMSE':>8s} {'Delta':>8s} {'N':>6s}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    val_meta = meta.iloc[val_idx].reset_index(drop=True)
    val_groups = val_meta["drug"].map(drug_to_group).values

    per_group_results = []
    for grp in sorted(set(val_groups)):
        mask = val_groups == grp
        if mask.sum() < 10:
            continue
        mt_grp = compute_metrics(y_val[mask], mt_preds_val[mask])
        st_grp = compute_metrics(y_val[mask], st_preds_val[mask])
        delta = st_grp["rmse"] - mt_grp["rmse"]
        better = "MT" if delta > 0 else "ST"
        logger.info(
            f"  {grp:<25s} {mt_grp['rmse']:>8.4f} {st_grp['rmse']:>8.4f} "
            f"{delta:>+8.4f} {mask.sum():>6d}"
        )
        per_group_results.append({
            "pathway_group": grp,
            "mt_rmse": mt_grp["rmse"],
            "mt_mae": mt_grp["mae"],
            "mt_r2": mt_grp["r2"],
            "st_rmse": st_grp["rmse"],
            "st_mae": st_grp["mae"],
            "st_r2": st_grp["r2"],
            "delta_rmse": round(delta, 4),
            "better": better,
            "n_samples": mask.sum(),
        })

    # ==================================================================
    # B6. Extract drug embeddings
    # ==================================================================
    logger.info("\n--- B6: Extracting drug embeddings ---")

    # Use full dataset (scaled) for embeddings
    X_full_scaled = scaler.transform(X).astype(np.float32)
    drug_emb_df = extract_drug_embeddings(
        mt_model, X_full_scaled, drug_indices, unique_drugs, meta,
    )

    # ==================================================================
    # B7. Save everything
    # ==================================================================
    logger.info("\n--- B7: Saving results ---")

    # Overall comparison
    comparison = pd.DataFrame([
        {"model": "single_task_nn_gdsc2", "dataset": "GDSC2",
         "rmse": 14.48, "mae": None, "r2": None, "n_samples": 719},
        {"model": "multi_task_nn_gdsc2", "dataset": "GDSC2",
         "rmse": 15.25, "mae": None, "r2": None, "n_samples": 719},
        {"model": "single_task_nn_prism", "dataset": "PRISM",
         **st_metrics},
        {"model": "multi_task_nn_prism", "dataset": "PRISM",
         **mt_metrics},
    ])
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    logger.info(f"  Saved model_comparison.csv")

    # Per-group results
    pg_df = pd.DataFrame(per_group_results)
    pg_df.to_csv(RESULTS_DIR / "per_pathway_group_comparison.csv", index=False)
    logger.info(f"  Saved per_pathway_group_comparison.csv")

    # Training logs
    pd.DataFrame(mt_log).to_csv(RESULTS_DIR / "multitask_training_log.csv", index=False)
    pd.DataFrame(st_log).to_csv(RESULTS_DIR / "singletask_training_log.csv", index=False)
    logger.info(f"  Saved training logs")

    # Drug embeddings
    emb_path = DATA_CACHE / "prism_multitask_drug_embeddings.parquet"
    drug_emb_df.to_parquet(emb_path, index=False)
    logger.info(f"  Saved drug embeddings to {emb_path}")

    # Model state dicts
    torch.save(mt_model.state_dict(), RESULTS_DIR / "multitask_nn_state.pt")
    torch.save(st_model.state_dict(), RESULTS_DIR / "singletask_nn_state.pt")
    logger.info(f"  Saved model state dicts")

    # Save scaler
    import joblib
    joblib.dump(scaler, RESULTS_DIR / "prism_scaler.joblib")

    # Save a summary report
    elapsed = time.time() - t_start
    summary_lines = [
        "PRISM Pan-Cancer Multi-Task NN Experiment",
        "=" * 50,
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total time: {elapsed:.0f}s",
        f"",
        f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_drugs)} drugs",
        f"Train/Val: {len(X_train)}/{len(X_val)} (90/10)",
        f"Pathway groups: {len(unique_groups)}",
        f"",
        "Model Comparison:",
        f"  Single-task NN (GDSC2):    RMSE = 14.48  (719 samples)",
        f"  Multi-task NN (GDSC2):     RMSE = 15.25  (719 samples)",
        f"  Single-task NN (PRISM):    RMSE = {st_metrics['rmse']:.4f}, R2 = {st_metrics['r2']:.4f}  ({st_metrics['n_samples']} samples)",
        f"  Multi-task NN (PRISM):     RMSE = {mt_metrics['rmse']:.4f}, R2 = {mt_metrics['r2']:.4f}  ({mt_metrics['n_samples']} samples)",
        f"",
        "Per-Pathway-Group RMSE (Multi-Task):",
    ]
    for row in per_group_results:
        summary_lines.append(
            f"  {row['pathway_group']:<25s}: MT={row['mt_rmse']:.4f}, ST={row['st_rmse']:.4f}, "
            f"delta={row['delta_rmse']:+.4f} ({row['n_samples']} samples)"
        )
    summary_lines.extend([
        "",
        f"Drug embeddings: {len(drug_emb_df)} drugs x 128 dims",
        f"Saved to: {emb_path}",
    ])

    summary_text = "\n".join(summary_lines)
    with open(RESULTS_DIR / "experiment_summary.txt", "w") as f:
        f.write(summary_text)
    logger.info(f"\n{summary_text}")

    logger.info(f"\nAll results saved to {RESULTS_DIR}/")
    logger.info(f"Total time: {elapsed:.0f}s")

    return {
        "mt_metrics": mt_metrics,
        "st_metrics": st_metrics,
        "per_group": per_group_results,
        "drug_embeddings": drug_emb_df,
    }


if __name__ == "__main__":
    run_prism_multitask_experiment()
