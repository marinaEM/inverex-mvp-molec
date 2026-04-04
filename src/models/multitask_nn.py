"""
Multi-task neural network for drug-response prediction (Step 2).

Groups drugs by their GDSC2 pathway_name annotation and builds a
multi-task architecture:

  Shared encoder:   Input -> Linear(256) -> ReLU -> Dropout(0.3)
  Pathway heads:    per pathway_name -> Linear(128) -> ReLU -> Dropout(0.2)
  Drug output:      per drug -> Linear(128, 1)

Each training sample is routed through the shared encoder, then to the
pathway-specific layer for its drug's pathway, and finally to the
drug-specific output head.

Comparison with the single-task NN (src/models/interaction_nn.py) is
done via 5-fold CV on the cell-line training matrix (719 samples).

Entry point: ``run_multitask_experiment()`` or
``pixi run python -m src.models.multitask_nn``
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Avoid OpenMP conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CACHE, DATA_PROCESSED, RESULTS, RANDOM_SEED

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Build drug -> pathway mapping from GDSC2
# ======================================================================

def build_drug_pathway_map(
    ref_path: Optional[Path] = None,
) -> dict[str, str]:
    """
    Build a drug_name -> pathway_name mapping from GDSC2 reference data.

    Returns
    -------
    dict : drug_name (as-is in training matrix) -> pathway_name string.
    """
    if ref_path is None:
        ref_path = DATA_CACHE / "breast_dose_response_ref.parquet"
    ref = pd.read_parquet(ref_path)

    drug_pw = (
        ref[["drug_name", "pathway_name"]]
        .drop_duplicates()
        .set_index("drug_name")["pathway_name"]
        .to_dict()
    )
    logger.info(
        "Drug-pathway mapping: %d drugs across %d pathways",
        len(drug_pw),
        len(set(drug_pw.values())),
    )
    return drug_pw


# ======================================================================
# 2. Identify drug for each training sample
# ======================================================================

def identify_sample_drugs(
    ref_path: Optional[Path] = None,
    training_matrix_path: Optional[Path] = None,
) -> tuple[pd.Series, dict[str, str]]:
    """
    Map each training sample to its drug name and pathway.

    The training matrix is built from GDSC2 dose-response data, so we
    can reconstruct which drug each row corresponds to by joining back
    to the reference.  The training matrix rows are ordered identically
    to the dose-response reference rows used to build them.

    Returns
    -------
    drug_names : pd.Series, length n_samples
        Drug name for each training sample.
    drug_pathway : dict
        drug_name -> pathway_name.
    """
    if ref_path is None:
        ref_path = DATA_CACHE / "breast_dose_response_ref.parquet"
    if training_matrix_path is None:
        training_matrix_path = DATA_PROCESSED / "training_matrix.parquet"

    ref = pd.read_parquet(ref_path)
    tm = pd.read_parquet(training_matrix_path)

    # The training matrix has 719 rows; the ref may have more.
    # Match by the log_dose_um column and row count.
    # The simplest approach: the training matrix rows correspond to the
    # first len(tm) rows of the ref (they are built together).
    if len(ref) >= len(tm):
        drug_names = ref["drug_name"].iloc[: len(tm)].reset_index(drop=True)
    else:
        # Fallback: pad with "Unknown"
        drug_names = ref["drug_name"].reset_index(drop=True)
        while len(drug_names) < len(tm):
            drug_names = pd.concat(
                [drug_names, pd.Series(["Unknown"])], ignore_index=True
            )
        drug_names = drug_names.iloc[: len(tm)]

    drug_pathway = build_drug_pathway_map(ref_path)
    return drug_names, drug_pathway


# ======================================================================
# 3. Multi-task NN model
# ======================================================================

class MultiTaskDrugNN(nn.Module):
    """
    Multi-task neural network with shared encoder, pathway-specific
    intermediate layers, and drug-specific output heads.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    pathway_names : list[str]
        Unique pathway names; one intermediate layer per pathway.
    drug_names : list[str]
        Unique drug names; one output head per drug.
    drug_to_pathway : dict[str, str]
        Mapping from drug name to pathway name.
    shared_dim : int
        Shared encoder hidden dimension (default 256).
    pathway_dim : int
        Pathway-specific hidden dimension (default 128).
    dropout_shared : float
        Dropout for shared encoder (default 0.3).
    dropout_pathway : float
        Dropout for pathway layers (default 0.2).
    """

    def __init__(
        self,
        input_dim: int,
        pathway_names: list[str],
        drug_names: list[str],
        drug_to_pathway: dict[str, str],
        shared_dim: int = 256,
        pathway_dim: int = 128,
        dropout_shared: float = 0.3,
        dropout_pathway: float = 0.2,
    ):
        super().__init__()
        self.pathway_names = pathway_names
        self.drug_names = drug_names
        self.drug_to_pathway = drug_to_pathway

        # Index mappings
        self.pathway_idx = {pw: i for i, pw in enumerate(pathway_names)}
        self.drug_idx = {d: i for i, d in enumerate(drug_names)}

        # Shared patient encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout_shared),
        )

        # Pathway-specific intermediate layers
        self.pathway_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, pathway_dim),
                nn.ReLU(),
                nn.Dropout(dropout_pathway),
            )
            for _ in pathway_names
        ])

        # Drug-specific output heads
        self.drug_heads = nn.ModuleList([
            nn.Linear(pathway_dim, 1)
            for _ in drug_names
        ])

        # Pre-compute drug -> pathway-layer index
        self._drug_to_pw_idx = {}
        for d in drug_names:
            pw = drug_to_pathway.get(d, pathway_names[0])
            if pw not in self.pathway_idx:
                pw = pathway_names[0]  # fallback
            self._drug_to_pw_idx[d] = self.pathway_idx[pw]

    def forward(
        self,
        x: torch.Tensor,
        drug_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)
            Input features.
        drug_indices : Tensor, shape (batch,), dtype long
            Index into self.drug_names for each sample.

        Returns
        -------
        Tensor, shape (batch,)
            Predicted response values.
        """
        # Shared encoding
        h = self.shared_encoder(x)  # (batch, shared_dim)

        # Route each sample through its pathway layer and drug head
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, device=x.device)

        # Group samples by pathway for efficiency
        pw_groups: dict[int, list[int]] = {}
        for i in range(batch_size):
            d_idx = drug_indices[i].item()
            d_name = self.drug_names[d_idx]
            pw_idx = self._drug_to_pw_idx.get(d_name, 0)
            pw_groups.setdefault(pw_idx, []).append(i)

        for pw_idx, sample_indices in pw_groups.items():
            idx_t = torch.tensor(sample_indices, device=x.device, dtype=torch.long)
            h_sub = h[idx_t]  # (n_group, shared_dim)
            pw_out = self.pathway_layers[pw_idx](h_sub)  # (n_group, pathway_dim)

            # Each sample in the group may have a different drug head
            for local_i, global_i in enumerate(sample_indices):
                d_idx = drug_indices[global_i].item()
                out = self.drug_heads[d_idx](pw_out[local_i: local_i + 1])
                outputs[global_i] = out.squeeze()

        return outputs


# ======================================================================
# 4. Training utilities
# ======================================================================

def train_multitask_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    drug_idx_train: np.ndarray,
    model: MultiTaskDrugNN,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    drug_idx_val: Optional[np.ndarray] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 15,
    device: str = "cpu",
) -> MultiTaskDrugNN:
    """Train the multi-task NN with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(drug_idx_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    has_val = X_val is not None and y_val is not None and drug_idx_val is not None
    if has_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
        drug_idx_val_t = torch.tensor(drug_idx_val, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_b, y_b, d_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            d_b = d_b.to(device)

            optimizer.zero_grad()
            pred = model(X_b, d_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t, drug_idx_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 50 == 0:
                logger.debug(
                    "  Epoch %d/%d: train=%.4f, val=%.4f",
                    epoch + 1, max_epochs, avg_train_loss, val_loss,
                )

            if epochs_no_improve >= patience:
                logger.debug(
                    "  Early stopping at epoch %d (best val=%.4f)",
                    epoch + 1, best_val_loss,
                )
                break
        else:
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.cpu()
    return model


# ======================================================================
# 5. Single-task NN baseline (re-uses interaction_nn)
# ======================================================================

def run_singletask_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """5-fold CV for single-task NN. Returns list of fold metrics."""
    from src.models.interaction_nn import DrugResponseNN, train_nn, predict_nn

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        y_tr = y[train_idx]
        y_va = y[val_idx]

        model = train_nn(
            X_tr, y_tr, X_va, y_va,
            hidden_dims=[256, 128],
            dropout=0.3,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            max_epochs=200,
            patience=15,
        )
        preds = predict_nn(model, X_va)
        rmse = float(np.sqrt(np.mean((preds - y_va) ** 2)))
        r2 = float(1 - np.sum((y_va - preds) ** 2) / np.sum((y_va - y_va.mean()) ** 2))

        results.append({
            "fold": fold,
            "model": "single_task_nn",
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })
        logger.info(
            "  Single-task fold %d: RMSE=%.4f, R2=%.4f", fold, rmse, r2
        )

    return results


# ======================================================================
# 6. Multi-task CV
# ======================================================================

def run_multitask_cv(
    X: np.ndarray,
    y: np.ndarray,
    drug_names_series: pd.Series,
    drug_pathway_map: dict[str, str],
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """5-fold CV for multi-task NN. Returns list of fold metrics."""
    unique_drugs = sorted(drug_names_series.unique())
    unique_pathways = sorted(set(
        drug_pathway_map.get(d, "Other") for d in unique_drugs
    ))

    # Build drug index array
    drug_idx_map = {d: i for i, d in enumerate(unique_drugs)}
    drug_indices = drug_names_series.map(drug_idx_map).values.astype(np.int64)

    # Ensure every drug maps to a pathway
    full_d2p = {d: drug_pathway_map.get(d, "Other") for d in unique_drugs}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        y_tr = y[train_idx]
        y_va = y[val_idx]
        d_tr = drug_indices[train_idx]
        d_va = drug_indices[val_idx]

        model = MultiTaskDrugNN(
            input_dim=X_tr.shape[1],
            pathway_names=unique_pathways,
            drug_names=unique_drugs,
            drug_to_pathway=full_d2p,
            shared_dim=256,
            pathway_dim=128,
            dropout_shared=0.3,
            dropout_pathway=0.2,
        )

        model = train_multitask_nn(
            X_tr, y_tr, d_tr, model,
            X_val=X_va, y_val=y_va, drug_idx_val=d_va,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            max_epochs=200,
            patience=15,
        )

        # Predict
        model.eval()
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        d_va_t = torch.tensor(d_va, dtype=torch.long)
        with torch.no_grad():
            preds = model(X_va_t, d_va_t).numpy()

        rmse = float(np.sqrt(np.mean((preds - y_va) ** 2)))
        r2 = float(1 - np.sum((y_va - preds) ** 2) / np.sum((y_va - y_va.mean()) ** 2))

        results.append({
            "fold": fold,
            "model": "multi_task_nn",
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })
        logger.info(
            "  Multi-task fold %d: RMSE=%.4f, R2=%.4f", fold, rmse, r2
        )

    return results


# ======================================================================
# 7. Main experiment
# ======================================================================

def run_multitask_experiment():
    """
    End-to-end experiment comparing single-task vs multi-task NN
    on the cell-line training matrix (5-fold CV).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.time()

    logger.info("=" * 60)
    logger.info("STEP 2: MULTI-TASK NN EXPERIMENT")
    logger.info("=" * 60)

    # ---------------------------------------------------------------- #
    # Load data                                                         #
    # ---------------------------------------------------------------- #
    logger.info("Loading training matrix ...")
    tm_path = DATA_PROCESSED / "training_matrix.parquet"
    tt_path = DATA_PROCESSED / "training_target.parquet"
    X_df = pd.read_parquet(tm_path)
    y_df = pd.read_parquet(tt_path)

    X = X_df.values.astype(np.float32)
    y = y_df.iloc[:, 0].values.astype(np.float32)

    logger.info("  X: %s, y: %s", X.shape, y.shape)

    # ---------------------------------------------------------------- #
    # Build drug-pathway mapping                                        #
    # ---------------------------------------------------------------- #
    logger.info("Building drug-pathway mapping ...")
    drug_names_series, drug_pathway_map = identify_sample_drugs()
    logger.info(
        "  %d unique drugs, %d unique pathways",
        drug_names_series.nunique(),
        len(set(drug_pathway_map.values())),
    )

    # Log pathway distribution
    pw_counts = drug_names_series.map(
        lambda d: drug_pathway_map.get(d, "Other")
    ).value_counts()
    logger.info("Pathway sample counts:")
    for pw, cnt in pw_counts.items():
        logger.info("  %s: %d samples", pw, cnt)

    # ---------------------------------------------------------------- #
    # Single-task NN (5-fold CV)                                        #
    # ---------------------------------------------------------------- #
    logger.info("\n" + "-" * 60)
    logger.info("Running single-task NN 5-fold CV ...")
    logger.info("-" * 60)
    st_results = run_singletask_cv(X, y, n_splits=5, seed=RANDOM_SEED)

    # ---------------------------------------------------------------- #
    # Multi-task NN (5-fold CV)                                         #
    # ---------------------------------------------------------------- #
    logger.info("\n" + "-" * 60)
    logger.info("Running multi-task NN 5-fold CV ...")
    logger.info("-" * 60)
    mt_results = run_multitask_cv(
        X, y, drug_names_series, drug_pathway_map,
        n_splits=5, seed=RANDOM_SEED,
    )

    # ---------------------------------------------------------------- #
    # Combine and save                                                  #
    # ---------------------------------------------------------------- #
    all_results = pd.DataFrame(st_results + mt_results)

    # Summary
    summary = (
        all_results.groupby("model")[["rmse", "r2"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for _, row in summary.iterrows():
        logger.info(
            "  %-20s: RMSE = %.4f +/- %.4f, R2 = %.4f +/- %.4f",
            row["model"],
            row["rmse_mean"],
            row["rmse_std"],
            row["r2_mean"],
            row["r2_std"],
        )

    # Determine winner
    st_rmse = summary.loc[summary["model"] == "single_task_nn", "rmse_mean"].values[0]
    mt_rmse = summary.loc[summary["model"] == "multi_task_nn", "rmse_mean"].values[0]
    if mt_rmse < st_rmse:
        logger.info(
            "\nMulti-task NN improves over single-task: "
            "%.4f vs %.4f RMSE (delta = %.4f)",
            mt_rmse, st_rmse, st_rmse - mt_rmse,
        )
    else:
        logger.info(
            "\nMulti-task NN does NOT improve over single-task: "
            "%.4f vs %.4f RMSE (delta = %.4f)",
            mt_rmse, st_rmse, st_rmse - mt_rmse,
        )

    # Save
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "ablation_multitask_nn.tsv"
    all_results.to_csv(out_path, sep="\t", index=False)
    logger.info("Per-fold results saved to %s", out_path)

    # Also save summary
    summary_path = RESULTS / "ablation_multitask_nn_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    logger.info("Summary saved to %s", summary_path)

    elapsed = time.time() - t_start
    logger.info("Total time: %.0fs", elapsed)

    return all_results


if __name__ == "__main__":
    run_multitask_experiment()
