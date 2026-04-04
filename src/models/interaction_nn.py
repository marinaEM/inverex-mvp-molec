"""
Shallow neural network for drug-response prediction with gene-gene interactions.

Unlike LightGBM (which treats features independently via axis-aligned splits),
a neural network with hidden layers naturally learns pairwise and higher-order
interactions between gene expression features, drug fingerprint bits, and dose.

Architecture:
  Input (2003) -> Dense(256) -> ReLU -> Dropout -> Dense(128) -> ReLU -> Dropout -> Dense(1)

This captures gene-gene interactions in the first hidden layer via learned
weight combinations, while remaining shallow enough to avoid overfitting
on small datasets (~719 cell-line samples).
"""
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

# Set before importing torch to avoid OpenMP conflict with LightGBM on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class DrugResponseNN(nn.Module):
    """
    Shallow feedforward network for drug-response regression.

    Parameters
    ----------
    input_dim : int
        Number of input features (gene z-scores + ECFP bits + dose).
    hidden_dims : list[int]
        Sizes of hidden layers. Default [256, 128] gives two layers
        that can capture pairwise gene interactions.
    dropout : float
        Dropout probability for regularization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    hidden_dims: Optional[list[int]] = None,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 15,
    device: str = "cpu",
) -> DrugResponseNN:
    """
    Train a DrugResponseNN with early stopping.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data (already scaled).
    X_val, y_val : np.ndarray or None
        Validation data for early stopping. If None, uses training loss
        (not recommended).
    hidden_dims : list[int]
        Hidden layer sizes.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate for Adam.
    weight_decay : float
        L2 regularization.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience (epochs without val loss improvement).
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    DrugResponseNN
        Trained model (in eval mode, on CPU).
    """
    if hidden_dims is None:
        hidden_dims = [256, 128]

    input_dim = X_train.shape[1]
    model = DrugResponseNN(input_dim, hidden_dims, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Build data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 50 == 0:
                logger.debug(
                    f"  Epoch {epoch+1}/{max_epochs}: "
                    f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if epochs_no_improve >= patience:
                logger.debug(
                    f"  Early stopping at epoch {epoch+1} "
                    f"(best val_loss={best_val_loss:.4f})"
                )
                break
        else:
            # No validation: just track training loss
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.cpu()
    return model


def predict_nn(
    model: DrugResponseNN,
    X: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate predictions from a trained DrugResponseNN.

    Parameters
    ----------
    model : DrugResponseNN
        Trained model (eval mode).
    X : np.ndarray
        Feature matrix (already scaled with the same scaler used in training).
    device : str
        Device for inference.

    Returns
    -------
    np.ndarray
        Predicted percent inhibition values.
    """
    model.eval()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds
