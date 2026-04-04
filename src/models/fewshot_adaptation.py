"""
Few-shot leave-one-dataset-out (LODO) benchmark for domain adaptation.

Tests whether a small support set from the held-out dataset can improve
predictions over a zero-shot baseline trained on all other datasets.

Methods:
  1. L1-logistic regression (zero-shot baseline)
  2. Feature calibration (shift + scale)
  3. Neural network fine-tuning (2-layer NN, low-lr fine-tune on support)
  4. MAML (conditional, first-order)

See docs/meta_learning_protocol.md for the full protocol.
"""
import logging
import os
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Adaptation method interface
# ══════════════════════════════════════════════════════════════════════════

class AdaptationMethod(ABC):
    """Base class for adaptation methods."""

    name: str = "base"

    @abstractmethod
    def train(self, X_source: np.ndarray, y_source: np.ndarray) -> "AdaptationMethod":
        """Train on source data. Returns self."""
        ...

    @abstractmethod
    def adapt(self, X_support: np.ndarray, y_support: np.ndarray) -> "AdaptationMethod":
        """Adapt to the target domain using a small support set. Returns self."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities on query data. Returns 1D array of P(y=1)."""
        ...


# ══════════════════════════════════════════════════════════════════════════
# Method 1: L1-Logistic Regression (baseline)
# ══════════════════════════════════════════════════════════════════════════

class L1LogisticMethod(AdaptationMethod):
    """L1-penalized logistic regression. Zero-shot or retrain-with-support."""

    name = "l1_logistic"

    def __init__(self, C: float = 0.05, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self._X_source = None
        self._y_source = None

    def train(self, X_source: np.ndarray, y_source: np.ndarray) -> "L1LogisticMethod":
        self._X_source = X_source
        self._y_source = y_source
        self.scaler.fit(X_source)
        X_s = self.scaler.transform(X_source)
        X_s = np.nan_to_num(X_s, 0.0)
        self.model = LogisticRegression(
            C=self.C, penalty="l1", solver="liblinear",
            max_iter=2000, random_state=self.random_state,
            class_weight="balanced",
        )
        self.model.fit(X_s, y_source)
        return self

    def adapt(self, X_support: np.ndarray, y_support: np.ndarray) -> "L1LogisticMethod":
        """Retrain on source + support (simple pooling)."""
        X_all = np.concatenate([self._X_source, X_support], axis=0)
        y_all = np.concatenate([self._y_source, y_support], axis=0)
        self.scaler.fit(X_all)
        X_s = np.nan_to_num(self.scaler.transform(X_all), 0.0)
        self.model = LogisticRegression(
            C=self.C, penalty="l1", solver="liblinear",
            max_iter=2000, random_state=self.random_state,
            class_weight="balanced",
        )
        self.model.fit(X_s, y_all)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_s = np.nan_to_num(self.scaler.transform(X), 0.0)
        proba = self.model.predict_proba(X_s)
        return proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]


# ══════════════════════════════════════════════════════════════════════════
# Method 2: Feature Calibration
# ══════════════════════════════════════════════════════════════════════════

class FeatureCalibrator:
    """Lightweight domain shift correction via mean/std alignment."""

    def __init__(self):
        self.shift = None
        self.scale = None

    def fit(self, source_mean: np.ndarray, source_std: np.ndarray,
            support_X: np.ndarray) -> "FeatureCalibrator":
        support_mean = support_X.mean(axis=0)
        support_std = support_X.std(axis=0).clip(min=1e-6)
        self.shift = source_mean - support_mean
        self.scale = source_std / support_std
        # Clip extreme scales to prevent blow-up
        self.scale = np.clip(self.scale, 0.1, 10.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.shift is None:
            return X
        return (X + self.shift) * self.scale


class FeatureCalibrationMethod(AdaptationMethod):
    """L1-logistic with feature calibration for domain shift."""

    name = "feature_calibration"

    def __init__(self, C: float = 0.05, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.calibrator = FeatureCalibrator()
        self.model = None
        self._source_mean = None
        self._source_std = None

    def train(self, X_source: np.ndarray, y_source: np.ndarray) -> "FeatureCalibrationMethod":
        self._source_mean = X_source.mean(axis=0)
        self._source_std = X_source.std(axis=0).clip(min=1e-6)
        self.scaler.fit(X_source)
        X_s = np.nan_to_num(self.scaler.transform(X_source), 0.0)
        self.model = LogisticRegression(
            C=self.C, penalty="l1", solver="liblinear",
            max_iter=2000, random_state=self.random_state,
            class_weight="balanced",
        )
        self.model.fit(X_s, y_source)
        return self

    def adapt(self, X_support: np.ndarray, y_support: np.ndarray) -> "FeatureCalibrationMethod":
        """Calibrate features to align support set with source distribution."""
        self.calibrator.fit(self._source_mean, self._source_std, X_support)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_cal = self.calibrator.transform(X)
        X_s = np.nan_to_num(self.scaler.transform(X_cal), 0.0)
        proba = self.model.predict_proba(X_s)
        return proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]


# ══════════════════════════════════════════════════════════════════════════
# Method 3: Neural Network Fine-tuning
# ══════════════════════════════════════════════════════════════════════════

class NNFineTuneMethod(AdaptationMethod):
    """2-layer NN trained on source, fine-tuned on support with low lr."""

    name = "nn_finetune"

    def __init__(
        self,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        source_lr: float = 1e-3,
        finetune_lr: float = 1e-4,
        source_epochs: int = 100,
        finetune_epochs: int = 15,
        batch_size: int = 64,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.source_lr = source_lr
        self.finetune_lr = finetune_lr
        self.source_epochs = source_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self._base_state = None

    def train(self, X_source: np.ndarray, y_source: np.ndarray) -> "NNFineTuneMethod":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.scaler.fit(X_source)
        X_s = np.nan_to_num(self.scaler.transform(X_source), 0.0).astype(np.float32)

        input_dim = X_s.shape[1]
        self.model = self._build_model(input_dim)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.source_lr, weight_decay=1e-4,
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([y_source.sum() / max(1, (1 - y_source).sum())])
        )

        dataset = TensorDataset(
            torch.tensor(X_s), torch.tensor(y_source.astype(np.float32)),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        no_improve = 0

        for epoch in range(self.source_epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                logits = self.model(X_b).squeeze(-1)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                self._base_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        # Restore best
        if self._base_state:
            self.model.load_state_dict(self._base_state)
        self.model.eval()
        return self

    def adapt(self, X_support: np.ndarray, y_support: np.ndarray) -> "NNFineTuneMethod":
        """Fine-tune the NN on the support set with low lr."""
        import torch
        import torch.nn as nn

        if self.model is None or self._base_state is None:
            return self

        # Reset to base model
        self.model.load_state_dict(self._base_state)
        self.model.train()

        X_s = np.nan_to_num(self.scaler.transform(X_support), 0.0).astype(np.float32)
        X_t = torch.tensor(X_s)
        y_t = torch.tensor(y_support.astype(np.float32))

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.finetune_lr, weight_decay=1e-4,
        )
        n_pos = y_support.sum()
        n_neg = max(1, len(y_support) - n_pos)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(n_neg) / float(max(1, n_pos))])
        )

        for _ in range(self.finetune_epochs):
            optimizer.zero_grad()
            logits = self.model(X_t).squeeze(-1)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            return np.full(len(X), 0.5)

        X_s = np.nan_to_num(self.scaler.transform(X), 0.0).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_s)).squeeze(-1)
            proba = torch.sigmoid(logits).numpy()
        return proba

    def _build_model(self, input_dim: int):
        import torch.nn as nn

        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════
# Few-Shot LODO Benchmark
# ══════════════════════════════════════════════════════════════════════════

def stratified_split(
    X: np.ndarray, y: np.ndarray, k: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split: draw k samples as support, rest as query.

    Returns (X_support, y_support, X_query, y_query).
    If k >= len(y), returns all as support and empty query.
    """
    rng = np.random.RandomState(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Stratified: roughly k * prevalence positives, rest negatives
    n_pos_support = max(1, int(round(k * len(pos_idx) / len(y))))
    n_neg_support = k - n_pos_support

    # Clamp
    n_pos_support = min(n_pos_support, len(pos_idx) - 1)
    n_neg_support = min(n_neg_support, len(neg_idx) - 1)

    # Ensure at least 1 in each class for support if k >= 2
    if k >= 2 and n_pos_support < 1:
        n_pos_support = 1
        n_neg_support = k - 1
    if k >= 2 and n_neg_support < 1:
        n_neg_support = 1
        n_pos_support = k - 1

    pos_perm = rng.permutation(len(pos_idx))
    neg_perm = rng.permutation(len(neg_idx))

    support_pos = pos_idx[pos_perm[:n_pos_support]]
    support_neg = neg_idx[neg_perm[:n_neg_support]]
    support_idx = np.concatenate([support_pos, support_neg])

    query_pos = pos_idx[pos_perm[n_pos_support:]]
    query_neg = neg_idx[neg_perm[n_neg_support:]]
    query_idx = np.concatenate([query_pos, query_neg])

    return X[support_idx], y[support_idx], X[query_idx], y[query_idx]


class FewShotLODOBenchmark:
    """
    Run few-shot LODO evaluation across multiple adaptation methods.

    Parameters
    ----------
    datasets : dict[str, dict]
        {geo_id: {"X": np.ndarray, "y": np.ndarray, "endpoint_family": str, "drug": str}}
    methods : list[AdaptationMethod]
        Adaptation methods to evaluate.
    support_sizes : list[int]
        Number of support samples (k=0 is zero-shot baseline).
    n_repeats : int
        Number of random repeats per (dataset, method, k).
    min_dataset_size : int
        Minimum dataset size for inclusion.
    """

    def __init__(
        self,
        datasets: dict,
        methods: list[AdaptationMethod],
        support_sizes: Optional[list[int]] = None,
        n_repeats: int = 5,
        min_dataset_size: int = 30,
    ):
        self.datasets = datasets
        self.methods = methods
        self.support_sizes = support_sizes or [0, 5, 10, 20]
        self.n_repeats = n_repeats
        self.min_dataset_size = min_dataset_size

    def run(self) -> pd.DataFrame:
        """Run the full benchmark. Returns a DataFrame of all results."""
        all_geos = sorted(self.datasets.keys())

        # Filter by size
        eligible = [
            g for g in all_geos
            if len(self.datasets[g]["y"]) >= self.min_dataset_size
        ]
        logger.info(
            f"FewShot LODO: {len(eligible)}/{len(all_geos)} datasets "
            f"meet min size {self.min_dataset_size}"
        )

        if len(eligible) < 2:
            logger.error("Need >= 2 eligible datasets")
            return pd.DataFrame()

        # Align feature dimensions
        dims = [self.datasets[g]["X"].shape[1] for g in eligible]
        min_dim = min(dims)

        results = []

        for holdout_idx, holdout in enumerate(eligible):
            logger.info(
                f"[{holdout_idx + 1}/{len(eligible)}] Holdout: {holdout} "
                f"(n={len(self.datasets[holdout]['y'])})"
            )

            # Source data
            source_geos = [g for g in eligible if g != holdout]
            X_source = np.concatenate(
                [self.datasets[g]["X"][:, :min_dim] for g in source_geos], axis=0,
            )
            y_source = np.concatenate(
                [self.datasets[g]["y"] for g in source_geos], axis=0,
            )

            X_holdout = self.datasets[holdout]["X"][:, :min_dim]
            y_holdout = self.datasets[holdout]["y"]
            ef = self.datasets[holdout].get("endpoint_family", "unknown")
            drug = self.datasets[holdout].get("drug", "")

            if len(np.unique(y_source)) < 2:
                logger.warning(f"  Skipping {holdout}: source has only one class")
                continue

            for method in self.methods:
                # Create a fresh copy for each holdout to avoid state leakage
                m = deepcopy(method)

                try:
                    m.train(X_source, y_source)
                except Exception as e:
                    logger.warning(f"  {holdout}/{m.name}: train failed: {e}")
                    continue

                for k in self.support_sizes:
                    if k == 0:
                        # Zero-shot: no support, evaluate on full holdout
                        try:
                            y_score = m.predict_proba(X_holdout)
                            if len(np.unique(y_holdout)) < 2:
                                continue
                            auroc = roc_auc_score(y_holdout, y_score)
                            auprc = average_precision_score(y_holdout, y_score)
                        except Exception as e:
                            logger.warning(f"  {holdout}/{m.name}/k=0: {e}")
                            auroc = 0.5
                            auprc = float("nan")

                        results.append({
                            "held_out_dataset": holdout,
                            "method": m.name,
                            "k": 0,
                            "seed": 0,
                            "endpoint_family": ef,
                            "drug": drug,
                            "n_holdout": len(y_holdout),
                            "n_query": len(y_holdout),
                            "auroc": round(auroc, 4),
                            "auprc": round(auprc, 4),
                        })
                        logger.info(
                            f"    {m.name} k=0: AUROC={auroc:.3f}"
                        )
                        continue

                    if k >= len(y_holdout) - 5:
                        # Not enough samples for meaningful query set
                        continue

                    for seed in range(self.n_repeats):
                        m_copy = deepcopy(m)
                        # Re-train on source so we start from same base
                        try:
                            m_copy.train(X_source, y_source)
                        except Exception:
                            continue

                        X_sup, y_sup, X_query, y_query = stratified_split(
                            X_holdout, y_holdout, k, seed=seed + k * 100,
                        )

                        if len(np.unique(y_query)) < 2 or len(y_query) < 5:
                            continue
                        if len(np.unique(y_sup)) < 2 and k >= 2:
                            continue

                        try:
                            m_copy.adapt(X_sup, y_sup)
                            y_score = m_copy.predict_proba(X_query)
                            auroc = roc_auc_score(y_query, y_score)
                            auprc = average_precision_score(y_query, y_score)
                        except Exception as e:
                            logger.debug(f"    {holdout}/{m.name}/k={k}/s={seed}: {e}")
                            auroc = 0.5
                            auprc = float("nan")

                        results.append({
                            "held_out_dataset": holdout,
                            "method": m_copy.name,
                            "k": k,
                            "seed": seed,
                            "endpoint_family": ef,
                            "drug": drug,
                            "n_holdout": len(y_holdout),
                            "n_query": len(y_query),
                            "auroc": round(auroc, 4),
                            "auprc": round(auprc, 4),
                        })

                    # Log mean for this (method, k) across seeds
                    k_results = [r for r in results
                                 if r["held_out_dataset"] == holdout
                                 and r["method"] == m.name and r["k"] == k]
                    if k_results:
                        mean_auc = np.mean([r["auroc"] for r in k_results])
                        logger.info(f"    {m.name} k={k}: mean AUROC={mean_auc:.3f}")

        return pd.DataFrame(results)


def aggregate_fewshot_results(
    results_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate few-shot results into summary tables.

    Returns (summary_df, by_endpoint_df).
    """
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Summary: mean across seeds and datasets, per (method, k)
    summary_rows = []
    for (method, k), grp in results_df.groupby(["method", "k"]):
        # Mean across seeds first, then across datasets
        per_dataset = grp.groupby("held_out_dataset")["auroc"].mean()
        summary_rows.append({
            "method": method,
            "k": k,
            "mean_auroc": round(per_dataset.mean(), 4),
            "std_auroc": round(per_dataset.std(), 4),
            "median_auroc": round(per_dataset.median(), 4),
            "n_datasets": len(per_dataset),
            "mean_auprc": round(
                grp.groupby("held_out_dataset")["auprc"].mean().mean(), 4
            ),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["method", "k"])

    # By endpoint family
    endpoint_rows = []
    for (method, k, ef), grp in results_df.groupby(["method", "k", "endpoint_family"]):
        per_dataset = grp.groupby("held_out_dataset")["auroc"].mean()
        endpoint_rows.append({
            "method": method,
            "k": k,
            "endpoint_family": ef,
            "mean_auroc": round(per_dataset.mean(), 4),
            "std_auroc": round(per_dataset.std(), 4),
            "n_datasets": len(per_dataset),
        })
    endpoint_df = pd.DataFrame(endpoint_rows).sort_values(["endpoint_family", "method", "k"])

    return summary_df, endpoint_df
