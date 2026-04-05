"""
Foundation Model Evaluation
===========================
LODO (Leave-One-Dataset-Out) evaluation on 10 representative datasets.

For each held-out dataset with response labels:
    1. Load pretrained Perceiver encoder (relaxed: pretrained on ALL data)
    2. Extract [CLS] embeddings (128-dim) for patients
    3. Fine-tune a linear classifier: [CLS] → response (20 epochs)
    4. Report AUC

Comparison:
    A. LightGBM on 212 landmark genes only
    B. Foundation [CLS] → linear probe
    C. LightGBM on 212 genes + [CLS] 128-dim
"""

from __future__ import annotations

import os, sys, time, logging, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]  # inverex-mvp
sys.path.insert(0, str(ROOT / "src"))

from models.foundation.gene_universe import build_gene_universe, discover_all_datasets
from models.foundation.expression_tokenizer import ExpressionTokenizer
from models.foundation.expression_encoder import ExpressionEncoder, EncoderConfig

logger = logging.getLogger(__name__)

RESULTS = ROOT / "results" / "foundation"
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pretrained_encoder() -> tuple[ExpressionEncoder, EncoderConfig]:
    """Load the pretrained encoder from disk."""
    with open(RESULTS / "pretrain_config.json") as f:
        config = json.load(f)

    cfg = EncoderConfig(
        n_genes=config["n_genes"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_latents=config.get("n_latents", 16),
    )
    encoder = ExpressionEncoder(cfg)
    state = torch.load(RESULTS / "encoder.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder, cfg


def load_tokenizer(gene_list, gene2idx) -> ExpressionTokenizer:
    tok = ExpressionTokenizer(gene_list, gene2idx)
    tok.load_stats()
    return tok


@torch.no_grad()
def extract_embeddings(
    encoder: ExpressionEncoder,
    tokenizer: ExpressionTokenizer,
    expr_df: pd.DataFrame,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract [CLS] embeddings for all samples."""
    encoder.eval()
    embeddings = []

    for start in range(0, len(expr_df), batch_size):
        end = min(start + batch_size, len(expr_df))
        batch_df = expr_df.iloc[start:end]
        tokens = tokenizer.tokenize_batch_fast(batch_df)

        cls_emb, _ = encoder(
            tokens["gene_ids"], tokens["ranks"], tokens["mag_bins"]
        )
        embeddings.append(cls_emb.numpy())

    return np.concatenate(embeddings, axis=0)


def discover_labeled_datasets() -> list[tuple[str, Path, Path]]:
    """Find datasets with both expression + response labels."""
    labeled = []

    ctrdb = DATA_RAW / "ctrdb"
    if ctrdb.exists():
        for gse_dir in sorted(ctrdb.iterdir()):
            if not gse_dir.is_dir():
                continue
            expr_files = list(gse_dir.glob("*_expression.parquet"))
            resp_file = gse_dir / "response_labels.parquet"
            if expr_files and resp_file.exists():
                labeled.append((gse_dir.name, expr_files[0], resp_file))

    ispy2_expr = DATA_RAW / "ispy2" / "GSE194040_expression.parquet"
    ispy2_resp = DATA_RAW / "ispy2" / "response_labels.parquet"
    if ispy2_expr.exists() and ispy2_resp.exists():
        labeled.append(("ISPY2", ispy2_expr, ispy2_resp))

    br_expr = DATA_RAW / "brightness" / "GSE164458_expression.parquet"
    br_resp = DATA_RAW / "brightness" / "response_labels.parquet"
    if br_expr.exists() and br_resp.exists():
        labeled.append(("BrighTNess", br_expr, br_resp))

    return labeled


def select_representative_datasets(
    labeled: list[tuple[str, Path, Path]], n: int = 10
) -> list[tuple[str, Path, Path]]:
    """Select n largest datasets for evaluation."""
    if len(labeled) <= n:
        return labeled

    sizes = []
    for name, expr_path, resp_path in labeled:
        try:
            resp = pd.read_parquet(resp_path)
            sizes.append(len(resp))
        except Exception:
            sizes.append(0)

    indexed = sorted(enumerate(sizes), key=lambda x: -x[1])
    selected = sorted([idx for idx, _ in indexed[:n]])
    return [labeled[i] for i in selected]


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.fc = nn.Linear(d_in, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def finetune_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_epochs: int = 20,
    lr: float = 1e-3,
) -> np.ndarray:
    d = X_train.shape[1]
    model = LinearProbe(d)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)

    model.train()
    for epoch in range(n_epochs):
        logits = model(X_tr)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te)).numpy()
    return probs


# ---------------------------------------------------------------------------
# LightGBM baseline
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, X_test):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=100, learning_rate=0.05,
        max_depth=4, num_leaves=15, min_child_samples=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        verbosity=-1, random_state=42,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return probs


# ---------------------------------------------------------------------------
# LODO evaluation
# ---------------------------------------------------------------------------

def evaluate_lodo(n_eval_datasets: int = 10, n_finetune_epochs: int = 20):
    """LODO evaluation: A. LightGBM-212, B. Linear-probe, C. Combined."""
    RESULTS.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pretrained model …")
    gene_list, gene2idx = build_gene_universe()
    encoder, cfg = load_pretrained_encoder()
    tokenizer = load_tokenizer(gene_list, gene2idx)

    # Landmark genes
    landmark_path = DATA_CACHE / "geneinfo_beta_input.txt"
    landmark_genes = pd.read_csv(landmark_path, sep="\t")["gene_symbol"].tolist()
    landmark_genes = [g for g in landmark_genes if isinstance(g, str) and len(g) > 0]
    logger.info("Landmark genes: %d", len(landmark_genes))

    # Find labeled datasets
    labeled = discover_labeled_datasets()
    logger.info("Found %d labeled datasets", len(labeled))

    eval_datasets = select_representative_datasets(labeled, n=n_eval_datasets)
    logger.info("Evaluating on %d representative datasets", len(eval_datasets))

    # Load ALL labeled data
    all_expr = {}
    all_resp = {}
    for name, expr_path, resp_path in labeled:
        try:
            expr = pd.read_parquet(expr_path)
            resp = pd.read_parquet(resp_path)
            if "response" not in resp.columns:
                continue
            if "sample_id" in resp.columns:
                resp = resp.set_index("sample_id")
            common = expr.index.intersection(resp.index)
            if len(common) >= 10:
                all_expr[name] = expr.loc[common]
                all_resp[name] = resp.loc[common, "response"].values.astype(int)
                logger.info("  %s: %d samples", name, len(common))
        except Exception as e:
            logger.warning("  %s: failed: %s", name, e)

    # LODO loop
    results = []
    for name, expr_path, resp_path in eval_datasets:
        if name not in all_expr:
            continue

        logger.info("=" * 50)
        logger.info("LODO held-out: %s (%d samples)", name, len(all_expr[name]))

        X_test_expr = all_expr[name]
        y_test = all_resp[name]

        if len(np.unique(y_test)) < 2:
            logger.warning("  Single class, skipping")
            continue

        train_names = [n for n in all_expr if n != name]
        if not train_names:
            continue

        y_tr = np.concatenate([all_resp[n] for n in train_names])

        # A. LightGBM on landmark genes
        avail_lm = [g for g in landmark_genes
                     if g in X_test_expr.columns
                     and all(g in all_expr[n].columns for n in train_names)]
        if len(avail_lm) < 10:
            avail_lm = [g for g in landmark_genes if g in X_test_expr.columns]

        try:
            X_tr_lgbm = np.vstack([
                all_expr[n].reindex(columns=avail_lm, fill_value=0).values
                for n in train_names
            ])
            X_te_lgbm = X_test_expr.reindex(columns=avail_lm, fill_value=0).values

            probs_a = train_lightgbm(X_tr_lgbm, y_tr, X_te_lgbm)
            auc_a = roc_auc_score(y_test, probs_a)
        except Exception as e:
            logger.warning("  LightGBM-212 failed: %s", e)
            auc_a = np.nan

        # B. Foundation [CLS] linear probe
        try:
            emb_test = extract_embeddings(encoder, tokenizer, X_test_expr)
            emb_train = np.vstack([
                extract_embeddings(encoder, tokenizer, all_expr[n])
                for n in train_names
            ])

            probs_b = finetune_linear_probe(
                emb_train, y_tr, emb_test, n_epochs=n_finetune_epochs
            )
            auc_b = roc_auc_score(y_test, probs_b)
        except Exception as e:
            logger.warning("  Linear probe failed: %s", e)
            auc_b = np.nan

        # C. LightGBM on landmarks + [CLS]
        try:
            X_tr_combined = np.hstack([X_tr_lgbm, emb_train])
            X_te_combined = np.hstack([X_te_lgbm, emb_test])

            probs_c = train_lightgbm(X_tr_combined, y_tr, X_te_combined)
            auc_c = roc_auc_score(y_test, probs_c)
        except Exception as e:
            logger.warning("  Combined failed: %s", e)
            auc_c = np.nan

        logger.info(
            "  AUC — LightGBM-212: %.3f | Linear-probe: %.3f | Combined: %.3f",
            auc_a, auc_b, auc_c,
        )

        results.append({
            "dataset": name,
            "n_test": len(y_test),
            "n_train": len(y_tr),
            "response_rate": round(float(y_test.mean()), 3),
            "auc_lightgbm_212": round(auc_a, 4) if not np.isnan(auc_a) else None,
            "auc_linear_probe": round(auc_b, 4) if not np.isnan(auc_b) else None,
            "auc_combined": round(auc_c, 4) if not np.isnan(auc_c) else None,
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "lodo_results.csv", index=False)
    logger.info("Saved lodo_results.csv (%d datasets)", len(results_df))

    # Summary comparison
    valid = results_df.dropna(subset=["auc_lightgbm_212", "auc_linear_probe", "auc_combined"])
    summary = {
        "method": [
            "A. LightGBM 212 genes",
            "B. Foundation [CLS] linear probe",
            "C. LightGBM 212 + [CLS] 128",
        ],
        "mean_auc": [
            round(valid["auc_lightgbm_212"].mean(), 4),
            round(valid["auc_linear_probe"].mean(), 4),
            round(valid["auc_combined"].mean(), 4),
        ],
        "median_auc": [
            round(valid["auc_lightgbm_212"].median(), 4),
            round(valid["auc_linear_probe"].median(), 4),
            round(valid["auc_combined"].median(), 4),
        ],
        "n_datasets": [len(valid)] * 3,
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS / "comparison.csv", index=False)
    logger.info("\n%s", summary_df.to_string(index=False))
    logger.info("Saved comparison.csv")

    return results_df, summary_df


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS / "evaluate.log"),
        ],
    )
    evaluate_lodo()
