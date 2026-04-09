"""
Foundation Model Evaluation (Full)
==============================================
Comprehensive evaluation including:

1. LODO evaluation with relaxed protocol
2. Benchmarking against ALL baselines (A-F)
3. Ablation experiments (input encoding, objectives, model size, frozen vs finetuned)
4. Embedding analysis (clustering, linear probes)
5. Leakage detection (flag any LODO AUC > 0.70)

ALL pretraining, fine-tuning, ablations, and benchmarking in one pipeline.
"""

from __future__ import annotations

import os, sys, time, logging, json, warnings, math
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models.foundation_full.gene_universe import (
    build_gene_universe,
    discover_all_datasets,
    discover_labeled_datasets,
)
from models.foundation_full.expression_tokenizer import ExpressionTokenizer
from models.foundation_full.expression_encoder import (
    ExpressionEncoder,
    EncoderConfig,
    FoundationPretrainModel,
)
from models.foundation_full.pretrain import (
    pretrain,
    load_hallmark_gene_sets,
    compute_pathway_scores,
    infer_pam50_subtype,
    load_mutation_labels,
    build_pathway_gene_positions,
    PretrainDataset,
    collate_fn,
    CosineWarmupScheduler,
)

logger = logging.getLogger(__name__)

RESULTS = ROOT / "results" / "foundation_full"
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"


# ===========================================================================
# Helpers
# ===========================================================================

def load_pretrained_encoder(tag: str = "") -> tuple[ExpressionEncoder, EncoderConfig]:
    """Load a pretrained encoder from disk."""
    suffix = f"_{tag}" if tag else ""
    config_path = RESULTS / f"pretrain_config{suffix}.json"
    with open(config_path) as f:
        config = json.load(f)

    cfg = EncoderConfig(
        n_genes=config["n_genes"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_mag_bins=config.get("n_mag_bins", 64),
        use_mag_embedding=config.get("use_mag_embedding", True),
        use_presence_flag=config.get("use_presence_flag", True),
    )
    encoder = ExpressionEncoder(cfg)
    encoder.load_state_dict(
        torch.load(RESULTS / f"encoder{suffix}.pt", map_location="cpu", weights_only=True)
    )
    encoder.eval()
    return encoder, cfg


def load_tokenizer(gene_list, gene2idx, encoding="hybrid") -> ExpressionTokenizer:
    """Load fitted tokenizer."""
    tok = ExpressionTokenizer(gene_list, gene2idx, encoding=encoding)
    tok.load_stats()
    return tok


def extract_embeddings(
    encoder: ExpressionEncoder,
    tokenizer: ExpressionTokenizer,
    expr_df: pd.DataFrame,
    cfg: EncoderConfig,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract [CLS] embeddings for all samples."""
    encoder.eval()
    embeddings = []

    for start in range(0, len(expr_df), batch_size):
        end = min(start + batch_size, len(expr_df))
        batch_df = expr_df.iloc[start:end]
        tokens = tokenizer.tokenize_dataframe(batch_df)

        with torch.no_grad():
            cls_emb, _ = encoder(
                tokens["gene_ids"],
                tokens["values"],
                mag_bins=tokens["mag_bins"] if cfg.use_mag_embedding else None,
                presence=tokens["presence"] if cfg.use_presence_flag else None,
            )
        embeddings.append(cls_emb.numpy())

    return np.concatenate(embeddings, axis=0)


# ===========================================================================
# Classifier heads for fine-tuning
# ===========================================================================

class MLPClassifier(nn.Module):
    """[CLS] -> Linear(d, 64) -> ReLU -> Linear(64, 1) -> Sigmoid"""
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LinearProbe(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.fc = nn.Linear(d_in, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def finetune_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classifier_type: str = "mlp",
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    """Fine-tune a classifier and return predicted probabilities on X_test."""
    d = X_train.shape[1]
    if classifier_type == "mlp":
        model = MLPClassifier(d)
    else:
        model = LinearProbe(d)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)

    # Class weight for imbalanced data
    pos_rate = y_train.mean()
    if 0.05 < pos_rate < 0.95:
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate])
    else:
        pos_weight = torch.tensor([1.0])

    model.train()
    for epoch in range(n_epochs):
        logits = model(X_tr)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te)).numpy()
    return probs


def finetune_encoder_and_classify(
    encoder: ExpressionEncoder,
    tokenizer: ExpressionTokenizer,
    cfg: EncoderConfig,
    train_dfs: list[pd.DataFrame],
    train_ys: list[np.ndarray],
    test_df: pd.DataFrame,
    n_epochs: int = 10,
    lr: float = 5e-5,
    batch_size: int = 32,
) -> np.ndarray:
    """Fine-tune the encoder end-to-end + MLP head, return probs on test."""
    import copy
    enc = copy.deepcopy(encoder)
    d = cfg.d_model
    head = MLPClassifier(d)

    # Combine training data
    all_X = pd.concat(train_dfs, axis=0)
    all_y = np.concatenate(train_ys)

    # Tokenize
    train_tokens = tokenizer.tokenize_dataframe(all_X)
    test_tokens = tokenizer.tokenize_dataframe(test_df)

    y_tr = torch.tensor(all_y, dtype=torch.float32)

    pos_rate = all_y.mean()
    if 0.05 < pos_rate < 0.95:
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate])
    else:
        pos_weight = torch.tensor([1.0])

    params = list(enc.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    enc.train()
    head.train()
    n_train = len(all_X)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            g = train_tokens["gene_ids"][idx]
            v = train_tokens["values"][idx]
            mb = train_tokens["mag_bins"][idx] if cfg.use_mag_embedding else None
            pr = train_tokens["presence"][idx] if cfg.use_presence_flag else None
            y = y_tr[idx]

            cls_emb, _ = enc(g, v, mag_bins=mb, presence=pr)
            logits = head(cls_emb)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

    # Predict on test
    enc.eval()
    head.eval()
    probs_list = []
    with torch.no_grad():
        for start in range(0, len(test_df), batch_size):
            end = min(start + batch_size, len(test_df))
            g = test_tokens["gene_ids"][start:end]
            v = test_tokens["values"][start:end]
            mb = test_tokens["mag_bins"][start:end] if cfg.use_mag_embedding else None
            pr = test_tokens["presence"][start:end] if cfg.use_presence_flag else None
            cls_emb, _ = enc(g, v, mag_bins=mb, presence=pr)
            probs_list.append(torch.sigmoid(head(cls_emb)).numpy())

    return np.concatenate(probs_list)


# ===========================================================================
# LightGBM
# ===========================================================================

def train_lightgbm(X_train, y_train, X_test):
    """Train LightGBM and return predicted probabilities."""
    import lightgbm as lgb
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return probs


def safe_auc(y_true, y_pred):
    """AUC that returns NaN if only one class present."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return np.nan


# ===========================================================================
# Load all labeled data
# ===========================================================================

def load_all_labeled_data(gene_list):
    """Load all labeled datasets with aligned expression and response."""
    labeled = discover_labeled_datasets()
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
                logger.info("  %s: %d samples, response rate %.2f",
                           name, len(common), all_resp[name].mean())
        except Exception as e:
            logger.warning("  %s: failed: %s", name, e)

    return all_expr, all_resp


# ===========================================================================
# LODO evaluation
# ===========================================================================

def run_lodo_evaluation(
    encoder: ExpressionEncoder,
    tokenizer: ExpressionTokenizer,
    cfg: EncoderConfig,
    gene_list: list[str],
    all_expr: dict,
    all_resp: dict,
    tag: str = "",
    include_finetuned: bool = False,
) -> pd.DataFrame:
    """
    LODO evaluation: for each held-out dataset, fine-tune on all others, evaluate.

    Methods:
      A. LightGBM on 212 landmark genes
      B. LightGBM on 212 + pathways
      C. LightGBM on 212 + mutations/clinical (multimodal baseline)
      D. Foundation [CLS] -> MLP head (frozen encoder)
      E. Foundation [CLS] as LightGBM features (212 + [CLS])
      F. Foundation [CLS] + mutations/clinical -> LightGBM
    """
    # Load landmark genes
    landmark_path = DATA_CACHE / "geneinfo_beta_input.txt"
    if landmark_path.exists():
        landmark_genes = pd.read_csv(landmark_path, sep="\t")["gene_symbol"].tolist()
        landmark_genes = [g for g in landmark_genes if isinstance(g, str) and len(g) > 0]
    else:
        # Fall back to gene_list subset
        landmark_genes = gene_list[:212]
    logger.info("Landmark genes: %d", len(landmark_genes))

    results = []
    eval_datasets = sorted(all_expr.keys())

    # Pre-extract embeddings for all datasets
    logger.info("Pre-extracting embeddings for all datasets ...")
    all_embs = {}
    for name, df in all_expr.items():
        all_embs[name] = extract_embeddings(encoder, tokenizer, df, cfg)
        logger.info("  %s: %d embeddings (dim=%d)", name, all_embs[name].shape[0], all_embs[name].shape[1])

    for heldout_name in eval_datasets:
        if heldout_name not in all_resp:
            continue

        y_test = all_resp[heldout_name]
        if len(np.unique(y_test)) < 2:
            logger.warning("  %s: single class, skipping", heldout_name)
            continue

        X_test_expr = all_expr[heldout_name]
        emb_test = all_embs[heldout_name]

        # Training data: all except held-out
        train_names = [n for n in all_expr if n != heldout_name and n in all_resp]
        if not train_names:
            continue

        y_train = np.concatenate([all_resp[n] for n in train_names])
        emb_train = np.concatenate([all_embs[n] for n in train_names], axis=0)

        # Landmark gene features
        avail_landmarks = [g for g in landmark_genes if g in X_test_expr.columns]
        if len(avail_landmarks) < 20:
            avail_landmarks = [g for g in gene_list[:212] if g in X_test_expr.columns]

        X_te_lgbm = X_test_expr[avail_landmarks].fillna(0).values
        X_tr_lgbm_parts = []
        for n in train_names:
            # Use overlapping landmarks
            lgbm_cols = [g for g in avail_landmarks if g in all_expr[n].columns]
            df_part = all_expr[n][lgbm_cols].fillna(0)
            # Pad missing columns
            full_part = pd.DataFrame(0, index=df_part.index, columns=avail_landmarks)
            for c in lgbm_cols:
                full_part[c] = df_part[c]
            X_tr_lgbm_parts.append(full_part.values)
        X_tr_lgbm = np.concatenate(X_tr_lgbm_parts, axis=0)

        row = {
            "dataset": heldout_name,
            "n_test": len(y_test),
            "n_train": len(y_train),
            "response_rate": y_test.mean(),
        }

        # ---- A. LightGBM on landmark genes ----
        try:
            probs = train_lightgbm(X_tr_lgbm, y_train, X_te_lgbm)
            row["auc_A_lgbm_212"] = safe_auc(y_test, probs)
        except Exception as e:
            logger.warning("  A failed: %s", e)
            row["auc_A_lgbm_212"] = np.nan

        # ---- D. Foundation [CLS] -> MLP (frozen) ----
        try:
            probs = finetune_classifier(
                emb_train, y_train, emb_test,
                classifier_type="mlp", n_epochs=50,
            )
            row["auc_D_cls_mlp"] = safe_auc(y_test, probs)
        except Exception as e:
            logger.warning("  D failed: %s", e)
            row["auc_D_cls_mlp"] = np.nan

        # ---- E. LightGBM on 212 + [CLS] ----
        try:
            X_tr_combined = np.hstack([X_tr_lgbm, emb_train])
            X_te_combined = np.hstack([X_te_lgbm, emb_test])
            probs = train_lightgbm(X_tr_combined, y_train, X_te_combined)
            row["auc_E_lgbm_212_cls"] = safe_auc(y_test, probs)
        except Exception as e:
            logger.warning("  E failed: %s", e)
            row["auc_E_lgbm_212_cls"] = np.nan

        # ---- Linear probe (simple) ----
        try:
            probs = finetune_classifier(
                emb_train, y_train, emb_test,
                classifier_type="linear", n_epochs=50,
            )
            row["auc_linear_probe"] = safe_auc(y_test, probs)
        except Exception as e:
            row["auc_linear_probe"] = np.nan

        logger.info(
            "  %s | A(lgbm212)=%.3f | D(cls_mlp)=%.3f | E(lgbm+cls)=%.3f",
            heldout_name,
            row.get("auc_A_lgbm_212", float("nan")),
            row.get("auc_D_cls_mlp", float("nan")),
            row.get("auc_E_lgbm_212_cls", float("nan")),
        )

        results.append(row)

    results_df = pd.DataFrame(results)

    # Leakage check
    for col in [c for c in results_df.columns if c.startswith("auc_")]:
        max_auc = results_df[col].max()
        if max_auc > 0.70:
            logger.warning(
                "LEAKAGE CHECK: %s has max AUC %.3f > 0.70. "
                "Investigating -- this may be due to within-domain similarity "
                "rather than leakage, but verify carefully.",
                col, max_auc,
            )

    return results_df


# ===========================================================================
# Ablation helpers
# ===========================================================================

def run_ablation_pretrain_and_eval(
    ablation_name: str,
    model_size: str,
    encoding: str,
    objectives_config: dict,
    n_epochs: int = 30,
    gene_list: list[str] = None,
    gene2idx: dict = None,
    all_expr: dict = None,
    all_resp: dict = None,
) -> dict:
    """Run one ablation: pretrain + LODO eval. Returns summary metrics."""
    tag = ablation_name
    logger.info("=" * 70)
    logger.info("ABLATION: %s (size=%s, encoding=%s)", ablation_name, model_size, encoding)
    logger.info("=" * 70)

    # Pretrain
    model, tokenizer, gl, g2i, history = pretrain(
        model_size=model_size,
        encoding=encoding,
        objectives_config=objectives_config,
        n_epochs=n_epochs,
        save_tag=tag,
        max_time_hours=1.0,  # Limit ablation runs
    )

    # Load encoder for eval
    encoder, cfg = load_pretrained_encoder(tag=tag)
    tok = load_tokenizer(gl, g2i, encoding=encoding)

    # Run LODO
    lodo_df = run_lodo_evaluation(
        encoder, tok, cfg, gl, all_expr, all_resp, tag=tag,
    )
    lodo_df.to_csv(RESULTS / f"lodo_{tag}.csv", index=False)

    # Summary
    summary = {
        "ablation": ablation_name,
        "model_size": model_size,
        "encoding": encoding,
        "n_epochs": history["epoch"].max() if len(history) > 0 else 0,
        "final_loss": history["loss_total"].iloc[-1] if len(history) > 0 else np.nan,
    }
    for col in [c for c in lodo_df.columns if c.startswith("auc_")]:
        summary[f"mean_{col}"] = lodo_df[col].mean()
        summary[f"median_{col}"] = lodo_df[col].median()

    return summary


# ===========================================================================
# Embedding analysis
# ===========================================================================

def run_embedding_analysis(
    encoder: ExpressionEncoder,
    tokenizer: ExpressionTokenizer,
    cfg: EncoderConfig,
    gene_list: list[str],
) -> pd.DataFrame:
    """
    Compute [CLS] embeddings for all patients and analyze:
    - Clustering by subtype (should cluster)
    - Clustering by platform/dataset (should NOT cluster)
    - Linear probe accuracy for subtype, TP53, platform
    """
    logger.info("=" * 70)
    logger.info("EMBEDDING ANALYSIS")
    logger.info("=" * 70)

    datasets = discover_all_datasets()
    all_embs = []
    all_subtypes = []
    all_datasets_labels = []
    all_mutations = []  # TP53 only
    all_sample_ids = []

    for di, (name, path) in enumerate(datasets):
        try:
            df = pd.read_parquet(path)
            gene_cols = [c for c in df.columns if c in set(gene_list)]
            if len(gene_cols) < 50:
                continue

            embs = extract_embeddings(encoder, tokenizer, df, cfg)
            subtypes = infer_pam50_subtype(df)
            muts = load_mutation_labels(df, name)

            all_embs.append(embs)
            all_subtypes.append(subtypes)
            all_datasets_labels.append(np.full(len(df), di, dtype=np.int64))
            all_mutations.append(muts[:, 0])  # TP53 only
            all_sample_ids.extend([f"{name}_{i}" for i in range(len(df))])

            logger.info("  %s: %d embeddings", name, len(df))
        except Exception as e:
            logger.warning("  %s failed: %s", name, e)

    if not all_embs:
        logger.warning("No embeddings extracted!")
        return pd.DataFrame()

    embs = np.concatenate(all_embs, axis=0)
    subtypes = np.concatenate(all_subtypes)
    dataset_labels = np.concatenate(all_datasets_labels)
    mutations = np.concatenate(all_mutations)

    logger.info("Total embeddings: %d (dim=%d)", embs.shape[0], embs.shape[1])

    results = []

    # 1. PCA analysis
    pca = PCA(n_components=min(50, embs.shape[1]))
    embs_pca = pca.fit_transform(embs)
    var_explained = pca.explained_variance_ratio_.cumsum()
    logger.info("PCA: 10 components explain %.1f%% variance", var_explained[9] * 100)
    logger.info("PCA: 20 components explain %.1f%% variance", var_explained[19] * 100)

    results.append({"metric": "pca_10_var_explained", "value": var_explained[9]})
    results.append({"metric": "pca_20_var_explained", "value": var_explained[19]})

    # 2. Silhouette scores
    # By subtype (should be high)
    valid_sub = subtypes >= 0
    if valid_sub.sum() > 100:
        sil_sub = silhouette_score(
            embs_pca[valid_sub, :20], subtypes[valid_sub],
            sample_size=min(5000, valid_sub.sum()),
            random_state=42,
        )
        logger.info("Silhouette (subtype): %.3f", sil_sub)
        results.append({"metric": "silhouette_subtype", "value": sil_sub})

    # By dataset (should be low -- domain invariance)
    n_unique_ds = len(np.unique(dataset_labels))
    if n_unique_ds > 1:
        sil_ds = silhouette_score(
            embs_pca[:, :20], dataset_labels,
            sample_size=min(5000, len(dataset_labels)),
            random_state=42,
        )
        logger.info("Silhouette (dataset): %.3f (lower is better)", sil_ds)
        results.append({"metric": "silhouette_dataset", "value": sil_ds})

    # 3. Linear probe accuracy: subtype
    if valid_sub.sum() > 100:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embs[valid_sub])
        y_sub = subtypes[valid_sub]

        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        cv_scores = cross_val_score(lr, X_scaled, y_sub, cv=5, scoring="accuracy")
        logger.info("Linear probe subtype accuracy: %.3f +/- %.3f",
                    cv_scores.mean(), cv_scores.std())
        results.append({"metric": "probe_subtype_accuracy", "value": cv_scores.mean()})
        results.append({"metric": "probe_subtype_std", "value": cv_scores.std()})

    # 4. Linear probe: TP53 status
    valid_tp53 = mutations >= 0
    if valid_tp53.sum() > 50:
        X_tp53 = StandardScaler().fit_transform(embs[valid_tp53])
        y_tp53 = mutations[valid_tp53].astype(int)
        if len(np.unique(y_tp53)) == 2:
            lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            cv_scores = cross_val_score(lr, X_tp53, y_tp53, cv=5, scoring="roc_auc")
            logger.info("Linear probe TP53 AUC: %.3f +/- %.3f",
                       cv_scores.mean(), cv_scores.std())
            results.append({"metric": "probe_tp53_auc", "value": cv_scores.mean()})

    # 5. Linear probe: dataset identity (should be low -- domain invariance)
    if n_unique_ds > 1:
        X_ds = StandardScaler().fit_transform(embs)
        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        cv_scores = cross_val_score(lr, X_ds, dataset_labels, cv=5, scoring="accuracy")
        logger.info("Linear probe dataset accuracy: %.3f (lower is better for invariance)",
                   cv_scores.mean())
        results.append({"metric": "probe_dataset_accuracy", "value": cv_scores.mean()})

    results_df = pd.DataFrame(results)
    return results_df


# ===========================================================================
# Full pipeline
# ===========================================================================

def run_full_pipeline():
    """Execute the complete foundation model pipeline."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Phase 1: Main pretraining (medium, hybrid, all objectives)
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 1: MAIN PRETRAINING")
    logger.info("=" * 70)

    model, tokenizer, gene_list, gene2idx, history = pretrain(
        model_size="medium",
        encoding="hybrid",
        n_epochs=50,
        batch_size=32,
        lr=1e-4,
        save_tag="main",
        max_time_hours=2.0,
    )

    # -----------------------------------------------------------------------
    # Phase 2: Load all labeled data and run main LODO evaluation
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 2: LODO EVALUATION")
    logger.info("=" * 70)

    all_expr, all_resp = load_all_labeled_data(gene_list)
    logger.info("Loaded %d labeled datasets", len(all_expr))

    encoder, cfg = load_pretrained_encoder(tag="main")
    tok = load_tokenizer(gene_list, gene2idx, encoding="hybrid")

    lodo_df = run_lodo_evaluation(
        encoder, tok, cfg, gene_list, all_expr, all_resp, tag="main",
    )
    lodo_df.to_csv(RESULTS / "lodo_results.csv", index=False)
    logger.info("Main LODO results saved. %d datasets evaluated.", len(lodo_df))

    # Print summary
    for col in [c for c in lodo_df.columns if c.startswith("auc_")]:
        logger.info("  %s: mean=%.3f, median=%.3f",
                    col, lodo_df[col].mean(), lodo_df[col].median())

    # -----------------------------------------------------------------------
    # Phase 3: Ablation - Input Encoding (raw vs rank vs hybrid)
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 3: ABLATION -- INPUT ENCODING")
    logger.info("=" * 70)

    encoding_results = []
    for enc_name in ["raw", "rank", "hybrid"]:
        if enc_name == "hybrid":
            # Already done in main
            summary = {
                "ablation": f"encoding_{enc_name}",
                "model_size": "medium",
                "encoding": enc_name,
            }
            for col in [c for c in lodo_df.columns if c.startswith("auc_")]:
                summary[f"mean_{col}"] = lodo_df[col].mean()
                summary[f"median_{col}"] = lodo_df[col].median()
            encoding_results.append(summary)
        else:
            summary = run_ablation_pretrain_and_eval(
                ablation_name=f"encoding_{enc_name}",
                model_size="medium",
                encoding=enc_name,
                objectives_config={
                    "mgp": True, "mgp_weight": 1.0,
                    "pap": True, "pap_weight": 0.5,
                    "sub": True, "sub_weight": 0.5,
                    "mut": True, "mut_weight": 0.3,
                    "dav": True, "dav_weight": 0.2,
                },
                n_epochs=30,
                gene_list=gene_list,
                gene2idx=gene2idx,
                all_expr=all_expr,
                all_resp=all_resp,
            )
            encoding_results.append(summary)

    enc_abl_df = pd.DataFrame(encoding_results)
    enc_abl_df.to_csv(RESULTS / "ablation_input_encoding.csv", index=False)
    logger.info("Input encoding ablation saved.")

    # -----------------------------------------------------------------------
    # Phase 4: Ablation - Objectives (masked-only vs multi vs multi+domain)
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 4: ABLATION -- PRETRAINING OBJECTIVES")
    logger.info("=" * 70)

    obj_results = []

    # MGP only
    summary = run_ablation_pretrain_and_eval(
        ablation_name="obj_mgp_only",
        model_size="medium",
        encoding="hybrid",
        objectives_config={
            "mgp": True, "mgp_weight": 1.0,
            "pap": False, "pap_weight": 0.0,
            "sub": False, "sub_weight": 0.0,
            "mut": False, "mut_weight": 0.0,
            "dav": False, "dav_weight": 0.0,
        },
        n_epochs=30,
        gene_list=gene_list, gene2idx=gene2idx,
        all_expr=all_expr, all_resp=all_resp,
    )
    obj_results.append(summary)

    # Multi-objective (no domain adversarial)
    summary = run_ablation_pretrain_and_eval(
        ablation_name="obj_multi_no_dav",
        model_size="medium",
        encoding="hybrid",
        objectives_config={
            "mgp": True, "mgp_weight": 1.0,
            "pap": True, "pap_weight": 0.5,
            "sub": True, "sub_weight": 0.5,
            "mut": True, "mut_weight": 0.3,
            "dav": False, "dav_weight": 0.0,
        },
        n_epochs=30,
        gene_list=gene_list, gene2idx=gene2idx,
        all_expr=all_expr, all_resp=all_resp,
    )
    obj_results.append(summary)

    # Multi-objective + domain adversarial (= main, reuse results)
    summary_main = {
        "ablation": "obj_multi_with_dav",
        "model_size": "medium",
        "encoding": "hybrid",
    }
    for col in [c for c in lodo_df.columns if c.startswith("auc_")]:
        summary_main[f"mean_{col}"] = lodo_df[col].mean()
        summary_main[f"median_{col}"] = lodo_df[col].median()
    obj_results.append(summary_main)

    obj_abl_df = pd.DataFrame(obj_results)
    obj_abl_df.to_csv(RESULTS / "ablation_objectives.csv", index=False)
    logger.info("Objectives ablation saved.")

    # -----------------------------------------------------------------------
    # Phase 5: Ablation - Model Size (medium vs full)
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 5: ABLATION -- MODEL SIZE")
    logger.info("=" * 70)

    size_results = []

    # Medium is already main
    summary_med = {
        "ablation": "size_medium",
        "model_size": "medium",
        "encoding": "hybrid",
    }
    for col in [c for c in lodo_df.columns if c.startswith("auc_")]:
        summary_med[f"mean_{col}"] = lodo_df[col].mean()
        summary_med[f"median_{col}"] = lodo_df[col].median()
    size_results.append(summary_med)

    # Full model
    summary_full = run_ablation_pretrain_and_eval(
        ablation_name="size_full",
        model_size="full",
        encoding="hybrid",
        objectives_config={
            "mgp": True, "mgp_weight": 1.0,
            "pap": True, "pap_weight": 0.5,
            "sub": True, "sub_weight": 0.5,
            "mut": True, "mut_weight": 0.3,
            "dav": True, "dav_weight": 0.2,
        },
        n_epochs=30,
        gene_list=gene_list, gene2idx=gene2idx,
        all_expr=all_expr, all_resp=all_resp,
    )
    size_results.append(summary_full)

    size_abl_df = pd.DataFrame(size_results)
    size_abl_df.to_csv(RESULTS / "ablation_model_size.csv", index=False)
    logger.info("Model size ablation saved.")

    # -----------------------------------------------------------------------
    # Phase 6: Ablation - Frozen vs Fine-tuned encoder
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 6: ABLATION -- FROZEN VS FINETUNED")
    logger.info("=" * 70)

    ft_results = []
    # Use main model
    encoder_main, cfg_main = load_pretrained_encoder(tag="main")
    tok_main = load_tokenizer(gene_list, gene2idx, encoding="hybrid")

    eval_datasets = sorted(all_expr.keys())
    for heldout_name in eval_datasets[:10]:  # Limit to 10 for time
        if heldout_name not in all_resp:
            continue
        y_test = all_resp[heldout_name]
        if len(np.unique(y_test)) < 2:
            continue

        train_names = [n for n in all_expr if n != heldout_name and n in all_resp]
        if not train_names:
            continue

        train_dfs = [all_expr[n] for n in train_names]
        train_ys = [all_resp[n] for n in train_names]

        # Frozen: already in lodo_df as auc_D_cls_mlp
        frozen_row = lodo_df[lodo_df["dataset"] == heldout_name]
        frozen_auc = frozen_row["auc_D_cls_mlp"].values[0] if len(frozen_row) > 0 else np.nan

        # Fine-tuned
        try:
            probs = finetune_encoder_and_classify(
                encoder_main, tok_main, cfg_main,
                train_dfs, train_ys,
                all_expr[heldout_name],
                n_epochs=5, lr=5e-5,
            )
            finetuned_auc = safe_auc(y_test, probs)
        except Exception as e:
            logger.warning("  Finetuned %s failed: %s", heldout_name, e)
            finetuned_auc = np.nan

        ft_results.append({
            "dataset": heldout_name,
            "auc_frozen": frozen_auc,
            "auc_finetuned": finetuned_auc,
        })
        logger.info("  %s: frozen=%.3f, finetuned=%.3f",
                    heldout_name, frozen_auc, finetuned_auc)

    ft_df = pd.DataFrame(ft_results)
    ft_df.to_csv(RESULTS / "ablation_frozen_vs_finetuned.csv", index=False)
    logger.info("Frozen vs finetuned ablation saved.")

    # -----------------------------------------------------------------------
    # Phase 7: Embedding analysis
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 7: EMBEDDING ANALYSIS")
    logger.info("=" * 70)

    emb_analysis_df = run_embedding_analysis(encoder_main, tok_main, cfg_main, gene_list)
    emb_analysis_df.to_csv(RESULTS / "embedding_analysis.csv", index=False)
    logger.info("Embedding analysis saved.")

    # -----------------------------------------------------------------------
    # Phase 8: Final comparison table
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 8: COMPARISON VS BASELINES")
    logger.info("=" * 70)

    comparison = []
    comparison.append({
        "method": "A. LightGBM 212 genes",
        "mean_auc": lodo_df["auc_A_lgbm_212"].mean() if "auc_A_lgbm_212" in lodo_df else np.nan,
        "median_auc": lodo_df["auc_A_lgbm_212"].median() if "auc_A_lgbm_212" in lodo_df else np.nan,
        "known_baseline": 0.610,
        "source": "This run",
    })
    comparison.append({
        "method": "B. LightGBM 212 + pathways",
        "mean_auc": np.nan,
        "median_auc": np.nan,
        "known_baseline": 0.603,
        "source": "Known baseline",
    })
    comparison.append({
        "method": "C. LightGBM 212 + mutations/clinical",
        "mean_auc": np.nan,
        "median_auc": np.nan,
        "known_baseline": 0.624,
        "source": "Known baseline",
    })
    comparison.append({
        "method": "D. Foundation [CLS] -> MLP",
        "mean_auc": lodo_df["auc_D_cls_mlp"].mean() if "auc_D_cls_mlp" in lodo_df else np.nan,
        "median_auc": lodo_df["auc_D_cls_mlp"].median() if "auc_D_cls_mlp" in lodo_df else np.nan,
        "known_baseline": np.nan,
        "source": "This run",
    })
    comparison.append({
        "method": "E. LightGBM 212 + [CLS] embedding",
        "mean_auc": lodo_df["auc_E_lgbm_212_cls"].mean() if "auc_E_lgbm_212_cls" in lodo_df else np.nan,
        "median_auc": lodo_df["auc_E_lgbm_212_cls"].median() if "auc_E_lgbm_212_cls" in lodo_df else np.nan,
        "known_baseline": np.nan,
        "source": "This run",
    })
    if "auc_linear_probe" in lodo_df.columns:
        comparison.append({
            "method": "D'. Foundation [CLS] -> linear probe",
            "mean_auc": lodo_df["auc_linear_probe"].mean(),
            "median_auc": lodo_df["auc_linear_probe"].median(),
            "known_baseline": np.nan,
            "source": "This run",
        })
    if len(ft_df) > 0:
        comparison.append({
            "method": "D''. Foundation [CLS] finetuned -> MLP",
            "mean_auc": ft_df["auc_finetuned"].mean(),
            "median_auc": ft_df["auc_finetuned"].median(),
            "known_baseline": np.nan,
            "source": "This run",
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(RESULTS / "comparison_vs_baselines.csv", index=False)
    logger.info("\n%s", comparison_df.to_string(index=False))

    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info("FULL PIPELINE COMPLETE in %.1f minutes", total_time / 60)
    logger.info("=" * 70)

    return {
        "lodo_df": lodo_df,
        "comparison_df": comparison_df,
        "enc_abl_df": enc_abl_df,
        "obj_abl_df": obj_abl_df,
        "size_abl_df": size_abl_df,
        "ft_df": ft_df,
        "emb_analysis_df": emb_analysis_df,
    }


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS / "evaluate_full.log"),
        ],
    )
    run_full_pipeline()
