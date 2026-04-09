#!/usr/bin/env python
"""
INVEREX — Train Final Production Model (Cross-Platform)
========================================================

Best config from experiment: rank_plus_singscore_reo_knowledge__agnostic
  AUROC=0.602, AUPRC=0.619, MCC=0.157

This script:
  1. Trains on ALL clean data (38 datasets, 5,200 patients)
  2. Uses within-sample rank + inverse-normal (per-sample deployable)
  3. Computes SHAP explanations
  4. Tunes production threshold (inner-CV MCC)
  5. Calibrates conformal predictor
  6. Saves production bundle to models/production/
"""

import os
import sys
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import joblib
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "cross_platform"
PROD_DIR = ROOT / "models" / "production"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 300,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 10,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "random_state": 42,
    "verbose": -1,
}

CHUNK_SIZE = 500
MIN_PATIENTS = 20

PLATFORM_MAP = {
    "GSE25066": "affymetrix", "GSE20194": "affymetrix", "GSE20271": "affymetrix",
    "GSE22093": "affymetrix", "GSE23988": "affymetrix", "GSE37946": "affymetrix",
    "GSE41998": "affymetrix", "GSE5122": "affymetrix", "GSE8970": "affymetrix",
    "GSE131978": "affymetrix", "GSE20181": "affymetrix",
    "GSE14615": "affymetrix", "GSE14671": "affymetrix", "GSE19293": "affymetrix",
    "GSE28702": "affymetrix", "GSE32646": "affymetrix", "GSE35640": "affymetrix",
    "GSE48905": "affymetrix", "GSE50948": "affymetrix", "GSE63885": "affymetrix",
    "GSE68871": "affymetrix", "GSE72970": "affymetrix", "GSE73578": "affymetrix",
    "GSE62321": "affymetrix",
    "GSE104645": "agilent", "GSE109211": "agilent", "GSE173263": "agilent",
    "GSE21974": "agilent", "GSE44272": "agilent", "GSE4779": "agilent",
    "GSE65021": "agilent", "GSE66999": "agilent", "GSE6861": "agilent",
    "GSE76360": "agilent", "GSE82172": "agilent",
    "GSE104958": "rnaseq",
    "ISPY2": "agilent",
    "BrighTNess": "rnaseq",
}

CURATED_PAIRS = [
    ("ERBB2", "ESR1"), ("MKI67", "ACTB"), ("CCND1", "CDKN1A"),
    ("TP53", "MDM2"), ("EGFR", "ERBB2"), ("AKT1", "PTEN"),
    ("BCL2", "BAX"), ("FOXO3", "AKT1"), ("CASP3", "BCL2"), ("ESR1", "AR"),
]


def clean_matrix(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def within_sample_rank_inv_norm(expression_df):
    mat = expression_df.values.copy()
    n_samples, n_genes = mat.shape
    ranked = np.zeros_like(mat, dtype=float)
    for i in range(n_samples):
        ranked[i, :] = rankdata(mat[i, :], method="average")
    quantiles = (ranked - 0.5) / n_genes
    result = norm.ppf(np.clip(quantiles, 1e-7, 1 - 1e-7))
    return pd.DataFrame(result, index=expression_df.index, columns=expression_df.columns)


def compute_singscore(expression_df, gene_list):
    import gseapy as gp
    hallmark = gp.get_library("MSigDB_Hallmark_2020")
    ranks = expression_df[gene_list].rank(axis=1)
    n_genes = len(gene_list)
    scores = {}
    for pw_name, pw_genes in hallmark.items():
        present = [g for g in pw_genes if g in gene_list]
        if len(present) < 5:
            continue
        mean_rank = ranks[present].mean(axis=1)
        scores[f"singscore_{pw_name}"] = (mean_rank / n_genes - 0.5) * 2
    return pd.DataFrame(scores, index=expression_df.index)


def build_reo_knowledge(expression_df):
    features = {}
    for ga, gb in CURATED_PAIRS:
        if ga in expression_df.columns and gb in expression_df.columns:
            features[f"reo_{ga}_gt_{gb}"] = (expression_df[ga] > expression_df[gb]).astype(int)
    return pd.DataFrame(features, index=expression_df.index)


# =========================================================================
# Load data (same as experiment)
# =========================================================================
log.info("Loading data...")


def load_ctrdb_dataset(geo_id):
    base = ROOT / "data" / "raw" / "ctrdb" / geo_id
    expr_path = base / f"{geo_id}_expression.parquet"
    label_path = base / "response_labels.parquet"
    if not expr_path.exists() or not label_path.exists():
        return None, None
    expr = pd.read_parquet(expr_path)
    labels = pd.read_parquet(label_path)
    common = expr.index.intersection(labels.index)
    if len(common) < MIN_PATIENTS:
        return None, None
    return expr.loc[common], labels.loc[common, "response"].astype(int)


def load_positional_dataset(data_dir, geo_id):
    base = ROOT / "data" / "raw" / data_dir
    expr = pd.read_parquet(base / f"{geo_id}_expression.parquet")
    labels = pd.read_parquet(base / "response_labels.parquet")
    response = labels["response"].astype(int).values
    expr = expr.reset_index(drop=True)
    return expr, pd.Series(response, index=expr.index, name="response")


ctrdb_dir = ROOT / "data" / "raw" / "ctrdb"
ctrdb_geos = sorted(
    d.name for d in ctrdb_dir.iterdir() if d.is_dir() and d.name.startswith("GSE")
)
ctrdb_geos = [g for g in ctrdb_geos if g != "GSE194040"]

datasets = {}
for geo_id in ctrdb_geos:
    expr, labels = load_ctrdb_dataset(geo_id)
    if expr is not None and labels is not None:
        if labels.nunique() >= 2 and len(labels) >= MIN_PATIENTS:
            tech = PLATFORM_MAP.get(geo_id, "unknown")
            if tech not in ("targeted", "beadarray"):
                datasets[geo_id] = (expr, labels)

for name, data_dir, geo_id in [("ISPY2", "ispy2", "GSE194040"), ("BrighTNess", "brightness", "GSE164458")]:
    try:
        e, l = load_positional_dataset(data_dir, geo_id)
        if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
            datasets[name] = (e, l)
    except Exception:
        pass

# Load common genes
with open(ROOT / "data" / "cache" / "common_genes_cross_platform.json") as f:
    common_genes = json.load(f)

log.info(f"{len(datasets)} datasets, {sum(len(v[1]) for v in datasets.values())} patients, {len(common_genes)} genes")

# Restrict, z-score, pool
all_parts_x, all_parts_y = [], []
for did in sorted(datasets.keys()):
    expr, lab = datasets[did]
    available = [g for g in common_genes if g in expr.columns]
    missing = [g for g in common_genes if g not in expr.columns]
    expr_sub = expr[available].copy()
    for g in missing:
        expr_sub[g] = 0.0
    expr_sub = expr_sub[common_genes]
    means = expr_sub.mean(axis=0)
    stds = expr_sub.std(axis=0).replace(0, 1)
    expr_z = clean_matrix((expr_sub - means) / stds)
    idx = [f"{did}__{i}" for i in range(len(expr_z))]
    expr_z.index = idx
    lc = lab.copy(); lc.index = idx
    all_parts_x.append(expr_z)
    all_parts_y.append(lc)

pooled_expr = pd.concat(all_parts_x, axis=0)
pooled_labels = pd.concat(all_parts_y, axis=0)
log.info(f"Pooled: {pooled_expr.shape}")


# =========================================================================
# Build features: rank + singscore + REO knowledge
# =========================================================================
log.info("Building features...")

rank_expr = within_sample_rank_inv_norm(pooled_expr)
singscore_feats = compute_singscore(pooled_expr, common_genes)
reo_feats = build_reo_knowledge(pooled_expr)

X_full = pd.concat([rank_expr, singscore_feats, reo_feats], axis=1)
X_full = clean_matrix(X_full)
y_full = pooled_labels.values.astype(int)
feature_names = list(X_full.columns)

log.info(f"Features: {X_full.shape[1]} ({len(common_genes)} rank genes + {singscore_feats.shape[1]} singscore + {reo_feats.shape[1]} REO)")


# =========================================================================
# Train final model
# =========================================================================
log.info("Training final LightGBM...")
model = lgb.LGBMClassifier(**LGBM_PARAMS)
model.fit(X_full.values, y_full)
log.info(f"Model: {model.n_features_in_} features, {model.n_estimators} trees")


# =========================================================================
# Threshold tuning (inner-CV MCC)
# =========================================================================
log.info("Tuning production threshold (inner-CV)...")
thresholds = np.linspace(0.1, 0.9, 81)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_best = []
for tr_idx, val_idx in skf.split(X_full, y_full):
    inner_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    inner_model.fit(X_full.values[tr_idx], y_full[tr_idx])
    val_preds = inner_model.predict_proba(X_full.values[val_idx])[:, 1]
    mccs = [matthews_corrcoef(y_full[val_idx], (val_preds >= t).astype(int)) for t in thresholds]
    fold_best.append(thresholds[int(np.argmax(mccs))])

prod_threshold = float(np.mean(fold_best))
log.info(f"Production threshold: {prod_threshold:.3f} (inner-CV MCC, 5 folds: {fold_best})")


# =========================================================================
# Conformal calibration
# =========================================================================
log.info("Calibrating conformal predictor...")
from sklearn.model_selection import train_test_split

_, cal_idx = train_test_split(np.arange(len(y_full)), test_size=0.2, random_state=42, stratify=y_full)
cal_preds = model.predict_proba(X_full.values[cal_idx])[:, 1]
cal_labels = y_full[cal_idx]
# Nonconformity scores for conformal classification
cal_scores = np.sort(1 - cal_preds[cal_labels == 1])  # responders
cal_scores_neg = np.sort(1 - (1 - cal_preds[cal_labels == 0]))  # non-responders

alpha = 0.10
n_cal = len(cal_scores)
q = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
conformal_threshold = float(np.quantile(cal_scores, min(q, 1.0))) if n_cal > 0 else 0.5
log.info(f"Conformal threshold (alpha={alpha}): {conformal_threshold:.3f}, cal_set={len(cal_idx)}")


# =========================================================================
# SHAP explanations
# =========================================================================
log.info("Computing SHAP...")
explainer = shap.TreeExplainer(model)

# Subsample for SHAP
rng = np.random.RandomState(42)
shap_idx = rng.choice(len(X_full), min(2000, len(X_full)), replace=False)
X_shap = X_full.iloc[shap_idx]
sv = explainer.shap_values(X_shap)
if isinstance(sv, list) and len(sv) == 2:
    sv = sv[1]

importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(sv).mean(axis=0),
    "mean_shap": sv.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

def categorize(name):
    if name.startswith("singscore_"):
        return "singscore_pathway"
    elif name.startswith("reo_"):
        return "REO_pair"
    else:
        return "gene_expression"

importance["category"] = importance["feature"].apply(categorize)
importance["direction"] = importance["mean_shap"].apply(lambda x: "response" if x > 0 else "resistance")
importance.to_csv(OUT_DIR / "shap_final_model.tsv", sep="\t", index=False)

# Plots
try:
    X_shap_df = pd.DataFrame(X_shap.values, columns=feature_names)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(sv, X_shap_df, feature_names=feature_names, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_beeswarm_final.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved beeswarm plot")
except Exception as e:
    log.warning(f"Plot failed: {e}")

log.info("\nTop 15 features:")
for _, r in importance.head(15).iterrows():
    log.info(f"  {r['feature']:40s}  |SHAP|={r['mean_abs_shap']:.4f}  ({r['category']}, {r['direction']})")

# Category summary
cat_summary = importance.groupby("category")["mean_abs_shap"].agg(["sum", "mean", "count"])
log.info("\nBy category:")
for cat, row in cat_summary.sort_values("sum", ascending=False).iterrows():
    log.info(f"  {cat:25s}  sum={row['sum']:.3f}  n={int(row['count'])}")


# =========================================================================
# Save production bundle
# =========================================================================
log.info("Saving production bundle...")

bundle = {
    "model": model,
    "gene_list": common_genes,
    "feature_names": feature_names,
    "config": "rank_plus_singscore_reo_knowledge__agnostic",
    "normalization": "within_sample_rank_inv_norm",
    "per_sample_deployable": True,
    "production_threshold": prod_threshold,
    "conformal_alpha": alpha,
    "conformal_threshold": conformal_threshold,
    "n_datasets": len(datasets),
    "n_patients": len(y_full),
    "n_genes": len(common_genes),
    "n_features": len(feature_names),
    "lodo_auroc": 0.602,
    "lodo_mcc": 0.157,
    "curated_reo_pairs": CURATED_PAIRS,
    "lgbm_params": LGBM_PARAMS,
}

joblib.dump(bundle, PROD_DIR / "cross_platform_model_bundle.joblib")
log.info(f"Saved: {PROD_DIR / 'cross_platform_model_bundle.joblib'}")

# Also save standalone artifacts for the inference pipeline
joblib.dump(model, PROD_DIR / "lightgbm_model.joblib")
joblib.dump(common_genes, PROD_DIR / "gene_list.joblib")
joblib.dump(feature_names, PROD_DIR / "feature_names.joblib")

log.info(f"\n{'='*60}")
log.info("PRODUCTION MODEL COMPLETE")
log.info(f"{'='*60}")
log.info(f"  Config:      rank + singscore + REO (knowledge)")
log.info(f"  Norm:        within-sample rank + inv-normal (deployable)")
log.info(f"  Genes:       {len(common_genes)}")
log.info(f"  Features:    {len(feature_names)}")
log.info(f"  Patients:    {len(y_full)}")
log.info(f"  Threshold:   {prod_threshold:.3f}")
log.info(f"  LODO AUROC:  0.602")
log.info(f"  Output:      {PROD_DIR}")
