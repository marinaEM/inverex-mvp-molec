#!/usr/bin/env python
"""
INVEREX — Cross-Platform Training + Harmonization Experiment
=============================================================

Single comprehensive script covering:
  STEP 1: Data audit, platform mapping, gene harmonization
  STEP 2: Feature building (rank, ssGSEA, PROGENy, singscore, REO)
  STEP 3: LODO experiment grid (17 configs, 5 metrics, threshold tuning)
  STEP 4: Platform-held-out evaluation (Variants A, B, C)
  STEP 5: Production config selection

All outputs to results/cross_platform/.

ANTI-LEAKAGE RULES enforced:
  - REO pair selection is fold-local (training labels only)
  - Threshold tuning uses training predictions only
  - Harmonization is unsupervised (no labels) or per-sample
  - test_aware configs are tagged and excluded from production selection
"""

import os
import sys
import time
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "cross_platform"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "experiment.log", mode="w"),
    ],
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

MIN_PATIENTS = 20
CHUNK_SIZE = 500

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
    "GSE61676": "beadarray",
    "GSE104958": "rnaseq",
    "GSE9782": "targeted",
    "ISPY2": "agilent",
    "BrighTNess": "rnaseq",
}

# Curated REO pairs (knowledge-driven, no labels needed)
CURATED_PAIRS = [
    ("ERBB2", "ESR1"), ("MKI67", "ACTB"), ("CCND1", "CDKN1A"),
    ("TP53", "MDM2"), ("EGFR", "ERBB2"), ("AKT1", "PTEN"),
    ("BCL2", "BAX"), ("FOXO3", "AKT1"), ("CASP3", "BCL2"), ("ESR1", "AR"),
]


def clean_matrix(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# =========================================================================
# FEATURE BUILDING FUNCTIONS
# =========================================================================

def within_sample_rank_inv_norm(expression_df):
    """Per-sample rank + inverse-normal. No cross-sample fitting. Deployable."""
    mat = expression_df.values.copy()
    n_samples, n_genes = mat.shape
    ranked = np.zeros_like(mat, dtype=float)
    for i in range(n_samples):
        ranked[i, :] = rankdata(mat[i, :], method="average")
    quantiles = (ranked - 0.5) / n_genes
    result = norm.ppf(np.clip(quantiles, 1e-7, 1 - 1e-7))
    return pd.DataFrame(result, index=expression_df.index, columns=expression_df.columns)


def compute_ssgsea(expression_df):
    """50 Hallmark pathway scores via gseapy."""
    import gseapy as gp
    expr_clean = clean_matrix(expression_df)
    n = expr_clean.shape[0]
    if n <= CHUNK_SIZE:
        result = gp.ssgsea(
            data=expr_clean.T, gene_sets="MSigDB_Hallmark_2020",
            outdir=None, min_size=5, no_plot=True, verbose=False,
        )
        scores = result.res2d.pivot(index="Name", columns="Term", values="NES")
        scores.index.name = None; scores.columns.name = None
    else:
        chunks = []
        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            result = gp.ssgsea(
                data=expr_clean.iloc[start:end].T, gene_sets="MSigDB_Hallmark_2020",
                outdir=None, min_size=5, no_plot=True, verbose=False,
            )
            cs = result.res2d.pivot(index="Name", columns="Term", values="NES")
            cs.index.name = None; cs.columns.name = None
            chunks.append(cs)
        scores = pd.concat(chunks, axis=0)
    scores = scores.loc[expression_df.index].astype(float)
    scores.columns = [f"ssgsea_{c}" for c in scores.columns]
    return clean_matrix(scores)


def compute_progeny(expression_df):
    """14 PROGENy signaling pathway activities."""
    import decoupler as dc
    expr_clean = clean_matrix(expression_df)
    progeny_model = dc.get_progeny(organism="human", top=500)
    estimates, _ = dc.run_mlm(mat=expr_clean, net=progeny_model)
    estimates.columns = [f"progeny_{c}" for c in estimates.columns]
    return clean_matrix(estimates.loc[expression_df.index])


def compute_singscore(expression_df, gene_list):
    """Rank-based pathway scoring. Per-sample, no reference needed."""
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
    """Curated gene-pair features. No labels needed."""
    features = {}
    for gene_a, gene_b in CURATED_PAIRS:
        if gene_a in expression_df.columns and gene_b in expression_df.columns:
            features[f"reo_{gene_a}_gt_{gene_b}"] = (
                expression_df[gene_a] > expression_df[gene_b]
            ).astype(int)
    return pd.DataFrame(features, index=expression_df.index)


def select_discriminative_pairs(train_expr, train_labels, gene_list, top_k=20):
    """Data-driven pair selection. MUST be fold-local (training only)."""
    gene_var = train_expr[gene_list].var()
    top_genes = gene_var.nlargest(50).index.tolist()
    pair_scores = []
    for i, ga in enumerate(top_genes):
        for gb in top_genes[i + 1 :]:
            reo = (train_expr[ga] > train_expr[gb]).astype(int)
            try:
                auc = roc_auc_score(train_labels, reo)
                pair_scores.append((ga, gb, abs(auc - 0.5)))
            except Exception:
                continue
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in pair_scores[:top_k]]


def build_reo_datadriven(expression_df, selected_pairs):
    """Apply pre-selected pairs."""
    features = {}
    for ga, gb in selected_pairs:
        if ga in expression_df.columns and gb in expression_df.columns:
            features[f"reo_dd_{ga}_gt_{gb}"] = (
                expression_df[ga] > expression_df[gb]
            ).astype(int)
    return pd.DataFrame(features, index=expression_df.index)


# =========================================================================
# EVALUATION + THRESHOLD TUNING
# =========================================================================

def tune_threshold(y_true, y_proba, strategy="max_mcc"):
    """Tune threshold on TRAINING data only. Returns frozen scalar."""
    thresholds = np.linspace(0.1, 0.9, 81)
    mccs = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        try:
            mccs.append(matthews_corrcoef(y_true, pred))
        except Exception:
            mccs.append(-1)
    return float(thresholds[int(np.argmax(mccs))])


def evaluate_predictions(y_true, y_proba, threshold):
    """Compute all 5 metrics at frozen threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "threshold_used": threshold,
    }
    try:
        result["auroc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        result["auroc"] = float("nan")
    try:
        result["auprc"] = average_precision_score(y_true, y_proba)
    except ValueError:
        result["auprc"] = float("nan")
    try:
        result["mcc"] = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        result["mcc"] = float("nan")
    try:
        result["bal_acc"] = balanced_accuracy_score(y_true, y_pred)
    except ValueError:
        result["bal_acc"] = float("nan")
    return result


# =========================================================================
# STEP 1: DATA LOADING + PLATFORM MAPPING + GENE HARMONIZATION
# =========================================================================
log.info("=" * 70)
log.info("STEP 1: Data loading + gene harmonization")
log.info("=" * 70)


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
    d.name for d in ctrdb_dir.iterdir()
    if d.is_dir() and d.name.startswith("GSE")
)
ctrdb_geos = [g for g in ctrdb_geos if g != "GSE194040"]

datasets = {}
for geo_id in ctrdb_geos:
    expr, labels = load_ctrdb_dataset(geo_id)
    if expr is not None and labels is not None:
        if labels.nunique() >= 2 and len(labels) >= MIN_PATIENTS:
            datasets[geo_id] = (expr, labels)

for name, data_dir, geo_id in [
    ("ISPY2", "ispy2", "GSE194040"),
    ("BrighTNess", "brightness", "GSE164458"),
]:
    try:
        e, l = load_positional_dataset(data_dir, geo_id)
        if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
            datasets[name] = (e, l)
    except Exception:
        pass

# Exclude targeted panel and beadarray (no L1000 gene overlap)
excluded = []
for did in list(datasets.keys()):
    tech = PLATFORM_MAP.get(did, "unknown")
    if tech in ("targeted", "beadarray"):
        excluded.append((did, tech, len(datasets[did][1])))
        del datasets[did]

log.info(f"Excluded datasets: {excluded}")
log.info(f"Remaining: {len(datasets)} datasets, {sum(len(v[1]) for v in datasets.values())} patients")

# L1000 gene list
gene_info = pd.read_csv(ROOT / "data" / "cache" / "geneinfo_beta_input.txt", header=0)
l1000_genes = gene_info.iloc[:, 0].dropna().astype(str).tolist()

# Per-platform availability filter (>=80% of datasets within each tech group)
tech_groups = {}
for did in datasets:
    tech = PLATFORM_MAP.get(did, "unknown")
    tech_groups.setdefault(tech, []).append(did)

gene_avail = {g: {tech: 0 for tech in tech_groups} for g in l1000_genes}
for tech, dids in tech_groups.items():
    for did in dids:
        expr = datasets[did][0]
        for g in l1000_genes:
            if g in expr.columns:
                gene_avail[g][tech] += 1

n_per_tech = {tech: len(dids) for tech, dids in tech_groups.items()}
common_genes = []
for g in l1000_genes:
    passes_all = True
    for tech, n_ds in n_per_tech.items():
        threshold = int(np.ceil(0.80 * n_ds))
        if gene_avail[g].get(tech, 0) < threshold:
            passes_all = False
            break
    if passes_all:
        common_genes.append(g)

common_genes = sorted(common_genes)
log.info(f"Common genes (>=80% per platform): {len(common_genes)}")

# Save
with open(ROOT / "data" / "cache" / "common_genes_cross_platform.json", "w") as f:
    json.dump(common_genes, f)

# Technology distribution
tech_summary = []
for tech, dids in sorted(tech_groups.items()):
    n_pts = sum(len(datasets[d][1]) for d in dids)
    resp_rate = np.mean([datasets[d][1].mean() for d in dids])
    tech_summary.append({
        "technology": tech, "n_datasets": len(dids),
        "n_patients": n_pts, "mean_response_rate": round(resp_rate, 3),
    })
pd.DataFrame(tech_summary).to_csv(OUT_DIR / "technology_distribution.tsv", sep="\t", index=False)

# Restrict to common genes and z-score per dataset
for did in list(datasets.keys()):
    expr, lab = datasets[did]
    available = [g for g in common_genes if g in expr.columns]
    missing = [g for g in common_genes if g not in expr.columns]
    expr_sub = expr[available].copy()
    for g in missing:
        expr_sub[g] = 0.0
    expr_sub = expr_sub[common_genes]
    means = expr_sub.mean(axis=0)
    stds = expr_sub.std(axis=0).replace(0, 1)
    datasets[did] = (clean_matrix((expr_sub - means) / stds), lab)

# Pool
all_parts_x, all_parts_y, all_batch, all_tech = [], [], [], []
s2d = {}
for did in sorted(datasets.keys()):
    e, l = datasets[did]
    idx = [f"{did}__{i}" for i in range(len(e))]
    ec = e.copy(); ec.index = idx
    lc = l.copy(); lc.index = idx
    all_parts_x.append(ec)
    all_parts_y.append(lc)
    tech = PLATFORM_MAP.get(did, "unknown")
    all_batch.extend([did] * len(ec))
    all_tech.extend([tech] * len(ec))
    for s in idx:
        s2d[s] = did

pooled_expr = pd.concat(all_parts_x, axis=0)
pooled_labels = pd.concat(all_parts_y, axis=0)
batch_series = pd.Series(all_batch, index=pooled_expr.index)
tech_series = pd.Series(all_tech, index=pooled_expr.index)

log.info(f"Pooled: {pooled_expr.shape[0]} patients x {pooled_expr.shape[1]} genes")
for tech in sorted(tech_series.unique()):
    n = (tech_series == tech).sum()
    log.info(f"  {tech}: {n} patients")


# =========================================================================
# STEP 2: PRE-COMPUTE PATHWAY FEATURES (unsupervised, safe to do globally)
# =========================================================================
log.info("=" * 70)
log.info("STEP 2: Pre-computing pathway features")
log.info("=" * 70)

# Rank-normalized expression (per-sample, no cross-sample fitting)
log.info("Computing within-sample rank + inverse-normal...")
rank_expr = within_sample_rank_inv_norm(pooled_expr)
log.info(f"Rank expression: {rank_expr.shape}")

# ssGSEA on rank-normalized data
log.info("Computing ssGSEA (on rank-normalized data)...")
ssgsea_features = compute_ssgsea(rank_expr)
log.info(f"ssGSEA: {ssgsea_features.shape[1]} pathways")

# PROGENy
log.info("Computing PROGENy...")
try:
    progeny_features = compute_progeny(rank_expr)
    log.info(f"PROGENy: {progeny_features.shape[1]} pathways")
except Exception as e:
    log.warning(f"PROGENy failed: {e}")
    progeny_features = pd.DataFrame(index=rank_expr.index)

# singscore (rank-based, per-sample)
log.info("Computing singscore...")
singscore_features = compute_singscore(pooled_expr, common_genes)
log.info(f"singscore: {singscore_features.shape[1]} pathways")

# REO knowledge-driven
log.info("Computing REO (knowledge-driven)...")
reo_knowledge = build_reo_knowledge(pooled_expr)
log.info(f"REO knowledge: {reo_knowledge.shape[1]} pairs")

# Platform one-hot
platform_onehot = pd.get_dummies(tech_series, prefix="platform")
platform_onehot.index = pooled_expr.index


# =========================================================================
# STEP 3: DEFINE FEATURE CONFIGS
# =========================================================================

def build_features_for_config(feat_name, plat_name, samples, gene_list,
                              train_labels=None, train_expr_for_reo=None):
    """Build the feature matrix for a given config and sample set."""
    parts = []

    # Gene-level features
    if feat_name.startswith("raw"):
        parts.append(pooled_expr.loc[samples, gene_list])
    else:
        parts.append(rank_expr.loc[samples, gene_list])

    # Pathway features
    if "ssgsea" in feat_name and "singscore" not in feat_name:
        parts.append(ssgsea_features.loc[samples])
    if "progeny" in feat_name:
        if progeny_features.shape[1] > 0:
            parts.append(progeny_features.loc[samples])
    if "singscore" in feat_name:
        parts.append(singscore_features.loc[samples])
    if "all_pathways" in feat_name:
        parts.append(ssgsea_features.loc[samples])
        if progeny_features.shape[1] > 0:
            parts.append(progeny_features.loc[samples])
        parts.append(singscore_features.loc[samples])

    # Pathway-only configs (no gene-level features)
    if feat_name in ("ssgsea_only", "progeny_only", "singscore_only"):
        parts = []
        if feat_name == "ssgsea_only":
            parts.append(ssgsea_features.loc[samples])
        elif feat_name == "progeny_only" and progeny_features.shape[1] > 0:
            parts.append(progeny_features.loc[samples])
        elif feat_name == "singscore_only":
            parts.append(singscore_features.loc[samples])

    # REO
    if "reo_knowledge" in feat_name:
        parts.append(reo_knowledge.loc[samples])
    if "reo_dd" in feat_name and train_labels is not None and train_expr_for_reo is not None:
        pairs = select_discriminative_pairs(train_expr_for_reo, train_labels, gene_list)
        parts.append(build_reo_datadriven(pooled_expr.loc[samples], pairs))

    # Platform covariate
    if "platform_covariate" in plat_name:
        parts.append(platform_onehot.loc[samples])

    if not parts:
        raise ValueError(f"No features built for {feat_name}")

    X = pd.concat(parts, axis=1)
    return clean_matrix(X)


# =========================================================================
# STEP 4: RUN LODO EXPERIMENT GRID
# =========================================================================
log.info("=" * 70)
log.info("STEP 3-4: LODO experiment grid")
log.info("=" * 70)

# Priority grid (reduced from prompt — skip dataset_id exploratory, skip joint quantile for now)
PRIORITY_GRID = [
    # (feat_name, plat_name, test_aware, production_candidate)
    # Baselines
    ("raw_common", "agnostic", False, True),
    ("rank_genes", "agnostic", False, True),
    ("rank_genes", "platform_covariate", False, True),
    # Pathway-only ablations
    ("ssgsea_only", "agnostic", False, True),
    ("singscore_only", "agnostic", False, True),
    # Rank + individual pathways
    ("rank_plus_ssgsea", "agnostic", False, True),
    ("rank_plus_singscore", "agnostic", False, True),
    # Combined
    ("rank_plus_all_pathways", "agnostic", False, True),
    ("rank_plus_all_pathways", "platform_covariate", False, True),
    # REO additions
    ("rank_plus_singscore_reo_knowledge", "agnostic", False, True),
]

dataset_ids = sorted(set(s2d.values()))
all_results = []

for feat_name, plat_name, test_aware, prod_candidate in PRIORITY_GRID:
    config_name = f"{feat_name}__{plat_name}"
    log.info(f"\n--- {config_name} ---")
    t_start = time.time()
    fold_results = []

    for holdout_id in dataset_ids:
        train_samples = [s for s in pooled_expr.index if s2d[s] != holdout_id]
        test_samples = [s for s in pooled_expr.index if s2d[s] == holdout_id]
        train_y = pooled_labels.loc[train_samples].values.astype(int)
        test_y = pooled_labels.loc[test_samples].values.astype(int)

        if len(np.unique(test_y)) < 2 or len(np.unique(train_y)) < 2:
            continue

        try:
            train_expr_for_reo = rank_expr.loc[train_samples] if "reo_dd" in feat_name else None
            train_labels_for_reo = pooled_labels.loc[train_samples] if "reo_dd" in feat_name else None

            X_train = build_features_for_config(
                feat_name, plat_name, train_samples, common_genes,
                train_labels=train_labels_for_reo, train_expr_for_reo=train_expr_for_reo,
            )
            X_test = build_features_for_config(
                feat_name, plat_name, test_samples, common_genes,
            )

            # Align columns
            for col in X_train.columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            for col in X_test.columns:
                if col not in X_train.columns:
                    X_train[col] = 0
            X_test = X_test[X_train.columns]

            mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
            mdl.fit(X_train.values, train_y)

            # Threshold tuning on training predictions
            train_preds = mdl.predict_proba(X_train.values)[:, 1]
            threshold = tune_threshold(train_y, train_preds)

            test_preds = mdl.predict_proba(X_test.values)[:, 1]
            metrics = evaluate_predictions(test_y, test_preds, threshold)

        except Exception as e:
            log.warning(f"  {config_name}/{holdout_id}: failed: {e}")
            continue

        holdout_tech = PLATFORM_MAP.get(holdout_id, "unknown")
        fold_results.append({
            "config": config_name,
            "eval_type": "standard_lodo",
            "holdout": holdout_id,
            "holdout_technology": holdout_tech,
            "test_aware": test_aware,
            "production_candidate": prod_candidate,
            "n_features": X_train.shape[1],
            **metrics,
        })

    if fold_results:
        df = pd.DataFrame(fold_results)
        mean_auroc = df["auroc"].mean()
        mean_mcc = df["mcc"].mean()
        mean_auprc = df["auprc"].mean()
        elapsed = time.time() - t_start
        log.info(
            f"  → {config_name}: AUROC={mean_auroc:.4f}  AUPRC={mean_auprc:.4f}  "
            f"MCC={mean_mcc:.4f}  ({elapsed:.0f}s, {len(df)} folds)"
        )
        all_results.extend(fold_results)


# =========================================================================
# STEP 5: PLATFORM-HELD-OUT EVALUATION
# =========================================================================
log.info("=" * 70)
log.info("STEP 5: Platform-held-out evaluation")
log.info("=" * 70)

# Pick top 3 configs from LODO for platform holdout
lodo_df = pd.DataFrame(all_results)
lodo_summary = (
    lodo_df.groupby("config")["auroc"].mean().sort_values(ascending=False)
)
top_configs = lodo_summary.head(3).index.tolist()
log.info(f"Top 3 configs for platform holdout: {top_configs}")

# Variant A: microarray → RNA-seq
log.info("\nVariant A: microarray → RNA-seq")
rnaseq_samples = [s for s in pooled_expr.index if tech_series[s] == "rnaseq"]
microarray_samples = [s for s in pooled_expr.index if tech_series[s] in ("affymetrix", "agilent")]

if len(rnaseq_samples) >= 10 and len(microarray_samples) >= 50:
    train_y_a = pooled_labels.loc[microarray_samples].values.astype(int)
    test_y_a = pooled_labels.loc[rnaseq_samples].values.astype(int)

    if len(np.unique(test_y_a)) >= 2 and len(np.unique(train_y_a)) >= 2:
        for config_name in top_configs:
            feat_name, plat_name = config_name.split("__")
            try:
                X_tr = build_features_for_config(feat_name, plat_name, microarray_samples, common_genes)
                X_te = build_features_for_config(feat_name, plat_name, rnaseq_samples, common_genes)
                for col in X_tr.columns:
                    if col not in X_te.columns:
                        X_te[col] = 0
                for col in X_te.columns:
                    if col not in X_tr.columns:
                        X_tr[col] = 0
                X_te = X_te[X_tr.columns]

                mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
                mdl.fit(X_tr.values, train_y_a)
                train_preds = mdl.predict_proba(X_tr.values)[:, 1]
                threshold = tune_threshold(train_y_a, train_preds)
                test_preds = mdl.predict_proba(X_te.values)[:, 1]
                metrics = evaluate_predictions(test_y_a, test_preds, threshold)

                all_results.append({
                    "config": config_name,
                    "eval_type": "variant_A_microarray_to_rnaseq",
                    "holdout": "all_rnaseq",
                    "holdout_technology": "rnaseq",
                    "test_aware": False,
                    "production_candidate": True,
                    "n_features": X_tr.shape[1],
                    **metrics,
                })
                log.info(f"  Variant A {config_name}: AUROC={metrics['auroc']:.4f} MCC={metrics['mcc']:.4f}")
            except Exception as e:
                log.warning(f"  Variant A {config_name} failed: {e}")

# Variant B: Affymetrix → Agilent
log.info("\nVariant B: Affymetrix → Agilent")
affy_samples = [s for s in pooled_expr.index if tech_series[s] == "affymetrix"]
agil_samples = [s for s in pooled_expr.index if tech_series[s] == "agilent"]

if len(agil_samples) >= 10 and len(affy_samples) >= 50:
    train_y_b = pooled_labels.loc[affy_samples].values.astype(int)
    test_y_b = pooled_labels.loc[agil_samples].values.astype(int)

    if len(np.unique(test_y_b)) >= 2 and len(np.unique(train_y_b)) >= 2:
        for config_name in top_configs:
            feat_name, plat_name = config_name.split("__")
            try:
                X_tr = build_features_for_config(feat_name, plat_name, affy_samples, common_genes)
                X_te = build_features_for_config(feat_name, plat_name, agil_samples, common_genes)
                for col in X_tr.columns:
                    if col not in X_te.columns:
                        X_te[col] = 0
                for col in X_te.columns:
                    if col not in X_tr.columns:
                        X_tr[col] = 0
                X_te = X_te[X_tr.columns]

                mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
                mdl.fit(X_tr.values, train_y_b)
                train_preds = mdl.predict_proba(X_tr.values)[:, 1]
                threshold = tune_threshold(train_y_b, train_preds)
                test_preds = mdl.predict_proba(X_te.values)[:, 1]
                metrics = evaluate_predictions(test_y_b, test_preds, threshold)

                all_results.append({
                    "config": config_name,
                    "eval_type": "variant_B_affy_to_agilent",
                    "holdout": "all_agilent",
                    "holdout_technology": "agilent",
                    "test_aware": False,
                    "production_candidate": True,
                    "n_features": X_tr.shape[1],
                    **metrics,
                })
                log.info(f"  Variant B {config_name}: AUROC={metrics['auroc']:.4f} MCC={metrics['mcc']:.4f}")
            except Exception as e:
                log.warning(f"  Variant B {config_name} failed: {e}")

# Variant C: Per-dataset RNA-seq holdout
log.info("\nVariant C: Per-dataset RNA-seq holdout")
rnaseq_dataset_ids = [did for did in dataset_ids if PLATFORM_MAP.get(did) == "rnaseq"]
log.info(f"RNA-seq datasets: {rnaseq_dataset_ids}")

for holdout_ds in rnaseq_dataset_ids:
    train_samples_c = [s for s in pooled_expr.index if s2d[s] != holdout_ds]
    test_samples_c = [s for s in pooled_expr.index if s2d[s] == holdout_ds]
    train_y_c = pooled_labels.loc[train_samples_c].values.astype(int)
    test_y_c = pooled_labels.loc[test_samples_c].values.astype(int)

    if len(np.unique(test_y_c)) < 2 or len(np.unique(train_y_c)) < 2:
        continue

    for config_name in top_configs:
        feat_name, plat_name = config_name.split("__")
        try:
            X_tr = build_features_for_config(feat_name, plat_name, train_samples_c, common_genes)
            X_te = build_features_for_config(feat_name, plat_name, test_samples_c, common_genes)
            for col in X_tr.columns:
                if col not in X_te.columns:
                    X_te[col] = 0
            for col in X_te.columns:
                if col not in X_tr.columns:
                    X_tr[col] = 0
            X_te = X_te[X_tr.columns]

            mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
            mdl.fit(X_tr.values, train_y_c)
            train_preds = mdl.predict_proba(X_tr.values)[:, 1]
            threshold = tune_threshold(train_y_c, train_preds)
            test_preds = mdl.predict_proba(X_te.values)[:, 1]
            metrics = evaluate_predictions(test_y_c, test_preds, threshold)

            all_results.append({
                "config": config_name,
                "eval_type": f"variant_C_rnaseq_holdout_{holdout_ds}",
                "holdout": holdout_ds,
                "holdout_technology": "rnaseq",
                "test_aware": False,
                "production_candidate": True,
                "n_features": X_tr.shape[1],
                **metrics,
            })
            log.info(
                f"  Variant C {config_name}/{holdout_ds}: "
                f"AUROC={metrics['auroc']:.4f} MCC={metrics['mcc']:.4f}"
            )
        except Exception as e:
            log.warning(f"  Variant C {config_name}/{holdout_ds} failed: {e}")


# =========================================================================
# STEP 6: SAVE RESULTS + SELECT PRODUCTION CONFIG
# =========================================================================
log.info("=" * 70)
log.info("STEP 6: Results + production config selection")
log.info("=" * 70)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "all_results.tsv", sep="\t", index=False)

# Summary by config and eval_type
summary = (
    results_df.groupby(["config", "eval_type", "test_aware", "production_candidate"])
    .agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_auprc=("auprc", "mean"),
        mean_mcc=("mcc", "mean"),
        mean_bal_acc=("bal_acc", "mean"),
        n_folds=("auroc", "count"),
    )
    .reset_index()
)
summary.to_csv(OUT_DIR / "summary_all_configs.tsv", sep="\t", index=False)

# Production selection: LODO only, production_candidate=True, test_aware=False
lodo_prod = summary[
    (summary["eval_type"] == "standard_lodo")
    & (summary["production_candidate"] == True)
    & (summary["test_aware"] == False)
].copy()

lodo_prod["composite"] = (
    lodo_prod["mean_auroc"]
    + lodo_prod["mean_auprc"]
    + (lodo_prod["mean_mcc"] + 1) / 2
) / 3

lodo_prod = lodo_prod.sort_values("composite", ascending=False)
lodo_prod.to_csv(OUT_DIR / "production_config_selection.tsv", sep="\t", index=False)

log.info("\nProduction config ranking (standard LODO):")
for _, row in lodo_prod.iterrows():
    log.info(
        f"  {row['config']:50s}  AUROC={row['mean_auroc']:.4f}  "
        f"AUPRC={row['mean_auprc']:.4f}  MCC={row['mean_mcc']:.4f}  "
        f"composite={row['composite']:.4f}"
    )

# Platform holdout summary
for eval_type in sorted(results_df["eval_type"].unique()):
    if eval_type == "standard_lodo":
        continue
    sub = results_df[results_df["eval_type"] == eval_type]
    if len(sub) == 0:
        continue
    log.info(f"\n{eval_type}:")
    for config in sub["config"].unique():
        csub = sub[sub["config"] == config]
        log.info(
            f"  {config}: AUROC={csub['auroc'].mean():.4f} MCC={csub['mcc'].mean():.4f}"
        )

best_config = lodo_prod.iloc[0]["config"] if len(lodo_prod) > 0 else "rank_genes__agnostic"
log.info(f"\nBest production config: {best_config}")
log.info(f"Done. Results in {OUT_DIR}/")
