#!/usr/bin/env python
"""
INVEREX — Scope Investigation: Why 0.761 vs 0.666?
====================================================

FINDING: Both runs use the SAME 40 datasets and 5,598 patients.
There are NO "4 extra datasets." The AUC gap comes from:

  v3 (0.761):           212 genes, 256 features (gene + ssGSEA + PROGENy)
                        ComBat with response as continuous covariate
  leakage-free (0.666): 954 genes, 1001 features (gene + ssGSEA)
                        ComBat with response as continuous covariate

Both leaked — the question is: why does the 212-gene model leak MORE?

This script runs a controlled experiment isolating each variable:
  A. 212 genes + leaked ComBat (with labels)    → should reproduce ~0.761
  B. 954 genes + leaked ComBat (with labels)    → should reproduce ~0.666
  C. 212 genes + leaked ComBat (NO labels)      → isolates label leakage
  D. 954 genes + leaked ComBat (NO labels)      → isolates label leakage
  E. 212 genes + quantile (leakage-free)        → honest AUC, 212 genes
  F. 954 genes + quantile (leakage-free)        → honest AUC, 954 genes (= 0.603)

This also produces per-dataset profiles and the comprehensive comparison.

All outputs to results/scope_investigation/ (no conflicts with parallel agent).
"""

import os
import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "scope_investigation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "investigation.log", mode="w"),
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


def clean_matrix(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_ssgsea(expression_df, label=""):
    import gseapy as gp
    expr_clean = clean_matrix(expression_df)
    n_samples = expr_clean.shape[0]
    if n_samples <= CHUNK_SIZE:
        result = gp.ssgsea(
            data=expr_clean.T, gene_sets="MSigDB_Hallmark_2020",
            outdir=None, min_size=5, no_plot=True, verbose=False,
        )
        scores = result.res2d.pivot(index="Name", columns="Term", values="NES")
        scores.index.name = None; scores.columns.name = None
    else:
        chunks = []
        for start in range(0, n_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_samples)
            chunk = expr_clean.iloc[start:end]
            result = gp.ssgsea(
                data=chunk.T, gene_sets="MSigDB_Hallmark_2020",
                outdir=None, min_size=5, no_plot=True, verbose=False,
            )
            cs = result.res2d.pivot(index="Name", columns="Term", values="NES")
            cs.index.name = None; cs.columns.name = None
            chunks.append(cs)
        scores = pd.concat(chunks, axis=0)
    scores = scores.loc[expression_df.index].astype(float)
    scores.columns = [f"ssgsea_{c}" for c in scores.columns]
    return clean_matrix(scores)


def quantile_normalize(expression_df):
    """Within-sample rank → inverse-normal transform."""
    mat = expression_df.values.copy()
    n_samples, n_genes = mat.shape
    ranked = np.zeros_like(mat)
    for i in range(n_samples):
        ranked[i, :] = rankdata(mat[i, :], method="average")
    quantiles = (ranked - 0.5) / n_genes
    ranked = norm.ppf(np.clip(quantiles, 1e-7, 1 - 1e-7))
    return pd.DataFrame(ranked, index=expression_df.index, columns=expression_df.columns)


# =========================================================================
# Load datasets (identical to other scripts)
# =========================================================================
log.info("Loading datasets...")


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
treatment_map = {}

for geo_id in ctrdb_geos:
    expr, labels = load_ctrdb_dataset(geo_id)
    if expr is not None and labels is not None:
        if labels.nunique() >= 2 and len(labels) >= MIN_PATIENTS:
            datasets[geo_id] = (expr, labels)
            treatment_map[geo_id] = "chemo"

for name, data_dir, geo_id in [
    ("ISPY2", "ispy2", "GSE194040"),
    ("BrighTNess", "brightness", "GSE164458"),
]:
    try:
        e, l = load_positional_dataset(data_dir, geo_id)
        if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
            datasets[name] = (e, l)
            treatment_map[name] = "combination" if name == "ISPY2" else "parp"
    except Exception:
        pass

total_patients = sum(len(v[1]) for v in datasets.values())
log.info(f"Loaded {len(datasets)} datasets, {total_patients} patients")


# =========================================================================
# STEP 1: Dataset profiling
# =========================================================================
log.info("=" * 60)
log.info("STEP 1: Dataset profiling")
log.info("=" * 60)

profiles = []
for did, (expr, lab) in sorted(datasets.items()):
    profiles.append({
        "dataset_id": did,
        "n_patients": len(lab),
        "n_responders": int(lab.sum()),
        "n_nonresponders": int(len(lab) - lab.sum()),
        "response_rate": round(lab.mean(), 3),
        "n_genes_available": expr.shape[1],
        "treatment_class": treatment_map.get(did, "unknown"),
    })

profiles_df = pd.DataFrame(profiles).sort_values("n_patients", ascending=False)
profiles_df.to_csv(OUT_DIR / "dataset_profiles.tsv", sep="\t", index=False)
log.info(f"Saved dataset profiles ({len(profiles_df)} datasets)")
for _, r in profiles_df.iterrows():
    log.info(
        f"  {r['dataset_id']:15s}  n={r['n_patients']:4d}  "
        f"resp={r['response_rate']:.2f}  genes={r['n_genes_available']}"
    )


# =========================================================================
# STEP 2: Build two gene sets (212 vs 954)
# =========================================================================
log.info("=" * 60)
log.info("STEP 2: Building two gene sets")
log.info("=" * 60)

# 954-gene set (60% threshold, as in leakage-free run)
gene_info = pd.read_csv(ROOT / "data" / "cache" / "geneinfo_beta_input.txt", header=0)
l1000_all = gene_info.iloc[:, 0].dropna().astype(str).tolist()

gene_counts = {}
for did, (expr, _) in datasets.items():
    for g in expr.columns:
        gene_counts[g] = gene_counts.get(g, 0) + 1

n_ds = len(datasets)
genes_954 = sorted(
    g for g in l1000_all if gene_counts.get(g, 0) >= int(np.ceil(0.60 * n_ds))
)

# 212-gene set: the v3 used a different geneinfo file. Let's figure out which.
# The v3 log says "L1000 gene list: 212 genes" from geneinfo_beta_input.txt.
# But geneinfo_beta_input.txt has 954 entries now. The file may have been
# overwritten. Let's check if there's a cached 212-gene list.
gene_list_212_path = ROOT / "results" / "full_retrain" / "gene_list.txt"
if gene_list_212_path.exists():
    # The definitive_retrain_full.py saved the gene list used
    with open(gene_list_212_path) as f:
        genes_212_raw = [line.strip() for line in f if line.strip()]
    # Filter to those available in our datasets
    genes_212 = sorted(g for g in genes_212_raw if g in gene_counts)
else:
    # Reconstruct: v3 used 60% threshold but on a SMALLER initial list
    # The v3 comment says "L1000 gene list: 212 genes" — the input was 212.
    # This means geneinfo_beta_input.txt originally had 212 entries.
    # Since we can't recover the original file, approximate with the
    # feature names from the v3 model.
    try:
        v3_bundle = pd.read_csv(ROOT / "results" / "full_retrain" / "feature_importances.tsv", sep="\t")
        genes_212 = sorted([
            f.replace("gene_", "") for f in v3_bundle["feature"].tolist()
            if f.startswith("gene_")
        ])
    except Exception:
        genes_212 = genes_954[:212]  # fallback

log.info(f"Gene set 212: {len(genes_212)} genes")
log.info(f"Gene set 954: {len(genes_954)} genes")
log.info(f"Overlap: {len(set(genes_212) & set(genes_954))} genes")


# =========================================================================
# Prepare data for both gene sets
# =========================================================================
def prepare_datasets(gene_list):
    """Restrict to gene list, z-score per dataset, pool."""
    prepared = {}
    for did, (expr, lab) in datasets.items():
        available = [g for g in gene_list if g in expr.columns]
        missing = [g for g in gene_list if g not in expr.columns]
        expr_sub = expr[available].copy()
        for g in missing:
            expr_sub[g] = 0.0
        expr_sub = expr_sub[gene_list]
        # Per-dataset z-score
        means = expr_sub.mean(axis=0)
        stds = expr_sub.std(axis=0).replace(0, 1)
        expr_z = clean_matrix((expr_sub - means) / stds)
        prepared[did] = (expr_z, lab)
    return prepared


def pool_datasets(prepared):
    """Pool all prepared datasets into a single matrix."""
    parts_x, parts_y, batch = [], [], []
    s2d = {}
    for did in sorted(prepared.keys()):
        e, l = prepared[did]
        idx = [f"{did}__{i}" for i in range(len(e))]
        ec = e.copy(); ec.index = idx
        lc = l.copy(); lc.index = idx
        parts_x.append(ec)
        parts_y.append(lc)
        batch.extend([did] * len(ec))
        for s in idx:
            s2d[s] = did
    X = pd.concat(parts_x, axis=0)
    y = pd.concat(parts_y, axis=0)
    b = pd.Series(batch, index=X.index)
    return X, y, b, s2d


def run_lodo(X_feat, y, s2d, config_name):
    """Run LODO evaluation on pre-built feature matrix."""
    ds_arr = np.array([s2d[s] for s in X_feat.index])
    y_arr = y.values.astype(int)
    results = []
    for test_did in sorted(set(ds_arr)):
        test_mask = ds_arr == test_did
        train_mask = ~test_mask
        X_tr = X_feat.values[train_mask]
        y_tr = y_arr[train_mask]
        X_te = X_feat.values[test_mask]
        y_te = y_arr[test_mask]
        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            continue
        mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
        mdl.fit(X_tr, y_tr)
        yp = mdl.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y_te, yp)
        except ValueError:
            auc = np.nan
        results.append({
            "config": config_name,
            "dataset_id": test_did,
            "n_test": int(test_mask.sum()),
            "n_test_pos": int(y_te.sum()),
            "auc": auc,
        })
    return pd.DataFrame(results)


# =========================================================================
# STEP 3: Controlled experiment — 6 configurations
# =========================================================================
log.info("=" * 60)
log.info("STEP 3: Controlled experiment")
log.info("=" * 60)

all_results = []

for gene_label, gene_list in [("212", genes_212), ("954", genes_954)]:
    log.info(f"\n--- Gene set: {gene_label} ({len(gene_list)} genes) ---")
    prep = prepare_datasets(gene_list)
    X_pooled, y_pooled, batch_pooled, s2d = pool_datasets(prep)
    log.info(f"Pooled: {X_pooled.shape}")

    # --- Config A/B: Leaked ComBat WITH labels ---
    try:
        from neuroCombat import neuroCombat
        covars_with_labels = pd.DataFrame({
            "batch": batch_pooled.values,
            "response": y_pooled.values.astype(float),
        }, index=X_pooled.index)

        log.info(f"  Running leaked ComBat WITH labels ({gene_label} genes)...")
        t0 = time.time()
        result = neuroCombat(
            dat=X_pooled.values.T,
            covars=covars_with_labels,
            batch_col="batch",
            continuous_cols=["response"],
        )
        combat_labels = pd.DataFrame(
            result["data"].T, columns=X_pooled.columns, index=X_pooled.index,
        )
        combat_labels = clean_matrix(combat_labels)
        log.info(f"  ComBat+labels done ({time.time()-t0:.1f}s)")

        # ssGSEA
        log.info(f"  ssGSEA ({gene_label}, combat+labels)...")
        ssgsea_cl = compute_ssgsea(combat_labels, f"{gene_label}_cl")
        X_feat_cl = pd.concat([combat_labels, ssgsea_cl], axis=1)
        X_feat_cl = clean_matrix(X_feat_cl)

        config_name = f"{gene_label}g_leaked_combat_with_labels"
        log.info(f"  LODO: {config_name} ({X_feat_cl.shape[1]} features)...")
        res = run_lodo(X_feat_cl, y_pooled, s2d, config_name)
        all_results.append(res)
        log.info(f"  → {config_name}: AUC = {res['auc'].mean():.4f}")
    except Exception as e:
        log.warning(f"  ComBat+labels failed: {e}")

    # --- Config C/D: Leaked ComBat WITHOUT labels ---
    try:
        covars_no_labels = pd.DataFrame({
            "batch": batch_pooled.values,
        }, index=X_pooled.index)

        log.info(f"  Running leaked ComBat NO labels ({gene_label} genes)...")
        t0 = time.time()
        result = neuroCombat(
            dat=X_pooled.values.T,
            covars=covars_no_labels,
            batch_col="batch",
        )
        combat_nolabels = pd.DataFrame(
            result["data"].T, columns=X_pooled.columns, index=X_pooled.index,
        )
        combat_nolabels = clean_matrix(combat_nolabels)
        log.info(f"  ComBat no-labels done ({time.time()-t0:.1f}s)")

        log.info(f"  ssGSEA ({gene_label}, combat_nolabels)...")
        ssgsea_cn = compute_ssgsea(combat_nolabels, f"{gene_label}_cn")
        X_feat_cn = pd.concat([combat_nolabels, ssgsea_cn], axis=1)
        X_feat_cn = clean_matrix(X_feat_cn)

        config_name = f"{gene_label}g_leaked_combat_no_labels"
        log.info(f"  LODO: {config_name} ({X_feat_cn.shape[1]} features)...")
        res = run_lodo(X_feat_cn, y_pooled, s2d, config_name)
        all_results.append(res)
        log.info(f"  → {config_name}: AUC = {res['auc'].mean():.4f}")
    except Exception as e:
        log.warning(f"  ComBat no-labels failed: {e}")

    # --- Config E/F: Quantile normalization (leakage-free) ---
    log.info(f"  Quantile normalization ({gene_label} genes)...")
    quantile_expr = quantile_normalize(X_pooled)
    quantile_expr = clean_matrix(quantile_expr)

    log.info(f"  ssGSEA ({gene_label}, quantile)...")
    ssgsea_q = compute_ssgsea(quantile_expr, f"{gene_label}_q")
    X_feat_q = pd.concat([quantile_expr, ssgsea_q], axis=1)
    X_feat_q = clean_matrix(X_feat_q)

    config_name = f"{gene_label}g_quantile"
    log.info(f"  LODO: {config_name} ({X_feat_q.shape[1]} features)...")
    res = run_lodo(X_feat_q, y_pooled, s2d, config_name)
    all_results.append(res)
    log.info(f"  → {config_name}: AUC = {res['auc'].mean():.4f}")


# =========================================================================
# STEP 4: Build comprehensive comparison
# =========================================================================
log.info("=" * 60)
log.info("STEP 4: Comprehensive comparison")
log.info("=" * 60)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(OUT_DIR / "controlled_experiment_detail.tsv", sep="\t", index=False)

summary = (
    combined.groupby("config")["auc"]
    .agg(["mean", "std", "median", "count"])
    .sort_values("mean", ascending=False)
)
summary.to_csv(OUT_DIR / "controlled_experiment_summary.tsv", sep="\t")

log.info("\n" + "=" * 60)
log.info("CONTROLLED EXPERIMENT RESULTS")
log.info("=" * 60)
for config, row in summary.iterrows():
    log.info(f"  {config:45s}  AUC = {row['mean']:.4f} ± {row['std']:.4f}")

# Decomposition
log.info("\n" + "=" * 60)
log.info("DECOMPOSITION OF 0.761 → 0.666 GAP")
log.info("=" * 60)

aucs = {}
for config, row in summary.iterrows():
    aucs[config] = row["mean"]

# Effects
if "212g_leaked_combat_with_labels" in aucs and "954g_leaked_combat_with_labels" in aucs:
    gene_effect = aucs["212g_leaked_combat_with_labels"] - aucs["954g_leaked_combat_with_labels"]
    log.info(f"  Gene count effect (212 vs 954, same ComBat+labels): {gene_effect:+.4f}")

if "212g_leaked_combat_with_labels" in aucs and "212g_leaked_combat_no_labels" in aucs:
    label_effect_212 = aucs["212g_leaked_combat_with_labels"] - aucs["212g_leaked_combat_no_labels"]
    log.info(f"  Label covariate effect (212 genes): {label_effect_212:+.4f}")

if "954g_leaked_combat_with_labels" in aucs and "954g_leaked_combat_no_labels" in aucs:
    label_effect_954 = aucs["954g_leaked_combat_with_labels"] - aucs["954g_leaked_combat_no_labels"]
    log.info(f"  Label covariate effect (954 genes): {label_effect_954:+.4f}")

if "212g_leaked_combat_no_labels" in aucs and "212g_quantile" in aucs:
    combat_effect_212 = aucs["212g_leaked_combat_no_labels"] - aucs["212g_quantile"]
    log.info(f"  ComBat batch leakage (212 genes, no labels): {combat_effect_212:+.4f}")

if "954g_leaked_combat_no_labels" in aucs and "954g_quantile" in aucs:
    combat_effect_954 = aucs["954g_leaked_combat_no_labels"] - aucs["954g_quantile"]
    log.info(f"  ComBat batch leakage (954 genes, no labels): {combat_effect_954:+.4f}")

log.info("\nHonest (leakage-free) AUCs:")
for key in ["212g_quantile", "954g_quantile"]:
    if key in aucs:
        log.info(f"  {key}: {aucs[key]:.4f}")


# =========================================================================
# STEP 5: Per-dataset heatmap data
# =========================================================================
log.info("=" * 60)
log.info("STEP 5: Per-dataset comparison heatmap")
log.info("=" * 60)

# Pivot: dataset_id x config → AUC
pivot = combined.pivot_table(index="dataset_id", columns="config", values="auc")

# Add profiles
pivot = pivot.merge(profiles_df.set_index("dataset_id"), left_index=True, right_index=True, how="left")
pivot = pivot.sort_values("n_patients", ascending=False)
pivot.to_csv(OUT_DIR / "comprehensive_dataset_comparison.tsv", sep="\t")

log.info(f"Saved comprehensive comparison: {pivot.shape}")
log.info(f"\nDone. All results in {OUT_DIR}/")
