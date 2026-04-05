#!/usr/bin/env python
"""
INVEREX Pan-Cancer Patient Model — Full Combined Retraining Pipeline v3
=======================================================================
Combines ALL improvements into one pipeline:
  - 38+ datasets (~5,200+ patients): CTR-DB + I-SPY2 + BrighTNess
  - L1000 landmark gene z-scores
  - ssGSEA Hallmark pathway scores
  - PROGENy pathway activities (14 pathways)
  - ComBat batch correction (neuroCombat)
  - LightGBM binary classifier with LODO cross-validation
  - Ablation comparison across 5 feature configs (A-E)
  - Stratification by treatment class

API notes:
  - neuroCombat: use categorical_cols= for biological covariate
  - decoupler v2: dc.op.progeny() and dc.mt.mlm(data=..., net=...)
  - Fill NaN/inf before decoupler MLM and LightGBM
  - ComBat: response as continuous covariate (0/1 float) to avoid
    rank-deficiency issues with many batches + few categories
"""

import os
import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import joblib
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

OUT_DIR = ROOT / "results" / "full_retrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "retrain.log", mode="w"),
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
GENE_PRESENCE_THRESHOLD = 0.60
CHUNK_SIZE = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_matrix(df):
    """Replace NaN and inf with 0 in a DataFrame."""
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_ssgsea(expression_df, label=""):
    """Compute ssGSEA Hallmark pathway scores. Input: samples x genes."""
    import gseapy as gp

    expr_clean = clean_matrix(expression_df)
    n_samples = expr_clean.shape[0]

    if n_samples <= CHUNK_SIZE:
        result = gp.ssgsea(
            data=expr_clean.T,
            gene_sets="MSigDB_Hallmark_2020",
            outdir=None,
            min_size=5,
            no_plot=True,
            verbose=False,
        )
        scores = result.res2d.pivot(index="Name", columns="Term", values="NES")
        scores.index.name = None
        scores.columns.name = None
    else:
        chunks = []
        for start in range(0, n_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_samples)
            chunk = expr_clean.iloc[start:end]
            log.info(f"  ssGSEA {label} chunk {start}-{end} / {n_samples}")
            result = gp.ssgsea(
                data=chunk.T,
                gene_sets="MSigDB_Hallmark_2020",
                outdir=None,
                min_size=5,
                no_plot=True,
                verbose=False,
            )
            cs = result.res2d.pivot(index="Name", columns="Term", values="NES")
            cs.index.name = None
            cs.columns.name = None
            chunks.append(cs)
        scores = pd.concat(chunks, axis=0)

    scores = scores.loc[expression_df.index].astype(float)
    scores.columns = [f"ssgsea_{c}" for c in scores.columns]
    return clean_matrix(scores)


def compute_progeny(expression_df, label=""):
    """Compute PROGENy pathway activities. Input: samples x genes."""
    import decoupler as dc

    expr_clean = clean_matrix(expression_df)
    log.info(f"  Getting PROGENy model... {label}")
    progeny_model = dc.op.progeny(organism="human", top=500)
    log.info(f"  PROGENy model: {progeny_model.shape[0]} gene-pathway pairs")
    log.info(f"  Running PROGENy MLM... {label}")
    estimates, pvalues = dc.mt.mlm(data=expr_clean, net=progeny_model)
    estimates.columns = [f"progeny_{c}" for c in estimates.columns]
    estimates = estimates.loc[expression_df.index]
    return clean_matrix(estimates)


# ---------------------------------------------------------------------------
# Step 1: Load all datasets
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 1: Loading all datasets")
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
    assert len(expr) == len(labels)
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

try:
    e, l = load_positional_dataset("ispy2", "GSE194040")
    if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
        datasets["ISPY2"] = (e, l)
        treatment_map["ISPY2"] = "combination"
        log.info(f"  ISPY2: {len(l)} patients")
except Exception as exc:
    log.warning(f"  ISPY2 failed: {exc}")

try:
    e, l = load_positional_dataset("brightness", "GSE164458")
    if l.nunique() >= 2 and len(l) >= MIN_PATIENTS:
        datasets["BrighTNess"] = (e, l)
        treatment_map["BrighTNess"] = "parp"
        log.info(f"  BrighTNess: {len(l)} patients")
except Exception as exc:
    log.warning(f"  BrighTNess failed: {exc}")

total_patients = sum(len(v[1]) for v in datasets.values())
log.info(f"Loaded {len(datasets)} datasets, {total_patients} total patients")
for did, (expr, lab) in sorted(datasets.items()):
    log.info(
        f"  {did}: {len(lab)} pts, {lab.sum()} resp, "
        f"{len(lab)-lab.sum()} non-resp, {expr.shape[1]} genes"
    )

# ---------------------------------------------------------------------------
# Step 2: Restrict to L1000 landmark genes
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 2: Restricting to L1000 landmark genes")
log.info("=" * 70)

gene_info = pd.read_csv(ROOT / "data" / "cache" / "geneinfo_beta_input.txt", header=0)
l1000_genes = gene_info.iloc[:, 0].dropna().astype(str).tolist()
log.info(f"L1000 gene list: {len(l1000_genes)} genes")

gene_counts = {}
for did, (expr, _) in datasets.items():
    for g in expr.columns:
        gene_counts[g] = gene_counts.get(g, 0) + 1

n_datasets = len(datasets)
threshold_count = int(np.ceil(GENE_PRESENCE_THRESHOLD * n_datasets))
common_l1000 = sorted(
    g for g in l1000_genes if gene_counts.get(g, 0) >= threshold_count
)
log.info(
    f"L1000 genes in >= {GENE_PRESENCE_THRESHOLD*100:.0f}% of datasets "
    f"({threshold_count}/{n_datasets}): {len(common_l1000)}"
)

for did in list(datasets.keys()):
    expr, lab = datasets[did]
    available = [g for g in common_l1000 if g in expr.columns]
    missing = [g for g in common_l1000 if g not in expr.columns]
    expr_sub = expr[available].copy()
    for g in missing:
        expr_sub[g] = 0.0
    datasets[did] = (expr_sub[common_l1000], lab)

log.info(f"Gene matrix: {len(common_l1000)} features per patient")

# ---------------------------------------------------------------------------
# Step 3: Z-score normalize per gene within each dataset
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 3: Z-score normalization per dataset")
log.info("=" * 70)

for did in datasets:
    expr, lab = datasets[did]
    means = expr.mean(axis=0)
    stds = expr.std(axis=0)
    stds = stds.replace(0, 1)
    expr_z = (expr - means) / stds
    # Replace any remaining NaN/inf from constant columns
    expr_z = clean_matrix(expr_z)
    datasets[did] = (expr_z, lab)
    log.info(f"  {did}: z-scored ({expr_z.shape})")

# ---------------------------------------------------------------------------
# Step 4: Pool all datasets
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 4: Pooling all datasets")
log.info("=" * 70)

all_expr, all_labels, all_batch = [], [], []
sample_dataset_map = {}

for did in sorted(datasets.keys()):
    expr, lab = datasets[did]
    new_idx = [f"{did}__{i}" for i in range(len(expr))]
    ec = expr.copy(); ec.index = new_idx
    lc = lab.copy(); lc.index = new_idx
    all_expr.append(ec); all_labels.append(lc)
    all_batch.extend([did] * len(ec))
    for s in new_idx:
        sample_dataset_map[s] = did

pooled_expr = pd.concat(all_expr, axis=0)
pooled_labels = pd.concat(all_labels, axis=0)
batch_series = pd.Series(all_batch, index=pooled_expr.index, name="batch")

log.info(f"Pooled: {pooled_expr.shape[0]} samples x {pooled_expr.shape[1]} genes")
log.info(f"Labels: {pooled_labels.value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# Step 5: Apply ComBat batch correction
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 5: ComBat batch correction")
log.info("=" * 70)

combat_success = False
corrected_expr = pooled_expr.copy()

try:
    from neuroCombat import neuroCombat

    # Use response as continuous covariate (avoids rank-deficiency with
    # many batches + binary categorical, which can make the design matrix
    # singular for some batches)
    covariates = pd.DataFrame(
        {
            "batch": batch_series.values,
            "response": pooled_labels.values.astype(float),
        },
        index=pooled_expr.index,
    )

    log.info("Running neuroCombat (response as continuous covariate)...")
    t0 = time.time()
    combat_result = neuroCombat(
        dat=pooled_expr.values.T,
        covars=covariates,
        batch_col="batch",
        continuous_cols=["response"],
    )
    corrected_raw = pd.DataFrame(
        combat_result["data"].T,
        columns=pooled_expr.columns,
        index=pooled_expr.index,
    )
    # Check quality
    n_nan = np.isnan(corrected_raw.values).sum()
    n_inf = np.isinf(corrected_raw.values).sum()
    var_per_gene = corrected_raw.var(axis=0)
    n_zero_var = (var_per_gene < 1e-12).sum()
    log.info(f"  ComBat output: NaN={n_nan}, Inf={n_inf}, zero-var genes={n_zero_var}")

    corrected_expr = clean_matrix(corrected_raw)
    combat_success = True
    elapsed = time.time() - t0
    log.info(f"ComBat completed in {elapsed:.1f}s, shape={corrected_expr.shape}")
except Exception as e:
    log.warning(f"ComBat FAILED: {e}")
    log.warning("Falling back to per-dataset z-scoring (already applied)")
    corrected_expr = pooled_expr.copy()

# ---------------------------------------------------------------------------
# Step 6: ssGSEA Hallmark pathway scores
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 6: ssGSEA Hallmark pathway scores")
log.info("=" * 70)

ssgsea_combat = None
try:
    log.info("ssGSEA on ComBat-corrected data...")
    t0 = time.time()
    ssgsea_combat = compute_ssgsea(corrected_expr, "(combat)")
    log.info(f"ssGSEA (combat) done in {time.time()-t0:.1f}s -- {ssgsea_combat.shape[1]} pathways")
except Exception as e:
    log.warning(f"ssGSEA (combat) FAILED: {e}")

ssgsea_nocombat = None
if combat_success:
    try:
        log.info("ssGSEA on uncorrected z-scored data...")
        t0 = time.time()
        ssgsea_nocombat = compute_ssgsea(pooled_expr, "(nocombat)")
        log.info(f"ssGSEA (nocombat) done in {time.time()-t0:.1f}s -- {ssgsea_nocombat.shape[1]} pathways")
    except Exception as e:
        log.warning(f"ssGSEA (nocombat) FAILED: {e}")
else:
    ssgsea_nocombat = ssgsea_combat

# ---------------------------------------------------------------------------
# Step 7: PROGENy pathway activities
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 7: PROGENy pathway activities")
log.info("=" * 70)

progeny_combat = None
try:
    log.info("PROGENy on ComBat-corrected data...")
    t0 = time.time()
    progeny_combat = compute_progeny(corrected_expr, "(combat)")
    log.info(f"PROGENy (combat) done in {time.time()-t0:.1f}s -- {progeny_combat.shape[1]} pathways")
except Exception as e:
    log.warning(f"PROGENy (combat) FAILED: {e}")

progeny_nocombat = None
if combat_success:
    try:
        log.info("PROGENy on uncorrected z-scored data...")
        t0 = time.time()
        progeny_nocombat = compute_progeny(pooled_expr, "(nocombat)")
        log.info(f"PROGENy (nocombat) done in {time.time()-t0:.1f}s -- {progeny_nocombat.shape[1]} pathways")
    except Exception as e:
        log.warning(f"PROGENy (nocombat) FAILED: {e}")
else:
    progeny_nocombat = progeny_combat

# ---------------------------------------------------------------------------
# Step 8: Assemble feature configs for ablation
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 8: Assembling feature configs")
log.info("=" * 70)

gene_nc = pooled_expr.copy()
gene_nc.columns = [f"gene_{c}" for c in gene_nc.columns]
gene_cb = corrected_expr.copy()
gene_cb.columns = [f"gene_{c}" for c in gene_cb.columns]

feature_configs = {}

# A: genes only (no ComBat)
feature_configs["A_genes_only"] = gene_nc.copy()
log.info(f"  A: {feature_configs['A_genes_only'].shape[1]} features")

# B: genes + ssGSEA (no ComBat)
if ssgsea_nocombat is not None:
    feature_configs["B_genes_ssgsea"] = pd.concat([gene_nc, ssgsea_nocombat], axis=1)
    log.info(f"  B: {feature_configs['B_genes_ssgsea'].shape[1]} features")
else:
    log.warning("  B skipped")

# C: genes + PROGENy (no ComBat)
if progeny_nocombat is not None:
    feature_configs["C_genes_progeny"] = pd.concat([gene_nc, progeny_nocombat], axis=1)
    log.info(f"  C: {feature_configs['C_genes_progeny'].shape[1]} features")
else:
    log.warning("  C skipped")

# D: genes + ssGSEA + PROGENy (no ComBat)
if ssgsea_nocombat is not None and progeny_nocombat is not None:
    feature_configs["D_genes_ssgsea_progeny"] = pd.concat(
        [gene_nc, ssgsea_nocombat, progeny_nocombat], axis=1
    )
    log.info(f"  D: {feature_configs['D_genes_ssgsea_progeny'].shape[1]} features")
elif ssgsea_nocombat is not None:
    # Fallback: D = B if no PROGENy
    feature_configs["D_genes_ssgsea_progeny"] = feature_configs["B_genes_ssgsea"].copy()
    log.info(f"  D (fallback=B): {feature_configs['D_genes_ssgsea_progeny'].shape[1]} features")
else:
    log.warning("  D skipped")

# E: full pipeline + ComBat
if combat_success:
    parts = [gene_cb]
    if ssgsea_combat is not None:
        parts.append(ssgsea_combat)
    if progeny_combat is not None:
        parts.append(progeny_combat)
    feature_configs["E_full_combat"] = pd.concat(parts, axis=1)
    log.info(f"  E: {feature_configs['E_full_combat'].shape[1]} features (ComBat)")
else:
    if "D_genes_ssgsea_progeny" in feature_configs:
        feature_configs["E_full_combat"] = feature_configs["D_genes_ssgsea_progeny"].copy()
        log.info(f"  E (fallback=D): {feature_configs['E_full_combat'].shape[1]} features")

for cn, cd in sorted(feature_configs.items()):
    log.info(f"  Final {cn}: {cd.shape[1]} feat, {cd.shape[0]} samples")

# ---------------------------------------------------------------------------
# Step 9-10: LODO cross-validation with ablation
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 9-10: LODO cross-validation")
log.info("=" * 70)


def run_lodo(X, y, ds_arr, cfg_name):
    results = []
    for test_did in sorted(set(ds_arr)):
        test = ds_arr == test_did
        train = ~test
        X_tr, y_tr = X[train], y[train]
        X_te, y_te = X[test], y[test]

        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            log.info(f"    {cfg_name}/{test_did}: skipped (single class)")
            continue

        mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
        mdl.fit(X_tr, y_tr)
        yp = mdl.predict_proba(X_te)[:, 1]

        try:
            auc = roc_auc_score(y_te, yp)
        except ValueError:
            auc = np.nan

        results.append({
            "config": cfg_name,
            "test_dataset": test_did,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "n_test_pos": int(y_te.sum()),
            "n_test_neg": int((1 - y_te).sum()),
            "auc": auc,
            "treatment_class": treatment_map.get(test_did, "unknown"),
        })
        log.info(f"    {cfg_name}/{test_did}: AUC={auc:.4f} (n={test.sum()}, pos={int(y_te.sum())})")
    return results


ds_arr = np.array([sample_dataset_map[s] for s in pooled_expr.index])
y_arr = pooled_labels.values.astype(int)

all_lodo = []
for cfg_name in sorted(feature_configs.keys()):
    cfg_df = feature_configs[cfg_name]
    log.info(f"\n--- LODO {cfg_name} ({cfg_df.shape[1]} features) ---")
    X = clean_matrix(cfg_df).values
    all_lodo.extend(run_lodo(X, y_arr, ds_arr, cfg_name))

# ---------------------------------------------------------------------------
# Build result tables
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("Building result tables")
log.info("=" * 70)

lodo_df = pd.DataFrame(all_lodo)
lodo_df.to_csv(OUT_DIR / "lodo_per_dataset.tsv", sep="\t", index=False)
log.info(f"Saved lodo_per_dataset.tsv ({len(lodo_df)} rows)")

ablation = (
    lodo_df.groupby("config")
    .agg(
        mean_auc=("auc", "mean"),
        median_auc=("auc", "median"),
        std_auc=("auc", "std"),
        min_auc=("auc", "min"),
        max_auc=("auc", "max"),
        n_folds=("auc", "count"),
        n_features=("config", lambda x: feature_configs[x.iloc[0]].shape[1]),
    )
    .reset_index()
    .sort_values("mean_auc", ascending=False)
)
ablation.to_csv(OUT_DIR / "lodo_ablation_summary.tsv", sep="\t", index=False)
log.info("Saved lodo_ablation_summary.tsv")

log.info("\n" + "=" * 70)
log.info("ABLATION SUMMARY")
log.info("=" * 70)
for _, r in ablation.iterrows():
    log.info(
        f"  {r['config']}: mean_AUC={r['mean_auc']:.4f} "
        f"(+/-{r['std_auc']:.4f}), {int(r['n_features'])} feat, {int(r['n_folds'])} folds"
    )

# ---------------------------------------------------------------------------
# Step 11: Stratify by treatment class
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 11: Treatment class stratification")
log.info("=" * 70)

treat_summary = (
    lodo_df.groupby(["config", "treatment_class"])
    .agg(
        mean_auc=("auc", "mean"),
        median_auc=("auc", "median"),
        std_auc=("auc", "std"),
        n_folds=("auc", "count"),
    )
    .reset_index()
    .sort_values(["treatment_class", "mean_auc"], ascending=[True, False])
)
treat_summary.to_csv(OUT_DIR / "lodo_by_treatment_class.tsv", sep="\t", index=False)
log.info("Saved lodo_by_treatment_class.tsv")
for tc in treat_summary["treatment_class"].unique():
    log.info(f"\n  {tc}:")
    for _, r in treat_summary[treat_summary["treatment_class"] == tc].iterrows():
        log.info(f"    {r['config']}: mean_AUC={r['mean_auc']:.4f} (n={int(r['n_folds'])})")

# ---------------------------------------------------------------------------
# Step 12: Train final model
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 12: Training final model")
log.info("=" * 70)

best_config = ablation.iloc[0]["config"]
log.info(f"Best config: {best_config}")

best_feat = feature_configs[best_config]
X_final = clean_matrix(best_feat).values
y_final = y_arr

final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
final_model.fit(X_final, y_final)
log.info(f"Final model: {X_final.shape[0]} samples, {X_final.shape[1]} features")

importances = pd.DataFrame({
    "feature": best_feat.columns,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False)
importances.to_csv(OUT_DIR / "feature_importances.tsv", sep="\t", index=False)
log.info("Saved feature_importances.tsv (top-10):")
for _, r in importances.head(10).iterrows():
    log.info(f"  {r['feature']}: {r['importance']}")

# ---------------------------------------------------------------------------
# Step 13: Save everything
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 13: Saving artifacts")
log.info("=" * 70)

model_path = ROOT / "results" / "full_retrain_patient_model.joblib"
joblib.dump({
    "model": final_model,
    "config": best_config,
    "feature_names": list(best_feat.columns),
    "lgbm_params": LGBM_PARAMS,
    "n_datasets": len(datasets),
    "n_patients": len(y_final),
    "combat_applied": combat_success,
    "common_l1000_genes": common_l1000,
}, model_path)
log.info(f"Saved model: {model_path}")

best_row = ablation[ablation["config"] == best_config].iloc[0]
headline = pd.DataFrame([
    {"metric": "n_datasets", "value": len(datasets)},
    {"metric": "n_patients", "value": len(y_final)},
    {"metric": "n_responders", "value": int(y_final.sum())},
    {"metric": "n_nonresponders", "value": int((1 - y_final).sum())},
    {"metric": "best_config", "value": best_config},
    {"metric": "n_features", "value": int(best_row["n_features"])},
    {"metric": "mean_auc_lodo", "value": round(best_row["mean_auc"], 4)},
    {"metric": "median_auc_lodo", "value": round(best_row["median_auc"], 4)},
    {"metric": "std_auc_lodo", "value": round(best_row["std_auc"], 4)},
    {"metric": "combat_applied", "value": combat_success},
    {"metric": "ssgsea_computed", "value": ssgsea_combat is not None},
    {"metric": "progeny_computed", "value": progeny_combat is not None},
    {"metric": "n_l1000_genes", "value": len(common_l1000)},
])
headline.to_csv(OUT_DIR / "headline_metrics.tsv", sep="\t", index=False)
log.info("Saved headline_metrics.tsv")

log.info("\n" + "=" * 70)
log.info("PIPELINE COMPLETE")
log.info("=" * 70)
log.info(f"  Datasets:      {len(datasets)}")
log.info(f"  Patients:      {len(y_final)}")
log.info(f"  Best config:   {best_config}")
log.info(f"  Mean LODO AUC: {best_row['mean_auc']:.4f}")
log.info(f"  ComBat:        {'Yes' if combat_success else 'No'}")
log.info(f"  ssGSEA:        {'Yes' if ssgsea_combat is not None else 'No'}")
log.info(f"  PROGENy:       {'Yes' if progeny_combat is not None else 'No'}")
log.info(f"  Output:        {OUT_DIR}")
log.info(f"  Model:         {model_path}")
