#!/usr/bin/env python
"""
INVEREX Pan-Cancer Patient Model — Full Combined Retraining Pipeline
====================================================================
Combines ALL improvements into one pipeline:
  - 38 datasets (~5,200 patients): 36 CTR-DB + I-SPY2 + BrighTNess
  - L1000 landmark gene z-scores
  - ssGSEA Hallmark pathway scores (48 pathways)
  - PROGENy pathway activities (14 pathways)
  - ComBat batch correction (neuroCombat)
  - LightGBM binary classifier with LODO cross-validation
  - Ablation comparison across 5 feature configs
  - Stratification by treatment class
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
GENE_PRESENCE_THRESHOLD = 0.60  # gene must be in >=60% of datasets

# ---------------------------------------------------------------------------
# Step 1: Load all 38 datasets
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 1: Loading all datasets")
log.info("=" * 70)


def load_ctrdb_dataset(geo_id: str) -> tuple:
    """Load a CTR-DB dataset — indices align directly."""
    base = ROOT / "data" / "raw" / "ctrdb" / geo_id
    expr_path = base / f"{geo_id}_expression.parquet"
    label_path = base / "response_labels.parquet"
    if not expr_path.exists() or not label_path.exists():
        return None, None
    expr = pd.read_parquet(expr_path)
    labels = pd.read_parquet(label_path)
    # Align on common index
    common = expr.index.intersection(labels.index)
    if len(common) < MIN_PATIENTS:
        return None, None
    expr = expr.loc[common]
    labels = labels.loc[common, "response"].astype(int)
    return expr, labels


def load_positional_dataset(data_dir: str, geo_id: str) -> tuple:
    """Load I-SPY2 or BrighTNess — positional alignment."""
    base = ROOT / "data" / "raw" / data_dir
    expr_path = base / f"{geo_id}_expression.parquet"
    label_path = base / "response_labels.parquet"
    expr = pd.read_parquet(expr_path)
    labels = pd.read_parquet(label_path)
    # Positional alignment: reset both indices
    assert len(expr) == len(labels), (
        f"{data_dir}: expression ({len(expr)}) != labels ({len(labels)})"
    )
    response = labels["response"].astype(int).values
    # Use expression index as the canonical sample id
    expr = expr.reset_index(drop=True)
    labels_series = pd.Series(response, index=expr.index, name="response")
    return expr, labels_series


# Discover CTR-DB datasets
ctrdb_dir = ROOT / "data" / "raw" / "ctrdb"
ctrdb_geos = sorted(
    [
        d.name
        for d in ctrdb_dir.iterdir()
        if d.is_dir() and d.name.startswith("GSE")
    ]
)
# Exclude GSE194040 from CTR-DB (it's I-SPY2, stored there too)
ctrdb_geos = [g for g in ctrdb_geos if g != "GSE194040"]

datasets = {}  # dataset_id -> (expr_df, labels_series)
treatment_map = {}  # dataset_id -> treatment_class

for geo_id in ctrdb_geos:
    expr, labels = load_ctrdb_dataset(geo_id)
    if expr is not None and labels is not None:
        # Need both classes
        if labels.nunique() >= 2 and len(labels) >= MIN_PATIENTS:
            datasets[geo_id] = (expr, labels)
            treatment_map[geo_id] = "chemo"

# I-SPY2
try:
    expr_ispy, lab_ispy = load_positional_dataset("ispy2", "GSE194040")
    if lab_ispy.nunique() >= 2 and len(lab_ispy) >= MIN_PATIENTS:
        datasets["ISPY2"] = (expr_ispy, lab_ispy)
        treatment_map["ISPY2"] = "combination"
        log.info(f"  ISPY2: {len(lab_ispy)} patients loaded")
except Exception as e:
    log.warning(f"  ISPY2 load failed: {e}")

# BrighTNess
try:
    expr_bright, lab_bright = load_positional_dataset("brightness", "GSE164458")
    if lab_bright.nunique() >= 2 and len(lab_bright) >= MIN_PATIENTS:
        datasets["BrighTNess"] = (expr_bright, lab_bright)
        treatment_map["BrighTNess"] = "parp"
        log.info(f"  BrighTNess: {len(lab_bright)} patients loaded")
except Exception as e:
    log.warning(f"  BrighTNess load failed: {e}")

total_patients = sum(len(v[1]) for v in datasets.values())
log.info(
    f"Loaded {len(datasets)} datasets, {total_patients} total patients"
)
for did, (expr, lab) in sorted(datasets.items()):
    log.info(
        f"  {did}: {len(lab)} pts, {lab.sum()} resp, "
        f"{len(lab) - lab.sum()} non-resp, {expr.shape[1]} genes"
    )

# ---------------------------------------------------------------------------
# Step 2: Restrict to L1000 landmark genes present in >=60% of datasets
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 2: Restricting to L1000 landmark genes")
log.info("=" * 70)

gene_info = pd.read_csv(
    ROOT / "data" / "cache" / "geneinfo_beta_input.txt", header=0
)
l1000_genes = gene_info.iloc[:, 0].dropna().astype(str).tolist()
log.info(f"L1000 gene list: {len(l1000_genes)} genes")

# Count gene presence across datasets
gene_counts = {}
for did, (expr, _) in datasets.items():
    for g in expr.columns:
        gene_counts[g] = gene_counts.get(g, 0) + 1

n_datasets = len(datasets)
threshold_count = int(np.ceil(GENE_PRESENCE_THRESHOLD * n_datasets))

# Keep L1000 genes present in >= threshold datasets
common_l1000 = sorted(
    [
        g
        for g in l1000_genes
        if gene_counts.get(g, 0) >= threshold_count
    ]
)
log.info(
    f"L1000 genes in >= {GENE_PRESENCE_THRESHOLD*100:.0f}% of datasets "
    f"({threshold_count}/{n_datasets}): {len(common_l1000)}"
)

# Subset each dataset to common L1000 genes (fill missing with NaN then 0)
for did in list(datasets.keys()):
    expr, lab = datasets[did]
    available = [g for g in common_l1000 if g in expr.columns]
    missing = [g for g in common_l1000 if g not in expr.columns]
    expr_sub = expr[available].copy()
    if missing:
        for g in missing:
            expr_sub[g] = 0.0
    expr_sub = expr_sub[common_l1000]  # consistent column order
    datasets[did] = (expr_sub, lab)

log.info(f"Gene matrix: {len(common_l1000)} features per patient")

# ---------------------------------------------------------------------------
# Step 3: Z-score normalize per gene within each dataset
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 3: Z-score normalization per dataset")
log.info("=" * 70)

for did in datasets:
    expr, lab = datasets[did]
    # Z-score per column (gene)
    means = expr.mean(axis=0)
    stds = expr.std(axis=0)
    stds = stds.replace(0, 1)  # avoid division by zero
    expr_z = (expr - means) / stds
    datasets[did] = (expr_z, lab)
    log.info(f"  {did}: z-scored ({expr_z.shape})")

# ---------------------------------------------------------------------------
# Step 4: Pool all datasets with dataset_id as batch label
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 4: Pooling all datasets")
log.info("=" * 70)

all_expr_list = []
all_labels_list = []
all_batch_list = []
sample_dataset_map = {}

for did in sorted(datasets.keys()):
    expr, lab = datasets[did]
    # Make unique sample indices
    new_idx = [f"{did}__{i}" for i in range(len(expr))]
    expr_copy = expr.copy()
    expr_copy.index = new_idx
    lab_copy = lab.copy()
    lab_copy.index = new_idx

    all_expr_list.append(expr_copy)
    all_labels_list.append(lab_copy)
    all_batch_list.extend([did] * len(expr_copy))
    for sid in new_idx:
        sample_dataset_map[sid] = did

pooled_expr = pd.concat(all_expr_list, axis=0)
pooled_labels = pd.concat(all_labels_list, axis=0)
batch_labels = pd.Series(all_batch_list, index=pooled_expr.index, name="batch")

log.info(
    f"Pooled matrix: {pooled_expr.shape[0]} samples x {pooled_expr.shape[1]} genes"
)
log.info(f"Label distribution: {pooled_labels.value_counts().to_dict()}")

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

    # neuroCombat expects: dat = genes x samples (rows=genes, cols=samples)
    covariates = pd.DataFrame(
        {"batch": batch_labels.values, "response": pooled_labels.values},
        index=pooled_expr.index,
    )

    log.info("Running neuroCombat...")
    t0 = time.time()
    combat_result = neuroCombat(
        dat=pooled_expr.values.T,  # genes x samples
        covars=covariates,
        batch_col="batch",
        mod=[
            "response"
        ],  # preserve biological signal
    )
    corrected_expr = pd.DataFrame(
        combat_result["data"].T,
        columns=pooled_expr.columns,
        index=pooled_expr.index,
    )
    combat_success = True
    elapsed = time.time() - t0
    log.info(f"ComBat completed in {elapsed:.1f}s")
    log.info(f"Corrected matrix: {corrected_expr.shape}")
except Exception as e:
    log.warning(f"ComBat FAILED: {e}")
    log.warning("Falling back to per-dataset z-scoring (already applied)")
    corrected_expr = pooled_expr.copy()

# ---------------------------------------------------------------------------
# Step 6: Compute ssGSEA Hallmark pathway scores
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 6: ssGSEA Hallmark pathway scores")
log.info("=" * 70)

ssgsea_scores = None
try:
    import gseapy as gp

    log.info("Running ssGSEA with MSigDB_Hallmark_2020...")
    t0 = time.time()

    # ssGSEA expects genes as rows, samples as columns
    # Process in chunks if too large
    n_samples = corrected_expr.shape[0]
    CHUNK_SIZE = 500

    if n_samples <= CHUNK_SIZE:
        result = gp.ssgsea(
            data=corrected_expr.T,
            gene_sets="MSigDB_Hallmark_2020",
            outdir=None,
            min_size=5,
            no_plot=True,
            verbose=False,
        )
        ssgsea_scores = result.res2d.pivot(
            index="Name", columns="Term", values="NES"
        )
        # Name = sample, Term = pathway
        ssgsea_scores.index.name = None
        ssgsea_scores.columns.name = None
    else:
        # Process in chunks to avoid memory / speed issues
        chunks = []
        for start in range(0, n_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_samples)
            chunk_expr = corrected_expr.iloc[start:end]
            log.info(
                f"  ssGSEA chunk {start}-{end} / {n_samples}"
            )
            result = gp.ssgsea(
                data=chunk_expr.T,
                gene_sets="MSigDB_Hallmark_2020",
                outdir=None,
                min_size=5,
                no_plot=True,
                verbose=False,
            )
            chunk_scores = result.res2d.pivot(
                index="Name", columns="Term", values="NES"
            )
            chunk_scores.index.name = None
            chunk_scores.columns.name = None
            chunks.append(chunk_scores)
        ssgsea_scores = pd.concat(chunks, axis=0)

    # Ensure aligned with pooled samples
    ssgsea_scores = ssgsea_scores.loc[corrected_expr.index]
    ssgsea_scores = ssgsea_scores.astype(float)
    # Prefix columns
    ssgsea_scores.columns = [f"ssgsea_{c}" for c in ssgsea_scores.columns]

    elapsed = time.time() - t0
    log.info(
        f"ssGSEA completed in {elapsed:.1f}s — "
        f"{ssgsea_scores.shape[1]} pathways"
    )
except Exception as e:
    log.warning(f"ssGSEA FAILED: {e}")
    log.warning("Proceeding without ssGSEA features")
    ssgsea_scores = None

# ---------------------------------------------------------------------------
# Step 7: Compute PROGENy pathway activities
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 7: PROGENy pathway activities")
log.info("=" * 70)

progeny_scores = None
try:
    import decoupler as dc

    log.info("Getting PROGENy model...")
    progeny_model = dc.get_progeny(organism="human", top=500)
    log.info(f"PROGENy model: {progeny_model.shape[0]} gene-pathway pairs")

    log.info("Running PROGENy (MLM)...")
    t0 = time.time()
    activities = dc.run_mlm(
        mat=corrected_expr, net=progeny_model
    )

    # dc.run_mlm returns a tuple (estimates, pvalues) or just the estimates
    if isinstance(activities, tuple):
        progeny_scores = activities[0]
    else:
        progeny_scores = activities

    # Prefix columns
    progeny_scores.columns = [
        f"progeny_{c}" for c in progeny_scores.columns
    ]
    progeny_scores = progeny_scores.loc[corrected_expr.index]

    elapsed = time.time() - t0
    log.info(
        f"PROGENy completed in {elapsed:.1f}s — "
        f"{progeny_scores.shape[1]} pathways"
    )
except Exception as e:
    log.warning(f"PROGENy FAILED: {e}")
    log.warning("Proceeding without PROGENy features")
    progeny_scores = None

# ---------------------------------------------------------------------------
# Step 8: Assemble combined feature matrix
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 8: Assembling combined feature matrix")
log.info("=" * 70)

# Gene features (from corrected expression)
gene_features = corrected_expr.copy()
gene_features.columns = [f"gene_{c}" for c in gene_features.columns]

# Build feature configs for ablation
feature_configs = {}

# A: genes only
feature_configs["A_genes_only"] = gene_features.copy()
log.info(
    f"  Config A (genes only): {feature_configs['A_genes_only'].shape[1]} features"
)

# B: genes + ssGSEA
if ssgsea_scores is not None:
    feature_configs["B_genes_ssgsea"] = pd.concat(
        [gene_features, ssgsea_scores], axis=1
    )
    log.info(
        f"  Config B (genes+ssGSEA): {feature_configs['B_genes_ssgsea'].shape[1]} features"
    )
else:
    log.warning("  Config B skipped (no ssGSEA scores)")

# C: genes + PROGENy
if progeny_scores is not None:
    feature_configs["C_genes_progeny"] = pd.concat(
        [gene_features, progeny_scores], axis=1
    )
    log.info(
        f"  Config C (genes+PROGENy): {feature_configs['C_genes_progeny'].shape[1]} features"
    )
else:
    log.warning("  Config C skipped (no PROGENy scores)")

# D: genes + ssGSEA + PROGENy
if ssgsea_scores is not None and progeny_scores is not None:
    feature_configs["D_genes_ssgsea_progeny"] = pd.concat(
        [gene_features, ssgsea_scores, progeny_scores], axis=1
    )
    log.info(
        f"  Config D (genes+ssGSEA+PROGENy): {feature_configs['D_genes_ssgsea_progeny'].shape[1]} features"
    )
else:
    log.warning("  Config D skipped (missing ssGSEA or PROGENy)")

# E: config D + ComBat (if ComBat was applied, this is already the case)
# If ComBat failed, we build config E from uncorrected (same as D)
if combat_success:
    # Config D already uses ComBat-corrected data
    if "D_genes_ssgsea_progeny" in feature_configs:
        feature_configs["E_full_combat"] = feature_configs[
            "D_genes_ssgsea_progeny"
        ].copy()
        log.info(
            f"  Config E (D+ComBat): {feature_configs['E_full_combat'].shape[1]} features (ComBat applied)"
        )
    else:
        log.warning("  Config E skipped (no config D available)")
else:
    # Build E without ComBat = same as D but flagged
    if "D_genes_ssgsea_progeny" in feature_configs:
        feature_configs["E_full_combat"] = feature_configs[
            "D_genes_ssgsea_progeny"
        ].copy()
        log.info(
            f"  Config E (D, no ComBat): {feature_configs['E_full_combat'].shape[1]} features (ComBat failed)"
        )

# For configs A-D without ComBat, we need to rebuild from uncorrected data
# Since we z-scored per dataset in step 3, the pooled_expr is the uncorrected+zscored version
# corrected_expr is the ComBat-corrected version
# We need to also compute ssGSEA and PROGENy on uncorrected data for fair comparison
if combat_success:
    log.info("Rebuilding configs A-D from pre-ComBat (z-scored) data for fair ablation...")

    gene_features_nocombat = pooled_expr.copy()
    gene_features_nocombat.columns = [f"gene_{c}" for c in gene_features_nocombat.columns]

    # Compute ssGSEA on uncorrected data
    ssgsea_nocombat = None
    try:
        import gseapy as gp
        n_samples = pooled_expr.shape[0]
        if n_samples <= CHUNK_SIZE:
            result = gp.ssgsea(
                data=pooled_expr.T, gene_sets="MSigDB_Hallmark_2020",
                outdir=None, min_size=5, no_plot=True, verbose=False,
            )
            ssgsea_nocombat = result.res2d.pivot(index="Name", columns="Term", values="NES")
            ssgsea_nocombat.index.name = None
            ssgsea_nocombat.columns.name = None
        else:
            chunks = []
            for start in range(0, n_samples, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n_samples)
                chunk_expr = pooled_expr.iloc[start:end]
                log.info(f"  ssGSEA (no-combat) chunk {start}-{end}")
                result = gp.ssgsea(
                    data=chunk_expr.T, gene_sets="MSigDB_Hallmark_2020",
                    outdir=None, min_size=5, no_plot=True, verbose=False,
                )
                chunk_scores = result.res2d.pivot(index="Name", columns="Term", values="NES")
                chunk_scores.index.name = None
                chunk_scores.columns.name = None
                chunks.append(chunk_scores)
            ssgsea_nocombat = pd.concat(chunks, axis=0)
        ssgsea_nocombat = ssgsea_nocombat.loc[pooled_expr.index].astype(float)
        ssgsea_nocombat.columns = [f"ssgsea_{c}" for c in ssgsea_nocombat.columns]
    except Exception as e:
        log.warning(f"ssGSEA (no-combat) failed: {e}")

    # Compute PROGENy on uncorrected data
    progeny_nocombat = None
    try:
        import decoupler as dc
        progeny_model_nc = dc.get_progeny(organism="human", top=500)
        activities_nc = dc.run_mlm(mat=pooled_expr, net=progeny_model_nc)
        if isinstance(activities_nc, tuple):
            progeny_nocombat = activities_nc[0]
        else:
            progeny_nocombat = activities_nc
        progeny_nocombat.columns = [f"progeny_{c}" for c in progeny_nocombat.columns]
        progeny_nocombat = progeny_nocombat.loc[pooled_expr.index]
    except Exception as e:
        log.warning(f"PROGENy (no-combat) failed: {e}")

    # Rebuild A-D with no-combat data
    feature_configs["A_genes_only"] = gene_features_nocombat.copy()

    if ssgsea_nocombat is not None:
        feature_configs["B_genes_ssgsea"] = pd.concat(
            [gene_features_nocombat, ssgsea_nocombat], axis=1
        )

    if progeny_nocombat is not None:
        feature_configs["C_genes_progeny"] = pd.concat(
            [gene_features_nocombat, progeny_nocombat], axis=1
        )

    if ssgsea_nocombat is not None and progeny_nocombat is not None:
        feature_configs["D_genes_ssgsea_progeny"] = pd.concat(
            [gene_features_nocombat, ssgsea_nocombat, progeny_nocombat], axis=1
        )

# Log final configs
for cfg_name, cfg_df in sorted(feature_configs.items()):
    log.info(f"  Final {cfg_name}: {cfg_df.shape[1]} features, {cfg_df.shape[0]} samples")

# ---------------------------------------------------------------------------
# Step 9 & 10: LODO cross-validation with ablation
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 9-10: LODO cross-validation with ablation")
log.info("=" * 70)

dataset_ids = sorted(datasets.keys())
lodo_results = []  # per dataset, per config


def run_lodo(X, y, dataset_ids_arr, config_name):
    """Run leave-one-dataset-out CV and return per-fold results."""
    unique_datasets = sorted(set(dataset_ids_arr))
    results = []

    for test_did in unique_datasets:
        test_mask = dataset_ids_arr == test_did
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Skip if test set has only one class
        if len(np.unique(y_test)) < 2:
            log.info(
                f"    {config_name} / {test_did}: skipped (single class in test)"
            )
            continue

        if len(np.unique(y_train)) < 2:
            log.info(
                f"    {config_name} / {test_did}: skipped (single class in train)"
            )
            continue

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        results.append(
            {
                "config": config_name,
                "test_dataset": test_did,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_test_pos": int(y_test.sum()),
                "n_test_neg": int((1 - y_test).sum()),
                "auc": auc,
                "treatment_class": treatment_map.get(test_did, "unknown"),
            }
        )
        log.info(
            f"    {config_name} / {test_did}: AUC={auc:.4f} "
            f"(n_test={test_mask.sum()}, pos={int(y_test.sum())})"
        )

    return results


# Build dataset_ids array aligned with pooled data
dataset_ids_arr = np.array(
    [sample_dataset_map[sid] for sid in pooled_expr.index]
)
y_arr = pooled_labels.values.astype(int)

all_lodo_results = []

for config_name in sorted(feature_configs.keys()):
    cfg_df = feature_configs[config_name]
    log.info(
        f"\n--- LODO for {config_name} ({cfg_df.shape[1]} features) ---"
    )
    # Handle NaN
    X = cfg_df.fillna(0).values
    results = run_lodo(X, y_arr, dataset_ids_arr, config_name)
    all_lodo_results.extend(results)

# ---------------------------------------------------------------------------
# Build result tables
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("Building result tables")
log.info("=" * 70)

lodo_df = pd.DataFrame(all_lodo_results)

# Per-dataset results
lodo_df.to_csv(OUT_DIR / "lodo_per_dataset.tsv", sep="\t", index=False)
log.info(f"Saved lodo_per_dataset.tsv ({len(lodo_df)} rows)")

# Ablation summary
ablation_summary = (
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
ablation_summary.to_csv(
    OUT_DIR / "lodo_ablation_summary.tsv", sep="\t", index=False
)
log.info(f"Saved lodo_ablation_summary.tsv")

# Print ablation summary
log.info("\n" + "=" * 70)
log.info("ABLATION SUMMARY")
log.info("=" * 70)
for _, row in ablation_summary.iterrows():
    log.info(
        f"  {row['config']}: mean_AUC={row['mean_auc']:.4f} "
        f"(+/-{row['std_auc']:.4f}), {int(row['n_features'])} features, "
        f"{int(row['n_folds'])} folds"
    )

# ---------------------------------------------------------------------------
# Step 11: Stratify by treatment class
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 11: Stratification by treatment class")
log.info("=" * 70)

treatment_summary = (
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
treatment_summary.to_csv(
    OUT_DIR / "lodo_by_treatment_class.tsv", sep="\t", index=False
)
log.info(f"Saved lodo_by_treatment_class.tsv")

for tc in treatment_summary["treatment_class"].unique():
    log.info(f"\n  Treatment class: {tc}")
    subset = treatment_summary[treatment_summary["treatment_class"] == tc]
    for _, row in subset.iterrows():
        log.info(
            f"    {row['config']}: mean_AUC={row['mean_auc']:.4f} "
            f"(n_folds={int(row['n_folds'])})"
        )

# ---------------------------------------------------------------------------
# Step 12: Train final model on ALL data with best config
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 12: Training final model")
log.info("=" * 70)

# Pick best config by mean AUC
best_config = ablation_summary.iloc[0]["config"]
log.info(f"Best config: {best_config}")

best_features = feature_configs[best_config]
X_final = best_features.fillna(0).values
y_final = y_arr

final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
final_model.fit(X_final, y_final)
log.info(
    f"Final model trained on {X_final.shape[0]} samples, "
    f"{X_final.shape[1]} features"
)

# Feature importances
importances = pd.DataFrame(
    {
        "feature": best_features.columns,
        "importance": final_model.feature_importances_,
    }
).sort_values("importance", ascending=False)
importances.to_csv(
    OUT_DIR / "feature_importances.tsv", sep="\t", index=False
)
log.info(f"Saved feature_importances.tsv (top-10 below)")
for _, row in importances.head(10).iterrows():
    log.info(f"  {row['feature']}: {row['importance']}")

# ---------------------------------------------------------------------------
# Step 13: Save everything
# ---------------------------------------------------------------------------
log.info("=" * 70)
log.info("STEP 13: Saving final artifacts")
log.info("=" * 70)

# Save final model
model_path = ROOT / "results" / "full_retrain_patient_model.joblib"
joblib.dump(
    {
        "model": final_model,
        "config": best_config,
        "feature_names": list(best_features.columns),
        "lgbm_params": LGBM_PARAMS,
        "n_datasets": len(datasets),
        "n_patients": len(y_final),
        "combat_applied": combat_success,
        "common_l1000_genes": common_l1000,
    },
    model_path,
)
log.info(f"Saved model: {model_path}")

# Headline metrics
best_auc_row = ablation_summary[
    ablation_summary["config"] == best_config
].iloc[0]
headline = pd.DataFrame(
    [
        {
            "metric": "n_datasets",
            "value": len(datasets),
        },
        {
            "metric": "n_patients",
            "value": len(y_final),
        },
        {
            "metric": "n_responders",
            "value": int(y_final.sum()),
        },
        {
            "metric": "n_nonresponders",
            "value": int((1 - y_final).sum()),
        },
        {
            "metric": "best_config",
            "value": best_config,
        },
        {
            "metric": "n_features",
            "value": int(best_auc_row["n_features"]),
        },
        {
            "metric": "mean_auc_lodo",
            "value": round(best_auc_row["mean_auc"], 4),
        },
        {
            "metric": "median_auc_lodo",
            "value": round(best_auc_row["median_auc"], 4),
        },
        {
            "metric": "std_auc_lodo",
            "value": round(best_auc_row["std_auc"], 4),
        },
        {
            "metric": "combat_applied",
            "value": combat_success,
        },
        {
            "metric": "ssgsea_computed",
            "value": ssgsea_scores is not None,
        },
        {
            "metric": "progeny_computed",
            "value": progeny_scores is not None,
        },
        {
            "metric": "n_l1000_genes",
            "value": len(common_l1000),
        },
    ]
)
headline.to_csv(OUT_DIR / "headline_metrics.tsv", sep="\t", index=False)
log.info(f"Saved headline_metrics.tsv")

log.info("\n" + "=" * 70)
log.info("PIPELINE COMPLETE")
log.info("=" * 70)
log.info(f"  Datasets:      {len(datasets)}")
log.info(f"  Patients:      {len(y_final)}")
log.info(f"  Best config:   {best_config}")
log.info(f"  Mean LODO AUC: {best_auc_row['mean_auc']:.4f}")
log.info(f"  ComBat:        {'Yes' if combat_success else 'No'}")
log.info(f"  ssGSEA:        {'Yes' if ssgsea_scores is not None else 'No'}")
log.info(f"  PROGENy:       {'Yes' if progeny_scores is not None else 'No'}")
log.info(f"  Output dir:    {OUT_DIR}")
log.info(f"  Model:         {model_path}")
