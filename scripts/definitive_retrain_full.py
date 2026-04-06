#!/usr/bin/env python
"""
INVEREX Definitive Retraining Pipeline (Full)
=============================================
Trains the patient response model using:
- 947 L1000 landmark genes (present in >=80% of datasets)
- Excludes 4 bottleneck datasets (<500 landmarks)
- NaN for missing genes (LightGBM handles natively)
- Per-dataset z-scoring (no batch effect leakage)
- Optuna HP optimization (50 trials, 5-fold CV)
- LODO evaluation with optimized HPs
- Stratified analysis by treatment class and endpoint
- Final model training on all data
- TCGA-BRCA treatability re-ranking
"""

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("definitive_retrain")

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "definitive_retrain_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CTRDB_DIR = DATA_RAW / "ctrdb"
ISPY2_DIR = DATA_RAW / "ispy2"
BRIGHTNESS_DIR = DATA_RAW / "brightness"

# Thresholds
MIN_LANDMARK_GENES = 500
MIN_PATIENTS = 20
COVERAGE_THRESHOLD = 0.80  # gene must be in >=80% of datasets
BOTTLENECK_DATASETS = {"GSE37138", "GSE61676", "GSE9782", "GSE62321"}

# ====================================================================
# STEP 1: Build the proper gene list
# ====================================================================
def step1_build_gene_list():
    """Get 978 L1000 landmarks, filter to those in >=80% of datasets."""
    log.info("=" * 70)
    log.info("STEP 1: Building proper gene list")
    log.info("=" * 70)

    # Load 978 L1000 landmarks
    gi = pd.read_parquet(DATA_CACHE / "GSE92742_gene_info.parquet")
    landmarks_978 = sorted(gi[gi["pr_is_lm"] == 1]["pr_gene_symbol"].unique().tolist())
    log.info("L1000 landmarks: %d genes", len(landmarks_978))

    # Survey all datasets for gene coverage
    dataset_genes = {}

    # CTR-DB datasets
    for gse in sorted(os.listdir(CTRDB_DIR)):
        if not gse.startswith("GSE"):
            continue
        expr_path = CTRDB_DIR / gse / f"{gse}_expression.parquet"
        if not expr_path.exists():
            continue
        if gse in BOTTLENECK_DATASETS:
            log.info("  Excluding bottleneck: %s", gse)
            continue
        expr = pd.read_parquet(expr_path)
        dataset_genes[gse] = set(expr.columns)

    # ISPY2
    ispy2_path = ISPY2_DIR / "GSE194040_expression.parquet"
    if ispy2_path.exists():
        expr = pd.read_parquet(ispy2_path)
        dataset_genes["ispy2"] = set(expr.columns)

    # BRIGHTNESS
    brightness_path = BRIGHTNESS_DIR / "GSE164458_expression.parquet"
    if brightness_path.exists():
        expr = pd.read_parquet(brightness_path)
        dataset_genes["brightness"] = set(expr.columns)

    n_datasets = len(dataset_genes)
    log.info("Total datasets surveyed (excl. bottlenecks): %d", n_datasets)

    # Filter landmarks: present in >=80% of datasets
    landmark_set = set(landmarks_978)
    gene_coverage = {}
    for gene in landmarks_978:
        count = sum(1 for ds_genes in dataset_genes.values() if gene in ds_genes)
        gene_coverage[gene] = count / n_datasets

    genes_filtered = sorted(
        [g for g, cov in gene_coverage.items() if cov >= COVERAGE_THRESHOLD]
    )
    log.info(
        "Genes with >=%d%% coverage: %d / %d",
        int(COVERAGE_THRESHOLD * 100),
        len(genes_filtered),
        len(landmarks_978),
    )

    # Save gene list
    gene_list_path = OUT_DIR / "gene_list.txt"
    with open(gene_list_path, "w") as f:
        f.write("gene_symbol\n")
        for g in genes_filtered:
            f.write(g + "\n")
    log.info("Saved gene list to %s", gene_list_path)

    # Update geneinfo_beta_input.txt
    geneinfo_path = DATA_CACHE / "geneinfo_beta_input.txt"
    with open(geneinfo_path, "w") as f:
        f.write("gene_symbol\n")
        for g in genes_filtered:
            f.write(g + "\n")
    log.info("Updated %s with %d genes", geneinfo_path, len(genes_filtered))

    # Coverage stats
    coverage_vals = list(gene_coverage.values())
    log.info(
        "Coverage stats: min=%.2f, median=%.2f, mean=%.2f",
        min(coverage_vals),
        np.median(coverage_vals),
        np.mean(coverage_vals),
    )

    return genes_filtered, dataset_genes


# ====================================================================
# STEP 2: Load ALL datasets with proper preprocessing
# ====================================================================
def step2_load_datasets(gene_list, dataset_genes_map):
    """Load all datasets, z-score per gene, NaN for missing genes."""
    log.info("=" * 70)
    log.info("STEP 2: Loading and preprocessing all datasets")
    log.info("=" * 70)

    gene_set = set(gene_list)

    # Load catalogs for drug/treatment info
    catalog_info = {}
    for cat_name in ["catalog.csv", "pan_cancer_catalog.csv"]:
        cat_path = CTRDB_DIR / cat_name
        if cat_path.exists():
            cat = pd.read_csv(cat_path)
            for _, row in cat.iterrows():
                gse = row["geo_source"]
                if gse not in catalog_info:
                    catalog_info[gse] = {
                        "drug": row.get("drug", "unknown"),
                        "response_grouping": row.get("response_grouping", ""),
                        "dataset_type": row.get("dataset_type", ""),
                    }

    datasets = []

    # ---- CTR-DB datasets ----
    for gse in sorted(os.listdir(CTRDB_DIR)):
        if not gse.startswith("GSE"):
            continue
        if gse in BOTTLENECK_DATASETS:
            continue

        expr_path = CTRDB_DIR / gse / f"{gse}_expression.parquet"
        resp_path = CTRDB_DIR / gse / "response_labels.parquet"
        if not expr_path.exists() or not resp_path.exists():
            continue

        try:
            expr = pd.read_parquet(expr_path)
            resp_df = pd.read_parquet(resp_path)

            # Get response column
            if "response" in resp_df.columns:
                resp_series = resp_df["response"]
            else:
                resp_series = resp_df.squeeze()

            # Align
            common_idx = expr.index.intersection(resp_series.index)
            if len(common_idx) < MIN_PATIENTS:
                log.info("  %s: skipped (only %d samples after alignment)", gse, len(common_idx))
                continue

            expr = expr.loc[common_idx]
            resp_series = resp_series.loc[common_idx]

            # Check class balance
            n_pos = int(resp_series.sum())
            n_neg = len(resp_series) - n_pos
            if n_pos < 3 or n_neg < 3:
                log.info("  %s: skipped (class imbalance: %d/%d)", gse, n_pos, n_neg)
                continue

            # Check gene coverage
            available_genes = sorted(gene_set & set(expr.columns))
            if len(available_genes) < MIN_LANDMARK_GENES:
                log.info("  %s: skipped (only %d landmark genes)", gse, len(available_genes))
                continue

            # Z-score per gene WITHIN the dataset
            expr_filtered = expr[available_genes].astype(np.float64)
            expr_z = (expr_filtered - expr_filtered.mean()) / expr_filtered.std()
            expr_z = expr_z.replace([np.inf, -np.inf], np.nan)

            # Build full matrix with NaN for missing genes
            full_matrix = pd.DataFrame(
                np.nan, index=expr.index, columns=gene_list, dtype=np.float64
            )
            full_matrix[available_genes] = expr_z[available_genes].values

            # Clinical features (ER/HER2/PR)
            clinical = _extract_clinical_ctrdb(resp_df, common_idx)

            # Get drug/treatment info from catalog
            info = catalog_info.get(gse, {})
            drug_string = info.get("drug", "unknown")
            treatment_class = _classify_treatment(drug_string)
            endpoint_family = _classify_endpoint(info.get("response_grouping", ""))

            datasets.append({
                "dataset_id": gse,
                "source": "ctrdb",
                "X": full_matrix,
                "y": resp_series.values.astype(int),
                "n_genes_available": len(available_genes),
                "n_patients": len(common_idx),
                "n_responders": n_pos,
                "drug": drug_string,
                "treatment_class": treatment_class,
                "endpoint_family": endpoint_family,
                "clinical": clinical,
            })
            log.info(
                "  %s: %d patients, %d/%d genes, resp=%d/%d, drug=%s, class=%s",
                gse, len(common_idx), len(available_genes), len(gene_list),
                n_pos, len(common_idx), drug_string[:40], treatment_class,
            )

        except Exception as e:
            log.warning("  %s: failed to load (%s)", gse, e)

    # ---- ISPY2 ----
    ispy2_expr_path = ISPY2_DIR / "GSE194040_expression.parquet"
    ispy2_resp_path = ISPY2_DIR / "response_labels.parquet"
    if ispy2_expr_path.exists() and ispy2_resp_path.exists():
        try:
            expr = pd.read_parquet(ispy2_expr_path)
            resp_df = pd.read_parquet(ispy2_resp_path)

            # ISPY2: reset indices for alignment (expression uses internal IDs,
            # labels use integer index)
            expr = expr.reset_index(drop=True)
            resp_df = resp_df.reset_index(drop=True)

            resp_series = resp_df["response"]
            common_idx = expr.index.intersection(resp_series.index)
            expr = expr.loc[common_idx]
            resp_series = resp_series.loc[common_idx]

            available_genes = sorted(gene_set & set(expr.columns))
            n_pos = int(resp_series.sum())
            n_neg = len(resp_series) - n_pos

            if len(available_genes) >= MIN_LANDMARK_GENES and n_pos >= 3 and n_neg >= 3:
                expr_filtered = expr[available_genes].astype(np.float64)
                expr_z = (expr_filtered - expr_filtered.mean()) / expr_filtered.std()
                expr_z = expr_z.replace([np.inf, -np.inf], np.nan)

                full_matrix = pd.DataFrame(
                    np.nan, index=common_idx, columns=gene_list, dtype=np.float64
                )
                full_matrix[available_genes] = expr_z[available_genes].values

                # ISPY2 has clinical features
                clinical = pd.DataFrame(index=common_idx)
                if "char_hr" in resp_df.columns:
                    clinical["ER_status"] = pd.to_numeric(
                        resp_df.loc[common_idx, "char_hr"], errors="coerce"
                    )
                if "char_her2" in resp_df.columns:
                    clinical["HER2_status"] = pd.to_numeric(
                        resp_df.loc[common_idx, "char_her2"], errors="coerce"
                    )

                # Determine treatment from arm info
                drug_str = "Paclitaxel+combination (ISPY2)"
                if "char_arm" in resp_df.columns:
                    arms = resp_df.loc[common_idx, "char_arm"].unique()
                    drug_str = f"ISPY2 arms: {', '.join(str(a) for a in arms[:5])}"

                datasets.append({
                    "dataset_id": "ispy2",
                    "source": "ispy2",
                    "X": full_matrix,
                    "y": resp_series.values.astype(int),
                    "n_genes_available": len(available_genes),
                    "n_patients": len(common_idx),
                    "n_responders": n_pos,
                    "drug": drug_str,
                    "treatment_class": "combination",
                    "endpoint_family": "pathologic",
                    "clinical": clinical,
                })
                log.info(
                    "  ispy2: %d patients, %d/%d genes, resp=%d/%d",
                    len(common_idx), len(available_genes), len(gene_list),
                    n_pos, len(common_idx),
                )
        except Exception as e:
            log.warning("  ispy2: failed to load (%s)", e)

    # ---- BRIGHTNESS ----
    bright_expr_path = BRIGHTNESS_DIR / "GSE164458_expression.parquet"
    bright_resp_path = BRIGHTNESS_DIR / "response_labels.parquet"
    if bright_expr_path.exists() and bright_resp_path.exists():
        try:
            expr = pd.read_parquet(bright_expr_path)
            resp_df = pd.read_parquet(bright_resp_path)

            # Reset indices for alignment
            expr = expr.reset_index(drop=True)
            resp_df = resp_df.reset_index(drop=True)

            resp_series = resp_df["response"]
            common_idx = expr.index.intersection(resp_series.index)
            expr = expr.loc[common_idx]
            resp_series = resp_series.loc[common_idx]

            available_genes = sorted(gene_set & set(expr.columns))
            n_pos = int(resp_series.sum())
            n_neg = len(resp_series) - n_pos

            if len(available_genes) >= MIN_LANDMARK_GENES and n_pos >= 3 and n_neg >= 3:
                expr_filtered = expr[available_genes].astype(np.float64)
                expr_z = (expr_filtered - expr_filtered.mean()) / expr_filtered.std()
                expr_z = expr_z.replace([np.inf, -np.inf], np.nan)

                full_matrix = pd.DataFrame(
                    np.nan, index=common_idx, columns=gene_list, dtype=np.float64
                )
                full_matrix[available_genes] = expr_z[available_genes].values

                # BRIGHTNESS has limited clinical info
                clinical = pd.DataFrame(index=common_idx)

                datasets.append({
                    "dataset_id": "brightness",
                    "source": "brightness",
                    "X": full_matrix,
                    "y": resp_series.values.astype(int),
                    "n_genes_available": len(available_genes),
                    "n_patients": len(common_idx),
                    "n_responders": n_pos,
                    "drug": "Veliparib+Carboplatin+Paclitaxel (BRIGHTNESS)",
                    "treatment_class": "parp",
                    "endpoint_family": "pathologic",
                    "clinical": clinical,
                })
                log.info(
                    "  brightness: %d patients, %d/%d genes, resp=%d/%d",
                    len(common_idx), len(available_genes), len(gene_list),
                    n_pos, len(common_idx),
                )
        except Exception as e:
            log.warning("  brightness: failed to load (%s)", e)

    log.info("\nTotal datasets loaded: %d", len(datasets))
    total_patients = sum(d["n_patients"] for d in datasets)
    log.info("Total patients: %d", total_patients)

    return datasets


def _extract_clinical_ctrdb(resp_df, common_idx):
    """Extract ER/HER2/PR clinical features from CTR-DB phenotype columns."""
    clinical = pd.DataFrame(index=common_idx)

    # Look for ER/HER2/PR in characteristics columns
    char_cols = [c for c in resp_df.columns if "char" in c.lower() or "characteristics" in c.lower()]
    for col in char_cols:
        col_lower = col.lower()
        if "er_status" in col_lower or "estrogen" in col_lower:
            vals = resp_df.loc[common_idx, col] if col in resp_df.columns else None
            if vals is not None:
                clinical["ER_status"] = _parse_binary_status(vals)
        elif "her2" in col_lower:
            vals = resp_df.loc[common_idx, col] if col in resp_df.columns else None
            if vals is not None:
                clinical["HER2_status"] = _parse_binary_status(vals)
        elif "pr_status" in col_lower or "progesterone" in col_lower:
            vals = resp_df.loc[common_idx, col] if col in resp_df.columns else None
            if vals is not None:
                clinical["PR_status"] = _parse_binary_status(vals)
        elif col_lower in ("char_hr",):
            vals = resp_df.loc[common_idx, col] if col in resp_df.columns else None
            if vals is not None:
                clinical["ER_status"] = _parse_binary_status(vals)

    return clinical


def _parse_binary_status(series):
    """Parse a series to binary (1=positive, 0=negative, NaN=unknown)."""
    result = pd.Series(np.nan, index=series.index)
    s = series.astype(str).str.lower().str.strip()
    result[s.isin(["1", "positive", "pos", "+", "yes", "true"])] = 1.0
    result[s.isin(["0", "negative", "neg", "-", "no", "false"])] = 0.0
    return result


def _classify_treatment(drug_string):
    """Classify treatment into treatment class."""
    d = drug_string.lower()
    if any(w in d for w in ["tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant"]):
        return "endocrine"
    if any(w in d for w in ["trastuzumab", "lapatinib", "pertuzumab", "neratinib", "t-dm1"]):
        if any(w in d for w in ["paclitaxel", "docetaxel", "doxorubicin", "cyclophosphamide"]):
            return "targeted+chemo"
        return "targeted"
    if any(w in d for w in ["veliparib", "olaparib", "talazoparib", "niraparib", "rucaparib"]):
        return "parp"
    if any(w in d for w in ["pembrolizumab", "atezolizumab", "nivolumab", "durvalumab", "ipilimumab"]):
        return "immunotherapy"
    if any(w in d for w in [
        "paclitaxel", "docetaxel", "doxorubicin", "epirubicin", "cyclophosphamide",
        "fluorouracil", "cisplatin", "carboplatin", "methotrexate", "capecitabine",
        "ixabepilone", "taxane", "anthracycline", "fac", "fec", "cef",
        "gemcitabine", "oxaliplatin", "irinotecan", "vincristine",
    ]):
        return "chemo"
    if "mk-2206" in d or "ganitumab" in d or "ganetespib" in d or "capivasertib" in d:
        return "targeted"
    if "+" in drug_string or "combination" in d:
        return "combination"
    return "other"


def _classify_endpoint(response_grouping):
    """Classify response endpoint into family."""
    r = response_grouping.lower() if response_grouping else ""
    if any(w in r for w in ["pcr", "rcb", "pathologic"]):
        return "pathologic"
    if any(w in r for w in ["cr", "pr", "sd", "pd", "recist"]):
        return "pharmacodynamic"
    if any(w in r for w in ["survival", "os", "pfs", "dfs"]):
        return "survival"
    if any(w in r for w in ["radiographic", "imaging"]):
        return "radiographic"
    return "pathologic"  # default for breast cancer neoadjuvant


# ====================================================================
# STEP 3: Optuna HP optimization
# ====================================================================
def step3_optuna_optimization(datasets, gene_list):
    """Optimize LightGBM hyperparameters using Optuna with 5-fold CV."""
    log.info("=" * 70)
    log.info("STEP 3: Optuna HP optimization (50 trials, 5-fold CV)")
    log.info("=" * 70)

    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pool all data
    X_all = np.vstack([d["X"].values for d in datasets])
    y_all = np.concatenate([d["y"] for d in datasets])

    log.info("Pooled data: %d samples, %d features", X_all.shape[0], X_all.shape[1])
    log.info("Class balance: %d responders, %d non-responders",
             int(y_all.sum()), int((1 - y_all).sum()))

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        for train_idx, val_idx in skf.split(X_all, y_all):
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

            try:
                proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, proba)
                aucs.append(auc)
            except Exception:
                aucs.append(0.5)

        return np.mean(aucs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    best_params["objective"] = "binary"
    best_params["metric"] = "auc"
    best_params["random_state"] = 42
    best_params["verbose"] = -1

    log.info("Best trial: AUC=%.4f", study.best_value)
    log.info("Best params: %s", json.dumps(best_params, indent=2))

    # Save best params
    params_path = OUT_DIR / "optuna_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Saved best params to %s", params_path)

    # Check for leakage
    if study.best_value > 0.75:
        log.warning(
            "WARNING: Best 5-fold CV AUC=%.4f > 0.75. "
            "This is pooled CV (not LODO), so some inflation is expected from "
            "intra-dataset correlation. Will verify with LODO.",
            study.best_value,
        )

    return best_params


# ====================================================================
# STEP 4: LODO evaluation
# ====================================================================
def step4_lodo_evaluation(datasets, gene_list, best_params):
    """Full LODO evaluation with configs A (genes only) and B (genes+clinical)."""
    log.info("=" * 70)
    log.info("STEP 4: LODO evaluation with optimized HPs")
    log.info("=" * 70)

    import lightgbm as lgb

    results = []
    dataset_ids = [d["dataset_id"] for d in datasets]

    for i, held_out in enumerate(datasets):
        held_out_id = held_out["dataset_id"]
        log.info("  LODO fold %d/%d: holding out %s (%d patients)",
                 i + 1, len(datasets), held_out_id, held_out["n_patients"])

        # Build training data (all except held-out)
        X_train_list = []
        y_train_list = []
        X_train_clin_list = []

        for d in datasets:
            if d["dataset_id"] == held_out_id:
                continue
            X_train_list.append(d["X"].values)
            y_train_list.append(d["y"])

            # Clinical features
            clin = d["clinical"]
            clin_matrix = _build_clinical_matrix(clin, len(d["y"]))
            X_train_clin_list.append(clin_matrix)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_train_clin = np.vstack(X_train_clin_list)

        X_test = held_out["X"].values
        y_test = held_out["y"]
        clin_test = _build_clinical_matrix(held_out["clinical"], len(y_test))

        # Config A: genes only
        auc_a = _train_and_predict(X_train, y_train, X_test, y_test, best_params)

        # Config B: genes + clinical
        X_train_b = np.hstack([X_train, X_train_clin])
        X_test_b = np.hstack([X_test, clin_test])
        auc_b = _train_and_predict(X_train_b, y_train, X_test_b, y_test, best_params)

        results.append({
            "dataset": held_out_id,
            "source": held_out["source"],
            "n": held_out["n_patients"],
            "resp": held_out["n_responders"],
            "resp_rate": round(held_out["n_responders"] / held_out["n_patients"], 3),
            "n_genes": held_out["n_genes_available"],
            "auc_A_genes_only": round(auc_a, 4),
            "auc_B_genes_clinical": round(auc_b, 4),
            "treatment_class": held_out["treatment_class"],
            "endpoint_family": held_out["endpoint_family"],
            "drug": held_out["drug"],
            "has_clinical": not held_out["clinical"].empty and held_out["clinical"].notna().any().any(),
        })

        log.info(
            "    AUC: A=%.4f, B=%.4f | n=%d, resp=%d, class=%s",
            auc_a, auc_b, held_out["n_patients"],
            held_out["n_responders"], held_out["treatment_class"],
        )

        # Leakage check
        for config_name, auc_val in [("A", auc_a), ("B", auc_b)]:
            if auc_val > 0.75:
                log.warning(
                    "    LEAKAGE CHECK: %s AUC=%.4f > 0.75 for %s. "
                    "Investigating... n=%d, resp_rate=%.2f",
                    config_name, auc_val, held_out_id,
                    held_out["n_patients"],
                    held_out["n_responders"] / held_out["n_patients"],
                )

    lodo_df = pd.DataFrame(results)

    # Save per-dataset results
    lodo_df.to_csv(OUT_DIR / "lodo_per_dataset.csv", index=False)
    log.info("Saved per-dataset LODO results")

    # Summary
    summary = pd.DataFrame({
        "config": ["A_genes_only", "B_genes_clinical"],
        "mean_auc": [
            lodo_df["auc_A_genes_only"].mean(),
            lodo_df["auc_B_genes_clinical"].mean(),
        ],
        "median_auc": [
            lodo_df["auc_A_genes_only"].median(),
            lodo_df["auc_B_genes_clinical"].median(),
        ],
        "std_auc": [
            lodo_df["auc_A_genes_only"].std(),
            lodo_df["auc_B_genes_clinical"].std(),
        ],
        "n_datasets": [len(lodo_df), len(lodo_df)],
    }).round(4)

    summary.to_csv(OUT_DIR / "lodo_summary.csv", index=False)
    log.info("\nLODO Summary:")
    log.info(summary.to_string(index=False))

    return lodo_df


def _build_clinical_matrix(clinical_df, n_patients):
    """Build a fixed-width clinical feature matrix (ER, HER2, PR)."""
    matrix = np.full((n_patients, 3), np.nan)  # ER, HER2, PR
    if clinical_df is not None and not clinical_df.empty:
        if "ER_status" in clinical_df.columns:
            matrix[:, 0] = clinical_df["ER_status"].values[:n_patients]
        if "HER2_status" in clinical_df.columns:
            matrix[:, 1] = clinical_df["HER2_status"].values[:n_patients]
        if "PR_status" in clinical_df.columns:
            matrix[:, 2] = clinical_df["PR_status"].values[:n_patients]
    return matrix


def _train_and_predict(X_train, y_train, X_test, y_test, params):
    """Train LightGBM and return AUC on test set."""
    import lightgbm as lgb

    try:
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        return auc
    except Exception as e:
        log.warning("    Train/predict failed: %s", e)
        return 0.5


# ====================================================================
# STEP 5: Stratified evaluation
# ====================================================================
def step5_stratified_evaluation(lodo_df):
    """Report LODO AUC stratified by treatment class and endpoint family."""
    log.info("=" * 70)
    log.info("STEP 5: Stratified evaluation")
    log.info("=" * 70)

    # By treatment class
    by_treatment = (
        lodo_df.groupby("treatment_class")
        .agg(
            n_datasets=("dataset", "count"),
            total_patients=("n", "sum"),
            mean_auc_A=("auc_A_genes_only", "mean"),
            median_auc_A=("auc_A_genes_only", "median"),
            mean_auc_B=("auc_B_genes_clinical", "mean"),
            median_auc_B=("auc_B_genes_clinical", "median"),
        )
        .round(4)
        .reset_index()
    )
    by_treatment.to_csv(OUT_DIR / "lodo_by_treatment_class.csv", index=False)
    log.info("\nBy treatment class:")
    log.info(by_treatment.to_string(index=False))

    # By endpoint family
    by_endpoint = (
        lodo_df.groupby("endpoint_family")
        .agg(
            n_datasets=("dataset", "count"),
            total_patients=("n", "sum"),
            mean_auc_A=("auc_A_genes_only", "mean"),
            median_auc_A=("auc_A_genes_only", "median"),
            mean_auc_B=("auc_B_genes_clinical", "mean"),
            median_auc_B=("auc_B_genes_clinical", "median"),
        )
        .round(4)
        .reset_index()
    )
    by_endpoint.to_csv(OUT_DIR / "lodo_by_endpoint_family.csv", index=False)
    log.info("\nBy endpoint family:")
    log.info(by_endpoint.to_string(index=False))

    return by_treatment, by_endpoint


# ====================================================================
# STEP 6: Train final model on ALL data
# ====================================================================
def step6_train_final_model(datasets, gene_list, best_params, lodo_df):
    """Train the final model on all patients with best config."""
    log.info("=" * 70)
    log.info("STEP 6: Training final model on ALL data")
    log.info("=" * 70)

    import lightgbm as lgb

    # Determine best config
    mean_a = lodo_df["auc_A_genes_only"].mean()
    mean_b = lodo_df["auc_B_genes_clinical"].mean()

    if mean_b > mean_a + 0.005:
        best_config = "B"
        log.info("Best config: B (genes+clinical), AUC=%.4f vs A=%.4f", mean_b, mean_a)
    else:
        best_config = "A"
        log.info("Best config: A (genes only), AUC=%.4f vs B=%.4f", mean_a, mean_b)

    # Build full training data
    if best_config == "A":
        X_all = np.vstack([d["X"].values for d in datasets])
        feature_names = list(gene_list)
    else:
        X_gene_list = [d["X"].values for d in datasets]
        X_clin_list = [_build_clinical_matrix(d["clinical"], d["n_patients"]) for d in datasets]
        X_all = np.hstack([np.vstack(X_gene_list), np.vstack(X_clin_list)])
        feature_names = list(gene_list) + ["ER_status", "HER2_status", "PR_status"]

    y_all = np.concatenate([d["y"] for d in datasets])

    log.info("Training on %d patients, %d features", X_all.shape[0], X_all.shape[1])

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_all, y_all)

    # Save model
    model_path = RESULTS / "definitive_patient_model.joblib"
    joblib.dump({
        "model": model,
        "gene_list": gene_list,
        "feature_names": feature_names,
        "best_config": best_config,
        "best_params": best_params,
        "n_training_patients": len(y_all),
        "n_features": X_all.shape[1],
    }, model_path)
    log.info("Saved final model to %s", model_path)

    # Feature importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(OUT_DIR / "feature_importances.csv", index=False)
    log.info("\nTop 20 features:")
    log.info(feat_imp.head(20).to_string(index=False))

    return model, feature_names, best_config


# ====================================================================
# STEP 7: Re-rank TCGA patients
# ====================================================================
def step7_rerank_tcga(model_bundle, gene_list, best_config):
    """Use the new model to update treatability scores for TCGA-BRCA patients."""
    log.info("=" * 70)
    log.info("STEP 7: Re-ranking TCGA-BRCA patients")
    log.info("=" * 70)

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    # Load TCGA expression
    tcga_expr = pd.read_parquet(DATA_CACHE / "tcga_brca_expression.parquet")
    tcga_clinical = pd.read_parquet(DATA_CACHE / "tcga_brca_clinical.parquet")

    log.info("TCGA expression: %d patients, %d genes", tcga_expr.shape[0], tcga_expr.shape[1])

    # Filter to gene list
    gene_set = set(gene_list)
    available_genes = sorted(gene_set & set(tcga_expr.columns))
    log.info("Available landmarks in TCGA: %d / %d", len(available_genes), len(gene_list))

    # Z-score within TCGA cohort
    expr_filtered = tcga_expr[available_genes].astype(np.float64)
    expr_z = (expr_filtered - expr_filtered.mean()) / expr_filtered.std()
    expr_z = expr_z.replace([np.inf, -np.inf], np.nan)

    # Build full matrix
    full_matrix = pd.DataFrame(
        np.nan, index=tcga_expr.index, columns=gene_list, dtype=np.float64
    )
    full_matrix[available_genes] = expr_z[available_genes].values

    if best_config == "B":
        # Add clinical features
        clin_matrix = np.full((len(tcga_expr), 3), np.nan)

        # Get ER status
        common_patients = tcga_expr.index.intersection(tcga_clinical.index)
        if "ER_Status_nature2012" in tcga_clinical.columns:
            er = tcga_clinical.loc[common_patients, "ER_Status_nature2012"]
            er_binary = pd.Series(np.nan, index=tcga_expr.index)
            er_binary[common_patients] = (er == "Positive").astype(float)
            clin_matrix[:, 0] = er_binary.values

        if "HER2_Final_Status_nature2012" in tcga_clinical.columns:
            her2 = tcga_clinical.loc[common_patients, "HER2_Final_Status_nature2012"]
            her2_binary = pd.Series(np.nan, index=tcga_expr.index)
            her2_binary[common_patients] = (her2 == "Positive").astype(float)
            clin_matrix[:, 1] = her2_binary.values

        if "PR_Status_nature2012" in tcga_clinical.columns:
            pr = tcga_clinical.loc[common_patients, "PR_Status_nature2012"]
            pr_binary = pd.Series(np.nan, index=tcga_expr.index)
            pr_binary[common_patients] = (pr == "Positive").astype(float)
            clin_matrix[:, 2] = pr_binary.values

        X_tcga = np.hstack([full_matrix.values, clin_matrix])
    else:
        X_tcga = full_matrix.values

    # Predict treatability scores
    new_scores = model.predict_proba(X_tcga)[:, 1]

    # Get PAM50 subtypes
    pam50_col = "PAM50Call_RNAseq"
    if pam50_col in tcga_clinical.columns:
        subtypes = tcga_clinical.reindex(tcga_expr.index)[pam50_col]
    else:
        subtypes = pd.Series("Unknown", index=tcga_expr.index)

    tcga_results = pd.DataFrame({
        "patient_id": tcga_expr.index,
        "new_treatability_score": new_scores,
        "pam50_subtype": subtypes.values,
    })

    # Try to load old scores for comparison
    old_scores = None
    old_model_path = RESULTS / "full_retrain_patient_model.joblib"
    if old_model_path.exists():
        try:
            old_bundle = joblib.load(old_model_path)
            if isinstance(old_bundle, dict) and "model" in old_bundle:
                old_model = old_bundle["model"]
                old_gene_list = old_bundle.get("gene_list", old_bundle.get("feature_names", []))
                # Build old features
                old_available = sorted(set(old_gene_list) & set(tcga_expr.columns))
                if len(old_available) > 10:
                    old_expr = tcga_expr[old_available].astype(np.float64)
                    old_z = (old_expr - old_expr.mean()) / old_expr.std()
                    old_z = old_z.replace([np.inf, -np.inf], np.nan)
                    old_full = pd.DataFrame(
                        np.nan, index=tcga_expr.index, columns=old_gene_list, dtype=np.float64
                    )
                    for g in old_available:
                        if g in old_full.columns:
                            old_full[g] = old_z[g].values
                    X_old = old_full.values
                    n_feats_expected = old_model.n_features_in_ if hasattr(old_model, 'n_features_in_') else X_old.shape[1]
                    if X_old.shape[1] == n_feats_expected:
                        old_scores = old_model.predict_proba(X_old)[:, 1]
                    else:
                        log.info("Old model feature mismatch: %d vs %d", X_old.shape[1], n_feats_expected)
            else:
                # It might be just the model object
                old_model = old_bundle
                log.info("Old model is not a dict bundle, skipping comparison")
        except Exception as e:
            log.warning("Could not load old model for comparison: %s", e)

    if old_scores is not None:
        tcga_results["old_treatability_score"] = old_scores
        tcga_results["score_change"] = new_scores - old_scores

    # Comparison by subtype
    comparison = (
        tcga_results.groupby("pam50_subtype")
        .agg(
            n_patients=("patient_id", "count"),
            new_mean=("new_treatability_score", "mean"),
            new_std=("new_treatability_score", "std"),
            new_median=("new_treatability_score", "median"),
        )
        .round(4)
        .reset_index()
    )

    if old_scores is not None:
        old_comparison = (
            tcga_results.groupby("pam50_subtype")
            .agg(
                old_mean=("old_treatability_score", "mean"),
                old_median=("old_treatability_score", "median"),
            )
            .round(4)
            .reset_index()
        )
        comparison = comparison.merge(old_comparison, on="pam50_subtype")

    comparison.to_csv(OUT_DIR / "treatability_comparison.csv", index=False)
    log.info("\nTreatability comparison by PAM50 subtype:")
    log.info(comparison.to_string(index=False))

    return tcga_results, comparison


# ====================================================================
# STEP 8: Summary report
# ====================================================================
def step8_summary_report(
    gene_list, datasets, best_params, lodo_df,
    by_treatment, by_endpoint, tcga_comparison,
    best_config, model_bundle,
):
    """Write comprehensive summary report."""
    log.info("=" * 70)
    log.info("STEP 8: Writing summary report")
    log.info("=" * 70)

    total_patients = sum(d["n_patients"] for d in datasets)
    total_responders = sum(d["n_responders"] for d in datasets)

    mean_a = lodo_df["auc_A_genes_only"].mean()
    mean_b = lodo_df["auc_B_genes_clinical"].mean()
    median_a = lodo_df["auc_A_genes_only"].median()
    median_b = lodo_df["auc_B_genes_clinical"].median()

    # Previous results for comparison
    prev_results = {}
    prev_path = RESULTS / "definitive_retrain" / "summary.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        for _, row in prev.iterrows():
            prev_results[row["config"]] = row["mean_auc"]

    excluded_datasets = sorted(BOTTLENECK_DATASETS)

    report = f"""# INVEREX Definitive Retraining Report (Full)

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 1. Gene Set

- **Total L1000 landmarks**: 978
- **Genes with >={int(COVERAGE_THRESHOLD*100)}% dataset coverage**: {len(gene_list)}
- **Coverage threshold**: {COVERAGE_THRESHOLD*100:.0f}% of non-bottleneck datasets

## 2. Datasets

- **Total datasets loaded**: {len(datasets)}
- **Total patients**: {total_patients}
- **Total responders**: {total_responders} ({total_responders/total_patients*100:.1f}%)
- **Excluded (bottleneck, <{MIN_LANDMARK_GENES} landmarks)**: {', '.join(excluded_datasets)}
- **Sources**: CTR-DB ({sum(1 for d in datasets if d['source']=='ctrdb')}), ISPY2 ({sum(1 for d in datasets if d['source']=='ispy2')}), BRIGHTNESS ({sum(1 for d in datasets if d['source']=='brightness')})

### Dataset Details

| Dataset | Source | N | Resp | Resp Rate | Genes | Treatment | Endpoint |
|---------|--------|---|------|-----------|-------|-----------|----------|
"""
    for d in sorted(datasets, key=lambda x: x["n_patients"], reverse=True):
        report += (
            f"| {d['dataset_id']} | {d['source']} | {d['n_patients']} | "
            f"{d['n_responders']} | {d['n_responders']/d['n_patients']:.2f} | "
            f"{d['n_genes_available']} | {d['treatment_class']} | "
            f"{d['endpoint_family']} |\n"
        )

    report += f"""
## 3. Optuna HP Optimization

- **Trials**: 50
- **CV Strategy**: 5-fold stratified on pooled data
- **Best params**:

```json
{json.dumps(best_params, indent=2)}
```

## 4. LODO Evaluation Results

### Overall

| Config | Mean AUC | Median AUC | Std AUC | N datasets |
|--------|----------|------------|---------|------------|
| A: Genes only ({len(gene_list)} genes) | {mean_a:.4f} | {median_a:.4f} | {lodo_df['auc_A_genes_only'].std():.4f} | {len(lodo_df)} |
| B: Genes + Clinical | {mean_b:.4f} | {median_b:.4f} | {lodo_df['auc_B_genes_clinical'].std():.4f} | {len(lodo_df)} |

**Best config**: {best_config} ({'Genes only' if best_config == 'A' else 'Genes + Clinical'})

### By Treatment Class

{by_treatment.to_csv(index=False)}

### By Endpoint Family

{by_endpoint.to_csv(index=False)}

## 5. Comparison to Previous Runs

| Run | Mean AUC | N Features | Notes |
|-----|----------|------------|-------|
| Previous 212 genes | {prev_results.get('A_212_genes_only', 'N/A')} | 212 | Original curated list |
| Previous 978 genes | {prev_results.get('previous_978_genes', 'N/A')} | 978 | All landmarks, all datasets |
| Previous 212 + clinical | {prev_results.get('D_genes+clinical', 'N/A')} | 215 | 212 genes + ER/HER2/PR |
| **This run: A ({len(gene_list)} genes)** | **{mean_a:.4f}** | **{len(gene_list)}** | **Filtered landmarks, no bottlenecks, NaN fill** |
| **This run: B ({len(gene_list)} genes + clinical)** | **{mean_b:.4f}** | **{len(gene_list)+3}** | **+ ER/HER2/PR** |

## 6. Treatability Score Distribution (TCGA-BRCA)

{tcga_comparison.to_csv(index=False)}

## 7. Key Observations

"""
    # Leakage check
    high_auc = lodo_df[lodo_df["auc_A_genes_only"] > 0.75]
    if len(high_auc) > 0:
        report += f"### Datasets with AUC > 0.75 (leakage check)\n\n"
        for _, row in high_auc.iterrows():
            report += (
                f"- **{row['dataset']}**: AUC={row['auc_A_genes_only']:.4f}, "
                f"n={row['n']}, resp_rate={row['resp_rate']:.2f}, "
                f"drug={row['drug'][:50]}\n"
            )
        report += "\n"

    report += f"""
## 8. Model Details

- **Algorithm**: LightGBM (binary classification)
- **Training patients**: {total_patients}
- **Features**: {model_bundle['n_features']}
- **Config**: {best_config}
- **Model path**: `results/definitive_patient_model.joblib`

## 9. Output Files

```
results/definitive_retrain_full/
  gene_list.txt                  -- {len(gene_list)} L1000 landmarks (>={int(COVERAGE_THRESHOLD*100)}% coverage)
  optuna_best_params.json        -- Optimized HPs (50 Optuna trials)
  lodo_per_dataset.csv           -- Per-dataset AUC for configs A and B
  lodo_summary.csv               -- Mean AUC comparison
  lodo_by_treatment_class.csv    -- Stratified by treatment class
  lodo_by_endpoint_family.csv    -- Stratified by endpoint family
  treatability_comparison.csv    -- Old vs new scores by PAM50 subtype
  feature_importances.csv        -- Top features from final model
  summary.md                     -- This report
results/definitive_patient_model.joblib  -- Final trained model
```
"""

    report_path = OUT_DIR / "summary.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved summary report to %s", report_path)


# ====================================================================
# MAIN
# ====================================================================
def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("INVEREX DEFINITIVE RETRAINING PIPELINE (FULL)")
    log.info("=" * 70)

    # Step 1: Build gene list
    gene_list, dataset_genes_map = step1_build_gene_list()

    # Step 2: Load datasets
    datasets = step2_load_datasets(gene_list, dataset_genes_map)

    if len(datasets) < 5:
        log.error("Only %d datasets loaded. Need at least 5 for meaningful LODO.", len(datasets))
        return

    # Step 3: Optuna optimization
    best_params = step3_optuna_optimization(datasets, gene_list)

    # Step 4: LODO evaluation
    lodo_df = step4_lodo_evaluation(datasets, gene_list, best_params)

    # Step 5: Stratified evaluation
    by_treatment, by_endpoint = step5_stratified_evaluation(lodo_df)

    # Step 6: Train final model
    model, feature_names, best_config = step6_train_final_model(
        datasets, gene_list, best_params, lodo_df
    )
    model_bundle = {
        "model": model,
        "gene_list": gene_list,
        "feature_names": feature_names,
        "best_config": best_config,
        "n_features": len(feature_names),
    }

    # Step 7: Re-rank TCGA
    tcga_results, tcga_comparison = step7_rerank_tcga(
        model_bundle, gene_list, best_config
    )

    # Step 8: Summary report
    step8_summary_report(
        gene_list, datasets, best_params, lodo_df,
        by_treatment, by_endpoint, tcga_comparison,
        best_config, model_bundle,
    )

    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE in %.1f minutes", elapsed / 60)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
