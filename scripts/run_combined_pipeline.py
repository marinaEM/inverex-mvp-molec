#!/usr/bin/env python
"""
Combined pipeline using ALL winning improvements from the INVEREX feature sprint.

Winners combined:
  1. ComBat batch correction                    (AUC +0.017)
  2. Pathway features (Hallmark ssGSEA)         (AUC +0.091)
  3. ChemBERTa embeddings                       (CV RMSE -3.57 vs ECFP)
  4. Dose-aware signatures                      (AUC +0.017)
  5. Shallow NN                                 (CV RMSE -2.53 vs LightGBM)

Steps:
  1. Load & ComBat-correct CTR-DB patient data
  2. Compute pathway features for patients and LINCS signatures
  3. Compute dose-aware reversal features
  4. Build combined reversal feature matrix
  5. LODO evaluation with L1-logistic regression
  6. Cell-line model evaluation (LightGBM + NN) with ChemBERTa + pathways
  7. Test combined cell-line model on CTR-DB patients
  8. Full comparison table

Usage:
    pixi run python scripts/run_combined_pipeline.py
"""

import gc
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

# Prevent OMP duplicate-library crash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_CACHE,
    DATA_PROCESSED,
    DATA_RAW,
    LIGHTGBM_DEFAULT_PARAMS,
    RANDOM_SEED,
    RESULTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("combined_pipeline")

# Suppress convergence warnings during CV
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Constants ─────────────────────────────────────────────────────────────
META_COLS = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
CTRDB_DIR = DATA_RAW / "ctrdb"
LODO_C = 0.05


# =========================================================================
# Drug-name parsing utilities (from evaluate_batch_correction.py)
# =========================================================================
DRUG_ALIASES = {
    "5-fluorouracil": "fluorouracil",
    "5-fu": "fluorouracil",
    "5fu": "fluorouracil",
    "adriamycin": "doxorubicin",
    "taxol": "paclitaxel",
    "nolvadex": "tamoxifen",
    "xeloda": "capecitabine",
    "taxotere": "docetaxel",
    "gemzar": "gemcitabine",
    "mtx": "methotrexate",
    "ctx": "cyclophosphamide",
    "cytoxan": "cyclophosphamide",
    "epirubicin": "epirubicin",
    "ixabepilone": "ixabepilone",
    "mk-2206": "MK-2206",
    "trastuzumab": "trastuzumab",
}


def _normalise_drug_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[\s\-]+", "", s)
    for alias, canonical in DRUG_ALIASES.items():
        if s == re.sub(r"[\s\-]+", "", alias):
            return canonical.lower()
    return s


def parse_regimen_components(drug_string: str) -> list[str]:
    s = drug_string.strip()
    paren_match = re.search(r"\(([^)]+)\)", s)
    inner = paren_match.group(1) if paren_match else s
    parts = re.split(r"[+/]", inner)
    components = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^[A-Z]{1,5}\s+", "", p)
        norm = _normalise_drug_name(p)
        if norm and len(norm) > 1:
            components.append(norm)
    return list(dict.fromkeys(components))


def match_drugs_to_lincs(components: list[str], lincs_drug_set: set[str]) -> list[str]:
    lincs_norm = {_normalise_drug_name(d): d for d in lincs_drug_set}
    matched = []
    for comp in components:
        cn = _normalise_drug_name(comp)
        if cn in lincs_norm:
            matched.append(lincs_norm[cn])
    return matched


# =========================================================================
# Data loading
# =========================================================================

def load_all_datasets():
    """Load all CTR-DB datasets from disk."""
    cat_path = CTRDB_DIR / "catalog.csv"
    pan_cat_path = CTRDB_DIR / "pan_cancer_catalog.csv"
    catalog = pd.read_csv(cat_path) if cat_path.exists() else pd.DataFrame()
    pan_catalog = pd.read_csv(pan_cat_path) if pan_cat_path.exists() else pd.DataFrame()

    expr_datasets = {}
    label_datasets = {}
    for ds_dir in sorted(CTRDB_DIR.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("GSE"):
            continue
        geo_id = ds_dir.name
        expr_files = list(ds_dir.glob("*_expression.parquet"))
        label_file = ds_dir / "response_labels.parquet"
        if not expr_files or not label_file.exists():
            continue
        expr = pd.read_parquet(expr_files[0])
        labels = pd.read_parquet(label_file)["response"]
        common_samples = expr.index.intersection(labels.index)
        if len(common_samples) < 5:
            continue
        expr_datasets[geo_id] = expr.loc[common_samples]
        label_datasets[geo_id] = labels.loc[common_samples]

    logger.info(f"Loaded {len(expr_datasets)} CTR-DB datasets from disk")
    return expr_datasets, label_datasets, catalog, pan_catalog


def load_landmark_genes() -> list[str]:
    gi = pd.read_csv(DATA_CACHE / "geneinfo_beta_input.txt", sep="\t")
    return gi["gene_symbol"].tolist()


def load_lincs_all_cell():
    sigs = pd.read_parquet(DATA_CACHE / "all_cellline_drug_signatures.parquet")
    gene_cols = [c for c in sigs.columns if c not in META_COLS]
    drug_set = set(sigs["pert_iname"].str.lower().unique())
    return sigs, gene_cols, drug_set


def get_drug_for_dataset(geo_id, catalog, pan_catalog):
    for cat in [catalog, pan_catalog]:
        if cat.empty:
            continue
        row = cat[cat["geo_source"] == geo_id]
        if not row.empty:
            return str(row.iloc[0]["drug"])
    return ""


# =========================================================================
# STEP 1: ComBat batch correction
# =========================================================================

def step1_combat_correction(expr_datasets, label_datasets, landmark_genes):
    """Apply ComBat batch correction to CTR-DB datasets."""
    logger.info("=" * 70)
    logger.info("STEP 1: ComBat batch correction")
    logger.info("=" * 70)

    from src.preprocessing.batch_correction import apply_batch_correction

    # Filter datasets with >= 20 patients
    valid_datasets = {}
    valid_labels = {}
    for geo_id in expr_datasets:
        if geo_id in label_datasets and len(label_datasets[geo_id]) >= 20:
            valid_datasets[geo_id] = expr_datasets[geo_id]
            valid_labels[geo_id] = label_datasets[geo_id]

    logger.info(f"Datasets with >= 20 patients: {len(valid_datasets)}")

    try:
        X_combat, y_combat, ds_combat, common_genes = apply_batch_correction(
            method_name="combat",
            datasets=valid_datasets,
            labels=valid_labels,
            landmark_genes=landmark_genes,
        )
        logger.info(
            f"ComBat corrected: {X_combat.shape[0]} patients x "
            f"{X_combat.shape[1]} genes, {ds_combat.nunique()} datasets"
        )
        return X_combat, y_combat, ds_combat, common_genes
    except Exception as e:
        logger.error(f"ComBat failed: {e}. Falling back to per-dataset z-score.")
        X_zscore, y_zscore, ds_zscore, common_genes = apply_batch_correction(
            method_name="per_dataset_zscore",
            datasets=valid_datasets,
            labels=valid_labels,
            landmark_genes=landmark_genes,
        )
        return X_zscore, y_zscore, ds_zscore, common_genes


# =========================================================================
# STEP 2: Pathway features
# =========================================================================

def step2_pathway_features(expr_datasets, label_datasets, lincs_sigs):
    """Compute pathway features for patients and LINCS signatures."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Pathway features (Hallmark ssGSEA)")
    logger.info("=" * 70)

    from src.features.pathway_features import (
        HALLMARK,
        transform_ctrdb_to_pathways,
        transform_lincs_to_pathways,
    )

    # 2a: Transform LINCS to pathway scores (uses cache)
    logger.info("Transforming LINCS signatures to pathway scores ...")
    try:
        lincs_pathway = transform_lincs_to_pathways(
            lincs_df=lincs_sigs,
            gene_sets=HALLMARK,
            min_size=5,
            threads=4,
        )
        pw_meta = {"sig_id", "pert_id", "pert_iname", "cell_id", "dose_um", "pert_idose"}
        pw_cols = [c for c in lincs_pathway.columns if c not in pw_meta]
        logger.info(f"LINCS pathway features: {len(pw_cols)} pathways")
    except Exception as e:
        logger.error(f"LINCS pathway transformation failed: {e}")
        lincs_pathway = None
        pw_cols = []

    # 2b: Transform CTR-DB to pathway scores (uses cache)
    logger.info("Transforming CTR-DB to pathway scores ...")
    ctrdb_for_pathway = {
        geo_id: (expr_datasets[geo_id], label_datasets[geo_id])
        for geo_id in expr_datasets
        if geo_id in label_datasets
    }
    try:
        ctrdb_pathway = transform_ctrdb_to_pathways(
            ctrdb_datasets=ctrdb_for_pathway,
            gene_sets=HALLMARK,
            min_size=5,
            threads=4,
        )
        logger.info(f"CTR-DB pathway datasets: {len(ctrdb_pathway)}")
    except Exception as e:
        logger.error(f"CTR-DB pathway transformation failed: {e}")
        ctrdb_pathway = {}

    return lincs_pathway, pw_cols, ctrdb_pathway


# =========================================================================
# STEP 3: Dose-aware signatures
# =========================================================================

def step3_dose_aware_signatures(lincs_all_cell):
    """Compute dose-stratified LINCS signatures."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Dose-aware signatures")
    logger.info("=" * 70)

    from src.features.dose_aware_signatures import (
        compute_dose_averaged_signatures,
        compute_dose_stratified_signatures,
    )

    strat_sigs = compute_dose_stratified_signatures(lincs_all_cell)
    avg_sigs = compute_dose_averaged_signatures(lincs_all_cell)

    logger.info(
        f"Stratified sigs: {len(strat_sigs)} drug-bin combos; "
        f"Averaged sigs: {len(avg_sigs)} drugs"
    )
    return strat_sigs, avg_sigs


# =========================================================================
# STEP 4+5: Build combined reversal features + LODO evaluation
# =========================================================================

def step4_5_combined_reversal_lodo(
    X_combat, y_combat, ds_combat, common_genes,
    ctrdb_pathway, lincs_sigs, lincs_pathway, pw_cols,
    strat_sigs, avg_sigs,
    lincs_gene_cols, lincs_drug_set,
    catalog, pan_catalog,
):
    """Build combined reversal features and run LODO evaluation."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4+5: Combined reversal features + LODO evaluation")
    logger.info("=" * 70)

    from src.features.dose_aware_signatures import (
        assign_dose_bin,
        compute_reversal_features_stratified,
    )

    # Build per-drug mean gene signatures
    drug_mean_gene = {}
    for drug, grp in lincs_sigs.groupby(lincs_sigs["pert_iname"].str.lower()):
        drug_mean_gene[drug] = grp[lincs_gene_cols].mean(axis=0)

    # Build per-drug mean pathway signatures
    drug_mean_pw = {}
    if lincs_pathway is not None and pw_cols:
        for drug, grp in lincs_pathway.groupby(lincs_pathway["pert_iname"].str.lower()):
            drug_mean_pw[drug] = grp[pw_cols].mean()

    # Build per-drug dose-stratified signatures (gene level)
    dose_bins_labels = ["low", "medium", "high"]
    dose_numeric = {"low": 0.25, "medium": 2.75, "high": 7.5}

    # Pre-compute per-dataset info
    unique_datasets = ds_combat.unique()
    dataset_drug_map = {}
    for geo_id in unique_datasets:
        drug_str = get_drug_for_dataset(geo_id, catalog, pan_catalog)
        if not drug_str:
            continue
        components = parse_regimen_components(drug_str)
        matched = match_drugs_to_lincs(components, lincs_drug_set)
        if matched:
            dataset_drug_map[geo_id] = (drug_str, matched)

    logger.info(
        f"Drug matching: {len(dataset_drug_map)}/{len(unique_datasets)} datasets matched"
    )

    # Determine global pathway set from CTR-DB pathway cache
    # Use majority vote instead of strict intersection (some datasets have few)
    pw_counts = {}
    n_pw_datasets = 0
    for geo_id in dataset_drug_map:
        if geo_id in ctrdb_pathway:
            pw_df, _ = ctrdb_pathway[geo_id]
            ds_pw = set(pw_df.columns) & set(pw_cols)
            for p in ds_pw:
                pw_counts[p] = pw_counts.get(p, 0) + 1
            n_pw_datasets += 1
    # Keep pathways present in >= 50% of pathway datasets
    pw_threshold = max(1, n_pw_datasets // 2)
    global_pw = sorted([p for p, cnt in pw_counts.items() if cnt >= pw_threshold])
    logger.info(f"Global common pathways: {len(global_pw)} (threshold >= {pw_threshold}/{n_pw_datasets} datasets)")

    # Pre-compute per-dataset feature bundles
    dataset_features = {}

    for geo_id in unique_datasets:
        if geo_id not in dataset_drug_map:
            continue
        drug_str, matched_drugs = dataset_drug_map[geo_id]

        mask = ds_combat == geo_id
        X_ds = X_combat[mask].values
        y_ds = y_combat[mask].values

        if len(y_ds) < 20:
            continue
        n_pos = int(y_ds.sum())
        n_neg = len(y_ds) - n_pos
        if n_pos < 3 or n_neg < 3:
            continue

        # --- Gene-level reversal features (ComBat-corrected) ---
        drug_sig_gene = np.zeros(len(common_genes), dtype=np.float64)
        n_d = 0
        for d in matched_drugs:
            dl = d.lower()
            if dl in drug_mean_gene:
                sig = drug_mean_gene[dl].reindex(common_genes).values.astype(np.float64)
                drug_sig_gene += np.nan_to_num(sig, 0.0)
                n_d += 1
        if n_d > 0:
            drug_sig_gene /= n_d
        else:
            continue

        X_gene_rev = X_ds * drug_sig_gene[np.newaxis, :]

        # --- Pathway-level reversal features ---
        # Always produce a fixed-size pathway reversal array (zero-filled if unavailable)
        n_pw_feats = len(global_pw) if global_pw else 0
        X_pw_rev = np.zeros((len(y_ds), max(n_pw_feats, 1)), dtype=np.float64)
        if global_pw and geo_id in ctrdb_pathway:
            pw_df, pw_labels = ctrdb_pathway[geo_id]
            drug_sig_pw = np.zeros(len(global_pw), dtype=np.float64)
            n_d = 0
            for d in matched_drugs:
                dl = d.lower()
                if dl in drug_mean_pw:
                    sig = drug_mean_pw[dl].reindex(global_pw).values.astype(np.float64)
                    drug_sig_pw += np.nan_to_num(sig, 0.0)
                    n_d += 1
            if n_d > 0:
                drug_sig_pw /= n_d
                pw_aligned = pw_df.reindex(columns=global_pw).fillna(0.0)
                n_pw = min(len(pw_aligned), len(y_ds))
                if n_pw == len(y_ds):
                    X_pw = pw_aligned.iloc[:n_pw][global_pw].values.astype(np.float64)
                    X_pw = np.nan_to_num(X_pw, 0.0)
                    X_pw_rev = X_pw * drug_sig_pw[np.newaxis, :]
                else:
                    logger.debug(
                        f"  {geo_id}: pathway sample count mismatch "
                        f"({n_pw} vs {len(y_ds)}), using zeros"
                    )

        # --- Dose-aware reversal features ---
        # Compute reversal at low/medium/high dose + slope + max
        dose_features = np.zeros((len(y_ds), 5), dtype=np.float64)

        strat_lookup = strat_sigs.set_index(["pert_iname", "dose_bin"])
        strat_gene_cols = [c for c in strat_sigs.columns if c not in {"pert_iname", "dose_bin"}]
        strat_common = sorted(set(strat_gene_cols) & set(common_genes))

        if strat_common:
            # Patient z-scores aligned to strat_common genes
            gene_idx_map = {g: i for i, g in enumerate(common_genes)}
            strat_col_idx = [gene_idx_map[g] for g in strat_common if g in gene_idx_map]
            patient_z = X_ds[:, strat_col_idx]

            bin_reversal = {}
            for d in matched_drugs:
                for dbin in dose_bins_labels:
                    if (d, dbin) not in strat_lookup.index:
                        continue
                    row_vals = strat_lookup.loc[(d, dbin)]
                    drug_sig_bin = row_vals[strat_common].values.astype(np.float64)
                    drug_sig_bin = np.nan_to_num(drug_sig_bin, 0.0)
                    # Reversal = mean of element-wise product across genes
                    rev = np.mean(patient_z * drug_sig_bin[np.newaxis, :], axis=1)
                    if dbin not in bin_reversal:
                        bin_reversal[dbin] = rev
                    else:
                        bin_reversal[dbin] = (bin_reversal[dbin] + rev) / 2

            for i, dbin in enumerate(dose_bins_labels):
                if dbin in bin_reversal:
                    dose_features[:, i] = bin_reversal[dbin]

            # Slope: linear fit across available dose bins
            available_bins = [b for b in dose_bins_labels if b in bin_reversal]
            if len(available_bins) >= 2:
                x_vals = np.array([dose_numeric[b] for b in available_bins])
                y_mat = np.column_stack([bin_reversal[b] for b in available_bins])
                for row_i in range(len(y_ds)):
                    if np.all(np.isfinite(y_mat[row_i])):
                        dose_features[row_i, 3] = np.polyfit(x_vals, y_mat[row_i], 1)[0]

            # Max reversal across bins
            if available_bins:
                all_bin_scores = np.column_stack([bin_reversal[b] for b in available_bins])
                dose_features[:, 4] = np.nanmax(all_bin_scores, axis=1)

        # --- Assemble combined features ---
        # Gene reversal (n_genes) + Pathway reversal (n_pw) + Dose features (5)
        feature_parts = [X_gene_rev, X_pw_rev, dose_features]
        feat_dim_desc = [
            f"gene_rev({X_gene_rev.shape[1]})",
            f"pw_rev({X_pw_rev.shape[1]})",
            f"dose({dose_features.shape[1]})",
        ]

        X_combined = np.hstack(feature_parts)

        dataset_features[geo_id] = {
            "X_gene_rev": X_gene_rev,
            "X_combined": X_combined,
            "y": y_ds,
            "drug": drug_str,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    if dataset_features:
        first_geo = next(iter(dataset_features))
        n_gene = dataset_features[first_geo]["X_gene_rev"].shape[1]
        n_comb = dataset_features[first_geo]["X_combined"].shape[1]
        logger.info(
            f"Built combined features for {len(dataset_features)} datasets "
            f"(gene_rev={n_gene}, combined={n_comb} dims)"
        )
    else:
        logger.warning("No datasets had valid features")

    if len(dataset_features) < 2:
        logger.warning("Not enough datasets for LODO evaluation")
        return pd.DataFrame()

    # ── LODO evaluation ──
    logger.info("Running LODO evaluation ...")

    results = []
    all_geos = sorted(dataset_features.keys())

    for feature_label, feature_key in [
        ("gene_reversal_only", "X_gene_rev"),
        ("combined_all", "X_combined"),
    ]:
        for held_out in all_geos:
            train_geos = [g for g in all_geos if g != held_out]

            X_train_parts = []
            y_train_parts = []
            for tg in train_geos:
                X_train_parts.append(dataset_features[tg][feature_key])
                y_train_parts.append(dataset_features[tg]["y"])

            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            X_test = dataset_features[held_out][feature_key]
            y_test = dataset_features[held_out]["y"]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_train_s = np.nan_to_num(X_train_s, 0.0)
            X_test_s = np.nan_to_num(X_test_s, 0.0)

            clf = LogisticRegression(
                C=LODO_C, solver="liblinear", l1_ratio=1.0,
                max_iter=2000, random_state=42, class_weight="balanced",
            )
            try:
                clf.fit(X_train_s, y_train)
                y_prob = clf.predict_proba(X_test_s)
                if y_prob.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    auc = 0.5
            except Exception as e:
                logger.warning(f"  LODO {held_out}/{feature_label}: {e}")
                auc = 0.5

            results.append({
                "feature_set": feature_label,
                "held_out_dataset": held_out,
                "drug": dataset_features[held_out]["drug"],
                "auc": round(auc, 4),
                "n_test": len(y_test),
                "n_train": len(y_train),
                "n_features": X_train.shape[1],
            })
            logger.info(
                f"  [{feature_label}] {held_out}: AUC={auc:.3f} "
                f"(n_test={len(y_test)}, feats={X_train.shape[1]})"
            )

    return pd.DataFrame(results)


# =========================================================================
# STEP 6: Cell-line model evaluation with ChemBERTa + pathways
# =========================================================================

def step6_cellline_models():
    """Evaluate LightGBM and NN with ChemBERTa features via 5-fold CV."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 6: Cell-line model with ChemBERTa + pathways")
    logger.info("=" * 70)

    import lightgbm as lgb
    from src.models.interaction_nn import DrugResponseNN, train_nn, predict_nn

    # Load training data
    X_full = pd.read_parquet(DATA_PROCESSED / "training_matrix.parquet")
    y = pd.read_parquet(DATA_PROCESSED / "training_target.parquet")["pct_inhibition"].values.astype(np.float32)

    # Identify column groups
    ecfp_cols = [c for c in X_full.columns if c.startswith("ecfp_")]
    gene_cols = [c for c in X_full.columns if not c.startswith("ecfp_") and c != "log_dose_um"]
    dose_col = ["log_dose_um"]

    logger.info(f"Training matrix: {X_full.shape[0]} samples x {X_full.shape[1]} features")
    logger.info(f"Gene cols: {len(gene_cols)}, ECFP cols: {len(ecfp_cols)}, dose: 1")

    # Load ChemBERTa embeddings
    emb_path = DATA_CACHE / "chemberta_embeddings.parquet"
    if not emb_path.exists():
        logger.error(f"ChemBERTa embeddings not found at {emb_path}")
        return pd.DataFrame(), None, None, None

    emb_df = pd.read_parquet(emb_path)
    chemberta_cols = [c for c in emb_df.columns if c.startswith("chemberta_")]
    logger.info(f"ChemBERTa: {len(emb_df)} compounds x {len(chemberta_cols)} dims")

    # Map embeddings to training rows
    matched_path = DATA_CACHE / "lincs_pharmacodb_matched.parquet"
    matched = pd.read_parquet(matched_path)
    compound_names = matched["pert_iname"].values

    emb_lookup = emb_df.set_index("compound_name")[chemberta_cols]
    chemberta_matrix = np.zeros((len(X_full), len(chemberta_cols)), dtype=np.float32)
    for i, name in enumerate(compound_names):
        if name in emb_lookup.index:
            chemberta_matrix[i] = emb_lookup.loc[name].values

    chemberta_df = pd.DataFrame(chemberta_matrix, columns=chemberta_cols, index=X_full.index)

    # Feature configurations
    # Original: genes + ECFP + dose
    X_original = X_full[gene_cols + ecfp_cols + dose_col]

    # ChemBERTa: genes + ChemBERTa + dose
    X_chemberta = pd.concat([X_full[gene_cols], chemberta_df, X_full[dose_col]], axis=1)

    feature_configs = {
        "genes_ecfp_dose": X_original,
        "genes_chemberta_dose": X_chemberta,
    }

    # 5-Fold CV for each config x model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    # Also store the best NN model trained on ChemBERTa features for Step 7
    best_nn_model = None
    best_nn_scaler = None
    best_nn_features = None

    for config_name, X_config in feature_configs.items():
        X_arr = X_config.values.astype(np.float32)
        feature_names = list(X_config.columns)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ── LightGBM ──
            lgbm_params = LIGHTGBM_DEFAULT_PARAMS.copy()
            lgbm_model = lgb.LGBMRegressor(**lgbm_params)
            lgbm_model.fit(
                X_train, y_train,
                feature_name=feature_names,
                callbacks=[lgb.log_evaluation(0)],
            )
            lgbm_preds = lgbm_model.predict(X_test)
            lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_preds))

            # ── Shallow NN ──
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            rng = np.random.RandomState(RANDOM_SEED + fold_idx)
            n_train = len(X_train_s)
            val_mask = rng.rand(n_train) < 0.2
            train_mask = ~val_mask

            nn_model = train_nn(
                X_train=X_train_s[train_mask],
                y_train=y_train[train_mask],
                X_val=X_train_s[val_mask],
                y_val=y_train[val_mask],
                hidden_dims=[256, 128],
                dropout=0.3,
                lr=1e-3,
                weight_decay=1e-4,
                batch_size=64,
                max_epochs=200,
                patience=15,
                device="cpu",
            )
            nn_preds = predict_nn(nn_model, X_test_s)
            nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))

            cv_results.append({
                "config": config_name,
                "fold": fold_idx + 1,
                "lgbm_rmse": round(lgbm_rmse, 4),
                "nn_rmse": round(nn_rmse, 4),
            })

            logger.info(
                f"  [{config_name}] Fold {fold_idx+1}: "
                f"LightGBM RMSE={lgbm_rmse:.3f}, NN RMSE={nn_rmse:.3f}"
            )

    cv_df = pd.DataFrame(cv_results)

    # Train final NN model on all data with ChemBERTa for Step 7
    logger.info("Training final NN on all data with ChemBERTa features for patient transfer ...")
    X_chemberta_arr = X_chemberta.values.astype(np.float32)
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_chemberta_arr)

    rng = np.random.RandomState(RANDOM_SEED)
    n_all = len(X_all_scaled)
    val_mask = rng.rand(n_all) < 0.15
    train_mask = ~val_mask

    final_nn = train_nn(
        X_train=X_all_scaled[train_mask],
        y_train=y[train_mask],
        X_val=X_all_scaled[val_mask],
        y_val=y[val_mask],
        hidden_dims=[256, 128],
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        max_epochs=200,
        patience=15,
        device="cpu",
    )

    # Also train final LightGBM on all data
    logger.info("Training final LightGBM on all data ...")
    final_lgbm = lgb.LGBMRegressor(**LIGHTGBM_DEFAULT_PARAMS)
    final_lgbm.fit(
        X_chemberta_arr, y,
        feature_name=list(X_chemberta.columns),
        callbacks=[lgb.log_evaluation(0)],
    )

    return cv_df, final_nn, final_scaler, final_lgbm, gene_cols, chemberta_cols, dose_col


# =========================================================================
# STEP 7: Test cell-line model on CTR-DB patients
# =========================================================================

def step7_patient_transfer(
    final_nn, final_scaler, final_lgbm,
    gene_cols, chemberta_cols, dose_col,
    expr_datasets, label_datasets,
    lincs_sigs, lincs_drug_set,
    catalog, pan_catalog,
):
    """Test cell-line models on CTR-DB patients."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 7: Cell-line model -> CTR-DB patient transfer")
    logger.info("=" * 70)

    from src.models.interaction_nn import predict_nn

    # Load ChemBERTa embeddings
    emb_df = pd.read_parquet(DATA_CACHE / "chemberta_embeddings.parquet")
    emb_cols = [c for c in emb_df.columns if c.startswith("chemberta_")]
    emb_lookup = emb_df.set_index("compound_name")[emb_cols]

    # Load LINCS signatures for gene-drug matching
    lincs_gene_cols = [c for c in lincs_sigs.columns if c not in META_COLS]

    # Build drug -> average gene z-scores + chemberta embedding
    drug_gene_mean = {}
    for drug, grp in lincs_sigs.groupby("pert_iname"):
        drug_gene_mean[drug.lower()] = grp[lincs_gene_cols].mean()

    # Use median dose from training data
    import math
    median_log_dose = math.log10(10.0)  # typical dose ~10 uM

    results = []

    for geo_id in sorted(expr_datasets.keys()):
        if geo_id not in label_datasets:
            continue
        labels = label_datasets[geo_id]
        if len(labels) < 20:
            continue
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos < 3 or n_neg < 3:
            continue

        drug_str = get_drug_for_dataset(geo_id, catalog, pan_catalog)
        if not drug_str:
            continue
        components = parse_regimen_components(drug_str)
        matched = match_drugs_to_lincs(components, lincs_drug_set)
        if not matched:
            continue

        # For each matched drug, build patient features
        expr = expr_datasets[geo_id]

        for drug in matched:
            dl = drug.lower()
            if dl not in drug_gene_mean:
                continue

            # Get ChemBERTa embedding for this drug
            drug_chemberta = np.zeros(len(emb_cols), dtype=np.float32)
            if drug in emb_lookup.index:
                drug_chemberta = emb_lookup.loc[drug].values.astype(np.float32)
            elif dl in emb_lookup.index:
                drug_chemberta = emb_lookup.loc[dl].values.astype(np.float32)

            # Build feature matrix: gene z-scores + ChemBERTa + log_dose
            # Gene features: patient expression z-scored per gene
            common_genes = [g for g in gene_cols if g in expr.columns]
            if len(common_genes) < 100:
                continue

            X_genes = expr.reindex(columns=gene_cols).fillna(0.0).values.astype(np.float32)
            # Z-score per gene
            g_mean = np.nanmean(X_genes, axis=0)
            g_std = np.nanstd(X_genes, axis=0)
            g_std[g_std == 0] = 1
            X_genes = (X_genes - g_mean) / g_std

            n_patients = X_genes.shape[0]
            X_chem = np.tile(drug_chemberta, (n_patients, 1))
            X_dose = np.full((n_patients, 1), median_log_dose, dtype=np.float32)

            X_patient = np.hstack([X_genes, X_chem, X_dose])

            # Predict with NN
            try:
                X_patient_s = final_scaler.transform(X_patient)
                nn_preds = predict_nn(final_nn, X_patient_s)

                # Higher predicted inhibition -> responder
                y_true = labels.values
                auc_nn = roc_auc_score(y_true, nn_preds)
            except Exception as e:
                logger.warning(f"  NN prediction failed for {geo_id}/{drug}: {e}")
                auc_nn = 0.5

            # Predict with LightGBM
            try:
                lgbm_preds = final_lgbm.predict(X_patient)
                auc_lgbm = roc_auc_score(y_true, lgbm_preds)
            except Exception as e:
                logger.warning(f"  LightGBM prediction failed for {geo_id}/{drug}: {e}")
                auc_lgbm = 0.5

            results.append({
                "geo_id": geo_id,
                "drug": drug_str,
                "matched_drug": drug,
                "n_patients": len(y_true),
                "n_responders": n_pos,
                "auc_nn": round(auc_nn, 4),
                "auc_lgbm": round(auc_lgbm, 4),
            })

            logger.info(
                f"  {geo_id} ({drug}): NN AUC={auc_nn:.3f}, "
                f"LightGBM AUC={auc_lgbm:.3f} (n={len(y_true)})"
            )

    return pd.DataFrame(results)


# =========================================================================
# STEP 8: Full comparison table
# =========================================================================

def step8_comparison_table(lodo_results, cv_results, transfer_results):
    """Build the final comparison table from all results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 8: Full comparison table")
    logger.info("=" * 70)

    rows = []

    # ── Row 1: Original cell-line LightGBM (from existing results) ──
    rows.append({
        "model": "Original cell-line LightGBM",
        "features": "genes (978) + ECFP (1024) + dose",
        "metric_type": "CV RMSE",
        "metric_value": 21.20,
        "metric_type_2": "Patient AUC",
        "metric_value_2": 0.467,
        "source": "baseline",
    })

    # ── Rows from CV results ──
    if cv_results is not None and not cv_results.empty:
        for config in cv_results["config"].unique():
            cfg_data = cv_results[cv_results["config"] == config]

            lgbm_rmse = cfg_data["lgbm_rmse"].mean()
            nn_rmse = cfg_data["nn_rmse"].mean()

            if config == "genes_ecfp_dose":
                feat_desc = "genes (978) + ECFP (1024) + dose"
            else:
                feat_desc = "genes (978) + ChemBERTa (384) + dose"

            rows.append({
                "model": f"Cell-line LightGBM ({config})",
                "features": feat_desc,
                "metric_type": "CV RMSE",
                "metric_value": round(lgbm_rmse, 2),
                "metric_type_2": "",
                "metric_value_2": np.nan,
                "source": "step6",
            })
            rows.append({
                "model": f"Cell-line NN ({config})",
                "features": feat_desc,
                "metric_type": "CV RMSE",
                "metric_value": round(nn_rmse, 2),
                "metric_type_2": "",
                "metric_value_2": np.nan,
                "source": "step6",
            })

    # ── Rows from LODO reversal evaluation ──
    if lodo_results is not None and not lodo_results.empty:
        for feat_set in lodo_results["feature_set"].unique():
            fs_data = lodo_results[lodo_results["feature_set"] == feat_set]
            mean_auc = fs_data["auc"].mean()
            n_datasets = len(fs_data)
            n_feats = fs_data["n_features"].iloc[0] if len(fs_data) > 0 else 0

            if feat_set == "gene_reversal_only":
                model_name = "Reversal (ComBat, genes only)"
                feat_desc = f"gene reversal ({n_feats})"
            else:
                model_name = "Reversal (ComBat + pathways + dose)"
                feat_desc = f"gene + pathway + dose reversal ({n_feats})"

            rows.append({
                "model": model_name,
                "features": feat_desc,
                "metric_type": "LODO AUC",
                "metric_value": round(mean_auc, 4),
                "metric_type_2": f"n_datasets={n_datasets}",
                "metric_value_2": np.nan,
                "source": "step5",
            })

    # ── Reference: existing results from previous sprints ──
    rows.append({
        "model": "Reversal (uniform, genes only)",
        "features": "978 gene reversal",
        "metric_type": "LODO AUC",
        "metric_value": 0.46,
        "metric_type_2": "",
        "metric_value_2": np.nan,
        "source": "reference (ctrdb_validation)",
    })

    rows.append({
        "model": "Pan-cancer recalibrated reversal",
        "features": "LODO gene weights",
        "metric_type": "LODO AUC",
        "metric_value": 0.60,
        "metric_type_2": "",
        "metric_value_2": np.nan,
        "source": "reference (recalibration_validation_pancancer)",
    })

    rows.append({
        "model": "Direct patient model (breast LODO)",
        "features": "genes",
        "metric_type": "AUC",
        "metric_value": 0.69,
        "metric_type_2": "",
        "metric_value_2": np.nan,
        "source": "reference",
    })

    rows.append({
        "model": "Direct patient model (pan-cancer LODO)",
        "features": "genes",
        "metric_type": "AUC",
        "metric_value": 0.73,
        "metric_type_2": "",
        "metric_value_2": np.nan,
        "source": "reference",
    })

    # ── Rows from patient transfer (Step 7) ──
    if transfer_results is not None and not transfer_results.empty:
        mean_nn_auc = transfer_results["auc_nn"].mean()
        mean_lgbm_auc = transfer_results["auc_lgbm"].mean()
        n_ds = transfer_results["geo_id"].nunique()

        rows.append({
            "model": "Cell-line NN + ChemBERTa -> patients",
            "features": "genes + ChemBERTa + dose",
            "metric_type": "Patient AUC",
            "metric_value": round(mean_nn_auc, 4),
            "metric_type_2": f"n_datasets={n_ds}",
            "metric_value_2": np.nan,
            "source": "step7",
        })
        rows.append({
            "model": "Cell-line LightGBM + ChemBERTa -> patients",
            "features": "genes + ChemBERTa + dose",
            "metric_type": "Patient AUC",
            "metric_value": round(mean_lgbm_auc, 4),
            "metric_type_2": f"n_datasets={n_ds}",
            "metric_value_2": np.nan,
            "source": "step7",
        })

    comparison_df = pd.DataFrame(rows)
    return comparison_df


# =========================================================================
# MAIN
# =========================================================================

def main():
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("COMBINED PIPELINE: ALL WINNING IMPROVEMENTS")
    logger.info("=" * 70)
    logger.info("Winners: ComBat + Pathway + ChemBERTa + Dose-aware + Shallow NN")
    logger.info("")

    # ── Load base data ──
    logger.info("Loading base data ...")
    expr_datasets, label_datasets, catalog, pan_catalog = load_all_datasets()
    landmark_genes = load_landmark_genes()
    lincs_sigs, lincs_gene_cols, lincs_drug_set = load_lincs_all_cell()

    # Also load breast LINCS signatures for pathway computation
    breast_lincs_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    if breast_lincs_path.exists():
        breast_lincs = pd.read_parquet(breast_lincs_path)
    else:
        breast_lincs = lincs_sigs
    logger.info(f"LINCS all-cell: {len(lincs_sigs)} sigs, breast: {len(breast_lincs)} sigs")

    # ── STEP 1: ComBat ──
    try:
        X_combat, y_combat, ds_combat, common_genes = step1_combat_correction(
            expr_datasets, label_datasets, landmark_genes
        )
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        X_combat, y_combat, ds_combat, common_genes = None, None, None, []

    # ── STEP 2: Pathway features ──
    try:
        lincs_pathway, pw_cols, ctrdb_pathway = step2_pathway_features(
            expr_datasets, label_datasets, breast_lincs
        )
    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        lincs_pathway, pw_cols, ctrdb_pathway = None, [], {}

    # ── STEP 3: Dose-aware signatures ──
    try:
        strat_sigs, avg_sigs = step3_dose_aware_signatures(lincs_sigs)
    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        strat_sigs, avg_sigs = pd.DataFrame(), pd.DataFrame()

    # ── STEP 4+5: Combined reversal LODO ──
    lodo_results = pd.DataFrame()
    if X_combat is not None:
        try:
            lodo_results = step4_5_combined_reversal_lodo(
                X_combat, y_combat, ds_combat, common_genes,
                ctrdb_pathway, lincs_sigs, lincs_pathway, pw_cols,
                strat_sigs, avg_sigs,
                lincs_gene_cols, lincs_drug_set,
                catalog, pan_catalog,
            )
        except Exception as e:
            logger.error(f"Step 4+5 failed: {e}", exc_info=True)

    # ── STEP 6: Cell-line models ──
    cv_results = pd.DataFrame()
    final_nn = None
    final_scaler = None
    final_lgbm = None
    gene_cols_cl = []
    chemberta_cols_cl = []
    dose_col_cl = []
    try:
        result = step6_cellline_models()
        if result is not None:
            cv_results, final_nn, final_scaler, final_lgbm, gene_cols_cl, chemberta_cols_cl, dose_col_cl = result
    except Exception as e:
        logger.error(f"Step 6 failed: {e}", exc_info=True)

    # ── STEP 7: Patient transfer ──
    transfer_results = pd.DataFrame()
    if final_nn is not None and final_lgbm is not None:
        try:
            transfer_results = step7_patient_transfer(
                final_nn, final_scaler, final_lgbm,
                gene_cols_cl, chemberta_cols_cl, dose_col_cl,
                expr_datasets, label_datasets,
                lincs_sigs, lincs_drug_set,
                catalog, pan_catalog,
            )
        except Exception as e:
            logger.error(f"Step 7 failed: {e}", exc_info=True)

    # ── STEP 8: Comparison table ──
    comparison_df = step8_comparison_table(lodo_results, cv_results, transfer_results)

    # ── Save results ──
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "combined_pipeline_results.csv"
    comparison_df.to_csv(out_path, index=False)
    logger.info(f"\nFull comparison saved to {out_path}")

    # Save detailed LODO results
    if not lodo_results.empty:
        lodo_path = RESULTS / "combined_lodo_detail.csv"
        lodo_results.to_csv(lodo_path, index=False)
        logger.info(f"LODO detail saved to {lodo_path}")

    # Save CV results
    if not cv_results.empty:
        cv_path = RESULTS / "combined_cv_detail.csv"
        cv_results.to_csv(cv_path, index=False)
        logger.info(f"CV detail saved to {cv_path}")

    # Save transfer results
    if not transfer_results.empty:
        transfer_path = RESULTS / "combined_transfer_detail.csv"
        transfer_results.to_csv(transfer_path, index=False)
        logger.info(f"Transfer detail saved to {transfer_path}")

    # ── Print final summary ──
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY — COMBINED PIPELINE")
    logger.info("=" * 70)

    # Print comparison table
    for _, row in comparison_df.iterrows():
        logger.info(
            f"  {row['model']:50s}  {row['metric_type']:12s} = {row['metric_value']:.4f}"
        )

    # Print LODO summary
    if not lodo_results.empty:
        logger.info("")
        logger.info("LODO AUC summary:")
        for feat_set in lodo_results["feature_set"].unique():
            fs_data = lodo_results[lodo_results["feature_set"] == feat_set]
            logger.info(
                f"  {feat_set:30s}: mean={fs_data['auc'].mean():.4f}, "
                f"median={fs_data['auc'].median():.4f}, "
                f"std={fs_data['auc'].std():.4f}, n={len(fs_data)}"
            )

    # Print CV summary
    if not cv_results.empty:
        logger.info("")
        logger.info("Cell-line CV RMSE summary:")
        for config in cv_results["config"].unique():
            cfg = cv_results[cv_results["config"] == config]
            logger.info(
                f"  {config:30s}: LightGBM={cfg['lgbm_rmse'].mean():.3f} +/- {cfg['lgbm_rmse'].std():.3f}, "
                f"NN={cfg['nn_rmse'].mean():.3f} +/- {cfg['nn_rmse'].std():.3f}"
            )

    # Print transfer summary
    if not transfer_results.empty:
        logger.info("")
        logger.info("Patient transfer AUC summary:")
        logger.info(
            f"  NN mean AUC:       {transfer_results['auc_nn'].mean():.4f}"
        )
        logger.info(
            f"  LightGBM mean AUC: {transfer_results['auc_lgbm'].mean():.4f}"
        )

    logger.info("")
    logger.info(f"Total pipeline time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
