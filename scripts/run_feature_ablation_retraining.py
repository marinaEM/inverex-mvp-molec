#!/usr/bin/env python3
"""
Structured ablation retraining for INVEREX.

Configs evaluated (all use L1-logistic regression, C=0.05, LODO):

    A: baseline (gene reversal features only)         [existing results]
    B: + drug-target interaction features              [retrain]
    C: + DepMap dependency features                    [retrain]
    D: + PROGENy pathway features                      [retrain]
    E: + drug-target + DepMap combined                 [retrain]
    F: + drug-target + DepMap + PROGENy                [retrain]
    G: + all features + ssGSEA kept                    [retrain]

Outputs:
    results/feature_ablation_retraining.tsv
    results/lodo_predictions_config_B.tsv .. _G.tsv
    results/full_improvement_trajectory.tsv

Entry point: ``pixi run python scripts/run_feature_ablation_retraining.py``
"""

import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_CACHE, DATA_RAW, RESULTS
from src.features.drug_target_interactions import (
    parse_drug_targets,
    parse_drug_pathways,
    load_ctrdb_datasets,
    match_ctrdb_drug_to_gdsc,
    compute_target_expression_features,
    compute_pathway_context_features,
    compute_compatibility_features,
    load_hallmark_gene_sets,
)
from src.features.depmap_priors import get_depmap_features_for_sample
from src.features.progeny_features import (
    DRUG_PROGENY_PATHWAY,
    PROGENY_PATHWAYS,
    compute_progeny_activities,
    get_drug_progeny_pathways,
    build_drug_specific_progeny_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CTRDB_DIR = DATA_RAW / "ctrdb"
META_COLS = frozenset(
    {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
)


# ===================================================================
# Drug matching helpers (self-contained to avoid modifying existing)
# ===================================================================

def _normalize_drug_name(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", name.strip().lower())


_CLASS_TO_DRUGS = {
    "anthracycline": ["doxorubicin", "epirubicin", "daunorubicin"],
    "taxane": ["paclitaxel", "docetaxel"],
    "platinum": ["cisplatin", "carboplatin", "oxaliplatin"],
    "glucocorticoids": ["dexamethasone", "prednisone", "prednisolone"],
}


def _parse_regimen_components(drug_string: str) -> list[str]:
    """Parse CTR-DB drug string into individual component names."""
    s = drug_string.strip()
    paren_match = re.search(r"\(([^)]+)\)", s)
    if paren_match:
        inner = paren_match.group(1)
    else:
        inner = s
    parts = re.split(r"[+/]", inner)
    components = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^[A-Z]{1,5}\s+", "", p)
        norm = _normalize_drug_name(p)
        if norm and len(norm) > 1:
            components.append(norm)
    return list(dict.fromkeys(components))


def _match_to_lincs(drug_string: str, lincs_drug_set: set[str]) -> list[str]:
    """Match CTR-DB drug string to LINCS drug names."""
    components = _parse_regimen_components(drug_string)
    lincs_norm = {_normalize_drug_name(d): d for d in lincs_drug_set}
    matched = []
    for comp in components:
        if comp in lincs_norm:
            matched.append(lincs_norm[comp])
            continue
        if comp in _CLASS_TO_DRUGS:
            for specific in _CLASS_TO_DRUGS[comp]:
                sn = _normalize_drug_name(specific)
                if sn in lincs_norm:
                    matched.append(lincs_norm[sn])
            continue
        for ln, orig in lincs_norm.items():
            if comp in ln or ln in comp:
                matched.append(orig)
                break
    return list(dict.fromkeys(matched))


# ===================================================================
# Feature computation for a single dataset
# ===================================================================

def _compute_reversal_features(
    patient_z: np.ndarray,
    drug_sig: np.ndarray,
) -> np.ndarray:
    """
    Compute gene reversal summary features (5 features per patient).
    """
    n = patient_z.shape[0]
    rev = patient_z * drug_sig[np.newaxis, :]
    rev_mean = np.nanmean(rev, axis=1, keepdims=True)
    rev_std = np.nanstd(rev, axis=1, keepdims=True)
    rev_min = np.nanmin(rev, axis=1, keepdims=True)
    rev_neg_frac = (
        np.sum(rev < 0, axis=1, keepdims=True).astype(np.float64)
        / rev.shape[1]
    )
    rev_scores = np.array([
        -pearsonr(patient_z[i], drug_sig)[0]
        if np.std(patient_z[i]) > 0 else 0.0
        for i in range(n)
    ]).reshape(-1, 1)
    return np.hstack([rev_mean, rev_std, rev_min, rev_neg_frac, rev_scores])


def _compute_drug_target_features(
    patient_z: np.ndarray,
    gene_names: list[str],
    target_genes: list[str],
    hallmark: dict[str, set[str]],
) -> np.ndarray:
    """
    Drug-target features: target expression (3) + pathway context (1) +
    compatibility (1) = 5 features.
    """
    n = patient_z.shape[0]
    if not target_genes:
        return np.zeros((n, 5), dtype=np.float64)

    tgt_expr = compute_target_expression_features(
        patient_z, gene_names, target_genes
    )
    pw_ctx = compute_pathway_context_features(
        patient_z, gene_names, target_genes, hallmark
    )
    compat = compute_compatibility_features(
        patient_z, gene_names, target_genes
    )
    return np.hstack([tgt_expr, pw_ctx, compat])


def _compute_depmap_features_for_dataset(
    depmap_cache: pd.DataFrame,
    drug_string: str,
    gdsc_drug_names: set[str],
    drug_targets_map: dict[str, list[str]],
    n_patients: int,
    default_subtype: str = "LumA",
) -> np.ndarray:
    """
    DepMap features: 4 features per patient (broadcast from drug-level).
    Since DepMap features are drug/subtype-level (not patient-level),
    we broadcast the same values to all patients in the dataset.
    """
    feat_cols = [
        "depmap_target_essentiality",
        "depmap_target_selectivity",
        "depmap_n_essential_targets",
        "depmap_mutation_vulnerability",
    ]

    gdsc_matched = match_ctrdb_drug_to_gdsc(drug_string, gdsc_drug_names)

    # Aggregate DepMap features across matched GDSC drugs
    feat_values = np.zeros(len(feat_cols), dtype=np.float64)
    n_matched = 0

    for gd in gdsc_matched:
        feats = get_depmap_features_for_sample(
            depmap_cache, gd, default_subtype
        )
        vals = np.array([feats[c] for c in feat_cols], dtype=np.float64)
        if np.any(vals != 0):
            feat_values += vals
            n_matched += 1

    if n_matched > 0:
        feat_values /= n_matched

    # Broadcast to all patients
    return np.tile(feat_values, (n_patients, 1))


def _compute_progeny_features_for_dataset(
    patient_expr: pd.DataFrame,
    drug_string: str,
    lincs_progeny_mean: dict[str, pd.Series],
    lincs_drug_set: set[str],
    progeny_pathway_names: list[str],
) -> np.ndarray:
    """
    PROGENy features: reversal + drug-specific amplification.
    Returns n_patients x n_pathways array.
    """
    n_patients = patient_expr.shape[0]
    n_pw = len(progeny_pathway_names)

    # Compute patient PROGENy activities
    try:
        patient_progeny = compute_progeny_activities(patient_expr)
    except Exception as e:
        logger.warning("  PROGENy computation failed: %s", e)
        return np.zeros((n_patients, n_pw), dtype=np.float64)

    # Align to standard pathway names
    X_patient = np.zeros((n_patients, n_pw), dtype=np.float64)
    for i, pw in enumerate(progeny_pathway_names):
        if pw in patient_progeny.columns:
            X_patient[:, i] = patient_progeny[pw].values.astype(np.float64)

    # Mean LINCS PROGENy drug signature
    matched = _match_to_lincs(drug_string, lincs_drug_set)
    drug_sig = np.zeros(n_pw, dtype=np.float64)
    n_d = 0
    for d in matched:
        dl = d.lower()
        if dl in lincs_progeny_mean:
            sig = lincs_progeny_mean[dl]
            vals = sig.reindex(progeny_pathway_names).values.astype(np.float64)
            drug_sig += np.nan_to_num(vals, 0.0)
            n_d += 1
    if n_d > 0:
        drug_sig /= n_d

    # Parse drug components for drug-specific amplification
    components = _parse_regimen_components(drug_string)

    # Build drug-specific PROGENy features
    X_progeny = build_drug_specific_progeny_features(
        patient_progeny=X_patient,
        drug_progeny=drug_sig,
        pathway_names=progeny_pathway_names,
        drug_components=components,
    )

    return np.nan_to_num(X_progeny, 0.0)


def _compute_ssgsea_features_for_dataset(
    patient_expr: pd.DataFrame,
    drug_string: str,
    lincs_hallmark_mean: dict[str, pd.Series],
    lincs_drug_set: set[str],
    hallmark_pathway_names: list[str],
    cache_dir: Path = None,
    gse_id: str = "",
) -> np.ndarray:
    """
    ssGSEA Hallmark pathway reversal features.
    """
    n_patients = patient_expr.shape[0]
    n_pw = len(hallmark_pathway_names)

    # Try loading cached ssGSEA scores
    if cache_dir is None:
        cache_dir = DATA_CACHE / "pathway_ctrdb_MSigDB_Hallmark_2020"

    cache_path = cache_dir / f"{gse_id}_pathway.parquet" if gse_id else None

    if cache_path and cache_path.exists():
        pw_scores = pd.read_parquet(cache_path)
        pw_scores.index = pw_scores.index.astype(str)
        # Align to patient expression index
        common = patient_expr.index.intersection(pw_scores.index)
        if len(common) >= n_patients * 0.5:
            X_patient = np.zeros((n_patients, n_pw), dtype=np.float64)
            for i, pw in enumerate(hallmark_pathway_names):
                if pw in pw_scores.columns:
                    aligned = pw_scores.reindex(patient_expr.index)
                    X_patient[:, i] = aligned[pw].fillna(0).values.astype(np.float64)
        else:
            X_patient = np.zeros((n_patients, n_pw), dtype=np.float64)
    else:
        # Compute on the fly
        try:
            from src.features.pathway_features import compute_ssgsea_scores
            pw_scores = compute_ssgsea_scores(patient_expr, gene_sets="MSigDB_Hallmark_2020")
            X_patient = np.zeros((n_patients, n_pw), dtype=np.float64)
            for i, pw in enumerate(hallmark_pathway_names):
                if pw in pw_scores.columns:
                    X_patient[:, i] = pw_scores[pw].values.astype(np.float64)
        except Exception as e:
            logger.warning("  ssGSEA failed for %s: %s", gse_id, e)
            X_patient = np.zeros((n_patients, n_pw), dtype=np.float64)

    # Mean LINCS Hallmark drug signature
    matched = _match_to_lincs(drug_string, lincs_drug_set)
    drug_sig = np.zeros(n_pw, dtype=np.float64)
    n_d = 0
    for d in matched:
        dl = d.lower()
        if dl in lincs_hallmark_mean:
            sig = lincs_hallmark_mean[dl]
            vals = sig.reindex(hallmark_pathway_names).values.astype(np.float64)
            drug_sig += np.nan_to_num(vals, 0.0)
            n_d += 1
    if n_d > 0:
        drug_sig /= n_d

    # Reversal features
    X_rev = X_patient * drug_sig[np.newaxis, :]
    return np.nan_to_num(X_rev, 0.0)


# ===================================================================
# Build all ablation features for one dataset
# ===================================================================

def build_ablation_features_for_dataset(
    ds: dict,
    lincs_drugs: set[str],
    avg_sigs: pd.DataFrame,
    gene_cols: list[str],
    drug_targets_map: dict[str, list[str]],
    hallmark_gene_sets: dict[str, set[str]],
    gdsc_drug_names: set[str],
    depmap_cache: pd.DataFrame,
    lincs_progeny_mean: dict[str, pd.Series],
    progeny_pathway_names: list[str],
    lincs_hallmark_mean: dict[str, pd.Series],
    hallmark_pathway_names: list[str],
) -> dict | None:
    """
    Build feature matrices for all ablation configs (A-G) for one dataset.

    Returns dict with X_A ... X_G and y, or None if dataset unusable.
    """
    expr = ds["expression"]
    resp = ds["response"]
    drug_string = ds["drug_string"]

    # Match to LINCS
    lincs_matched = _match_to_lincs(drug_string, lincs_drugs)
    if not lincs_matched:
        return None

    # Common genes
    common_genes = sorted(set(gene_cols) & set(expr.columns))
    if len(common_genes) < 50:
        return None

    # Patient z-scores
    patient_mat = expr[common_genes].values.astype(np.float64)
    patient_mat = np.nan_to_num(patient_mat, nan=0.0)
    p_mean = patient_mat.mean(axis=0)
    p_std = patient_mat.std(axis=0)
    p_std[p_std == 0] = 1.0
    patient_z = (patient_mat - p_mean) / p_std

    # Mean drug signature
    sig_lookup = avg_sigs.set_index("pert_iname")
    drug_sig = np.zeros(len(common_genes), dtype=np.float64)
    n_d = 0
    for d in lincs_matched:
        if d in sig_lookup.index:
            vals = sig_lookup.loc[d].reindex(common_genes).values.astype(np.float64)
            drug_sig += np.nan_to_num(vals, nan=0.0)
            n_d += 1
    if n_d == 0:
        return None
    drug_sig /= n_d

    n_patients = patient_z.shape[0]
    y = resp.values.astype(int)

    # Match to GDSC for targets
    gdsc_matched = match_ctrdb_drug_to_gdsc(drug_string, gdsc_drug_names)
    all_targets: set[str] = set()
    for gd in gdsc_matched:
        gd_lower = gd.lower()
        if gd_lower in drug_targets_map:
            all_targets.update(drug_targets_map[gd_lower])
    target_list = sorted(all_targets)

    # --- Config A: Gene reversal features (5 features) ---
    X_reversal = _compute_reversal_features(patient_z, drug_sig)

    # --- Config B: + drug-target interaction features (5 more) ---
    X_dt = _compute_drug_target_features(
        patient_z, common_genes, target_list, hallmark_gene_sets
    )

    # --- Config C: + DepMap dependency features (4 more) ---
    X_depmap = _compute_depmap_features_for_dataset(
        depmap_cache, drug_string, gdsc_drug_names,
        drug_targets_map, n_patients
    )

    # --- Config D: + PROGENy features ---
    X_progeny = _compute_progeny_features_for_dataset(
        expr, drug_string, lincs_progeny_mean, lincs_drugs,
        progeny_pathway_names
    )

    # --- Config G: + ssGSEA Hallmark ---
    gse_id = ds.get("gse_id", "")
    X_ssgsea = _compute_ssgsea_features_for_dataset(
        expr, drug_string, lincs_hallmark_mean, lincs_drugs,
        hallmark_pathway_names, gse_id=gse_id
    )

    # Assemble configs
    X_A = X_reversal
    X_B = np.hstack([X_reversal, X_dt])
    X_C = np.hstack([X_reversal, X_depmap])
    X_D = np.hstack([X_reversal, X_progeny])
    X_E = np.hstack([X_reversal, X_dt, X_depmap])
    X_F = np.hstack([X_reversal, X_dt, X_depmap, X_progeny])
    X_G = np.hstack([X_reversal, X_dt, X_depmap, X_progeny, X_ssgsea])

    return {
        "X_A": X_A,
        "X_B": X_B,
        "X_C": X_C,
        "X_D": X_D,
        "X_E": X_E,
        "X_F": X_F,
        "X_G": X_G,
        "y": y,
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "drug": drug_string,
        "n_targets": len(target_list),
        "gse_id": gse_id,
        "sample_ids": list(expr.index),
    }


# ===================================================================
# LODO ablation
# ===================================================================

CONFIGS = {
    "A_baseline_reversal": "X_A",
    "B_plus_drug_target": "X_B",
    "C_plus_depmap": "X_C",
    "D_plus_progeny": "X_D",
    "E_dt_plus_depmap": "X_E",
    "F_dt_depmap_progeny": "X_F",
    "G_all_plus_ssgsea": "X_G",
}


def run_lodo_ablation(
    dataset_features: dict[str, dict],
    C: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Leave-One-Dataset-Out ablation across all configs.

    Returns:
        results_df: Per-fold, per-config AUC table.
        predictions: config_name -> DataFrame of LODO predictions.
    """
    all_geos = sorted(dataset_features.keys())
    if len(all_geos) < 2:
        logger.warning("Need >= 2 datasets for LODO; got %d", len(all_geos))
        return pd.DataFrame(), {}

    results = []
    predictions: dict[str, list[dict]] = {k: [] for k in CONFIGS}

    for held_out in all_geos:
        train_geos = [g for g in all_geos if g != held_out]

        for config_name, feat_key in CONFIGS.items():
            # Check that all datasets have this feature key
            if feat_key not in dataset_features[held_out]:
                continue

            X_train_parts, y_train_parts = [], []
            for tg in train_geos:
                if feat_key in dataset_features[tg]:
                    X_train_parts.append(dataset_features[tg][feat_key])
                    y_train_parts.append(dataset_features[tg]["y"])

            if not X_train_parts:
                continue

            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            X_test = dataset_features[held_out][feat_key]
            y_test = dataset_features[held_out]["y"]

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_train_s = np.nan_to_num(X_train_s, nan=0.0)
            X_test_s = np.nan_to_num(X_test_s, nan=0.0)

            # Train L1-logistic regression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = LogisticRegression(
                    C=C,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                )
                try:
                    clf.fit(X_train_s, y_train)
                except Exception as e:
                    logger.warning(
                        "  LODO %s / %s: fit failed (%s)",
                        held_out, config_name, e,
                    )
                    continue

            # Predict
            try:
                proba = clf.predict_proba(X_test_s)
                if proba.shape[1] == 2:
                    preds = proba[:, 1]
                    auc = roc_auc_score(y_test, preds)
                else:
                    preds = np.full(len(y_test), 0.5)
                    auc = 0.5
            except Exception:
                preds = np.full(len(y_test), 0.5)
                auc = 0.5

            results.append({
                "held_out_dataset": held_out,
                "config": config_name,
                "auc": round(auc, 4),
                "n_test": len(y_test),
                "n_test_pos": int(y_test.sum()),
                "n_train": len(y_train),
                "n_train_pos": int(y_train.sum()),
                "n_features": X_train.shape[1],
                "drug": dataset_features[held_out]["drug"],
            })

            # Save predictions
            sample_ids = dataset_features[held_out].get("sample_ids", [])
            if len(sample_ids) != len(y_test):
                sample_ids = [f"sample_{i}" for i in range(len(y_test))]

            for i in range(len(y_test)):
                predictions[config_name].append({
                    "dataset_id": held_out,
                    "sample_id": sample_ids[i],
                    "prediction": round(float(preds[i]), 6),
                    "true_label": int(y_test[i]),
                })

            logger.info(
                "  LODO %s [%s]: AUC=%.3f (test=%d, feats=%d)",
                held_out, config_name, auc,
                len(y_test), X_train.shape[1],
            )

    results_df = pd.DataFrame(results)
    pred_dfs = {k: pd.DataFrame(v) for k, v in predictions.items() if v}
    return results_df, pred_dfs


# ===================================================================
# Main pipeline
# ===================================================================

def run_ablation_pipeline():
    """
    End-to-end structured ablation retraining:
    1. Load LINCS signatures and build drug profiles
    2. Load Hallmark gene sets and PROGENy/ssGSEA drug-level signatures
    3. Load DepMap priors cache
    4. Load CTR-DB datasets
    5. Build features per dataset for all configs
    6. Run LODO ablation
    7. Save per-dataset results + predictions + trajectory table
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.time()

    logger.info("=" * 70)
    logger.info("STRUCTURED ABLATION RETRAINING PIPELINE")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1: Load LINCS signatures                                       #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 1: Loading LINCS signatures ...")

    lincs_path = DATA_CACHE / "all_cellline_drug_signatures.parquet"
    if not lincs_path.exists():
        lincs_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    if not lincs_path.exists():
        logger.error("No LINCS signatures found.")
        return

    lincs_sigs = pd.read_parquet(lincs_path)
    gene_cols = [c for c in lincs_sigs.columns if c not in META_COLS]
    lincs_drugs = set(lincs_sigs["pert_iname"].unique())
    logger.info(
        "  %d signatures, %d drugs, %d genes",
        len(lincs_sigs), len(lincs_drugs), len(gene_cols),
    )

    avg_sigs = (
        lincs_sigs.groupby("pert_iname")[gene_cols]
        .mean()
        .reset_index()
    )
    logger.info("  %d dose-averaged drug profiles", len(avg_sigs))

    # ------------------------------------------------------------------ #
    # Step 2: Load feature dependencies                                   #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 2: Loading feature dependencies ...")

    # Drug targets
    drug_targets_map = parse_drug_targets()
    logger.info("  %d drugs with gene-level targets", len(drug_targets_map))

    # Hallmark gene sets
    hallmark_gene_sets = load_hallmark_gene_sets()
    logger.info("  %d Hallmark gene sets", len(hallmark_gene_sets))

    # GDSC drug names
    ref = pd.read_parquet(DATA_CACHE / "breast_dose_response_ref.parquet")
    gdsc_drug_names = set(ref["drug_name"].unique())
    logger.info("  %d GDSC2 drugs", len(gdsc_drug_names))

    # DepMap priors cache
    depmap_cache_path = DATA_CACHE / "depmap_target_priors.parquet"
    if depmap_cache_path.exists():
        depmap_cache = pd.read_parquet(depmap_cache_path)
        logger.info("  DepMap priors: %d rows", len(depmap_cache))
    else:
        logger.warning("  DepMap priors not found; using zeros")
        depmap_cache = pd.DataFrame(columns=[
            "drug_name", "subtype", "depmap_target_essentiality",
            "depmap_target_selectivity", "depmap_n_essential_targets",
            "depmap_mutation_vulnerability",
        ])

    # PROGENy LINCS signatures
    lincs_progeny_path = DATA_CACHE / "lincs_progeny.parquet"
    if lincs_progeny_path.exists():
        lincs_progeny_df = pd.read_parquet(lincs_progeny_path)
        progeny_meta = {"sig_id", "pert_id", "pert_iname", "cell_id",
                        "dose_um", "pert_idose"}
        progeny_cols = [c for c in lincs_progeny_df.columns if c not in progeny_meta]
        progeny_pathway_names = sorted(progeny_cols)
        lincs_progeny_mean = {}
        for drug, grp in lincs_progeny_df.groupby(
            lincs_progeny_df["pert_iname"].str.lower()
        ):
            lincs_progeny_mean[drug] = grp[progeny_cols].mean()
        logger.info("  LINCS PROGENy: %d drugs, %d pathways",
                     len(lincs_progeny_mean), len(progeny_pathway_names))
    else:
        logger.warning("  LINCS PROGENy not cached; PROGENy features will be computed per-dataset")
        progeny_pathway_names = sorted(PROGENY_PATHWAYS)
        lincs_progeny_mean = {}

    # Hallmark LINCS signatures
    lincs_hallmark_path = DATA_CACHE / "lincs_pathway_MSigDB_Hallmark_2020.parquet"
    if lincs_hallmark_path.exists():
        lincs_hallmark_df = pd.read_parquet(lincs_hallmark_path)
        hallmark_meta = {"sig_id", "pert_id", "pert_iname", "cell_id",
                         "dose_um", "pert_idose"}
        hallmark_pw_cols = [c for c in lincs_hallmark_df.columns if c not in hallmark_meta]
        hallmark_pathway_names = sorted(hallmark_pw_cols)
        lincs_hallmark_mean = {}
        for drug, grp in lincs_hallmark_df.groupby(
            lincs_hallmark_df["pert_iname"].str.lower()
        ):
            lincs_hallmark_mean[drug] = grp[hallmark_pw_cols].mean()
        logger.info("  LINCS Hallmark: %d drugs, %d pathways",
                     len(lincs_hallmark_mean), len(hallmark_pathway_names))
    else:
        logger.warning("  LINCS Hallmark not cached; ssGSEA features will be computed per-dataset")
        hallmark_pathway_names = []
        lincs_hallmark_mean = {}

    # ------------------------------------------------------------------ #
    # Step 3: Load CTR-DB datasets                                        #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 3: Loading CTR-DB datasets ...")
    datasets = load_ctrdb_datasets()
    logger.info("  %d CTR-DB datasets loaded", len(datasets))

    if len(datasets) < 2:
        logger.error("Not enough CTR-DB datasets for LODO evaluation.")
        return

    # ------------------------------------------------------------------ #
    # Step 4: Build features per dataset for all configs                  #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 4: Building features per dataset for all ablation configs ...")

    dataset_features: dict[str, dict] = {}
    for ds in datasets:
        gse_id = ds["gse_id"]
        logger.info("  Processing %s (drug=%s) ...", gse_id, ds["drug_string"])

        feats = build_ablation_features_for_dataset(
            ds=ds,
            lincs_drugs=lincs_drugs,
            avg_sigs=avg_sigs,
            gene_cols=gene_cols,
            drug_targets_map=drug_targets_map,
            hallmark_gene_sets=hallmark_gene_sets,
            gdsc_drug_names=gdsc_drug_names,
            depmap_cache=depmap_cache,
            lincs_progeny_mean=lincs_progeny_mean,
            progeny_pathway_names=progeny_pathway_names,
            lincs_hallmark_mean=lincs_hallmark_mean,
            hallmark_pathway_names=hallmark_pathway_names,
        )

        if feats is not None:
            dataset_features[gse_id] = feats
            logger.info(
                "    A=%d, B=%d, C=%d, D=%d, E=%d, F=%d, G=%d feats "
                "(targets=%d, n=%d)",
                feats["X_A"].shape[1],
                feats["X_B"].shape[1],
                feats["X_C"].shape[1],
                feats["X_D"].shape[1],
                feats["X_E"].shape[1],
                feats["X_F"].shape[1],
                feats["X_G"].shape[1],
                feats.get("n_targets", 0),
                len(feats["y"]),
            )
        else:
            logger.info("    %s: skipped (no LINCS match or too few genes)", gse_id)

    logger.info(
        "\n  %d / %d datasets ready for LODO",
        len(dataset_features), len(datasets),
    )

    if len(dataset_features) < 2:
        logger.error("Not enough datasets with valid features for LODO.")
        return

    # ------------------------------------------------------------------ #
    # Step 5: LODO ablation                                               #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: LODO ablation evaluation")
    logger.info("=" * 70)

    lodo_results, predictions = run_lodo_ablation(dataset_features, C=0.05)

    # ------------------------------------------------------------------ #
    # Step 6: Save results                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Saving results")
    logger.info("=" * 70)

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Per-fold results
    if not lodo_results.empty:
        ablation_path = RESULTS / "feature_ablation_retraining.tsv"
        lodo_results.to_csv(ablation_path, sep="\t", index=False)
        logger.info("  Saved per-fold results to %s", ablation_path)

    # Predictions per config
    for config_name, pred_df in predictions.items():
        if not pred_df.empty:
            # Extract letter from config name
            letter = config_name.split("_")[0]
            pred_path = RESULTS / f"lodo_predictions_config_{letter}.tsv"
            pred_df.to_csv(pred_path, sep="\t", index=False)
            logger.info("  Saved predictions for %s to %s", config_name, pred_path)

    # ------------------------------------------------------------------ #
    # Step 7: Summary and trajectory table                                #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 70)
    logger.info("Step 7: Ablation summary and improvement trajectory")
    logger.info("=" * 70)

    if not lodo_results.empty:
        # Summary by config
        summary = (
            lodo_results.groupby("config")["auc"]
            .agg(["mean", "std", "median", "count"])
            .round(4)
            .reset_index()
        )
        summary.columns = ["config", "mean_auc", "std_auc", "median_auc", "n_datasets"]

        logger.info("\nABLATION SUMMARY (all datasets):")
        for _, row in summary.iterrows():
            logger.info(
                "  %-35s: AUC = %.4f +/- %.4f  (median=%.4f, n=%d)",
                row["config"],
                row["mean_auc"],
                row["std_auc"],
                row["median_auc"],
                int(row["n_datasets"]),
            )

        # Separate pCR datasets (breast cancer pathologic_response)
        # Load catalog to identify pCR datasets
        cat_path = CTRDB_DIR / "catalog.csv"
        pcr_datasets = set()
        if cat_path.exists():
            cat = pd.read_csv(cat_path)
            for _, row in cat.iterrows():
                resp_grp = str(row.get("response_grouping", "")).lower()
                if "pcr" in resp_grp or "pathologic" in resp_grp:
                    pcr_datasets.add(row["geo_source"])
        logger.info("  pCR datasets: %s", sorted(pcr_datasets))

        # pCR-specific AUC
        pcr_results = lodo_results[
            lodo_results["held_out_dataset"].isin(pcr_datasets)
        ]
        if not pcr_results.empty:
            pcr_summary = (
                pcr_results.groupby("config")["auc"]
                .agg(["mean", "count"])
                .round(4)
                .reset_index()
            )
            pcr_summary.columns = ["config", "pcr_mean_auc", "n_pcr_datasets"]
            logger.info("\npCR SUBSET:")
            for _, row in pcr_summary.iterrows():
                logger.info(
                    "  %-35s: pCR AUC = %.4f (n=%d)",
                    row["config"],
                    row["pcr_mean_auc"],
                    int(row["n_pcr_datasets"]),
                )
        else:
            pcr_summary = pd.DataFrame(columns=["config", "pcr_mean_auc", "n_pcr_datasets"])

        # Build trajectory table
        _build_trajectory_table(summary, pcr_summary)

    elapsed = time.time() - t_start
    logger.info("\nTotal pipeline time: %.0fs (%.1f min)", elapsed, elapsed / 60)


def _build_trajectory_table(
    summary: pd.DataFrame,
    pcr_summary: pd.DataFrame,
) -> None:
    """Build and save the full improvement trajectory table."""

    # Map config names to readable labels
    config_labels = {
        "A_baseline_reversal":    "A. Baseline (gene reversal)",
        "B_plus_drug_target":     "B. + drug-target features",
        "C_plus_depmap":          "C. + DepMap features",
        "D_plus_progeny":         "D. + PROGENy features",
        "E_dt_plus_depmap":       "E. + drug-target + DepMap",
        "F_dt_depmap_progeny":    "F. + drug-target + DepMap + PROGENy",
        "G_all_plus_ssgsea":      "G. + all + ssGSEA",
    }

    rows = []
    for _, row in summary.iterrows():
        config = row["config"]
        label = config_labels.get(config, config)
        pan_auc = row["mean_auc"]

        # Get pCR AUC if available
        pcr_match = pcr_summary[pcr_summary["config"] == config]
        pcr_auc = float(pcr_match["pcr_mean_auc"].iloc[0]) if not pcr_match.empty else None

        rows.append({
            "Config": label,
            "Pan-cancer_AUC": round(pan_auc, 4),
            "pCR_AUC": round(pcr_auc, 4) if pcr_auc is not None else "N/A",
            "Retrained": "Yes",
        })

    # Add reference rows from existing results
    # CDS-DB reversal baseline
    lodo_summary_path = RESULTS / "lodo_summary.tsv"
    if lodo_summary_path.exists():
        lodo_summary = pd.read_csv(lodo_summary_path, sep="\t")
        overall_auc = lodo_summary.loc[
            lodo_summary["metric"] == "mean_auroc", "overall"
        ]
        pcr_ref_auc = lodo_summary.loc[
            lodo_summary["metric"] == "mean_auroc", "pathologic_response"
        ]
        if not overall_auc.empty:
            rows.append({
                "Config": "CDS-DB reversal (existing baseline)",
                "Pan-cancer_AUC": round(float(overall_auc.iloc[0]), 4),
                "pCR_AUC": round(float(pcr_ref_auc.iloc[0]), 4) if not pcr_ref_auc.empty else "N/A",
                "Retrained": "No",
            })

    # Ensemble
    ensemble_path = RESULTS / "ensemble_comparison.tsv"
    if ensemble_path.exists():
        ens = pd.read_csv(ensemble_path, sep="\t")
        if "auc_ensemble" in ens.columns:
            ens_mean = ens["auc_ensemble"].mean()
            rows.append({
                "Config": "Stacking ensemble",
                "Pan-cancer_AUC": round(float(ens_mean), 4),
                "pCR_AUC": "N/A",
                "Retrained": "Meta-learner",
            })

    # Pan-cancer patient model
    pan_cancer_path = RESULTS / "pan_cancer_model_lodo_results.csv"
    if pan_cancer_path.exists():
        pan = pd.read_csv(pan_cancer_path)
        if "auc" in pan.columns:
            pan_mean = pan["auc"].mean()
            # pCR subset from pan-cancer
            pan_breast = pan[pan.get("cancer_type", pan.get("held_out_dataset", "")).str.contains("Breast|breast|GSE20194|GSE20271|GSE25066|GSE32646|GSE4779|GSE6861", case=False, na=False)]
            pan_pcr = pan_breast["auc"].mean() if not pan_breast.empty else None
            rows.append({
                "Config": "Pan-cancer patient model",
                "Pan-cancer_AUC": round(float(pan_mean), 4),
                "pCR_AUC": round(float(pan_pcr), 4) if pan_pcr is not None else "N/A",
                "Retrained": "Ref",
            })

    trajectory = pd.DataFrame(rows)
    traj_path = RESULTS / "full_improvement_trajectory.tsv"
    trajectory.to_csv(traj_path, sep="\t", index=False)
    logger.info("\n  Saved improvement trajectory to %s", traj_path)

    logger.info("\n" + "=" * 70)
    logger.info("FULL IMPROVEMENT TRAJECTORY")
    logger.info("=" * 70)
    for _, row in trajectory.iterrows():
        logger.info(
            "  %-45s  Pan-cancer=%-8s  pCR=%-8s  %s",
            row["Config"],
            str(row["Pan-cancer_AUC"]),
            str(row["pCR_AUC"]),
            row["Retrained"],
        )


if __name__ == "__main__":
    run_ablation_pipeline()
