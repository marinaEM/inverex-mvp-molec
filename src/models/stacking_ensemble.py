"""
Stacking ensemble for patient drug-response prediction (Step 3).

Collects LODO out-of-fold predictions from multiple base models, then
trains a logistic-regression meta-learner on the stacked predictions.
The meta-learner itself is evaluated with LODO to prevent meta-leakage.

Base models
-----------
1. Reversal (uniform): -corr(patient_sig, drug_sig)
2. Reversal + drug-target features (Step 1)
3. Patient LightGBM: L1-logistic on gene reversal features
4. ssGSEA pathway reversal: from src/features/pathway_features.py

Meta-learner
------------
LogisticRegressionCV(cv=3, max_iter=1000), trained in LODO fashion.

Entry point: ``run_stacking_ensemble()`` or
``pixi run python -m src.models.stacking_ensemble``
"""

import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CACHE, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# Meta columns in LINCS signatures
META_COLS = frozenset(
    {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
)


# ======================================================================
# 1. Load shared data
# ======================================================================

def _load_ctrdb_catalog() -> pd.DataFrame:
    """Load CTR-DB catalog."""
    cat_path = DATA_RAW / "ctrdb" / "catalog.csv"
    return pd.read_csv(cat_path)


def _load_ctrdb_datasets() -> list[dict]:
    """
    Load CTR-DB datasets with expression and response labels.
    Returns list of dicts with gse_id, drug_string, expression, response.
    """
    ctrdb_dir = DATA_RAW / "ctrdb"
    catalog = _load_ctrdb_catalog()
    gse_drug_map = dict(zip(catalog["geo_source"], catalog["drug"]))

    datasets = []
    for gse_id in sorted(os.listdir(ctrdb_dir)):
        gse_dir = ctrdb_dir / gse_id
        if not gse_dir.is_dir() or not gse_id.startswith("GSE"):
            continue

        expr_path = gse_dir / f"{gse_id}_expression.parquet"
        resp_path = gse_dir / "response_labels.parquet"
        if not expr_path.exists() or not resp_path.exists():
            continue

        drug_string = gse_drug_map.get(gse_id)
        if drug_string is None:
            continue

        try:
            expression = pd.read_parquet(expr_path)
            response = pd.read_parquet(resp_path).squeeze()

            common = expression.index.intersection(response.index)
            if len(common) < 10:
                continue

            expression = expression.loc[common]
            response = response.loc[common]

            n_pos = int(response.sum())
            n_neg = len(response) - n_pos
            if n_pos < 3 or n_neg < 3:
                continue

            datasets.append({
                "gse_id": gse_id,
                "drug_string": drug_string,
                "expression": expression,
                "response": response,
            })
        except Exception as e:
            logger.warning("Failed to load %s: %s", gse_id, e)

    logger.info("Loaded %d CTR-DB datasets", len(datasets))
    return datasets


def _load_lincs_signatures() -> tuple[pd.DataFrame, list[str]]:
    """Load LINCS signatures, return (df, gene_cols)."""
    lincs_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    lincs = pd.read_parquet(lincs_path)
    gene_cols = [c for c in lincs.columns if c not in META_COLS]
    return lincs, gene_cols


def _normalise_drug_name(name: str) -> str:
    """Lower-case, strip hyphens/spaces."""
    return re.sub(r"[\s\-_]+", "", name.strip().lower())


def _parse_regimen_components(drug_string: str) -> list[str]:
    """Parse a CTR-DB drug string into normalised component names."""
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
        norm = _normalise_drug_name(p)
        if norm and len(norm) > 1:
            components.append(norm)
    return list(dict.fromkeys(components))


def _match_drugs_to_lincs(
    components: list[str],
    lincs_drug_set: set[str],
) -> list[str]:
    """Return subset of components found in LINCS."""
    lincs_norm = {_normalise_drug_name(d): d for d in lincs_drug_set}
    matched = []
    for comp in components:
        cn = _normalise_drug_name(comp)
        if cn in lincs_norm:
            matched.append(lincs_norm[cn])
    return matched


# ======================================================================
# 2. Base model: Uniform reversal score
# ======================================================================

def compute_reversal_scores(
    patient_z: np.ndarray,
    drug_sig: np.ndarray,
) -> np.ndarray:
    """
    Compute -corr(patient, drug_sig) for each patient.
    Returns array of shape (n_patients,).
    """
    n = patient_z.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p = patient_z[i]
        if np.std(p) > 0 and np.std(drug_sig) > 0:
            r, _ = pearsonr(p, drug_sig)
            scores[i] = -r
        else:
            scores[i] = 0.0
    return scores


# ======================================================================
# 3. Base model: Gene reversal features for L1-logistic
# ======================================================================

def compute_gene_reversal_features(
    patient_z: np.ndarray,
    drug_sig: np.ndarray,
) -> np.ndarray:
    """
    Element-wise product + summary features.
    Returns (n_patients, 5) array.
    """
    rev_product = patient_z * drug_sig[np.newaxis, :]
    rev_mean = np.nanmean(rev_product, axis=1, keepdims=True)
    rev_std = np.nanstd(rev_product, axis=1, keepdims=True)
    rev_min = np.nanmin(rev_product, axis=1, keepdims=True)
    rev_neg_frac = (
        np.sum(rev_product < 0, axis=1, keepdims=True).astype(np.float64)
        / max(rev_product.shape[1], 1)
    )
    rev_scores = compute_reversal_scores(patient_z, drug_sig).reshape(-1, 1)
    return np.hstack([rev_mean, rev_std, rev_min, rev_neg_frac, rev_scores])


# ======================================================================
# 4. Base model: Drug-target features (from Step 1)
# ======================================================================

def compute_drug_target_features(
    patient_z: np.ndarray,
    common_genes: list[str],
    drug_string: str,
    drug_targets: dict[str, list[str]],
    gdsc_drug_names: set[str],
    hallmark: dict[str, set[str]],
) -> np.ndarray:
    """
    Compute drug-target interaction features (Step 1 features).
    Returns (n_patients, 5) for target expression + pathway context + compat.
    """
    from src.features.drug_target_interactions import (
        compute_target_expression_features,
        compute_pathway_context_features,
        compute_compatibility_features,
        match_ctrdb_drug_to_gdsc,
    )

    gdsc_matched = match_ctrdb_drug_to_gdsc(drug_string, gdsc_drug_names)
    all_targets: set[str] = set()
    for gd in gdsc_matched:
        gd_lower = gd.lower()
        if gd_lower in drug_targets:
            all_targets.update(drug_targets[gd_lower])

    n_patients = patient_z.shape[0]
    target_list = sorted(all_targets)

    if target_list:
        X_tgt = compute_target_expression_features(
            patient_z, common_genes, target_list
        )
        X_pw = compute_pathway_context_features(
            patient_z, common_genes, target_list, hallmark
        )
        X_compat = compute_compatibility_features(
            patient_z, common_genes, target_list
        )
    else:
        X_tgt = np.zeros((n_patients, 3), dtype=np.float64)
        X_pw = np.zeros((n_patients, 1), dtype=np.float64)
        X_compat = np.zeros((n_patients, 1), dtype=np.float64)

    return np.hstack([X_tgt, X_pw, X_compat])


# ======================================================================
# 5. Base model: ssGSEA pathway reversal
# ======================================================================

def compute_pathway_reversal_score(
    patient_pw_scores: np.ndarray,
    drug_pw_sig: np.ndarray,
) -> np.ndarray:
    """
    Reversal score at pathway level: -corr(patient_pathway, drug_pathway).
    Returns (n_patients,).
    """
    n = patient_pw_scores.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p = patient_pw_scores[i]
        if np.std(p) > 0 and np.std(drug_pw_sig) > 0:
            r, _ = pearsonr(p, drug_pw_sig)
            scores[i] = -r
        else:
            scores[i] = 0.0
    return scores


# ======================================================================
# 6. LODO base-model predictions
# ======================================================================

def generate_base_model_predictions(
    datasets: list[dict],
    lincs: pd.DataFrame,
    gene_cols: list[str],
) -> pd.DataFrame:
    """
    For each CTR-DB dataset, compute predictions from each base model.

    Returns DataFrame with columns:
        dataset_id, sample_idx, true_label,
        pred_reversal, pred_reversal_dt, pred_lgbm, pred_ssgsea
    """
    lincs_drugs = set(lincs["pert_iname"].unique())

    # Pre-compute dose-averaged signatures
    avg_sigs = lincs.groupby("pert_iname")[gene_cols].mean()

    # Load drug targets for Step 1 features
    from src.features.drug_target_interactions import (
        parse_drug_targets,
        load_hallmark_gene_sets,
    )
    drug_targets = parse_drug_targets()
    ref = pd.read_parquet(DATA_CACHE / "breast_dose_response_ref.parquet")
    gdsc_drug_names = set(ref["drug_name"].unique())

    try:
        hallmark = load_hallmark_gene_sets()
    except Exception:
        hallmark = {}
        logger.warning("Could not load Hallmark gene sets; drug-target pathway features will be zero")

    # Try to load cached pathway scores for ssGSEA
    pw_cache_dir = DATA_CACHE / "pathway_ctrdb_MSigDB_Hallmark_2020"
    lincs_pw_path = DATA_CACHE / "lincs_pathway_MSigDB_Hallmark_2020.parquet"

    has_pathway = False
    lincs_pw = None
    if lincs_pw_path.exists():
        lincs_pw = pd.read_parquet(lincs_pw_path)
        pw_meta = {"sig_id", "pert_id", "pert_iname", "cell_id", "dose_um", "pert_idose"}
        pw_cols = [c for c in lincs_pw.columns if c not in pw_meta]
        if pw_cols:
            has_pathway = True
            # Per-drug mean pathway signatures
            drug_mean_pw = {}
            for drug, grp in lincs_pw.groupby(lincs_pw["pert_iname"].str.lower()):
                drug_mean_pw[drug] = grp[pw_cols].mean().values

    all_records = []

    for ds in datasets:
        gse_id = ds["gse_id"]
        drug_string = ds["drug_string"]
        expr = ds["expression"]
        resp = ds["response"]

        # Match to LINCS
        components = _parse_regimen_components(drug_string)
        matched = _match_drugs_to_lincs(components, lincs_drugs)
        if not matched:
            logger.info("  %s: no LINCS match, skipping", gse_id)
            continue

        # Common genes
        common_genes = sorted(set(gene_cols) & set(expr.columns))
        if len(common_genes) < 50:
            logger.info("  %s: too few common genes (%d)", gse_id, len(common_genes))
            continue

        # Z-score patient expression
        patient_mat = expr[common_genes].values.astype(np.float64)
        patient_mat = np.nan_to_num(patient_mat, nan=0.0)
        p_mean = patient_mat.mean(axis=0)
        p_std = patient_mat.std(axis=0)
        p_std[p_std == 0] = 1.0
        patient_z = (patient_mat - p_mean) / p_std

        # Mean drug signature across matched LINCS drugs
        drug_sig = np.zeros(len(common_genes), dtype=np.float64)
        n_d = 0
        for d in matched:
            if d in avg_sigs.index:
                vals = avg_sigs.loc[d].reindex(common_genes).values.astype(np.float64)
                drug_sig += np.nan_to_num(vals, nan=0.0)
                n_d += 1
        if n_d == 0:
            logger.info("  %s: no LINCS sigs for matched drugs", gse_id)
            continue
        drug_sig /= n_d

        n_patients = patient_z.shape[0]
        y = resp.values.astype(int)

        # --- Model 1: Uniform reversal ---
        pred_reversal = compute_reversal_scores(patient_z, drug_sig)

        # --- Model 2: Reversal + drug-target features ---
        X_reversal = compute_gene_reversal_features(patient_z, drug_sig)
        X_dt = compute_drug_target_features(
            patient_z, common_genes, drug_string,
            drug_targets, gdsc_drug_names, hallmark,
        )
        X_rev_dt = np.hstack([X_reversal, X_dt])
        # Store for LODO (we'll compute predictions in LODO loop later)
        # For now, store the raw feature values
        pred_reversal_dt = np.full(n_patients, np.nan)

        # --- Model 3: L1-logistic on gene reversal features ---
        pred_lgbm = np.full(n_patients, np.nan)

        # --- Model 4: ssGSEA pathway reversal ---
        pred_ssgsea = np.full(n_patients, np.nan)
        if has_pathway and pw_cache_dir.exists():
            pw_path = pw_cache_dir / f"{gse_id}_pathway.parquet"
            if pw_path.exists():
                pw_scores = pd.read_parquet(pw_path)
                pw_scores.index = pw_scores.index.astype(str)
                common_idx = pw_scores.index.intersection(resp.index)
                if len(common_idx) > 5:
                    pw_patient = pw_scores.loc[
                        resp.index.intersection(pw_scores.index)
                    ]
                    # Mean drug pathway sig
                    drug_pw_sig_arr = np.zeros(len(pw_cols), dtype=np.float64)
                    nd = 0
                    for d in matched:
                        dl = d.lower()
                        if dl in drug_mean_pw:
                            drug_pw_sig_arr += drug_mean_pw[dl]
                            nd += 1
                    if nd > 0:
                        drug_pw_sig_arr /= nd
                        # Align columns
                        pw_avail = [c for c in pw_cols if c in pw_patient.columns]
                        pw_idx_map = {c: i for i, c in enumerate(pw_cols)}
                        if len(pw_avail) >= 5:
                            pw_mat = pw_patient[pw_avail].values.astype(np.float64)
                            pw_mat = np.nan_to_num(pw_mat, nan=0.0)
                            pw_drug_aligned = np.array([
                                drug_pw_sig_arr[pw_idx_map[c]] for c in pw_avail
                            ])
                            ssgsea_scores = compute_pathway_reversal_score(
                                pw_mat, pw_drug_aligned
                            )
                            # Map back to full patient index
                            overlap = resp.index.intersection(pw_patient.index)
                            for ii, sample_id in enumerate(resp.index):
                                if sample_id in pw_patient.index:
                                    loc_in_pw = list(pw_patient.index).index(sample_id)
                                    if loc_in_pw < len(ssgsea_scores):
                                        pred_ssgsea[ii] = ssgsea_scores[loc_in_pw]

        for i in range(n_patients):
            all_records.append({
                "dataset_id": gse_id,
                "sample_idx": i,
                "true_label": y[i],
                "pred_reversal": pred_reversal[i],
                "drug_string": drug_string,
            })

        # Store feature matrices for LODO training of models 2, 3, 4
        ds["_common_genes"] = common_genes
        ds["_patient_z"] = patient_z
        ds["_drug_sig"] = drug_sig
        ds["_X_reversal"] = X_reversal
        ds["_X_rev_dt"] = X_rev_dt
        ds["_y"] = y
        ds["_pred_ssgsea_raw"] = pred_ssgsea
        ds["_matched_drugs"] = matched

    return pd.DataFrame(all_records), datasets


# ======================================================================
# 7. LODO evaluation for all models + meta-learner
# ======================================================================

def run_lodo_all_models(
    datasets: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LODO evaluation for all base models and the stacking ensemble.

    For models that require training (2, 3), the LODO split is used.
    For the meta-learner, an outer LODO loop prevents meta-leakage.

    Returns
    -------
    per_dataset : DataFrame
        Per-dataset AUC for each model + ensemble.
    meta_weights : DataFrame
        Meta-learner coefficients.
    """
    # Filter to datasets with valid features
    valid_ds = [ds for ds in datasets if "_patient_z" in ds]
    if len(valid_ds) < 2:
        logger.warning("Need >= 2 datasets for LODO, got %d", len(valid_ds))
        return pd.DataFrame(), pd.DataFrame()

    all_geos = [ds["gse_id"] for ds in valid_ds]
    logger.info("LODO with %d datasets: %s", len(all_geos), all_geos)

    per_dataset_results = []
    all_meta_preds = []
    all_meta_weights = []

    for hold_idx, held_out_ds in enumerate(valid_ds):
        held_out_geo = held_out_ds["gse_id"]
        train_ds_list = [ds for ds in valid_ds if ds["gse_id"] != held_out_geo]

        X_test_rev = held_out_ds["_X_reversal"]
        X_test_rev_dt = held_out_ds["_X_rev_dt"]
        y_test = held_out_ds["_y"]
        n_test = len(y_test)

        # ---- Model 1: Uniform reversal (no training needed) ----
        pred_reversal = compute_reversal_scores(
            held_out_ds["_patient_z"], held_out_ds["_drug_sig"]
        )

        # ---- Model 2: Reversal + drug-target (L1-logistic, LODO) ----
        X_train_parts_dt = [ds["_X_rev_dt"] for ds in train_ds_list]
        y_train_parts_dt = [ds["_y"] for ds in train_ds_list]
        X_train_dt = np.concatenate(X_train_parts_dt, axis=0)
        y_train_dt = np.concatenate(y_train_parts_dt, axis=0)

        scaler_dt = StandardScaler()
        X_train_dt_s = scaler_dt.fit_transform(X_train_dt)
        X_test_dt_s = scaler_dt.transform(X_test_rev_dt)
        X_train_dt_s = np.nan_to_num(X_train_dt_s, nan=0.0)
        X_test_dt_s = np.nan_to_num(X_test_dt_s, nan=0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf_dt = LogisticRegression(
                C=0.05, solver="liblinear", max_iter=2000, random_state=42,
            )
            try:
                clf_dt.fit(X_train_dt_s, y_train_dt)
                pred_rev_dt = clf_dt.predict_proba(X_test_dt_s)[:, 1]
            except Exception:
                pred_rev_dt = np.full(n_test, 0.5)

        # ---- Model 3: L1-logistic on gene reversal features ----
        X_train_parts_rev = [ds["_X_reversal"] for ds in train_ds_list]
        y_train_parts_rev = [ds["_y"] for ds in train_ds_list]
        X_train_rev = np.concatenate(X_train_parts_rev, axis=0)
        y_train_rev = np.concatenate(y_train_parts_rev, axis=0)

        scaler_rev = StandardScaler()
        X_train_rev_s = scaler_rev.fit_transform(X_train_rev)
        X_test_rev_s = scaler_rev.transform(X_test_rev)
        X_train_rev_s = np.nan_to_num(X_train_rev_s, nan=0.0)
        X_test_rev_s = np.nan_to_num(X_test_rev_s, nan=0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf_rev = LogisticRegression(
                C=0.05, solver="liblinear", max_iter=2000, random_state=42,
            )
            try:
                clf_rev.fit(X_train_rev_s, y_train_rev)
                pred_lgbm = clf_rev.predict_proba(X_test_rev_s)[:, 1]
            except Exception:
                pred_lgbm = np.full(n_test, 0.5)

        # ---- Model 4: ssGSEA pathway reversal score ----
        pred_ssgsea = held_out_ds.get("_pred_ssgsea_raw", np.full(n_test, 0.0))
        if pred_ssgsea is None or np.all(np.isnan(pred_ssgsea)):
            pred_ssgsea = np.zeros(n_test, dtype=np.float64)
        pred_ssgsea = np.nan_to_num(pred_ssgsea, nan=0.0)

        # ---- Compute per-model AUCs ----
        model_preds = {
            "reversal_uniform": pred_reversal,
            "reversal_plus_dt": pred_rev_dt,
            "lgbm_reversal": pred_lgbm,
            "ssgsea_pathway": pred_ssgsea,
        }

        row = {"dataset_id": held_out_geo, "drug": held_out_ds["drug_string"],
               "n_test": n_test, "n_pos": int(y_test.sum())}

        for mname, mpred in model_preds.items():
            try:
                auc = roc_auc_score(y_test, mpred)
            except ValueError:
                auc = 0.5
            row[f"auc_{mname}"] = round(auc, 4)

        # ---- Stack predictions for meta-learner ----
        X_meta_test = np.column_stack([
            pred_reversal, pred_rev_dt, pred_lgbm, pred_ssgsea
        ])

        # Meta-learner: train on all other datasets' predictions
        X_meta_train_parts = []
        y_meta_train_parts = []
        for tr_ds in train_ds_list:
            tr_geo = tr_ds["gse_id"]
            # We need to compute base model predictions for training datasets too
            # For the meta-learner training, use inner-LODO predictions
            # Simplified: train each base model leaving out tr_ds as well,
            # but this is expensive. Use a pragmatic approach: leave-one-out
            # from the remaining training datasets.
            tr_y = tr_ds["_y"]
            tr_n = len(tr_y)

            # Reversal score (no training needed)
            tr_reversal = compute_reversal_scores(
                tr_ds["_patient_z"], tr_ds["_drug_sig"]
            )

            # For trained models: train on all except tr_ds and held_out
            inner_train = [
                d for d in valid_ds
                if d["gse_id"] != held_out_geo and d["gse_id"] != tr_geo
            ]
            if not inner_train:
                # Can't do inner LODO, use self-predictions (biased but only option)
                tr_rev_dt = np.full(tr_n, 0.5)
                tr_lgbm = np.full(tr_n, 0.5)
            else:
                # Model 2: reversal + DT
                Xi_dt = np.concatenate([d["_X_rev_dt"] for d in inner_train])
                yi_dt = np.concatenate([d["_y"] for d in inner_train])
                sc_dt = StandardScaler()
                Xi_dt_s = sc_dt.fit_transform(Xi_dt)
                Xt_dt_s = sc_dt.transform(tr_ds["_X_rev_dt"])
                Xi_dt_s = np.nan_to_num(Xi_dt_s, nan=0.0)
                Xt_dt_s = np.nan_to_num(Xt_dt_s, nan=0.0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    try:
                        c2 = LogisticRegression(
                            C=0.05, solver="liblinear", max_iter=2000, random_state=42,
                        )
                        c2.fit(Xi_dt_s, yi_dt)
                        tr_rev_dt = c2.predict_proba(Xt_dt_s)[:, 1]
                    except Exception:
                        tr_rev_dt = np.full(tr_n, 0.5)

                # Model 3: gene reversal L1-logistic
                Xi_rev = np.concatenate([d["_X_reversal"] for d in inner_train])
                yi_rev = np.concatenate([d["_y"] for d in inner_train])
                sc_rev = StandardScaler()
                Xi_rev_s = sc_rev.fit_transform(Xi_rev)
                Xt_rev_s = sc_rev.transform(tr_ds["_X_reversal"])
                Xi_rev_s = np.nan_to_num(Xi_rev_s, nan=0.0)
                Xt_rev_s = np.nan_to_num(Xt_rev_s, nan=0.0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    try:
                        c3 = LogisticRegression(
                            C=0.05, solver="liblinear", max_iter=2000, random_state=42,
                        )
                        c3.fit(Xi_rev_s, yi_rev)
                        tr_lgbm = c3.predict_proba(Xt_rev_s)[:, 1]
                    except Exception:
                        tr_lgbm = np.full(tr_n, 0.5)

            # Model 4: ssGSEA (pre-computed)
            tr_ssgsea = tr_ds.get("_pred_ssgsea_raw", np.zeros(tr_n))
            if tr_ssgsea is None or np.all(np.isnan(tr_ssgsea)):
                tr_ssgsea = np.zeros(tr_n, dtype=np.float64)
            tr_ssgsea = np.nan_to_num(tr_ssgsea, nan=0.0)

            X_meta_tr = np.column_stack([
                tr_reversal, tr_rev_dt, tr_lgbm, tr_ssgsea
            ])
            X_meta_train_parts.append(X_meta_tr)
            y_meta_train_parts.append(tr_y)

        if X_meta_train_parts:
            X_meta_train = np.concatenate(X_meta_train_parts, axis=0)
            y_meta_train = np.concatenate(y_meta_train_parts, axis=0)

            # Scale meta features
            meta_scaler = StandardScaler()
            X_meta_train_s = meta_scaler.fit_transform(X_meta_train)
            X_meta_test_s = meta_scaler.transform(X_meta_test)
            X_meta_train_s = np.nan_to_num(X_meta_train_s, nan=0.0)
            X_meta_test_s = np.nan_to_num(X_meta_test_s, nan=0.0)

            # Train meta-learner
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                try:
                    meta_model = LogisticRegressionCV(
                        cv=min(3, max(2, int(np.unique(y_meta_train).shape[0]))),
                        max_iter=1000,
                        random_state=42,
                        solver="liblinear",
                    )
                    meta_model.fit(X_meta_train_s, y_meta_train)
                    pred_ensemble = meta_model.predict_proba(X_meta_test_s)[:, 1]

                    # Record weights
                    weights = meta_model.coef_[0]
                    all_meta_weights.append({
                        "held_out": held_out_geo,
                        "w_reversal": round(float(weights[0]), 4),
                        "w_rev_dt": round(float(weights[1]), 4),
                        "w_lgbm": round(float(weights[2]), 4),
                        "w_ssgsea": round(float(weights[3]), 4),
                        "intercept": round(float(meta_model.intercept_[0]), 4),
                        "C_best": round(float(meta_model.C_[0]), 6),
                    })
                except Exception as e:
                    logger.warning("  Meta-learner failed for %s: %s", held_out_geo, e)
                    pred_ensemble = np.full(n_test, 0.5)
        else:
            pred_ensemble = np.full(n_test, 0.5)

        try:
            auc_ensemble = roc_auc_score(y_test, pred_ensemble)
        except ValueError:
            auc_ensemble = 0.5
        row["auc_ensemble"] = round(auc_ensemble, 4)

        per_dataset_results.append(row)
        logger.info(
            "  LODO %s: rev=%.3f, rev_dt=%.3f, lgbm=%.3f, ssgsea=%.3f, ensemble=%.3f",
            held_out_geo,
            row["auc_reversal_uniform"],
            row["auc_reversal_plus_dt"],
            row["auc_lgbm_reversal"],
            row["auc_ssgsea_pathway"],
            row["auc_ensemble"],
        )

    per_dataset_df = pd.DataFrame(per_dataset_results)
    meta_weights_df = pd.DataFrame(all_meta_weights)

    return per_dataset_df, meta_weights_df


# ======================================================================
# 8. Main pipeline
# ======================================================================

def run_stacking_ensemble():
    """
    End-to-end stacking ensemble pipeline:
    1. Load data
    2. Generate base-model predictions for all datasets
    3. Run LODO evaluation of all models + meta-learner
    4. Save results
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.time()

    logger.info("=" * 60)
    logger.info("STEP 3: STACKING ENSEMBLE")
    logger.info("=" * 60)

    # ---------------------------------------------------------------- #
    # Load data                                                         #
    # ---------------------------------------------------------------- #
    logger.info("Loading LINCS signatures ...")
    lincs, gene_cols = _load_lincs_signatures()
    logger.info("  %d signatures, %d genes", len(lincs), len(gene_cols))

    logger.info("Loading CTR-DB datasets ...")
    datasets = _load_ctrdb_datasets()
    logger.info("  %d datasets", len(datasets))

    if len(datasets) < 2:
        logger.error("Not enough datasets for LODO evaluation")
        return

    # ---------------------------------------------------------------- #
    # Generate base-model features                                      #
    # ---------------------------------------------------------------- #
    logger.info("\nGenerating base-model features ...")
    _, datasets = generate_base_model_predictions(datasets, lincs, gene_cols)

    valid_count = sum(1 for ds in datasets if "_patient_z" in ds)
    logger.info("  %d / %d datasets with valid features", valid_count, len(datasets))

    # ---------------------------------------------------------------- #
    # LODO evaluation                                                   #
    # ---------------------------------------------------------------- #
    logger.info("\n" + "=" * 60)
    logger.info("LODO evaluation: all models + ensemble")
    logger.info("=" * 60)

    per_dataset_df, meta_weights_df = run_lodo_all_models(datasets)

    # ---------------------------------------------------------------- #
    # Results                                                           #
    # ---------------------------------------------------------------- #
    if per_dataset_df.empty:
        logger.warning("No LODO results generated")
        return

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    # Summary
    auc_cols = [c for c in per_dataset_df.columns if c.startswith("auc_")]
    summary = per_dataset_df[auc_cols].agg(["mean", "std", "median"]).round(4)
    logger.info("\nPer-model summary:")
    for col in auc_cols:
        model_name = col.replace("auc_", "")
        logger.info(
            "  %-25s: mean=%.4f, std=%.4f, median=%.4f",
            model_name,
            summary.loc["mean", col],
            summary.loc["std", col],
            summary.loc["median", col],
        )

    # Meta-learner weights
    if not meta_weights_df.empty:
        logger.info("\nMeta-learner weights (mean across folds):")
        weight_cols = [c for c in meta_weights_df.columns if c.startswith("w_")]
        for wc in weight_cols:
            logger.info("  %-20s: %.4f", wc, meta_weights_df[wc].mean())
        logger.info("  intercept           : %.4f", meta_weights_df["intercept"].mean())

    # ---------------------------------------------------------------- #
    # Save                                                              #
    # ---------------------------------------------------------------- #
    RESULTS.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS / "ensemble_comparison.tsv"
    per_dataset_df.to_csv(out_path, sep="\t", index=False)
    logger.info("\nPer-dataset results saved to %s", out_path)

    if not meta_weights_df.empty:
        weights_path = RESULTS / "ensemble_meta_weights.tsv"
        meta_weights_df.to_csv(weights_path, sep="\t", index=False)
        logger.info("Meta-learner weights saved to %s", weights_path)

    elapsed = time.time() - t_start
    logger.info("Total time: %.0fs", elapsed)

    return per_dataset_df


if __name__ == "__main__":
    run_stacking_ensemble()
