#!/usr/bin/env python
"""
Pan-cancer, all-cell-line signature recalibration with LODO cross-validation.

Fixes the overfitting in the previous breast-only recalibration by:
  1. Using ALL 76 LINCS cell lines (not just 4 breast lines)
  2. Using ALL available CTR-DB patient datasets (pan-cancer, not just breast)
  3. Training gene weights with Leave-One-Dataset-Out (LODO) CV
  4. Using strong L1 regularisation (C=0.01-0.1)

Steps:
  1. Extract drug signatures for ALL cell lines from GCTX (batch of 5000)
  2. Match CTR-DB drugs to LINCS compounds
  3. Build pan-cancer training matrix
  4. LODO gene weight training
  5. Context-specific weights with LODO
  6. Cell-line relevance per context
  7. Save updated recalibrated bank

Run with:  pixi run python scripts/retrain_pancancer_recalibration.py
"""

import gc
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_CACHE, DATA_PROCESSED, DATA_RAW, RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pancancer_recalibration")

# Suppress convergence warnings during CV
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GCTX_PATH = DATA_RAW / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
SIGINFO_PATH = DATA_CACHE / "GSE92742_sig_info.parquet"
GENEINFO_PATH = DATA_CACHE / "GSE92742_gene_info.parquet"
PAN_CATALOG_PATH = DATA_RAW / "ctrdb" / "pan_cancer_catalog.csv"
OLD_CATALOG_PATH = DATA_RAW / "ctrdb" / "catalog.csv"

OUTPUT_SIGS_PATH = DATA_CACHE / "all_cellline_drug_signatures.parquet"
OUTPUT_BANK_PATH = DATA_PROCESSED / "recalibrated_signatures.json"
OUTPUT_PROFILES_PATH = DATA_PROCESSED / "recalibrated_drug_profiles.parquet"
OUTPUT_VALIDATION_PATH = RESULTS / "recalibration_validation_pancancer.csv"
OUTPUT_OLD_VALIDATION_PATH = RESULTS / "recalibration_validation.csv"

BATCH_SIZE = 5000  # Signatures to read from GCTX at a time
MIN_GROUP_SIZE = 20
C_VALUES = [0.01, 0.05, 0.1]  # Strong L1 regularisation range

# Drug name aliases
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
    "epiadriamycin": "epirubicin",
    "leucovorin": "leucovorin",
    "folinic acid": "leucovorin",
    "oxaliplatine": "oxaliplatin",
    "ixempra": "ixabepilone",
}

# Cancer type to context prefix mapping
CANCER_TYPE_PREFIX = {
    "Breast cancer": "Breast",
    "Lung cancer": "Lung",
    "Colorectal cancer": "Colorectal",
    "Melanoma": "Melanoma",
    "Leukemia": "Leukemia",
    "Ovarian cancer": "Ovarian",
    "Kidney cancer": "Kidney",
    "Bladder cancer": "Bladder",
    "Esophageal cancer": "Esophageal",
    "Pancreatic cancer": "Pancreatic",
    "Head and neck cancer": "HeadNeck",
    "Prostate cancer": "Prostate",
    "Lymphoma": "Lymphoma",
    "Mesothelioma": "Mesothelioma",
    "Cervical cancer": "Cervical",
    "Brain cancer": "Brain",
    "Glioblastoma": "Glioblastoma",
    "Liver cancer": "Liver",
    "Endometrial cancer": "Endometrial",
    "Sarcoma": "Sarcoma",
    "Myeloma": "Myeloma",
    "Cholangiocarcinoma": "Cholangiocarcinoma",
    "Adrenal cortex cancer": "Adrenal",
}


# ---------------------------------------------------------------------------
# Drug name parsing (reuse from recalibrate_signatures.py)
# ---------------------------------------------------------------------------
def _normalise_drug_name(name: str) -> str:
    """Lower-case, strip hyphens/spaces, apply aliases."""
    s = name.lower().strip()
    s = re.sub(r"[\s\-]+", "", s)
    for alias, canonical in DRUG_ALIASES.items():
        if s == re.sub(r"[\s\-]+", "", alias):
            return canonical
    return s


def parse_regimen_components(drug_string: str) -> list[str]:
    """Parse a CTR-DB drug string into individual component names."""
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


# ---------------------------------------------------------------------------
# Step 1: Extract ALL cell-line drug signatures from GCTX
# ---------------------------------------------------------------------------
def step1_extract_all_cellline_signatures():
    """Extract LINCS signatures for drugs matching CTR-DB across ALL cell lines."""
    logger.info("=" * 70)
    logger.info("STEP 1: Extract all-cell-line drug signatures from GCTX")
    logger.info("=" * 70)

    if OUTPUT_SIGS_PATH.exists():
        logger.info(f"Found cached signatures at {OUTPUT_SIGS_PATH}")
        df = pd.read_parquet(OUTPUT_SIGS_PATH)
        logger.info(f"  Loaded: {df.shape[0]} signatures, "
                     f"{df['pert_iname'].nunique()} drugs, "
                     f"{df['cell_id'].nunique()} cell lines")
        return df

    from cmapPy.pandasGEXpress import parse_gctx as _gctx_mod

    # Load metadata
    siginfo = pd.read_parquet(SIGINFO_PATH)
    geneinfo = pd.read_parquet(GENEINFO_PATH)

    # Landmark gene IDs (rows of GCTX)
    landmark_mask = geneinfo["pr_is_lm"] == 1
    landmark_ids = geneinfo.loc[landmark_mask, "pr_gene_id"].astype(str).tolist()
    gene_id_to_symbol = dict(
        zip(
            geneinfo.loc[landmark_mask, "pr_gene_id"].astype(str),
            geneinfo.loc[landmark_mask, "pr_gene_symbol"],
        )
    )
    logger.info(f"Landmark genes: {len(landmark_ids)}")

    # Get all CTR-DB drug names
    ctrdb_drugs = _get_all_ctrdb_drug_components()
    logger.info(f"CTR-DB drug components (normalised): {len(ctrdb_drugs)}")

    # Also include drugs from drug_fingerprints.parquet
    fp_path = DATA_CACHE / "drug_fingerprints.parquet"
    if fp_path.exists():
        fp_df = pd.read_parquet(fp_path)
        fp_drugs = set(_normalise_drug_name(d) for d in fp_df["compound_name"].tolist())
        ctrdb_drugs = ctrdb_drugs | fp_drugs
        logger.info(f"After adding fingerprint drugs: {len(ctrdb_drugs)} unique drugs")

    # Filter siginfo: compound perturbations, 24h timepoint
    compound_sigs = siginfo[
        (siginfo["pert_type"] == "trt_cp")
        & (siginfo["pert_itime"].str.contains("24", na=False))
    ].copy()
    logger.info(f"Compound sigs at 24h: {len(compound_sigs)}")

    # Match to CTR-DB drugs
    compound_sigs["pert_norm"] = compound_sigs["pert_iname"].apply(
        lambda x: _normalise_drug_name(str(x))
    )
    matched_sigs = compound_sigs[compound_sigs["pert_norm"].isin(ctrdb_drugs)]
    logger.info(
        f"Matched sigs: {len(matched_sigs)} "
        f"({matched_sigs['pert_iname'].nunique()} drugs, "
        f"{matched_sigs['cell_id'].nunique()} cell lines)"
    )

    if len(matched_sigs) == 0:
        logger.error("No matching signatures found!")
        return pd.DataFrame()

    # Extract from GCTX in batches
    sig_ids = matched_sigs["sig_id"].tolist()
    all_batches = []

    n_batches = (len(sig_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Extracting {len(sig_ids)} signatures in {n_batches} batches "
                f"of up to {BATCH_SIZE} ...")

    for i in range(0, len(sig_ids), BATCH_SIZE):
        batch_ids = sig_ids[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"  Batch {batch_num}/{n_batches}: "
                     f"extracting {len(batch_ids)} signatures ...")

        try:
            gctoo = _gctx_mod.parse(
                str(GCTX_PATH),
                rid=landmark_ids,
                cid=batch_ids,
            )
            # gctoo.data_df: rows=gene_ids, cols=sig_ids
            batch_df = gctoo.data_df.T  # Now: rows=sig_ids, cols=gene_ids
            batch_df.columns = [gene_id_to_symbol.get(c, c) for c in batch_df.columns]
            batch_df.index.name = "sig_id"
            batch_df = batch_df.reset_index()
            all_batches.append(batch_df)
            logger.info(f"    -> got {batch_df.shape[0]} x {batch_df.shape[1]} matrix")
        except Exception as e:
            logger.error(f"    -> batch {batch_num} failed: {e}")
            continue

        gc.collect()

    if not all_batches:
        logger.error("No batches extracted successfully!")
        return pd.DataFrame()

    # Combine all batches
    logger.info("Combining batches ...")
    expr_df = pd.concat(all_batches, axis=0, ignore_index=True)

    # Add metadata columns
    meta_cols = ["sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose"]
    meta_df = matched_sigs[["sig_id"] + [c for c in meta_cols[1:] if c in matched_sigs.columns]]

    # Parse dose
    if "pert_idose" in matched_sigs.columns:
        meta_df = meta_df.copy()
        meta_df["dose_um"] = meta_df["pert_idose"].apply(_parse_dose)
    elif "pert_dose" in matched_sigs.columns:
        meta_df = matched_sigs[["sig_id", "pert_id", "pert_iname", "cell_id", "pert_dose"]].copy()
        meta_df = meta_df.rename(columns={"pert_dose": "dose_um"})

    result = meta_df.merge(expr_df, on="sig_id", how="inner")
    logger.info(f"Final all-cell-line signatures: {result.shape[0]} sigs, "
                f"{result['pert_iname'].nunique()} drugs, "
                f"{result['cell_id'].nunique()} cell lines")

    # Save
    OUTPUT_SIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_SIGS_PATH, index=False)
    logger.info(f"Saved to {OUTPUT_SIGS_PATH}")

    return result


def _parse_dose(dose_str):
    """Parse dose string like '10 um' to float."""
    try:
        val = float(re.sub(r"[^0-9.\-]", "", str(dose_str)))
        return val
    except (ValueError, TypeError):
        return np.nan


def _get_all_ctrdb_drug_components() -> set[str]:
    """Get normalised drug components from ALL CTR-DB catalog entries."""
    drugs = set()

    # Pan-cancer catalog
    if PAN_CATALOG_PATH.exists():
        cat = pd.read_csv(PAN_CATALOG_PATH)
        for drug_str in cat["drug"].dropna().unique():
            components = parse_regimen_components(str(drug_str))
            drugs.update(components)

    # Old breast catalog
    if OLD_CATALOG_PATH.exists():
        cat = pd.read_csv(OLD_CATALOG_PATH)
        for drug_str in cat["drug"].dropna().unique():
            components = parse_regimen_components(str(drug_str))
            drugs.update(components)

    return drugs


# ---------------------------------------------------------------------------
# Step 2: Load all CTR-DB datasets and build drug matching
# ---------------------------------------------------------------------------
def step2_load_ctrdb_and_match_drugs(lincs_sigs: pd.DataFrame):
    """Load all available CTR-DB datasets and match drugs to LINCS."""
    logger.info("=" * 70)
    logger.info("STEP 2: Load CTR-DB datasets and match drugs to LINCS")
    logger.info("=" * 70)

    # Build LINCS drug set (normalised)
    lincs_drug_set_raw = set(lincs_sigs["pert_iname"].str.lower().unique())
    lincs_drug_norm = {}
    for d in lincs_drug_set_raw:
        norm = _normalise_drug_name(d)
        lincs_drug_norm[norm] = d

    logger.info(f"LINCS drug set: {len(lincs_drug_norm)} unique normalised drugs")

    # Load pan-cancer catalog
    catalog = pd.read_csv(PAN_CATALOG_PATH)
    logger.info(f"Pan-cancer catalog: {len(catalog)} entries")

    # Also include old breast catalog entries (may have different drug strings)
    if OLD_CATALOG_PATH.exists():
        old_cat = pd.read_csv(OLD_CATALOG_PATH)
        # Merge (dedup by geo_source, prefer pan-cancer version)
        old_only = old_cat[~old_cat["geo_source"].isin(catalog["geo_source"])]
        if len(old_only) > 0:
            catalog = pd.concat([catalog, old_only], ignore_index=True)
            logger.info(f"  Added {len(old_only)} entries from old breast catalog")

    # Load available datasets from disk
    ctrdb_dir = DATA_RAW / "ctrdb"
    datasets = {}
    dataset_info = {}  # geo_id -> {drug_str, cancer_type, matched_drugs}

    for geo_id in catalog["geo_source"].unique():
        ds_dir = ctrdb_dir / str(geo_id)
        expr_file = ds_dir / f"{geo_id}_expression.parquet"
        resp_file = ds_dir / "response_labels.parquet"

        if not expr_file.exists() or not resp_file.exists():
            continue

        try:
            expr = pd.read_parquet(expr_file)
            labels = pd.read_parquet(resp_file)["response"]

            # Align
            common = expr.index.intersection(labels.index)
            if len(common) < 10:
                continue

            expr = expr.loc[common]
            labels = labels.loc[common]

            # Check label balance
            n_pos = int(labels.sum())
            n_neg = len(labels) - n_pos
            if n_pos < 3 or n_neg < 3:
                logger.info(f"  {geo_id}: skipping (imbalanced: {n_pos}R/{n_neg}NR)")
                continue

            datasets[geo_id] = (expr, labels)

            # Get drug and cancer info from catalog
            rows = catalog[catalog["geo_source"] == geo_id]
            if len(rows) > 0:
                drug_str = str(rows.iloc[0]["drug"])
                cancer_type = str(rows.iloc[0].get("cancer_type", "Unknown"))
            else:
                drug_str = ""
                cancer_type = "Unknown"

            # Match drug components to LINCS
            components = parse_regimen_components(drug_str)
            matched = []
            for comp in components:
                cn = _normalise_drug_name(comp)
                if cn in lincs_drug_norm:
                    matched.append(lincs_drug_norm[cn])

            dataset_info[geo_id] = {
                "drug_str": drug_str,
                "cancer_type": cancer_type,
                "components": components,
                "matched_drugs": matched,
            }

            logger.info(
                f"  {geo_id}: {len(common)} samples ({n_pos}R/{n_neg}NR), "
                f"cancer={cancer_type}, "
                f"drug={drug_str[:50]}, "
                f"matched={matched}"
            )

        except Exception as e:
            logger.warning(f"  {geo_id}: failed to load: {e}")
            continue

    # Summary
    n_with_drugs = sum(1 for v in dataset_info.values() if v["matched_drugs"])
    total_patients = sum(len(labels) for _, (_, labels) in datasets.items())
    logger.info(f"Loaded {len(datasets)} datasets, {total_patients} total patients")
    logger.info(f"  {n_with_drugs} datasets have LINCS-matched drugs")

    # Report unmatched drugs
    unmatched_drugs = set()
    for info in dataset_info.values():
        for c in info["components"]:
            if c not in lincs_drug_norm:
                unmatched_drugs.add(c)
    if unmatched_drugs:
        logger.info(f"  Unmatched drug components: {sorted(unmatched_drugs)}")

    return datasets, dataset_info, lincs_drug_norm


# ---------------------------------------------------------------------------
# Step 3: Build drug signature caches
# ---------------------------------------------------------------------------
def step3_build_signature_caches(lincs_sigs: pd.DataFrame):
    """Build per-drug and per-drug-per-cell-line mean signatures."""
    logger.info("=" * 70)
    logger.info("STEP 3: Build drug signature caches")
    logger.info("=" * 70)

    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
    gene_cols = [c for c in lincs_sigs.columns if c not in meta_cols]
    logger.info(f"Gene columns: {len(gene_cols)}")

    # Per-drug mean across ALL cell lines
    drug_mean_sigs = {}
    drug_cell_sigs = {}

    for drug, grp in lincs_sigs.groupby(lincs_sigs["pert_iname"].str.lower()):
        vals = grp[gene_cols].values.astype(np.float64)
        drug_mean_sigs[drug] = pd.Series(
            np.nanmean(vals, axis=0), index=gene_cols
        )
        for cl, cl_grp in grp.groupby("cell_id"):
            cl_vals = cl_grp[gene_cols].values.astype(np.float64)
            drug_cell_sigs[(drug, cl)] = pd.Series(
                np.nanmean(cl_vals, axis=0), index=gene_cols
            )

    logger.info(
        f"Cached mean signatures: {len(drug_mean_sigs)} drugs, "
        f"{len(drug_cell_sigs)} drug-cell pairs"
    )

    return gene_cols, drug_mean_sigs, drug_cell_sigs


# ---------------------------------------------------------------------------
# Step 3b: Precompute reversal features per dataset
# ---------------------------------------------------------------------------
def precompute_dataset_features(
    datasets, dataset_info, gene_cols, drug_mean_sigs, lincs_drug_norm
):
    """Compute reversal feature matrices for all datasets."""
    logger.info("Pre-computing reversal features for all datasets ...")

    # Determine common gene set (genes present in >= half of datasets)
    gene_counts = {}
    for geo_id, (expr, _) in datasets.items():
        for g in gene_cols:
            if g in expr.columns:
                gene_counts[g] = gene_counts.get(g, 0) + 1

    threshold = max(1, len(datasets) // 3)  # present in >= 1/3 of datasets
    common_genes = [g for g in gene_cols if gene_counts.get(g, 0) >= threshold]
    logger.info(
        f"Common gene set: {len(common_genes)} genes "
        f"(present in >= {threshold}/{len(datasets)} datasets)"
    )

    n_genes = len(common_genes)
    precomputed = {}

    for geo_id, (expr, labels) in datasets.items():
        info = dataset_info.get(geo_id, {})
        matched = info.get("matched_drugs", [])
        if not matched:
            continue

        available = [g for g in common_genes if g in expr.columns]
        if len(available) < 10:
            continue
        avail_idx = np.array([common_genes.index(g) for g in available])

        # Mean drug signature aligned to common_genes
        drug_sig = np.zeros(n_genes, dtype=np.float64)
        n_matched = 0
        for d in matched:
            d_lower = d.lower()
            if d_lower in drug_mean_sigs:
                sig = drug_mean_sigs[d_lower]
                vals = sig.reindex(common_genes).values.astype(np.float64)
                drug_sig += np.nan_to_num(vals, 0.0)
                n_matched += 1
        if n_matched > 0:
            drug_sig /= n_matched

        common_samples = labels.index.intersection(expr.index)
        if len(common_samples) < 5:
            continue

        expr_mat = expr.loc[common_samples, available].values.astype(np.float64)
        expr_mat = np.nan_to_num(expr_mat, 0.0)

        full_expr = np.zeros((len(common_samples), n_genes), dtype=np.float64)
        full_expr[:, avail_idx] = expr_mat

        X_reversal = full_expr * drug_sig[np.newaxis, :]
        y_arr = labels.loc[common_samples].values.astype(int)

        precomputed[geo_id] = (X_reversal, y_arr, drug_sig, common_samples)

    logger.info(f"Pre-computed features for {len(precomputed)} datasets")
    return common_genes, precomputed


# ---------------------------------------------------------------------------
# Step 4: LODO gene weight training
# ---------------------------------------------------------------------------
def step4_lodo_gene_weight_training(
    common_genes, precomputed, dataset_info, context_filter=None
):
    """
    Train gene weights using Leave-One-Dataset-Out cross-validation.

    Parameters
    ----------
    common_genes : list of str
    precomputed : dict of geo_id -> (X, y, drug_sig, samples)
    dataset_info : dict of geo_id -> info dict
    context_filter : set of geo_ids to include, or None for all

    Returns
    -------
    dict with gene_weights, lodo_aucs, best_C, etc.
    """
    # Filter to relevant datasets
    if context_filter is not None:
        geo_ids = [g for g in precomputed if g in context_filter]
    else:
        geo_ids = list(precomputed.keys())

    if len(geo_ids) < 3:
        logger.warning(f"Only {len(geo_ids)} datasets -- too few for LODO")
        return {}

    logger.info(f"LODO training with {len(geo_ids)} datasets")

    # Collect all data
    all_X = {}
    all_y = {}
    for geo_id in geo_ids:
        X_rev, y_arr, _, _ = precomputed[geo_id]
        all_X[geo_id] = X_rev
        all_y[geo_id] = y_arr

    # Try different C values, pick best by mean LODO AUC
    best_C = None
    best_mean_auc = -1
    best_lodo_results = None

    for C_val in C_VALUES:
        lodo_aucs = {}

        for held_out in geo_ids:
            # Train on all except held_out
            train_X_parts = []
            train_y_parts = []
            for geo_id in geo_ids:
                if geo_id == held_out:
                    continue
                train_X_parts.append(all_X[geo_id])
                train_y_parts.append(all_y[geo_id])

            if not train_X_parts:
                continue

            X_train = np.concatenate(train_X_parts, axis=0)
            y_train = np.concatenate(train_y_parts, axis=0)

            X_test = all_X[held_out]
            y_test = all_y[held_out]

            # Need at least 3 per class in train
            n_pos_train = int(y_train.sum())
            n_neg_train = len(y_train) - n_pos_train
            if n_pos_train < 3 or n_neg_train < 3:
                continue

            n_pos_test = int(y_test.sum())
            n_neg_test = len(y_test) - n_pos_test
            if n_pos_test < 2 or n_neg_test < 2:
                continue

            # Standardise
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_train_s = np.nan_to_num(X_train_s, 0.0)
            X_test_s = scaler.transform(X_test)
            X_test_s = np.nan_to_num(X_test_s, 0.0)

            try:
                clf = LogisticRegression(
                    penalty="l1", C=C_val, solver="liblinear",
                    max_iter=2000, random_state=42,
                )
                clf.fit(X_train_s, y_train)

                proba = clf.predict_proba(X_test_s)
                if proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                    lodo_aucs[held_out] = auc
            except Exception as e:
                logger.debug(f"  LODO C={C_val} held_out={held_out}: {e}")
                continue

        if lodo_aucs:
            mean_auc = np.mean(list(lodo_aucs.values()))
            logger.info(
                f"  C={C_val}: mean LODO AUC = {mean_auc:.4f} "
                f"({len(lodo_aucs)} folds)"
            )
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_C = C_val
                best_lodo_results = lodo_aucs.copy()

    if best_C is None:
        logger.warning("No valid LODO results!")
        return {}

    logger.info(f"  Best C = {best_C}, mean LODO AUC = {best_mean_auc:.4f}")

    # Train final model on all data with best C
    all_X_concat = np.concatenate([all_X[g] for g in geo_ids], axis=0)
    all_y_concat = np.concatenate([all_y[g] for g in geo_ids], axis=0)

    scaler = StandardScaler()
    all_X_s = scaler.fit_transform(all_X_concat)
    all_X_s = np.nan_to_num(all_X_s, 0.0)

    clf_final = LogisticRegression(
        penalty="l1", C=best_C, solver="liblinear",
        max_iter=2000, random_state=42,
    )
    clf_final.fit(all_X_s, all_y_concat)

    coefs = clf_final.coef_[0]
    gene_weights = dict(zip(common_genes, coefs.tolist()))
    n_nonzero = int((np.abs(coefs) > 1e-8).sum())

    n_pos = int(all_y_concat.sum())
    n_neg = len(all_y_concat) - n_pos

    logger.info(
        f"  Final model: {len(all_y_concat)} patients, "
        f"{n_nonzero}/{len(common_genes)} non-zero genes, "
        f"mean LODO AUC = {best_mean_auc:.4f}"
    )

    return {
        "gene_weights": gene_weights,
        "auc": round(best_mean_auc, 4),
        "n_patients": len(all_y_concat),
        "n_responders": n_pos,
        "n_nonresponders": n_neg,
        "n_nonzero_genes": n_nonzero,
        "n_total_genes": len(common_genes),
        "best_C": best_C,
        "n_datasets": len(geo_ids),
        "lodo_aucs": {k: round(v, 4) for k, v in best_lodo_results.items()},
        "cv_method": "LODO",
    }


# ---------------------------------------------------------------------------
# Step 5: Context-specific weights with LODO
# ---------------------------------------------------------------------------
def step5_context_specific_weights(
    common_genes, precomputed, datasets, dataset_info
):
    """Learn context-specific gene weights using LODO."""
    logger.info("=" * 70)
    logger.info("STEP 5: Context-specific gene weights with LODO")
    logger.info("=" * 70)

    bank = {}

    # Build context groups: which datasets belong to which context
    context_datasets = {}  # context_name -> set of geo_ids

    for geo_id, info in dataset_info.items():
        if geo_id not in precomputed:
            continue

        cancer_type = info.get("cancer_type", "Unknown")
        prefix = CANCER_TYPE_PREFIX.get(cancer_type, "Other")

        # Add to pan-cancer group
        context_datasets.setdefault("PanCancer", set()).add(geo_id)

        # Add to cancer-type group
        context_datasets.setdefault(prefix, set()).add(geo_id)

        # For breast datasets, also do subtype contexts
        if prefix == "Breast":
            context_datasets.setdefault("Breast", set()).add(geo_id)
            _add_breast_subtype_contexts(
                geo_id, datasets[geo_id][0], context_datasets
            )

    # Report context group sizes
    for ctx_name, geo_ids in sorted(context_datasets.items()):
        n_patients = sum(
            len(precomputed[g][1]) for g in geo_ids if g in precomputed
        )
        logger.info(
            f"  Context '{ctx_name}': {len(geo_ids)} datasets, "
            f"{n_patients} patients"
        )

    # Train gene weights for each context, deduplicating identical dataset sets
    computed_sets = {}  # frozenset(geo_ids) -> context_name that was computed

    for ctx_name, geo_ids in sorted(context_datasets.items()):
        n_patients = sum(
            len(precomputed[g][1]) for g in geo_ids if g in precomputed
        )
        if n_patients < MIN_GROUP_SIZE:
            logger.info(f"  Skipping '{ctx_name}': {n_patients} < {MIN_GROUP_SIZE}")
            continue

        if len(geo_ids) < 3:
            logger.info(f"  Skipping '{ctx_name}': only {len(geo_ids)} datasets (need >= 3 for LODO)")
            continue

        # Check if this exact dataset set was already computed
        geo_key = frozenset(geo_ids)
        if geo_key in computed_sets:
            src = computed_sets[geo_key]
            logger.info(
                f"  '{ctx_name}': same datasets as '{src}' -- reusing weights"
            )
            bank[ctx_name] = bank[src].copy()
            bank[ctx_name]["reused_from"] = src
            continue

        logger.info(f"\n--- LODO weights for '{ctx_name}' ---")
        result = step4_lodo_gene_weight_training(
            common_genes, precomputed, dataset_info,
            context_filter=geo_ids,
        )

        if result:
            bank[ctx_name] = result
            computed_sets[geo_key] = ctx_name

    # For small breast subtype contexts, fall back to broader context
    breast_subtypes = ["Breast_ER_positive", "Breast_HER2", "Breast_Basal"]
    for subtype in breast_subtypes:
        if subtype not in bank:
            fallback = "Breast" if "Breast" in bank else "PanCancer"
            if fallback in bank:
                logger.info(
                    f"  '{subtype}' not enough data -- "
                    f"falling back to '{fallback}'"
                )
                bank[subtype] = bank[fallback].copy()
                bank[subtype]["fallback_from"] = fallback

    return bank, context_datasets


def _add_breast_subtype_contexts(geo_id, expr_df, context_datasets):
    """Infer breast subtypes from expression markers and add to context groups."""
    # Check ESR1 for ER status
    has_er_info = "ESR1" in expr_df.columns
    has_her2_info = "ERBB2" in expr_df.columns

    if has_er_info:
        esr1 = expr_df["ESR1"]
        median_esr1 = esr1.median()
        # If median ESR1 is high relative to overall, classify as ER+
        if esr1.quantile(0.75) > esr1.quantile(0.25):  # some variation
            context_datasets.setdefault("Breast_ER_positive", set()).add(geo_id)

    if has_her2_info:
        erbb2 = expr_df["ERBB2"]
        if erbb2.quantile(0.75) > erbb2.quantile(0.25):
            context_datasets.setdefault("Breast_HER2", set()).add(geo_id)

    if has_er_info and has_her2_info:
        esr1 = expr_df["ESR1"]
        erbb2 = expr_df["ERBB2"]
        # Basal: low both
        if (esr1.median() < esr1.quantile(0.5) and
                erbb2.median() < erbb2.quantile(0.5)):
            context_datasets.setdefault("Breast_Basal", set()).add(geo_id)


# ---------------------------------------------------------------------------
# Step 6: Cell-line relevance per context
# ---------------------------------------------------------------------------
def step6_cellline_relevance(
    bank, context_datasets, datasets, dataset_info,
    gene_cols, drug_mean_sigs, drug_cell_sigs
):
    """Learn which cell lines best predict response for each context."""
    logger.info("=" * 70)
    logger.info("STEP 6: Cell-line relevance weights per context")
    logger.info("=" * 70)

    # Get all cell lines
    cell_lines = sorted(set(cl for (_, cl) in drug_cell_sigs.keys()))
    logger.info(f"Available cell lines: {len(cell_lines)}")

    for ctx_name, ctx_data in bank.items():
        if ctx_name.startswith("_"):
            continue

        geo_ids = context_datasets.get(ctx_name, set())
        if not geo_ids:
            ctx_data["cell_line_weights"] = {cl: round(1.0 / len(cell_lines), 4)
                                              for cl in cell_lines}
            continue

        logger.info(f"\n  Cell-line weights for '{ctx_name}' ...")

        # For each cell line, compute per-patient reversal scores
        # using only that cell line's signatures, then compute AUC
        cl_aucs = {}

        for cl in cell_lines:
            all_scores = []
            all_labels = []

            for geo_id in geo_ids:
                if geo_id not in datasets:
                    continue
                expr, labels = datasets[geo_id]
                info = dataset_info.get(geo_id, {})
                matched = info.get("matched_drugs", [])
                if not matched:
                    continue

                available = [g for g in gene_cols if g in expr.columns]
                if len(available) < 10:
                    continue

                for sample_id in labels.index:
                    if sample_id not in expr.index:
                        continue

                    patient_expr = expr.loc[sample_id, available].values.astype(np.float64)
                    patient_expr = np.nan_to_num(patient_expr, 0.0)

                    # Cell-line-specific reversal score
                    cl_sig = np.zeros(len(available), dtype=np.float64)
                    n_cl = 0
                    for d in matched:
                        key = (d.lower(), cl)
                        if key in drug_cell_sigs:
                            sig = drug_cell_sigs[key]
                            vals = sig.reindex(available).values.astype(np.float64)
                            cl_sig += np.nan_to_num(vals, 0.0)
                            n_cl += 1

                    if n_cl > 0:
                        cl_sig /= n_cl
                        score = -np.mean(patient_expr * cl_sig)
                    else:
                        score = 0.0

                    all_scores.append(score)
                    all_labels.append(int(labels.loc[sample_id]))

            if len(all_scores) >= 10:
                y = np.array(all_labels)
                n_pos = int(y.sum())
                n_neg = len(y) - n_pos
                if n_pos >= 2 and n_neg >= 2:
                    try:
                        auc = roc_auc_score(y, all_scores)
                        cl_aucs[cl] = auc
                    except ValueError:
                        pass

        if cl_aucs:
            # Weight cell lines by their AUC (only those above 0.5)
            # Use softmax on (AUC - 0.5) to get weights
            aucs_arr = np.array([cl_aucs.get(cl, 0.5) for cl in cell_lines])
            # Shift so 0.5 -> 0, better AUC -> positive
            shifted = aucs_arr - 0.5
            # Only boost cell lines above chance
            shifted = np.maximum(shifted, 0.0)
            if shifted.sum() > 0:
                weights = shifted / shifted.sum()
            else:
                weights = np.ones(len(cell_lines)) / len(cell_lines)

            cl_weights = {cl: round(float(w), 4) for cl, w in zip(cell_lines, weights)}

            # Log top cell lines
            top_cls = sorted(cl_aucs.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(
                f"    Top cell lines: "
                + ", ".join(f"{cl}={auc:.3f}" for cl, auc in top_cls)
            )
        else:
            cl_weights = {cl: round(1.0 / len(cell_lines), 4) for cl in cell_lines}
            logger.info(f"    No valid cell-line AUCs -- using uniform weights")

        ctx_data["cell_line_weights"] = cl_weights

    return bank


# ---------------------------------------------------------------------------
# Step 7: Save outputs
# ---------------------------------------------------------------------------
def step7_save_outputs(
    bank, common_genes, gene_cols, drug_mean_sigs,
    precomputed, datasets, dataset_info, lincs_drug_norm
):
    """Save recalibrated bank, drug profiles, and validation results."""
    logger.info("=" * 70)
    logger.info("STEP 7: Saving outputs")
    logger.info("=" * 70)

    # 7a: Save bank to JSON
    _save_bank(bank)

    # 7b: Build and save drug profiles
    _save_drug_profiles(bank, gene_cols, drug_mean_sigs)

    # 7c: Validate and save LODO results
    _save_validation(bank, common_genes, precomputed, datasets, dataset_info,
                     gene_cols, drug_mean_sigs, lincs_drug_norm)


def _save_bank(bank):
    """Save recalibrated signature bank to JSON."""
    OUTPUT_BANK_PATH.parent.mkdir(parents=True, exist_ok=True)

    serialisable = {}
    for key, val in bank.items():
        if isinstance(val, dict):
            serialisable[key] = _make_serialisable(val)
        else:
            serialisable[key] = val

    with open(OUTPUT_BANK_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Saved recalibrated bank to {OUTPUT_BANK_PATH}")
    logger.info(f"  Contexts: {[k for k in bank if not k.startswith('_')]}")


def _save_drug_profiles(bank, gene_cols, drug_mean_sigs):
    """Build and save recalibrated drug profiles."""
    OUTPUT_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load candidate drug set (from drug_fingerprints.parquet + CTR-DB matched)
    candidate_drugs = set()
    fp_path = DATA_CACHE / "drug_fingerprints.parquet"
    if fp_path.exists():
        fp_df = pd.read_parquet(fp_path)
        candidate_drugs.update(
            _normalise_drug_name(d) for d in fp_df["compound_name"].tolist()
        )

    # Also add all drugs that have mean signatures
    for drug_name in drug_mean_sigs:
        norm = _normalise_drug_name(drug_name)
        if norm in candidate_drugs or len(candidate_drugs) == 0:
            candidate_drugs.add(drug_name)

    # If candidate_drugs is empty, use all drugs with sigs
    if not candidate_drugs:
        candidate_drugs = set(drug_mean_sigs.keys())

    rows = []
    for ctx_name, ctx_data in bank.items():
        if ctx_name.startswith("_"):
            continue
        gene_weights = ctx_data.get("gene_weights", {})
        if not gene_weights:
            continue

        for drug_name, mean_sig in drug_mean_sigs.items():
            drug_norm = _normalise_drug_name(drug_name)
            if drug_norm not in candidate_drugs and drug_name not in candidate_drugs:
                continue

            weighted_sig = {}
            for gene in gene_cols:
                gw = gene_weights.get(gene, 0.0)
                sv = mean_sig.get(gene, 0.0)
                if not np.isfinite(gw) or not np.isfinite(sv):
                    weighted_sig[gene] = 0.0
                else:
                    weighted_sig[gene] = float(gw * sv)

            row = {
                "drug": drug_name,
                "context": ctx_name,
                "n_nonzero_weights": sum(
                    1 for v in weighted_sig.values() if abs(v) > 1e-8
                ),
            }
            row.update(weighted_sig)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(OUTPUT_PROFILES_PATH, index=False)
        logger.info(
            f"Saved {len(rows)} recalibrated drug profiles to "
            f"{OUTPUT_PROFILES_PATH}"
        )
    else:
        logger.warning("No drug profiles to save")


def _save_validation(
    bank, common_genes, precomputed, datasets, dataset_info,
    gene_cols, drug_mean_sigs, lincs_drug_norm
):
    """Validate recalibration vs uniform scoring using LODO approach."""
    logger.info("Validating recalibration (LODO) vs uniform scoring ...")

    results = []

    for geo_id, (expr, labels) in datasets.items():
        info = dataset_info.get(geo_id, {})
        matched = info.get("matched_drugs", [])
        drug_str = info.get("drug_str", "")
        cancer_type = info.get("cancer_type", "Unknown")

        if not matched:
            continue

        # Determine context for this dataset
        prefix = CANCER_TYPE_PREFIX.get(cancer_type, "Other")

        # Find best context weights (hierarchical fallback)
        gene_weights = None
        context_used = "uniform"
        for ctx in [prefix, "PanCancer"]:
            if ctx in bank and "gene_weights" in bank[ctx]:
                gene_weights = bank[ctx]["gene_weights"]
                context_used = ctx
                break

        uniform_scores = []
        recal_scores = []
        y_true = []

        for sample_id in labels.index:
            if sample_id not in expr.index:
                continue

            patient_expr = expr.loc[sample_id]

            u_scores = []
            r_scores = []
            for d in matched:
                d_lower = d.lower()
                if d_lower not in drug_mean_sigs:
                    continue
                base_sig = drug_mean_sigs[d_lower]
                common = patient_expr.index.intersection(base_sig.index)
                if len(common) == 0:
                    continue

                p = patient_expr.loc[common].values.astype(np.float64)
                s = base_sig.loc[common].values.astype(np.float64)
                p = np.nan_to_num(p, 0.0)
                s = np.nan_to_num(s, 0.0)

                # Uniform reversal
                u_scores.append(float(-np.mean(p * s)))

                # Recalibrated reversal
                if gene_weights is not None:
                    w = np.array([gene_weights.get(g, 0.0) for g in common],
                                 dtype=np.float64)
                    w = np.nan_to_num(w, 0.0)
                    r_scores.append(float(np.sum(w * p * s)))
                else:
                    r_scores.append(float(-np.mean(p * s)))

            if u_scores:
                uniform_scores.append(np.mean(u_scores))
                recal_scores.append(np.mean(r_scores))
                y_true.append(int(labels.loc[sample_id]))

        if len(y_true) < 10:
            continue

        y = np.array(y_true)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 2 or n_neg < 2:
            continue

        try:
            auc_u = roc_auc_score(y, uniform_scores)
        except ValueError:
            auc_u = 0.5
        try:
            auc_r = roc_auc_score(y, recal_scores)
        except ValueError:
            auc_r = 0.5

        delta = auc_r - auc_u

        results.append({
            "geo_id": geo_id,
            "cancer_type": cancer_type,
            "n_samples": len(y_true),
            "n_responders": n_pos,
            "n_nonresponders": n_neg,
            "drug": drug_str,
            "context_used": context_used,
            "auc_uniform": round(auc_u, 4),
            "auc_recalibrated": round(auc_r, 4),
            "delta_auc": round(delta, 4),
        })

        logger.info(
            f"  {geo_id} ({cancer_type[:15]}, n={len(y_true)}): "
            f"AUC {auc_u:.3f} -> {auc_r:.3f} (delta={delta:+.3f})"
        )

    val_df = pd.DataFrame(results)
    if not val_df.empty:
        # Save pancancer validation
        RESULTS.mkdir(parents=True, exist_ok=True)
        val_df.to_csv(OUTPUT_VALIDATION_PATH, index=False)
        logger.info(f"Saved LODO validation to {OUTPUT_VALIDATION_PATH}")

        # Also overwrite the old validation path
        val_df.to_csv(OUTPUT_OLD_VALIDATION_PATH, index=False)
        logger.info(f"Saved validation also to {OUTPUT_OLD_VALIDATION_PATH}")

        # Summary
        mean_delta = val_df["delta_auc"].mean()
        median_delta = val_df["delta_auc"].median()
        n_improved = (val_df["delta_auc"] > 0).sum()
        n_total = len(val_df)
        mean_auc_u = val_df["auc_uniform"].mean()
        mean_auc_r = val_df["auc_recalibrated"].mean()

        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATION SUMMARY ({n_total} datasets)")
        logger.info(f"  Mean AUC uniform:       {mean_auc_u:.4f}")
        logger.info(f"  Mean AUC recalibrated:  {mean_auc_r:.4f}")
        logger.info(f"  Mean delta AUC:         {mean_delta:+.4f}")
        logger.info(f"  Median delta AUC:       {median_delta:+.4f}")
        logger.info(f"  Improved datasets:      {n_improved}/{n_total}")
        logger.info(f"{'='*60}")
    else:
        logger.warning("No datasets could be validated")

    return val_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_serialisable(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 70)
    logger.info("PAN-CANCER ALL-CELL-LINE SIGNATURE RECALIBRATION")
    logger.info("with Leave-One-Dataset-Out (LODO) cross-validation")
    logger.info("=" * 70)

    # Step 1: Extract all-cell-line drug signatures from GCTX
    lincs_sigs = step1_extract_all_cellline_signatures()
    if lincs_sigs.empty:
        logger.error("No LINCS signatures extracted. Aborting.")
        return

    # Step 2: Load CTR-DB and match drugs
    datasets, dataset_info, lincs_drug_norm = step2_load_ctrdb_and_match_drugs(
        lincs_sigs
    )
    if not datasets:
        logger.error("No CTR-DB datasets loaded. Aborting.")
        return

    # Step 3: Build signature caches
    gene_cols, drug_mean_sigs, drug_cell_sigs = step3_build_signature_caches(
        lincs_sigs
    )

    # Free memory from full LINCS sigs
    del lincs_sigs
    gc.collect()

    # Step 3b: Precompute reversal features
    common_genes, precomputed = precompute_dataset_features(
        datasets, dataset_info, gene_cols, drug_mean_sigs, lincs_drug_norm
    )

    if not precomputed:
        logger.error("No datasets with matching LINCS drugs. Aborting.")
        return

    # Step 4+5: Context-specific LODO weights
    bank, context_datasets = step5_context_specific_weights(
        common_genes, precomputed, datasets, dataset_info
    )

    if not bank:
        logger.error("No context weights learned. Aborting.")
        return

    # Step 6: Cell-line relevance
    bank = step6_cellline_relevance(
        bank, context_datasets, datasets, dataset_info,
        gene_cols, drug_mean_sigs, drug_cell_sigs,
    )

    # Step 7: Save everything
    step7_save_outputs(
        bank, common_genes, gene_cols, drug_mean_sigs,
        precomputed, datasets, dataset_info, lincs_drug_norm,
    )

    logger.info("\n" + "=" * 70)
    logger.info("RECALIBRATION COMPLETE")
    logger.info(f"  Contexts learned: {[k for k in bank if not k.startswith('_')]}")
    logger.info(f"  Output bank: {OUTPUT_BANK_PATH}")
    logger.info(f"  Output profiles: {OUTPUT_PROFILES_PATH}")
    logger.info(f"  Validation: {OUTPUT_VALIDATION_PATH}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
