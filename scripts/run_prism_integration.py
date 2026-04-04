#!/usr/bin/env python
"""
PRISM Integration Pipeline for INVEREX.

Downloads PRISM secondary screen data, analyzes drug coverage,
builds an expanded training matrix (GDSC2 + PRISM), retrains
the LightGBM model, and validates on CTR-DB patient datasets.

Usage:
    pixi run python scripts/run_prism_integration.py
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_CACHE, DATA_RAW, DATA_PROCESSED, RESULTS, ECFP_NBITS
from src.data_ingestion.prism import (
    download_prism_data,
    download_depmap_cell_info,
    load_prism_treatment_info,
    load_prism_dose_response,
    load_prism_viability,
    load_depmap_cell_info,
    identify_breast_cell_lines,
    filter_breast_prism,
    build_prism_training_matrix,
    build_prism_viability_training,
    compare_prism_gdsc2_overlap,
    build_drug_coverage_table,
    _normalize_drug,
    _normalize_cell,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("prism_integration")


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PRISM Integration Pipeline — Starting")
    logger.info("=" * 70)

    # ── Step 1: Download PRISM data ─────────────────────────────────
    logger.info("\n[Step 1] Downloading PRISM data...")
    try:
        prism_files = download_prism_data()
        logger.info("PRISM data downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download PRISM data: {e}")
        logger.info("Attempting to proceed with any existing files...")

    # Download DepMap cell line info
    try:
        download_depmap_cell_info()
    except Exception as e:
        logger.warning(f"Could not download DepMap cell info: {e}")

    # ── Step 2: Load and analyze PRISM data ─────────────────────────
    logger.info("\n[Step 2] Loading and analyzing PRISM data...")

    try:
        treatment_info = load_prism_treatment_info()
    except FileNotFoundError:
        logger.error("PRISM treatment info not available. Cannot proceed.")
        return

    try:
        dose_response = load_prism_dose_response()
    except FileNotFoundError:
        logger.warning("PRISM dose-response not available.")
        dose_response = pd.DataFrame()

    # Log basic stats
    drug_col = "name" if "name" in treatment_info.columns else treatment_info.columns[0]
    n_total_drugs = treatment_info[drug_col].nunique()
    logger.info(f"\nPRISM Treatment Info:")
    logger.info(f"  Columns: {list(treatment_info.columns)}")
    logger.info(f"  Total unique drugs: {n_total_drugs}")

    if "moa" in treatment_info.columns:
        n_with_moa = treatment_info["moa"].notna().sum()
        logger.info(f"  Drugs with MOA annotation: {n_with_moa}")
        logger.info(f"  Top MOAs: {treatment_info['moa'].value_counts().head(10).to_dict()}")

    if "target" in treatment_info.columns:
        n_with_target = treatment_info["target"].notna().sum()
        logger.info(f"  Drugs with target annotation: {n_with_target}")

    if "phase" in treatment_info.columns:
        logger.info(f"  Clinical phase distribution: {treatment_info['phase'].value_counts().to_dict()}")

    if len(dose_response) > 0:
        logger.info(f"\nPRISM Dose-Response:")
        logger.info(f"  Columns: {list(dose_response.columns)}")
        logger.info(f"  Total records: {len(dose_response)}")
        if "depmap_id" in dose_response.columns:
            logger.info(f"  Unique cell lines: {dose_response['depmap_id'].nunique()}")
        if "name" in dose_response.columns:
            logger.info(f"  Unique drugs: {dose_response['name'].nunique()}")

    # ── Step 3: Identify breast cancer cell lines ───────────────────
    logger.info("\n[Step 3] Identifying breast cancer cell lines...")
    try:
        cell_info = load_depmap_cell_info()
        breast_cells = identify_breast_cell_lines(cell_info)
        breast_depmap_ids = set()
        for col in ["DepMap_ID", "depmap_id"]:
            if col in breast_cells.columns:
                breast_depmap_ids = set(breast_cells[col].dropna().unique())
                break
        logger.info(f"  Breast cancer DepMap IDs: {len(breast_depmap_ids)}")

        # Log cell line names
        for col in ["stripped_cell_line_name", "cell_line_name"]:
            if col in breast_cells.columns:
                names = sorted(breast_cells[col].unique()[:30])
                logger.info(f"  Breast cell lines: {names}")
                break
    except Exception as e:
        logger.warning(f"Could not load DepMap cell info: {e}")
        breast_depmap_ids = set()
        cell_info = pd.DataFrame()
        breast_cells = pd.DataFrame()

    # ── Step 4: Filter PRISM to breast cancer ───────────────────────
    logger.info("\n[Step 4] Filtering PRISM to breast cancer...")
    breast_prism_dr = pd.DataFrame()
    if len(dose_response) > 0 and breast_depmap_ids:
        breast_prism_dr = filter_breast_prism(dose_response, breast_depmap_ids)
        logger.info(f"  Breast PRISM dose-response records: {len(breast_prism_dr)}")
        if "name" in breast_prism_dr.columns:
            logger.info(f"  Breast PRISM unique drugs: {breast_prism_dr['name'].nunique()}")
    else:
        logger.warning("  Cannot filter PRISM - missing dose-response or cell line data")

    # ── Step 5: Load LINCS and GDSC2 for comparison ─────────────────
    logger.info("\n[Step 5] Loading existing pipeline data for comparison...")

    # LINCS signatures
    lincs_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    if lincs_path.exists():
        lincs_sigs = pd.read_parquet(lincs_path)
        logger.info(f"  LINCS breast sigs: {len(lincs_sigs)} ({lincs_sigs['pert_iname'].nunique()} drugs)")
    else:
        logger.warning("  LINCS breast signatures not found. Using all cell line signatures.")
        alt_path = DATA_CACHE / "all_cellline_drug_signatures.parquet"
        if alt_path.exists():
            lincs_sigs = pd.read_parquet(alt_path)
            logger.info(f"  All cellline sigs: {len(lincs_sigs)} ({lincs_sigs['pert_iname'].nunique()} drugs)")
        else:
            logger.error("  No LINCS signatures available.")
            lincs_sigs = pd.DataFrame()

    # GDSC2 dose-response
    gdsc_path = DATA_CACHE / "gdsc2_dose_response.parquet"
    if gdsc_path.exists():
        gdsc_dr = pd.read_parquet(gdsc_path)
        logger.info(f"  GDSC2 dose-response: {len(gdsc_dr)} records ({gdsc_dr['DRUG_NAME'].nunique()} drugs)")
    else:
        logger.warning("  GDSC2 dose-response not found.")
        gdsc_dr = pd.DataFrame()

    # Drug fingerprints
    fp_path = DATA_CACHE / "drug_fingerprints.parquet"
    if fp_path.exists():
        drug_fps = pd.read_parquet(fp_path)
        logger.info(f"  Drug fingerprints: {len(drug_fps)} compounds")
    else:
        logger.warning("  Drug fingerprints not found.")
        drug_fps = pd.DataFrame()

    # Existing training matrix
    tm_path = DATA_PROCESSED / "training_matrix.parquet"
    tt_path = DATA_PROCESSED / "training_target.parquet"
    if tm_path.exists() and tt_path.exists():
        existing_X = pd.read_parquet(tm_path)
        existing_y = pd.read_parquet(tt_path).squeeze()
        logger.info(f"  Existing training matrix: {existing_X.shape[0]} samples x {existing_X.shape[1]} features")
    else:
        logger.warning("  Existing training matrix not found.")
        existing_X = pd.DataFrame()
        existing_y = pd.Series(dtype=float)

    # ── Step 6: Drug coverage comparison ────────────────────────────
    logger.info("\n[Step 6] Comparing PRISM vs GDSC2 drug coverage...")
    if len(treatment_info) > 0 and len(lincs_sigs) > 0:
        overlap_summary = compare_prism_gdsc2_overlap(
            treatment_info, gdsc_dr, lincs_sigs
        )
        overlap_summary.to_csv(
            RESULTS / "prism_integration_analysis.csv", index=False
        )
        logger.info(f"  Saved to {RESULTS / 'prism_integration_analysis.csv'}")
    else:
        overlap_summary = pd.DataFrame()

    # ── Step 7: Build PRISM training data ───────────────────────────
    logger.info("\n[Step 7] Building PRISM training matrix...")
    prism_X = pd.DataFrame()
    prism_y = pd.Series(dtype=float)

    if len(breast_prism_dr) > 0 and len(lincs_sigs) > 0 and len(drug_fps) > 0:
        try:
            prism_X, prism_y = build_prism_training_matrix(
                breast_prism_dr, lincs_sigs, drug_fps, cell_info,
                cache_dir=DATA_CACHE,
            )
            if len(prism_X) > 0:
                logger.info(f"  PRISM training matrix: {prism_X.shape[0]} samples")
            else:
                logger.warning("  PRISM dose-response matching produced 0 samples")
        except Exception as e:
            logger.error(f"  Failed to build PRISM training matrix: {e}")
            import traceback
            traceback.print_exc()

    # Also try viability-based matching if dose-response gave few samples
    if len(prism_X) < 50:
        logger.info("  Trying viability-based matching as supplement...")
        try:
            viability_path = DATA_RAW / "prism" / "secondary-screen-replicate-collapsed-logfold-change.csv"
            if viability_path.exists():
                logger.info("  Loading viability matrix (this may take a while)...")
                viability_df = load_prism_viability()
                via_X, via_y = build_prism_viability_training(
                    viability_df, treatment_info, lincs_sigs,
                    drug_fps, breast_depmap_ids, cell_info,
                    cache_dir=DATA_CACHE,
                )
                if len(via_X) > len(prism_X):
                    logger.info(f"  Viability-based matching: {len(via_X)} samples (using this)")
                    prism_X = via_X
                    prism_y = via_y
            else:
                logger.info("  Viability file not available, skipping")
        except Exception as e:
            logger.warning(f"  Viability-based matching failed: {e}")

    # ── Step 8: Combine GDSC2 + PRISM training data ────────────────
    logger.info("\n[Step 8] Combining GDSC2 + PRISM training data...")

    if len(existing_X) > 0 and len(prism_X) > 0:
        # Align columns
        common_cols = list(set(existing_X.columns) & set(prism_X.columns))
        if len(common_cols) < 10:
            logger.warning(
                f"  Only {len(common_cols)} common columns between GDSC2 and PRISM matrices. "
                f"Attempting to align..."
            )
            # The training matrices should have the same feature schema
            # GDSC2 columns: gene_cols + ecfp_cols + log_dose_um
            # PRISM columns: gene_cols + ecfp_cols + log_dose_um
            # Align by adding missing columns as zeros
            all_cols = list(dict.fromkeys(list(existing_X.columns) + list(prism_X.columns)))
            for col in all_cols:
                if col not in existing_X.columns:
                    existing_X[col] = 0.0
                if col not in prism_X.columns:
                    prism_X[col] = 0.0
            prism_X = prism_X[existing_X.columns]

        # Ensure same column order
        prism_X = prism_X.reindex(columns=existing_X.columns, fill_value=0)

        combined_X = pd.concat([existing_X, prism_X], ignore_index=True)
        combined_y = pd.concat(
            [existing_y.reset_index(drop=True), prism_y.reset_index(drop=True)],
            ignore_index=True,
        )

        logger.info(f"  GDSC2-only samples: {len(existing_X)}")
        logger.info(f"  PRISM samples: {len(prism_X)}")
        logger.info(f"  Combined samples: {len(combined_X)}")
        logger.info(f"  Combined features: {combined_X.shape[1]}")
    elif len(existing_X) > 0:
        combined_X = existing_X
        combined_y = existing_y
        logger.info(f"  Using GDSC2-only data ({len(existing_X)} samples)")
    else:
        logger.error("  No training data available.")
        return

    # ── Step 9: Train and compare models ────────────────────────────
    logger.info("\n[Step 9] Training and comparing models...")

    from sklearn.model_selection import KFold, cross_val_score
    import lightgbm as lgb
    from src.config import LIGHTGBM_DEFAULT_PARAMS, RANDOM_SEED

    results = []

    # Model 1: GDSC2-only
    if len(existing_X) > 0 and len(existing_y) > 0:
        logger.info(f"  Training GDSC2-only model ({len(existing_X)} samples)...")
        model_gdsc = lgb.LGBMRegressor(**LIGHTGBM_DEFAULT_PARAMS)
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        # Handle any NaN/inf
        X_clean = existing_X.copy()
        y_clean = existing_y.copy()
        valid = np.isfinite(y_clean) & (~X_clean.isna().any(axis=1))
        X_clean = X_clean[valid].reset_index(drop=True)
        y_clean = y_clean[valid].reset_index(drop=True)

        if len(X_clean) > 10:
            scores = cross_val_score(
                model_gdsc, X_clean, y_clean, cv=cv,
                scoring="neg_root_mean_squared_error",
            )
            rmse_gdsc = -scores.mean()
            rmse_std_gdsc = scores.std()
            logger.info(f"    GDSC2-only CV RMSE: {rmse_gdsc:.3f} +/- {rmse_std_gdsc:.3f}")
            results.append({
                "model": "GDSC2-only",
                "n_samples": len(X_clean),
                "n_features": X_clean.shape[1],
                "cv_rmse_mean": round(rmse_gdsc, 4),
                "cv_rmse_std": round(rmse_std_gdsc, 4),
            })

    # Model 2: GDSC2+PRISM combined
    if len(combined_X) > len(existing_X):
        logger.info(f"  Training GDSC2+PRISM model ({len(combined_X)} samples)...")
        model_combined = lgb.LGBMRegressor(**LIGHTGBM_DEFAULT_PARAMS)
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        # Handle any NaN/inf
        X_comb_clean = combined_X.copy()
        y_comb_clean = combined_y.copy()
        valid = np.isfinite(y_comb_clean) & (~X_comb_clean.isna().any(axis=1))
        X_comb_clean = X_comb_clean[valid].reset_index(drop=True)
        y_comb_clean = y_comb_clean[valid].reset_index(drop=True)

        if len(X_comb_clean) > 10:
            scores = cross_val_score(
                model_combined, X_comb_clean, y_comb_clean, cv=cv,
                scoring="neg_root_mean_squared_error",
            )
            rmse_combined = -scores.mean()
            rmse_std_combined = scores.std()
            logger.info(f"    GDSC2+PRISM CV RMSE: {rmse_combined:.3f} +/- {rmse_std_combined:.3f}")
            results.append({
                "model": "GDSC2+PRISM",
                "n_samples": len(X_comb_clean),
                "n_features": X_comb_clean.shape[1],
                "cv_rmse_mean": round(rmse_combined, 4),
                "cv_rmse_std": round(rmse_std_combined, 4),
            })

            # Train the combined model on all data and save
            logger.info("  Training final combined model on all data...")
            model_combined.fit(X_comb_clean, y_comb_clean)
            import joblib
            prism_model_path = RESULTS / "lightgbm_gdsc2_prism_model.joblib"
            joblib.dump(model_combined, prism_model_path)
            logger.info(f"  Saved combined model to {prism_model_path}")

    # Model 3: PRISM-only (if enough samples)
    if len(prism_X) > 30:
        logger.info(f"  Training PRISM-only model ({len(prism_X)} samples)...")
        model_prism = lgb.LGBMRegressor(**LIGHTGBM_DEFAULT_PARAMS)
        cv = KFold(n_splits=min(5, len(prism_X) // 5), shuffle=True, random_state=RANDOM_SEED)

        X_p_clean = prism_X.copy()
        y_p_clean = prism_y.copy()
        # Align features to existing_X columns (if needed)
        if len(existing_X) > 0:
            X_p_clean = X_p_clean.reindex(columns=existing_X.columns, fill_value=0)

        valid = np.isfinite(y_p_clean) & (~X_p_clean.isna().any(axis=1))
        X_p_clean = X_p_clean[valid].reset_index(drop=True)
        y_p_clean = y_p_clean[valid].reset_index(drop=True)

        if len(X_p_clean) > 10:
            scores = cross_val_score(
                model_prism, X_p_clean, y_p_clean, cv=cv,
                scoring="neg_root_mean_squared_error",
            )
            rmse_prism = -scores.mean()
            rmse_std_prism = scores.std()
            logger.info(f"    PRISM-only CV RMSE: {rmse_prism:.3f} +/- {rmse_std_prism:.3f}")
            results.append({
                "model": "PRISM-only",
                "n_samples": len(X_p_clean),
                "n_features": X_p_clean.shape[1],
                "cv_rmse_mean": round(rmse_prism, 4),
                "cv_rmse_std": round(rmse_std_prism, 4),
            })

    # Save model comparison
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS / "prism_model_comparison.csv", index=False)
        logger.info(f"\n  Model comparison saved to {RESULTS / 'prism_model_comparison.csv'}")
        logger.info("\n  Model Comparison:")
        logger.info(results_df.to_string(index=False))

    # ── Step 10: Validate on CTR-DB patients ────────────────────────
    logger.info("\n[Step 10] Validating on CTR-DB patients...")

    prism_model_path = RESULTS / "lightgbm_gdsc2_prism_model.joblib"
    gdsc_model_path = RESULTS / "lightgbm_drug_model.joblib"

    validation_results = []

    for model_name, model_path in [
        ("GDSC2-only", gdsc_model_path),
        ("GDSC2+PRISM", prism_model_path),
    ]:
        if not model_path.exists():
            logger.info(f"  {model_name} model not found at {model_path}, skipping validation")
            continue

        logger.info(f"\n  Validating {model_name} model...")
        try:
            from src.models.validate_on_patients import validate_on_ctrdb_patients
            val_results = validate_on_ctrdb_patients(
                model_path=model_path,
                fp_path=fp_path if fp_path.exists() else DATA_CACHE / "drug_fingerprints.parquet",
                output_path=RESULTS / f"ctrdb_validation_{model_name.lower().replace('+', '_')}.csv",
            )
            if len(val_results) > 0:
                val_results["model"] = model_name
                validation_results.append(val_results)
                mean_auc = val_results["auc"].mean()
                logger.info(f"    {model_name} mean AUC: {mean_auc:.3f}")
        except Exception as e:
            logger.warning(f"    Validation failed for {model_name}: {e}")

    if validation_results:
        all_val = pd.concat(validation_results, ignore_index=True)
        all_val.to_csv(RESULTS / "prism_patient_validation_comparison.csv", index=False)
        logger.info(f"\n  Patient validation comparison saved to {RESULTS / 'prism_patient_validation_comparison.csv'}")

        # Report per-drug results
        for model_name in all_val["model"].unique():
            subset = all_val[all_val["model"] == model_name]
            logger.info(f"\n  {model_name} patient validation:")
            for _, row in subset.iterrows():
                logger.info(
                    f"    {row['geo_id']} ({row['drug']}): "
                    f"AUC={row['auc']:.3f}, p={row['wilcoxon_pvalue']:.4f}"
                )

    # ── Step 11: Drug coverage table ────────────────────────────────
    logger.info("\n[Step 11] Building drug coverage table...")
    if len(treatment_info) > 0 and len(gdsc_dr) > 0 and len(lincs_sigs) > 0:
        coverage_table = build_drug_coverage_table(
            treatment_info, gdsc_dr, lincs_sigs,
            gdsc2_n_samples=len(existing_X),
            prism_n_samples=len(prism_X),
        )
        logger.info("\nDrug Coverage Comparison:")
        logger.info(coverage_table.to_string(index=False))
        coverage_table.to_csv(RESULTS / "prism_drug_coverage_table.csv", index=False)

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("PRISM Integration Pipeline — Complete")
    logger.info("=" * 70)
    logger.info(f"  Total time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"  PRISM drugs: {n_total_drugs}")
    logger.info(f"  Breast cell lines in PRISM: {len(breast_depmap_ids)}")
    logger.info(f"  Breast PRISM dose-response records: {len(breast_prism_dr)}")
    logger.info(f"  PRISM training samples: {len(prism_X)}")
    logger.info(f"  Combined training samples: {len(existing_X) + len(prism_X)}")
    logger.info(f"\n  Output files:")
    for fname in [
        "prism_integration_analysis.csv",
        "prism_model_comparison.csv",
        "prism_drug_coverage_table.csv",
        "prism_patient_validation_comparison.csv",
        "lightgbm_gdsc2_prism_model.joblib",
    ]:
        fpath = RESULTS / fname
        if fpath.exists():
            logger.info(f"    {fpath}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
