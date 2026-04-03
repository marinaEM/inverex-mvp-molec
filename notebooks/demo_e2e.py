#!/usr/bin/env python3
"""
INVEREX MVP — End-to-end demo.

Runs the full pipeline on synthetic data to verify all components work:
  1. Build training matrix (LINCS × PharmacoDB × PubChem schema)
  2. Train LightGBM drug-response model
  3. Create mock patients with realistic breast cancer profiles
  4. Predict drug rankings per patient
  5. Compare reversal baseline vs LightGBM
  6. Generate patient-level report
"""
import json
import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_CACHE, DATA_PROCESSED, ECFP_NBITS, RANDOM_SEED, RESULTS
from src.data_ingestion.lincs import load_landmark_genes
from src.features.build_training_matrix import build_training_matrix
from src.models.predict_patients import (
    EVAL_DOSES,
    compute_reversal_score,
    predict_drugs_for_patient,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 1. BUILD TRAINING MATRIX
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("PHASE 1: Building training matrix")
logger.info("=" * 60)

X, y, features = build_training_matrix(
    use_demo=True, cache_dir=DATA_CACHE, output_dir=DATA_PROCESSED
)

gene_feats = [f for f in features if not f.startswith("ecfp_") and f != "log_dose_um"]
ecfp_feats = [f for f in features if f.startswith("ecfp_")]

print(f"\n  Training matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features")
print(f"  Gene z-scores:  {len(gene_feats)}")
print(f"  ECFP4 bits:     {len(ecfp_feats)}")
print(f"  Dose feature:   1")
print(f"  Target range:   [{y.min():.1f}%, {y.max():.1f}%]")


# ══════════════════════════════════════════════════════════════════════
# 2. TRAIN LIGHTGBM
# ══════════════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("PHASE 2: Training LightGBM")
logger.info("=" * 60)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 200,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(X, y)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(
    lgb.LGBMRegressor(**params), X, y, cv=cv, scoring="neg_root_mean_squared_error"
)
rmse = -cv_scores.mean()
print(f"\n  CV RMSE: {rmse:.3f} ± {cv_scores.std():.3f}")

# Feature importance
imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
imp = imp.sort_values("importance", ascending=False)
gene_imp = imp[~imp["feature"].str.startswith("ecfp_") & (imp["feature"] != "log_dose_um")]
print(f"\n  Top 10 gene features by importance:")
for _, row in gene_imp.head(10).iterrows():
    print(f"    {row['feature']:15s} importance={row['importance']:.0f}")

RESULTS.mkdir(parents=True, exist_ok=True)
joblib.dump(model, RESULTS / "lightgbm_drug_model.joblib")


# ══════════════════════════════════════════════════════════════════════
# 3. CREATE MOCK BREAST CANCER PATIENTS
# ══════════════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("PHASE 3: Creating mock breast cancer patient profiles")
logger.info("=" * 60)

gene_df = load_landmark_genes(DATA_CACHE)
gene_symbols = gene_df["gene_symbol"].tolist()
rng = np.random.default_rng(42)

# Define 5 archetypal breast cancer patient profiles
patient_profiles = {
    "BRCA_HER2_001": {
        "subtype": "HER2-enriched",
        "overrides": {"ERBB2": 5.0, "EGFR": 2.5, "MKI67": 3.0,
                       "ESR1": -1.5, "PGR": -1.8, "TOP2A": 2.5},
        "mutations": ["TP53", "PIK3CA"],
        "description": "HER2-amplified, ER-negative, high proliferation",
    },
    "BRCA_LUMA_002": {
        "subtype": "Luminal A",
        "overrides": {"ESR1": 3.5, "PGR": 3.0, "GATA3": 2.5, "FOXA1": 2.0,
                       "ERBB2": -0.5, "MKI67": -1.0},
        "mutations": ["PIK3CA", "CDH1"],
        "description": "ER+/PR+, low proliferation, PIK3CA-mutant",
    },
    "BRCA_LUMB_003": {
        "subtype": "Luminal B",
        "overrides": {"ESR1": 2.0, "PGR": 0.5, "MKI67": 3.5,
                       "ERBB2": 1.5, "CCND1": 2.5, "CDK4": 2.0},
        "mutations": ["TP53"],
        "description": "ER+/HER2-low, high proliferation, aggressive",
    },
    "BRCA_BASAL_004": {
        "subtype": "Basal-like (TNBC)",
        "overrides": {"ESR1": -3.0, "PGR": -2.5, "ERBB2": -1.0,
                       "MKI67": 4.0, "EGFR": 3.0, "TOP2A": 3.5,
                       "BRCA1": -2.0},
        "mutations": ["TP53", "BRCA1"],
        "description": "Triple-negative, BRCA1-deficient, highly proliferative",
    },
    "BRCA_PIK3CA_005": {
        "subtype": "Luminal A with PIK3CA + AKT activation",
        "overrides": {"ESR1": 3.0, "PGR": 2.5, "PIK3CA": 2.5,
                       "AKT1": 2.0, "MTOR": 1.5, "PTEN": -2.0},
        "mutations": ["PIK3CA", "AKT1"],
        "description": "ER+, PI3K/AKT/mTOR pathway activated",
    },
}

# Generate patient signatures
patient_signatures = {}
for pid, profile in patient_profiles.items():
    sig = pd.Series(rng.standard_normal(len(gene_symbols)) * 0.5, index=gene_symbols)
    for gene, val in profile["overrides"].items():
        if gene in sig.index:
            sig[gene] = val
    patient_signatures[pid] = sig
    print(f"\n  {pid}: {profile['description']}")
    print(f"    Subtype: {profile['subtype']}")
    print(f"    Mutations: {', '.join(profile['mutations'])}")
    top_up = sig.nlargest(3)
    top_dn = sig.nsmallest(3)
    print(f"    Top up: {', '.join(f'{g}={v:.1f}' for g, v in top_up.items())}")
    print(f"    Top dn: {', '.join(f'{g}={v:.1f}' for g, v in top_dn.items())}")


# ══════════════════════════════════════════════════════════════════════
# 4. PREDICT DRUG RANKINGS
# ══════════════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("PHASE 4: Predicting drug rankings for each patient")
logger.info("=" * 60)

# Build drug fingerprints (mock)
drug_names = [
    "lapatinib", "neratinib", "tucatinib",           # HER2-targeted
    "tamoxifen", "fulvestrant", "letrozole",           # Endocrine
    "palbociclib", "ribociclib", "abemaciclib",        # CDK4/6
    "alpelisib", "everolimus", "ipatasertib",          # PI3K/AKT/mTOR
    "olaparib", "talazoparib",                         # PARP
    "paclitaxel", "doxorubicin", "capecitabine",       # Chemo
    "pembrolizumab", "atezolizumab",                   # IO
    "dasatinib", "trametinib", "vorinostat",           # Other targeted
]

fp_rows = []
for i, drug in enumerate(drug_names):
    row = {"compound_name": drug, "smiles": "DEMO"}
    # Use different random seeds per drug for distinct fingerprints
    drug_rng = np.random.default_rng(i * 1000 + 7)
    bits = drug_rng.random(ECFP_NBITS) < 0.1
    for j, bit in enumerate(bits):
        row[f"ecfp_{j}"] = int(bit)
    fp_rows.append(row)
fp_df = pd.DataFrame(fp_rows)

all_rankings = {}
for pid, sig in patient_signatures.items():
    rankings = predict_drugs_for_patient(
        sample_id=pid,
        model=model,
        drug_fingerprints=fp_df,
        patient_signature=sig,
        top_k=10,
    )
    all_rankings[pid] = rankings

    print(f"\n  {pid} ({patient_profiles[pid]['subtype']}):")
    print(f"  {'Rank':<5} {'Drug':<18} {'Inhibition':>10} {'Dose (µM)':>10} {'Confidence':>12}")
    print(f"  {'-'*55}")
    for rank, (_, r) in enumerate(rankings.head(5).iterrows(), 1):
        print(
            f"  {rank:<5} {r['drug_name']:<18} {r['predicted_inhibition']:>9.1f}% "
            f"{r['best_dose_um']:>9.2f}  {r['confidence']:>12}"
        )


# ══════════════════════════════════════════════════════════════════════
# 5. GENERATE PATIENT REPORTS
# ══════════════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("PHASE 5: Generating patient reports")
logger.info("=" * 60)

reports = []
for pid, rankings in all_rankings.items():
    profile = patient_profiles[pid]
    sig = patient_signatures[pid]

    report = {
        "patient_id": pid,
        "subtype": profile["subtype"],
        "description": profile["description"],
        "mutations": profile["mutations"],
        "top_dysregulated_up": sig.nlargest(5).to_dict(),
        "top_dysregulated_down": sig.nsmallest(5).to_dict(),
        "drug_rankings": rankings.head(10).to_dict("records"),
    }
    reports.append(report)

    # Save per-patient CSV
    rankings.to_csv(RESULTS / f"drug_rankings_{pid}.csv", index=False)

# Save combined report
with open(RESULTS / "patient_reports.json", "w") as f:
    json.dump(reports, f, indent=2, default=str)

print(f"\n  Reports saved to {RESULTS}")
print(f"  Files:")
for f in sorted(RESULTS.iterdir()):
    size = f.stat().st_size
    print(f"    {f.name:40s}  {size:>8,} bytes")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("INVEREX MVP — Demo Complete")
print("=" * 60)
print(f"""
Pipeline validated end-to-end with synthetic data:

  ✓ Training matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features
    (gene z-scores + ECFP4 drug fingerprints + dose)
  ✓ LightGBM model: CV RMSE = {rmse:.3f}
  ✓ {len(patient_profiles)} mock patients with realistic BRCA profiles
  ✓ Drug rankings generated for {len(drug_names)} breast cancer drugs
  ✓ Patient reports saved

NOTE: These results use SYNTHETIC data. For meaningful predictions:
  1. Download LINCS L1000 Level 5 signatures (breast cell lines)
  2. Download GDSC2 dose-response data
  3. Fetch SMILES from PubChem → compute real ECFP4 fingerprints
  4. Run: python -m src.run_pipeline --real --optimize
  5. Load real TCGA-BRCA patient data
""")
