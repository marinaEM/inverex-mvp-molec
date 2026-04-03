"""
INVEREX MVP — Central configuration.

All paths, constants, and breast-cancer-specific settings live here.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

for d in [DATA_RAW, DATA_PROCESSED, DATA_CACHE, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Breast cancer cell lines (well-represented in LINCS L1000) ─────────
# Maps common name → Cellosaurus ID for PharmacoDB matching
BREAST_CELL_LINES = {
    "MCF7":       "CVCL_0031",
    "T47D":       "CVCL_0553",
    "MDA-MB-231": "CVCL_0062",
    "BT474":      "CVCL_0179",
    "SKBR3":      "CVCL_0033",
    "HS578T":     "CVCL_0332",
    "MDA-MB-468": "CVCL_0419",
    "HCC1937":    "CVCL_0290",
    "ZR751":      "CVCL_0588",
    "MDA-MB-436": "CVCL_0623",
    "CAL51":      "CVCL_1110",
    "HCC1806":    "CVCL_1258",
    "BT549":      "CVCL_1092",
    "MDA-MB-157": "CVCL_0618",
}

# LINCS L1000 cell_id values for breast lines (from siginfo)
BREAST_CELL_IDS_LINCS = [
    "MCF7", "T47D", "MDAMB231", "BT474", "SKBR3",
    "HS578T", "MDAMB468", "HCC1937",
]

# ── L1000 landmark genes ───────────────────────────────────────────────
# The 978 "landmark" genes measured directly in L1000.
# Will be loaded from the gene info file; this is the expected count.
N_LANDMARK_GENES = 978

# ── Drug fingerprint settings ──────────────────────────────────────────
ECFP_RADIUS = 2          # ECFP4 = radius 2
ECFP_NBITS = 1024        # bit-vector length

# ── PharmacoDB settings ────────────────────────────────────────────────
PHARMACODB_API = "https://pharmacodb.ca/api/v1"

# ── TCGA-BRCA settings (via UCSC Xena) ────────────────────────────────
XENA_HUB = "https://tcga.xenahubs.net/download"
TCGA_EXPRESSION_DATASET = "TCGA.BRCA.sampleMap/HiSeqV2"
TCGA_MUTATION_DATASET = "TCGA.BRCA.sampleMap/mutation_curated_wustl"
TCGA_CLINICAL_DATASET = "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"
TCGA_CNV_DATASET = "TCGA.BRCA.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"

# ── BRCA PAM50 subtypes ───────────────────────────────────────────────
PAM50_SUBTYPES = ["LumA", "LumB", "Her2", "Basal", "Normal"]

# ── ClinicalTrials.gov API ─────────────────────────────────────────────
CTGOV_API = "https://clinicaltrials.gov/api/v2"

# ── Model settings ─────────────────────────────────────────────────────
RANDOM_SEED = 42
LIGHTGBM_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
