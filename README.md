# INVEREX MVP — Breast Cancer Drug Response & Ranking Pipeline

Molecular-profile-driven drug response prediction and personalized drug ranking for breast cancer patients.

> Patient molecular profile → response prediction → drug ranking → clinical trial suggestions → interpretable rationale

## Models

### 1. Patient Response Model (primary)

LightGBM classifier trained directly on **pooled CTR-DB patient data** (37 neoadjuvant chemotherapy datasets, 5,143 patients) with leave-one-dataset-out (LODO) cross-validation.

| Metric | Value |
|--------|-------|
| **LODO AUC (expression only)** | 0.597 |
| **LODO AUC (expression + clinical)** | **0.610** |
| Pathologic response (pCR) AUC | 0.663 |
| Pathologic response (pCR) + harmonized datasets AUC  | 0.767 |
| Features | 954 L1000 landmark genes + ER/HER2 status |
| Hyperparameters | Optuna-optimized (50 trials) |

Top predictive features: ORC1, CCND1, PTGS2, ER status, PSMB10, HER2 status.

See `results/definitive_retrain_full/summary.md` for full results.

### 2. Cell-Line Drug-Response Model (auxiliary)

LightGBM regressor trained on matched LINCS L1000 × GDSC2 × PubChem data, scoped to breast cancer cell lines.

- **Features**: L1000 gene z-scores (~978) + ECFP4 drug fingerprints (1024-bit) + log-dose
- **Target**: Percent cell inhibition (continuous, from PharmacoDB dose-response curves)
- Used as an auxiliary prior in the personalized ranker

### 3. Composite Personalized Ranker

Modular ranking layer combining multiple signals per patient-drug pair:

- **RNA reversal score**: patient dysregulation vs. LINCS perturbation signatures
- **Mutation / pathway score**: biomarker-pathway bonuses (HER2, PI3K/AKT/mTOR)
- **Subtype / tissue-context score**: PAM50-aware breast-cancer rules
- **Clinical actionability score**: favors clinically relevant agents
- **ML prior**: patient response model + cell-line model predictions

### 4. Foundation Expression Encoder (experimental)

Gene-aware transformer with multi-objective pretraining (masked gene prediction, pathway activity, subtype, mutation proxy, domain adversarial). Architecture validated on CPU; GPU training would unlock full potential. See `docs/foundation_model_results.md`.

## Data Sources

| Source | What we use | Access |
|--------|------------|--------|
| CTR-DB | 37 breast cancer patient response datasets (GEO) | GEO bulk downloads |
| LINCS L1000 (Level 5) | Drug perturbation signatures for breast cell lines | GEO GSE92742 / clue.io |
| GDSC2 / PharmacoDB | Dose-response viability for cell lines | pharmacodb.ca API |
| PubChem | SMILES → ECFP4 fingerprints | PubChem PUG-REST |
| TCGA-BRCA | Bulk RNA-seq + mutations + clinical | UCSC Xena / GDC |
| ClinicalTrials.gov | Active breast cancer trials | CT.gov API v2 |

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Download and cache all data
python -m src.data_ingestion.run_all

# 2. Train patient response model (CTR-DB LODO)
python scripts/definitive_retrain_full.py

# 3. Build cell-line training matrix (LINCS x PharmacoDB x PubChem)
python -m src.features.build_training_matrix

# 4. Train cell-line LightGBM
python -m src.models.train_lightgbm

# 5. Run inference on TCGA-BRCA patients
python -m src.models.predict_patients

# 6. Personalized ranking
# from src.ranking.personalized_ranker import rank_tcga_patient
# rankings, summary = rank_tcga_patient("TCGA-A2-A04W-01", top_k=15)

# 7. Launch Streamlit app
streamlit run app/main.py
```

## Key Results

| Experiment | AUC | Notes |
|-----------|-----|-------|
| Patient model (954 genes + clinical) | **0.610** | Production model, honest LODO |
| Patient model (212 curated genes) | 0.617 | Fewer but hand-picked genes |
| ComBat with response labels (leaked) | 0.767 | Label leakage — not valid for production |
| ComBat without labels | 0.604 | Honest batch correction |
| Foundation encoder | TBD | CPU-limited, architecture validated |

See `results/EXPERIMENT_SUMMARY.md` for the full analysis.
