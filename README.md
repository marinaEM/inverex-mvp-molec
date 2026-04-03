# INVEREX MVP — Breast Cancer Drug Ranking Pipeline

**Retrospective mock demo** using TCGA-BRCA patient data.  
**Not** a clinical response predictor. Shows an end-to-end pipeline:

> TCGA-BRCA patient molecular profile → disease signature → drug ranking → mapped breast-cancer trial suggestions → interpretable rationale

## Architecture

### Composite Personalized Ranker

The ranking layer is now explicit and modular for each patient-drug pair:

- **RNA reversal score**: patient RNA dysregulation versus breast LINCS perturbation signatures
- **Mutation / pathway score**: curated biomarker-pathway bonuses such as HER2 and PI3K/AKT/mTOR
- **Subtype / tissue-context score**: PAM50-aware breast-cancer context rules
- **Clinical actionability score**: favors clinically plausible breast-cancer agents and penalizes tool compounds
- **Optional ML prior**: LightGBM cell-line inhibition model used only as an auxiliary ranking prior

Final rankings are therefore not driven by perturbation potency alone.

### Auxiliary LightGBM Drug-Response Model (inspired by scTherapy)

Trained on the same data schema as scTherapy (Ianevski & Nader et al., Nat Commun 2024),
but **scoped to breast cancer cell lines** and **rebuilt locally** for interpretability:

- **Features**: L1000 landmark gene fold-changes (~978) + ECFP4 drug fingerprints (1024-bit) + log-dose
- **Target**: Percent cell inhibition (continuous, from PharmacoDB dose-response curves)
- **Model**: LightGBM regressor with Bayesian hyperparameter optimization

Important: this model is not treated as a validated patient response predictor. It is one component inside the broader personalized rationale stack.

### Baseline: Signature Reversal Score

Classical connectivity-map-style anti-correlation between patient disease signatures
and L1000 drug perturbation signatures.

## Data Sources

| Source | What we use | Access |
|--------|------------|--------|
| LINCS L1000 (Level 5) | Drug perturbation signatures for breast cell lines | GEO GSE92742 / clue.io |
| PharmacoDB | Dose-response viability for matched cell lines | pharmacodb.ca API |
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

# 2. Build matched training matrix (LINCS ↔ PharmacoDB ↔ PubChem)
python -m src.features.build_training_matrix

# 3. Train LightGBM
python -m src.models.train_lightgbm

# 4. Run inference on TCGA-BRCA patients
python -m src.models.predict_patients

# 4b. Programmatic personalized ranking
# from src.ranking.personalized_ranker import rank_tcga_patient
# rankings, summary = rank_tcga_patient("TCGA-A2-A04W-01", top_k=15)

# 5. Launch Streamlit app
streamlit run app/main.py
```

## Disclaimer

This is a **retrospective mock ranking model** with no clinical validation.
All outputs are for research demonstration purposes only.
