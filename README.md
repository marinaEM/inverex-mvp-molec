# INVEREX MVP — Breast Cancer Drug Ranking Pipeline

**Retrospective mock demo** using TCGA-BRCA patient data.  
**Not** a clinical response predictor. Shows an end-to-end pipeline:

> TCGA-BRCA patient molecular profile → disease signature → drug ranking → mapped breast-cancer trial suggestions → interpretable rationale

## Architecture

### Drug Ranking Model (inspired by scTherapy)

Trained on the same data schema as scTherapy (Ianevski & Nader et al., Nat Commun 2024),
but **scoped to breast cancer cell lines** and **rebuilt locally** for full interpretability:

- **Features**: L1000 landmark gene fold-changes (~978) + ECFP4 drug fingerprints (1024-bit) + log-dose
- **Target**: Percent cell inhibition (continuous, from PharmacoDB dose-response curves)
- **Model**: LightGBM regressor with Bayesian hyperparameter optimization

At inference, TCGA-BRCA patient DEGs (vs. subtype-aware centroid) substitute for
drug-induced expression changes, and the model predicts which drugs at which doses
would produce the strongest inhibition.

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

# 5. Launch Streamlit app
streamlit run app/main.py
```

## Disclaimer

This is a **retrospective mock ranking model** with no clinical validation.
All outputs are for research demonstration purposes only.
