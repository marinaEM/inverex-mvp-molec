# Current Pipeline Audit

## Scope

This audit covers the current breast-cancer-only MVP code paths for:

- TCGA ingestion
- patient signature generation
- LINCS processing
- model training
- patient ranking
- trial mapping

## Repository Map

### Core pipeline modules

- `src/data_ingestion/tcga.py`
  - `load_tcga_expression()`
  - `load_tcga_clinical()`
  - `load_tcga_mutations()`
  - `load_tcga_cnv()`
  - `build_patient_cohort()`
  - `compute_patient_signature()`
- `src/data_ingestion/lincs.py`
  - `load_lincs_siginfo()`
  - `filter_breast_signatures()`
  - `build_breast_signature_matrix()`
- `src/data_ingestion/pharmacodb.py`
  - breast-cell-line dose-response ingestion and LINCS matching
- `src/data_ingestion/pubchem.py`
  - SMILES lookup and ECFP4 fingerprint generation
- `src/features/build_training_matrix.py`
  - LINCS + PharmacoDB + PubChem training matrix assembly
- `src/models/train_lightgbm.py`
  - LightGBM regressor training and feature importances
- `src/models/predict_patients.py`
  - patient-level inference and CSV export
  - `compute_reversal_score()` baseline helper
- `src/trials/match_trials.py`
  - ClinicalTrials.gov mapping by drug name
- `app/main.py`
  - Streamlit UI reading precomputed ranking CSVs and patient metadata

### Current cached / generated artifacts used by the pipeline

- `data/cache/tcga_brca_expression.parquet`
- `data/cache/tcga_brca_cohort.parquet`
- `data/cache/breast_l1000_signatures.parquet`
- `data/cache/lincs_pharmacodb_matched.parquet`
- `data/cache/drug_fingerprints.parquet`
- `data/processed/training_matrix.parquet`
- `results/lightgbm_drug_model.joblib`
- `results/feature_importances.csv`
- `results/drug_rankings_*.csv`
- `results/patient_reports.json`

## Current End-to-End Flow

### 1. TCGA ingestion and cohort building

`src/data_ingestion/tcga.py` downloads or loads:

- BRCA bulk RNA expression from UCSC Xena
- BRCA clinical metadata including PAM50 labels when available
- BRCA mutation calls
- BRCA gene-level CNV

`build_patient_cohort()` merges expression-backed samples with clinical metadata and adds:

- `pam50_subtype`
- `er_status`, `pr_status`, `her2_status`
- mutation flags for key genes including `TP53`, `PIK3CA`, `ERBB2`, `PTEN`, `AKT1`, `ESR1`, `BRCA1`, `BRCA2`
- `ERBB2_amp` from CNV

Result: the cohort metadata is reasonably rich, and mutation/CNV flags already exist in cache.

### 2. Patient signature generation

`compute_patient_signature()` produces one patient RNA signature by:

- taking the patient expression values for available LINCS landmark genes
- comparing the patient either to:
  - the subtype centroid if PAM50 is available and there are at least 10 peers, or
  - the full BRCA cohort centroid otherwise
- returning per-gene z-scores

This is the only molecular signal currently passed into the ranking model.

### 3. LINCS processing

`src/data_ingestion/lincs.py`:

- loads LINCS signature metadata
- filters to breast cell lines, compound perturbations, and 24h time points
- builds a breast-only Level 5 matrix when the `.gctx` file is present

The cached matrix `data/cache/breast_l1000_signatures.parquet` currently contains many breast-line signatures, but the final patient ranking code does not explicitly use these signatures at scoring time.

### 4. Training data assembly

`src/features/build_training_matrix.py` constructs one row per matched `(drug, dose, cell line)`:

- LINCS landmark-gene perturbation z-scores
- ECFP4 fingerprint bits from PubChem/RDKit
- `log_dose_um`

Target:

- `pct_inhibition` from matched PharmacoDB/GDSC dose-response measurements

The cached matched matrix currently includes 719 matched rows across 103 compounds.

### 5. LightGBM training

`src/models/train_lightgbm.py` trains a LightGBM regressor on:

- gene perturbation features
- ECFP fingerprint features
- dose

Outputs:

- `results/lightgbm_drug_model.joblib`
- `results/feature_importances.csv`
- `results/lightgbm_metrics.json`

Observed current state:

- `results/feature_importances.csv` shows `log_dose_um` as the top feature.
- The training code does not actually include subtype, mutation, pathway, or clinical metadata despite comments in the file header mentioning optional subtype indicators.

### 6. Patient ranking

`src/models/predict_patients.py::predict_drugs_for_patient()`:

1. computes a patient RNA signature
2. aligns the patient signature to the model's gene feature columns
3. concatenates:
   - patient RNA z-scores
   - each drug fingerprint
   - each evaluation dose
4. predicts inhibition for every drug-dose pair
5. keeps the best dose per drug

This is the current ranking engine used to create `results/drug_rankings_*.csv`.

### 7. Trial mapping

`src/trials/match_trials.py` and `app/main.py` query ClinicalTrials.gov by:

- condition = breast cancer
- intervention = exact drug name string from the ranking output

This is a thin string-based mapping layer with no synonym management, no actionability gating, and no biomarker-aware matching.

## What Currently Makes the Ranking Patient-Specific

- The patient RNA signature is computed separately for each TCGA-BRCA sample.
- The RNA signature is subtype-relative when PAM50 labels are available.
- The patient-specific RNA vector changes the LightGBM prediction input.

That is the main personalization mechanism in the current codebase.

## What Is Not Currently Personalized

- Mutation flags and CNV are not used in the final ranking score.
- PAM50 subtype is not used directly as a ranking rule or feature after signature generation.
- There is no explicit subtype-context bonus for HER2, Luminal, or Basal/TNBC settings.
- There is no explicit drug-level clinical actionability filter.
- The reversal-score helper exists but is not used in the exported patient ranking CSVs.
- Trial mapping does not use biomarker logic, aliases, or evidence tiers.

## Where Clinical Rationale Is Missing

- Current outputs expose only:
  - predicted inhibition
  - best dose
  - a generic list of "top contributing genes"
- There is no decomposition into:
  - RNA reversal
  - mutation/pathway relevance
  - subtype context
  - clinical plausibility
- There is no evidence tiering.
- The same "top contributing genes" string is often repeated across many drugs for the same patient because it is based on model importance times patient RNA magnitude, not drug-specific rationale.

## Where Research / Tool Compounds Slip Through

The current training/ranking path has no explicit exclusion or penalty for tool compounds.

Evidence from current cached data:

- `data/cache/lincs_pharmacodb_matched.parquet` contains `MG-132` rows.
- The same cache also contains `lestaurtinib`.
- Existing `results/drug_rankings_*.csv` files place `lestaurtinib` in the top 15 for hundreds of patients.
- Current ranking outputs for HER2-positive / PIK3CA-mutant / TP53-mutant cases are dominated by:
  - taxanes / vinca agents
  - mTOR inhibitors
  - tool or weakly actionable compounds

## Summary of the Current Failure Mode

The current stack is best described as:

`patient RNA z-score vector -> generic drug-response regressor trained on breast cell-line perturbation strength`

It is partially patient-specific on the RNA side, but it is not explicitly:

- mutation-aware
- CNV-aware
- subtype-rule-aware
- clinically filtered
- evidence-tiered
- rationale-driven

That is why outputs can look plausible at the proliferation/stress-response level while still missing obvious HER2, PI3K-pathway, and clinical-actionability context.
