# Personalization Gap Analysis

## Executive Summary

The current MVP ranking is only weakly personalized. The per-patient RNA signature changes the model input, but the final ranking is still dominated by global drug priors learned from LINCS/PharmacoDB breast cell lines and by dose effects. Mutation, CNV, subtype rules, and clinical actionability are present in metadata but absent from the final score.

## Key Gaps

### 1. Rankings are dominated by global drug priors and dose

Code path:

- `src/features/build_training_matrix.py::build_training_matrix()`
- `src/models/train_lightgbm.py::train_lightgbm()`
- `src/models/predict_patients.py::predict_drugs_for_patient()`

Observed evidence:

- The training matrix uses only LINCS perturbation z-scores, drug ECFP fingerprints, and `log_dose_um`.
- `results/feature_importances.csv` currently ranks `log_dose_um` as the single most important feature.
- Many exported ranking files share nearly identical top drugs across unrelated patients, which is consistent with a strong global prior.

Technical implication:

- At inference, the model sees the same drug fingerprint and dose grid for every patient.
- If the trained regressor has learned that some drug classes are broadly inhibitory in matched breast cell lines, those classes will recur across patients even when subtype or mutation context should separate them.

### 2. The patient ranking path does not explicitly use mutations or CNV

Code path:

- mutation/CNV ingestion exists in `src/data_ingestion/tcga.py::build_patient_cohort()`
- final ranking path is `src/models/predict_patients.py::predict_drugs_for_patient()`

Observed evidence:

- `build_patient_cohort()` adds `mut_TP53`, `mut_PIK3CA`, `mut_ERBB2`, `mut_PTEN`, `mut_AKT1`, `mut_ESR1`, `mut_BRCA1`, `mut_BRCA2`, and `ERBB2_amp`.
- `predict_drugs_for_patient()` only consumes:
  - patient RNA signature
  - drug fingerprint
  - dose
- None of the mutation/CNV columns are included in the ranking score.

Technical implication:

- A HER2-positive / PIK3CA-mutant patient and a HER2-negative / PIK3CA-wildtype patient can receive very similar rankings if their RNA signatures map similarly onto the LightGBM prior.

### 3. PAM50 subtype is only used indirectly for the reference centroid

Code path:

- `src/data_ingestion/tcga.py::compute_patient_signature()`

Observed evidence:

- Subtype is used only to choose a reference cohort for z-scoring.
- After that, subtype disappears from the ranking logic.

Technical implication:

- HER2 subtype does not create an explicit HER2-targeting bonus.
- Luminal context does not explicitly prefer endocrine / CDK4/6 logic.
- Basal/TNBC context does not explicitly distinguish DNA-damage / immune / chemotherapy logic from luminal logic.

### 4. LINCS reversal exists as a helper but is not part of the deployed ranking output

Code path:

- `src/models/predict_patients.py::compute_reversal_score()`

Observed evidence:

- The helper computes the expected anti-correlation baseline between a patient signature and a drug signature.
- The exported patient ranking pipeline does not call it when generating `results/drug_rankings_*.csv`.

Technical implication:

- The app and CSV outputs are not combining an explicit transcriptomic reversal score with the ML score.
- The current ranking is therefore not auditable as "RNA reversal + other evidence"; it is mostly "ML-predicted inhibition".

### 5. Drug rationale is not actually drug-specific

Code path:

- `src/models/predict_patients.py::predict_drugs_for_patient()`

Observed evidence:

- The column `top_contributing_genes` is derived from:
  - model feature importance
  - patient RNA magnitude
- It does not use drug-specific perturbation evidence.
- For a given patient, many drugs receive the same rationale string.

Technical implication:

- The rationale panel does not explain why one drug outranks another.
- It mostly explains which patient genes the model globally considers important.

### 6. Clinical actionability filters are missing

Code path:

- no current actionability module exists
- trial mapping in `src/trials/match_trials.py` is a string lookup only

Observed evidence:

- `data/cache/lincs_pharmacodb_matched.parquet` includes `MG-132`.
- The same matched set includes compounds such as `lestaurtinib`.
- Existing patient ranking CSVs place `lestaurtinib` in the top 15 for a large fraction of patients.

Technical implication:

- Compounds can rank highly because they are potent in breast cell lines, even if they are tool compounds, weakly actionable, or not clinically plausible for breast cancer.

### 7. The current LightGBM objective is not a clinical response model

Code path:

- `src/models/train_lightgbm.py`

Observed evidence:

- Labels are percent inhibition from cell-line dose response, not patient treatment response.
- Training rows are `(drug, dose, cell line)` matched from LINCS and PharmacoDB.
- The current code does not model patient-level biomarkers or clinical outcomes.

Technical implication:

- The LightGBM model can be useful as a ranking aid, but it should not be the dominant score.
- If left uncorrected, it will preferentially surface broadly cytotoxic or potency-heavy compounds.

## Example Failure Pattern

For real HER2-positive / PAM50 Her2 / TP53-mutant / PIK3CA-mutant TCGA cases such as:

- `TCGA-A2-A04W-01`
- `TCGA-A2-A0D1-01`

current top-ranked drugs include:

- `temsirolimus`
- `sirolimus`
- `docetaxel`
- `paclitaxel`
- `vinorelbine`
- `vinblastine`
- `lestaurtinib`
- `methotrexate`

This pattern is consistent with:

- a strong generic cytotoxic prior
- a strong mTOR/cell-growth prior
- insufficient HER2-specific contextualization
- no explicit clinical filter against weakly actionable compounds

## Diagnosis

The personalization gap is not caused by one bug. It is structural:

1. patient RNA is the only deployed patient-specific signal
2. mutations and CNV are ingested but unused in ranking
3. subtype is treated as a reference-selection detail, not a ranking feature
4. clinical plausibility is not encoded
5. the LightGBM component is allowed to dominate even though it is trained on weakly supervised cell-line inhibition labels

## Required Direction of Fix

The ranking layer needs to become an explicit composite model:

- RNA reversal should remain important, but as one component
- mutation/CNV-pathway relevance must directly alter the score
- subtype/tissue context must directly alter the score
- clinical actionability must be encoded, with tool compounds excluded or heavily penalized
- the LightGBM prediction should be retained only as an auxiliary prior

That architecture is the minimal change needed to move the MVP from "generic breast-cell-line potency ranking" toward "retrospective personalized breast-cancer ranking with explicit rationale."
