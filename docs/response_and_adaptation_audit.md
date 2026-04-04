# Response Endpoint and Adaptation Pipeline Audit

## 1. Data Ingestion: CTR-DB Response Labels (`src/data_ingestion/ctrdb.py`)

### 1.1 Label storage format
- Response labels are stored as `response_labels.parquet` in each `data/raw/ctrdb/GSE*/` directory.
- Each file contains a single column `response` with integer values 0 (non-responder) or 1 (responder).
- The original response categories (pCR, RD, CR, PR, SD, PD, etc.) are collapsed to binary at download time and the native labels are discarded.

### 1.2 Label parsing pipeline
The function `parse_geo_response_labels()` has a three-tier strategy:
1. **CTR-DB predefined grouping** (`_extract_labels_from_predefined_grouping`): Parses the string "Response: X; Non_response: Y" from the API. Most reliable source.
2. **CTR-DB original response grouping** (`_extract_labels_from_ctrdb_grouping`): Parses "pCR:46;RD:165;" style strings and applies pattern matching (neg_patterns, pos_patterns) to classify groups.
3. **Fallback heuristic** (`_heuristic_label_extraction`): Searches GEO phenotype columns for response-related keywords.

### 1.3 Binary conversion rules (hardcoded)
- pCR, pathologic complete, response, responder, sensitive, CR, PR, no relapse, no recurrence, rcb-0, rcb-i -> 1 (responder)
- RD, residual disease, npCR, nCR, non-pcr, non-response, non-responder, resistant, relapse, recurrence, rcb-ii, rcb-iii, progressive, SD, PD -> 0 (non-responder)
- **RECIST handling**: CR+PR are grouped as responders, SD+PD as non-responders. There is no configurable policy for SD. The predefined_grouping from CTR-DB sometimes specifies "Response: CR and PR; Non_response: SD and PD" (e.g., GSE82172, GSE41998, GSE66999).
- **Survival endpoints**: "no relapse"/"no recurrence" mapped to 1, "relapse"/"recurrence" mapped to 0. No time-to-event handling.
- **Continuous endpoints**: Not supported. All labels are forced to binary.

### 1.4 Information loss
- The native labels (e.g., ordinal RECIST categories CR/PR/SD/PD, or continuous measurements) are not preserved anywhere after download.
- The `response_grouping` and `predefined_grouping` strings are stored in the catalog CSV files but not in the per-dataset directories.
- No logging of which conversion rule was applied to each dataset.

## 2. Dataset Inventory

### 2.1 Available datasets
- 60 GSE directories exist under `data/raw/ctrdb/`.
- 40 datasets have both expression and response labels (binary, usable).
- 20 directories lack labels or expression data.
- All labels are already binary (0/1); no dataset retains native multi-category labels.

### 2.2 Catalog files
- `catalog.csv`: 15 breast-cancer datasets with columns including `response_grouping` and `predefined_grouping`.
- `pan_cancer_catalog.csv`: 673 datasets across 23 cancer types. Most have `predefined_grouping` metadata.

### 2.3 Observed endpoint types in catalogs
- **Pathologic response (pCR/RD)**: Most common. Examples: GSE25066, GSE20194, GSE20271, GSE6861, GSE4779, GSE32646, GSE50948.
- **RECIST (CR/PR/SD/PD)**: GSE82172, GSE41998, GSE66999, GSE66305.
- **RCB-based (RCB-0/I vs RCB-II/III)**: GSE25066 variant, GSE32646.
- **Relapse/recurrence (survival-derived)**: Present in fallback catalog (GSE9893, GSE17705) but those datasets lack labels on disk.
- **Pharmacodynamic/continuous**: Not present in current dataset.

## 3. Evaluation Pipeline (`src/models/validate_on_patients.py`)

### 3.1 Metrics
- AUROC (sklearn `roc_auc_score`)
- Wilcoxon rank-sum p-value (scipy `mannwhitneyu`)
- Mean score difference between responders and non-responders
- Cohen's d effect size
- All metrics assume binary labels.

### 3.2 Prediction approach
- Uses the cell-line-trained LightGBM model (trained on LINCS x GDSC2 data).
- Predicts one "general drug sensitivity" score per patient using mean drug fingerprint and standard dose.
- No per-drug-specific predictions.

### 3.3 Limitations
- Only binary evaluation (AUROC). No ordinal, continuous, or time-to-event metrics.
- No stratification by endpoint type or endpoint family.
- No per-dataset endpoint-aware evaluation.

## 4. Patient Model Training (`src/models/train_patient_model.py`)

### 4.1 Architecture
- LightGBM classifier trained on pooled CTR-DB breast-cancer datasets.
- Features: L1000 landmark gene z-scores (z-scored within each dataset).
- Labels: Binary response.

### 4.2 LODO-CV
- Leave-one-dataset-out cross-validation.
- Metrics: AUROC, balanced accuracy, sensitivity, specificity.
- Skip conditions: held-out dataset needs >= 10 samples, >= 3 in each class.
- Conservative hyperparameters: `n_estimators=200`, `learning_rate=0.05`, `max_depth=5`, `class_weight=balanced`.

### 4.3 No endpoint awareness
- All datasets treated identically regardless of whether the underlying endpoint is pCR/RD, RECIST, or survival-derived.
- No aggregation by endpoint family.

## 5. Pan-Cancer Patient Model (`src/models/train_pancancer_patient_model.py`)

### 5.1 Same structure as breast-only
- Same LightGBM LODO-CV approach but across all cancer types.
- Uses genes present in >= 80% of datasets (union approach with fill_value=0).
- Compares pan-cancer vs breast-only AUC on breast held-out datasets.

### 5.2 Response label handling
- Calls `load_ctrdb_dataset()` which reads pre-binarized labels.
- No native endpoint handling.
- Cancer type tracked per-patient for reporting but not used as a feature.

## 6. Combined Pipeline (`scripts/run_combined_pipeline.py`)

### 6.1 LODO approach
- Uses L1-logistic regression (C=0.05, solver=liblinear).
- Features: gene reversal scores, pathway reversal scores, dose-aware features.
- LODO evaluation with binary AUROC only.
- Drug matching: parses regimen components from catalog, matches to LINCS drug names.
- ComBat batch correction applied before LODO (but correction uses all datasets together, not LODO-aware).

### 6.2 Feature construction
- Gene reversal: element-wise product of patient z-scores and drug signature.
- Pathway reversal: ssGSEA pathway scores multiplied by drug pathway signature.
- Dose-aware: reversal at low/medium/high dose bins, slope, max.
- All features are continuous but outcome is binary classification.

### 6.3 No adaptation
- No few-shot adaptation, fine-tuning, or domain-specific calibration.
- All training datasets treated equally regardless of cancer type, drug, or endpoint.

## 7. Summary of Gaps

1. **Native labels discarded**: Original multi-category labels (CR/PR/SD/PD, RCB classes) are collapsed to binary at ingestion time. No recovery possible from stored data without re-downloading.
2. **No endpoint taxonomy**: No metadata tracking whether a dataset uses pathologic response, RECIST, survival, or pharmacodynamic endpoints.
3. **Single metric (AUROC)**: No endpoint-appropriate metrics (Spearman for continuous, concordance index for survival).
4. **No within-family aggregation**: LODO results not stratified by endpoint type.
5. **No adaptation pipeline**: No few-shot learning, domain adaptation, or feature calibration.
6. **SD handling is fixed**: RECIST SD is always grouped with PD as non-responders. No configurable policy.
7. **Survival endpoints conflated**: Relapse/recurrence are treated as binary without considering time-to-event information.
8. **ComBat leakage risk**: In `run_combined_pipeline.py`, ComBat correction is applied to all datasets together before LODO splits, potentially leaking held-out dataset statistics into training.
