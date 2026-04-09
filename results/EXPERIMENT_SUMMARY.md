# INVEREX Model Experiment Summary (2026-04-06)

## What the model does

The current model predicts **binary drug response** (responder vs non-responder) from **patient gene expression only**. It is a general treatment-responsiveness classifier trained on 38 CTR-DB datasets spanning multiple drugs/regimens. It does NOT take drug identity as input and therefore cannot rank specific drugs for a patient. Per-patient-per-drug SHAP explanations are not possible with this architecture.

## Training data

- **38 datasets**, **5,200 patients** (after excluding 1 targeted panel with 102 genes and 1 BeadArray dataset with no L1000 gene overlap)
- **918 L1000 genes** passing cross-platform availability filter (present in >=80% of datasets within each technology group)
- **3 technology platforms**: Affymetrix microarray (24 datasets, ~3,100 pts), Agilent microarray (12 datasets, ~1,800 pts), RNA-seq (2 datasets, ~530 pts)
- **Response labels**: pCR, clinical response, relapse-free survival — heterogeneous across datasets

## The 0.761 was fake

The original "best" result (0.761 AUROC) was entirely due to data leakage. A controlled experiment isolating each variable showed:

| Configuration (212 curated genes) | AUROC | What it measures |
|-----------------------------------|-------|------------------|
| ComBat WITH response labels as covariate (leaked) | 0.763 | Label leakage through batch correction |
| ComBat WITHOUT labels | 0.594 | Batch correction alone |
| Quantile normalization (no ComBat) | 0.594 | Honest baseline |
| Supervised ComBat (training labels only, mean imputation) | 0.566 | Worse than no labels |
| Supervised ComBat (training labels only, two-step) | 0.567 | Worse than no labels |

**The +0.169 AUROC "gain" was entirely from passing response labels to neuroCombat as a continuous covariate (`continuous_cols=["response"]`).** This let ComBat's design matrix learn which genes separate responders from non-responders, then encode that separation into the corrected expression values. The model just picked up the pre-separated signal.

ComBat batch correction WITHOUT labels provides zero benefit over quantile normalization. Supervised ComBat (training-only labels) actually hurts by creating a train-test distribution mismatch.

## The honest model

### Best production config: `rank_genes + singscore + REO (knowledge-driven), platform-agnostic`

**Normalization**: Within-sample rank + inverse-normal transform (per-sample, no cross-sample fitting, deployable on a single new patient)

**Features (970 total)**:
- 918 rank-normalized gene expression values
- 47 singscore pathway scores (rank-based, per-sample)
- 5 curated gene-pair REO features (ERBB2>ESR1, MKI67>ACTB, CCND1>CDKN1A, AKT1>PTEN, CASP3>BCL2)

**Production threshold**: 0.376 (tuned via 5-fold inner-CV MCC optimization on training data, frozen before test evaluation)

### Standard LODO evaluation (38 datasets, leave-one-dataset-out)

All configs are leakage-free, per-sample deployable, with MCC-optimal threshold tuned on training only.

| Config | AUROC | AUPRC | MCC | Bal. Acc |
|--------|-------|-------|-----|---------|
| **rank + singscore + REO** | **0.602** | **0.619** | **0.157** | 0.588 |
| rank genes only | 0.600 | 0.606 | 0.163 | 0.592 |
| rank + ssGSEA | 0.595 | 0.613 | 0.146 | 0.577 |
| rank + singscore | 0.595 | 0.615 | 0.138 | 0.579 |
| rank + all pathways | 0.595 | 0.610 | 0.139 | 0.573 |
| rank + platform covariate | 0.593 | 0.613 | 0.136 | 0.572 |
| raw expression (no rank) | 0.578 | 0.612 | 0.138 | 0.570 |
| singscore only (no genes) | 0.577 | 0.583 | 0.112 | 0.559 |
| ssGSEA only (no genes) | 0.544 | 0.558 | 0.053 | 0.528 |

**Key findings**:
- Rank normalization improves over raw expression (+0.02 AUROC)
- Pathway features add marginal value (+0.002 AUROC with singscore+REO)
- Platform covariate hurts (-0.007 AUROC) — rank normalization already handles platform differences
- ssGSEA alone is near-random; singscore alone is marginally better

### Cross-technology transfer evaluation

| Evaluation | Config | AUROC | MCC |
|-----------|--------|-------|-----|
| **Microarray → RNA-seq** (Variant A) | rank genes | **0.614** | 0.149 |
| **BrighTNess RNA-seq holdout** (Variant C) | rank+all_pathways | **0.611** | 0.127 |
| **GSE104958 RNA-seq holdout** (Variant C) | rank+singscore+REO | **0.767** | 0.366 |
| Affymetrix → Agilent (Variant B) | all configs | ~0.49 | ~0.00 |

RNA-seq transfer works (0.60-0.77). Affymetrix-to-Agilent transfer fails completely (~random).

## SHAP feature importance (global, final model)

Top 15 features by mean |SHAP|:

| Rank | Feature | |SHAP| | Direction | Biology |
|------|---------|--------|-----------|---------|
| 1 | CCND1 | 0.106 | Resistance | Cyclin D1, cell cycle driver, CDK4/6i target |
| 2 | HMGA2 | 0.047 | Resistance | Chromatin remodeling, EMT marker |
| 3 | FRS2 | 0.047 | Resistance | FGFR signaling adaptor |
| 4 | AKT1 | 0.043 | Resistance | PI3K/AKT pathway, survival signaling |
| 5 | CYB561 | 0.043 | Response | Electron transport, iron metabolism |
| 6 | CEBPD | 0.041 | Response | Inflammatory differentiation |
| 7 | TCEAL4 | 0.037 | Resistance | Transcription elongation |
| 8 | KIF5C | 0.034 | Response | Kinesin motor protein |
| 9 | FOXO3 | 0.034 | Resistance | Tumor suppressor / apoptosis |
| 10 | ABCB6 | 0.033 | Resistance | ABC transporter (drug efflux family) |
| 11 | HOXA10 | 0.032 | Response | Homeobox transcription factor |
| 12 | GATA3 | 0.032 | Response | Luminal differentiation marker |
| 13 | singscore_Spermatogenesis | 0.032 | Resistance | Pathway activity score |
| 14 | ZW10 | 0.031 | Response | Kinetochore/mitotic checkpoint |
| 15 | SPAG4 | 0.031 | Response | Nuclear envelope, sperm-associated |

Gene expression features dominate (sum=5.78), singscore pathways contribute modestly (sum=0.26), REO pairs contribute minimally (sum=0.003).

## Critical limitation

**This model does not include drug features.** It predicts "will this patient respond to treatment?" not "which drug should this patient get?" The SHAP explanations are global — they show which genes drive treatment responsiveness in general, not why a specific drug is ranked higher than another for a specific patient.

The original INVEREX design included drug features (LINCS L1000 signatures + ECFP4 fingerprints from GDSC2), enabling per-patient drug ranking with drug-specific SHAP. That cell-line-based model achieved 0.47 AUROC. The patient-expression-only model achieves 0.60 but loses the drug-specificity that makes per-drug ranking meaningful.

## Production artifacts

```
models/production/
  cross_platform_model_bundle.joblib  — full bundle (model + genes + features + threshold + conformal)
  lightgbm_model.joblib               — standalone LightGBM model
  gene_list.joblib                     — 918 L1000 genes
  feature_names.joblib                 — 970 feature names

results/cross_platform/
  all_results.tsv                      — per-fold, per-config, 5 metrics
  summary_all_configs.tsv              — aggregated summary
  production_config_selection.tsv      — ranked production candidates
  shap_final_model.tsv                 — 970 features ranked by SHAP
  shap_beeswarm_final.png             — top-30 beeswarm visualization
  technology_distribution.tsv          — platform breakdown
  experiment.log                       — full experiment log
```
