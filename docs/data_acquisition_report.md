# Data Acquisition Report: Breaking the Chemotherapy Bias

Generated: 2026-04-05 13:16:12

## Objective

INVEREX training data was 80% chemotherapy. This acquisition adds endocrine,
targeted, immunotherapy, PARP inhibitor, and ADC response data from GEO.

## Existing Data Audit

- **CTR-DB**: 60 GEO datasets total, 18 breast cancer datasets
- **CDS-DB**: 4 GEO datasets (paired pre/post treatment)
- **Breast cancer treatment composition before acquisition**:

| Treatment Class | N patients | % |
|-----------------|-----------|---|
| chemotherapy | 1975 | 78.4% |
| endocrine | 345 | 13.7% |
| targeted | 199 | 7.9% |
| **TOTAL** | 2519 | 100% |

## Datasets Successfully Downloaded

### Tier 1 (Highest Priority)

#### A. I-SPY2 (GSE194040) - THE key dataset
- **988 patients**, 19,134 genes (Agilent microarray)
- **14 treatment arms** including:
  - Paclitaxel control (179)
  - Targeted: Neratinib (114), Ganitumab (107), Ganetespib (93), MK-2206 (94), AMG 386 (134), Trastuzumab combos (75)
  - PARP inhibitor: Veliparib + Carboplatin (71)
  - Immunotherapy: Pembrolizumab (69)
  - ADC: T-DM1 + Pertuzumab (52)
- **Response**: pCR (binary: 0/1), 319 pCR vs 669 RD
- **Status**: Expression + response labels saved
- Files: `data/raw/ispy2/GSE194040_expression.parquet`, `data/raw/ispy2/response_labels.parquet`

#### B. BrighTNess (GSE164458) - TNBC + PARP
- **482 patients**, 24,613 genes (RNA-seq)
- **3 arms**: Paclitaxel +/- Carboplatin +/- Veliparib
- **Response**: pCR and RCB class (0-3)
- Files: `data/raw/brightness/GSE164458_expression.parquet`, `data/raw/brightness/response_labels.parquet`

#### C. Hurvitz HER2+ (GSE130788) - ADC
- **199 samples**, 41,000 features (microarray)
- HER2+ breast cancer treated with T-DM1 + Pertuzumab
- **Limitation**: No response labels in GEO metadata (expression only)
- Files: `data/raw/hurvitz_her2/GSE130788_expression.parquet`

### Additional Endocrine Datasets

#### GSE59515 - Letrozole response prediction
- **75 patients**, 47,323 features (Illumina BeadArray)
- Pre-treatment, 2-week, and 3-month timepoints
- **Response**: Responder vs Non-responder (explicit labels)
- Files: `data/raw/endocrine_response_letrozole/GSE59515_expression.parquet`

#### GSE87411 - TransNEOS/Z1031 paired endocrine
- **218 samples**, 29,032 features (Agilent)
- Paired pre/post aromatase inhibitor treatment
- Patient IDs match Z1031 trial
- Files: `data/raw/transneos/GSE87411_expression.parquet`

#### GSE78958 - Z1031 trial baseline
- **424 patients**, 22,277 features (Affymetrix U133A)
- Baseline tumor expression with molecular subtype
- Treatment: neoadjuvant aromatase inhibitor
- Files: `data/raw/z1031/GSE78958_expression.parquet`

#### GSE145325 - Letrozole RNA-seq
- **58 patients**, 58,652 genes (RNA-seq)
- ER+ breast cancer: **responder vs nonresponder** (explicit labels)
- Files: `data/raw/letrozole_rnaseq/GSE145325_expression.parquet`

#### GSE111563 - Endocrine resistance
- **101 patients**, 22,119 features (Illumina)
- Dormant vs acquired resistant ER+ breast tumors
- Files: `data/raw/endocrine_resistance_er/GSE111563_expression.parquet`

### Additional Immunotherapy/PARP Datasets

#### GSE173839 - Durvalumab + Olaparib (I-SPY2 arm)
- **105 patients**, 32,146 features (Agilent)
- **Arms**: Durvalumab/Olaparib vs Control
- **Response**: pCR (0/1), also MP (metaplastic)
- Files: `data/raw/durva_olap_breast/GSE173839_expression.parquet`

#### GSE157284 - PD-L1 in TNBC
- **82 patients**, 54,675 features (Affymetrix U133 Plus 2.0)
- PD-L1 SP142 positive vs negative (immunotherapy biomarker)
- Files: `data/raw/pdl1_tnbc/GSE157284_expression.parquet`

### Additional Mixed/Response Datasets

#### GSE191127 - Hoogstraat pre/post NAC (RNA-seq)
- **46 samples** (23 patients, pre/post), 39,116 genes
- Neoadjuvant Response Index (NRI)
- Files: `data/raw/hoogstraat_1/GSE191127_expression.parquet`

#### GSE192341 - Hoogstraat NAC cohort (RNA-seq)
- **93 samples**, 42,876 genes
- pCR and NRI labels, molecular subtype
- Files: `data/raw/hoogstraat_2/GSE192341_expression.parquet`

#### GSE240671 - Neoadjuvant chemo + adjuvant endocrine (RNA-seq)
- **122 samples**, 58,243 genes
- RCB class (no residual / minimal / moderate / extensive)
- Hormone therapy type (Femara, Nolvadex, Arimidex, etc.)
- Files: `data/raw/neoadj_letrozole_rnaseq/GSE240671_expression.parquet`

## Failed/Skipped Datasets

| Dataset | Accession | Issue |
|---------|-----------|-------|
| IMPACT endocrine | GSE36339 | **Wrong dataset** - Sparus aurata (fish) study, not breast cancer |

## Treatment Composition: Before vs After

| Treatment Class      | Before (n)   | Before (%)   | After (n)    | After (%)    |
|----------------------|--------------|--------------|--------------|--------------|
| adc                  | 0            | 0.0        % | 251          | 4.6        % |
| chemotherapy         | 1975         | 78.4       % | 2681         | 48.7       % |
| endocrine            | 345          | 13.7       % | 1270         | 23.1       % |
| immunotherapy        | 0            | 0.0        % | 204          | 3.7        % |
| parp_inhibitor       | 0            | 0.0        % | 284          | 5.2        % |
| targeted             | 199          | 7.9        % | 816          | 14.8       % |
| **TOTAL**            | 2519         | 100.0%       | 5506         | 100.0%       |

## Key Improvements

1. **Immunotherapy**: 0% -> 3.7% (204 patients)
   - Pembrolizumab (I-SPY2), Durvalumab+Olaparib, PD-L1 TNBC biomarker
2. **PARP inhibitor**: 0% -> 5.2% (284 patients)
   - Veliparib (I-SPY2, BrighTNess), Olaparib (GSE173839)
3. **ADC**: 0% -> 4.6% (251 patients)
   - T-DM1 + Pertuzumab (I-SPY2, GSE130788)
4. **Endocrine**: 13.7% -> 23.1% (1270 patients)
   - Letrozole, Anastrozole, Tamoxifen, Fulvestrant, Exemestane
5. **Targeted**: 7.9% -> 14.8% (816 patients)
   - Neratinib, Trastuzumab, Pertuzumab, MK-2206, Ganitumab, etc.
6. **Chemotherapy**: 78.4% -> 48.7% (reduced from dominant majority)

## Files Created (no existing files modified)

### New data directories:
- `data/raw/ispy2/` - I-SPY2 multi-arm trial
- `data/raw/brightness/` - BrighTNess PARP trial
- `data/raw/hurvitz_her2/` - HER2+ ADC
- `data/raw/transneos/` - TransNEOS paired endocrine
- `data/raw/z1031/` - Z1031 endocrine baseline
- `data/raw/hoogstraat_1/` - Hoogstraat pre/post NAC
- `data/raw/hoogstraat_2/` - Hoogstraat NAC cohort
- `data/raw/durva_olap_breast/` - Durvalumab+Olaparib
- `data/raw/endocrine_response_letrozole/` - Letrozole response
- `data/raw/endocrine_resistance_er/` - Endocrine resistance
- `data/raw/letrozole_rnaseq/` - Letrozole RNA-seq
- `data/raw/pdl1_tnbc/` - PD-L1 TNBC
- `data/raw/neoadj_letrozole_rnaseq/` - Neoadjuvant + endocrine

### New metadata:
- `data/metadata/all_datasets_inventory.tsv`

### Documentation:
- `docs/data_acquisition_report.md` (this file)

### Scripts (all new):
- `scripts/download_treatment_diversity_data.py`
- `scripts/download_supplementary_expression.py`
- `scripts/fix_and_complete_downloads.py`
- `scripts/download_additional_datasets.py`
- `scripts/process_rnaseq_supplements.py`
- `scripts/fix_gse240671.py`
- `scripts/build_final_inventory.py`
