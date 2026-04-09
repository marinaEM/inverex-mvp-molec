# Foundation Model: Training Data Documentation

## Data Sources

### 1. CTR-DB (Chemotherapy Response Database)
- **Location**: `data/raw/ctrdb/GSE*/`
- **Datasets**: 62 GEO directories, 36+ with expression parquet files
- **Content**: Microarray gene expression from neoadjuvant chemotherapy trials in breast cancer
- **Use in pretraining**: All expression data (self-supervised, no response labels)
- **Use in LODO**: Datasets with `response_labels.parquet` used for fine-tuning and evaluation
- **Response**: pCR (pathological complete response) vs non-pCR (binary)

### 2. I-SPY2
- **Location**: `data/raw/ispy2/GSE194040_expression.parquet`
- **Samples**: 988 patients
- **Content**: Gene expression from multi-arm adaptive neoadjuvant trial
- **Response labels**: `data/raw/ispy2/response_labels.parquet`
- **Use in pretraining**: Expression data (self-supervised)
- **Use in LODO**: Fine-tuning and evaluation

### 3. BrighTNess
- **Location**: `data/raw/brightness/GSE164458_expression.parquet`
- **Samples**: 482 patients
- **Content**: Gene expression from triple-negative breast cancer trial (DNA-damage regimen)
- **Response labels**: `data/raw/brightness/response_labels.parquet`
- **Use in pretraining**: Expression data (self-supervised)
- **Use in LODO**: Fine-tuning and evaluation

### 4. TCGA-BRCA
- **Location**: `data/cache/tcga_brca_expression.parquet`
- **Samples**: 1,218 patients
- **Content**: RNA-seq gene expression (20,530 genes)
- **Response labels**: NONE (no treatment response labels)
- **Use in pretraining**: Expression data only (self-supervised)
- **Use in LODO**: NOT used (no response labels)
- **Additional data used**:
  - `data/cache/tcga_brca_mutations.parquet`: Somatic mutation calls for TP53, PIK3CA, ERBB2 (used as mutation-proxy pretraining labels)
  - Genes with mutations: PIK3CA (355), TP53 (304), CDH1 (114), GATA3 (101), MAP3K1 (98), PTEN (37), ERBB2 (22), BRCA2 (17), BRCA1 (13), ESR1 (5), AKT1 (2)

## Gene Universe

- **Threshold**: Genes present in >= 40% of all 52 datasets (>= 20 datasets)
- **Size**: 1,000 genes (capped for CPU feasibility with transformer attention)
- **Selection**: Priority genes forced first (50 key breast cancer genes), then by prevalence
- **Priority genes include**: ESR1, PGR, ERBB2, EGFR, MKI67, TP53, PIK3CA, BRCA1, BRCA2, GATA3, MAP3K1, PTEN, AURKA, TOP2A, BCL2, BIRC5, GRB7, FOXA1, FOXC1, KRT5, KRT14, KRT17, and more
- **Note**: All 1,000 selected genes are present in >= 48/52 datasets (very high prevalence)
- **Comparison**: Mini pilot used 2,000 genes but without priority selection, missing key genes like MKI67/TP53

### Why not more genes?
The transformer self-attention complexity is O(n^2) in sequence length. On CPU:
- 500 genes: ~1.6s/batch -- feasible
- 1,000 genes: ~5s/batch -- practical for comprehensive training
- 2,000 genes: ~25s/batch -- too slow for ablations
- 3,000 genes: ~60s/batch -- impractical

With 1,000 well-selected genes covering all major breast cancer pathways, the biological coverage is comprehensive.

## What Is Used for What

### Pretraining (self-supervised, NO response labels)
All expression data from all sources:
- CTR-DB: All datasets with expression parquet files
- I-SPY2: 988 samples
- BrighTNess: 482 samples
- TCGA-BRCA: 1,218 samples
- **Total**: ~7,000-8,000 samples (varies by gene overlap threshold)

Pretraining uses ONLY gene expression values. **Response labels are never seen during pretraining.**

### Auxiliary pretraining labels (derived from expression, NOT from response)
- **Pathway activity scores**: Computed via mean-rank of MSigDB Hallmark 2020 gene sets
- **Inferred subtypes**: Derived from ESR1/ERBB2/MKI67 expression (not clinical PAM50): LumA (ESR1-high, MKI67-low), LumB (ESR1-high, MKI67-high), HER2+ (ERBB2-high), Basal (ESR1-low, ERBB2-low)
- **Mutation status**: TP53/PIK3CA/ERBB2 from TCGA mutation calls only (1,068 samples)
- **Dataset identity**: Integer label per source dataset (for domain adversarial objective)

### LODO evaluation (response labels used here)
Relaxed LODO protocol:
1. Pretrain encoder on ALL data (self-supervised) -- done once
2. For each held-out dataset:
   a. Fine-tune classifier head on all OTHER labeled datasets (using response labels)
   b. Evaluate on the held-out dataset

TCGA-BRCA is NEVER in LODO evaluation (no treatment response labels).

## Landmark Genes (for baselines)
- **Source**: `data/cache/geneinfo_beta_input.txt` (L1000 landmark genes)
- **Count**: ~212 genes after filtering
- **Use**: LightGBM baseline comparisons

## Data Integrity Notes
- Expression values are log2(x+1) transformed during tokenization
- Per-gene mean/std computed across ALL pretraining samples for z-scoring
- Missing genes handled with presence=0 flag (hybrid encoding)
- Response labels binary: 1 = responder (pCR), 0 = non-responder
- No data augmentation
- Train/test split at dataset level (LODO), never at sample level
- TCGA mutations: sample is labeled WT (0) unless a mutation is found in the calls file
