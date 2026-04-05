# Agent C: Mini vs Full Foundation Model

## Overview

The INVEREX foundation model has two versions: the **mini pilot** (Agent C-mini) and the **full model** (Agent C-full). This document explains why the mini pilot is insufficient and what the full model adds.

## Mini Pilot (Agent C-mini)

| Parameter | Value |
|-----------|-------|
| Gene universe | 2,000 genes (top by prevalence, >=50% of datasets) |
| Model dimension | d_model=128 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Total parameters | ~500K |
| Pretraining objectives | 3 (MGP, PAP, SUB) |
| Input encoding | Hybrid only (no ablation) |
| Domain adversarial | No |
| Mutation proxy | No |
| Pathway masking | No |
| Evaluation | 10 representative datasets |

### What mini proved
- The transformer architecture can process gene expression into [CLS] embeddings
- Masked gene prediction converges on CPU
- Pathway activity prediction is feasible as a pretraining objective
- The pipeline (tokenize -> encode -> pretrain -> evaluate) works end-to-end

### What mini cannot tell us
- Whether the encoder captures enough biology to beat tabular baselines
- Whether domain adversarial training improves cross-dataset generalization
- Which input encoding works best (no ablation was run)
- Whether multi-objective pretraining helps vs masked prediction alone
- Whether mutation-proxy prediction adds useful signal
- Whether the embeddings separate subtypes better than datasets (domain invariance)

## Full Model (Agent C-full)

| Parameter | Value |
|-----------|-------|
| Gene universe | 1,000 priority-selected genes (>=40% prevalence, key BC genes forced) |
| Model dimension | d_model=256 (medium) or 384 (full) |
| Transformer layers | 4 (medium) or 6 (full) |
| Attention heads | 4 (medium) or 8 (full) |
| Total parameters | ~3.8M (medium) or ~5M+ (full) |
| Pretraining objectives | 5 (MGP, PAP, SUB, MUT, DAV) |
| Input encoding | 3 modes with ablation (raw, rank, hybrid) |
| Domain adversarial | Yes (gradient reversal with progressive alpha ramp) |
| Mutation proxy | Yes (TP53, PIK3CA, ERBB2 from TCGA) |
| Pathway masking | Yes (20% probability of masking entire pathway) |
| Evaluation | All labeled datasets (LODO) |
| Pretraining data | 8,362 samples from 50 datasets |

**Note on gene count**: Initial design called for 5,000+ genes. On CPU, transformer self-attention is O(n^2) in sequence length, making >1,000 genes impractical for multiple ablation runs. The 1,000-gene universe uses priority selection to ensure all biologically important breast cancer genes (ESR1, ERBB2, TP53, MKI67, etc.) are included. The mini pilot's 2,000 genes were selected purely by prevalence and actually missed key genes like MKI67 and TP53.

## Why Mini is Insufficient

### 1. Poor gene selection (prevalence-only vs priority-aware)
The mini model selected 2,000 genes purely by prevalence, which caused it to miss biologically critical genes like MKI67 and TP53 (present in 48/52 datasets but alphabetically sorted out). The full model uses a priority-aware selection: 50 key breast cancer genes are forced into the universe, ensuring complete coverage of ER/HER2/proliferation/EMT/immune pathways. Despite using 1,000 genes (reduced for CPU feasibility), the biological coverage is better.

### 2. Too shallow (2 layers vs 4-6)
With only 2 transformer layers, the mini model has limited capacity for learning gene-gene interactions. Biological processes involve complex multi-gene regulatory networks. The full model's 4-6 layers can capture deeper interaction patterns, similar to how deeper language models capture longer-range dependencies.

### 3. No domain invariance
The mini model has no mechanism to learn platform/dataset-invariant representations. The domain adversarial objective (gradient reversal) in the full model actively encourages the encoder to produce embeddings that are informative about biology but not about the source dataset. This is critical for cross-dataset generalization in the LODO setting.

### 4. No ablation evidence
Without systematic ablation studies, we cannot draw any conclusions about:
- Which input encoding is most effective
- Whether multi-objective pretraining helps
- Whether the foundation approach has any advantage over tabular baselines
- What the optimal model size is

The full model runs ablations across input encodings, objective combinations, model sizes, and frozen vs. fine-tuned settings.

### 5. No mutation integration
The mini model cannot predict mutation status from expression, missing an important biological validation signal. If the encoder captures biology well, it should be able to predict TP53/PIK3CA/ERBB2 mutation status from expression alone (since these mutations have known transcriptomic signatures).

### 6. Incomplete evaluation
The mini evaluated on only 10 representative datasets and compared against only 3 baselines. The full model evaluates on all labeled datasets and compares against 6+ baselines including the known Agent A result (LightGBM + mutations/clinical at AUC 0.624).

## Decision Framework

The full model produces a definitive answer to the question: **Does a gene-aware expression foundation model beat strong tabular baselines for treatment response prediction?**

If the answer is no (which is a scientifically valid and important finding), we know this with confidence because we tested multiple configurations, encodings, and objective combinations. If the answer is yes, we know which configuration works best and by how much.
