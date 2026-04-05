# Agent C Results Summary: Gene-Aware Expression Foundation Model

## Architecture Description

### Gene Universe
- **1,000 priority-selected genes** from 52 datasets (>= 40% prevalence threshold)
- Priority gene selection ensures coverage of key breast cancer biology:
  - ER pathway: ESR1, PGR, FOXA1
  - HER2 pathway: ERBB2, GRB7
  - Proliferation: MKI67, AURKA, AURKB, TOP2A, CCND1, CCNE1, CDK4, CDK6
  - Tumor suppressors: TP53, BRCA1, BRCA2, PTEN, RB1
  - Signaling: PIK3CA, AKT1, MTOR, EGFR, STAT3, NOTCH1
  - EMT: VIM, SNAI1, SNAI2, ZEB1, TWIST1
  - Basal markers: KRT5, KRT14, KRT17, FOXC1
  - Immune: PDCD1 (PD-1), CD274 (PD-L1), CTLA4

### Model Configurations

| Config | d_model | n_layers | n_heads | FFN dim | Total params | Encoder params |
|--------|---------|----------|---------|---------|-------------|----------------|
| medium | 256 | 4 | 4 | 1024 | ~3.77M | ~3.50M |
| full | 384 | 6 | 8 | 1536 | ~5M+ | ~4.5M+ |

Primary configuration: **medium** (selected for CPU feasibility).

### Input Encoding: Hybrid (primary)
Three modes tested in ablation:
1. **Raw**: z-scored log2(expr+1) per gene
2. **Rank**: within-sample percentile rank (0-1)
3. **Hybrid** (primary): rank + binned magnitude (64 bins) + gene-presence flag

### Pretraining Objectives (5 total, all modular)
| Objective | Weight | Description |
|-----------|--------|-------------|
| MGP (Masked Gene Prediction) | 1.0 | Reconstruct held-out gene values from context |
| PAP (Pathway Activity Prediction) | 0.5 | Predict Hallmark pathway scores from [CLS] |
| SUB (Subtype Prediction) | 0.5 | Predict inferred PAM50 subtype from [CLS] |
| MUT (Mutation Proxy) | 0.3 | Predict TP53/PIK3CA/ERBB2 status (TCGA only) |
| DAV (Domain Adversarial) | 0.2 | Gradient reversal against dataset identity |

## Training Protocol

- **Data**: 8,362 samples across 50 datasets (CTR-DB + I-SPY2 + BrighTNess + TCGA-BRCA)
- **Optimizer**: AdamW, lr=1e-4, weight_decay=0.01
- **Schedule**: Cosine with 3-epoch warmup
- **Masking**: 15% of genes masked per sample, 20% probability of pathway-wise masking
- **Batch size**: 32
- **Gradient clipping**: max_norm=1.0
- **Domain adversarial alpha**: Progressive ramp from 0 to 1 over training
- **Time limit**: 2 hours per pretraining run (CPU constraint)
- **Checkpointing**: Every 10 epochs

## CPU Feasibility Analysis

The transformer self-attention is O(n^2) in sequence length. On this Mac CPU:
- 500 genes + d=256 + 4 layers: ~1.6s/batch
- 1,000 genes + d=256 + 4 layers: ~5s/batch (~22 min/epoch)
- 2,000 genes + d=256 + 4 layers: ~25s/batch (~90 min/epoch)

With 1,000 genes and CPU contention from concurrent processes, observed training speed was ~10s/batch (~44 min/epoch). The 2-hour time limit allowed ~2-3 epochs. This is acknowledged as a significant limitation -- a GPU would allow 50+ epochs with 5,000+ genes in the same time.

## Benchmark Results

*Results below are from the model that completed within the 2-hour CPU time limit. See `results/agent_c_full/` for raw data.*

### LODO Evaluation Summary

| Method | Mean AUC | Median AUC | Source |
|--------|----------|------------|--------|
| A. LightGBM 212 genes | 0.610 | - | Known baseline |
| B. LightGBM 212 + pathways | 0.603 | - | Known baseline |
| C. LightGBM 212 + mutations/clinical | 0.624 | - | Known (Agent A) |
| D. Foundation [CLS] -> MLP | TBD | TBD | This run |
| E. LightGBM 212 + [CLS] | TBD | TBD | This run |
| F. Foundation [CLS] + mutations/clinical | TBD | TBD | This run |

*Note: Results populated after pretraining completes. Run `pixi run python src/models/foundation_full/evaluate.py` to generate.*

### Ablation Results

#### Input Encoding
| Encoding | Mean AUC (D) | Mean AUC (E) |
|----------|-------------|-------------|
| Raw | TBD | TBD |
| Rank | TBD | TBD |
| Hybrid | TBD | TBD |

#### Pretraining Objectives
| Objectives | Mean AUC (D) | Mean AUC (E) |
|------------|-------------|-------------|
| MGP only | TBD | TBD |
| Multi (no DAV) | TBD | TBD |
| Multi + DAV | TBD | TBD |

#### Model Size
| Size | Params | Mean AUC (D) |
|------|--------|-------------|
| Medium | ~3.8M | TBD |
| Full | ~5M+ | TBD |

#### Frozen vs Fine-tuned
| Mode | Mean AUC |
|------|----------|
| Frozen encoder + MLP | TBD |
| Fine-tuned encoder + MLP | TBD |

## Embedding Quality Analysis

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Silhouette (subtype) | TBD | Higher = better subtype separation |
| Silhouette (dataset) | TBD | Lower = better domain invariance |
| Linear probe: subtype accuracy | TBD | Biological signal captured |
| Linear probe: TP53 AUC | TBD | Mutation signature captured |
| Linear probe: dataset accuracy | TBD | Lower = better domain invariance |
| PCA 10-component variance | TBD | Embedding dimensionality |

## Decision Statement

**Does the foundation model beat strong tabular baselines?**

*This section will be completed with honest assessment after results are available.*

Key considerations:
1. The foundation model trains on the same data used for baselines, so any advantage must come from better representation learning
2. With only 1,000 genes (vs. 212 for the landmark baseline), the foundation model has access to more biological information
3. The domain adversarial objective should help cross-dataset generalization
4. CPU limitations mean the model has had limited training (2-3 epochs vs. ideal 50+)
5. If the foundation model does not beat LightGBM, this is an important negative result that suggests tabular methods may be sufficient for this data size and task

## Limitations

1. **CPU-only training**: Transformer models benefit enormously from GPU acceleration. The 2-hour CPU time limit severely constrains training depth. Results should be interpreted as "what is achievable on CPU" rather than "what the architecture can do."
2. **Gene count reduced from spec**: The specification called for 5,000+ genes; we use 1,000 due to O(n^2) attention cost on CPU. The 1,000 genes are priority-selected for breast cancer biology.
3. **Concurrent CPU contention**: Another training process was running simultaneously, further reducing effective training speed by ~50%.
4. **Limited ablation epochs**: Ablation runs use fewer epochs than the main model, which may not capture the full effect of each configuration change.
5. **Relaxed LODO**: The encoder sees all data during pretraining (including held-out expression, but NOT response labels). This is the standard protocol for self-supervised pretraining but means embeddings are not fully out-of-distribution.

## How to Run

```bash
# Full pipeline (pretraining + evaluation + ablations)
pixi run python src/models/foundation_full/run_all.py

# Just evaluation (requires pretrained model)
pixi run python src/models/foundation_full/evaluate.py

# Just pretraining
pixi run python src/models/foundation_full/pretrain.py
```

## Files

### Source Code
- `src/models/foundation_full/__init__.py` -- Module init
- `src/models/foundation_full/gene_universe.py` -- Gene universe discovery
- `src/models/foundation_full/expression_tokenizer.py` -- Three input encodings
- `src/models/foundation_full/expression_encoder.py` -- Transformer encoder + all objective heads
- `src/models/foundation_full/pretrain.py` -- Multi-objective pretraining
- `src/models/foundation_full/evaluate.py` -- LODO evaluation, ablations, embedding analysis
- `src/models/foundation_full/run_all.py` -- Full pipeline runner

### Results
- `results/agent_c_full/pretrain_history*.csv` -- Training loss curves
- `results/agent_c_full/lodo_results.csv` -- LODO evaluation per dataset
- `results/agent_c_full/ablation_input_encoding.csv` -- Encoding ablation
- `results/agent_c_full/ablation_objectives.csv` -- Objectives ablation
- `results/agent_c_full/ablation_model_size.csv` -- Model size ablation
- `results/agent_c_full/ablation_frozen_vs_finetuned.csv` -- Frozen/finetuned ablation
- `results/agent_c_full/embedding_analysis.csv` -- Embedding quality metrics
- `results/agent_c_full/comparison_vs_baselines.csv` -- Final comparison table

### Documentation
- `docs/agent_c_mini_vs_full.md` -- Mini vs full comparison
- `docs/agent_c_training_data.md` -- Training data documentation
- `docs/agent_c_results_summary.md` -- This document
