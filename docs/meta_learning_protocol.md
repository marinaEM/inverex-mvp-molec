# Few-Shot LODO Meta-Learning Protocol

## Overview

This document describes the few-shot leave-one-dataset-out (LODO) evaluation
protocol for testing domain adaptation methods on CTR-DB clinical trial datasets.

## Motivation

Standard LODO trains on N-1 datasets and evaluates on the held-out dataset
with zero knowledge of the target domain.  In practice, a small number of
labeled samples from the target domain may be available (e.g., from a pilot
cohort or early responders).  Few-shot adaptation tests whether methods can
leverage k labeled target samples to improve predictions on the remaining
target samples.

## Protocol

### Definitions
- **Source datasets**: All CTR-DB datasets except the held-out dataset.
- **Held-out dataset**: The target dataset for evaluation.
- **Support set**: k labeled samples drawn from the held-out dataset (used for adaptation).
- **Query set**: Remaining held-out samples (used for evaluation only).

### Procedure

For each held-out dataset D_i with at least 30 samples:

1. **Source training**: Train the base model on all source datasets (all datasets except D_i).
2. **Support/query split**: For each support size k in {0, 5, 10, 20}:
   a. For each repeat r in {1, ..., 5} (different random seeds):
      - Draw k samples from D_i using stratified sampling (preserving class ratio).
      - The remaining samples form the query set.
      - **k=0 is the zero-shot baseline** (no adaptation; evaluate source model directly on query).
3. **Adaptation**: Apply each adaptation method using only the support set.
4. **Evaluation**: Compute metrics on the query set.
5. **Log**: Record (dataset, method, k, seed, metrics).

### Leakage prevention
- The held-out dataset is never included in source training data.
- The support set is never included in the query set.
- Batch correction (if any) does not use held-out dataset statistics.
- Feature standardization is fit on source training data only; support and query are transformed using source statistics.

### Methods evaluated

1. **L1-logistic baseline (zero-shot)**: L1-penalized logistic regression (C=0.05) trained on source data only.
2. **Feature calibration**: Shift and scale source features to match support set statistics.  Lightweight; works with any classifier.
3. **NN fine-tuning**: Train a 2-layer neural network (256->128) on source data, then fine-tune on the support set with low learning rate (1e-4) for 5-20 epochs.
4. **MAML** (conditional): First-order MAML with inner loop (3-5 steps, lr=0.01) and outer loop (Adam, lr=1e-3).  Only attempted if NN fine-tuning shows improvement over zero-shot.

### Metrics
- Primary: AUROC (for binary endpoints)
- Secondary: AUPRC
- Aggregation: Mean across repeats per (dataset, method, k), then mean across datasets.

### Stop conditions
- If NN zero-shot AUROC < L1-logistic zero-shot AUROC - 0.05: skip MAML.
- If NN fine-tuning does not beat NN zero-shot (mean across datasets): skip MAML.
- Datasets with fewer than 30 samples are excluded from the benchmark.

## Expected outputs

- `results/fewshot_by_dataset.tsv`: Per-dataset, per-method, per-k, per-seed results.
- `results/fewshot_summary.tsv`: Aggregated results (mean/std across datasets and seeds).
- `results/fewshot_by_endpoint.tsv`: Results stratified by endpoint family.
