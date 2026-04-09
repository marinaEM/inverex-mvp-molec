"""
INVEREX Foundation Model (Full)
==========================================
Comprehensive gene-aware expression encoder with multi-objective pretraining,
domain-adversarial training, input-encoding ablations, and rigorous LODO evaluation.

Modules:
    gene_universe        -- discover 5K-8K genes present in >=40% of datasets
    expression_tokenizer -- three input encodings: raw, rank, hybrid (ablatable)
    expression_encoder   -- gene-aware transformer (medium: 2M, full: 5M params)
    pretrain             -- multi-objective pretraining with domain adversarial
    evaluate             -- LODO evaluation, ablations, embedding analysis, benchmarking
"""
