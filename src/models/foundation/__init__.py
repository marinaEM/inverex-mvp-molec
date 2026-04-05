"""
INVEREX Foundation Model: Biologically structured expression encoder.

Modules:
    gene_universe       – discover top-2000 genes present in >=50% of datasets
    expression_tokenizer – rank + magnitude encoding per sample
    expression_encoder  – gene-aware transformer → [CLS] patient embedding
    pretrain_objectives – masked-gene, pathway-activity, subtype prediction
    pretrain            – self-supervised pretraining loop
    evaluate            – LODO fine-tune + comparison vs LightGBM baseline
"""
