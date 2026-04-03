# Case Review: TCGA-A2-A04W-01

## Patient Summary

- PAM50 subtype: Her2
- ER / PR / HER2: Negative / Negative / Positive
- Active alterations: TP53, PIK3CA, ERBB2_amp
- Top up genes: SHH, GLI1, JUN, HES1, ACTB
- Top down genes: CCNA2, TYMS, EZH2, PARP2, MSH2

## Before Improvement: Top 15

| drug_name     |   predicted_inhibition |
|:--------------|-----------------------:|
| temsirolimus  |               100      |
| sirolimus     |                99.9497 |
| docetaxel     |                67.138  |
| paclitaxel    |                66.4422 |
| vinorelbine   |                50.4076 |
| vinblastine   |                49.0142 |
| NVP-AUY922    |                34.3334 |
| NVP-BEZ235    |                34.2757 |
| lestaurtinib  |                33.4008 |
| methotrexate  |                33.2383 |
| epirubicin    |                31.6787 |
| camptothecin  |                30.9868 |
| gemcitabine   |                30.4204 |
| GDC-0941      |                28.8678 |
| staurosporine |                28.5497 |

## After Improvement: Top 15

| drug_name    |   final_score | evidence_tier   | rationale_short                                                                |
|:-------------|--------------:|:----------------|:-------------------------------------------------------------------------------|
| lapatinib    |      0.534081 | Tier 1          | RNA 0.21; mutation/pathway 0.75; context 0.70; clinical 1.00                   |
| NVP-AUY922   |      0.440268 | Tier 1          | RNA 0.23; mutation/pathway 0.50; context 0.67; clinical 0.57; ML prior 34.3%   |
| tanespimycin |      0.35148  | Tier 1          | RNA 0.21; mutation/pathway 0.50; context 0.67; clinical 0.57; ML prior 25.6%   |
| afatinib     |      0.346662 | Tier 1          | RNA 0.12; mutation/pathway 0.50; context 0.47; clinical 0.65                   |
| NVP-BEZ235   |      0.311949 | Tier 2          | RNA 0.17; mutation/pathway 0.60; context 0.00; clinical 0.57; ML prior 34.3%   |
| paclitaxel   |      0.311449 | Tier 1          | RNA 0.17; mutation/pathway 0.00; context 0.15; clinical 1.00; ML prior 66.4%   |
| gefitinib    |      0.28213  | Tier 2          | RNA 0.21; mutation/pathway 0.25; context 0.23; clinical 0.65                   |
| buparlisib   |      0.269423 | Tier 2          | RNA 0.12; mutation/pathway 0.60; context 0.00; clinical 0.57                   |
| erlotinib    |      0.250136 | Tier 2          | RNA 0.12; mutation/pathway 0.25; context 0.23; clinical 0.65                   |
| temsirolimus |      0.227719 | Tier 2          | RNA -0.15; mutation/pathway 0.40; context 0.00; clinical 0.65; ML prior 100.0% |
| vinorelbine  |      0.225946 | Tier 1          | RNA -0.03; mutation/pathway 0.00; context 0.10; clinical 1.00; ML prior 50.4%  |
| olaparib     |      0.222822 | Tier 1          | RNA 0.15; mutation/pathway 0.00; context 0.00; clinical 1.00                   |
| MK-2206      |      0.218989 | Tier 2          | RNA 0.10; mutation/pathway 0.40; context 0.00; clinical 0.57                   |
| fluorouracil |      0.214566 | Tier 1          | RNA 0.13; mutation/pathway 0.00; context 0.00; clinical 1.00                   |
| vorinostat   |      0.21199  | Tier 2          | RNA 0.29; mutation/pathway 0.00; context 0.00; clinical 0.65                   |

## What Changed

- `lapatinib` moved from rank None to rank 1, driven by HER2-specific mutation and subtype context plus strong clinical actionability.
- `temsirolimus` moved from rank 1 to rank 10; it remains mechanistically plausible for PI3K/mTOR biology but is no longer allowed to dominate a HER2-positive case.
- `paclitaxel` moved from rank 4 to rank 6; standard cytotoxic agents remain plausible but are now behind more subtype-aware HER2 logic.
- `lestaurtinib` moved from rank 9 to rank None; weak breast-cancer relevance is now penalized.
- `MG-132` rank after improvement: None. Tool compounds are excluded from recommendation outputs.
