# INVEREX Definitive Retraining Report

## Gene Set
954 L1000 landmarks (>=80pct coverage)

## Datasets
37 datasets, 5143 patients
Excluded: GSE37138 GSE61676 GSE9782 GSE62321

## Optuna Params
{
  "n_estimators": 800,
  "num_leaves": 60,
  "max_depth": 10,
  "min_child_samples": 16,
  "learning_rate": 0.07161286789808947,
  "subsample": 0.7084225406362931,
  "colsample_bytree": 0.647667396154867,
  "reg_alpha": 0.07957571804043045,
  "reg_lambda": 0.010289206923135653,
  "objective": "binary",
  "metric": "auc",
  "random_state": 42,
  "verbose": -1
}

## LODO AUC
A (genes only): 0.5972
B (genes+clinical): 0.6103

## By Treatment
treatment_class  n_datasets  total_patients  mean_auc_A  median_auc_A  mean_auc_B  median_auc_B
          chemo          19            2362      0.6621        0.6379      0.6570        0.5848
    combination           3            1162      0.5304        0.5223      0.5791        0.5882
      endocrine           3             345      0.4585        0.4598      0.4954        0.4959
          other           7             437      0.5360        0.5060      0.5798        0.5521
           parp           1             482      0.5902        0.5902      0.6169        0.6169
       targeted           3             199      0.5553        0.5159      0.5643        0.4841
 targeted+chemo           1             156      0.5400        0.5400      0.5049        0.5049

## By Endpoint
endpoint_family  n_datasets  total_patients  mean_auc_A  median_auc_A  mean_auc_B  median_auc_B
     pathologic          22            3857      0.6625        0.6014      0.6635        0.6008
pharmacodynamic          14            1246      0.4998        0.5002      0.5218        0.5186
       survival           1              40      0.5233        0.5233      0.6774        0.6774

## Previous Comparison
Prev 212 genes: 0.617
Prev 978 genes: 0.572
This run A: 0.5972
This run B: 0.6103

## TCGA Treatability
pam50_subtype  n_patients  new_mean  new_std  new_median
        Basal         142    0.2563   0.2331      0.1782
         Her2          67    0.2604   0.2538      0.2067
         LumA         434    0.0316   0.0711      0.0087
         LumB         194    0.0497   0.1108      0.0156
       Normal         119    0.2171   0.1817      0.1655

## Top 20 Features
    feature  importance
       ORC1         146
      CCND1         144
      PTGS2         129
  ER_status         117
     PSMB10         116
       CHN1         114
HER2_status         103
       SOX2          95
     TSPAN4          94
     HOXA10          90
      PSMG1          87
      FOXJ3          87
     INSIG1          87
      CASP3          84
      KDM3A          82
    SLC25A4          82
    HERPUD1          82
      SATB1          82
      DDX42          81
     MAPK13          80
