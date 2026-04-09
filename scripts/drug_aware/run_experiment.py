#!/usr/bin/env python
"""
INVEREX — Drug-Aware Modeling Sprint
=====================================

Adds drug features (ECFP4, drug-target interaction) to the clean
patient expression model. Tests whether drug identity improves
prediction beyond the general responsiveness baseline (0.602).

PHASE 0: GSE104958 investigation
PHASE 1: Drug feature construction
PHASE 2: LODO experiment grid (10 configs)
PHASE 3: Per-drug analysis

All outputs to results/drug_aware/.
"""

import os
import sys
import time
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "drug_aware"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "experiment.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

LGBM_PARAMS = {
    "objective": "binary", "metric": "auc", "n_estimators": 300,
    "num_leaves": 31, "max_depth": 5, "min_child_samples": 10,
    "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 2.0, "random_state": 42, "verbose": -1,
}
CHUNK_SIZE = 500

# ── Dataset → Drug mapping (curated from GEO metadata + CTR-DB catalog) ──
DATASET_DRUG_MAP = {
    "GSE25066":  "anthracycline+taxane",
    "GSE20194":  "cyclophosphamide+doxorubicin+fluorouracil+paclitaxel",
    "GSE20271":  "cyclophosphamide+doxorubicin+fluorouracil",
    "GSE22093":  "taxane+anthracycline",
    "GSE23988":  "docetaxel",
    "GSE37946":  "trastuzumab+chemotherapy",
    "GSE41998":  "cyclophosphamide+doxorubicin+ixabepilone",
    "GSE5122":   "tipifarnib",
    "GSE8970":   "tipifarnib",
    "GSE131978": "platinum_chemotherapy",
    "GSE20181":  "letrozole",
    "GSE14615":  "doxorubicin+mercaptopurine+methotrexate+prednisone+vincristine",
    "GSE14671":  "imatinib",
    "GSE19293":  "melphalan",
    "GSE28702":  "fluorouracil+leucovorin+oxaliplatin",
    "GSE32646":  "cyclophosphamide+epirubicin+fluorouracil+paclitaxel",
    "GSE35640":  "immunotherapy_peptide_vaccine",
    "GSE48905":  "tamoxifen",
    "GSE50948":  "doxorubicin+paclitaxel+cyclophosphamide+fluorouracil+methotrexate+trastuzumab",
    "GSE63885":  "platinum_chemotherapy",
    "GSE68871":  "bortezomib+thalidomide+dexamethasone",
    "GSE72970":  "chemotherapy+targeted",
    "GSE73578":  "glucocorticoids",
    "GSE82172":  "tamoxifen",
    "GSE104645": "fluorouracil+leucovorin+oxaliplatin",
    "GSE104958": "cisplatin+docetaxel+fluorouracil",
    "GSE109211": "sorafenib",
    "GSE173263": "rituximab+cyclophosphamide+doxorubicin+vincristine+prednisolone",
    "GSE21974":  "docetaxel",
    "GSE44272":  "trastuzumab",
    "GSE4779":   "cyclophosphamide+epirubicin+fluorouracil",
    "GSE65021":  "cetuximab+platinum",
    "GSE66999":  "docetaxel+epirubicin+pegfilgrastim",
    "GSE6861":   "cyclophosphamide+epirubicin+fluorouracil",
    "GSE76360":  "trastuzumab",
    "GSE62321":  "chemotherapy",
    "ISPY2":     "docetaxel+doxorubicin+cyclophosphamide",
    "BrighTNess": "veliparib+carboplatin+paclitaxel",
}

# Anchor drug for ECFP4 (primary active agent for combos)
ANCHOR_DRUG = {
    "anthracycline+taxane": "paclitaxel",
    "cyclophosphamide+doxorubicin+fluorouracil+paclitaxel": "paclitaxel",
    "cyclophosphamide+doxorubicin+fluorouracil": "doxorubicin",
    "taxane+anthracycline": "paclitaxel",
    "docetaxel": "docetaxel",
    "trastuzumab+chemotherapy": "trastuzumab",
    "cyclophosphamide+doxorubicin+ixabepilone": "ixabepilone",
    "tipifarnib": "tipifarnib",
    "platinum_chemotherapy": "cisplatin",
    "letrozole": "letrozole",
    "doxorubicin+mercaptopurine+methotrexate+prednisone+vincristine": "doxorubicin",
    "imatinib": "imatinib",
    "melphalan": "melphalan",
    "fluorouracil+leucovorin+oxaliplatin": "oxaliplatin",
    "cyclophosphamide+epirubicin+fluorouracil+paclitaxel": "paclitaxel",
    "immunotherapy_peptide_vaccine": None,
    "tamoxifen": "tamoxifen",
    "doxorubicin+paclitaxel+cyclophosphamide+fluorouracil+methotrexate+trastuzumab": "paclitaxel",
    "bortezomib+thalidomide+dexamethasone": "bortezomib",
    "chemotherapy+targeted": None,
    "glucocorticoids": "dexamethasone",
    "cisplatin+docetaxel+fluorouracil": "docetaxel",
    "sorafenib": "sorafenib",
    "rituximab+cyclophosphamide+doxorubicin+vincristine+prednisolone": "doxorubicin",
    "trastuzumab": "trastuzumab",
    "cyclophosphamide+epirubicin+fluorouracil": "epirubicin",
    "cetuximab+platinum": "cetuximab",
    "docetaxel+epirubicin+pegfilgrastim": "docetaxel",
    "chemotherapy": None,
    "docetaxel+doxorubicin+cyclophosphamide": "docetaxel",
    "veliparib+carboplatin+paclitaxel": "paclitaxel",
}

CURATED_PAIRS = [
    ("ERBB2", "ESR1"), ("MKI67", "ACTB"), ("CCND1", "CDKN1A"),
    ("AKT1", "PTEN"), ("CASP3", "BCL2"),
]

FALLBACK_SMILES = {
    "docetaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(O)C(NC(=O)OC(C)(C)C)c5ccccc5)O)OC(=O)c6ccccc6)O)OC(=O)[C@@H](O)C(NC(=O)OC(C)(C)C)c7ccccc7)O)O",
    "paclitaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)c5ccccc5NC(=O)c6ccccc6)O)OC(=O)c7ccccc7)OC(=O)C)O)OC(=O)C)O",
    "doxorubicin": "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(OC5CC(N)C(O)C(C)O5)c4c(O)c3C(=O)c12)C(=O)CO",
    "tamoxifen": "CCC(=C(c1ccccc1)c2ccc(OCCN(C)C)cc2)c3ccccc3",
    "trastuzumab": None,
    "letrozole": "N#Cc1ccc(C(c2ccc(C#N)cc2)n3ccnc3)cc1",
    "cisplatin": "[NH3][Pt]([NH3])(Cl)Cl",
    "oxaliplatin": "O=C1O[Pt]2(OC1=O)[NH2]C3CCCCC3[NH2]2",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(-c5cccnc5)n4",
    "sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "epirubicin": "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(OC5CC(N)C(O)C(C)O5)c4c(O)c3C(=O)c12)C(=O)CO",
    "ixabepilone": "CC1CCCC2(CC(C(C(=O)C(CC(=CC(C1O)C)C)OC(=O)N2)C)O)C",
    "bortezomib": "CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)B(O)O",
    "cetuximab": None,
    "melphalan": "NC(Cc1ccc(N(CCCl)CCCl)cc1)C(=O)O",
    "dexamethasone": "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
    "tipifarnib": "Cc1nc2ccc(Cl)cc2n1C3CC(c4ccc(Cl)cc4)(c5cccnc5)C(=O)N3",
    "veliparib": "CC1(c2cc3c(cc2F)C(=O)N(C3=O)C4CCC(NC1=O)C4)F",
    "fluorouracil": "O=c1[nH]cc(F)c(=O)[nH]1",
}


def clean_matrix(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def within_sample_rank_inv_norm(expression_df):
    mat = expression_df.values.copy()
    n_samples, n_genes = mat.shape
    ranked = np.zeros_like(mat, dtype=float)
    for i in range(n_samples):
        ranked[i, :] = rankdata(mat[i, :], method="average")
    quantiles = (ranked - 0.5) / n_genes
    return pd.DataFrame(norm.ppf(np.clip(quantiles, 1e-7, 1 - 1e-7)),
                        index=expression_df.index, columns=expression_df.columns)

def compute_singscore(expression_df, gene_list):
    import gseapy as gp
    hallmark = gp.get_library("MSigDB_Hallmark_2020")
    ranks = expression_df[gene_list].rank(axis=1)
    n_genes = len(gene_list)
    scores = {}
    for pw, pw_genes in hallmark.items():
        present = [g for g in pw_genes if g in gene_list]
        if len(present) < 5: continue
        scores[f"singscore_{pw}"] = (ranks[present].mean(axis=1) / n_genes - 0.5) * 2
    return pd.DataFrame(scores, index=expression_df.index)

def build_reo_knowledge(expression_df):
    features = {}
    for ga, gb in CURATED_PAIRS:
        if ga in expression_df.columns and gb in expression_df.columns:
            features[f"reo_{ga}_gt_{gb}"] = (expression_df[ga] > expression_df[gb]).astype(int)
    return pd.DataFrame(features, index=expression_df.index)

def tune_threshold(y_true, y_proba):
    thresholds = np.linspace(0.1, 0.9, 81)
    mccs = [matthews_corrcoef(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(mccs))])

def evaluate_predictions(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    r = {"threshold_used": threshold, "n_pos": int(y_true.sum()), "n_neg": int(len(y_true) - y_true.sum())}
    try: r["auroc"] = roc_auc_score(y_true, y_proba)
    except: r["auroc"] = float("nan")
    try: r["auprc"] = average_precision_score(y_true, y_proba)
    except: r["auprc"] = float("nan")
    try: r["mcc"] = matthews_corrcoef(y_true, y_pred)
    except: r["mcc"] = float("nan")
    try: r["bal_acc"] = balanced_accuracy_score(y_true, y_pred)
    except: r["bal_acc"] = float("nan")
    return r


# =========================================================================
# LOAD DATA
# =========================================================================
log.info("Loading data...")

PLATFORM_MAP = {
    "GSE25066": "affymetrix", "GSE20194": "affymetrix", "GSE20271": "affymetrix",
    "GSE22093": "affymetrix", "GSE23988": "affymetrix", "GSE37946": "affymetrix",
    "GSE41998": "affymetrix", "GSE5122": "affymetrix", "GSE8970": "affymetrix",
    "GSE131978": "affymetrix", "GSE20181": "affymetrix",
    "GSE14615": "affymetrix", "GSE14671": "affymetrix", "GSE19293": "affymetrix",
    "GSE28702": "affymetrix", "GSE32646": "affymetrix", "GSE35640": "affymetrix",
    "GSE48905": "affymetrix", "GSE50948": "affymetrix", "GSE63885": "affymetrix",
    "GSE68871": "affymetrix", "GSE72970": "affymetrix", "GSE73578": "affymetrix",
    "GSE62321": "affymetrix",
    "GSE104645": "agilent", "GSE109211": "agilent", "GSE173263": "agilent",
    "GSE21974": "agilent", "GSE44272": "agilent", "GSE4779": "agilent",
    "GSE65021": "agilent", "GSE66999": "agilent", "GSE6861": "agilent",
    "GSE76360": "agilent", "GSE82172": "agilent",
    "GSE104958": "rnaseq", "ISPY2": "agilent", "BrighTNess": "rnaseq",
}

def load_ctrdb_dataset(geo_id):
    base = ROOT / "data" / "raw" / "ctrdb" / geo_id
    ep = base / f"{geo_id}_expression.parquet"
    lp = base / "response_labels.parquet"
    if not ep.exists() or not lp.exists(): return None, None
    expr = pd.read_parquet(ep); labels = pd.read_parquet(lp)
    common = expr.index.intersection(labels.index)
    if len(common) < 20: return None, None
    return expr.loc[common], labels.loc[common, "response"].astype(int)

def load_positional_dataset(data_dir, geo_id):
    base = ROOT / "data" / "raw" / data_dir
    expr = pd.read_parquet(base / f"{geo_id}_expression.parquet")
    labels = pd.read_parquet(base / "response_labels.parquet")
    expr = expr.reset_index(drop=True)
    return expr, pd.Series(labels["response"].astype(int).values, index=expr.index, name="response")

datasets = {}
for geo_id in sorted(d.name for d in (ROOT/"data"/"raw"/"ctrdb").iterdir() if d.is_dir() and d.name.startswith("GSE") and d.name != "GSE194040"):
    expr, labels = load_ctrdb_dataset(geo_id)
    if expr is not None and labels is not None:
        tech = PLATFORM_MAP.get(geo_id, "unknown")
        if tech not in ("targeted", "beadarray") and labels.nunique() >= 2:
            datasets[geo_id] = (expr, labels)

for name, ddir, gid in [("ISPY2","ispy2","GSE194040"),("BrighTNess","brightness","GSE164458")]:
    try:
        e, l = load_positional_dataset(ddir, gid)
        if l.nunique() >= 2: datasets[name] = (e, l)
    except: pass

# Load common genes
with open(ROOT / "data" / "cache" / "common_genes_cross_platform.json") as f:
    common_genes = json.load(f)

log.info(f"{len(datasets)} datasets, {sum(len(v[1]) for v in datasets.values())} patients, {len(common_genes)} genes")

# Restrict, z-score, pool
all_px, all_py, all_batch, all_tech, all_drug = [], [], [], [], []
s2d = {}
for did in sorted(datasets.keys()):
    e, l = datasets[did]
    avail = [g for g in common_genes if g in e.columns]
    miss = [g for g in common_genes if g not in e.columns]
    es = e[avail].copy()
    for g in miss: es[g] = 0.0
    es = es[common_genes]
    m = es.mean(axis=0); s = es.std(axis=0).replace(0, 1)
    es = clean_matrix((es - m) / s)
    idx = [f"{did}__{i}" for i in range(len(es))]
    es.index = idx; lc = l.copy(); lc.index = idx
    all_px.append(es); all_py.append(lc)
    tech = PLATFORM_MAP.get(did, "unknown")
    drug = DATASET_DRUG_MAP.get(did, "unknown")
    all_batch.extend([did]*len(es)); all_tech.extend([tech]*len(es)); all_drug.extend([drug]*len(es))
    for s_ in idx: s2d[s_] = did

pooled_expr = pd.concat(all_px); pooled_labels = pd.concat(all_py)
batch_series = pd.Series(all_batch, index=pooled_expr.index)
drug_series = pd.Series(all_drug, index=pooled_expr.index)
log.info(f"Pooled: {pooled_expr.shape}")


# =========================================================================
# PHASE 0: GSE104958 INVESTIGATION
# =========================================================================
log.info("=" * 60)
log.info("PHASE 0: GSE104958 investigation")
log.info("=" * 60)

gse_samples = [s for s in pooled_expr.index if s2d[s] == "GSE104958"]
train_samples_0 = [s for s in pooled_expr.index if s2d[s] != "GSE104958"]

rank_all = within_sample_rank_inv_norm(pooled_expr)
ss_all = compute_singscore(pooled_expr, common_genes)
reo_all = build_reo_knowledge(pooled_expr)

X_tr0 = pd.concat([rank_all.loc[train_samples_0], ss_all.loc[train_samples_0], reo_all.loc[train_samples_0]], axis=1)
X_te0 = pd.concat([rank_all.loc[gse_samples], ss_all.loc[gse_samples], reo_all.loc[gse_samples]], axis=1)
X_tr0 = clean_matrix(X_tr0); X_te0 = clean_matrix(X_te0)
for c in X_tr0.columns:
    if c not in X_te0.columns: X_te0[c] = 0
X_te0 = X_te0[X_tr0.columns]

y_tr0 = pooled_labels.loc[train_samples_0].values.astype(int)
y_te0 = pooled_labels.loc[gse_samples].values.astype(int)

mdl0 = lgb.LGBMClassifier(**LGBM_PARAMS); mdl0.fit(X_tr0.values, y_tr0)
preds0 = mdl0.predict_proba(X_te0.values)[:, 1]
point_auroc = roc_auc_score(y_te0, preds0)

rng = np.random.default_rng(42)
boot_aurocs = []
for _ in range(1000):
    idx = rng.choice(len(y_te0), size=len(y_te0), replace=True)
    try: boot_aurocs.append(roc_auc_score(y_te0[idx], preds0[idx]))
    except: continue

ci_lo = np.percentile(boot_aurocs, 2.5)
ci_hi = np.percentile(boot_aurocs, 97.5)
verdict = "RELIABLE" if ci_lo >= 0.60 else "WIDE_CI"

log.info(f"GSE104958: drug={DATASET_DRUG_MAP.get('GSE104958')}, n={len(y_te0)}, resp_rate={y_te0.mean():.2f}")
log.info(f"AUROC={point_auroc:.3f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f}), verdict={verdict}")

pd.DataFrame([{
    "dataset": "GSE104958", "drug": DATASET_DRUG_MAP.get("GSE104958"),
    "n": len(y_te0), "resp_rate": round(y_te0.mean(), 3),
    "auroc": round(point_auroc, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
    "verdict": verdict,
}]).to_csv(OUT_DIR / "gse104958_audit.tsv", sep="\t", index=False)


# =========================================================================
# PHASE 1: DRUG FEATURES
# =========================================================================
log.info("=" * 60)
log.info("PHASE 1: Drug feature construction")
log.info("=" * 60)

# Get unique anchor drugs
unique_regimens = set(DATASET_DRUG_MAP.values())
anchor_drugs = set(v for v in ANCHOR_DRUG.values() if v is not None)
log.info(f"Unique regimens: {len(unique_regimens)}, anchor drugs with SMILES: {len(anchor_drugs)}")

# ECFP4
drug_smiles = {d: FALLBACK_SMILES.get(d) for d in anchor_drugs}
ecfp4_rows = {}
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    for drug, smiles in drug_smiles.items():
        if not smiles:
            ecfp4_rows[drug] = np.zeros(1024, dtype=np.int8)
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            ecfp4_rows[drug] = np.zeros(1024, dtype=np.int8)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        ecfp4_rows[drug] = np.array(fp, dtype=np.int8)
    ecfp4_cols = [f"ecfp4_{i}" for i in range(1024)]
    ecfp4_df = pd.DataFrame(ecfp4_rows, index=ecfp4_cols).T
    log.info(f"ECFP4: {len(ecfp4_df)} drugs x {ecfp4_df.shape[1]} bits")
except ImportError:
    log.warning("rdkit not available, ECFP4 features skipped")
    ecfp4_df = pd.DataFrame()

# Drug-target features from GDSC2
drug_targets = {}
try:
    ref = pd.read_parquet(ROOT / "data" / "cache" / "breast_dose_response_ref.parquet")
    for drug, group in ref.groupby("drug_name"):
        raw = group["putative_target"].dropna()
        if len(raw) == 0: continue
        targets = [t.strip() for t in str(raw.iloc[0]).replace(";", ",").split(",") if t.strip()]
        drug_targets[drug.lower()] = targets
    log.info(f"GDSC2 drug targets: {len(drug_targets)} drugs")
except Exception as e:
    log.warning(f"GDSC2 targets failed: {e}")


def get_ecfp4_for_sample(dataset_id):
    """Get ECFP4 vector for a sample based on its dataset's anchor drug."""
    regimen = DATASET_DRUG_MAP.get(dataset_id, "unknown")
    anchor = ANCHOR_DRUG.get(regimen)
    if anchor and anchor in ecfp4_df.index:
        return ecfp4_df.loc[anchor].values
    return np.zeros(1024, dtype=np.int8)

def get_drug_target_features(rank_expr_row, dataset_id):
    """Drug-target interaction features for one sample."""
    regimen = DATASET_DRUG_MAP.get(dataset_id, "unknown")
    # Try each component drug for targets
    components = regimen.split("+")
    all_targets = []
    for comp in components:
        comp = comp.strip()
        all_targets.extend(drug_targets.get(comp, []))
    present = [t for t in all_targets if t in rank_expr_row.index]
    if present:
        vals = rank_expr_row[present]
        return {
            "dtf_mean_target": vals.mean(),
            "dtf_max_target": vals.max(),
            "dtf_n_targets": len(present),
            "dtf_frac_dysreg": (vals.abs() > 1.5).mean(),
        }
    return {"dtf_mean_target": 0, "dtf_max_target": 0, "dtf_n_targets": 0, "dtf_frac_dysreg": 0}


# Build per-sample drug features for all pooled data
log.info("Building per-sample drug features...")

ecfp4_per_sample = []
dtf_per_sample = []
for sid in pooled_expr.index:
    did = s2d[sid]
    ecfp4_per_sample.append(get_ecfp4_for_sample(did))
    dtf_per_sample.append(get_drug_target_features(rank_all.loc[sid], did))

ecfp4_matrix = pd.DataFrame(ecfp4_per_sample, index=pooled_expr.index, columns=ecfp4_cols) if ecfp4_df.shape[0] > 0 else pd.DataFrame(index=pooled_expr.index)
dtf_matrix = pd.DataFrame(dtf_per_sample, index=pooled_expr.index)

log.info(f"ECFP4 per-sample: {ecfp4_matrix.shape}")
log.info(f"Drug-target per-sample: {dtf_matrix.shape}")

# Dataset one-hot for exploratory config
dataset_onehot = pd.get_dummies(batch_series, prefix="ds")


# =========================================================================
# PHASE 2: LODO EXPERIMENT GRID
# =========================================================================
log.info("=" * 60)
log.info("PHASE 2: LODO experiment grid")
log.info("=" * 60)

# Pre-computed patient features
patient_feats = {
    "rank": rank_all,
    "singscore": ss_all,
    "reo": reo_all,
    "ecfp4": ecfp4_matrix,
    "dtf": dtf_matrix,
    "ds_onehot": dataset_onehot,
}

GRID = [
    # (name, rank, singscore, reo, ecfp4, dtf, ds_onehot, prod_candidate)
    ("baseline_rank_singscore_reo", True, True, True, False, False, False, True),
    ("D1_rank_ecfp4",              True, False, False, True, False, False, True),
    ("D4_rank_dtf",                True, False, False, False, True, False, True),
    ("D5_rank_ecfp4_dtf",          True, False, False, True, True, False, True),
    ("D6_baseline_ecfp4",          True, True, True, True, False, False, True),
    ("D8_baseline_ecfp4_dtf",      True, True, True, True, True, False, True),
    ("D10_rank_dataset_id",        True, False, False, False, False, True, False),
]

dataset_ids = sorted(set(s2d.values()))
all_results = []

for config_name, use_rank, use_ss, use_reo, use_ecfp, use_dtf, use_ds, prod in GRID:
    log.info(f"\n--- {config_name} ---")
    t0 = time.time()
    fold_results = []

    for holdout_id in dataset_ids:
        train_s = [s for s in pooled_expr.index if s2d[s] != holdout_id]
        test_s = [s for s in pooled_expr.index if s2d[s] == holdout_id]
        train_y = pooled_labels.loc[train_s].values.astype(int)
        test_y = pooled_labels.loc[test_s].values.astype(int)
        if len(np.unique(test_y)) < 2 or len(np.unique(train_y)) < 2:
            continue

        parts_tr, parts_te = [], []
        if use_rank: parts_tr.append(patient_feats["rank"].loc[train_s]); parts_te.append(patient_feats["rank"].loc[test_s])
        if use_ss: parts_tr.append(patient_feats["singscore"].loc[train_s]); parts_te.append(patient_feats["singscore"].loc[test_s])
        if use_reo: parts_tr.append(patient_feats["reo"].loc[train_s]); parts_te.append(patient_feats["reo"].loc[test_s])
        if use_ecfp and ecfp4_matrix.shape[1] > 0: parts_tr.append(patient_feats["ecfp4"].loc[train_s]); parts_te.append(patient_feats["ecfp4"].loc[test_s])
        if use_dtf: parts_tr.append(patient_feats["dtf"].loc[train_s]); parts_te.append(patient_feats["dtf"].loc[test_s])
        if use_ds: parts_tr.append(patient_feats["ds_onehot"].loc[train_s]); parts_te.append(patient_feats["ds_onehot"].loc[test_s])

        X_tr = clean_matrix(pd.concat(parts_tr, axis=1))
        X_te = clean_matrix(pd.concat(parts_te, axis=1))
        for c in X_tr.columns:
            if c not in X_te.columns: X_te[c] = 0
        for c in X_te.columns:
            if c not in X_tr.columns: X_tr[c] = 0
        X_te = X_te[X_tr.columns]

        mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
        mdl.fit(X_tr.values, train_y)
        tr_pred = mdl.predict_proba(X_tr.values)[:, 1]
        threshold = tune_threshold(train_y, tr_pred)
        te_pred = mdl.predict_proba(X_te.values)[:, 1]
        metrics = evaluate_predictions(test_y, te_pred, threshold)

        fold_results.append({
            "config": config_name, "holdout": holdout_id,
            "holdout_drug": DATASET_DRUG_MAP.get(holdout_id, "unknown"),
            "production_candidate": prod, "n_features": X_tr.shape[1],
            **metrics,
        })

    if fold_results:
        df = pd.DataFrame(fold_results)
        log.info(f"  → {config_name}: AUROC={df['auroc'].mean():.4f}  AUPRC={df['auprc'].mean():.4f}  MCC={df['mcc'].mean():.4f}  ({time.time()-t0:.0f}s)")
        all_results.extend(fold_results)


# =========================================================================
# RESULTS
# =========================================================================
log.info("=" * 60)
log.info("RESULTS")
log.info("=" * 60)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "all_results.tsv", sep="\t", index=False)

summary = (
    results_df.groupby(["config", "production_candidate"])
    .agg(mean_auroc=("auroc","mean"), std_auroc=("auroc","std"),
         mean_auprc=("auprc","mean"), mean_mcc=("mcc","mean"),
         mean_bal_acc=("bal_acc","mean"), n_folds=("auroc","count"))
    .reset_index().sort_values("mean_auroc", ascending=False)
)
summary.to_csv(OUT_DIR / "summary.tsv", sep="\t", index=False)

log.info("\nFull comparison:")
for _, r in summary.iterrows():
    tag = "" if r["production_candidate"] else " [EXPLORATORY]"
    log.info(f"  {r['config']:40s}  AUROC={r['mean_auroc']:.4f}  AUPRC={r['mean_auprc']:.4f}  MCC={r['mean_mcc']:.4f}{tag}")

# Per-drug analysis for best drug-aware config
best_drug_config = summary[summary["production_candidate"] & summary["config"].str.startswith("D")]["config"].iloc[0] if len(summary[summary["production_candidate"] & summary["config"].str.startswith("D")]) > 0 else None

if best_drug_config:
    baseline_folds = results_df[results_df["config"] == "baseline_rank_singscore_reo"].set_index("holdout")
    drug_folds = results_df[results_df["config"] == best_drug_config].set_index("holdout")
    common_holdouts = baseline_folds.index.intersection(drug_folds.index)

    per_drug = pd.DataFrame({
        "holdout_drug": baseline_folds.loc[common_holdouts, "holdout_drug"],
        "baseline_auroc": baseline_folds.loc[common_holdouts, "auroc"],
        "drug_aware_auroc": drug_folds.loc[common_holdouts, "auroc"],
    })
    per_drug["delta"] = per_drug["drug_aware_auroc"] - per_drug["baseline_auroc"]
    per_drug = per_drug.sort_values("delta", ascending=False)
    per_drug.to_csv(OUT_DIR / "per_drug_delta.tsv", sep="\t")

    log.info(f"\nPer-drug delta ({best_drug_config} vs baseline):")
    for holdout, row in per_drug.iterrows():
        marker = "+" if row["delta"] > 0.02 else "-" if row["delta"] < -0.02 else "="
        log.info(f"  {holdout:15s} ({row['holdout_drug']:30s}): {row['baseline_auroc']:.3f} → {row['drug_aware_auroc']:.3f} ({row['delta']:+.3f}) {marker}")

    improved = (per_drug["delta"] > 0.02).sum()
    hurt = (per_drug["delta"] < -0.02).sum()
    log.info(f"\n  Improved: {improved}/{len(per_drug)}, Hurt: {hurt}/{len(per_drug)}")

log.info(f"\nDone. Results in {OUT_DIR}/")
