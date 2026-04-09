#!/usr/bin/env python
"""
INVEREX — Response-Proxy Variable Analysis
============================================

PHASE 1: Build proxy library (metadata from SOFT + expression-derived)
PHASE 2: NMI analysis (proxy vs response/dataset/technology)
PHASE 3: Proxy-guided ComBat LODO experiment
PHASE 4: Information recovery analysis

All outputs to results/proxy_analysis/.
"""

import os, sys, time, json, gzip, re, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "proxy_analysis"
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


def clean_matrix(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def within_sample_rank_inv_norm(expression_df):
    mat = expression_df.values.copy()
    n_s, n_g = mat.shape
    ranked = np.zeros_like(mat, dtype=float)
    for i in range(n_s):
        ranked[i, :] = rankdata(mat[i, :], method="average")
    q = (ranked - 0.5) / n_g
    return pd.DataFrame(norm.ppf(np.clip(q, 1e-7, 1 - 1e-7)),
                        index=expression_df.index, columns=expression_df.columns)

def tune_threshold(y_true, y_proba):
    thresholds = np.linspace(0.1, 0.9, 81)
    mccs = [matthews_corrcoef(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(mccs))])

def evaluate_predictions(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    r = {"threshold_used": threshold}
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
# DATA LOADING
# =========================================================================
log.info("Loading data...")

def load_ds(geo_id):
    base = ROOT / "data" / "raw" / "ctrdb" / geo_id
    ep = base / f"{geo_id}_expression.parquet"
    lp = base / "response_labels.parquet"
    if not ep.exists() or not lp.exists(): return None, None
    expr = pd.read_parquet(ep); labels = pd.read_parquet(lp)
    common = expr.index.intersection(labels.index)
    if len(common) < 20: return None, None
    return expr.loc[common], labels.loc[common, "response"].astype(int)

def load_pos(ddir, gid):
    base = ROOT / "data" / "raw" / ddir
    expr = pd.read_parquet(base / f"{gid}_expression.parquet")
    labels = pd.read_parquet(base / "response_labels.parquet")
    expr = expr.reset_index(drop=True)
    return expr, pd.Series(labels["response"].astype(int).values, index=expr.index, name="response")

datasets = {}
for geo_id in sorted(d.name for d in (ROOT/"data"/"raw"/"ctrdb").iterdir() if d.is_dir() and d.name.startswith("GSE") and d.name != "GSE194040"):
    expr, labels = load_ds(geo_id)
    if expr is not None and labels is not None:
        tech = PLATFORM_MAP.get(geo_id, "unknown")
        if tech not in ("targeted", "beadarray") and labels.nunique() >= 2:
            datasets[geo_id] = (expr, labels)

for name, ddir, gid in [("ISPY2","ispy2","GSE194040"),("BrighTNess","brightness","GSE164458")]:
    try:
        e, l = load_pos(ddir, gid)
        if l.nunique() >= 2: datasets[name] = (e, l)
    except: pass

with open(ROOT / "data" / "cache" / "common_genes_cross_platform.json") as f:
    common_genes = json.load(f)

log.info(f"{len(datasets)} datasets, {sum(len(v[1]) for v in datasets.values())} patients, {len(common_genes)} genes")


# =========================================================================
# PHASE 1A: Parse SOFT files for ER/HER2/PR/PAM50 metadata
# =========================================================================
log.info("=" * 60)
log.info("PHASE 1A: Parsing SOFT files for clinical metadata")
log.info("=" * 60)

CLINICAL_PATTERNS = {
    "er_status": [
        r"er[_ ]status:\s*(\w+)",
        r"estrogen[_ ]receptor[_ ]status:\s*(\w+)",
        r"er[_ ]ihc:\s*(\w+)",
        r"^\s*er:\s*(\w+)",
    ],
    "her2_status": [
        r"her2[_ ]status:\s*(\w+)",
        r"her2[_ ]ihc:\s*(\w+)",
        r"^\s*her2:\s*(\w+)",
    ],
    "pr_status": [
        r"pr[_ ]status:\s*(\w+)",
        r"progesterone[_ ]receptor[_ ]status:\s*(\w+)",
    ],
    "pam50_subtype": [
        r"pam50:\s*(\w+)",
        r"pam50[_ ]subtype:\s*(\w+)",
        r"intrinsic[_ ]subtype:\s*(\w+)",
    ],
    "tumor_grade": [
        r"grade:\s*(\d)",
        r"tumor[_ ]grade:\s*(\d)",
    ],
}

def normalize_status(value):
    """Normalize ER/HER2/PR status values."""
    if value is None: return None
    v = str(value).strip().lower()
    if v in ("p", "pos", "positive", "+", "+ve", "yes", "1"):
        return "positive"
    if v in ("n", "neg", "negative", "-", "-ve", "no", "0"):
        return "negative"
    return None

def parse_soft_file(soft_path):
    """Extract sample-level clinical metadata from a GEO SOFT file."""
    samples = {}
    current_sample = None
    try:
        with gzip.open(soft_path, "rt", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("^SAMPLE"):
                    current_sample = line.split("=")[1].strip()
                    samples[current_sample] = {}
                elif line.startswith("!Sample_characteristics_ch1") and current_sample:
                    val = line.split("=", 1)[1].strip() if "=" in line else ""
                    for field, patterns in CLINICAL_PATTERNS.items():
                        for pat in patterns:
                            m = re.search(pat, val, re.IGNORECASE)
                            if m:
                                samples[current_sample][field] = m.group(1)
                                break
    except Exception as e:
        log.warning(f"Failed to parse {soft_path}: {e}")
        return {}
    return samples

# Parse SOFT files for all datasets
clinical_meta = {}
for did in datasets:
    if did in ("ISPY2", "BrighTNess"): continue
    soft_path = ROOT / "data" / "raw" / "ctrdb" / did / f"{did}_family.soft.gz"
    if soft_path.exists():
        meta = parse_soft_file(soft_path)
        if meta:
            for sid, fields in meta.items():
                if fields:
                    clinical_meta[sid] = fields

log.info(f"Parsed clinical metadata for {len(clinical_meta)} samples")

# Coverage report per field
field_coverage = {f: 0 for f in CLINICAL_PATTERNS.keys()}
for sid, fields in clinical_meta.items():
    for f in fields:
        field_coverage[f] += 1
for f, n in field_coverage.items():
    log.info(f"  {f}: {n} samples")


# =========================================================================
# PHASE 1B: Pool data + build expression proxies
# =========================================================================
log.info("=" * 60)
log.info("PHASE 1B: Pooling data and building expression proxies")
log.info("=" * 60)

# Restrict, z-score, pool
all_px, all_py, all_batch, all_orig_sid = [], [], [], []
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
    new_idx = [f"{did}__{i}" for i in range(len(es))]
    orig_idx = list(es.index)  # original GSM IDs
    es.index = new_idx
    lc = l.copy(); lc.index = new_idx
    all_px.append(es); all_py.append(lc)
    all_batch.extend([did]*len(es))
    all_orig_sid.extend(orig_idx)
    for s_ in new_idx: s2d[s_] = did

pooled_expr = pd.concat(all_px)
pooled_labels = pd.concat(all_py)
batch_series = pd.Series(all_batch, index=pooled_expr.index)
tech_series = pd.Series([PLATFORM_MAP.get(d,"unknown") for d in all_batch], index=pooled_expr.index)
orig_sid_series = pd.Series(all_orig_sid, index=pooled_expr.index)
log.info(f"Pooled: {pooled_expr.shape}")

# Map clinical metadata to pooled samples
def map_clinical(field):
    """Map a clinical field to pooled sample IDs."""
    out = {}
    for new_idx, orig_sid in zip(pooled_expr.index, all_orig_sid):
        meta = clinical_meta.get(orig_sid, {})
        out[new_idx] = normalize_status(meta.get(field)) if field in ("er_status", "her2_status", "pr_status") else meta.get(field)
    return pd.Series(out, name=field)

er_status = map_clinical("er_status")
her2_status = map_clinical("her2_status")
pr_status = map_clinical("pr_status")
pam50_meta = map_clinical("pam50_subtype")

log.info(f"Clinical metadata coverage in pooled data:")
log.info(f"  ER status: {er_status.notna().sum()}/{len(er_status)} ({er_status.notna().mean():.1%})")
log.info(f"  HER2 status: {her2_status.notna().sum()}/{len(her2_status)} ({her2_status.notna().mean():.1%})")
log.info(f"  PR status: {pr_status.notna().sum()}/{len(pr_status)} ({pr_status.notna().mean():.1%})")
log.info(f"  PAM50 metadata: {pam50_meta.notna().sum()}/{len(pam50_meta)} ({pam50_meta.notna().mean():.1%})")

# TNBC label (derived)
tnbc_label = pd.Series(index=pooled_expr.index, dtype=object)
for idx in pooled_expr.index:
    er = er_status.loc[idx] if idx in er_status.index else None
    her2 = her2_status.loc[idx] if idx in her2_status.index else None
    pr = pr_status.loc[idx] if idx in pr_status.index else None
    if all(v == "negative" for v in [er, her2, pr] if v is not None) and any(v is not None for v in [er, her2, pr]):
        tnbc_label.loc[idx] = "tnbc"
    elif any(v == "positive" for v in [er, her2, pr] if v is not None):
        tnbc_label.loc[idx] = "non_tnbc"

log.info(f"  TNBC derived: {tnbc_label.notna().sum()}/{len(tnbc_label)}")

# =========================================================================
# Build expression-derived proxies
# =========================================================================
log.info("Building expression-derived proxies...")
rank_expr = within_sample_rank_inv_norm(pooled_expr)

proxies = {}

# Family 1: Single-gene surrogates
SINGLE_GENE_PROXIES = {
    "proxy_MKI67_rank": "MKI67", "proxy_CCND1_rank": "CCND1", "proxy_TOP2A_rank": "TOP2A",
    "proxy_ESR1_rank": "ESR1", "proxy_PGR_rank": "PGR", "proxy_AR_rank": "AR",
    "proxy_ERBB2_rank": "ERBB2", "proxy_GRB7_rank": "GRB7",
    "proxy_KRT5_rank": "KRT5", "proxy_KRT14_rank": "KRT14", "proxy_VIM_rank": "VIM",
    "proxy_BCL2_rank": "BCL2", "proxy_FOXO3_rank": "FOXO3", "proxy_AKT1_rank": "AKT1",
    "proxy_GATA3_rank": "GATA3", "proxy_FOXA1_rank": "FOXA1",
}
for pname, gene in SINGLE_GENE_PROXIES.items():
    if gene in rank_expr.columns:
        proxies[pname] = rank_expr[gene]

# Family 2: Signature scores
def signature_score(genes, name):
    present = [g for g in genes if g in rank_expr.columns]
    if len(present) >= 3:
        proxies[name] = rank_expr[present].mean(axis=1)

signature_score(["MKI67", "TOP2A", "CCND1", "MCM2", "CDC20", "CCNB1", "PLK1", "AURKA", "PCNA"], "proxy_proliferation_score")
signature_score(["ESR1", "PGR", "GATA3", "FOXA1", "TFF1", "AREG", "BCL2"], "proxy_er_score")
signature_score(["ERBB2", "GRB7", "STARD3", "MIEN1", "PGAP3"], "proxy_her2_score")
signature_score(["KRT5", "KRT14", "KRT17", "VIM", "CDH3", "FOXC1"], "proxy_basal_score")
signature_score(["BRCA1", "BRCA2", "RAD51", "FANCD2", "RRM2", "TYMS"], "proxy_ddr_score")

# Family 3: Tertiles and binarized versions
for sn in ["proxy_proliferation_score", "proxy_er_score", "proxy_her2_score", "proxy_basal_score"]:
    if sn in proxies:
        try:
            proxies[f"{sn}_tertile"] = pd.qcut(proxies[sn], q=3, labels=["low","mid","high"], duplicates="drop").astype(str)
            proxies[f"{sn}_binary"] = (proxies[sn] > proxies[sn].median()).astype(int).astype(str)
        except: pass

# Family 4: PAM50 from expression (k-means approximation since centroids not loaded)
PAM50_GENES = [
    "ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1","CDC20","CDC6",
    "CDH3","CENPF","CEP55","CXXC5","DCK","EGFR","ERBB2","ESR1","EXO1","FGFR4","FOXA1",
    "FOXC1","GPR160","GRB7","KIF2C","KRT14","KRT17","KRT5","MAPT","MDM2","MELK","MIA",
    "MKI67","MLPH","MMP11","MYBL2","MYC","NAT1","NDC80","NUF2","ORC6","PGR","PHGDH",
    "PTTG1","RRM2","SFRP1","SLC39A6","TMEM45B","TYMS","UBE2C","UBE2T",
]
present_pam50 = [g for g in PAM50_GENES if g in rank_expr.columns]
log.info(f"PAM50 genes present: {len(present_pam50)}/50")

if len(present_pam50) >= 30:
    pam50_expr = rank_expr[present_pam50]
    km_pam50 = KMeans(n_clusters=5, random_state=42, n_init=10)
    pam50_clusters_int = km_pam50.fit_predict(pam50_expr.values)
    # Label clusters by their mean expression of marker genes
    cluster_to_label = {}
    cluster_means = pd.DataFrame(pam50_expr.values, index=pooled_expr.index).assign(cluster=pam50_clusters_int).groupby("cluster").mean()
    cluster_means.columns = present_pam50
    # Assign labels based on dominant marker
    for c in range(5):
        row = cluster_means.iloc[c]
        if "ERBB2" in row.index and row["ERBB2"] == cluster_means["ERBB2"].max():
            cluster_to_label[c] = "HER2"
        elif "KRT5" in row.index and row["KRT5"] == cluster_means["KRT5"].max():
            cluster_to_label[c] = "Basal"
        elif "MKI67" in row.index and row["MKI67"] == cluster_means["MKI67"].max():
            cluster_to_label[c] = "LumB"
        elif "ESR1" in row.index and row["ESR1"] == cluster_means["ESR1"].max():
            cluster_to_label[c] = "LumA"
        else:
            cluster_to_label[c] = "Normal"
    # Resolve duplicates
    used = set()
    for c in range(5):
        if cluster_to_label[c] in used:
            for fallback in ["LumA","LumB","HER2","Basal","Normal"]:
                if fallback not in used:
                    cluster_to_label[c] = fallback
                    break
        used.add(cluster_to_label[c])
    proxies["proxy_pam50_computed"] = pd.Series([cluster_to_label[c] for c in pam50_clusters_int], index=pooled_expr.index)
    log.info(f"  PAM50 computed distribution: {proxies['proxy_pam50_computed'].value_counts().to_dict()}")

# Family 5: Unsupervised k-means clusters on full expression
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rank_expr.values)
    proxies[f"proxy_kmeans_{k}"] = pd.Series(labels.astype(str), index=pooled_expr.index)

# Add metadata proxies (ER/HER2/PR/TNBC)
proxies["proxy_er_status_meta"] = er_status
proxies["proxy_her2_status_meta"] = her2_status
proxies["proxy_pr_status_meta"] = pr_status
proxies["proxy_tnbc_meta"] = tnbc_label

proxy_df = pd.DataFrame(proxies, index=pooled_expr.index)
proxy_df.to_parquet(ROOT / "data" / "cache" / "expression_proxies.parquet")
log.info(f"Built {len(proxies)} proxy variables")


# =========================================================================
# PHASE 2: NMI ANALYSIS
# =========================================================================
log.info("=" * 60)
log.info("PHASE 2: NMI analysis")
log.info("=" * 60)

def discretize(s, n_bins=5):
    if s.dtype.kind in 'fiu':
        valid = s.dropna()
        if valid.nunique() <= 1: return s.astype(str)
        try:
            kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            binned = kbd.fit_transform(valid.values.reshape(-1,1)).flatten().astype(int)
            out = pd.Series(index=s.index, dtype=object)
            out.loc[valid.index] = [str(int(b)) for b in binned]
            return out.fillna("missing")
        except:
            return s.astype(str)
    return s.fillna("missing").astype(str)

def compute_nmi(p, y, d, t):
    """NMI of proxy vs response, dataset, technology."""
    p_str = discretize(p)
    y_str = y.astype(str)
    d_str = d.astype(str)
    t_str = t.astype(str)
    valid = (p_str != "missing") & (p_str != "nan") & y.notna()
    if valid.sum() < 50:
        return None
    p_v, y_v, d_v, t_v = p_str[valid], y_str[valid], d_str[valid], t_str[valid]
    if p_v.nunique() < 2:
        return None
    return {
        "n_valid": int(valid.sum()),
        "n_unique": p_v.nunique(),
        "nmi_response": normalized_mutual_info_score(y_v, p_v, average_method="arithmetic"),
        "ami_response": adjusted_mutual_info_score(y_v, p_v),
        "nmi_dataset": normalized_mutual_info_score(d_v, p_v, average_method="arithmetic"),
        "nmi_technology": normalized_mutual_info_score(t_v, p_v, average_method="arithmetic"),
    }

nmi_results = []
for pname in proxy_df.columns:
    r = compute_nmi(proxy_df[pname], pooled_labels, batch_series, tech_series)
    if r is None:
        continue
    r["proxy"] = pname
    r["response_specificity"] = max(0, r["nmi_response"] - max(r["nmi_dataset"], r["nmi_technology"]))
    nmi_results.append(r)

nmi_df = pd.DataFrame(nmi_results).sort_values("response_specificity", ascending=False)
nmi_df.to_csv(OUT_DIR / "nmi_full_profile.tsv", sep="\t", index=False)

log.info(f"\nTop 15 proxies by response_specificity:")
for _, r in nmi_df.head(15).iterrows():
    log.info(f"  {r['proxy']:35s}  NMI(Y)={r['nmi_response']:.4f}  NMI(DS)={r['nmi_dataset']:.4f}  NMI(tech)={r['nmi_technology']:.4f}  spec={r['response_specificity']:.4f}  n={r['n_valid']}")

log.info(f"\nTop 10 by raw NMI(Y):")
for _, r in nmi_df.sort_values("nmi_response", ascending=False).head(10).iterrows():
    log.info(f"  {r['proxy']:35s}  NMI(Y)={r['nmi_response']:.4f}")


# =========================================================================
# PHASE 3: PROXY-GUIDED COMBAT LODO
# =========================================================================
log.info("=" * 60)
log.info("PHASE 3: Proxy-guided ComBat LODO")
log.info("=" * 60)

from neuroCombat import neuroCombat

def run_combat_with_covariates(expr, batches, covariate_dict):
    """ComBat with categorical covariates."""
    covars = pd.DataFrame({"batch": batches.values}, index=expr.index)
    cat_cols = []
    cont_cols = []
    for cname, cseries in covariate_dict.items():
        # Fill NaN with "unknown"
        c_clean = cseries.fillna("unknown").astype(str)
        # neuroCombat needs numeric for categorical too
        unique_vals = sorted(c_clean.unique())
        val_to_int = {v: i for i, v in enumerate(unique_vals)}
        covars[cname] = [val_to_int[v] for v in c_clean.values]
        cat_cols.append(cname)
    result = neuroCombat(
        dat=expr.values.T, covars=covars, batch_col="batch",
        categorical_cols=cat_cols if cat_cols else None,
    )
    return clean_matrix(pd.DataFrame(result["data"].T, columns=expr.columns, index=expr.index))

def run_combat_no_covariate(expr, batches):
    covars = pd.DataFrame({"batch": batches.values}, index=expr.index)
    result = neuroCombat(dat=expr.values.T, covars=covars, batch_col="batch")
    return clean_matrix(pd.DataFrame(result["data"].T, columns=expr.columns, index=expr.index))

# Configs to test
PROXY_CONFIGS = [
    ("baseline_no_combat",     None,                              True),
    ("combat_no_covariate",    {},                                True),
    ("combat_pam50_computed",  {"pam50": "proxy_pam50_computed"}, True),
    ("combat_kmeans_5",        {"k5": "proxy_kmeans_5"},          True),
    ("combat_prolif_tertile",  {"prolif": "proxy_proliferation_score_tertile"}, True),
    ("combat_er_score_binary", {"er_b": "proxy_er_score_binary"}, True),
    ("combat_her2_score_binary", {"her2_b": "proxy_her2_score_binary"}, True),
    ("combat_basal_score_binary", {"basal_b": "proxy_basal_score_binary"}, True),
    ("combat_prolif_er_combo", {"prolif": "proxy_proliferation_score_tertile", "er_b": "proxy_er_score_binary"}, True),
    ("combat_pam50_prolif",    {"pam50": "proxy_pam50_computed", "prolif": "proxy_proliferation_score_tertile"}, True),
    ("combat_response_LEAKED", {"label": "_LABEL"},               False),  # upper bound
]

dataset_ids = sorted(set(s2d.values()))
all_results = []

for config_name, cov_spec, prod_cand in PROXY_CONFIGS:
    log.info(f"\n--- {config_name} ---")
    t_start = time.time()
    fold_results = []

    # Pre-correct globally (proxy is unsupervised; ComBat is unsupervised)
    # NOTE: For baseline, use within-sample rank normalization (the deployable
    # honest baseline). For ComBat configs, use ComBat-corrected expression
    # directly (as in the original supervised ComBat experiment that produced
    # the 0.767 leaked result).
    try:
        if cov_spec is None:
            # Baseline: within-sample rank + inv-normal (the honest 0.602 baseline)
            corrected_global = within_sample_rank_inv_norm(pooled_expr)
        elif len(cov_spec) == 0:
            corrected_global = run_combat_no_covariate(pooled_expr, batch_series)
        else:
            cov_dict = {}
            for k, v in cov_spec.items():
                if v == "_LABEL":
                    cov_dict[k] = pooled_labels
                else:
                    cov_dict[k] = proxy_df[v]
            corrected_global = run_combat_with_covariates(pooled_expr, batch_series, cov_dict)
    except Exception as e:
        log.warning(f"  Correction failed: {e}")
        continue

    for holdout_id in dataset_ids:
        train_s = [s for s in pooled_expr.index if s2d[s] != holdout_id]
        test_s = [s for s in pooled_expr.index if s2d[s] == holdout_id]
        train_y = pooled_labels.loc[train_s].values.astype(int)
        test_y = pooled_labels.loc[test_s].values.astype(int)
        if len(np.unique(test_y)) < 2 or len(np.unique(train_y)) < 2:
            continue

        X_tr = clean_matrix(corrected_global.loc[train_s]).values
        X_te = clean_matrix(corrected_global.loc[test_s]).values

        mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
        mdl.fit(X_tr, train_y)
        tr_pred = mdl.predict_proba(X_tr)[:, 1]
        threshold = tune_threshold(train_y, tr_pred)
        te_pred = mdl.predict_proba(X_te)[:, 1]
        metrics = evaluate_predictions(test_y, te_pred, threshold)

        fold_results.append({
            "config": config_name, "holdout": holdout_id,
            "production_candidate": prod_cand,
            **metrics,
        })

    if fold_results:
        df = pd.DataFrame(fold_results)
        log.info(f"  → {config_name}: AUROC={df['auroc'].mean():.4f}  AUPRC={df['auprc'].mean():.4f}  MCC={df['mcc'].mean():.4f}  ({time.time()-t_start:.0f}s)")
        all_results.extend(fold_results)


# =========================================================================
# PHASE 4: INFORMATION RECOVERY
# =========================================================================
log.info("=" * 60)
log.info("PHASE 4: Information recovery")
log.info("=" * 60)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "all_results.tsv", sep="\t", index=False)

summary = (
    results_df.groupby(["config", "production_candidate"])
    .agg(mean_auroc=("auroc","mean"), mean_auprc=("auprc","mean"),
         mean_mcc=("mcc","mean"), mean_bal_acc=("bal_acc","mean"),
         n_folds=("auroc","count"))
    .reset_index().sort_values("mean_auroc", ascending=False)
)
summary.to_csv(OUT_DIR / "summary.tsv", sep="\t", index=False)

# Recovery analysis
baseline_auroc = summary[summary["config"] == "baseline_no_combat"]["mean_auroc"].iloc[0] if "baseline_no_combat" in summary["config"].values else 0.602
baseline_mcc = summary[summary["config"] == "baseline_no_combat"]["mean_mcc"].iloc[0] if "baseline_no_combat" in summary["config"].values else 0.157
leaked_auroc = summary[summary["config"] == "combat_response_LEAKED"]["mean_auroc"].iloc[0] if "combat_response_LEAKED" in summary["config"].values else 0.767
leaked_mcc = summary[summary["config"] == "combat_response_LEAKED"]["mean_mcc"].iloc[0] if "combat_response_LEAKED" in summary["config"].values else 0.457

summary["auroc_recovery"] = ((summary["mean_auroc"] - baseline_auroc) / (leaked_auroc - baseline_auroc)).clip(-0.5, 1.5)
summary["mcc_recovery"] = ((summary["mean_mcc"] - baseline_mcc) / (leaked_mcc - baseline_mcc)).clip(-0.5, 1.5)
summary.to_csv(OUT_DIR / "information_recovery.tsv", sep="\t", index=False)

log.info(f"\nBaseline (no ComBat): AUROC={baseline_auroc:.4f}, MCC={baseline_mcc:.4f}")
log.info(f"Leaked (response covariate): AUROC={leaked_auroc:.4f}, MCC={leaked_mcc:.4f}")
log.info(f"Gap to recover: AUROC={leaked_auroc-baseline_auroc:+.4f}, MCC={leaked_mcc-baseline_mcc:+.4f}")
log.info(f"\nFull comparison:")
for _, r in summary.iterrows():
    tag = "" if r["production_candidate"] else " [LEAKED upper bound]"
    log.info(f"  {r['config']:35s}  AUROC={r['mean_auroc']:.4f}  MCC={r['mean_mcc']:.4f}  recovery={r['auroc_recovery']:+.1%}{tag}")

# NMI vs recovery correlation
prod = summary[summary["production_candidate"]].copy()
log.info(f"\nDecision gate:")
best_proxy_auroc = prod["mean_auroc"].max()
log.info(f"  Best proxy AUROC: {best_proxy_auroc:.4f}")
if best_proxy_auroc >= 0.635:
    log.info(f"  ✓ PASSED — proxy harmonization recovers > 20% of leaked advantage")
elif best_proxy_auroc >= 0.620:
    log.info(f"  ~ MARGINAL — partial recovery (~10-20%)")
else:
    log.info(f"  ✗ FAILED — no proxy materially improves over baseline")

log.info(f"\nDone. Results in {OUT_DIR}/")
