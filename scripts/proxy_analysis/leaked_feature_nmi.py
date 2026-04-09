#!/usr/bin/env python
"""
Leaked-Feature NMI Analysis
============================

The previous proxy analysis asked: "Is there a variable derived from RAW
expression that has mutual information with response?" → Answer: No.

This script asks the sharper question: in the LEAKED ComBat output (where
labels were used as ComBat covariate, producing the 0.767 AUROC), which
specific features carry the leaked response information?

Steps:
  1. Run leaked ComBat (212 curated genes + labels as covariate)
  2. Compute ssGSEA pathways on the LEAKED ComBat output
  3. For each leaked feature (gene or pathway), compute NMI(feature, response)
  4. Compute the same NMI on the RAW (uncorrected) expression
  5. Compare: leaked_nmi vs raw_nmi

  - Features with high leaked_nmi but ~zero raw_nmi → fabricated by ComBat
  - Features with high leaked_nmi AND high raw_nmi → real biology amplified

This identifies which features ComBat injected response information into,
and validates whether the model is learning genuine biology or noise.

Output: results/proxy_analysis/leaked_feature_nmi.tsv
"""

import os, sys, time, json, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "proxy_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE = 500
MIN_PATIENTS = 20

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
    "GSE61676": "beadarray", "GSE104958": "rnaseq", "GSE9782": "targeted",
    "ISPY2": "agilent", "BrighTNess": "rnaseq",
}


def clean_matrix(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_ssgsea(expression_df, label=""):
    import gseapy as gp
    expr_clean = clean_matrix(expression_df)
    n = expr_clean.shape[0]
    if n <= CHUNK_SIZE:
        result = gp.ssgsea(
            data=expr_clean.T, gene_sets="MSigDB_Hallmark_2020",
            outdir=None, min_size=5, no_plot=True, verbose=False,
        )
        scores = result.res2d.pivot(index="Name", columns="Term", values="NES")
        scores.index.name = None; scores.columns.name = None
    else:
        chunks = []
        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            result = gp.ssgsea(
                data=expr_clean.iloc[start:end].T, gene_sets="MSigDB_Hallmark_2020",
                outdir=None, min_size=5, no_plot=True, verbose=False,
            )
            cs = result.res2d.pivot(index="Name", columns="Term", values="NES")
            cs.index.name = None; cs.columns.name = None
            chunks.append(cs)
        scores = pd.concat(chunks, axis=0)
    scores = scores.loc[expression_df.index].astype(float)
    scores.columns = [f"ssgsea_{c}" for c in scores.columns]
    return clean_matrix(scores)


def discretize(s, n_bins=5):
    """Discretize a continuous variable into n_bins quantile bins."""
    valid = s.dropna()
    if valid.nunique() <= 1:
        return pd.Series(["constant"] * len(s), index=s.index)
    n_bins = min(n_bins, valid.nunique())
    try:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        binned = kbd.fit_transform(valid.values.reshape(-1, 1)).flatten().astype(int)
        out = pd.Series(index=s.index, dtype=object)
        out.loc[valid.index] = [str(int(b)) for b in binned]
        return out.fillna("missing")
    except Exception:
        return s.astype(str)


def compute_feature_nmi(feature_series, response):
    """NMI between a single feature and response."""
    p_str = discretize(feature_series)
    valid = (p_str != "missing") & (p_str != "constant") & response.notna()
    if valid.sum() < 50:
        return np.nan
    p_v = p_str[valid]
    y_v = response[valid].astype(str)
    if p_v.nunique() < 2:
        return np.nan
    return normalized_mutual_info_score(y_v, p_v, average_method="arithmetic")


# =========================================================================
# Load data and 212 curated genes
# =========================================================================
log.info("Loading data + 212 curated genes...")

feat_imp = pd.read_csv(ROOT / "results" / "full_retrain" / "feature_importances.tsv", sep="\t")
genes_212 = sorted([f.replace("gene_", "") for f in feat_imp["feature"] if f.startswith("gene_")])
log.info(f"Curated genes: {len(genes_212)}")


def load_ds(geo_id):
    base = ROOT / "data" / "raw" / "ctrdb" / geo_id
    ep = base / f"{geo_id}_expression.parquet"
    lp = base / "response_labels.parquet"
    if not ep.exists() or not lp.exists(): return None, None
    expr = pd.read_parquet(ep); labels = pd.read_parquet(lp)
    common = expr.index.intersection(labels.index)
    if len(common) < MIN_PATIENTS: return None, None
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
    if expr is not None and labels is not None and labels.nunique() >= 2:
        datasets[geo_id] = (expr, labels)
for name, ddir, gid in [("ISPY2","ispy2","GSE194040"),("BrighTNess","brightness","GSE164458")]:
    try:
        e, l = load_pos(ddir, gid)
        if l.nunique() >= 2: datasets[name] = (e, l)
    except: pass

# Restrict to 212 genes, z-score per dataset, pool
all_px, all_py, all_batch = [], [], []
for did in sorted(datasets.keys()):
    e, l = datasets[did]
    avail = [g for g in genes_212 if g in e.columns]
    miss = [g for g in genes_212 if g not in e.columns]
    es = e[avail].copy()
    for g in miss: es[g] = 0.0
    es = es[genes_212]
    m = es.mean(axis=0); s = es.std(axis=0).replace(0, 1)
    es = clean_matrix((es - m) / s)
    idx = [f"{did}__{i}" for i in range(len(es))]
    es.index = idx; lc = l.copy(); lc.index = idx
    all_px.append(es); all_py.append(lc); all_batch.extend([did]*len(es))

pooled_expr = pd.concat(all_px)
pooled_labels = pd.concat(all_py)
batch_series = pd.Series(all_batch, index=pooled_expr.index)
log.info(f"Pooled: {pooled_expr.shape}")


# =========================================================================
# Step 1: Run leaked ComBat (with labels) and no-label ComBat for comparison
# =========================================================================
log.info("Running leaked ComBat (response as covariate)...")
from neuroCombat import neuroCombat

covars_leaked = pd.DataFrame({
    "batch": batch_series.values,
    "response": pooled_labels.values.astype(float),
}, index=pooled_expr.index)

result_leaked = neuroCombat(
    dat=pooled_expr.values.T,
    covars=covars_leaked,
    batch_col="batch",
    continuous_cols=["response"],
)
leaked_combat_expr = clean_matrix(pd.DataFrame(
    result_leaked["data"].T, columns=pooled_expr.columns, index=pooled_expr.index,
))
log.info(f"Leaked ComBat done")

log.info("Running ComBat WITHOUT labels...")
covars_nolabels = pd.DataFrame({"batch": batch_series.values}, index=pooled_expr.index)
result_nolabels = neuroCombat(
    dat=pooled_expr.values.T,
    covars=covars_nolabels,
    batch_col="batch",
)
nolabels_combat_expr = clean_matrix(pd.DataFrame(
    result_nolabels["data"].T, columns=pooled_expr.columns, index=pooled_expr.index,
))
log.info(f"No-label ComBat done")


# =========================================================================
# Step 2: Compute ssGSEA on raw, leaked, and no-label expression
# =========================================================================
log.info("Computing ssGSEA on raw expression...")
ssgsea_raw = compute_ssgsea(pooled_expr, "raw")
log.info(f"  ssGSEA raw: {ssgsea_raw.shape[1]} pathways")

log.info("Computing ssGSEA on LEAKED ComBat expression...")
ssgsea_leaked = compute_ssgsea(leaked_combat_expr, "leaked")
log.info(f"  ssGSEA leaked: {ssgsea_leaked.shape[1]} pathways")

log.info("Computing ssGSEA on no-label ComBat expression...")
ssgsea_nolabels = compute_ssgsea(nolabels_combat_expr, "nolabels")
log.info(f"  ssGSEA no-labels: {ssgsea_nolabels.shape[1]} pathways")


# =========================================================================
# Step 3: Compute NMI for each feature against response
# =========================================================================
log.info("Computing per-feature NMI...")

feature_nmi = []

# Genes
for gene in genes_212:
    nmi_raw = compute_feature_nmi(pooled_expr[gene], pooled_labels)
    nmi_leaked = compute_feature_nmi(leaked_combat_expr[gene], pooled_labels)
    nmi_nolabels = compute_feature_nmi(nolabels_combat_expr[gene], pooled_labels)
    feature_nmi.append({
        "feature": gene,
        "type": "gene",
        "nmi_raw": nmi_raw,
        "nmi_combat_no_labels": nmi_nolabels,
        "nmi_combat_leaked": nmi_leaked,
        "leaked_minus_raw": nmi_leaked - nmi_raw if (not np.isnan(nmi_leaked) and not np.isnan(nmi_raw)) else np.nan,
        "leaked_minus_nolabels": nmi_leaked - nmi_nolabels if (not np.isnan(nmi_leaked) and not np.isnan(nmi_nolabels)) else np.nan,
    })

# Pathways
common_paths = sorted(set(ssgsea_raw.columns) & set(ssgsea_leaked.columns) & set(ssgsea_nolabels.columns))
for path in common_paths:
    nmi_raw = compute_feature_nmi(ssgsea_raw[path], pooled_labels)
    nmi_leaked = compute_feature_nmi(ssgsea_leaked[path], pooled_labels)
    nmi_nolabels = compute_feature_nmi(ssgsea_nolabels[path], pooled_labels)
    feature_nmi.append({
        "feature": path,
        "type": "pathway",
        "nmi_raw": nmi_raw,
        "nmi_combat_no_labels": nmi_nolabels,
        "nmi_combat_leaked": nmi_leaked,
        "leaked_minus_raw": nmi_leaked - nmi_raw if (not np.isnan(nmi_leaked) and not np.isnan(nmi_raw)) else np.nan,
        "leaked_minus_nolabels": nmi_leaked - nmi_nolabels if (not np.isnan(nmi_leaked) and not np.isnan(nmi_nolabels)) else np.nan,
    })

nmi_df = pd.DataFrame(feature_nmi).sort_values("nmi_combat_leaked", ascending=False)
nmi_df.to_csv(OUT_DIR / "leaked_feature_nmi.tsv", sep="\t", index=False)


# =========================================================================
# Step 4: Report
# =========================================================================
log.info("\n" + "=" * 70)
log.info("LEAKED FEATURE NMI ANALYSIS")
log.info("=" * 70)
log.info("Comparing NMI(feature, response) across:")
log.info("  - raw:     z-scored expression (no ComBat)")
log.info("  - no-lab:  ComBat without labels")
log.info("  - LEAKED:  ComBat with response as continuous covariate")

log.info("\n--- TOP 25 FEATURES BY LEAKED NMI ---")
log.info(f"{'Feature':<45} {'Type':<10} {'NMI raw':>10} {'NMI nolab':>10} {'NMI leaked':>12} {'Δ leak-raw':>12}")
log.info("-" * 110)
for _, r in nmi_df.head(25).iterrows():
    log.info(
        f"{r['feature'][:44]:<45} {r['type']:<10} "
        f"{r['nmi_raw']:>10.4f} {r['nmi_combat_no_labels']:>10.4f} "
        f"{r['nmi_combat_leaked']:>12.4f} {r['leaked_minus_raw']:>+12.4f}"
    )

# Aggregate stats
log.info("\n--- AGGREGATE STATISTICS ---")
for col, label in [("nmi_raw", "Raw"), ("nmi_combat_no_labels", "ComBat no labels"), ("nmi_combat_leaked", "ComBat LEAKED")]:
    vals = nmi_df[col].dropna()
    log.info(
        f"  {label:25s}: mean={vals.mean():.4f}  median={vals.median():.4f}  "
        f"max={vals.max():.4f}  >0.05: {(vals > 0.05).sum()}/{len(vals)}  "
        f">0.10: {(vals > 0.10).sum()}/{len(vals)}"
    )

# How many features had leaked NMI > 5x raw NMI?
inflated = nmi_df[(nmi_df["nmi_raw"] < 0.01) & (nmi_df["nmi_combat_leaked"] > 0.05)]
log.info(f"\nFeatures with leaked NMI > 0.05 but raw NMI < 0.01 (fabricated by ComBat): {len(inflated)}")
if len(inflated) > 0:
    log.info("Top 15 fabricated:")
    for _, r in inflated.sort_values("nmi_combat_leaked", ascending=False).head(15).iterrows():
        log.info(f"  {r['feature']:<45} {r['type']:<10} raw={r['nmi_raw']:.4f}  leaked={r['nmi_combat_leaked']:.4f}")

# Features with high raw NMI (real biology)
real_biology = nmi_df[nmi_df["nmi_raw"] > 0.05].sort_values("nmi_raw", ascending=False)
log.info(f"\nFeatures with high RAW NMI (real biology, NMI(raw) > 0.05): {len(real_biology)}")
if len(real_biology) > 0:
    log.info("Top 15:")
    for _, r in real_biology.head(15).iterrows():
        log.info(f"  {r['feature']:<45} {r['type']:<10} raw={r['nmi_raw']:.4f}  leaked={r['nmi_combat_leaked']:.4f}")
else:
    log.info("  (none)")

log.info(f"\nDone. Full table: {OUT_DIR / 'leaked_feature_nmi.tsv'}")
