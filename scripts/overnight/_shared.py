"""
Shared utilities for the overnight benchmark sprint.

Provides:
  - Data loading (38 datasets, 5198 patients, 918 genes, with platform + drug metadata)
  - Feature builders (rank, singscore, ssGSEA, REO knowledge)
  - LODO loop with 5 metrics + threshold tuning
  - Run logging in standardized format
"""

import os, sys, json, time, warnings, hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, accuracy_score, precision_score,
    recall_score, brier_score_loss, confusion_matrix,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "results" / "overnight_model_signal_benchmark"
BENCH.mkdir(parents=True, exist_ok=True)
for sub in ("configs", "logs", "raw_metrics", "plots", "reports", "models", "diagnostics", "summaries"):
    (BENCH / sub).mkdir(exist_ok=True)

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
    "GSE104958": "rnaseq", "ISPY2": "agilent", "BrighTNess": "rnaseq",
}

DATASET_DRUG_MAP = {
    "GSE25066":  "anthracycline+taxane",
    "GSE20194":  "anthracycline+taxane",
    "GSE20271":  "anthracycline+taxane",
    "GSE22093":  "anthracycline+taxane",
    "GSE23988":  "taxane",
    "GSE37946":  "trastuzumab+chemo",
    "GSE41998":  "anthracycline+ixabepilone",
    "GSE5122":   "tipifarnib",
    "GSE8970":   "tipifarnib",
    "GSE131978": "platinum",
    "GSE20181":  "letrozole",
    "GSE14615":  "induction_ALL",
    "GSE14671":  "imatinib",
    "GSE19293":  "melphalan",
    "GSE28702":  "folfox",
    "GSE32646":  "anthracycline+taxane",
    "GSE35640":  "immunotherapy",
    "GSE48905":  "endocrine",
    "GSE50948":  "anthracycline+taxane+trastuzumab",
    "GSE63885":  "platinum",
    "GSE68871":  "vtd_myeloma",
    "GSE72970":  "chemo+targeted_crc",
    "GSE73578":  "glucocorticoids",
    "GSE82172":  "tamoxifen",
    "GSE104645": "folfox",
    "GSE104958": "platinum+taxane",
    "GSE109211": "sorafenib",
    "GSE173263": "rchop",
    "GSE21974":  "taxane",
    "GSE44272":  "trastuzumab",
    "GSE4779":   "anthracycline_chemo",
    "GSE65021":  "cetuximab+platinum",
    "GSE66999":  "anthracycline+taxane",
    "GSE6861":   "anthracycline_chemo",
    "GSE76360":  "trastuzumab",
    "GSE62321":  "chemo",
    "ISPY2":     "anthracycline+taxane",
    "BrighTNess": "parp_chemo",
}

# Cancer type / indication grouping
DATASET_CANCER_MAP = {
    "GSE25066": "breast", "GSE20194": "breast", "GSE20271": "breast",
    "GSE22093": "breast", "GSE23988": "breast", "GSE37946": "breast",
    "GSE41998": "breast", "GSE20181": "breast", "GSE14671": "cml",
    "GSE131978": "ovarian", "GSE14615": "all_leukemia",
    "GSE19293": "melanoma", "GSE28702": "colorectal",
    "GSE32646": "breast", "GSE35640": "melanoma_immuno",
    "GSE48905": "breast", "GSE50948": "breast", "GSE63885": "ovarian",
    "GSE68871": "myeloma", "GSE72970": "colorectal",
    "GSE73578": "all_leukemia", "GSE82172": "breast",
    "GSE104645": "colorectal", "GSE104958": "esophageal",
    "GSE109211": "hcc", "GSE173263": "dlbcl",
    "GSE21974": "breast", "GSE44272": "breast", "GSE4779": "breast",
    "GSE65021": "head_neck", "GSE66999": "breast", "GSE6861": "breast",
    "GSE76360": "breast", "GSE62321": "colorectal",
    "GSE5122": "breast", "GSE8970": "breast",
    "ISPY2": "breast", "BrighTNess": "breast",
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
    return pd.DataFrame(
        norm.ppf(np.clip(quantiles, 1e-7, 1 - 1e-7)),
        index=expression_df.index, columns=expression_df.columns,
    )


def compute_singscore(expression_df, gene_list):
    import gseapy as gp
    hallmark = gp.get_library("MSigDB_Hallmark_2020")
    ranks = expression_df[gene_list].rank(axis=1)
    n_genes = len(gene_list)
    scores = {}
    for pw, pw_genes in hallmark.items():
        present = [g for g in pw_genes if g in gene_list]
        if len(present) < 5:
            continue
        scores[f"singscore_{pw}"] = (ranks[present].mean(axis=1) / n_genes - 0.5) * 2
    return pd.DataFrame(scores, index=expression_df.index)


def compute_ssgsea(expression_df):
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


CURATED_PAIRS = [
    ("ERBB2", "ESR1"), ("MKI67", "ACTB"), ("CCND1", "CDKN1A"),
    ("AKT1", "PTEN"), ("CASP3", "BCL2"),
]

def build_reo_knowledge(expression_df):
    features = {}
    for ga, gb in CURATED_PAIRS:
        if ga in expression_df.columns and gb in expression_df.columns:
            features[f"reo_{ga}_gt_{gb}"] = (expression_df[ga] > expression_df[gb]).astype(int)
    return pd.DataFrame(features, index=expression_df.index)


def tune_threshold(y_true, y_proba):
    """MCC-optimal threshold on training predictions."""
    thresholds = np.linspace(0.05, 0.95, 91)
    mccs = [matthews_corrcoef(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(mccs))])


def evaluate_predictions(y_true, y_proba, threshold):
    """Compute all 8+ metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    r = {
        "threshold": threshold, "n_pos": n_pos, "n_neg": n_neg,
        "n_test": int(len(y_true)),
    }
    try: r["auroc"] = roc_auc_score(y_true, y_proba)
    except: r["auroc"] = float("nan")
    try: r["auprc"] = average_precision_score(y_true, y_proba)
    except: r["auprc"] = float("nan")
    try: r["mcc"] = matthews_corrcoef(y_true, y_pred)
    except: r["mcc"] = float("nan")
    try: r["bal_acc"] = balanced_accuracy_score(y_true, y_pred)
    except: r["bal_acc"] = float("nan")
    try: r["accuracy"] = accuracy_score(y_true, y_pred)
    except: r["accuracy"] = float("nan")
    try: r["precision"] = precision_score(y_true, y_pred, zero_division=0)
    except: r["precision"] = float("nan")
    try: r["recall"] = recall_score(y_true, y_pred, zero_division=0)
    except: r["recall"] = float("nan")
    # specificity = TN / (TN + FP)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        r["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    except:
        r["specificity"] = float("nan")
    try: r["brier"] = brier_score_loss(y_true, y_proba)
    except: r["brier"] = float("nan")
    return r


def load_baseline_data():
    """
    Load the canonical 38-dataset, 918-gene baseline.
    Returns dict with everything needed for any experiment.
    """
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

    # Restrict, z-score per dataset, pool
    parts_x, parts_y, batch_list = [], [], []
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
        parts_x.append(es); parts_y.append(lc); batch_list.extend([did]*len(es))
        for s_ in idx: s2d[s_] = did

    pooled_expr = pd.concat(parts_x)
    pooled_labels = pd.concat(parts_y)
    batch_series = pd.Series(batch_list, index=pooled_expr.index)
    tech_series = pd.Series([PLATFORM_MAP.get(d,"unknown") for d in batch_list], index=pooled_expr.index)
    drug_series = pd.Series([DATASET_DRUG_MAP.get(d,"unknown") for d in batch_list], index=pooled_expr.index)
    cancer_series = pd.Series([DATASET_CANCER_MAP.get(d,"unknown") for d in batch_list], index=pooled_expr.index)

    return {
        "datasets": datasets,
        "common_genes": common_genes,
        "pooled_expr": pooled_expr,
        "pooled_labels": pooled_labels,
        "batch_series": batch_series,
        "tech_series": tech_series,
        "drug_series": drug_series,
        "cancer_series": cancer_series,
        "s2d": s2d,
    }


def build_baseline_features(data):
    """Build the baseline 970-feature matrix: rank + singscore + REO."""
    pooled_expr = data["pooled_expr"]
    common_genes = data["common_genes"]

    rank_expr = within_sample_rank_inv_norm(pooled_expr)
    singscore = compute_singscore(pooled_expr, common_genes)
    reo = build_reo_knowledge(pooled_expr)

    X = pd.concat([rank_expr, singscore, reo], axis=1)
    return clean_matrix(X)


def run_lodo_loop(data, X, model_factory, config_name, log_fn=None):
    """
    Generic LODO loop. Returns list of fold dicts.

    model_factory: callable() → fresh model with .fit and .predict_proba
    """
    pooled_labels = data["pooled_labels"]
    s2d = data["s2d"]
    dataset_ids = sorted(set(s2d.values()))

    fold_results = []
    for holdout_id in dataset_ids:
        train_s = [s for s in X.index if s2d[s] != holdout_id]
        test_s = [s for s in X.index if s2d[s] == holdout_id]
        train_y = pooled_labels.loc[train_s].values.astype(int)
        test_y = pooled_labels.loc[test_s].values.astype(int)

        if len(np.unique(test_y)) < 2 or len(np.unique(train_y)) < 2:
            continue
        if len(test_y) < 5:
            continue

        X_tr = X.loc[train_s].values
        X_te = X.loc[test_s].values

        try:
            mdl = model_factory()
            mdl.fit(X_tr, train_y)
            tr_pred = mdl.predict_proba(X_tr)[:, 1]
            threshold = tune_threshold(train_y, tr_pred)
            te_pred = mdl.predict_proba(X_te)[:, 1]
            metrics = evaluate_predictions(test_y, te_pred, threshold)
        except Exception as e:
            if log_fn: log_fn(f"  {holdout_id}: failed: {e}")
            continue

        fold_results.append({
            "config": config_name,
            "holdout": holdout_id,
            "holdout_drug": DATASET_DRUG_MAP.get(holdout_id, "unknown"),
            "holdout_tech": PLATFORM_MAP.get(holdout_id, "unknown"),
            "holdout_cancer": DATASET_CANCER_MAP.get(holdout_id, "unknown"),
            **metrics,
        })

    return fold_results


def aggregate_run(fold_results):
    """Mean ± std for each metric across folds."""
    if not fold_results:
        return {}
    df = pd.DataFrame(fold_results)
    metric_cols = ["auroc", "auprc", "mcc", "bal_acc", "accuracy", "precision", "recall", "specificity", "brier"]
    summary = {}
    for m in metric_cols:
        if m in df.columns:
            summary[f"mean_{m}"] = df[m].mean()
            summary[f"std_{m}"] = df[m].std()
            summary[f"median_{m}"] = df[m].median()
    summary["n_folds"] = len(df)
    return summary


def write_run_metrics(agent_name, run_id, fold_results, config=None):
    """Write per-fold metrics to BENCH/raw_metrics/."""
    if not fold_results:
        return None
    df = pd.DataFrame(fold_results)
    df["run_id"] = run_id
    df["agent"] = agent_name
    if config:
        df["config_json"] = json.dumps(config)
    out_path = BENCH / "raw_metrics" / f"{agent_name}_{run_id}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def append_to_master(agent_name, run_id, summary, config=None, feature_set="baseline_970"):
    """Append a run's summary to master all_runs.csv."""
    master = BENCH / "summaries" / "all_runs.csv"
    row = {
        "agent": agent_name,
        "run_id": run_id,
        "feature_set": feature_set,
        "timestamp": datetime.now().isoformat(),
        **summary,
    }
    if config:
        row["config"] = json.dumps(config)

    df = pd.DataFrame([row])
    if master.exists():
        existing = pd.read_csv(master)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(master, index=False)


def setup_logger(agent_name):
    import logging
    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(BENCH / "logs" / f"{agent_name}.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(sh)
    return logger
