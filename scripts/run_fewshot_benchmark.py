#!/usr/bin/env python
"""
Run the full few-shot LODO adaptation benchmark.

Tests L1-logistic, feature calibration, and NN fine-tuning across
CTR-DB datasets with support sizes k=0,5,10,20 and 5 random repeats.

Saves:
  - results/fewshot_by_dataset.tsv
  - results/fewshot_summary.tsv
  - results/fewshot_by_endpoint.tsv

Usage:
    pixi run python scripts/run_fewshot_benchmark.py
"""
import logging
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_CACHE, DATA_RAW, RESULTS
from src.models.fewshot_adaptation import (
    FeatureCalibrationMethod,
    FewShotLODOBenchmark,
    L1LogisticMethod,
    NNFineTuneMethod,
    aggregate_fewshot_results,
)
from src.preprocessing.response_handler import load_response_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fewshot_benchmark")

CTRDB_DIR = DATA_RAW / "ctrdb"
META_COLS = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}

# ── Drug name utilities ──────────────────────────────────────────────────

DRUG_ALIASES = {
    "5-fluorouracil": "fluorouracil", "5-fu": "fluorouracil",
    "5fu": "fluorouracil", "adriamycin": "doxorubicin",
    "taxol": "paclitaxel", "nolvadex": "tamoxifen",
    "xeloda": "capecitabine", "taxotere": "docetaxel",
    "gemzar": "gemcitabine", "mtx": "methotrexate",
    "ctx": "cyclophosphamide", "cytoxan": "cyclophosphamide",
    "epirubicin": "epirubicin", "ixabepilone": "ixabepilone",
    "mk-2206": "MK-2206", "trastuzumab": "trastuzumab",
}


def _normalise_drug_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[\s\-]+", "", s)
    for alias, canonical in DRUG_ALIASES.items():
        if s == re.sub(r"[\s\-]+", "", alias):
            return canonical.lower()
    return s


def parse_regimen_components(drug_string: str) -> list[str]:
    s = drug_string.strip()
    paren_match = re.search(r"\(([^)]+)\)", s)
    inner = paren_match.group(1) if paren_match else s
    parts = re.split(r"[+/]", inner)
    components = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^[A-Z]{1,5}\s+", "", p)
        norm = _normalise_drug_name(p)
        if norm and len(norm) > 1:
            components.append(norm)
    return list(dict.fromkeys(components))


def match_drugs_to_lincs(components: list[str], lincs_drug_set: set[str]) -> list[str]:
    lincs_norm = {_normalise_drug_name(d): d for d in lincs_drug_set}
    matched = []
    for comp in components:
        cn = _normalise_drug_name(comp)
        if cn in lincs_norm:
            matched.append(lincs_norm[cn])
    return matched


def get_drug_for_dataset(geo_id, catalog, pan_catalog):
    for cat in [catalog, pan_catalog]:
        if cat.empty:
            continue
        row = cat[cat["geo_source"] == geo_id]
        if not row.empty:
            return str(row.iloc[0]["drug"])
    return ""


# ── Main ────���────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("FEW-SHOT LODO ADAPTATION BENCHMARK")
    logger.info("=" * 70)

    # Load response handler
    handler = load_response_handler(policy="lenient")

    # Load catalogs
    cat_path = CTRDB_DIR / "catalog.csv"
    pan_cat_path = CTRDB_DIR / "pan_cancer_catalog.csv"
    catalog = pd.read_csv(cat_path) if cat_path.exists() else pd.DataFrame()
    pan_catalog = pd.read_csv(pan_cat_path) if pan_cat_path.exists() else pd.DataFrame()

    # Load landmark genes
    gi = pd.read_csv(DATA_CACHE / "geneinfo_beta_input.txt", sep="\t")
    landmark_genes = gi["gene_symbol"].tolist()
    logger.info(f"Landmark genes: {len(landmark_genes)}")

    # Load LINCS signatures
    all_path = DATA_CACHE / "all_cellline_drug_signatures.parquet"
    breast_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    if all_path.exists():
        lincs_sigs = pd.read_parquet(all_path)
    elif breast_path.exists():
        lincs_sigs = pd.read_parquet(breast_path)
    else:
        logger.error("No LINCS signatures found")
        return

    lincs_gene_cols = [c for c in lincs_sigs.columns if c not in META_COLS]
    lincs_drug_set = set(lincs_sigs["pert_iname"].str.lower().unique())

    # Compute per-drug mean signatures
    drug_mean_gene = {}
    for drug, grp in lincs_sigs.groupby(lincs_sigs["pert_iname"].str.lower()):
        drug_mean_gene[drug] = grp[lincs_gene_cols].mean(axis=0)

    # Load and process datasets
    logger.info("Loading and processing datasets ...")
    datasets = {}

    for ds_dir in sorted(CTRDB_DIR.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("GSE"):
            continue
        geo_id = ds_dir.name
        expr_files = list(ds_dir.glob("*_expression.parquet"))
        label_file = ds_dir / "response_labels.parquet"
        if not expr_files or not label_file.exists():
            continue

        expr = pd.read_parquet(expr_files[0])
        labels = pd.read_parquet(label_file)["response"]
        common = expr.index.intersection(labels.index)
        if len(common) < 5:
            continue

        expr = expr.loc[common]
        labels = labels.loc[common]

        ef = handler.get_endpoint_family(geo_id)
        harmonized = handler.harmonize_labels(geo_id, labels, endpoint_family=ef)
        if harmonized is None:
            continue

        available_genes = [g for g in landmark_genes if g in expr.columns]
        if len(available_genes) < 50:
            continue

        # Z-score within dataset
        expr_lm = expr[available_genes].copy()
        cohort_mean = expr_lm.mean(axis=0)
        cohort_std = expr_lm.std(axis=0).replace(0, 1)
        expr_z = (expr_lm - cohort_mean) / cohort_std

        # Match drugs
        drug_str = get_drug_for_dataset(geo_id, catalog, pan_catalog)
        if not drug_str:
            continue
        components = parse_regimen_components(drug_str)
        matched_drugs = match_drugs_to_lincs(components, lincs_drug_set)
        if not matched_drugs:
            continue

        # Compute drug signature
        drug_sig = np.zeros(len(available_genes), dtype=np.float64)
        n_d = 0
        for d in matched_drugs:
            dl = d.lower()
            if dl in drug_mean_gene:
                sig = drug_mean_gene[dl].reindex(available_genes).values.astype(np.float64)
                drug_sig += np.nan_to_num(sig, 0.0)
                n_d += 1
        if n_d == 0:
            continue
        drug_sig /= n_d

        # Reversal features
        X_rev = expr_z.values * drug_sig[np.newaxis, :]
        y = harmonized.values.astype(int)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 3 or n_neg < 3:
            continue

        datasets[geo_id] = {
            "X": X_rev,
            "y": y,
            "endpoint_family": ef,
            "drug": drug_str,
        }

    logger.info(f"Prepared {len(datasets)} datasets for benchmark")
    handler.flush_log()

    if len(datasets) < 2:
        logger.error("Not enough datasets. Aborting.")
        return

    # ── Define methods ───────────────────────────────────────────────────
    methods = [
        L1LogisticMethod(C=0.05),
        FeatureCalibrationMethod(C=0.05),
        NNFineTuneMethod(
            hidden_dims=[256, 128],
            source_epochs=100,
            finetune_epochs=15,
            finetune_lr=1e-4,
            patience=10,
        ),
    ]

    # ── Run benchmark ────────────────────────────────────────────────────
    benchmark = FewShotLODOBenchmark(
        datasets=datasets,
        methods=methods,
        support_sizes=[0, 5, 10, 20],
        n_repeats=5,
        min_dataset_size=30,
    )

    logger.info("Running few-shot LODO benchmark ...")
    results_df = benchmark.run()

    if results_df.empty:
        logger.error("No results. Aborting.")
        return

    # Save per-dataset results
    results_path = RESULTS / "fewshot_by_dataset.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    logger.info(f"Per-dataset results saved to {results_path}")

    # Aggregate
    summary_df, endpoint_df = aggregate_fewshot_results(results_df)

    summary_path = RESULTS / "fewshot_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Summary saved to {summary_path}")

    endpoint_path = RESULTS / "fewshot_by_endpoint.tsv"
    endpoint_df.to_csv(endpoint_path, sep="\t", index=False)
    logger.info(f"By-endpoint results saved to {endpoint_path}")

    # ── Check stop conditions for MAML ───────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STOP CONDITION CHECK FOR MAML")
    logger.info("=" * 70)

    l1_zero = summary_df[(summary_df["method"] == "l1_logistic") & (summary_df["k"] == 0)]
    nn_zero = summary_df[(summary_df["method"] == "nn_finetune") & (summary_df["k"] == 0)]
    nn_adapted = summary_df[(summary_df["method"] == "nn_finetune") & (summary_df["k"] > 0)]

    if not l1_zero.empty and not nn_zero.empty:
        l1_auc = l1_zero["mean_auroc"].values[0]
        nn_auc = nn_zero["mean_auroc"].values[0]
        logger.info(f"L1-logistic zero-shot AUROC: {l1_auc:.4f}")
        logger.info(f"NN zero-shot AUROC: {nn_auc:.4f}")

        if nn_auc < l1_auc - 0.05:
            logger.info(
                "STOP: NN zero-shot < L1-logistic - 0.05. "
                "Skipping MAML."
            )
        elif not nn_adapted.empty:
            best_nn_adapted = nn_adapted["mean_auroc"].max()
            logger.info(f"Best NN adapted AUROC: {best_nn_adapted:.4f}")
            if best_nn_adapted <= nn_auc:
                logger.info(
                    "STOP: NN fine-tuning does not beat NN zero-shot. "
                    "Skipping MAML."
                )
            else:
                logger.info(
                    "NN fine-tuning improves over zero-shot. "
                    "MAML would be warranted but is not implemented yet "
                    "as learn2learn is not available."
                )

    # ── Print summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FEW-SHOT BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total evaluations: {len(results_df)}")
    logger.info(f"Datasets: {results_df['held_out_dataset'].nunique()}")
    logger.info("\nSummary table:")
    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['method']:25s} k={int(row['k']):2d}: "
            f"AUROC={row['mean_auroc']:.4f} +/- {row['std_auroc']:.4f} "
            f"({int(row['n_datasets'])} datasets)"
        )


if __name__ == "__main__":
    main()
