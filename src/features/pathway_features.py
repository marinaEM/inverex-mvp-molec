"""
Pathway-level feature engineering using ssGSEA (gseapy).

Collapses individual gene expression / drug z-score values into pathway
activity scores.  This reduces dimensionality from ~978 genes to ~50-300
pathway scores, which should generalise better across datasets and
platforms.

Gene set collections used
-------------------------
- MSigDB Hallmark 2020 (50 pathways) -- primary set
- KEGG 2021 Human (~300 pathways) -- secondary
- Reactome 2022 (~1600 pathways) -- optional, can be slow

The main entry point is ``compute_ssgsea_scores`` which wraps ``gseapy.ssgsea``
and returns a tidy samples x pathways DataFrame.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.config import DATA_CACHE, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gene set collection names (Enrichr / gseapy)
# ---------------------------------------------------------------------------
HALLMARK = "MSigDB_Hallmark_2020"
KEGG = "KEGG_2021_Human"
REACTOME = "Reactome_2022"

DEFAULT_COLLECTIONS = [HALLMARK, KEGG]  # Reactome is optional


# ---------------------------------------------------------------------------
# Core ssGSEA wrapper
# ---------------------------------------------------------------------------

def compute_ssgsea_scores(
    expr_df: pd.DataFrame,
    gene_sets: str = HALLMARK,
    min_size: int = 5,
    max_size: int = 500,
    threads: int = 4,
    use_nes: bool = True,
) -> pd.DataFrame:
    """
    Compute ssGSEA pathway activity scores for a samples x genes matrix.

    Parameters
    ----------
    expr_df : DataFrame
        Samples (rows) x genes (columns).  Gene symbols as column names.
    gene_sets : str
        Enrichr library name (e.g. 'MSigDB_Hallmark_2020') or path to .gmt.
    min_size : int
        Minimum gene overlap for a pathway to be scored.
    max_size : int
        Maximum gene set size.
    threads : int
        Parallel threads for ssGSEA.
    use_nes : bool
        If True, return NES (normalised enrichment scores); else raw ES.

    Returns
    -------
    DataFrame : samples (rows) x pathways (columns), float64.
    """
    import gseapy as gp

    n_samples, n_genes = expr_df.shape
    logger.info(
        f"Running ssGSEA: {n_samples} samples x {n_genes} genes, "
        f"gene_sets={gene_sets}, min_size={min_size}"
    )

    t0 = time.time()

    # gseapy.ssgsea expects genes x samples (index = gene symbols)
    data_t = expr_df.T

    result = gp.ssgsea(
        data=data_t,
        gene_sets=gene_sets,
        outdir=None,
        min_size=min_size,
        max_size=max_size,
        no_plot=True,
        threads=threads,
        verbose=False,
    )

    # result.res2d has columns: Name (sample), Term (pathway), ES, NES
    score_col = "NES" if use_nes else "ES"
    pivot = result.res2d.pivot_table(
        index="Name", columns="Term", values=score_col, aggfunc="first"
    )

    # Ensure sample order matches input
    pivot = pivot.reindex(expr_df.index)
    pivot = pivot.astype(np.float64)

    elapsed = time.time() - t0
    logger.info(
        f"ssGSEA complete: {pivot.shape[1]} pathways scored "
        f"in {elapsed:.1f}s"
    )

    return pivot


def compute_multi_collection_scores(
    expr_df: pd.DataFrame,
    collections: Optional[list[str]] = None,
    min_size: int = 5,
    threads: int = 4,
) -> pd.DataFrame:
    """
    Run ssGSEA on multiple gene set collections and concatenate.

    Parameters
    ----------
    expr_df : DataFrame
        Samples x genes.
    collections : list of str, optional
        Gene set library names.  Defaults to Hallmark + KEGG.
    min_size : int
        Min gene overlap.
    threads : int
        Parallel threads.

    Returns
    -------
    DataFrame : samples x pathways (prefixed by collection name).
    """
    if collections is None:
        collections = DEFAULT_COLLECTIONS

    parts = []
    for coll in collections:
        logger.info(f"Computing ssGSEA for {coll} ...")
        try:
            scores = compute_ssgsea_scores(
                expr_df,
                gene_sets=coll,
                min_size=min_size,
                threads=threads,
            )
            # Prefix columns with collection name for uniqueness
            prefix = coll.split("_")[0]  # e.g. "MSigDB", "KEGG", "Reactome"
            scores.columns = [f"{prefix}_{c}" for c in scores.columns]
            parts.append(scores)
            logger.info(f"  {coll}: {scores.shape[1]} pathways")
        except Exception as e:
            logger.warning(f"  {coll} failed: {e}")

    if not parts:
        raise RuntimeError("All gene set collections failed")

    combined = pd.concat(parts, axis=1)
    logger.info(
        f"Combined pathway scores: {combined.shape[0]} samples x "
        f"{combined.shape[1]} pathways"
    )
    return combined


# ---------------------------------------------------------------------------
# Apply to LINCS drug signatures
# ---------------------------------------------------------------------------

def transform_lincs_to_pathways(
    lincs_df: Optional[pd.DataFrame] = None,
    cache_path: Optional[Path] = None,
    gene_sets: str = HALLMARK,
    min_size: int = 5,
    threads: int = 4,
) -> pd.DataFrame:
    """
    Transform LINCS drug z-score signatures into pathway perturbation scores.

    Parameters
    ----------
    lincs_df : DataFrame, optional
        Raw LINCS signatures (samples x [meta + genes]).  If None, loaded
        from cache.
    cache_path : Path, optional
        Where to save / load cached pathway scores.
    gene_sets : str
        Gene set library.
    min_size : int
        Min gene overlap.
    threads : int
        Parallel threads.

    Returns
    -------
    DataFrame with meta columns + pathway score columns.
    """
    if cache_path is None:
        gs_tag = gene_sets.replace(" ", "_")
        cache_path = DATA_CACHE / f"lincs_pathway_{gs_tag}.parquet"

    if cache_path.exists():
        logger.info(f"Loading cached LINCS pathway scores from {cache_path}")
        return pd.read_parquet(cache_path)

    # Load raw signatures if not provided
    if lincs_df is None:
        raw_path = DATA_CACHE / "breast_l1000_signatures.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"LINCS signatures not found at {raw_path}"
            )
        lincs_df = pd.read_parquet(raw_path)

    # Separate meta and gene columns
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id",
                 "dose_um", "pert_idose"}
    gene_cols = [c for c in lincs_df.columns if c not in meta_cols]
    meta = lincs_df[[c for c in lincs_df.columns if c in meta_cols]].copy()
    expr = lincs_df[gene_cols].astype(np.float64)

    # Compute ssGSEA
    pathway_scores = compute_ssgsea_scores(
        expr, gene_sets=gene_sets, min_size=min_size, threads=threads
    )

    # Re-attach metadata
    result = pd.concat([meta.reset_index(drop=True),
                        pathway_scores.reset_index(drop=True)], axis=1)

    result.to_parquet(cache_path, index=False)
    logger.info(f"Saved LINCS pathway scores to {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Apply to CTR-DB patient expression
# ---------------------------------------------------------------------------

def transform_ctrdb_to_pathways(
    ctrdb_datasets: Optional[dict] = None,
    gene_sets: str = HALLMARK,
    min_size: int = 5,
    threads: int = 4,
    cache_dir: Optional[Path] = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Transform CTR-DB patient expression datasets to pathway scores.

    Parameters
    ----------
    ctrdb_datasets : dict, optional
        geo_id -> (expression_df, response_series).  If None, loaded via
        ``load_all_breast_ctrdb``.
    gene_sets : str
        Gene set library name.
    min_size : int
        Minimum gene overlap.
    threads : int
        Parallel threads.
    cache_dir : Path, optional
        Where to cache results.  Defaults to DATA_CACHE / 'pathway_ctrdb'.

    Returns
    -------
    dict : geo_id -> (pathway_scores_df, response_series)
    """
    if cache_dir is None:
        gs_tag = gene_sets.replace(" ", "_")
        cache_dir = DATA_CACHE / f"pathway_ctrdb_{gs_tag}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if ctrdb_datasets is None:
        from src.data_ingestion.ctrdb import load_all_breast_ctrdb
        ctrdb_datasets = load_all_breast_ctrdb()

    transformed = {}
    for geo_id, (expr, labels) in sorted(ctrdb_datasets.items()):
        cache_path = cache_dir / f"{geo_id}_pathway.parquet"

        if cache_path.exists():
            pw_scores = pd.read_parquet(cache_path)
            pw_scores.index = pw_scores.index.astype(str)
            # Align labels
            common = labels.index.intersection(pw_scores.index)
            if len(common) > 0:
                transformed[geo_id] = (pw_scores.loc[common], labels.loc[common])
                logger.info(
                    f"  {geo_id}: loaded cached pathway scores "
                    f"({pw_scores.shape[1]} pathways, {len(common)} samples)"
                )
            continue

        try:
            pw_scores = compute_ssgsea_scores(
                expr, gene_sets=gene_sets, min_size=min_size, threads=threads
            )
            # Drop any all-NaN columns
            pw_scores = pw_scores.dropna(axis=1, how="all")
            pw_scores = pw_scores.fillna(0.0)

            pw_scores.to_parquet(cache_path)

            common = labels.index.intersection(pw_scores.index)
            if len(common) > 0:
                transformed[geo_id] = (pw_scores.loc[common], labels.loc[common])
                logger.info(
                    f"  {geo_id}: {pw_scores.shape[1]} pathways, "
                    f"{len(common)} samples"
                )
            else:
                logger.warning(f"  {geo_id}: no sample overlap after ssGSEA")
        except Exception as e:
            logger.warning(f"  {geo_id}: ssGSEA failed ({e})")

    logger.info(
        f"Transformed {len(transformed)}/{len(ctrdb_datasets)} "
        f"CTR-DB datasets to pathway scores"
    )
    return transformed


# ---------------------------------------------------------------------------
# Build reversal features (gene-level, pathway-level, or combined)
# ---------------------------------------------------------------------------

def build_reversal_features(
    patient_expr: np.ndarray,
    drug_sig: np.ndarray,
) -> np.ndarray:
    """
    Element-wise product of patient expression and drug signature.

    Negative product means the drug reverses the patient's dysregulation.
    """
    return patient_expr * drug_sig


# ---------------------------------------------------------------------------
# LODO evaluation
# ---------------------------------------------------------------------------

def run_lodo_evaluation(
    ctrdb_gene: dict[str, tuple[pd.DataFrame, pd.Series]],
    ctrdb_pathway: dict[str, tuple[pd.DataFrame, pd.Series]],
    lincs_sigs: pd.DataFrame,
    lincs_pathway: pd.DataFrame,
    catalog: pd.DataFrame,
    C: float = 0.05,
) -> pd.DataFrame:
    """
    Leave-One-Dataset-Out evaluation comparing gene, pathway, and combined
    features for predicting drug response.

    Parameters
    ----------
    ctrdb_gene : dict
        geo_id -> (gene_expression_df, labels).
    ctrdb_pathway : dict
        geo_id -> (pathway_scores_df, labels).
    lincs_sigs : DataFrame
        LINCS drug signatures (raw gene z-scores).
    lincs_pathway : DataFrame
        LINCS drug signatures as pathway scores.
    catalog : DataFrame
        CTR-DB catalog with geo_source and drug columns.
    C : float
        Inverse regularisation for L1-logistic.

    Returns
    -------
    DataFrame with per-dataset AUC for each feature set.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from src.models.recalibrate_signatures import (
        _normalise_drug_name,
        match_drugs_to_lincs,
        parse_regimen_components,
    )

    # Build per-drug mean signatures (gene level)
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id",
                 "dose_um", "pert_idose"}
    gene_cols = [c for c in lincs_sigs.columns if c not in meta_cols]
    drug_mean_gene = {}
    for drug, grp in lincs_sigs.groupby(lincs_sigs["pert_iname"].str.lower()):
        drug_mean_gene[drug] = grp[gene_cols].mean()

    # Build per-drug mean signatures (pathway level)
    pw_meta = {"sig_id", "pert_id", "pert_iname", "cell_id",
               "dose_um", "pert_idose"}
    pw_cols = [c for c in lincs_pathway.columns if c not in pw_meta]
    drug_mean_pw = {}
    for drug, grp in lincs_pathway.groupby(
        lincs_pathway["pert_iname"].str.lower()
    ):
        drug_mean_pw[drug] = grp[pw_cols].mean()

    lincs_drug_set = set(lincs_sigs["pert_iname"].str.lower().unique())

    # Get drug for each dataset from catalog
    def _get_drug(geo_id):
        row = catalog[catalog["geo_source"] == geo_id]
        if row.empty:
            return ""
        return str(row.iloc[0]["drug"])

    # Identify datasets usable for all three feature sets
    common_datasets = sorted(
        set(ctrdb_gene.keys()) & set(ctrdb_pathway.keys())
    )
    logger.info(f"LODO: {len(common_datasets)} datasets with both gene and pathway features")

    # ------------------------------------------------------------------
    # Determine a GLOBAL common gene set and pathway set so all datasets
    # have the same feature dimensionality (required for LODO stacking).
    # ------------------------------------------------------------------

    # Genes present in LINCS and in at least half of CTR-DB datasets
    gene_presence: dict[str, int] = {}
    eligible_datasets = []
    for geo_id in common_datasets:
        drug_str = _get_drug(geo_id)
        components = parse_regimen_components(drug_str)
        matched = match_drugs_to_lincs(components, lincs_drug_set)
        if not matched:
            continue
        expr_gene, labels_gene = ctrdb_gene[geo_id]
        expr_pw, labels_pw = ctrdb_pathway[geo_id]
        common_samples = (
            labels_gene.index
            .intersection(labels_pw.index)
            .intersection(expr_gene.index)
            .intersection(expr_pw.index)
        )
        labels = labels_gene.loc[common_samples]
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if len(common_samples) < 10 or n_pos < 3 or n_neg < 3:
            continue
        eligible_datasets.append(geo_id)
        for g in gene_cols:
            if g in expr_gene.columns:
                gene_presence[g] = gene_presence.get(g, 0) + 1

    n_eligible = len(eligible_datasets)
    threshold = max(1, n_eligible // 2)
    global_genes = sorted(
        [g for g, cnt in gene_presence.items() if cnt >= threshold]
    )
    logger.info(
        f"Global common gene set: {len(global_genes)} genes "
        f"(present in >= {threshold}/{n_eligible} eligible datasets)"
    )
    if len(global_genes) < 10:
        logger.warning("Too few common genes for LODO evaluation")
        return pd.DataFrame()

    # Pathways: all CTR-DB pathway datasets have the same 50 Hallmark
    # pathways, but we intersect to be safe.
    global_pw = None
    for geo_id in eligible_datasets:
        expr_pw, _ = ctrdb_pathway[geo_id]
        ds_pw = set(expr_pw.columns) & set(pw_cols)
        if global_pw is None:
            global_pw = ds_pw
        else:
            global_pw = global_pw & ds_pw
    global_pw = sorted(global_pw) if global_pw else []
    logger.info(f"Global common pathway set: {len(global_pw)} pathways")
    if len(global_pw) < 3:
        logger.warning("Too few common pathways for LODO evaluation")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Pre-compute per-dataset features aligned to global feature sets
    # ------------------------------------------------------------------
    dataset_features = {}

    for geo_id in eligible_datasets:
        drug_str = _get_drug(geo_id)
        components = parse_regimen_components(drug_str)
        matched = match_drugs_to_lincs(components, lincs_drug_set)
        if not matched:
            logger.info(f"  {geo_id}: no LINCS-matched drugs, skipping")
            continue

        expr_gene, labels_gene = ctrdb_gene[geo_id]
        expr_pw, labels_pw = ctrdb_pathway[geo_id]

        # Common samples across both representations
        common_samples = (
            labels_gene.index
            .intersection(labels_pw.index)
            .intersection(expr_gene.index)
            .intersection(expr_pw.index)
        )
        if len(common_samples) < 10:
            continue

        labels = labels_gene.loc[common_samples]
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos < 3 or n_neg < 3:
            continue

        # --- Gene-level reversal features (aligned to global_genes) ---
        # Mean drug signature across matched LINCS drugs
        drug_sig_gene = np.zeros(len(global_genes), dtype=np.float64)
        n_d = 0
        for d in matched:
            dl = d.lower()
            if dl in drug_mean_gene:
                sig = drug_mean_gene[dl]
                vals = sig.reindex(global_genes).values.astype(np.float64)
                drug_sig_gene += np.nan_to_num(vals, 0.0)
                n_d += 1
        if n_d > 0:
            drug_sig_gene /= n_d
        else:
            continue

        # Patient expression aligned to global_genes (zero-fill missing)
        X_gene = np.zeros(
            (len(common_samples), len(global_genes)), dtype=np.float64
        )
        for i, g in enumerate(global_genes):
            if g in expr_gene.columns:
                X_gene[:, i] = (
                    expr_gene.loc[common_samples, g]
                    .values.astype(np.float64)
                )
        X_gene = np.nan_to_num(X_gene, 0.0)
        X_gene_rev = X_gene * drug_sig_gene[np.newaxis, :]

        # --- Pathway-level reversal features (aligned to global_pw) ---
        drug_sig_pw = np.zeros(len(global_pw), dtype=np.float64)
        n_d = 0
        for d in matched:
            dl = d.lower()
            if dl in drug_mean_pw:
                sig = drug_mean_pw[dl]
                vals = sig.reindex(global_pw).values.astype(np.float64)
                drug_sig_pw += np.nan_to_num(vals, 0.0)
                n_d += 1
        if n_d > 0:
            drug_sig_pw /= n_d
        else:
            continue

        X_pw = np.zeros(
            (len(common_samples), len(global_pw)), dtype=np.float64
        )
        for i, p in enumerate(global_pw):
            if p in expr_pw.columns:
                X_pw[:, i] = (
                    expr_pw.loc[common_samples, p]
                    .values.astype(np.float64)
                )
        X_pw = np.nan_to_num(X_pw, 0.0)
        X_pw_rev = X_pw * drug_sig_pw[np.newaxis, :]

        # --- Combined ---
        X_combined = np.hstack([X_gene_rev, X_pw_rev])

        y = labels.values.astype(int)

        dataset_features[geo_id] = {
            "X_gene": X_gene_rev,
            "X_pw": X_pw_rev,
            "X_combined": X_combined,
            "y": y,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "drug": drug_str,
        }

    logger.info(
        f"LODO: {len(dataset_features)} datasets with valid features "
        f"(gene={len(global_genes)}, pathway={len(global_pw)}, "
        f"combined={len(global_genes)+len(global_pw)})"
    )

    if len(dataset_features) < 2:
        logger.warning("Not enough datasets for LODO evaluation")
        return pd.DataFrame()

    # LODO loop
    results = []
    all_geos = sorted(dataset_features.keys())

    for held_out in all_geos:
        train_geos = [g for g in all_geos if g != held_out]

        for feat_name, feat_key in [
            ("gene_only", "X_gene"),
            ("pathway_only", "X_pw"),
            ("gene_plus_pathway", "X_combined"),
        ]:
            # Assemble training data
            X_train_parts = []
            y_train_parts = []
            for tg in train_geos:
                X_train_parts.append(dataset_features[tg][feat_key])
                y_train_parts.append(dataset_features[tg]["y"])

            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            X_test = dataset_features[held_out][feat_key]
            y_test = dataset_features[held_out]["y"]

            # Standardise
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_train_s = np.nan_to_num(X_train_s, 0.0)
            X_test_s = np.nan_to_num(X_test_s, 0.0)

            # Train L1-logistic
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = LogisticRegression(
                    C=C,
                    l1_ratio=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                )
                try:
                    clf.fit(X_train_s, y_train)
                except Exception as e:
                    logger.warning(
                        f"  LODO {held_out}/{feat_name}: fit failed ({e})"
                    )
                    continue

            # Predict
            try:
                proba = clf.predict_proba(X_test_s)
                if proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    auc = 0.5
            except Exception:
                auc = 0.5

            results.append({
                "held_out_dataset": held_out,
                "feature_set": feat_name,
                "auc": round(auc, 4),
                "n_test": len(y_test),
                "n_test_pos": int(y_test.sum()),
                "n_train": len(y_train),
                "n_train_pos": int(y_train.sum()),
                "n_features": X_train.shape[1],
                "drug": dataset_features[held_out]["drug"],
            })

            logger.info(
                f"  LODO {held_out} [{feat_name}]: "
                f"AUC={auc:.3f} "
                f"(test={len(y_test)}, features={X_train.shape[1]})"
            )

    results_df = pd.DataFrame(results)
    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_pathway_features_pipeline():
    """
    End-to-end pipeline:
    1. Transform LINCS signatures to pathway scores
    2. Transform CTR-DB patient expression to pathway scores
    3. Run LODO evaluation (gene vs pathway vs combined)
    4. Save results
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.time()

    # ------------------------------------------------------------------ #
    # Step 0: Load raw data                                               #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("PATHWAY FEATURES PIPELINE")
    logger.info("=" * 60)

    logger.info("Loading LINCS breast signatures ...")
    lincs_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    if not lincs_path.exists():
        logger.error(f"LINCS signatures not found at {lincs_path}")
        return
    lincs_sigs = pd.read_parquet(lincs_path)
    logger.info(
        f"  {lincs_sigs.shape[0]} signatures, "
        f"{lincs_sigs['pert_iname'].nunique()} drugs"
    )

    logger.info("Loading CTR-DB patient datasets ...")
    from src.data_ingestion.ctrdb import load_all_breast_ctrdb
    ctrdb_gene = load_all_breast_ctrdb()
    logger.info(f"  {len(ctrdb_gene)} datasets loaded")

    if len(ctrdb_gene) == 0:
        logger.error("No CTR-DB datasets available.")
        return

    # Load catalog
    cat_path = DATA_RAW / "ctrdb" / "catalog.csv"
    if cat_path.exists():
        catalog = pd.read_csv(cat_path)
    else:
        logger.error("CTR-DB catalog not found.")
        return

    # ------------------------------------------------------------------ #
    # Step 1: Transform LINCS signatures to Hallmark pathway scores       #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Transforming LINCS signatures to pathway scores")
    logger.info("=" * 60)

    lincs_pathway = transform_lincs_to_pathways(
        lincs_df=lincs_sigs,
        gene_sets=HALLMARK,
        min_size=5,
        threads=4,
    )
    pw_meta = {"sig_id", "pert_id", "pert_iname", "cell_id",
               "dose_um", "pert_idose"}
    pw_feature_cols = [c for c in lincs_pathway.columns if c not in pw_meta]
    logger.info(f"  LINCS pathway features: {len(pw_feature_cols)} pathways")

    # ------------------------------------------------------------------ #
    # Step 2: Transform CTR-DB expression to pathway scores               #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Transforming CTR-DB datasets to pathway scores")
    logger.info("=" * 60)

    ctrdb_pathway = transform_ctrdb_to_pathways(
        ctrdb_datasets=ctrdb_gene,
        gene_sets=HALLMARK,
        min_size=5,
        threads=4,
    )

    # ------------------------------------------------------------------ #
    # Step 3: LODO evaluation                                             #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: LODO evaluation (gene vs pathway vs combined)")
    logger.info("=" * 60)

    lodo_results = run_lodo_evaluation(
        ctrdb_gene=ctrdb_gene,
        ctrdb_pathway=ctrdb_pathway,
        lincs_sigs=lincs_sigs,
        lincs_pathway=lincs_pathway,
        catalog=catalog,
        C=0.05,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Save results                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Saving results")
    logger.info("=" * 60)

    RESULTS.mkdir(parents=True, exist_ok=True)

    if not lodo_results.empty:
        # Per-dataset comparison
        comparison_path = RESULTS / "pathway_features_comparison.csv"
        lodo_results.to_csv(comparison_path, index=False)
        logger.info(f"  Saved per-dataset results to {comparison_path}")

        # Summary across datasets
        summary = (
            lodo_results
            .groupby("feature_set")["auc"]
            .agg(["mean", "std", "median", "count"])
            .round(4)
            .reset_index()
        )
        summary.columns = [
            "feature_set", "mean_auc", "std_auc", "median_auc", "n_datasets"
        ]
        summary_path = RESULTS / "pathway_features_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"  Saved summary to {summary_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        for _, row in summary.iterrows():
            logger.info(
                f"  {row['feature_set']:25s}: "
                f"AUC = {row['mean_auc']:.4f} +/- {row['std_auc']:.4f} "
                f"(median={row['median_auc']:.4f}, n={int(row['n_datasets'])})"
            )
    else:
        logger.warning("No LODO results to save")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal pipeline time: {elapsed:.0f}s")


if __name__ == "__main__":
    run_pathway_features_pipeline()
