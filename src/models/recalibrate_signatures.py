"""
Context-stratified signature recalibration using patient response data.

Learns context-specific gene weights that recalibrate LINCS cell-line drug
signatures using real patient response data from CTR-DB.  The key insight:
not all gene expression changes caused by a drug are therapeutically relevant,
and which ones matter depends on the patient's molecular context (cancer type,
subtype, mutations).

Pipeline
--------
1. Load LINCS breast signatures (978 genes x ~20k signatures).
2. Load CTR-DB patient datasets (expression + binary response labels).
3. Infer molecular context for each patient from expression markers.
4. Group patients by context; ensure >= 20 per group.
5. For each context x drug, compute gene-level reversal features and train
   L1-regularized logistic regression to learn gene importance weights.
6. Learn cell-line relevance weights per context.
7. Learn combination drug weights per regimen.
8. Save a recalibrated signature bank and drug profiles.

Outputs
-------
- ``data/processed/recalibrated_signatures.json`` -- per-context gene weights,
  cell-line weights, AUCs, and patient counts.
- ``data/processed/recalibrated_drug_profiles.parquet`` -- context-weighted
  drug signatures emphasising therapeutically relevant gene changes.
- ``results/recalibration_validation.csv`` -- AUC comparison (uniform vs
  context-weighted reversal scores) per CTR-DB dataset.
"""

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CACHE, DATA_PROCESSED, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_GROUP_SIZE = 20          # Minimum patients for context-specific weights
MIN_GROUP_SIZE_BROAD = 30    # Minimum for broad "Breast" context
MARKER_GENES = {
    "ESR1": "er_status",     # ER status
    "ERBB2": "her2_status",  # HER2 status
    "MKI67": "proliferation",  # Proliferation
}

# Cell-line → molecular context mapping
CELL_LINE_CONTEXT = {
    "MCF7": "Breast_ER_positive",
    "SKBR3": "Breast_HER2_enriched",
    "MDAMB231": "Breast_Basal_like",
    "HS578T": "Breast_Basal_like",
}

# Drug name aliases for matching CTR-DB regimens to LINCS compounds
DRUG_ALIASES = {
    "5-fluorouracil": "fluorouracil",
    "5-fu": "fluorouracil",
    "5fu": "fluorouracil",
    "adriamycin": "doxorubicin",
    "taxol": "paclitaxel",
    "nolvadex": "tamoxifen",
    "xeloda": "capecitabine",
    "taxotere": "docetaxel",
    "gemzar": "gemcitabine",
    "mtx": "methotrexate",
    "ctx": "cyclophosphamide",
    "cytoxan": "cyclophosphamide",
}


# ---------------------------------------------------------------------------
# Drug name parsing
# ---------------------------------------------------------------------------

def _normalise_drug_name(name: str) -> str:
    """Lower-case, strip hyphens/spaces, apply aliases."""
    s = name.lower().strip()
    s = re.sub(r"[\s\-]+", "", s)
    # Check alias table (also stripped)
    for alias, canonical in DRUG_ALIASES.items():
        if s == re.sub(r"[\s\-]+", "", alias):
            return canonical
    return s


def parse_regimen_components(drug_string: str) -> list[str]:
    """
    Parse a CTR-DB drug string into individual component names.

    Examples
    --------
    >>> parse_regimen_components("TFAC (Cyclophosphamide+Doxorubicin+Fluorouracil+Paclitaxel)")
    ['cyclophosphamide', 'doxorubicin', 'fluorouracil', 'paclitaxel']
    >>> parse_regimen_components("Tamoxifen")
    ['tamoxifen']
    """
    s = drug_string.strip()

    # If there's a parenthetical list, extract it
    paren_match = re.search(r"\(([^)]+)\)", s)
    if paren_match:
        inner = paren_match.group(1)
    else:
        # Try the whole string -- might have + separators
        inner = s

    # Split on + or /
    parts = re.split(r"[+/]", inner)
    components = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Remove leading acronyms like "AC ", "FAC ", "FEC ", "TX ", "CMF "
        # These are regimen abbreviations, not drug names
        p = re.sub(r"^[A-Z]{1,5}\s+", "", p)
        norm = _normalise_drug_name(p)
        if norm and len(norm) > 1:
            components.append(norm)

    return list(dict.fromkeys(components))  # deduplicate, preserve order


def match_drugs_to_lincs(
    components: list[str],
    lincs_drug_set: set[str],
) -> list[str]:
    """Return the subset of *components* found in LINCS (normalised)."""
    matched = []
    lincs_norm = {_normalise_drug_name(d): d for d in lincs_drug_set}
    for comp in components:
        cn = _normalise_drug_name(comp)
        if cn in lincs_norm:
            matched.append(lincs_norm[cn])
    return matched


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SignatureRecalibrator:
    """Context-stratified signature recalibration using patient response data."""

    def __init__(
        self,
        lincs_signatures: pd.DataFrame,
        ctrdb_datasets: dict[str, tuple[pd.DataFrame, pd.Series]],
        landmark_genes: Optional[list[str]] = None,
    ):
        """
        Parameters
        ----------
        lincs_signatures : DataFrame
            Rows = signatures, columns include meta cols
            (sig_id, pert_id, pert_iname, cell_id, dose_um) + gene z-scores.
        ctrdb_datasets : dict
            geo_id -> (expression_df, response_series).
        landmark_genes : list of str, optional
            Gene symbols to use.  If None, derived from LINCS columns.
        """
        self.lincs_sigs = lincs_signatures.copy()

        # Identify gene columns (everything not a meta column)
        meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id",
                     "pert_idose", "dose_um"}
        self.gene_cols = [c for c in self.lincs_sigs.columns
                          if c not in meta_cols]

        if landmark_genes is not None:
            # Keep only those present in LINCS
            self.gene_cols = [g for g in self.gene_cols
                              if g in set(landmark_genes) or g in set(self.gene_cols)]
        logger.info(f"Using {len(self.gene_cols)} LINCS gene columns")

        self.ctrdb = ctrdb_datasets
        self.lincs_drug_set = set(
            self.lincs_sigs["pert_iname"].str.lower().unique()
        )

        # Build per-drug mean signatures (across all cell lines / doses)
        self._drug_mean_sigs: dict[str, pd.Series] = {}
        # Build per-drug-per-cell-line mean signatures
        self._drug_cell_sigs: dict[tuple[str, str], pd.Series] = {}
        self._build_drug_signature_cache()

        # Catalog for drug matching
        self._load_catalog()

        # Will be populated by build_recalibrated_bank()
        self.bank: dict = {}

    # -- Signature cache ---------------------------------------------------

    def _build_drug_signature_cache(self):
        """Pre-compute mean drug signatures (all cell lines, per cell line)."""
        logger.info("Building drug signature cache ...")
        df = self.lincs_sigs
        gene_arr = self.gene_cols

        for drug, grp in df.groupby(df["pert_iname"].str.lower()):
            vals = grp[gene_arr].values
            self._drug_mean_sigs[drug] = pd.Series(
                np.nanmean(vals, axis=0), index=gene_arr
            )
            for cl, cl_grp in grp.groupby("cell_id"):
                cl_vals = cl_grp[gene_arr].values
                self._drug_cell_sigs[(drug, cl)] = pd.Series(
                    np.nanmean(cl_vals, axis=0), index=gene_arr
                )

        logger.info(
            f"Cached mean signatures for {len(self._drug_mean_sigs)} drugs, "
            f"{len(self._drug_cell_sigs)} drug-cell combinations"
        )

    def _load_catalog(self):
        """Load the CTR-DB catalog for drug-to-dataset mapping."""
        cat_path = DATA_RAW / "ctrdb" / "catalog.csv"
        if cat_path.exists():
            self.catalog = pd.read_csv(cat_path)
        else:
            self.catalog = pd.DataFrame()

    def _get_drug_for_dataset(self, geo_id: str) -> str:
        """Return the drug string for a CTR-DB dataset."""
        if self.catalog.empty:
            return ""
        row = self.catalog[self.catalog["geo_source"] == geo_id]
        if row.empty:
            return ""
        return str(row.iloc[0]["drug"])

    def _precompute_dataset_features(
        self,
        common_genes: list[str],
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Pre-compute per-dataset reversal feature matrices for all patients.

        For each dataset: compute the drug signature aligned to common_genes,
        then compute reversal features for all patients at once (vectorised).

        Returns dict: geo_id -> (X_reversal, y_labels, drug_sig)
            X_reversal : (n_samples, n_genes) reversal feature matrix
            y_labels   : (n_samples,) binary response
            drug_sig   : (n_genes,) mean drug signature
        """
        result = {}
        n_genes = len(common_genes)

        for geo_id, (expr, labels) in self.ctrdb.items():
            drug_str = self._get_drug_for_dataset(geo_id)
            components = parse_regimen_components(drug_str)
            matched = match_drugs_to_lincs(components, self.lincs_drug_set)
            if not matched:
                continue

            # Genes available in this dataset
            available = [g for g in common_genes if g in expr.columns]
            if len(available) < 10:
                continue

            avail_idx = np.array([common_genes.index(g) for g in available])

            # Mean drug signature aligned to common_genes
            drug_sig = np.zeros(n_genes, dtype=np.float64)
            n_matched = 0
            for d in matched:
                d_lower = d.lower()
                if d_lower in self._drug_mean_sigs:
                    sig = self._drug_mean_sigs[d_lower]
                    vals = sig.reindex(common_genes).values.astype(np.float64)
                    drug_sig += np.nan_to_num(vals, 0.0)
                    n_matched += 1
            if n_matched > 0:
                drug_sig /= n_matched

            # Aligned patients
            common_samples = labels.index.intersection(expr.index)
            if len(common_samples) < 5:
                continue

            # Vectorised: extract expression for all patients at once
            expr_mat = expr.loc[common_samples, available].values.astype(
                np.float64
            )
            expr_mat = np.nan_to_num(expr_mat, 0.0)

            # Pad to full common_genes dimensionality
            full_expr = np.zeros(
                (len(common_samples), n_genes), dtype=np.float64
            )
            full_expr[:, avail_idx] = expr_mat

            # Reversal features: element-wise product
            X_reversal = full_expr * drug_sig[np.newaxis, :]

            y_arr = labels.loc[common_samples].values.astype(int)

            result[geo_id] = (X_reversal, y_arr, drug_sig, common_samples)

        return result

    # -- Context inference --------------------------------------------------

    def infer_molecular_context(
        self,
        expression_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Infer ER/HER2/Basal/Proliferation status from expression.

        Parameters
        ----------
        expression_df : DataFrame
            Samples x genes.

        Returns
        -------
        DataFrame with columns:
            er_status (bool), her2_status (bool), basal_like (bool),
            high_proliferation (bool), context (str)
        """
        result = pd.DataFrame(index=expression_df.index)

        # ER status from ESR1
        if "ESR1" in expression_df.columns:
            esr1 = expression_df["ESR1"]
            q75 = esr1.quantile(0.75)
            q25 = esr1.quantile(0.25)
            result["er_status"] = esr1 >= q75
            result["esr1_low"] = esr1 <= q25
        else:
            result["er_status"] = False
            result["esr1_low"] = True

        # HER2 status from ERBB2
        if "ERBB2" in expression_df.columns:
            erbb2 = expression_df["ERBB2"]
            q75 = erbb2.quantile(0.75)
            q25 = erbb2.quantile(0.25)
            result["her2_status"] = erbb2 >= q75
            result["erbb2_low"] = erbb2 <= q25
        else:
            result["her2_status"] = False
            result["erbb2_low"] = True

        # Basal-like: low ESR1 AND low ERBB2
        result["basal_like"] = result["esr1_low"] & result["erbb2_low"]

        # Proliferation from MKI67
        if "MKI67" in expression_df.columns:
            mki67 = expression_df["MKI67"]
            q75 = mki67.quantile(0.75)
            result["high_proliferation"] = mki67 >= q75
        else:
            result["high_proliferation"] = False

        # Assign primary context label (most specific applicable)
        def _assign_context(row):
            if row.get("her2_status", False):
                return "Breast_HER2_enriched"
            if row.get("basal_like", False):
                return "Breast_Basal_like"
            if row.get("er_status", False):
                if row.get("high_proliferation", False):
                    return "Breast_ER_positive_HighProlif"
                return "Breast_ER_positive"
            return "Breast"

        result["context"] = result.apply(_assign_context, axis=1)

        return result

    # -- Context grouping ---------------------------------------------------

    def build_context_groups(self) -> dict[str, list[tuple[str, str]]]:
        """
        Group CTR-DB patients by inferred molecular context.

        Returns
        -------
        dict mapping context_name -> list of (geo_id, sample_id) tuples
        """
        groups: dict[str, list[tuple[str, str]]] = {}

        for geo_id, (expr, labels) in self.ctrdb.items():
            ctx = self.infer_molecular_context(expr)
            for sample_id in labels.index:
                if sample_id not in ctx.index:
                    continue
                context = ctx.loc[sample_id, "context"]
                groups.setdefault(context, []).append((geo_id, sample_id))
                # Also add to broader "Breast" group
                groups.setdefault("Breast", []).append((geo_id, sample_id))
                # Add to proliferation group if applicable
                if ctx.loc[sample_id, "high_proliferation"]:
                    groups.setdefault("Breast_HighProliferation", []).append(
                        (geo_id, sample_id)
                    )

        # Report sizes
        for ctx_name, members in sorted(groups.items()):
            logger.info(f"Context group '{ctx_name}': {len(members)} patients")

        return groups

    # -- Gene weight learning -----------------------------------------------

    def _get_common_genes(
        self,
        expression_df: pd.DataFrame,
    ) -> list[str]:
        """Return gene symbols present in both CTR-DB expression and LINCS."""
        return [g for g in self.gene_cols if g in expression_df.columns]

    def learn_gene_weights(
        self,
        context_name: str,
        patient_indices: list[tuple[str, str]],
        alpha: float = 1.0,
        precomputed: Optional[dict] = None,
    ) -> dict:
        """
        Train L1-regularized logistic regression for gene importance.

        For each patient, the feature vector is:
            reversal_feature[gene] = patient_zscore[gene] * drug_signature[gene]

        Negative product means the drug reverses the patient's dysregulation
        for that gene (which is the therapeutic hypothesis).

        Parameters
        ----------
        context_name : str
            Name of the context group.
        patient_indices : list of (geo_id, sample_id) tuples.
        alpha : float
            Inverse regularisation strength (C parameter).
        precomputed : dict, optional
            Pre-computed dataset features from _precompute_dataset_features.

        Returns
        -------
        dict with keys: gene_weights, auc, n_patients, n_nonzero_genes
        """
        if precomputed is None:
            logger.warning(
                f"No precomputed features for '{context_name}' -- "
                f"call _precompute_dataset_features first"
            )
            return {}

        # Build a set of (geo_id, sample_id) for fast lookup
        patient_set: dict[str, set[str]] = {}
        for geo_id, sample_id in patient_indices:
            patient_set.setdefault(geo_id, set()).add(sample_id)

        # Collect features from precomputed data
        X_parts = []
        y_parts = []

        for geo_id, sample_ids in patient_set.items():
            if geo_id not in precomputed:
                continue
            X_rev, y_arr, drug_sig, ds_samples = precomputed[geo_id]
            # Build mask for samples in this context
            mask = np.array([s in sample_ids for s in ds_samples])
            if mask.sum() == 0:
                continue
            X_parts.append(X_rev[mask])
            y_parts.append(y_arr[mask])

        if not X_parts:
            logger.warning(
                f"Context '{context_name}': no valid samples -- skipping"
            )
            return {}

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)

        # Derive common_genes from the precomputed feature dimensionality
        # (same as self._common_genes set during precomputation)
        common_genes = self._precomputed_common_genes
        n_genes_actual = X.shape[1]
        logger.info(
            f"Context '{context_name}': {len(y)} patients, "
            f"{n_genes_actual} features"
        )

        if len(y) < MIN_GROUP_SIZE:
            logger.warning(
                f"Context '{context_name}': only {len(y)} valid "
                f"samples (need {MIN_GROUP_SIZE}) -- skipping gene weights"
            )
            return {}

        # Z-score features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.nan_to_num(X, 0.0)

        # Check label balance
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 3 or n_neg < 3:
            logger.warning(
                f"Context '{context_name}': too few positive ({n_pos}) or "
                f"negative ({n_neg}) samples -- skipping"
            )
            return {}

        # Cross-validated AUC
        cv_aucs = []
        n_splits = min(5, min(n_pos, n_neg))
        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=42)
            for train_idx, val_idx in skf.split(X, y):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LogisticRegression(
                        penalty="l1", C=alpha, solver="saga",
                        max_iter=2000, random_state=42,
                    )
                    clf.fit(X[train_idx], y[train_idx])
                proba = clf.predict_proba(X[val_idx])
                if proba.shape[1] == 2:
                    try:
                        auc = roc_auc_score(y[val_idx], proba[:, 1])
                        cv_aucs.append(auc)
                    except ValueError:
                        pass

        mean_auc = float(np.mean(cv_aucs)) if cv_aucs else 0.5

        # Fit on all data for final weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_final = LogisticRegression(
                C=alpha, solver="saga", l1_ratio=1.0,
                max_iter=2000, random_state=42,
            )
            clf_final.fit(X, y)

        coefs = clf_final.coef_[0]
        gene_weights = dict(zip(common_genes, coefs.tolist()))

        # Non-zero genes
        n_nonzero = int((np.abs(coefs) > 1e-8).sum())

        logger.info(
            f"Context '{context_name}': n={len(y)}, "
            f"AUC={mean_auc:.3f}, "
            f"non-zero genes={n_nonzero}/{len(common_genes)}"
        )

        return {
            "gene_weights": gene_weights,
            "auc": round(mean_auc, 4),
            "n_patients": len(y),
            "n_responders": n_pos,
            "n_nonresponders": n_neg,
            "n_nonzero_genes": n_nonzero,
            "n_total_genes": len(common_genes),
        }

    # -- Cell-line weight learning ------------------------------------------

    def learn_cell_line_weights(
        self,
        context_name: str,
        patient_indices: list[tuple[str, str]],
    ) -> dict[str, float]:
        """
        Learn which LINCS cell lines best predict response for this context.

        For each cell line, compute a mean reversal score per patient using
        only that cell line's signatures.  Then use logistic regression on
        the cell-line-specific reversal scores to learn relative importance.
        """
        cell_lines = sorted(self.lincs_sigs["cell_id"].unique())

        # For each patient, compute one reversal score per cell line
        X_rows = []
        y_vals = []

        for geo_id, sample_id in patient_indices:
            if geo_id not in self.ctrdb:
                continue
            expr, labels = self.ctrdb[geo_id]
            if sample_id not in labels.index or sample_id not in expr.index:
                continue

            drug_str = self._get_drug_for_dataset(geo_id)
            components = parse_regimen_components(drug_str)
            matched = match_drugs_to_lincs(components, self.lincs_drug_set)
            if not matched:
                continue

            ds_common = self._get_common_genes(expr)
            if len(ds_common) < 10:
                continue

            available = [g for g in ds_common if g in expr.columns]
            patient_expr = expr.loc[sample_id, available].values.astype(
                np.float64
            )
            patient_expr = np.nan_to_num(patient_expr, 0.0)

            # One reversal score per cell line
            cl_scores = []
            for cl in cell_lines:
                cl_sig = np.zeros(len(available), dtype=np.float64)
                n_cl = 0
                for d in matched:
                    key = (d.lower(), cl)
                    if key in self._drug_cell_sigs:
                        sig = self._drug_cell_sigs[key]
                        vals = sig.reindex(available).values.astype(
                            np.float64
                        )
                        cl_sig += np.nan_to_num(vals, 0.0)
                        n_cl += 1
                if n_cl > 0:
                    cl_sig /= n_cl
                    score = -np.mean(patient_expr * cl_sig)  # negative = reversal
                else:
                    score = 0.0
                cl_scores.append(score)

            X_rows.append(cl_scores)
            y_vals.append(int(labels.loc[sample_id]))

        if len(X_rows) < MIN_GROUP_SIZE:
            # Fallback: use biological priors
            return self._prior_cell_line_weights(context_name)

        X = np.array(X_rows)
        y = np.array(y_vals)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 3 or n_neg < 3:
            return self._prior_cell_line_weights(context_name)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.nan_to_num(X, 0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=2000, random_state=42,
            )
            clf.fit(X, y)

        coefs = clf.coef_[0]
        # Softmax to get weights
        exp_coefs = np.exp(coefs - coefs.max())
        weights = exp_coefs / exp_coefs.sum()

        result = {cl: round(float(w), 4) for cl, w in zip(cell_lines, weights)}
        logger.info(f"Cell-line weights for '{context_name}': {result}")
        return result

    def _prior_cell_line_weights(
        self,
        context_name: str,
    ) -> dict[str, float]:
        """Biological prior for cell-line weights when data is insufficient."""
        cell_lines = sorted(self.lincs_sigs["cell_id"].unique())
        n = len(cell_lines)
        uniform = {cl: round(1.0 / n, 4) for cl in cell_lines}

        # Apply biological priors
        if "ER_positive" in context_name:
            for cl in uniform:
                if cl == "MCF7":
                    uniform[cl] = 0.50
                elif cl == "SKBR3":
                    uniform[cl] = 0.10
        elif "HER2" in context_name:
            for cl in uniform:
                if cl == "SKBR3":
                    uniform[cl] = 0.50
                elif cl == "MCF7":
                    uniform[cl] = 0.15
        elif "Basal" in context_name:
            for cl in uniform:
                if cl in ("MDAMB231", "HS578T"):
                    uniform[cl] = 0.35

        # Renormalise
        total = sum(uniform.values())
        return {cl: round(w / total, 4) for cl, w in uniform.items()}

    # -- Combination weight learning ----------------------------------------

    def learn_combination_weights(
        self,
        drug_string: str,
        patient_geo_ids: list[str],
    ) -> dict[str, float]:
        """
        Learn optimal weights for combining individual drug signatures.

        Parameters
        ----------
        drug_string : str
            The regimen string (e.g. "TFAC (Cyclophosphamide+...)").
        patient_geo_ids : list
            GEO IDs of datasets using this drug.

        Returns
        -------
        dict mapping drug_name -> weight
        """
        components = parse_regimen_components(drug_string)
        matched = match_drugs_to_lincs(components, self.lincs_drug_set)

        if len(matched) <= 1:
            # Single drug or no match -- equal weight
            if matched:
                return {matched[0]: 1.0}
            return {}

        # For each patient, compute one reversal score per drug component
        X_rows = []
        y_vals = []

        for geo_id in patient_geo_ids:
            if geo_id not in self.ctrdb:
                continue
            expr, labels = self.ctrdb[geo_id]
            ds_common = self._get_common_genes(expr)
            if len(ds_common) < 10:
                continue

            available = [g for g in ds_common if g in expr.columns]
            if len(available) < 10:
                continue

            for sample_id in labels.index:
                if sample_id not in expr.index:
                    continue
                patient_expr = expr.loc[sample_id, available].values.astype(
                    np.float64
                )
                patient_expr = np.nan_to_num(patient_expr, 0.0)

                drug_scores = []
                for d in matched:
                    d_lower = d.lower()
                    if d_lower in self._drug_mean_sigs:
                        sig = self._drug_mean_sigs[d_lower]
                        vals = sig.reindex(available).values.astype(
                            np.float64
                        )
                        vals = np.nan_to_num(vals, 0.0)
                        score = -np.mean(patient_expr * vals)
                    else:
                        score = 0.0
                    drug_scores.append(score)

                X_rows.append(drug_scores)
                y_vals.append(int(labels.loc[sample_id]))

        if len(X_rows) < MIN_GROUP_SIZE:
            return {d: round(1.0 / len(matched), 4) for d in matched}

        X = np.array(X_rows)
        y = np.array(y_vals)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 3 or n_neg < 3:
            return {d: round(1.0 / len(matched), 4) for d in matched}

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.nan_to_num(X, 0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=2000, random_state=42,
            )
            clf.fit(X, y)

        coefs = clf.coef_[0]
        # Softmax
        exp_c = np.exp(coefs - coefs.max())
        weights = exp_c / exp_c.sum()

        result = {d: round(float(w), 4) for d, w in zip(matched, weights)}
        logger.info(
            f"Combination weights for '{drug_string}': {result}"
        )
        return result

    # -- Build full recalibrated bank ---------------------------------------

    def build_recalibrated_bank(self) -> dict:
        """
        Run full recalibration pipeline and return the signature bank.

        Steps
        -----
        1. Infer patient contexts and build context groups.
        2. For each context, learn gene weights and cell-line weights.
        3. For each regimen, learn combination weights.
        4. Save bank to JSON and drug profiles to parquet.
        """
        logger.info("=" * 60)
        logger.info("Building recalibrated signature bank")
        logger.info("=" * 60)

        # Step 1: group patients by context
        context_groups = self.build_context_groups()

        # Step 1b: determine common gene set and precompute features once
        logger.info("Pre-computing dataset reversal features ...")
        involved_geo_ids = set()
        gene_counts: dict[str, int] = {}
        for geo_id, (expr, _) in self.ctrdb.items():
            ds_genes = self._get_common_genes(expr)
            if len(ds_genes) < 50:
                continue
            involved_geo_ids.add(geo_id)
            for g in ds_genes:
                gene_counts[g] = gene_counts.get(g, 0) + 1

        threshold = max(1, len(involved_geo_ids) // 2)
        common_genes = [g for g in self.gene_cols
                        if gene_counts.get(g, 0) >= threshold]
        self._precomputed_common_genes = common_genes
        logger.info(
            f"Common gene set: {len(common_genes)} genes "
            f"(present in >= {threshold}/{len(involved_geo_ids)} datasets)"
        )

        precomputed = self._precompute_dataset_features(common_genes)
        logger.info(
            f"Pre-computed features for {len(precomputed)} datasets"
        )

        bank = {}

        # Step 2: learn weights per context
        for ctx_name, members in sorted(context_groups.items()):
            min_size = (MIN_GROUP_SIZE_BROAD
                        if ctx_name == "Breast"
                        else MIN_GROUP_SIZE)
            if len(members) < min_size:
                logger.info(
                    f"Skipping context '{ctx_name}': "
                    f"{len(members)} < {min_size} patients"
                )
                continue

            logger.info(f"\n--- Learning weights for '{ctx_name}' "
                        f"({len(members)} patients) ---")

            # Gene weights
            result = self.learn_gene_weights(
                ctx_name, members, precomputed=precomputed
            )
            if not result:
                continue

            # Cell-line weights
            cl_weights = self.learn_cell_line_weights(ctx_name, members)
            result["cell_line_weights"] = cl_weights

            bank[ctx_name] = result

        # Step 3: learn combination weights per regimen
        if not self.catalog.empty:
            regimen_datasets: dict[str, list[str]] = {}
            for _, row in self.catalog.iterrows():
                drug_str = str(row["drug"])
                geo_id = str(row["geo_source"])
                if geo_id in self.ctrdb:
                    regimen_datasets.setdefault(drug_str, []).append(geo_id)

            combo_weights = {}
            for drug_str, geo_ids in regimen_datasets.items():
                weights = self.learn_combination_weights(drug_str, geo_ids)
                if weights:
                    combo_weights[drug_str] = weights

            bank["_combination_weights"] = combo_weights
            logger.info(
                f"Learned combination weights for "
                f"{len(combo_weights)} regimens"
            )

        self.bank = bank

        # Save outputs
        self._save_bank(bank)
        self._save_drug_profiles(bank)

        return bank

    # -- Saving outputs -----------------------------------------------------

    def _save_bank(self, bank: dict):
        """Save recalibrated signature bank to JSON."""
        out_path = DATA_PROCESSED / "recalibrated_signatures.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Make JSON-serialisable
        serialisable = {}
        for key, val in bank.items():
            if isinstance(val, dict):
                serialisable[key] = _make_serialisable(val)
            else:
                serialisable[key] = val

        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"Saved recalibrated bank to {out_path}")

    def _save_drug_profiles(self, bank: dict):
        """
        Build and save recalibrated drug profiles.

        For each drug in LINCS that also appears in CTR-DB, produce a
        context-weighted signature.
        """
        out_path = DATA_PROCESSED / "recalibrated_drug_profiles.parquet"
        rows = []

        combo_weights = bank.get("_combination_weights", {})

        for ctx_name, ctx_data in bank.items():
            if ctx_name.startswith("_"):
                continue
            gene_weights = ctx_data.get("gene_weights", {})
            cl_weights = ctx_data.get("cell_line_weights", {})
            if not gene_weights:
                continue

            # For each drug with a signature
            for drug_name, mean_sig in self._drug_mean_sigs.items():
                # Compute context-weighted signature
                weighted_sig = {}
                for gene in self.gene_cols:
                    gw = gene_weights.get(gene, 0.0)
                    sv = mean_sig.get(gene, 0.0)
                    if not np.isfinite(gw) or not np.isfinite(sv):
                        weighted_sig[gene] = 0.0
                    else:
                        weighted_sig[gene] = float(gw * sv)

                row = {
                    "drug": drug_name,
                    "context": ctx_name,
                    "n_nonzero_weights": sum(
                        1 for v in weighted_sig.values() if abs(v) > 1e-8
                    ),
                }
                row.update(weighted_sig)
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(out_path, index=False)
            logger.info(
                f"Saved {len(rows)} recalibrated drug profiles to {out_path}"
            )
        else:
            logger.warning("No drug profiles to save")

    # -- Scoring interface --------------------------------------------------

    def get_recalibrated_signature(
        self,
        drug_name: str,
        context: str,
    ) -> pd.Series:
        """
        Get the recalibrated signature for a drug in a given context.

        Falls back through the context hierarchy:
            specific context -> Breast -> uniform (all genes weight 1)
        """
        drug_lower = drug_name.lower()
        if drug_lower not in self._drug_mean_sigs:
            logger.warning(f"Drug '{drug_name}' not found in LINCS signatures")
            return pd.Series(dtype=float)

        base_sig = self._drug_mean_sigs[drug_lower]

        # Find gene weights, with hierarchical fallback
        gene_weights = None
        for ctx in [context, "Breast"]:
            if ctx in self.bank and "gene_weights" in self.bank[ctx]:
                gene_weights = self.bank[ctx]["gene_weights"]
                break

        if gene_weights is None:
            return base_sig.copy()

        # Apply gene weights
        weighted = base_sig.copy()
        for gene in weighted.index:
            gw = gene_weights.get(gene, 0.0)
            weighted[gene] = weighted[gene] * gw

        return weighted

    def compute_recalibrated_reversal(
        self,
        patient_sig: pd.Series,
        drug_name: str,
        context: str,
    ) -> float:
        """
        Compute context-aware reversal score using learned weights.

        The score is the weighted sum of reversal features:
            score = sum(gene_weight_i * patient_expr_i * drug_sig_i)

        This is the linear predictor from the logistic regression model.
        Higher score -> more likely to respond.

        Parameters
        ----------
        patient_sig : Series
            Gene expression z-scores for a patient (indexed by gene symbol).
        drug_name : str
            Drug name (LINCS pert_iname).
        context : str
            Molecular context (e.g. "Breast_ER_positive").

        Returns
        -------
        float : weighted reversal score (higher = better predicted response)
        """
        drug_lower = drug_name.lower()
        if drug_lower not in self._drug_mean_sigs:
            return 0.0

        base_sig = self._drug_mean_sigs[drug_lower]

        # Find gene weights, with hierarchical fallback
        gene_weights = None
        for ctx in [context, "Breast"]:
            if ctx in self.bank and "gene_weights" in self.bank[ctx]:
                gene_weights = self.bank[ctx]["gene_weights"]
                break

        if gene_weights is None:
            # Fall back to uniform reversal
            return self.compute_uniform_reversal(patient_sig, drug_name)

        common = patient_sig.index.intersection(base_sig.index)
        if len(common) == 0:
            return 0.0

        p = patient_sig.loc[common].values.astype(np.float64)
        d = base_sig.loc[common].values.astype(np.float64)
        w = np.array([gene_weights.get(g, 0.0) for g in common],
                     dtype=np.float64)

        p = np.nan_to_num(p, 0.0)
        d = np.nan_to_num(d, 0.0)
        w = np.nan_to_num(w, 0.0)

        # Weighted reversal: sum of (weight * patient_expr * drug_sig)
        return float(np.sum(w * p * d))

    def compute_uniform_reversal(
        self,
        patient_sig: pd.Series,
        drug_name: str,
    ) -> float:
        """
        Compute standard (non-recalibrated) reversal score.

        All genes weighted equally.
        """
        drug_lower = drug_name.lower()
        if drug_lower not in self._drug_mean_sigs:
            return 0.0

        base_sig = self._drug_mean_sigs[drug_lower]
        common = patient_sig.index.intersection(base_sig.index)
        if len(common) == 0:
            return 0.0

        p = patient_sig.loc[common].values.astype(np.float64)
        d = base_sig.loc[common].values.astype(np.float64)
        p = np.nan_to_num(p, 0.0)
        d = np.nan_to_num(d, 0.0)

        return float(-np.mean(p * d))

    # -- Validation ---------------------------------------------------------

    def validate(self) -> pd.DataFrame:
        """
        Compare AUC using uniform vs context-weighted reversal scores
        across all CTR-DB datasets.

        Returns DataFrame with columns:
            geo_id, n_samples, n_responders, drug, context,
            auc_uniform, auc_recalibrated, delta_auc
        """
        logger.info("=" * 60)
        logger.info("Validating recalibration vs uniform scoring")
        logger.info("=" * 60)

        results = []

        for geo_id, (expr, labels) in self.ctrdb.items():
            drug_str = self._get_drug_for_dataset(geo_id)
            components = parse_regimen_components(drug_str)
            matched = match_drugs_to_lincs(components, self.lincs_drug_set)

            if not matched:
                logger.info(f"  {geo_id}: no LINCS-matched drugs -- skip")
                continue

            # Infer contexts for patients in this dataset
            ctx_df = self.infer_molecular_context(expr)

            uniform_scores = []
            recal_scores = []
            y_true = []

            for sample_id in labels.index:
                if sample_id not in expr.index:
                    continue
                patient_expr = expr.loc[sample_id]
                context = ctx_df.loc[sample_id, "context"]

                # Compute both types of score (average over matched drugs)
                u_scores = []
                r_scores = []
                for d in matched:
                    u = self.compute_uniform_reversal(patient_expr, d)
                    r = self.compute_recalibrated_reversal(
                        patient_expr, d, context
                    )
                    u_scores.append(u)
                    r_scores.append(r)

                uniform_scores.append(np.mean(u_scores))
                recal_scores.append(np.mean(r_scores))
                y_true.append(int(labels.loc[sample_id]))

            if len(y_true) < 10:
                continue

            y = np.array(y_true)
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos

            if n_pos < 2 or n_neg < 2:
                continue

            try:
                auc_u = roc_auc_score(y, uniform_scores)
            except ValueError:
                auc_u = 0.5

            try:
                auc_r = roc_auc_score(y, recal_scores)
            except ValueError:
                auc_r = 0.5

            # Majority context for this dataset
            majority_ctx = ctx_df.loc[labels.index.intersection(
                ctx_df.index
            ), "context"].mode()
            majority_ctx = majority_ctx.iloc[0] if len(majority_ctx) > 0 else "Breast"

            delta = auc_r - auc_u
            results.append({
                "geo_id": geo_id,
                "n_samples": len(y_true),
                "n_responders": n_pos,
                "n_nonresponders": n_neg,
                "drug": drug_str,
                "majority_context": majority_ctx,
                "auc_uniform": round(auc_u, 4),
                "auc_recalibrated": round(auc_r, 4),
                "delta_auc": round(delta, 4),
            })

            logger.info(
                f"  {geo_id} (n={len(y_true)}, drug={drug_str[:40]}): "
                f"AUC uniform={auc_u:.3f} -> recalibrated={auc_r:.3f} "
                f"(delta={delta:+.3f})"
            )

        val_df = pd.DataFrame(results)
        if not val_df.empty:
            out_path = RESULTS / "recalibration_validation.csv"
            val_df.to_csv(out_path, index=False)
            logger.info(f"Saved validation results to {out_path}")

            # Summary
            mean_delta = val_df["delta_auc"].mean()
            n_improved = (val_df["delta_auc"] > 0).sum()
            n_total = len(val_df)
            logger.info(
                f"Summary: mean delta AUC = {mean_delta:+.4f}, "
                f"improved {n_improved}/{n_total} datasets"
            )
        else:
            logger.warning("No datasets could be validated")

        return val_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serialisable(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_recalibration():
    """
    End-to-end recalibration pipeline.

    Loads data, builds recalibrated signature bank, validates, and saves.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading LINCS breast signatures ...")
    lincs_path = DATA_CACHE / "breast_l1000_signatures.parquet"
    if not lincs_path.exists():
        logger.error(f"LINCS signatures not found at {lincs_path}")
        return
    lincs_sigs = pd.read_parquet(lincs_path)
    logger.info(f"  {lincs_sigs.shape[0]} signatures, "
                f"{lincs_sigs['pert_iname'].nunique()} drugs")

    logger.info("Loading CTR-DB patient datasets ...")
    from src.data_ingestion.ctrdb import load_all_breast_ctrdb
    ctrdb = load_all_breast_ctrdb()
    logger.info(f"  {len(ctrdb)} datasets loaded")

    if len(ctrdb) == 0:
        logger.error("No CTR-DB datasets available. Run data download first.")
        return

    logger.info("Loading landmark gene list ...")
    from src.data_ingestion.lincs import load_landmark_genes
    gene_df = load_landmark_genes()
    landmark_genes = gene_df["gene_symbol"].tolist()
    logger.info(f"  {len(landmark_genes)} landmark genes")

    # Build recalibrator
    recalibrator = SignatureRecalibrator(
        lincs_signatures=lincs_sigs,
        ctrdb_datasets=ctrdb,
        landmark_genes=landmark_genes,
    )

    # Build bank
    bank = recalibrator.build_recalibrated_bank()

    # Validate
    val_df = recalibrator.validate()

    logger.info("=" * 60)
    logger.info("Recalibration complete.")
    logger.info(
        f"  Contexts learned: "
        f"{[k for k in bank if not k.startswith('_')]}"
    )
    if not val_df.empty:
        logger.info(
            f"  Mean delta AUC: {val_df['delta_auc'].mean():+.4f}"
        )
    logger.info("=" * 60)

    return recalibrator


if __name__ == "__main__":
    run_recalibration()
