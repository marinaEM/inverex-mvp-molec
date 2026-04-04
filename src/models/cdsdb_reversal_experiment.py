"""
CDS-DB patient reversal vs LINCS cell-line reversal experiment.

Head-to-head comparison: for drugs with BOTH LINCS and CDS-DB signatures,
which source better predicts CTR-DB patient clinical response?

CDS-DB drugs (patient-derived perturbation signatures):
  - Letrozole (GSE87455, GSE20181)
  - Anastrozole (GSE33658)
  - Trastuzumab + Lapatinib (GSE55374)

LINCS drugs: ~hundreds of compounds profiled in breast cancer cell lines.

CTR-DB datasets: breast cancer clinical trial cohorts with expression +
binary response labels (pCR vs RD, or CR/PR vs SD/PD).

Approach:
  1. Load CDS-DB perturbation signatures (post - pre log2FC per gene).
  2. Load LINCS cell-line signatures (average z-score per drug across cell lines).
  3. Identify drugs present in BOTH CDS-DB and LINCS.
  4. For each CTR-DB dataset whose treatment overlaps with a shared drug:
     - Compute LINCS reversal score: -corr(patient_disease_sig, lincs_drug_sig)
     - Compute CDS-DB reversal score: -corr(patient_disease_sig, cdsdb_drug_sig)
     - Compute AUC for responder vs non-responder discrimination.
  5. Also do a broader component-level match: if a CTR-DB regimen contains
     a drug that is in CDS-DB/LINCS (e.g. paclitaxel in TFAC), score
     using the single-agent signature for that component.

Output:
  - results/lincs_vs_cdsdb_reversal.tsv
"""
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.config import DATA_CACHE, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# =====================================================================
# Drug name normalisation
# =====================================================================

_CANONICAL = {
    "letrozole": "letrozole",
    "anastrozole": "anastrozole",
    "tamoxifen": "tamoxifen",
    "trastuzumab": "trastuzumab",
    "lapatinib": "lapatinib",
    "paclitaxel": "paclitaxel",
    "docetaxel": "docetaxel",
    "doxorubicin": "doxorubicin",
    "cyclophosphamide": "cyclophosphamide",
    "fluorouracil": "fluorouracil",
    "5-fluorouracil": "fluorouracil",
    "epirubicin": "epirubicin",
    "methotrexate": "methotrexate",
    "capecitabine": "capecitabine",
    "ixabepilone": "ixabepilone",
    "pegfilgrastim": "pegfilgrastim",
    "trastuzumab + lapatinib": "trastuzumab+lapatinib",
}


def _norm(name: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return name.strip().lower()


def _canon(name: str) -> str:
    n = _norm(name)
    return _CANONICAL.get(n, n)


def _extract_components(drug_field: str) -> list[str]:
    """
    Extract individual drug names from a CTR-DB drug field.

    Examples:
        'TFAC (Cyclophosphamide+Doxorubicin+Fluorouracil+Paclitaxel)'
        -> ['cyclophosphamide', 'doxorubicin', 'fluorouracil', 'paclitaxel']

        'Letrozole' -> ['letrozole']
        'Anthracycline+Taxane' -> ['doxorubicin', 'paclitaxel']
    """
    # Drug-class to specific-drug expansion
    _CLASS_MAP = {
        "anthracycline": ["doxorubicin", "epirubicin"],
        "taxane": ["paclitaxel", "docetaxel"],
    }

    text = drug_field
    components = set()

    # Extract names from inside parentheses
    for m in re.finditer(r"\(([^)]+)\)", text):
        inside = m.group(1)
        for part in re.split(r"[+/,]", inside):
            name = part.strip()
            if name and len(name) > 2:
                components.add(_canon(name))

    # Also look for names at the top level (outside parens)
    no_parens = re.sub(r"\([^)]*\)", "", text)
    for part in re.split(r"[+/,]", no_parens):
        name = part.strip()
        if name and len(name) > 2 and not re.match(r"^[A-Z]{1,5}$", name):
            components.add(_canon(name))

    # If nothing was extracted, try the whole field
    if not components:
        components.add(_canon(drug_field))

    # Expand drug classes
    expanded = set()
    for comp in components:
        if comp in _CLASS_MAP:
            expanded.update(_CLASS_MAP[comp])
        else:
            expanded.add(comp)

    return sorted(expanded)


# =====================================================================
# Step 1 & 2: Load CDS-DB perturbation signatures
# =====================================================================

def _load_cdsdb_signatures(
    data_dir: Path,
) -> dict[str, pd.Series]:
    """
    Load CDS-DB patient drug perturbation signatures, averaged per drug.

    Uses the enhanced loader from patient_signatures which can extract
    pre/post pairs from GEO metadata (GSE87455, GSE20181 for Letrozole;
    GSE33658 for Anastrozole).

    Returns dict: canonical_drug_name -> Series(gene_symbol -> mean_log_fc)
    """
    # Try enhanced loader first (extracts pairs via GEOparse)
    try:
        from src.features.patient_signatures import load_enhanced_cdsdb_signatures
        sigs = load_enhanced_cdsdb_signatures(data_dir)
    except Exception as exc:
        logger.warning(f"Enhanced CDS-DB loader failed: {exc}")
        sigs = pd.DataFrame()

    # Fallback to standard loader
    if sigs.empty:
        from src.data_ingestion.cdsdb import load_breast_perturbation_signatures
        sigs = load_breast_perturbation_signatures(data_dir)

    # Last resort: load perturbation parquets directly
    if sigs.empty:
        logger.warning("No CDS-DB perturbation signatures found in standard loader")
        sigs = _load_perturbation_parquets_directly(data_dir)

    if sigs.empty:
        logger.warning("CDS-DB: no signatures available at all")
        return {}

    logger.info(
        f"CDS-DB raw signatures: {len(sigs)} rows, "
        f"drugs: {sigs['drug'].unique().tolist()}, "
        f"patients: {sigs['patient_id'].nunique()}"
    )

    drug_sigs = {}
    for drug_name, grp in sigs.groupby("drug"):
        if "gene_symbol" not in grp.columns or "log_fc" not in grp.columns:
            continue
        # Average across patients per gene
        avg = grp.groupby("gene_symbol")["log_fc"].mean()
        if len(avg) < 50:
            logger.info(f"  CDS-DB {drug_name}: only {len(avg)} genes, skipping")
            continue
        canon = _canon(drug_name)
        drug_sigs[canon] = avg
        logger.info(
            f"  CDS-DB {drug_name} -> '{canon}': "
            f"{grp['patient_id'].nunique()} patients, {len(avg)} genes"
        )

    return drug_sigs


def _load_perturbation_parquets_directly(data_dir: Path) -> pd.DataFrame:
    """
    Directly load perturbation_signatures.parquet files without GEOparse.
    """
    catalog_path = data_dir / "catalog.csv"
    if not catalog_path.exists():
        return pd.DataFrame()

    catalog = pd.read_csv(catalog_path)
    parts = []
    for _, row in catalog.iterrows():
        geo_id = row["geo_id"]
        sig_path = data_dir / geo_id / "perturbation_signatures.parquet"
        if sig_path.exists():
            df = pd.read_parquet(sig_path)
            df["drug"] = row.get("drug", "unknown")
            df["geo_id"] = geo_id
            parts.append(df)
            logger.info(
                f"  Direct load {geo_id}: {len(df)} rows, "
                f"cols={list(df.columns)}"
            )

    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()


# =====================================================================
# Step 3a: Load LINCS signatures
# =====================================================================

def _load_lincs_signatures() -> dict[str, pd.Series]:
    """
    Load LINCS cell-line drug signatures, averaged per drug.

    Returns dict: canonical_drug_name -> Series(gene_symbol -> mean_zscore)
    """
    # Try the all-cell-line cache first (larger), then breast-only
    for fname in ["all_cellline_drug_signatures.parquet",
                  "breast_l1000_signatures.parquet"]:
        path = DATA_CACHE / fname
        if path.exists():
            logger.info(f"Loading LINCS from {path}")
            lincs = pd.read_parquet(path)
            break
    else:
        logger.error("No LINCS signature cache found")
        return {}

    meta_cols = {
        "sig_id", "pert_id", "pert_iname", "cell_id",
        "pert_idose", "dose_um",
    }
    gene_cols = [c for c in lincs.columns if c not in meta_cols]

    if "pert_iname" not in lincs.columns:
        logger.error("LINCS cache missing 'pert_iname' column")
        return {}

    drug_sigs = {}
    for drug_name, grp in lincs.groupby("pert_iname"):
        avg = grp[gene_cols].mean()
        # Drop NaN genes
        avg = avg.dropna()
        if len(avg) < 50:
            continue
        canon = _canon(drug_name)
        drug_sigs[canon] = avg

    logger.info(f"LINCS: {len(drug_sigs)} drugs loaded")
    return drug_sigs


# =====================================================================
# Step 3b: Load CTR-DB datasets
# =====================================================================

def _load_ctrdb_datasets() -> dict[str, tuple[pd.DataFrame, pd.Series, str]]:
    """
    Load CTR-DB clinical trial datasets with response labels.

    Returns dict: geo_id -> (expression_df, response_series, drug_field)
    """
    ctrdb_dir = DATA_RAW / "ctrdb"
    if not ctrdb_dir.exists():
        logger.error(f"CTR-DB directory not found: {ctrdb_dir}")
        return {}

    # Load catalog for drug mapping
    catalog_path = ctrdb_dir / "catalog.csv"
    geo_to_drug = {}
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        for _, row in catalog.iterrows():
            geo_to_drug[row["geo_source"]] = row["drug"]

    # Manual drug annotations for datasets NOT in the catalog but whose
    # treatment is known from CDS-DB or GEO metadata.  This is critical
    # for the CDS-DB comparison because several Letrozole datasets
    # were downloaded separately and are not in the CTR-DB catalog.
    _MANUAL_DRUG_MAP = {
        "GSE20181": "Letrozole",          # pre/post Letrozole in ER+ BC
        "GSE19293": "Letrozole",          # neoadjuvant Letrozole
        "GSE37946": "Letrozole",          # Letrozole treatment
        "GSE104645": "Anthracycline+Taxane",  # neoadjuvant chemo
    }

    datasets = {}
    for ds_dir in sorted(ctrdb_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        geo_id = ds_dir.name

        expr_files = list(ds_dir.glob("*_expression.parquet"))
        label_file = ds_dir / "response_labels.parquet"
        if not expr_files or not label_file.exists():
            continue

        try:
            expr = pd.read_parquet(expr_files[0])
            labels = pd.read_parquet(label_file)["response"]
        except Exception as e:
            logger.debug(f"Failed to load {geo_id}: {e}")
            continue

        common = expr.index.intersection(labels.index)
        if len(common) < 10:
            continue

        drug_field = geo_to_drug.get(
            geo_id, _MANUAL_DRUG_MAP.get(geo_id, "unknown")
        )
        datasets[geo_id] = (expr.loc[common], labels.loc[common], drug_field)

    logger.info(f"CTR-DB: loaded {len(datasets)} datasets with labels")
    return datasets


# =====================================================================
# Step 4: Find overlapping drugs
# =====================================================================

def _find_drug_overlaps(
    cdsdb_drugs: set[str],
    lincs_drugs: set[str],
    ctrdb_drug_fields: dict[str, str],   # geo_id -> drug_field
) -> dict[str, list[dict]]:
    """
    For each CTR-DB dataset, find which CDS-DB and LINCS drugs match
    (either exact or as components of the CTR-DB regimen).

    Returns dict: geo_id -> list of
        { 'component': str, 'in_cdsdb': bool, 'in_lincs': bool }
    """
    overlaps = {}
    for geo_id, drug_field in ctrdb_drug_fields.items():
        components = _extract_components(drug_field)
        matches = []
        for comp in components:
            in_cdsdb = comp in cdsdb_drugs
            in_lincs = comp in lincs_drugs
            if in_cdsdb or in_lincs:
                matches.append({
                    "component": comp,
                    "in_cdsdb": in_cdsdb,
                    "in_lincs": in_lincs,
                })
        if matches:
            overlaps[geo_id] = matches
    return overlaps


# =====================================================================
# Step 5: Reversal score + AUC
# =====================================================================

def _reversal_score(patient_expr: pd.Series, drug_sig: pd.Series) -> float:
    """Reversal = -corr(patient_disease_sig, drug_sig)."""
    common = patient_expr.index.intersection(drug_sig.index)
    if len(common) < 10:
        return np.nan
    p = patient_expr[common].values.astype(float)
    d = drug_sig[common].values.astype(float)
    valid = np.isfinite(p) & np.isfinite(d)
    if valid.sum() < 10:
        return np.nan
    corr = np.corrcoef(p[valid], d[valid])[0, 1]
    if np.isnan(corr):
        return np.nan
    return -corr


def _evaluate_source(
    expr_z: pd.DataFrame,
    labels: pd.Series,
    drug_sig: pd.Series,
    source: str,
    drug: str,
    geo_id: str,
) -> Optional[dict]:
    """
    Evaluate one signature source on one CTR-DB dataset.

    Returns a result dict or None if insufficient data.
    """
    common_samples = expr_z.index.intersection(labels.index)
    if len(common_samples) < 10:
        return None

    expr_sub = expr_z.loc[common_samples]
    lab_sub = labels.loc[common_samples]

    n_resp = int(lab_sub.sum())
    n_nonresp = len(lab_sub) - n_resp
    if n_resp < 2 or n_nonresp < 2:
        return None

    # Reversal scores per patient
    scores = []
    for sid in common_samples:
        rev = _reversal_score(expr_sub.loc[sid], drug_sig)
        scores.append(rev)

    scores_s = pd.Series(scores, index=common_samples)
    valid = scores_s.dropna()

    if len(valid) < 10:
        return None

    lab_valid = lab_sub.loc[valid.index]
    n_resp_v = int(lab_valid.sum())
    n_nonresp_v = len(lab_valid) - n_resp_v
    if n_resp_v < 2 or n_nonresp_v < 2:
        return None

    # AUC
    try:
        auc = roc_auc_score(lab_valid, valid)
    except ValueError:
        auc = np.nan

    # Mann-Whitney p-value
    resp_scores = valid[lab_valid == 1]
    nonresp_scores = valid[lab_valid == 0]
    try:
        _, mw_p = stats.mannwhitneyu(
            resp_scores, nonresp_scores, alternative="two-sided"
        )
    except ValueError:
        mw_p = np.nan

    n_genes_overlap = len(
        expr_sub.columns.intersection(drug_sig.dropna().index)
    )

    return {
        "geo_id": geo_id,
        "drug_component": drug,
        "source": source,
        "n_patients": len(valid),
        "n_responders": n_resp_v,
        "n_nonresponders": n_nonresp_v,
        "n_genes_overlap": n_genes_overlap,
        "auc": round(auc, 4) if not np.isnan(auc) else np.nan,
        "mannwhitney_p": round(mw_p, 6) if not np.isnan(mw_p) else np.nan,
        "mean_rev_responders": round(float(resp_scores.mean()), 4),
        "mean_rev_nonresponders": round(float(nonresp_scores.mean()), 4),
        "delta_rev": round(
            float(resp_scores.mean()) - float(nonresp_scores.mean()), 4
        ),
    }


# =====================================================================
# Main experiment
# =====================================================================

def run_experiment(
    output_path: Path = RESULTS / "lincs_vs_cdsdb_reversal.tsv",
) -> pd.DataFrame:
    """
    Run the head-to-head CDS-DB vs LINCS reversal experiment.
    """
    print("=" * 72)
    print("  INVEREX: CDS-DB patient reversal vs LINCS cell-line reversal")
    print("=" * 72)
    print()

    # ---- Load signatures ----
    cdsdb_sigs = _load_cdsdb_signatures(DATA_RAW / "cdsdb")
    lincs_sigs = _load_lincs_signatures()

    print(f"\n--- Signature banks ---")
    print(f"  CDS-DB drugs : {len(cdsdb_sigs)}  {list(cdsdb_sigs.keys())}")
    print(f"  LINCS drugs  : {len(lincs_sigs)}")
    if lincs_sigs:
        sample = list(lincs_sigs.keys())[:20]
        print(f"    (sample)   : {sample}")

    # ---- Load CTR-DB ----
    ctrdb = _load_ctrdb_datasets()
    print(f"\n--- CTR-DB datasets ---")
    print(f"  Loaded: {len(ctrdb)} datasets with response labels")
    for gid, (expr, lab, drug_f) in sorted(ctrdb.items()):
        print(
            f"    {gid}: {expr.shape[0]} patients, "
            f"{int(lab.sum())}R/{int((1-lab).sum())}NR, drug={drug_f[:60]}"
        )

    # ---- Find overlaps ----
    ctrdb_drug_fields = {gid: v[2] for gid, v in ctrdb.items()}
    overlaps = _find_drug_overlaps(
        set(cdsdb_sigs.keys()), set(lincs_sigs.keys()), ctrdb_drug_fields
    )

    # Count true 3-way overlaps (drug in CDS-DB AND LINCS AND CTR-DB)
    three_way = set()
    for geo_id, matches in overlaps.items():
        for m in matches:
            if m["in_cdsdb"] and m["in_lincs"]:
                three_way.add(m["component"])

    print(f"\n{'=' * 72}")
    print(f"  DRUG OVERLAP COUNT: {len(three_way)} drugs in CDS-DB + LINCS + CTR-DB")
    print(f"{'=' * 72}")
    if three_way:
        print(f"  Three-way overlap drugs: {sorted(three_way)}")
    else:
        print("  WARNING: No three-way drug overlap found.")
        print("  Will proceed with whatever partial overlaps exist.")

    # Show all overlaps
    cdsdb_in_ctrdb = set()
    lincs_in_ctrdb = set()
    for geo_id, matches in overlaps.items():
        for m in matches:
            if m["in_cdsdb"]:
                cdsdb_in_ctrdb.add(m["component"])
            if m["in_lincs"]:
                lincs_in_ctrdb.add(m["component"])

    print(f"\n  CDS-DB drugs matching CTR-DB regimens: {sorted(cdsdb_in_ctrdb)}")
    print(f"  LINCS drugs matching CTR-DB regimens : {sorted(lincs_in_ctrdb)}")

    # ---- Compute reversal scores ----
    print(f"\n--- Computing reversal scores ---")
    results = []

    for geo_id, (expr, labels, drug_field) in ctrdb.items():
        if geo_id not in overlaps:
            continue

        matches = overlaps[geo_id]

        # Z-score expression across the cohort (disease signature)
        gene_cols = [
            c for c in expr.columns
            if isinstance(c, str) and c not in {"sample_id", "patient_id"}
        ]
        if len(gene_cols) < 50:
            continue

        expr_sub = expr[gene_cols].copy()
        mu = expr_sub.mean(axis=0)
        sd = expr_sub.std(axis=0).replace(0, 1)
        expr_z = (expr_sub - mu) / sd

        for m in matches:
            comp = m["component"]
            print(f"  {geo_id} | drug_component={comp} | "
                  f"cdsdb={m['in_cdsdb']} lincs={m['in_lincs']}")

            if m["in_lincs"]:
                res = _evaluate_source(
                    expr_z, labels, lincs_sigs[comp],
                    "LINCS", comp, geo_id,
                )
                if res:
                    res["ctrdb_regimen"] = drug_field
                    results.append(res)
                    print(
                        f"    LINCS  AUC={res['auc']}  "
                        f"p={res['mannwhitney_p']}  "
                        f"n_genes={res['n_genes_overlap']}"
                    )
                else:
                    print(f"    LINCS  -- insufficient data --")

            if m["in_cdsdb"]:
                res = _evaluate_source(
                    expr_z, labels, cdsdb_sigs[comp],
                    "CDS-DB", comp, geo_id,
                )
                if res:
                    res["ctrdb_regimen"] = drug_field
                    results.append(res)
                    print(
                        f"    CDS-DB AUC={res['auc']}  "
                        f"p={res['mannwhitney_p']}  "
                        f"n_genes={res['n_genes_overlap']}"
                    )
                else:
                    print(f"    CDS-DB -- insufficient data --")

    # ---- If we also want to compare CDS-DB drugs using their
    #      own internal response labels (GSE20181 has responder
    #      annotations), note that here for completeness ----

    # ---- Build results table ----
    if results:
        df = pd.DataFrame(results)
        # Reorder columns
        col_order = [
            "drug_component", "ctrdb_regimen", "geo_id", "source",
            "n_patients", "n_responders", "n_nonresponders",
            "n_genes_overlap", "auc", "mannwhitney_p",
            "mean_rev_responders", "mean_rev_nonresponders", "delta_rev",
        ]
        df = df[[c for c in col_order if c in df.columns]]
        df = df.sort_values(["drug_component", "geo_id", "source"])
    else:
        df = pd.DataFrame()

    # ---- Save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"\nResults saved to {output_path}")

    # ---- Summary ----
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total result rows          : {len(df)}")
    print(f"  Three-way overlap drugs    : {len(three_way)}")
    print(f"  CDS-DB drugs tested        : {sorted(cdsdb_in_ctrdb)}")
    print(f"  LINCS drugs tested         : {sorted(lincs_in_ctrdb)}")

    if not df.empty:
        print(f"\n  Per-drug-source results:")
        print(df.to_string(index=False))

        # Head-to-head where both sources evaluated on same dataset+drug
        if len(three_way) > 0:
            print(f"\n  --- Head-to-head comparison (same drug, same dataset) ---")
            for comp in sorted(three_way):
                comp_rows = df[df["drug_component"] == comp]
                for gid in comp_rows["geo_id"].unique():
                    gid_rows = comp_rows[comp_rows["geo_id"] == gid]
                    lincs_row = gid_rows[gid_rows["source"] == "LINCS"]
                    cdsdb_row = gid_rows[gid_rows["source"] == "CDS-DB"]
                    if not lincs_row.empty and not cdsdb_row.empty:
                        la = lincs_row.iloc[0]["auc"]
                        ca = cdsdb_row.iloc[0]["auc"]
                        winner = (
                            "CDS-DB" if ca > la else
                            "LINCS" if la > ca else
                            "TIE"
                        )
                        print(
                            f"    {comp} @ {gid}: "
                            f"LINCS AUC={la:.4f}  CDS-DB AUC={ca:.4f}  "
                            f"-> {winner}"
                        )
    else:
        print("\n  No reversal results could be computed.")
        print("  This is a DATA LIMITATION: the CDS-DB drugs")
        print("  (letrozole, anastrozole, trastuzumab+lapatinib)")
        print("  do not appear as single agents in the downloaded")
        print("  CTR-DB datasets, which feature combo chemo regimens.")
        print()
        print("  Reporting qualitative comparison of available data:")
        _qualitative_report(cdsdb_sigs, lincs_sigs, ctrdb)

    return df


def _qualitative_report(
    cdsdb_sigs: dict[str, pd.Series],
    lincs_sigs: dict[str, pd.Series],
    ctrdb: dict,
) -> None:
    """
    When overlap is insufficient, report what signatures are available
    and any qualitative observations.
    """
    print()
    print("  CDS-DB signature sizes (genes per drug):")
    for drug, sig in cdsdb_sigs.items():
        print(f"    {drug}: {len(sig)} genes")

    print()
    print("  CTR-DB dataset drugs vs CDS-DB/LINCS:")
    for gid, (expr, lab, drug_f) in sorted(ctrdb.items()):
        components = _extract_components(drug_f)
        cdsdb_match = [c for c in components if c in cdsdb_sigs]
        lincs_match = [c for c in components if c in lincs_sigs]
        print(
            f"    {gid}: regimen={drug_f[:50]:50s} "
            f"components={components}  "
            f"cdsdb_match={cdsdb_match}  lincs_match={lincs_match}"
        )

    # Try to evaluate anyway on any partial match
    print()
    print("  Attempting partial evaluations with component-level matching...")

    for gid, (expr, lab, drug_f) in sorted(ctrdb.items()):
        components = _extract_components(drug_f)
        for comp in components:
            has_lincs = comp in lincs_sigs
            has_cdsdb = comp in cdsdb_sigs
            if has_lincs or has_cdsdb:
                gene_cols = [
                    c for c in expr.columns
                    if isinstance(c, str) and c not in {"sample_id", "patient_id"}
                ]
                if len(gene_cols) < 50:
                    continue
                expr_sub = expr[gene_cols]
                mu = expr_sub.mean(axis=0)
                sd = expr_sub.std(axis=0).replace(0, 1)
                expr_z = (expr_sub - mu) / sd

                for source, sigs in [("LINCS", lincs_sigs), ("CDS-DB", cdsdb_sigs)]:
                    if comp in sigs:
                        res = _evaluate_source(
                            expr_z, lab, sigs[comp], source, comp, gid
                        )
                        if res:
                            print(
                                f"    {gid} | {comp} | {source} | "
                                f"AUC={res['auc']} | p={res['mannwhitney_p']} | "
                                f"n={res['n_patients']} | genes={res['n_genes_overlap']}"
                            )


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_experiment()
