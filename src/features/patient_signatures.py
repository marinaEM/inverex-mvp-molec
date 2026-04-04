"""
CDS-DB vs LINCS signature comparison for drug response prediction.

Builds a hybrid signature bank from:
  - CDS-DB: patient-level drug perturbation signatures (pre/post treatment)
  - LINCS L1000: cell-line drug perturbation signatures

For each CTR-DB validation cohort, computes reversal scores using both
signature sources and compares AUC for responder/non-responder discrimination.

Workflow:
  1. Load CDS-DB patient perturbation signatures (with enhanced pairing)
  2. Load LINCS cell-line signatures
  3. Build hybrid signature bank (average per-drug signature from each source)
  4. For each CTR-DB dataset, compute reversal scores against both
  5. Compute AUC per drug per source, produce comparison tables
"""
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DATA_CACHE, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drug name normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_drug_name(name: str) -> str:
    """Lower-case, strip whitespace, replace common separators."""
    return name.strip().lower().replace("-", "").replace(" ", "")


def _extract_drug_components(combo_name: str) -> list[str]:
    """Split a CTR-DB combo drug string into individual drug names."""
    # Remove parenthesised abbreviations like "TFAC (...)"
    cleaned = re.sub(r"^[A-Z]+\s*\(", "(", combo_name)
    parts = re.split(r"[+/]", combo_name)
    components = []
    for p in parts:
        # Remove parenthesised abbreviation prefixes
        p = re.sub(r"^\s*[A-Z]+\s*\(", "", p)
        p = p.strip().strip("()")
        if p and len(p) > 2:
            components.append(p.strip())
    return components


# ---------------------------------------------------------------------------
# 1. Enhanced CDS-DB loading with manual pair extraction
# ---------------------------------------------------------------------------

def _extract_pairs_gse87455(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    GSE87455 – paired Baseline / Cycle 2 biopsies in breast cancer.
    Drug: Letrozole (aromatase inhibitor neoadjuvant).
    Titles: Patient_NNN_Baseline / Patient_NNN_Cycle 2
    """
    expr_path = data_dir / "GSE87455" / "GSE87455_expression.parquet"
    if not expr_path.exists():
        return None

    try:
        import GEOparse
    except ImportError:
        logger.warning("GEOparse not available; cannot extract GSE87455 pairs")
        return None

    expr = pd.read_parquet(expr_path)

    try:
        gse = GEOparse.get_GEO(
            geo="GSE87455", destdir=str(data_dir / "GSE87455"), silent=True
        )
    except Exception as exc:
        logger.warning(f"Could not load GSE87455 GEO object: {exc}")
        return None

    # Build sample metadata
    sample_meta = {}
    for gsm_name, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        sample_meta[gsm_name] = title

    # Parse pairs from titles: "Patient_101_Baseline" / "Patient_101_Cycle 2"
    patients = {}
    for sid, title in sample_meta.items():
        m = re.match(r"Patient_(\d+)_(.*)", title)
        if m:
            pid = m.group(1)
            timepoint = m.group(2).strip().lower()
            if pid not in patients:
                patients[pid] = {}
            if "baseline" in timepoint:
                patients[pid]["pre"] = sid
            elif "cycle" in timepoint:
                patients[pid]["post"] = sid

    pairs = {pid: (v["pre"], v["post"])
             for pid, v in patients.items()
             if "pre" in v and "post" in v}

    if not pairs:
        logger.info("GSE87455: no pairs found from titles")
        return None

    logger.info(f"GSE87455: found {len(pairs)} pre/post pairs")
    return _compute_lfc_from_pairs(expr, pairs, "Letrozole", "GSE87455")


def _extract_pairs_gse20181(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    GSE20181 – pre/post Letrozole in ER+ breast cancer.
    Titles: "10A;pretreatment;..." / "10B;...Letrozole..."
    A suffix = pre, B suffix = post, same patient number.
    """
    expr_path = data_dir / "GSE20181" / "GSE20181_expression.parquet"
    if not expr_path.exists():
        return None

    try:
        import GEOparse
    except ImportError:
        return None

    expr = pd.read_parquet(expr_path)

    try:
        gse = GEOparse.get_GEO(
            geo="GSE20181", destdir=str(data_dir / "GSE20181"), silent=True
        )
    except Exception as exc:
        logger.warning(f"Could not load GSE20181 GEO object: {exc}")
        return None

    # Parse sample titles
    pre_samples = {}   # patient_num -> (gsm_id, response)
    post_samples = {}

    for gsm_name, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        # Format: "10A;pretreatment;female;breast tumor; responder"
        parts = [p.strip() for p in title.split(";")]
        if not parts:
            continue

        sample_id_part = parts[0]  # e.g. "10A" or "10B"
        m = re.match(r"(\d+)([AB])", sample_id_part)
        if not m:
            continue
        pid = m.group(1)
        suffix = m.group(2)

        # Extract response if present
        response = None
        for p in parts:
            p_lower = p.strip().lower()
            if "responder" in p_lower and "non" not in p_lower:
                response = "responder"
            elif "nonresponder" in p_lower or "non-responder" in p_lower:
                response = "nonresponder"

        if suffix == "A":
            pre_samples[pid] = (gsm_name, response)
        elif suffix == "B":
            post_samples[pid] = (gsm_name, response)

    pairs = {}
    response_map = {}
    for pid in pre_samples:
        if pid in post_samples:
            pairs[pid] = (pre_samples[pid][0], post_samples[pid][0])
            resp = pre_samples[pid][1] or post_samples[pid][1]
            if resp:
                response_map[pid] = resp

    if not pairs:
        logger.info("GSE20181: no pairs found")
        return None

    logger.info(f"GSE20181: found {len(pairs)} pre/post pairs")
    sigs = _compute_lfc_from_pairs(expr, pairs, "Letrozole", "GSE20181")

    # Attach response labels
    if sigs is not None and response_map:
        sigs["patient_response"] = sigs["patient_id"].map(response_map)

    return sigs


def _extract_pairs_gse55374(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    GSE55374 – pre/post Letrozole in ILC breast cancer.
    Has 'subject' and 'treatment' fields in characteristics.
    """
    expr_path = data_dir / "GSE55374" / "GSE55374_expression.parquet"
    if not expr_path.exists():
        return None

    try:
        import GEOparse
    except ImportError:
        return None

    expr = pd.read_parquet(expr_path)

    try:
        gse = GEOparse.get_GEO(
            geo="GSE55374", destdir=str(data_dir / "GSE55374"), silent=True
        )
    except Exception as exc:
        logger.warning(f"Could not load GSE55374 GEO object: {exc}")
        return None

    pre_samples = {}
    post_samples = {}
    response_map = {}

    for gsm_name, gsm in gse.gsms.items():
        chars = gsm.metadata.get("characteristics_ch1", [])
        info = {}
        for item in chars:
            item_str = str(item).strip()
            if ":" in item_str:
                key, val = item_str.split(":", 1)
                info[key.strip().lower()] = val.strip()

        patient = info.get("subject", "").replace("patient ", "")
        treatment = info.get("treatment", "").lower()
        response = info.get("clinical response", "").lower()

        if not patient:
            continue

        if "pre" in treatment:
            pre_samples[patient] = gsm_name
        elif "post" in treatment or "on" in treatment:
            post_samples[patient] = gsm_name

        if response:
            response_map[patient] = response

    pairs = {}
    for pid in pre_samples:
        if pid in post_samples:
            pairs[pid] = (pre_samples[pid], post_samples[pid])

    if not pairs:
        # Try from titles: "Breast.Tumour.ILC.PreLet.patient182"
        for gsm_name, gsm in gse.gsms.items():
            title = gsm.metadata.get("title", [""])[0]
            m_pre = re.search(r"PreLet\.patient(\d+)", title)
            m_post = re.search(r"PostLet\.patient(\d+)", title)
            if m_pre:
                pre_samples[m_pre.group(1)] = gsm_name
            elif m_post:
                post_samples[m_post.group(1)] = gsm_name

        pairs = {pid: (pre_samples[pid], post_samples[pid])
                 for pid in pre_samples if pid in post_samples}

    if not pairs:
        logger.info("GSE55374: no pairs found")
        return None

    logger.info(f"GSE55374: found {len(pairs)} pre/post pairs")
    sigs = _compute_lfc_from_pairs(expr, pairs, "Letrozole", "GSE55374")

    if sigs is not None and response_map:
        sigs["patient_response"] = sigs["patient_id"].map(response_map)

    return sigs


def _compute_lfc_from_pairs(
    expr: pd.DataFrame,
    pairs: dict[str, tuple[str, str]],
    drug: str,
    geo_id: str,
) -> Optional[pd.DataFrame]:
    """Compute log fold-change signatures from pre/post pairs."""
    rows = []
    for patient_id, (pre_sid, post_sid) in pairs.items():
        if pre_sid not in expr.index or post_sid not in expr.index:
            continue
        lfc = expr.loc[post_sid] - expr.loc[pre_sid]
        for gene, val in lfc.items():
            if pd.notna(val):
                rows.append({
                    "patient_id": str(patient_id),
                    "gene_symbol": gene,
                    "log_fc": float(val),
                    "drug": drug,
                    "geo_id": geo_id,
                })
    if rows:
        return pd.DataFrame(rows)
    return None


def load_enhanced_cdsdb_signatures(
    data_dir: Path = DATA_RAW / "cdsdb",
) -> pd.DataFrame:
    """
    Load CDS-DB signatures with enhanced pair extraction.

    First tries the standard load_breast_perturbation_signatures,
    then augments with manually extracted pairs from known datasets.
    """
    from src.data_ingestion.cdsdb import load_breast_perturbation_signatures

    sigs = load_breast_perturbation_signatures(data_dir)
    parts = [sigs] if not sigs.empty else []

    logger.info("Attempting enhanced pair extraction for CDS-DB datasets ...")

    # Try each dataset with custom extraction
    extractors = [
        _extract_pairs_gse87455,
        _extract_pairs_gse20181,
        _extract_pairs_gse55374,
    ]

    for extractor in extractors:
        try:
            result = extractor(data_dir)
            if result is not None and len(result) > 0:
                # Avoid duplicates: skip if this geo_id already loaded
                geo_id = result["geo_id"].iloc[0]
                already = any(
                    not p.empty and (p["geo_id"] == geo_id).any()
                    for p in parts
                )
                if not already:
                    parts.append(result)
                    logger.info(
                        f"  Added {geo_id}: {result['patient_id'].nunique()} patients, "
                        f"{len(result)} signature rows"
                    )
        except Exception as exc:
            logger.warning(f"Enhanced extraction failed: {exc}")

    if parts:
        combined = pd.concat(parts, ignore_index=True)
        logger.info(
            f"Total CDS-DB signatures: {len(combined)} rows, "
            f"{combined['drug'].nunique()} drugs, "
            f"{combined['patient_id'].nunique()} patients"
        )
        return combined

    logger.warning("No CDS-DB perturbation signatures available")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. Build hybrid signature bank
# ---------------------------------------------------------------------------

def build_drug_signature_bank(
    cdsdb_sigs: pd.DataFrame,
    lincs_path: Path = DATA_CACHE / "all_cellline_drug_signatures.parquet",
    gene_info_path: Path = DATA_CACHE / "geneinfo_beta_input.txt",
) -> dict[str, dict]:
    """
    Build a signature bank with average drug signatures from both sources.

    Returns dict keyed by normalised drug name:
        {
            "drug_name": {
                "lincs_sig": pd.Series (gene -> z-score) or None,
                "cdsdb_sig": pd.Series (gene -> log_fc) or None,
                "lincs_n_sigs": int,
                "cdsdb_n_patients": int,
                "drug_display": str,
            }
        }
    """
    # Load landmark genes for consistent gene space
    gene_info = pd.read_csv(gene_info_path, sep="\t")
    landmark_genes = gene_info["gene_symbol"].tolist()
    logger.info(f"Landmark gene set: {len(landmark_genes)} genes")

    # Load LINCS signatures
    lincs = pd.read_parquet(lincs_path)
    meta_cols = {"sig_id", "pert_id", "pert_iname", "cell_id", "pert_idose", "dose_um"}
    lincs_gene_cols = [c for c in lincs.columns if c not in meta_cols]

    bank = {}

    # --- LINCS average signatures per drug ---
    for drug_name, group in lincs.groupby("pert_iname"):
        norm_name = _normalise_drug_name(drug_name)
        avg_sig = group[lincs_gene_cols].mean()

        # Restrict to landmark genes that exist in both
        common = [g for g in landmark_genes if g in avg_sig.index]
        if not common:
            continue

        bank[norm_name] = {
            "lincs_sig": avg_sig[common],
            "cdsdb_sig": None,
            "lincs_n_sigs": len(group),
            "cdsdb_n_patients": 0,
            "drug_display": drug_name,
        }

    # --- CDS-DB average signatures per drug ---
    if not cdsdb_sigs.empty:
        for drug_name, drug_group in cdsdb_sigs.groupby("drug"):
            norm_name = _normalise_drug_name(drug_name)

            # Average across patients: pivot to (patient x gene), then mean
            pivot = drug_group.pivot_table(
                index="patient_id", columns="gene_symbol",
                values="log_fc", aggfunc="mean",
            )
            avg_lfc = pivot.mean(axis=0)

            # Restrict to landmark genes
            common = [g for g in landmark_genes if g in avg_lfc.index]
            if not common:
                continue

            if norm_name in bank:
                bank[norm_name]["cdsdb_sig"] = avg_lfc[common]
                bank[norm_name]["cdsdb_n_patients"] = drug_group["patient_id"].nunique()
            else:
                bank[norm_name] = {
                    "lincs_sig": None,
                    "cdsdb_sig": avg_lfc[common],
                    "lincs_n_sigs": 0,
                    "cdsdb_n_patients": drug_group["patient_id"].nunique(),
                    "drug_display": drug_name,
                }

    # Summary
    both = sum(1 for v in bank.values() if v["lincs_sig"] is not None and v["cdsdb_sig"] is not None)
    lincs_only = sum(1 for v in bank.values() if v["lincs_sig"] is not None and v["cdsdb_sig"] is None)
    cdsdb_only = sum(1 for v in bank.values() if v["lincs_sig"] is None and v["cdsdb_sig"] is not None)
    logger.info(
        f"Signature bank: {len(bank)} drugs total — "
        f"both={both}, LINCS-only={lincs_only}, CDS-DB-only={cdsdb_only}"
    )

    return bank


# ---------------------------------------------------------------------------
# 3. Reversal score computation
# ---------------------------------------------------------------------------

def compute_reversal_score(
    patient_expr: pd.Series,
    drug_sig: pd.Series,
) -> float:
    """
    Compute reversal score as negative Pearson correlation between
    patient expression profile and drug perturbation signature.

    A high reversal score means the drug signature is anti-correlated
    with the patient's disease expression — the drug "reverses" the
    disease state.

    Parameters:
        patient_expr: gene expression values for one patient
        drug_sig: drug perturbation signature (z-scores or log-FC)

    Returns:
        reversal score (float); higher = more reversal
    """
    # Align on common genes
    common = patient_expr.index.intersection(drug_sig.index)
    if len(common) < 10:
        return np.nan

    p = patient_expr[common].values.astype(float)
    d = drug_sig[common].values.astype(float)

    # Remove NaNs
    valid = np.isfinite(p) & np.isfinite(d)
    if valid.sum() < 10:
        return np.nan

    corr = np.corrcoef(p[valid], d[valid])[0, 1]
    return -corr  # reversal = negative correlation


def compute_reversal_scores_for_cohort(
    expr_df: pd.DataFrame,
    drug_sig: pd.Series,
) -> pd.Series:
    """
    Compute reversal scores for all patients in a cohort.

    Parameters:
        expr_df: (patients x genes) expression matrix
        drug_sig: drug signature (genes,)

    Returns:
        Series indexed by patient IDs with reversal scores
    """
    scores = {}
    for patient_id in expr_df.index:
        scores[patient_id] = compute_reversal_score(expr_df.loc[patient_id], drug_sig)
    return pd.Series(scores)


# ---------------------------------------------------------------------------
# 4. CTR-DB validation
# ---------------------------------------------------------------------------

def _match_ctrdb_drug_to_bank(
    ctrdb_drug: str,
    bank: dict,
) -> list[str]:
    """
    Match a CTR-DB drug (possibly combo) to signature bank entries.

    Returns list of normalised drug names found in the bank.
    """
    # Try exact match first
    norm = _normalise_drug_name(ctrdb_drug)
    if norm in bank:
        return [norm]

    # Try component-wise matching for combos
    components = _extract_drug_components(ctrdb_drug)
    matches = []
    for comp in components:
        comp_norm = _normalise_drug_name(comp)
        if comp_norm in bank:
            matches.append(comp_norm)
        else:
            # Fuzzy: check if component is a substring of any bank key
            for bk in bank:
                if comp_norm in bk or bk in comp_norm:
                    matches.append(bk)
                    break

    return list(set(matches))


def _average_signatures(
    drug_names: list[str],
    bank: dict,
    source: str,
) -> Optional[pd.Series]:
    """
    Average signatures from multiple drugs (for combo regimens).

    source: "lincs_sig" or "cdsdb_sig"
    """
    sigs = []
    for dn in drug_names:
        if dn in bank and bank[dn][source] is not None:
            sigs.append(bank[dn][source])

    if not sigs:
        return None

    # Align on common genes
    common_genes = sigs[0].index
    for s in sigs[1:]:
        common_genes = common_genes.intersection(s.index)

    if len(common_genes) < 10:
        return None

    combined = pd.DataFrame({i: s[common_genes] for i, s in enumerate(sigs)})
    return combined.mean(axis=1)


def validate_on_ctrdb(
    bank: dict,
    ctrdb_dir: Path = DATA_RAW / "ctrdb",
) -> pd.DataFrame:
    """
    For each CTR-DB dataset, compute LINCS and CDS-DB reversal scores,
    then measure AUC for responder/non-responder discrimination.

    Returns a DataFrame with per-drug AUC comparison.
    """
    catalog = pd.read_csv(ctrdb_dir / "catalog.csv")
    downloaded = catalog[catalog["downloaded"] == True]

    results = []

    for _, row in downloaded.iterrows():
        geo_id = row["geo_source"]
        ctrdb_drug = row["drug"]
        dataset_id = row["dataset_id"]
        ds_dir = ctrdb_dir / geo_id

        expr_path = ds_dir / f"{geo_id}_expression.parquet"
        resp_path = ds_dir / "response_labels.parquet"

        if not expr_path.exists() or not resp_path.exists():
            logger.warning(f"Missing data for {geo_id}")
            continue

        expr = pd.read_parquet(expr_path)
        resp = pd.read_parquet(resp_path)

        # Align patients
        common_patients = expr.index.intersection(resp.index)
        if len(common_patients) < 10:
            logger.warning(f"{geo_id}: too few common patients ({len(common_patients)})")
            continue

        expr = expr.loc[common_patients]
        labels = resp.loc[common_patients, "response"]

        # Need both classes
        if labels.nunique() < 2:
            logger.warning(f"{geo_id}: only one response class")
            continue

        # Match drugs
        matched_drugs = _match_ctrdb_drug_to_bank(ctrdb_drug, bank)
        if not matched_drugs:
            logger.info(f"{geo_id} ({ctrdb_drug}): no LINCS/CDS-DB drug match")
            results.append({
                "dataset_id": dataset_id,
                "geo_id": geo_id,
                "ctrdb_drug": ctrdb_drug,
                "matched_drugs": "",
                "n_patients": len(common_patients),
                "n_responders": int(labels.sum()),
                "n_nonresponders": int((labels == 0).sum()),
                "auc_lincs": np.nan,
                "auc_cdsdb": np.nan,
                "lincs_available": False,
                "cdsdb_available": False,
                "n_common_genes_lincs": 0,
                "n_common_genes_cdsdb": 0,
            })
            continue

        matched_str = "+".join(matched_drugs)
        logger.info(
            f"{geo_id} ({ctrdb_drug}): matched to [{matched_str}], "
            f"{len(common_patients)} patients"
        )

        # Compute LINCS reversal scores
        lincs_sig = _average_signatures(matched_drugs, bank, "lincs_sig")
        auc_lincs = np.nan
        lincs_avail = lincs_sig is not None
        n_genes_lincs = 0

        if lincs_sig is not None:
            n_genes_lincs = len(lincs_sig.index.intersection(expr.columns))
            rev_lincs = compute_reversal_scores_for_cohort(expr, lincs_sig)
            rev_lincs = rev_lincs.loc[common_patients]
            valid = rev_lincs.notna()
            if valid.sum() >= 10 and labels[valid].nunique() == 2:
                try:
                    auc_lincs = roc_auc_score(labels[valid], rev_lincs[valid])
                except ValueError:
                    pass
                logger.info(f"  LINCS AUC = {auc_lincs:.3f} ({n_genes_lincs} genes)")

        # Compute CDS-DB reversal scores
        cdsdb_sig = _average_signatures(matched_drugs, bank, "cdsdb_sig")
        auc_cdsdb = np.nan
        cdsdb_avail = cdsdb_sig is not None
        n_genes_cdsdb = 0

        if cdsdb_sig is not None:
            n_genes_cdsdb = len(cdsdb_sig.index.intersection(expr.columns))
            rev_cdsdb = compute_reversal_scores_for_cohort(expr, cdsdb_sig)
            rev_cdsdb = rev_cdsdb.loc[common_patients]
            valid = rev_cdsdb.notna()
            if valid.sum() >= 10 and labels[valid].nunique() == 2:
                try:
                    auc_cdsdb = roc_auc_score(labels[valid], rev_cdsdb[valid])
                except ValueError:
                    pass
                logger.info(f"  CDS-DB AUC = {auc_cdsdb:.3f} ({n_genes_cdsdb} genes)")

        results.append({
            "dataset_id": dataset_id,
            "geo_id": geo_id,
            "ctrdb_drug": ctrdb_drug,
            "matched_drugs": matched_str,
            "n_patients": len(common_patients),
            "n_responders": int(labels.sum()),
            "n_nonresponders": int((labels == 0).sum()),
            "auc_lincs": auc_lincs,
            "auc_cdsdb": auc_cdsdb,
            "lincs_available": lincs_avail,
            "cdsdb_available": cdsdb_avail,
            "n_common_genes_lincs": n_genes_lincs,
            "n_common_genes_cdsdb": n_genes_cdsdb,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 4b. CDS-DB self-validation (leave-one-out)
# ---------------------------------------------------------------------------

def validate_on_cdsdb_internal(
    cdsdb_sigs: pd.DataFrame,
    bank: dict,
    cdsdb_dir: Path = DATA_RAW / "cdsdb",
) -> pd.DataFrame:
    """
    For CDS-DB datasets with embedded response labels (e.g. GSE20181),
    perform leave-one-out cross-validation:
      - For each patient, compute CDS-DB signature from OTHER patients
      - Compute reversal score on the held-out patient's pre-treatment expr
      - Also compute reversal using LINCS signatures
      - Compare AUCs

    This is the key comparison for drugs with both LINCS and CDS-DB data.
    """
    results = []

    # GSE20181 has response labels in the CDS-DB signatures
    gse20181_sigs = cdsdb_sigs[cdsdb_sigs["geo_id"] == "GSE20181"]
    if gse20181_sigs.empty or "patient_response" not in gse20181_sigs.columns:
        logger.info("GSE20181 CDS-DB sigs not available with response labels")
    else:
        result = _loo_validation_gse20181(gse20181_sigs, bank, cdsdb_dir)
        if result is not None:
            results.append(result)

    # Also try GSE87455 (letrozole) if it has response data somewhere
    # GSE87455 doesn't have embedded response labels, skip for now

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()


def _loo_validation_gse20181(
    sigs_with_response: pd.DataFrame,
    bank: dict,
    cdsdb_dir: Path,
) -> Optional[dict]:
    """
    Leave-one-out validation on GSE20181 (Letrozole) using pre-treatment
    expression and both LINCS and CDS-DB drug signatures.
    """
    # Load pre-treatment expression
    expr_path = cdsdb_dir / "GSE20181" / "GSE20181_expression.parquet"
    if not expr_path.exists():
        return None

    expr = pd.read_parquet(expr_path)

    # Get gene info for landmark genes
    gene_info = pd.read_csv(DATA_CACHE / "geneinfo_beta_input.txt", sep="\t")
    landmark_genes = gene_info["gene_symbol"].tolist()

    # Build patient response map from the CDS-DB sigs
    patient_response = (
        sigs_with_response[["patient_id", "patient_response"]]
        .drop_duplicates()
        .dropna(subset=["patient_response"])
        .set_index("patient_id")["patient_response"]
    )

    if len(patient_response) < 5:
        logger.info("GSE20181: too few patients with response labels")
        return None

    # We need to map patient IDs to pre-treatment sample IDs
    # In GSE20181, patient "10" -> pre-treatment sample "GSM125123" (10A)
    # Reload GEO to get this mapping
    try:
        import GEOparse
        gse = GEOparse.get_GEO(
            geo="GSE20181", destdir=str(cdsdb_dir / "GSE20181"), silent=True
        )
    except Exception as exc:
        logger.warning(f"Cannot load GSE20181 for LOO validation: {exc}")
        return None

    pre_sample_map = {}  # patient_id -> pre_treatment_sample_id
    for gsm_name, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        parts = [p.strip() for p in title.split(";")]
        if not parts:
            continue
        m = re.match(r"(\d+)([AB])", parts[0])
        if m and m.group(2) == "A":  # A = pre-treatment
            pre_sample_map[m.group(1)] = gsm_name

    # Build labels and expression for pre-treatment samples
    patient_ids = []
    sample_ids = []
    labels = []

    for pid, resp in patient_response.items():
        if pid in pre_sample_map and pre_sample_map[pid] in expr.index:
            patient_ids.append(pid)
            sample_ids.append(pre_sample_map[pid])
            labels.append(1 if resp == "responder" else 0)

    if len(patient_ids) < 5:
        logger.info("GSE20181: insufficient matched pre-treatment samples")
        return None

    labels = np.array(labels)
    logger.info(
        f"GSE20181 LOO validation: {len(patient_ids)} patients, "
        f"{labels.sum()} responders, {(labels==0).sum()} non-responders"
    )

    if len(np.unique(labels)) < 2:
        return None

    pre_expr = expr.loc[sample_ids]

    # --- LINCS reversal scores ---
    norm_drug = _normalise_drug_name("Letrozole")
    lincs_sig = bank.get(norm_drug, {}).get("lincs_sig")
    auc_lincs = np.nan
    n_genes_lincs = 0

    if lincs_sig is not None:
        n_genes_lincs = len(lincs_sig.index.intersection(pre_expr.columns))
        rev_lincs = compute_reversal_scores_for_cohort(pre_expr, lincs_sig)
        valid = rev_lincs.notna()
        if valid.sum() >= 5 and len(np.unique(labels[valid.values])) == 2:
            try:
                auc_lincs = roc_auc_score(labels[valid.values], rev_lincs[valid].values)
            except ValueError:
                pass
        logger.info(f"  LINCS AUC = {auc_lincs:.3f} ({n_genes_lincs} genes)")

    # --- CDS-DB leave-one-out reversal scores ---
    # For each patient, build average signature from OTHER patients, score
    all_sigs_pivot = sigs_with_response.pivot_table(
        index="patient_id", columns="gene_symbol",
        values="log_fc", aggfunc="mean",
    )
    # Restrict to landmark genes
    common_lm = [g for g in landmark_genes if g in all_sigs_pivot.columns]

    loo_scores = []
    for i, pid in enumerate(patient_ids):
        # Leave-one-out: average signature from all other patients
        other_pids = [p for p in all_sigs_pivot.index if p != pid]
        if len(other_pids) < 2:
            loo_scores.append(np.nan)
            continue
        loo_sig = all_sigs_pivot.loc[other_pids, common_lm].mean(axis=0)
        pre_sample = sample_ids[i]
        score = compute_reversal_score(pre_expr.loc[pre_sample], loo_sig)
        loo_scores.append(score)

    loo_scores = np.array(loo_scores)
    auc_cdsdb = np.nan
    n_genes_cdsdb = len(common_lm)
    valid_mask = np.isfinite(loo_scores)
    if valid_mask.sum() >= 5 and len(np.unique(labels[valid_mask])) == 2:
        try:
            auc_cdsdb = roc_auc_score(labels[valid_mask], loo_scores[valid_mask])
        except ValueError:
            pass
    logger.info(f"  CDS-DB LOO AUC = {auc_cdsdb:.3f} ({n_genes_cdsdb} genes)")

    return {
        "dataset_id": "CDSDB_GSE20181_LOO",
        "geo_id": "GSE20181",
        "ctrdb_drug": "Letrozole",
        "matched_drugs": "letrozole",
        "n_patients": len(patient_ids),
        "n_responders": int(labels.sum()),
        "n_nonresponders": int((labels == 0).sum()),
        "auc_lincs": auc_lincs,
        "auc_cdsdb": auc_cdsdb,
        "lincs_available": lincs_sig is not None,
        "cdsdb_available": True,
        "n_common_genes_lincs": n_genes_lincs,
        "n_common_genes_cdsdb": n_genes_cdsdb,
    }


# ---------------------------------------------------------------------------
# 4c. Signature correlation analysis
# ---------------------------------------------------------------------------

def compute_signature_correlations(bank: dict) -> pd.DataFrame:
    """
    For drugs with both LINCS and CDS-DB signatures, compute
    the Pearson correlation between the two signature vectors.

    This measures how similar cell-line and patient-derived
    signatures are for the same drug.
    """
    rows = []
    for name, info in bank.items():
        if info["lincs_sig"] is not None and info["cdsdb_sig"] is not None:
            lincs = info["lincs_sig"]
            cdsdb = info["cdsdb_sig"]
            common = lincs.index.intersection(cdsdb.index)
            if len(common) < 10:
                continue
            l_vals = lincs[common].values.astype(float)
            c_vals = cdsdb[common].values.astype(float)
            valid = np.isfinite(l_vals) & np.isfinite(c_vals)
            if valid.sum() < 10:
                continue
            corr = np.corrcoef(l_vals[valid], c_vals[valid])[0, 1]
            rows.append({
                "drug": info["drug_display"],
                "n_common_genes": int(valid.sum()),
                "lincs_cdsdb_correlation": round(corr, 4),
                "lincs_n_sigs": info["lincs_n_sigs"],
                "cdsdb_n_patients": info["cdsdb_n_patients"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Summary and reporting
# ---------------------------------------------------------------------------

def build_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregate summary comparing LINCS vs CDS-DB.
    """
    rows = []

    # Overall LINCS stats
    lincs_valid = comparison_df[comparison_df["auc_lincs"].notna()]
    rows.append({
        "metric": "LINCS: datasets with AUC",
        "value": len(lincs_valid),
    })
    if len(lincs_valid) > 0:
        rows.append({
            "metric": "LINCS: mean AUC",
            "value": round(lincs_valid["auc_lincs"].mean(), 3),
        })
        rows.append({
            "metric": "LINCS: median AUC",
            "value": round(lincs_valid["auc_lincs"].median(), 3),
        })
        rows.append({
            "metric": "LINCS: AUC > 0.5 (better than random)",
            "value": int((lincs_valid["auc_lincs"] > 0.5).sum()),
        })

    # Overall CDS-DB stats
    cdsdb_valid = comparison_df[comparison_df["auc_cdsdb"].notna()]
    rows.append({
        "metric": "CDS-DB: datasets with AUC",
        "value": len(cdsdb_valid),
    })
    if len(cdsdb_valid) > 0:
        rows.append({
            "metric": "CDS-DB: mean AUC",
            "value": round(cdsdb_valid["auc_cdsdb"].mean(), 3),
        })
        rows.append({
            "metric": "CDS-DB: median AUC",
            "value": round(cdsdb_valid["auc_cdsdb"].median(), 3),
        })
        rows.append({
            "metric": "CDS-DB: AUC > 0.5 (better than random)",
            "value": int((cdsdb_valid["auc_cdsdb"] > 0.5).sum()),
        })

    # Head-to-head where both available
    both = comparison_df[
        comparison_df["auc_lincs"].notna() & comparison_df["auc_cdsdb"].notna()
    ]
    rows.append({
        "metric": "Both available: dataset count",
        "value": len(both),
    })
    if len(both) > 0:
        rows.append({
            "metric": "Both: LINCS mean AUC",
            "value": round(both["auc_lincs"].mean(), 3),
        })
        rows.append({
            "metric": "Both: CDS-DB mean AUC",
            "value": round(both["auc_cdsdb"].mean(), 3),
        })
        lincs_wins = int((both["auc_lincs"] > both["auc_cdsdb"]).sum())
        cdsdb_wins = int((both["auc_cdsdb"] > both["auc_lincs"]).sum())
        ties = len(both) - lincs_wins - cdsdb_wins
        rows.append({"metric": "Both: LINCS wins", "value": lincs_wins})
        rows.append({"metric": "Both: CDS-DB wins", "value": cdsdb_wins})
        rows.append({"metric": "Both: ties", "value": ties})
        rows.append({
            "metric": "Both: mean AUC difference (CDS-DB - LINCS)",
            "value": round((both["auc_cdsdb"] - both["auc_lincs"]).mean(), 3),
        })

    # Signature bank stats
    rows.append({
        "metric": "Total CTR-DB datasets evaluated",
        "value": len(comparison_df),
    })
    rows.append({
        "metric": "CTR-DB datasets with drug match",
        "value": int((comparison_df["matched_drugs"] != "").sum()),
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_comparison(
    output_dir: Path = RESULTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full CDS-DB vs LINCS comparison pipeline.

    Returns:
        comparison_df: per-drug AUC comparison
        summary_df: aggregate summary
    """
    logger.info("=" * 70)
    logger.info("CDS-DB vs LINCS Signature Comparison")
    logger.info("=" * 70)

    # Step 1: Load CDS-DB signatures (with enhanced extraction)
    logger.info("\n--- Step 1: Loading CDS-DB patient signatures ---")
    cdsdb_sigs = load_enhanced_cdsdb_signatures()
    if cdsdb_sigs.empty:
        logger.warning("CDS-DB data unavailable — will only compute LINCS AUCs")

    # Step 2: Build hybrid signature bank
    logger.info("\n--- Step 2: Building hybrid signature bank ---")
    bank = build_drug_signature_bank(cdsdb_sigs)

    # Log what we have
    both_count = sum(
        1 for v in bank.values()
        if v["lincs_sig"] is not None and v["cdsdb_sig"] is not None
    )
    logger.info(f"Drugs with BOTH LINCS and CDS-DB signatures: {both_count}")
    for name, info in bank.items():
        if info["lincs_sig"] is not None and info["cdsdb_sig"] is not None:
            logger.info(
                f"  {info['drug_display']}: "
                f"LINCS={info['lincs_n_sigs']} sigs, "
                f"CDS-DB={info['cdsdb_n_patients']} patients"
            )

    # Step 3: Validate on CTR-DB
    logger.info("\n--- Step 3: Validating on CTR-DB cohorts ---")
    ctrdb_results = validate_on_ctrdb(bank)

    # Step 3b: CDS-DB self-validation (LOO) for drugs with both sources
    logger.info("\n--- Step 3b: CDS-DB internal LOO validation ---")
    loo_results = validate_on_cdsdb_internal(cdsdb_sigs, bank)

    # Step 3c: Signature correlation analysis
    logger.info("\n--- Step 3c: LINCS vs CDS-DB signature correlation ---")
    sig_corr = compute_signature_correlations(bank)
    if not sig_corr.empty:
        logger.info("Signature correlations (LINCS vs CDS-DB):")
        for _, row in sig_corr.iterrows():
            logger.info(
                f"  {row['drug']}: r={row['lincs_cdsdb_correlation']:.3f} "
                f"({row['n_common_genes']} genes)"
            )

    # Combine all results
    parts = [ctrdb_results]
    if not loo_results.empty:
        parts.append(loo_results)
    comparison_df = pd.concat(parts, ignore_index=True)

    # Step 4: Summary
    logger.info("\n--- Step 4: Building summary ---")
    summary_df = build_summary(comparison_df)

    # Add signature correlation info to summary
    if not sig_corr.empty:
        for _, row in sig_corr.iterrows():
            summary_df = pd.concat([summary_df, pd.DataFrame([{
                "metric": f"Signature correlation ({row['drug']}): LINCS vs CDS-DB",
                "value": row["lincs_cdsdb_correlation"],
            }])], ignore_index=True)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    comp_path = output_dir / "cdsdb_vs_lincs_comparison.csv"
    summ_path = output_dir / "cdsdb_vs_lincs_summary.csv"

    comparison_df.to_csv(comp_path, index=False)
    summary_df.to_csv(summ_path, index=False)

    logger.info(f"\nSaved: {comp_path}")
    logger.info(f"Saved: {summ_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    for _, row in summary_df.iterrows():
        logger.info(f"  {row['metric']}: {row['value']}")

    logger.info("\nPer-dataset comparison:")
    if not comparison_df.empty:
        display_cols = [
            "geo_id", "ctrdb_drug", "matched_drugs",
            "n_patients", "auc_lincs", "auc_cdsdb",
        ]
        existing_cols = [c for c in display_cols if c in comparison_df.columns]
        logger.info("\n" + comparison_df[existing_cols].to_string(index=False))

    return comparison_df, summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    comparison_df, summary_df = run_comparison()
