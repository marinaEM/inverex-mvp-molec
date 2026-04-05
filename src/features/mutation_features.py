"""
Extract mutation / CNV / clinical biomarker features from patient phenotype data.

For each patient, we extract:
  1. Clinical biomarker status (binary 0/1/NaN):
     - ER_positive, HER2_positive, PR_positive
  2. Key gene mutation proxies (binary 0/1/NaN):
     - TP53_mutated, PIK3CA_mutated, BRCA1_mutated, BRCA2_mutated,
       PTEN_mutated, CDH1_mutated, ESR1_mutated, ERBB2_mutated
  3. CNV proxies:
     - ERBB2_amplified (from HER2 IHC/FISH)
  4. Pathway burden features:
     - n_mutations_pi3k (PIK3CA + PTEN + AKT1)
     - n_mutations_ddr (BRCA1 + BRCA2)

Missing data is NaN -- LightGBM handles NaN natively.

Usage:
    pixi run python -m src.features.mutation_features
"""

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
DATA_METADATA = ROOT / "data" / "metadata"

for d in [DATA_CACHE, DATA_METADATA]:
    d.mkdir(parents=True, exist_ok=True)


# ── Column names for the output features ────────────────────────────────
BIOMARKER_COLS = ["ER_positive", "HER2_positive", "PR_positive"]
MUTATION_COLS = [
    "TP53_mutated", "PIK3CA_mutated", "BRCA1_mutated", "BRCA2_mutated",
    "PTEN_mutated", "CDH1_mutated", "ESR1_mutated", "ERBB2_mutated",
]
CNV_COLS = ["ERBB2_amplified"]
PATHWAY_COLS = ["n_mutations_pi3k", "n_mutations_ddr"]
ALL_FEATURE_COLS = BIOMARKER_COLS + MUTATION_COLS + CNV_COLS + PATHWAY_COLS


# ── Positive / negative value canonicalizers ─────────────────────────────
_POS_PATTERNS = re.compile(
    r"^(p|positive|\+|pos|1|yes|ihc\s*3\+?|amplified|mutated|mut)$",
    re.IGNORECASE,
)
_NEG_PATTERNS = re.compile(
    r"^(n|negative|-|neg|0|no|ihc\s*0|ihc\s*1\+?|not\s*amplified|wild.?type|wt)$",
    re.IGNORECASE,
)
_EQUIVOCAL_PATTERNS = re.compile(
    r"^(e|equivocal|indeterminate|intermediate|ihc\s*2\+?)$",
    re.IGNORECASE,
)


def _to_binary(val) -> float:
    """Convert a phenotype value to 0/1/NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s or s.lower() in ("nan", "na", "n/a", "--", "---", "unknown", "unk", ""):
        return np.nan
    if _POS_PATTERNS.match(s):
        return 1.0
    if _NEG_PATTERNS.match(s):
        return 0.0
    if _EQUIVOCAL_PATTERNS.match(s):
        return np.nan  # equivocal -> missing
    # Try numeric
    try:
        v = float(s)
        if v == 1:
            return 1.0
        if v == 0:
            return 0.0
        return np.nan
    except ValueError:
        return np.nan


# ======================================================================
# Per-dataset phenotype extractors
# ======================================================================

def _extract_from_geo_soft(geo_id: str, data_dir: Path) -> pd.DataFrame:
    """
    Parse the GEO SOFT file to extract per-sample phenotype features.
    Returns a DataFrame indexed by sample_id with columns from ALL_FEATURE_COLS.
    """
    soft_path = data_dir / f"{geo_id}_family.soft.gz"
    if not soft_path.exists():
        return pd.DataFrame(columns=ALL_FEATURE_COLS)

    try:
        import GEOparse
        gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)
    except Exception as e:
        logger.debug(f"Could not parse SOFT for {geo_id}: {e}")
        return pd.DataFrame(columns=ALL_FEATURE_COLS)

    rows = []
    for gsm_name, gsm in gse.gsms.items():
        chars = gsm.metadata.get("characteristics_ch1", [])
        # Build a key:value dict from the characteristics
        kv = {}
        for item in chars:
            s = str(item).strip()
            if ":" in s:
                k, v = s.split(":", 1)
                kv[k.strip().lower()] = v.strip()

        row = {"sample_id": gsm_name}

        # ── ER status ──
        for key in ["er_status_ihc", "er_status", "er",
                     "esr1_status", "estrogen receptor",
                     "er positive vs negative by immunohistochemistry",
                     "er positive vs negative",
                     "er positive vs negative by esr1 mrna gene expression (probe 205225_at)"]:
            if key in kv:
                val = kv[key]
                # Handle special text values
                vl = val.lower().strip()
                if vl in ("erpos", "er+", "er pos"):
                    row["ER_positive"] = 1.0
                elif vl in ("erneg", "er-", "er neg"):
                    row["ER_positive"] = 0.0
                else:
                    row["ER_positive"] = _to_binary(val)
                break
        # Also try er(%) -- numeric percentage
        if "ER_positive" not in row:
            for key in ["er(%)", "er.ct"]:
                if key in kv:
                    try:
                        pct = float(kv[key])
                        row["ER_positive"] = 1.0 if pct > 0 else 0.0
                    except (ValueError, TypeError):
                        pass
                    break
        # Also try erhswk0 (ER H-score at week 0)
        if "ER_positive" not in row and "erhswk0" in kv:
            try:
                hscore = float(kv["erhswk0"])
                row["ER_positive"] = 1.0 if hscore > 0 else 0.0
            except (ValueError, TypeError):
                pass

        # ── PR status ──
        for key in ["pr_status_ihc", "pr_status", "pr",
                     "progesterone receptor", "pgr status", "pr status"]:
            if key in kv:
                val = kv[key]
                vl = val.lower().strip()
                if vl in ("prpos", "pr+", "pr pos"):
                    row["PR_positive"] = 1.0
                elif vl in ("prneg", "pr-", "pr neg"):
                    row["PR_positive"] = 0.0
                else:
                    row["PR_positive"] = _to_binary(val)
                break
        if "PR_positive" not in row:
            for key in ["pr(%)", "prhswk0"]:
                if key in kv:
                    try:
                        pct = float(kv[key])
                        row["PR_positive"] = 1.0 if pct > 0 else 0.0
                    except (ValueError, TypeError):
                        pass
                    break

        # ── HER2 status ──
        for key in ["her2_status", "her2", "her 2 status", "erbb2_status",
                     "her2stat", "her2 status fish", "her2 status ihc",
                     "her2 ihc", "her2neu score"]:
            if key in kv:
                val = kv[key]
                vl = val.lower().strip()
                # Handle IHC scores: 0, 1+ = neg; 3+ = pos; 2+ = equivocal
                if vl in ("3+", "3", "ihc 3+", "ihc3+"):
                    row["HER2_positive"] = 1.0
                elif vl in ("0", "1+", "1", "ihc 0", "ihc 1+", "ihc0", "ihc1+"):
                    row["HER2_positive"] = 0.0
                elif vl in ("2+", "2", "ihc 2+", "ihc2+"):
                    row["HER2_positive"] = np.nan  # equivocal
                else:
                    row["HER2_positive"] = _to_binary(val)
                break

        # ── ERBB2 amplified (from FISH if available) ──
        for key in ["her2 fish", "her 2 fish", "erbb2_amplified"]:
            if key in kv:
                val = kv[key]
                row["ERBB2_amplified"] = _to_binary(val)
                break
        # If no FISH, use HER2 IHC as proxy for amplification
        if "ERBB2_amplified" not in row and "HER2_positive" in row:
            row["ERBB2_amplified"] = row["HER2_positive"]

        # ── Molecular subtype (can infer ER/HER2) ──
        for key in ["pam50_class", "tumor_subtype_(via_breastprs)",
                     "subtype", "molecular_category", "label"]:
            if key in kv:
                st = kv[key].lower()
                # Only fill if not already set
                if "ER_positive" not in row:
                    if any(s in st for s in ["luma", "lumb", "luminal", "lum"]):
                        row["ER_positive"] = 1.0
                    elif "basal" in st or "tn" in st or "tnbc" in st:
                        row["ER_positive"] = 0.0
                if "HER2_positive" not in row:
                    if "her2" in st and "hr" not in st:
                        row["HER2_positive"] = 1.0
                    elif "her2+" in st:
                        row["HER2_positive"] = 1.0
                break

        # ── TP53 status ──
        for key in ["tp53_mutation", "tp53_status", "tp53",
                     "p53 status", "p53 mutation"]:
            if key in kv:
                val = kv[key].lower().strip()
                if val in ("mut", "mutated", "mutant", "mutation"):
                    row["TP53_mutated"] = 1.0
                elif val in ("wt", "wild-type", "wild type", "wildtype",
                             "normal"):
                    row["TP53_mutated"] = 0.0
                else:
                    row["TP53_mutated"] = _to_binary(val)
                break

        for key in ["pik3ca_mutation", "pik3ca_status", "pik3ca"]:
            if key in kv:
                row["PIK3CA_mutated"] = _to_binary(kv[key])
                break

        for key in ["brca1_mutation", "brca1_status", "brca1",
                     "inflammatory.brca"]:
            if key in kv:
                row["BRCA1_mutated"] = _to_binary(kv[key])
                break

        for key in ["brca2_mutation", "brca2_status", "brca2"]:
            if key in kv:
                row["BRCA2_mutated"] = _to_binary(kv[key])
                break

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=ALL_FEATURE_COLS)

    df = pd.DataFrame(rows).set_index("sample_id")
    # Ensure all feature columns exist
    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df[ALL_FEATURE_COLS]


def _extract_from_labels_parquet(
    labels_df: pd.DataFrame,
    dataset_id: str,
) -> pd.DataFrame:
    """
    Extract clinical features from a response_labels.parquet that has
    char_ columns (I-SPY2, BrighTNess, durva_olap, etc.).
    """
    rows = []
    for idx, row_data in labels_df.iterrows():
        feat = {}

        # Determine sample_id from index or sample_id column
        if "sample_id" in labels_df.columns:
            sid = row_data["sample_id"]
        elif "geo_accession" in labels_df.columns:
            sid = row_data["geo_accession"]
        else:
            sid = str(idx)

        feat["sample_id"] = sid

        # ── I-SPY2 / durva_olap style: char_hr, char_her2, char_mp ──
        if "char_hr" in labels_df.columns:
            feat["ER_positive"] = _to_binary(row_data.get("char_hr"))
        if "char_her2" in labels_df.columns:
            feat["HER2_positive"] = _to_binary(row_data.get("char_her2"))
            feat["ERBB2_amplified"] = _to_binary(row_data.get("char_her2"))

        # ── Subtype column (hoogstraat, z1031, neoadj_letrozole) ──
        for subcol in ["char_subtype", "char_tumor_subtype_(via_breastprs)",
                       "char_molecular_category"]:
            if subcol in labels_df.columns:
                st = str(row_data.get(subcol, "")).lower()
                if "ER_positive" not in feat or pd.isna(feat.get("ER_positive")):
                    if any(s in st for s in ["lum", "luminal", "er+"]):
                        feat["ER_positive"] = 1.0
                    elif any(s in st for s in ["tn", "tnbc", "basal"]):
                        feat["ER_positive"] = 0.0
                        feat["HER2_positive"] = 0.0
                        feat["PR_positive"] = 0.0
                if "HER2_positive" not in feat or pd.isna(feat.get("HER2_positive")):
                    if "her2+" in st or ("her2" in st and "hr" not in st
                                         and "tn" not in st):
                        feat["HER2_positive"] = 1.0
                break

        # ── Herceptin binary (neoadj_letrozole) ──
        if "char_nac_herceptin_binary" in labels_df.columns:
            herceptin_val = str(row_data.get("char_nac_herceptin_binary", "")).lower()
            if herceptin_val == "yes":
                feat["HER2_positive"] = 1.0
                feat["ERBB2_amplified"] = 1.0
            elif herceptin_val == "no":
                # Don't override if already set
                if "HER2_positive" not in feat or pd.isna(feat.get("HER2_positive")):
                    feat["HER2_positive"] = 0.0

        # ── PD-L1 TNBC dataset (all TNBC -> ER-/HER2-) ──
        if dataset_id == "pdl1_tnbc":
            feat["ER_positive"] = 0.0
            feat["HER2_positive"] = 0.0
            feat["PR_positive"] = 0.0

        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=ALL_FEATURE_COLS)

    df = pd.DataFrame(rows).set_index("sample_id")
    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df[ALL_FEATURE_COLS]


# ======================================================================
# Pathway burden features
# ======================================================================

def _add_pathway_burdens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pathway mutation burden features from individual gene mutations.
    n_mutations_pi3k = PIK3CA + PTEN + AKT1 (AKT1 not tracked, so PIK3CA + PTEN)
    n_mutations_ddr  = BRCA1 + BRCA2
    """
    pi3k_genes = ["PIK3CA_mutated", "PTEN_mutated"]
    ddr_genes = ["BRCA1_mutated", "BRCA2_mutated"]

    # Only compute burden if at least one gene in the pathway is non-NaN
    pi3k_vals = df[pi3k_genes]
    ddr_vals = df[ddr_genes]

    df["n_mutations_pi3k"] = pi3k_vals.sum(axis=1, min_count=1)
    df["n_mutations_ddr"] = ddr_vals.sum(axis=1, min_count=1)

    return df


# ======================================================================
# Master builder
# ======================================================================

def build_all_mutation_features(
    treatment_splits_path: Path = DATA_METADATA / "treatment_class_splits.tsv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build mutation/CNV/biomarker features for every patient in every dataset.

    Returns:
        features_df: DataFrame indexed by (dataset_id, sample_id) with
                     columns from ALL_FEATURE_COLS.
        availability_df: Per-dataset summary of what data is available.
    """
    splits = pd.read_csv(treatment_splits_path, sep="\t")
    all_features = []
    availability_rows = []

    for _, row in splits.iterrows():
        ds_id = row["dataset_id"]
        geo = row["geo_accession"]
        source = row["source"]

        logger.info(f"Extracting mutation features for {ds_id} ...")

        feat_df = pd.DataFrame(columns=ALL_FEATURE_COLS)

        # ── CTR-DB datasets: use GEO SOFT files ──
        if source == "ctrdb":
            soft_dir = DATA_RAW / "ctrdb" / geo
            feat_df = _extract_from_geo_soft(geo, soft_dir)

        # ── I-SPY2: use response_labels char_ columns ──
        elif source == "ispy2":
            labels = pd.read_parquet(DATA_RAW / "ispy2" / "response_labels.parquet")
            # Filter to the arm
            arm_name = ds_id.replace("ispy2_", "").replace("_", " ").replace("plus", "+")
            # Reconstruct arm name from char_arm
            if "char_arm" in labels.columns:
                arm_mask = labels["char_arm"].apply(
                    lambda x: x.replace(" ", "_").replace("+", "plus")
                ) == ds_id.replace("ispy2_", "")
                labels = labels[arm_mask]
            feat_df = _extract_from_labels_parquet(labels, ds_id)
            # I-SPY2 uses geo_accession as sample ID
            if "geo_accession" in pd.read_parquet(
                DATA_RAW / "ispy2" / "response_labels.parquet"
            ).columns:
                # Re-index to match expression index which is positional
                feat_df = feat_df.reset_index(drop=True)

        # ── BrighTNess ──
        elif source == "brightness":
            labels = pd.read_parquet(DATA_RAW / "brightness" / "response_labels.parquet")
            feat_df = _extract_from_labels_parquet(labels, ds_id)
            # BrighTNess is TNBC -> all ER-/HER2-/PR-
            feat_df["ER_positive"] = 0.0
            feat_df["HER2_positive"] = 0.0
            feat_df["PR_positive"] = 0.0
            feat_df["ERBB2_amplified"] = 0.0
            feat_df = feat_df.reset_index(drop=True)

        # ── Durva + Olap ──
        elif source == "durva_olap":
            labels = pd.read_parquet(
                DATA_RAW / "durva_olap_breast" / "response_labels.parquet"
            )
            feat_df = _extract_from_labels_parquet(labels, ds_id)

        # ── Hoogstraat 2 ──
        elif source == "hoogstraat_2":
            labels = pd.read_parquet(
                DATA_RAW / "hoogstraat_2" / "response_labels.parquet"
            )
            feat_df = _extract_from_labels_parquet(labels, ds_id)
            feat_df = feat_df.reset_index(drop=True)

        # Add pathway burdens
        if len(feat_df) > 0:
            feat_df = _add_pathway_burdens(feat_df)

        # Record dataset_id in the index
        feat_df["dataset_id"] = ds_id
        all_features.append(feat_df)

        # Availability summary
        avail = {
            "dataset_id": ds_id,
            "geo_accession": geo,
            "source": source,
            "n_patients": len(feat_df),
            "has_er_status": feat_df["ER_positive"].notna().any() if len(feat_df) > 0 else False,
            "has_her2_status": feat_df["HER2_positive"].notna().any() if len(feat_df) > 0 else False,
            "has_pr_status": feat_df["PR_positive"].notna().any() if len(feat_df) > 0 else False,
            "has_tp53": feat_df["TP53_mutated"].notna().any() if len(feat_df) > 0 else False,
            "has_pik3ca": feat_df["PIK3CA_mutated"].notna().any() if len(feat_df) > 0 else False,
            "has_brca": (
                feat_df["BRCA1_mutated"].notna().any() or
                feat_df["BRCA2_mutated"].notna().any()
            ) if len(feat_df) > 0 else False,
            "has_erbb2_amplified": feat_df["ERBB2_amplified"].notna().any() if len(feat_df) > 0 else False,
            "er_frac": feat_df["ER_positive"].notna().mean() if len(feat_df) > 0 else 0,
            "her2_frac": feat_df["HER2_positive"].notna().mean() if len(feat_df) > 0 else 0,
        }
        availability_rows.append(avail)

        n_non_null = feat_df[ALL_FEATURE_COLS].notna().any(axis=1).sum()
        logger.info(f"  {ds_id}: {n_non_null}/{len(feat_df)} patients have any feature")

    # Combine all features
    if all_features:
        combined = pd.concat(all_features, axis=0, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=ALL_FEATURE_COLS + ["dataset_id"])

    availability = pd.DataFrame(availability_rows)
    return combined, availability


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 70)
    logger.info("Building mutation / CNV / biomarker features")
    logger.info("=" * 70)

    features, availability = build_all_mutation_features()

    # Save
    features.to_parquet(DATA_CACHE / "mutation_features.parquet", index=False)
    availability.to_csv(DATA_METADATA / "multimodal_availability.tsv", sep="\t", index=False)

    logger.info(f"\nFeatures shape: {features.shape}")
    logger.info(f"Saved features to {DATA_CACHE / 'mutation_features.parquet'}")
    logger.info(f"Saved availability to {DATA_METADATA / 'multimodal_availability.tsv'}")

    # Print availability summary
    logger.info("\n=== Multimodal Availability ===")
    logger.info(availability.to_string(index=False))

    # Feature coverage
    logger.info("\n=== Feature Coverage ===")
    for col in ALL_FEATURE_COLS:
        n_total = len(features)
        n_avail = features[col].notna().sum()
        logger.info(f"  {col}: {n_avail}/{n_total} ({100*n_avail/n_total:.1f}%)")
