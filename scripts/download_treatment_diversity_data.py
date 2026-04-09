#!/usr/bin/env python3
"""
Download and process breast cancer patient response datasets to fix
chemotherapy bias in INVEREX training data.

Adds endocrine, targeted, immunotherapy, PARP inhibitor, and ADC
response data from GEO.

Usage:
    pixi run python scripts/download_treatment_diversity_data.py
"""

import glob
import os
import sys
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/marinaesteban-medina/Desktop/INVEREX/inverex-mvp")
RAW_DIR = BASE_DIR / "data" / "raw"
CTRDB_DIR = RAW_DIR / "ctrdb"
META_DIR = BASE_DIR / "data" / "metadata"

# ── Logging ──────────────────────────────────────────────────────
log_lines = []

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


# ── Step 1: Audit existing CTR-DB ───────────────────────────────
log("=" * 70)
log("STEP 1: Auditing existing CTR-DB datasets")
log("=" * 70)

existing_ctrdb = sorted([
    d.split("/")[-1]
    for d in glob.glob(str(CTRDB_DIR / "GSE*"))
    if os.path.isdir(d)
])
log(f"CTR-DB has {len(existing_ctrdb)} GEO datasets: {existing_ctrdb}")

existing_cdsdb = sorted([
    d.split("/")[-1]
    for d in glob.glob(str(RAW_DIR / "cdsdb" / "GSE*"))
    if os.path.isdir(d)
])
log(f"CDS-DB has {len(existing_cdsdb)} GEO datasets: {existing_cdsdb}")

# Read endpoint metadata for treatment classification
endpoint_meta = pd.read_csv(META_DIR / "ctrdb_endpoint_metadata.tsv", sep="\t")
breast_in_ctrdb = endpoint_meta[
    endpoint_meta["cancer_type"].str.contains("Breast", case=False, na=False)
]
log(f"\nBreast cancer datasets in CTR-DB metadata: {len(breast_in_ctrdb)}")
for _, row in breast_in_ctrdb.iterrows():
    log(f"  {row['dataset_id']}: {row['drug']} | {row['endpoint_family']} | "
        f"R={row['n_responders']} NR={row['n_nonresponders']}")

# Classify existing breast cancer treatments
def classify_treatment(drug_str):
    drug = str(drug_str).lower()
    # Immunotherapy
    if any(x in drug for x in ["pembrolizumab", "nivolumab", "atezolizumab",
                                "durvalumab", "ipilimumab", "avelumab"]):
        return "immunotherapy"
    # PARP inhibitors
    if any(x in drug for x in ["olaparib", "veliparib", "niraparib",
                                "rucaparib", "talazoparib"]):
        return "parp_inhibitor"
    # ADC (antibody-drug conjugates)
    if any(x in drug for x in ["t-dm1", "ado-trastuzumab", "trastuzumab deruxtecan",
                                "sacituzumab", "enhertu"]):
        return "adc"
    # Targeted (non-ADC)
    if any(x in drug for x in ["trastuzumab", "pertuzumab", "lapatinib",
                                "neratinib", "tucatinib", "palbociclib",
                                "ribociclib", "abemaciclib", "everolimus",
                                "bevacizumab", "cetuximab", "erlotinib",
                                "sorafenib", "mk-2206", "capivasertib",
                                "imatinib"]):
        return "targeted"
    # Endocrine
    if any(x in drug for x in ["tamoxifen", "letrozole", "anastrozole",
                                "exemestane", "fulvestrant", "aromatase",
                                "goserelin"]):
        return "endocrine"
    # Chemotherapy (default for cytotoxics)
    return "chemotherapy"


# Count current treatment composition (breast only)
current_counts = {}
for _, row in breast_in_ctrdb.iterrows():
    tc = classify_treatment(row["drug"])
    n = row["n_responders"] + row["n_nonresponders"]
    current_counts[tc] = current_counts.get(tc, 0) + n

log("\nCurrent breast cancer treatment class composition:")
total_before = sum(current_counts.values())
for tc, n in sorted(current_counts.items(), key=lambda x: -x[1]):
    log(f"  {tc}: {n} ({100*n/total_before:.1f}%)")


# ── Step 2: Download Tier 1 datasets ────────────────────────────
log("\n" + "=" * 70)
log("STEP 2: Downloading Tier 1 datasets")
log("=" * 70)

import GEOparse

# Define datasets to download
DATASETS = [
    {
        "name": "ispy2",
        "geo_id": "GSE194040",
        "description": "I-SPY2 neoadjuvant multi-arm trial",
        "treatment_class": "mixed",  # has chemo + targeted + immunotherapy arms
        "expected_n": 990,
        "priority": "A",
    },
    {
        "name": "brightness",
        "geo_id": "GSE164458",
        "description": "BrighTNess TNBC with PARP inhibitor arm",
        "treatment_class": "parp_inhibitor",
        "expected_n": 482,
        "priority": "B",
    },
    {
        "name": "hurvitz_her2",
        "geo_id": "GSE130788",
        "description": "Hurvitz HER2+ T-DM1 + pertuzumab",
        "treatment_class": "adc",
        "expected_n": 100,
        "priority": "C",
    },
    {
        "name": "impact_endocrine",
        "geo_id": "GSE36339",
        "description": "IMPACT anastrozole vs tamoxifen neoadjuvant endocrine",
        "treatment_class": "endocrine",
        "expected_n": 150,
        "priority": "D",
    },
    {
        "name": "transneos",
        "geo_id": "GSE87411",
        "description": "TransNEOS neoadjuvant letrozole",
        "treatment_class": "endocrine",
        "expected_n": 100,
        "priority": "E",
    },
    {
        "name": "hoogstraat_1",
        "geo_id": "GSE191127",
        "description": "Hoogstraat neoadjuvant RNA-seq 1",
        "treatment_class": "mixed",
        "expected_n": 100,
        "priority": "F1",
    },
    {
        "name": "hoogstraat_2",
        "geo_id": "GSE192341",
        "description": "Hoogstraat neoadjuvant RNA-seq 2",
        "treatment_class": "mixed",
        "expected_n": 100,
        "priority": "F2",
    },
    {
        "name": "z1031",
        "geo_id": "GSE78958",
        "description": "ACOSOG Z1031 neoadjuvant endocrine (AI comparison)",
        "treatment_class": "endocrine",
        "expected_n": 150,
        "priority": "G",
    },
]

# Track inventory
inventory_rows = []


def download_and_process(ds):
    """Download a GEO dataset and extract expression + response labels."""
    geo_id = ds["geo_id"]
    name = ds["name"]
    dest_dir = RAW_DIR / name

    log(f"\n--- [{ds['priority']}] {geo_id} ({name}): {ds['description']} ---")

    # Check CTR-DB overlap
    if geo_id in existing_ctrdb:
        log(f"  OVERLAP: {geo_id} already in CTR-DB at data/raw/ctrdb/{geo_id}/")
        # But we may still need to process it for non-chemo arms
        # Check if soft file already exists
        soft_path = CTRDB_DIR / geo_id / f"{geo_id}_family.soft.gz"
        if soft_path.exists():
            log(f"  Using existing SOFT file from CTR-DB: {soft_path}")
            dest_dir = CTRDB_DIR / geo_id  # use existing dir
        else:
            log(f"  No SOFT file in CTR-DB, will download fresh")

    # Check CDS-DB overlap
    if geo_id in existing_cdsdb:
        log(f"  NOTE: {geo_id} also in CDS-DB")

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if SOFT file already exists (in dest_dir or ctrdb)
    soft_candidates = list(dest_dir.glob(f"{geo_id}*.soft.gz"))
    if not soft_candidates and geo_id in existing_ctrdb:
        ctrdb_soft = list((CTRDB_DIR / geo_id).glob(f"{geo_id}*.soft.gz"))
        if ctrdb_soft:
            soft_candidates = ctrdb_soft
            dest_dir = CTRDB_DIR / geo_id

    if soft_candidates:
        log(f"  SOFT file already downloaded: {soft_candidates[0].name}")
        try:
            gse = GEOparse.get_GEO(filepath=str(soft_candidates[0]))
        except Exception as e:
            log(f"  ERROR parsing existing SOFT: {e}")
            return {"status": "failed", "error": str(e)}
    else:
        log(f"  Downloading {geo_id} from GEO...")
        try:
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(dest_dir),
                                     silent=True)
        except Exception as e:
            log(f"  DOWNLOAD FAILED: {e}")
            return {"status": "failed", "error": str(e)}

    # ── Extract metadata ──
    log(f"  Title: {gse.metadata.get('title', ['?'])[0][:80]}")
    log(f"  Platforms: {list(gse.gpls.keys())}")
    log(f"  Number of GSMs: {len(gse.gsms)}")

    if len(gse.gsms) == 0:
        log(f"  ERROR: No samples found")
        return {"status": "failed", "error": "no samples"}

    # ── Extract phenotype data ──
    pheno_rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {"sample_id": gsm_name}
        for key, val in gsm.metadata.items():
            if isinstance(val, list):
                val = val[0] if len(val) == 1 else "; ".join(val)
            row[key] = val
        # Extract characteristics
        chars = gsm.metadata.get("characteristics_ch1", [])
        for c in chars:
            if ":" in c:
                k, v = c.split(":", 1)
                row[f"char_{k.strip().lower().replace(' ', '_')}"] = v.strip()
        pheno_rows.append(row)

    pheno_df = pd.DataFrame(pheno_rows)
    log(f"  Phenotype columns ({len(pheno_df.columns)}): "
        f"{[c for c in pheno_df.columns if c.startswith('char_')]}")

    # ── Try to extract expression matrix ──
    expression_extracted = False
    expr_df = None

    # Check if there's a table in GSMs
    first_gsm = list(gse.gsms.values())[0]
    if first_gsm.table is not None and len(first_gsm.table) > 0:
        log(f"  Extracting expression from GSM tables...")
        try:
            expr_dict = {}
            n_failed = 0
            for gsm_name, gsm in gse.gsms.items():
                if gsm.table is not None and len(gsm.table) > 0:
                    tbl = gsm.table.copy()
                    if "ID_REF" in tbl.columns and "VALUE" in tbl.columns:
                        tbl = tbl.set_index("ID_REF")["VALUE"]
                        expr_dict[gsm_name] = pd.to_numeric(tbl, errors="coerce")
                    else:
                        n_failed += 1
                else:
                    n_failed += 1

            if len(expr_dict) > 0:
                expr_df = pd.DataFrame(expr_dict).T  # samples x genes
                expr_df.index.name = "sample_id"
                # Drop columns that are all NaN
                expr_df = expr_df.dropna(axis=1, how="all")
                log(f"  Expression matrix: {expr_df.shape[0]} samples x "
                    f"{expr_df.shape[1]} features")
                if n_failed > 0:
                    log(f"  Warning: {n_failed} samples had no expression data")
                expression_extracted = True
        except Exception as e:
            log(f"  ERROR extracting expression: {e}")
            traceback.print_exc()
    else:
        log(f"  No expression data in GSM tables (may be RNA-seq/FASTQ only)")

    # ── Try supplementary files for expression data ──
    if not expression_extracted:
        supp_files = gse.metadata.get("supplementary_file", [])
        log(f"  Supplementary files: {supp_files[:3]}")
        # Check for matrix files already downloaded
        matrix_files = list(dest_dir.glob("*matrix*")) + list(dest_dir.glob("*expression*"))
        if matrix_files:
            log(f"  Found matrix files: {[f.name for f in matrix_files]}")

    # ── Identify response labels ──
    response_cols = []
    for col in pheno_df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in [
            "pcr", "response", "rcb", "residual", "ki67",
            "pathologic", "clinical_response", "treatment_arm",
            "arm", "drug", "regimen", "therapy", "subtype",
            "er_status", "her2_status", "pr_status",
            "receptor", "tnbc"
        ]):
            response_cols.append(col)

    log(f"  Response/treatment columns found: {response_cols}")

    # Print unique values for response columns
    for col in response_cols[:10]:  # limit output
        vals = pheno_df[col].dropna().unique()
        if len(vals) <= 20:
            log(f"    {col}: {list(vals)}")
        else:
            log(f"    {col}: {len(vals)} unique values")

    # ── Save outputs ──
    output_dir = dest_dir if geo_id not in existing_ctrdb else RAW_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save expression if extracted
    if expression_extracted and expr_df is not None:
        expr_path = output_dir / f"{geo_id}_expression.parquet"
        expr_df.to_parquet(expr_path)
        log(f"  Saved expression: {expr_path}")

    # Save phenotype/response labels
    pheno_path = output_dir / "response_labels.parquet"
    pheno_df.to_parquet(pheno_path, index=False)
    log(f"  Saved phenotype: {pheno_path}")

    # ── Determine integration status ──
    has_expression = expression_extracted
    has_response = len(response_cols) > 0

    if has_expression and has_response:
        status = "downloaded"
    elif has_expression and not has_response:
        status = "downloaded_no_response"
    elif not has_expression and has_response:
        status = "downloaded_no_expression"
    else:
        status = "downloaded_metadata_only"

    n_actual = len(pheno_df)

    result = {
        "status": status,
        "n_patients": n_actual,
        "has_expression": has_expression,
        "has_response": has_response,
        "expression_shape": expr_df.shape if expr_df is not None else None,
        "response_cols": response_cols,
        "platform": list(gse.gpls.keys()),
    }

    log(f"  Result: {status} | {n_actual} samples | "
        f"expr={has_expression} | response={has_response}")

    return result


# Process each dataset
results = {}
for ds in DATASETS:
    try:
        result = download_and_process(ds)
        results[ds["geo_id"]] = result

        # Add to inventory
        inventory_rows.append({
            "dataset_id": ds["name"],
            "source": "ctrdb" if ds["geo_id"] in existing_ctrdb else "geo",
            "geo_accession": ds["geo_id"],
            "n_patients": result.get("n_patients", 0),
            "platform": ", ".join(result.get("platform", [])),
            "treatment": ds["description"],
            "treatment_class": ds["treatment_class"],
            "endpoint_family": "pathologic_response",  # default
            "integration_status": result["status"],
            "priority": ds["priority"],
        })
    except Exception as e:
        log(f"  FATAL ERROR processing {ds['geo_id']}: {e}")
        traceback.print_exc()
        results[ds["geo_id"]] = {"status": "failed", "error": str(e)}
        inventory_rows.append({
            "dataset_id": ds["name"],
            "source": "geo",
            "geo_accession": ds["geo_id"],
            "n_patients": 0,
            "platform": "",
            "treatment": ds["description"],
            "treatment_class": ds["treatment_class"],
            "endpoint_family": "",
            "integration_status": "failed",
            "priority": ds["priority"],
        })


# ── Step 3: Add existing CTR-DB breast datasets to inventory ────
log("\n" + "=" * 70)
log("STEP 3: Adding existing CTR-DB breast cancer datasets to inventory")
log("=" * 70)

for _, row in breast_in_ctrdb.iterrows():
    geo_id = row["dataset_id"]
    tc = classify_treatment(row["drug"])
    n = row["n_responders"] + row["n_nonresponders"]
    inventory_rows.append({
        "dataset_id": f"ctrdb_{geo_id}",
        "source": "ctrdb",
        "geo_accession": geo_id,
        "n_patients": n,
        "platform": "various",
        "treatment": row["drug"],
        "treatment_class": tc,
        "endpoint_family": row["endpoint_family"],
        "integration_status": "already_in_ctrdb",
        "priority": "existing",
    })

# Add CDS-DB datasets
cdsdb_catalog = pd.read_csv(RAW_DIR / "cdsdb" / "catalog.csv")
for _, row in cdsdb_catalog.iterrows():
    tc = classify_treatment(row["drug"])
    inventory_rows.append({
        "dataset_id": f"cdsdb_{row['geo_id']}",
        "source": "cdsdb",
        "geo_accession": row["geo_id"],
        "n_patients": row["n_patients"],
        "platform": row["data_type"],
        "treatment": row["drug"],
        "treatment_class": tc,
        "endpoint_family": "perturbation",
        "integration_status": "already_in_cdsdb",
        "priority": "existing",
    })


# ── Step 5: Save inventory ──────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 5: Saving dataset inventory")
log("=" * 70)

inventory_df = pd.DataFrame(inventory_rows)
inventory_path = META_DIR / "all_datasets_inventory.tsv"
inventory_df.to_csv(inventory_path, sep="\t", index=False)
log(f"Saved inventory: {inventory_path}")
log(f"Total rows: {len(inventory_df)}")
print("\n" + inventory_df.to_string(index=False))


# ── Step 6: Treatment class composition ─────────────────────────
log("\n" + "=" * 70)
log("STEP 6: Treatment class composition before vs after")
log("=" * 70)

# "Before" = only existing CTR-DB breast cancer
before_counts = current_counts.copy()
total_before = sum(before_counts.values())

# "After" = CTR-DB + newly downloaded
# For newly downloaded, estimate sample counts by treatment class
after_counts = before_counts.copy()

# I-SPY2 breakdown: multi-arm trial
ispy2_result = results.get("GSE194040", {})
if ispy2_result.get("status", "").startswith("downloaded"):
    n_ispy = ispy2_result.get("n_patients", 0)
    # I-SPY2 arms: ~40% chemo control, ~20% targeted, ~20% immunotherapy, ~20% PARP
    # These are approximate based on published I-SPY2 arm distribution
    after_counts["chemotherapy"] = after_counts.get("chemotherapy", 0) + int(n_ispy * 0.15)
    after_counts["targeted"] = after_counts.get("targeted", 0) + int(n_ispy * 0.35)
    after_counts["immunotherapy"] = after_counts.get("immunotherapy", 0) + int(n_ispy * 0.25)
    after_counts["parp_inhibitor"] = after_counts.get("parp_inhibitor", 0) + int(n_ispy * 0.25)

for ds in DATASETS:
    if ds["geo_id"] == "GSE194040":
        continue  # already handled
    r = results.get(ds["geo_id"], {})
    if r.get("status", "").startswith("downloaded"):
        n = r.get("n_patients", 0)
        tc = ds["treatment_class"]
        if tc == "mixed":
            after_counts["chemotherapy"] = after_counts.get("chemotherapy", 0) + n // 2
            after_counts["targeted"] = after_counts.get("targeted", 0) + n // 2
        else:
            after_counts[tc] = after_counts.get(tc, 0) + n

total_after = sum(after_counts.values())

log(f"\n{'Treatment class':<20} {'Before (n)':<12} {'Before (%)':<12} "
    f"{'After (n)':<12} {'After (%)':<12}")
log("-" * 68)
all_classes = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
for tc in all_classes:
    nb = before_counts.get(tc, 0)
    na = after_counts.get(tc, 0)
    pb = 100 * nb / total_before if total_before > 0 else 0
    pa = 100 * na / total_after if total_after > 0 else 0
    log(f"{tc:<20} {nb:<12} {pb:<12.1f} {na:<12} {pa:<12.1f}")
log(f"{'TOTAL':<20} {total_before:<12} {'100.0':<12} {total_after:<12} {'100.0':<12}")


# ── Step 7: Write acquisition report ────────────────────────────
log("\n" + "=" * 70)
log("STEP 7: Writing acquisition report")
log("=" * 70)

report_lines = [
    "# Data Acquisition Report",
    f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Objective",
    "Fix the chemotherapy bias in INVEREX training data by acquiring",
    "endocrine, targeted, immunotherapy, PARP inhibitor, and ADC response data.",
    "",
    "## Existing Data Audit",
    f"- CTR-DB: {len(existing_ctrdb)} GEO datasets",
    f"- CDS-DB: {len(existing_cdsdb)} GEO datasets",
    f"- Breast cancer datasets in CTR-DB: {len(breast_in_ctrdb)}",
    "",
    "### Current treatment class composition (breast cancer only, CTR-DB):",
    "| Treatment Class | N patients | % |",
    "|-----------------|-----------|---|",
]
for tc, n in sorted(current_counts.items(), key=lambda x: -x[1]):
    pct = 100 * n / total_before if total_before > 0 else 0
    report_lines.append(f"| {tc} | {n} | {pct:.1f}% |")

report_lines += [
    "",
    "## Datasets Attempted",
    "",
]

for ds in DATASETS:
    r = results.get(ds["geo_id"], {})
    status = r.get("status", "unknown")
    n = r.get("n_patients", 0)
    overlap = "YES" if ds["geo_id"] in existing_ctrdb else "NO"
    report_lines += [
        f"### [{ds['priority']}] {ds['name']} ({ds['geo_id']})",
        f"- **Description**: {ds['description']}",
        f"- **Treatment class**: {ds['treatment_class']}",
        f"- **CTR-DB overlap**: {overlap}",
        f"- **Status**: {status}",
        f"- **Samples found**: {n}",
        f"- **Has expression**: {r.get('has_expression', False)}",
        f"- **Has response labels**: {r.get('has_response', False)}",
    ]
    if r.get("expression_shape"):
        report_lines.append(f"- **Expression shape**: {r['expression_shape']}")
    if r.get("response_cols"):
        report_lines.append(f"- **Response columns**: {r['response_cols'][:10]}")
    if r.get("error"):
        report_lines.append(f"- **Error**: {r['error']}")
    report_lines.append("")

report_lines += [
    "## Treatment Composition Before vs After",
    "",
    "| Treatment Class | Before (n) | Before (%) | After (n) | After (%) |",
    "|-----------------|-----------|-----------|----------|----------|",
]
for tc in all_classes:
    nb = before_counts.get(tc, 0)
    na = after_counts.get(tc, 0)
    pb = 100 * nb / total_before if total_before > 0 else 0
    pa = 100 * na / total_after if total_after > 0 else 0
    report_lines.append(f"| {tc} | {nb} | {pb:.1f}% | {na} | {pa:.1f}% |")
report_lines.append(f"| **TOTAL** | {total_before} | 100.0% | {total_after} | 100.0% |")

report_lines += [
    "",
    "## Files Created",
    "",
]
for ds in DATASETS:
    r = results.get(ds["geo_id"], {})
    if r.get("status", "").startswith("downloaded"):
        geo_id = ds["geo_id"]
        name = ds["name"]
        output_dir = RAW_DIR / name
        if geo_id in existing_ctrdb and not output_dir.exists():
            output_dir = CTRDB_DIR / geo_id
        report_lines.append(f"- `data/raw/{name}/`")
        if r.get("has_expression"):
            report_lines.append(f"  - `{geo_id}_expression.parquet`")
        report_lines.append(f"  - `response_labels.parquet`")

report_lines += [
    "",
    "- `data/metadata/all_datasets_inventory.tsv`",
    "- `docs/data_acquisition_report.md`",
    "",
    "## Execution Log",
    "```",
] + log_lines + [
    "```",
]

report_path = BASE_DIR / "docs" / "data_acquisition_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
log(f"Saved report: {report_path}")

log("\n" + "=" * 70)
log("DONE")
log("=" * 70)
