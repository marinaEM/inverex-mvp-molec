"""
CTR-DB 2.0 + GEO data ingestion for patient drug-response validation.

CTR-DB 2.0 (http://ctrdb2api.cloudna.cn) provides curated clinical trial
transcriptomics datasets with response labels.  The API exposes metadata
(dataset IDs, drug, sample sizes, responder counts, GEO accessions) while
the actual expression matrices live on GEO.

Workflow
--------
1. Query CTR-DB ``searchCancerApi`` for breast-cancer datasets.
2. Parse metadata via ``searchDatasetApi`` / ``singleDatasetApi``.
3. Download expression from GEO with GEOparse.
4. Align response labels to expression samples.
5. Filter to L1000 landmark genes, transpose to samples-x-genes.

If the CTR-DB API is unreachable, a hardcoded fallback catalog of 10 key
breast-cancer GEO datasets is used instead.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import DATA_CACHE, DATA_RAW, RESULTS

logger = logging.getLogger(__name__)

# ── CTR-DB API ────────────────────────────────────────────────────────
CTRDB_API_BASE = "http://ctrdb2api.cloudna.cn"

# ── Hardcoded fallback catalog (curated breast-cancer GEO datasets) ──
FALLBACK_BREAST_DATASETS = [
    {
        "dataset_id": "FALLBACK_GSE25066",
        "drug": "Taxane + Anthracycline (neoadjuvant)",
        "sample_size": 508,
        "n_responders": 99,
        "n_nonresponders": 409,
        "geo_source": "GSE25066",
        "platform": "GPL96",
        "response_col": "pCR",
        "response_positive": ["pCR"],
        "response_negative": ["RD"],
    },
    {
        "dataset_id": "FALLBACK_GSE20194",
        "drug": "Neoadjuvant TFAC",
        "sample_size": 278,
        "n_responders": 56,
        "n_nonresponders": 222,
        "geo_source": "GSE20194",
        "platform": "GPL96",
        "response_col": "characteristics_ch1",
        "response_positive": ["pCR"],
        "response_negative": ["RD"],
    },
    {
        "dataset_id": "FALLBACK_GSE22093",
        "drug": "Neoadjuvant taxane/anthracycline",
        "sample_size": 103,
        "n_responders": 27,
        "n_nonresponders": 76,
        "geo_source": "GSE22093",
        "platform": "GPL96",
        "response_col": "characteristics_ch1",
        "response_positive": ["pCR"],
        "response_negative": ["RD"],
    },
    {
        "dataset_id": "FALLBACK_GSE20271",
        "drug": "Neoadjuvant T/FAC",
        "sample_size": 178,
        "n_responders": 36,
        "n_nonresponders": 142,
        "geo_source": "GSE20271",
        "platform": "GPL96",
        "response_col": "characteristics_ch1",
        "response_positive": ["pCR"],
        "response_negative": ["RD"],
    },
    {
        "dataset_id": "FALLBACK_GSE41998",
        "drug": "Neoadjuvant +/- ixabepilone",
        "sample_size": 279,
        "n_responders": 55,
        "n_nonresponders": 224,
        "geo_source": "GSE41998",
        "platform": "GPL570",
        "response_col": "characteristics_ch1",
        "response_positive": ["pCR"],
        "response_negative": ["RD"],
    },
    {
        "dataset_id": "FALLBACK_GSE9893",
        "drug": "Tamoxifen adjuvant",
        "sample_size": 155,
        "n_responders": 78,
        "n_nonresponders": 77,
        "geo_source": "GSE9893",
        "platform": "GPL5345",
        "response_col": "characteristics_ch1",
        "response_positive": ["no relapse"],
        "response_negative": ["relapse"],
    },
    {
        "dataset_id": "FALLBACK_GSE6861",
        "drug": "Docetaxel neoadjuvant",
        "sample_size": 161,
        "n_responders": 30,
        "n_nonresponders": 131,
        "geo_source": "GSE6861",
        "platform": "GPL570",
        "response_col": "characteristics_ch1",
        "response_positive": ["pCR"],
        "response_negative": ["RD", "no pCR"],
    },
    {
        "dataset_id": "FALLBACK_GSE76360",
        "drug": "Trastuzumab + chemo",
        "sample_size": 111,
        "n_responders": 50,
        "n_nonresponders": 61,
        "geo_source": "GSE76360",
        "platform": "GPL570",
        "response_col": "characteristics_ch1",
        "response_positive": ["pCR"],
        "response_negative": ["non-pCR", "RD"],
    },
    {
        "dataset_id": "FALLBACK_GSE50948",
        "drug": "Letrozole neoadjuvant",
        "sample_size": 150,
        "n_responders": 50,
        "n_nonresponders": 100,
        "geo_source": "GSE50948",
        "platform": "GPL570",
        "response_col": "characteristics_ch1",
        "response_positive": ["responder", "high response"],
        "response_negative": ["non-responder", "low response"],
    },
    {
        "dataset_id": "FALLBACK_GSE17705",
        "drug": "Tamoxifen adjuvant",
        "sample_size": 298,
        "n_responders": 209,
        "n_nonresponders": 89,
        "geo_source": "GSE17705",
        "platform": "GPL96",
        "response_col": "characteristics_ch1",
        "response_positive": ["no recurrence"],
        "response_negative": ["recurrence"],
    },
]


# ── Public helpers ────────────────────────────────────────────────────
def fetch_breast_ctrdb_catalog(
    timeout: int = 20,
    min_sample_size: int = 20,
) -> pd.DataFrame:
    """
    Query CTR-DB 2.0 for all breast-cancer datasets.

    The API works in two steps:
      1. ``searchCancerApi`` returns cancer-type records, each containing
         a list of CTR-DB dataset ID strings (e.g. "CTR_Microarray_48-I").
      2. ``searchDatasetApi`` returns per-dataset metadata (drug, sample
         size, responder counts, GEO accession, platform, grouping).

    Returns a DataFrame with columns:
        dataset_id, drug, sample_size, n_responders, n_nonresponders,
        geo_source, platform, response_grouping
    """
    logger.info("Fetching breast-cancer catalog from CTR-DB API ...")
    try:
        # Step 1: get all breast-cancer dataset IDs
        resp = requests.post(
            f"{CTRDB_API_BASE}/searchCancerApi",
            json={"search": "Breast cancer"},
            timeout=timeout,
        )
        resp.raise_for_status()
        cancer_data = resp.json()

        # Extract dataset IDs from the response
        dataset_ids = _collect_dataset_ids(cancer_data)
        if len(dataset_ids) == 0:
            logger.warning("CTR-DB returned 0 dataset IDs; using fallback catalog.")
            return _fallback_catalog()

        logger.info(f"CTR-DB returned {len(dataset_ids)} dataset IDs")

        # Step 2: fetch metadata for each dataset
        records = _fetch_all_dataset_metadata(dataset_ids, timeout)
        if len(records) == 0:
            logger.warning("Could not fetch any dataset metadata; using fallback.")
            return _fallback_catalog()

        catalog = pd.DataFrame(records)

        # Filter to usable datasets (have GEO source and sufficient samples)
        catalog = catalog[
            catalog["geo_source"].str.startswith("GSE")
            & (catalog["sample_size"] >= min_sample_size)
        ].reset_index(drop=True)

        logger.info(
            f"CTR-DB catalog: {len(catalog)} usable breast-cancer datasets "
            f"(>= {min_sample_size} samples, have GEO source)"
        )
        return catalog

    except Exception as exc:
        logger.warning(f"CTR-DB API failed ({exc}); using fallback catalog.")
        return _fallback_catalog()


def download_geo_expression(
    geo_id: str,
    dest_dir: Path,
    timeout: int = 300,
) -> Optional[pd.DataFrame]:
    """
    Download an expression matrix from GEO using GEOparse.

    Returns a samples-x-genes DataFrame (rows = samples, columns = gene symbols),
    or None on failure.
    """
    import GEOparse

    cache_path = dest_dir / f"{geo_id}_expression.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached expression for {geo_id}")
        return pd.read_parquet(cache_path)

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {geo_id} from GEO ...")

    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(dest_dir), silent=True)
    except Exception as exc:
        logger.error(f"GEOparse failed for {geo_id}: {exc}")
        return None

    # Find the first GPL platform with expression data
    expr_df = None
    for gpl_name, gpl in gse.gpls.items():
        try:
            pivot = gse.pivot_samples("VALUE")
            if pivot.empty:
                continue
            expr_df = pivot
            break
        except Exception:
            continue

    if expr_df is None:
        logger.warning(f"No expression data extracted from {geo_id}")
        return None

    # Map probe IDs to gene symbols using the GPL annotation table
    expr_df = _map_probes_to_genes(expr_df, gse)

    # Transpose: samples x genes
    expr_df = expr_df.T
    expr_df.index.name = "sample_id"

    # Drop columns with mostly NaN
    nan_frac = expr_df.isna().mean()
    expr_df = expr_df.loc[:, nan_frac < 0.5]

    # For duplicate gene symbols, take the mean
    expr_df = expr_df.T.groupby(level=0).mean().T

    logger.info(
        f"{geo_id}: {expr_df.shape[0]} samples x {expr_df.shape[1]} genes"
    )

    expr_df.to_parquet(cache_path)
    return expr_df


def parse_geo_response_labels(
    geo_id: str,
    dest_dir: Path,
    dataset_meta: Optional[dict] = None,
) -> Optional[pd.Series]:
    """
    Extract binary response labels from a GEO dataset's sample metadata.

    GEO stores sample phenotype info in ``characteristics_ch1`` as a list
    of "key: value" strings (e.g. "pathologic_response_pcr_rd: pCR").
    We parse these into a wide table and then look for response-related
    columns.

    Returns a Series indexed by sample_id with values 1 (responder) /
    0 (non-responder), or None if labels cannot be determined.
    """
    import GEOparse

    logger.info(f"Extracting response labels for {geo_id} ...")

    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(dest_dir), silent=True)
    except Exception as exc:
        logger.error(f"Could not load {geo_id} for label extraction: {exc}")
        return None

    # Build expanded phenotype table from characteristics_ch1
    pheno_rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {"sample_id": gsm_name}

        # Parse characteristics_ch1 key:value pairs
        chars = gsm.metadata.get("characteristics_ch1", [])
        if isinstance(chars, list):
            for item in chars:
                item_str = str(item).strip()
                if ":" in item_str:
                    key, val = item_str.split(":", 1)
                    row[key.strip().lower()] = val.strip()
                else:
                    row[f"char_{len(row)}"] = item_str

        # Also include other useful metadata fields
        for field in ["title", "source_name_ch1", "description"]:
            vals = gsm.metadata.get(field, [])
            if isinstance(vals, list) and vals:
                row[field] = "; ".join(str(v) for v in vals)
            elif vals:
                row[field] = str(vals)

        pheno_rows.append(row)

    if not pheno_rows:
        return None

    pheno = pd.DataFrame(pheno_rows).set_index("sample_id")

    # If we have CTR-DB predefined grouping criteria (most reliable)
    if dataset_meta and dataset_meta.get("predefined_grouping"):
        result = _extract_labels_from_predefined_grouping(
            pheno, dataset_meta["predefined_grouping"], geo_id
        )
        if result is not None:
            return result

    # If we have CTR-DB response_grouping metadata, use it to guide extraction
    if dataset_meta and dataset_meta.get("response_grouping"):
        grouping = dataset_meta["response_grouping"]
        result = _extract_labels_from_ctrdb_grouping(pheno, grouping, geo_id)
        if result is not None:
            return result

    # If we have fallback catalog with explicit labels
    if dataset_meta and "response_positive" in dataset_meta:
        return _extract_labels_from_fallback_meta(pheno, dataset_meta)

    # Heuristic: look for columns with response-related keywords
    return _heuristic_label_extraction(pheno, geo_id)


def download_all_breast_ctrdb(
    max_datasets: int = 20,
    min_sample_size: int = 30,
    dest_dir: Path = DATA_RAW / "ctrdb",
) -> pd.DataFrame:
    """
    Download the top breast-cancer datasets (by sample size) from CTR-DB / GEO.

    Steps:
        1. Fetch catalog from CTR-DB (or fallback).
        2. Sort by sample_size, filter to >= min_sample_size.
        3. Download expression from GEO and extract response labels.
        4. Save each dataset as expression + labels parquets.

    Returns the catalog DataFrame with a ``downloaded`` column.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    catalog = fetch_breast_ctrdb_catalog()
    catalog = catalog.sort_values("sample_size", ascending=False).reset_index(drop=True)
    catalog = catalog[catalog["sample_size"] >= min_sample_size]

    # Deduplicate by GEO source — keep the entry with the largest sample size
    catalog = (
        catalog
        .sort_values("sample_size", ascending=False)
        .drop_duplicates(subset="geo_source", keep="first")
        .head(max_datasets)
        .reset_index(drop=True)
    )

    logger.info(
        f"Will attempt to download {len(catalog)} unique GEO datasets "
        f"(min size={min_sample_size})"
    )

    downloaded = []
    for idx, row in catalog.iterrows():
        geo_id = row["geo_source"]
        ds_dir = dest_dir / geo_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        success = False
        try:
            # Download expression
            expr = download_geo_expression(geo_id, ds_dir)
            if expr is None:
                logger.warning(f"Skipping {geo_id}: no expression data")
                downloaded.append(False)
                continue

            # Extract labels — pass the full row as metadata
            meta = row.to_dict()
            labels = parse_geo_response_labels(geo_id, ds_dir, dataset_meta=meta)
            if labels is None:
                logger.warning(f"Skipping {geo_id}: no response labels")
                downloaded.append(False)
                continue

            # Save labels
            labels.to_frame("response").to_parquet(ds_dir / "response_labels.parquet")

            # Verify overlap
            common_samples = expr.index.intersection(labels.index)
            if len(common_samples) < 10:
                logger.warning(
                    f"Skipping {geo_id}: only {len(common_samples)} "
                    f"samples overlap between expression and labels"
                )
                downloaded.append(False)
                continue

            success = True
            logger.info(
                f"OK  {geo_id}: {len(common_samples)} samples with "
                f"expression + labels "
                f"(R={int(labels[common_samples].sum())}, "
                f"NR={int((1-labels[common_samples]).sum())})"
            )
        except Exception as exc:
            logger.error(f"Failed {geo_id}: {exc}")

        downloaded.append(success)
        time.sleep(1)  # be polite to GEO servers

    catalog["downloaded"] = downloaded

    # Save catalog
    catalog.to_csv(dest_dir / "catalog.csv", index=False)
    logger.info(
        f"Downloaded {sum(downloaded)}/{len(catalog)} datasets "
        f"to {dest_dir}"
    )
    return catalog


def load_ctrdb_dataset(
    dataset_dir: Path,
) -> Optional[tuple[pd.DataFrame, pd.Series]]:
    """
    Load a single downloaded CTR-DB/GEO dataset.

    Returns (expression_df, response_series) or None if files missing.
    expression_df: samples x genes
    response_series: binary (1=responder, 0=non-responder)
    """
    # Find expression parquet
    expr_files = list(dataset_dir.glob("*_expression.parquet"))
    if not expr_files:
        logger.warning(f"No expression file in {dataset_dir}")
        return None

    label_file = dataset_dir / "response_labels.parquet"
    if not label_file.exists():
        logger.warning(f"No labels file in {dataset_dir}")
        return None

    expr = pd.read_parquet(expr_files[0])
    labels = pd.read_parquet(label_file)["response"]

    # Align
    common = expr.index.intersection(labels.index)
    if len(common) < 10:
        logger.warning(
            f"Only {len(common)} overlapping samples in {dataset_dir.name}"
        )
        return None

    return expr.loc[common], labels.loc[common]


def load_all_breast_ctrdb(
    data_dir: Path = DATA_RAW / "ctrdb",
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Load all successfully downloaded breast-cancer datasets.

    Returns dict: geo_id -> (expression_df, response_series)
    """
    if not data_dir.exists():
        logger.warning(f"CTR-DB data directory not found: {data_dir}")
        return {}

    datasets = {}
    for ds_dir in sorted(data_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        geo_id = ds_dir.name
        result = load_ctrdb_dataset(ds_dir)
        if result is not None:
            datasets[geo_id] = result
            expr, labels = result
            logger.info(
                f"Loaded {geo_id}: {expr.shape[0]} samples, "
                f"{int(labels.sum())} responders / "
                f"{int((1-labels).sum())} non-responders"
            )

    logger.info(f"Loaded {len(datasets)} CTR-DB datasets total")
    return datasets


# ── Internal helpers ──────────────────────────────────────────────────

def _fallback_catalog() -> pd.DataFrame:
    """Return the hardcoded fallback catalog as a DataFrame."""
    df = pd.DataFrame(FALLBACK_BREAST_DATASETS)
    return df


def _collect_dataset_ids(cancer_data) -> list[str]:
    """
    Extract CTR-DB dataset ID strings from searchCancerApi response.

    The API returns a list of cancer-type records; each has a
    ``CTR-DB dataset`` field containing a list of ID strings.
    """
    ids = []
    if not isinstance(cancer_data, list):
        return ids
    for record in cancer_data:
        if isinstance(record, dict) and "CTR-DB dataset" in record:
            ds_list = record["CTR-DB dataset"]
            if isinstance(ds_list, list):
                ids.extend(str(d) for d in ds_list)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for d in ids:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _parse_geo_id(source_str: str) -> str:
    """Extract GSE accession from Source field like 'GEO:GSE20194'."""
    if not source_str:
        return ""
    # Handle formats: "GEO:GSE20194", "GSE20194", "GEO: GSE20194"
    match = re.search(r"(GSE\d+)", str(source_str))
    return match.group(1) if match else ""


def _parse_response_grouping(grouping_str: str) -> dict[str, str]:
    """
    Parse 'Original resposne grouping' like 'pCR:46;RD:165;'
    into {group_name: count_str}.
    """
    groups = {}
    if not grouping_str:
        return groups
    for part in str(grouping_str).split(";"):
        part = part.strip()
        if ":" in part:
            name, count = part.rsplit(":", 1)
            groups[name.strip()] = count.strip()
    return groups


def _fetch_all_dataset_metadata(
    dataset_ids: list[str],
    timeout: int = 15,
    batch_pause: float = 0.2,
) -> list[dict]:
    """
    Fetch metadata for each CTR-DB dataset ID via searchDatasetApi.
    Returns a list of standardized dicts.
    """
    records = []
    for i, ds_id in enumerate(dataset_ids):
        try:
            resp = requests.post(
                f"{CTRDB_API_BASE}/searchDatasetApi",
                json={"search": ds_id},
                timeout=timeout,
            )
            resp.raise_for_status()
            api_data = resp.json()

            # Response is a list; take the first item
            if isinstance(api_data, list) and api_data:
                item = api_data[0]
            elif isinstance(api_data, dict):
                item = api_data
            else:
                continue

            rec = {
                "dataset_id": item.get("CTR-DB ID", ds_id),
                "drug": item.get("Therapeutic regimen", "unknown"),
                "sample_size": int(item.get("Sample size", 0)),
                "n_responders": int(
                    item.get('Sample size of the "response" group', 0)
                ),
                "n_nonresponders": int(
                    item.get('Sample size of the "non-response" group', 0)
                ),
                "geo_source": _parse_geo_id(item.get("Source", "")),
                "platform": item.get("Platform", ""),
                "response_grouping": item.get("Original resposne grouping", ""),
                "predefined_grouping": item.get(
                    "Predefined grouping criteria of response and non-response groups", ""
                ),
                "data_type": item.get("Data type", ""),
                "dataset_type": item.get("Dataset type", ""),
            }
            records.append(rec)

            if (i + 1) % 20 == 0:
                logger.info(f"  Fetched metadata for {i+1}/{len(dataset_ids)} datasets")

        except Exception as exc:
            logger.debug(f"Failed to fetch {ds_id}: {exc}")
            continue

        time.sleep(batch_pause)

    logger.info(f"Fetched metadata for {len(records)}/{len(dataset_ids)} datasets")
    return records


def _map_probes_to_genes(
    expr_df: pd.DataFrame, gse
) -> pd.DataFrame:
    """
    Map probe IDs in expression to gene symbols using GPL annotation.

    expr_df has probes as rows, samples as columns.
    Returns the same shape but with gene symbols as row index.
    """
    # Try to get gene symbol mapping from GPL
    probe_to_gene = {}
    for gpl_name, gpl in gse.gpls.items():
        table = gpl.table
        if table.empty:
            continue

        # Look for gene symbol columns (common names)
        symbol_col = None
        for candidate in [
            "Gene Symbol", "gene_symbol", "GENE_SYMBOL", "Symbol",
            "Gene symbol", "GENE", "gene_assignment", "Gene_Symbol",
            "gene", "ORF",
        ]:
            if candidate in table.columns:
                symbol_col = candidate
                break

        if symbol_col is None:
            # Try partial match
            for col in table.columns:
                if "gene" in col.lower() and "symbol" in col.lower():
                    symbol_col = col
                    break
                if col.lower() == "gene":
                    symbol_col = col
                    break

        if symbol_col is not None:
            id_col = "ID" if "ID" in table.columns else table.columns[0]
            for _, row in table.iterrows():
                probe_id = str(row[id_col])
                gene_sym = str(row[symbol_col]).strip()
                if gene_sym and gene_sym not in ("nan", "---", "", "NA"):
                    # Some platforms have multiple symbols separated by " /// "
                    first_sym = gene_sym.split("///")[0].strip()
                    if first_sym:
                        probe_to_gene[probe_id] = first_sym

    if not probe_to_gene:
        logger.warning("Could not map probes to gene symbols")
        return expr_df

    # Apply mapping
    new_index = [probe_to_gene.get(str(p), str(p)) for p in expr_df.index]
    expr_df.index = new_index
    # Remove rows that didn't map (still have probe IDs)
    mapped_mask = expr_df.index.isin(probe_to_gene.values())
    if mapped_mask.sum() > 100:
        expr_df = expr_df.loc[mapped_mask]

    return expr_df


def _extract_labels_from_predefined_grouping(
    pheno: pd.DataFrame,
    predefined: str,
    geo_id: str,
) -> Optional[pd.Series]:
    """
    Extract labels using CTR-DB 'Predefined grouping criteria' string.

    This string explicitly states which values are Response vs Non_response.
    Example: "Response: pCR; Non_response: RD. Annotation: ..."
             "Response: pcr: 1; Non_response: pcr: 0. Annotation: ..."
    """
    if not predefined:
        return None

    # Parse "Response: X; Non_response: Y" pattern
    # Strip annotation part
    main_part = predefined.split("Annotation:")[0].strip().rstrip(".")

    pos_terms = []
    neg_terms = []

    # Extract "Response: ..." and "Non_response: ..." parts
    resp_match = re.search(r"Response:\s*([^;]+)", main_part)
    nonresp_match = re.search(r"Non_response:\s*([^;.]+)", main_part)

    if resp_match:
        pos_terms = [resp_match.group(1).strip().lower()]
    if nonresp_match:
        neg_terms = [nonresp_match.group(1).strip().lower()]

    if not pos_terms or not neg_terms:
        return None

    logger.info(
        f"{geo_id}: Predefined grouping: pos={pos_terms}, neg={neg_terms}"
    )

    # Search phenotype columns for exact matches first, then partial
    for exact in [True, False]:
        for col in pheno.columns:
            col_values = pheno[col].astype(str).str.lower().str.strip()

            if exact:
                has_pos = any(col_values.eq(t).any() for t in pos_terms)
                has_neg = any(col_values.eq(t).any() for t in neg_terms)
            else:
                has_pos = any(col_values.str.contains(t, regex=False).any() for t in pos_terms)
                has_neg = any(col_values.str.contains(t, regex=False).any() for t in neg_terms)

            if has_pos and has_neg:
                labels = pd.Series(np.nan, index=pheno.index, dtype=float)
                for t in pos_terms:
                    if exact:
                        labels[col_values.eq(t)] = 1.0
                    else:
                        labels[col_values.str.contains(t, regex=False)] = 1.0
                for t in neg_terms:
                    if exact:
                        labels[col_values.eq(t)] = 0.0
                    else:
                        labels[col_values.str.contains(t, regex=False)] = 0.0

                labels = labels.dropna()
                if len(labels) >= 10:
                    n_pos = int(labels.sum())
                    n_neg = int((1 - labels).sum())
                    if n_pos >= 3 and n_neg >= 3:
                        logger.info(
                            f"{geo_id}: predefined labels from '{col}': "
                            f"{n_pos} responders, {n_neg} non-responders"
                        )
                        return labels.astype(int)

    return None


def _extract_labels_from_ctrdb_grouping(
    pheno: pd.DataFrame,
    grouping_str: str,
    geo_id: str,
) -> Optional[pd.Series]:
    """
    Extract labels using CTR-DB 'Original resposne grouping' string.

    The grouping string is like "pCR:46;RD:165;" which tells us the
    response group names.  We look for these as values in the phenotype
    table columns.
    """
    groups = _parse_response_grouping(grouping_str)
    if len(groups) < 2:
        return None

    # Determine which groups are response vs non-response.
    # Check negative patterns FIRST to catch "npCR", "nCR", "non-pCR" etc.
    neg_patterns = [
        "npcr", "ncr", "non-pcr", "non pcr", "npCR",
        "rd", "residual disease",
        "non-response", "non-responder", "nonresponder",
        "resistant", "acquired-resistant",
        "relapse", "recurrence",
        "rcb-ii", "rcb-iii", "rcb-ii/iii",
        "progressive", "sd", "pd",
    ]
    pos_patterns = [
        "pcr", "pathologic complete",
        "response", "responder", "sensitive",
        "cr", "pr",
        "no relapse", "no recurrence",
        "rcb-0", "rcb-i", "rcb-0/i",
    ]

    pos_groups = []
    neg_groups = []
    for group_name in groups:
        name_lower = group_name.lower().strip()

        # Check negative first (catches npCR, nCR before pcr matches)
        is_neg = any(pat in name_lower for pat in neg_patterns)
        # Also check for leading "n" prefix before "pcr"/"cr"
        if not is_neg and re.match(r"^n\s*p?cr", name_lower):
            is_neg = True

        is_pos = any(pat in name_lower for pat in pos_patterns)

        if is_neg:
            neg_groups.append(group_name)
        elif is_pos:
            pos_groups.append(group_name)
        else:
            # Unknown; will be handled below
            pass

    if not pos_groups or not neg_groups:
        # Fall back: first group = response, rest = non-response
        group_names = list(groups.keys())
        pos_groups = [group_names[0]]
        neg_groups = group_names[1:]

    logger.info(
        f"{geo_id}: CTR-DB groups: pos={pos_groups}, neg={neg_groups}"
    )

    # Search phenotype columns for matching values
    pos_terms = [g.lower().strip() for g in pos_groups]
    neg_terms = [g.lower().strip() for g in neg_groups]

    # Try each column
    for col in pheno.columns:
        col_values = pheno[col].astype(str).str.lower().str.strip()

        # Check if column values match the group names
        has_pos = any(col_values.eq(t).any() for t in pos_terms)
        has_neg = any(col_values.eq(t).any() for t in neg_terms)

        if has_pos and has_neg:
            labels = pd.Series(np.nan, index=pheno.index, dtype=float)
            for t in pos_terms:
                labels[col_values.eq(t)] = 1.0
            for t in neg_terms:
                labels[col_values.eq(t)] = 0.0
            labels = labels.dropna()
            if len(labels) >= 10:
                n_pos = int(labels.sum())
                n_neg = int((1 - labels).sum())
                if n_pos >= 3 and n_neg >= 3:
                    logger.info(
                        f"{geo_id}: CTR-DB grouping labels from '{col}': "
                        f"{n_pos} responders, {n_neg} non-responders"
                    )
                    return labels.astype(int)

    # Also try partial match (contains) for the group terms
    for col in pheno.columns:
        col_values = pheno[col].astype(str).str.lower().str.strip()

        has_pos = any(col_values.str.contains(t, regex=False).any() for t in pos_terms)
        has_neg = any(col_values.str.contains(t, regex=False).any() for t in neg_terms)

        if has_pos and has_neg:
            labels = pd.Series(np.nan, index=pheno.index, dtype=float)
            for t in pos_terms:
                labels[col_values.str.contains(t, regex=False)] = 1.0
            for t in neg_terms:
                labels[col_values.str.contains(t, regex=False)] = 0.0
            labels = labels.dropna()
            if len(labels) >= 10:
                n_pos = int(labels.sum())
                n_neg = int((1 - labels).sum())
                if n_pos >= 3 and n_neg >= 3:
                    logger.info(
                        f"{geo_id}: CTR-DB partial match labels from '{col}': "
                        f"{n_pos} responders, {n_neg} non-responders"
                    )
                    return labels.astype(int)

    return None


def _extract_labels_from_fallback_meta(
    pheno: pd.DataFrame, meta: dict
) -> Optional[pd.Series]:
    """
    Extract binary labels using the explicit response_positive / response_negative
    mappings from the fallback catalog.
    """
    pos_terms = [t.lower() for t in meta.get("response_positive", [])]
    neg_terms = [t.lower() for t in meta.get("response_negative", [])]

    if not pos_terms and not neg_terms:
        return None

    all_terms = pos_terms + neg_terms

    # Search across all phenotype columns for matching terms
    best_col = None
    best_match_count = 0

    for col in pheno.columns:
        col_values = pheno[col].astype(str).str.lower()
        match_count = sum(
            col_values.str.contains(term, regex=False).sum()
            for term in all_terms
        )
        if match_count > best_match_count:
            best_match_count = match_count
            best_col = col

    if best_col is None or best_match_count == 0:
        return _heuristic_label_extraction(pheno, meta.get("geo_source", ""))

    values = pheno[best_col].astype(str).str.lower()
    labels = pd.Series(np.nan, index=pheno.index, dtype=float)

    for term in pos_terms:
        labels[values.str.contains(term, regex=False)] = 1.0
    for term in neg_terms:
        labels[values.str.contains(term, regex=False)] = 0.0

    labels = labels.dropna()
    if len(labels) < 10:
        return _heuristic_label_extraction(pheno, meta.get("geo_source", ""))

    logger.info(
        f"Labels from fallback meta: {int(labels.sum())} responders, "
        f"{int((1-labels).sum())} non-responders"
    )
    return labels.astype(int)


def _heuristic_label_extraction(
    pheno: pd.DataFrame, geo_id: str
) -> Optional[pd.Series]:
    """
    Heuristic label extraction by scanning phenotype columns for
    response-related keywords.
    """
    # Keywords indicating response (positive class)
    pos_keywords = [
        "pcr", "pathologic complete response", "complete response",
        "responder", "sensitive", "no relapse", "no recurrence",
        "response", "cr", "pr",
    ]
    neg_keywords = [
        "rd", "residual disease", "non-responder", "resistant",
        "relapse", "recurrence", "progressive disease", "pd", "sd",
        "non-pcr", "no response",
    ]

    for col in pheno.columns:
        col_values = pheno[col].astype(str).str.lower()

        # Check if this column contains response-related values
        has_pos = any(col_values.str.contains(kw, regex=False).any() for kw in pos_keywords)
        has_neg = any(col_values.str.contains(kw, regex=False).any() for kw in neg_keywords)

        if has_pos and has_neg:
            labels = pd.Series(np.nan, index=pheno.index, dtype=float)
            for kw in pos_keywords:
                mask = col_values.str.contains(kw, regex=False)
                labels[mask] = 1.0
            for kw in neg_keywords:
                mask = col_values.str.contains(kw, regex=False)
                labels[mask] = 0.0

            labels = labels.dropna()
            if len(labels) >= 10:
                n_pos = int(labels.sum())
                n_neg = int((1 - labels).sum())
                # Sanity: both classes represented
                if n_pos >= 3 and n_neg >= 3:
                    logger.info(
                        f"{geo_id}: heuristic labels from column '{col}': "
                        f"{n_pos} responders, {n_neg} non-responders"
                    )
                    return labels.astype(int)

    logger.warning(f"{geo_id}: could not extract response labels heuristically")
    return None


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Phase 1: Fetch catalog
    logger.info("=" * 60)
    logger.info("Phase 1: Fetch CTR-DB breast-cancer catalog")
    logger.info("=" * 60)
    catalog = fetch_breast_ctrdb_catalog()
    print(f"\nCatalog: {len(catalog)} datasets")
    print(catalog[["dataset_id", "drug", "sample_size", "geo_source"]].to_string())

    # Save catalog
    out_dir = DATA_RAW / "ctrdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "catalog.csv", index=False)
    print(f"\nCatalog saved to {out_dir / 'catalog.csv'}")
