#!/usr/bin/env python3
"""
Search GEO for correct accessions for IMPACT and TransNEOS trials,
and try alternative known accessions.

Usage:
    pixi run python scripts/search_correct_accessions.py
"""

import os
import urllib.request
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/marinaesteban-medina/Desktop/INVEREX/inverex-mvp")
RAW_DIR = BASE_DIR / "data" / "raw"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def search_geo(term, max_results=10):
    """Search GEO using NCBI Entrez e-utils."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    # Search for GEO DataSets (GDS) or Series (GSE)
    search_url = (
        f"{base}/esearch.fcgi?db=gds&term={term.replace(' ', '+')}"
        f"&retmax={max_results}&retmode=json"
    )
    log(f"  Searching GEO: {term}")
    try:
        with urllib.request.urlopen(search_url, timeout=30) as resp:
            data = json.loads(resp.read())
        ids = data.get("esearchresult", {}).get("idlist", [])
        log(f"  Found {len(ids)} results")

        if not ids:
            return []

        # Fetch summaries
        id_str = ",".join(ids)
        summary_url = (
            f"{base}/esummary.fcgi?db=gds&id={id_str}&retmode=json"
        )
        with urllib.request.urlopen(summary_url, timeout=30) as resp:
            summary = json.loads(resp.read())

        results = []
        for uid in ids:
            entry = summary.get("result", {}).get(uid, {})
            accession = entry.get("accession", "")
            title = entry.get("title", "")
            gpl = entry.get("gpl", "")
            n_samples = entry.get("n_samples", "")
            summary_text = entry.get("summary", "")[:200]
            results.append({
                "uid": uid,
                "accession": accession,
                "title": title,
                "platform": gpl,
                "n_samples": n_samples,
                "summary": summary_text,
            })
            log(f"    {accession}: {title[:80]} ({n_samples} samples)")

        return results
    except Exception as e:
        log(f"  Search error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
# Search for IMPACT trial
# ═══════════════════════════════════════════════════════════════════
log("=" * 70)
log("Searching for IMPACT breast cancer endocrine trial")
log("=" * 70)

search_geo("IMPACT anastrozole tamoxifen breast neoadjuvant")
log("")
search_geo("IMPACT breast endocrine neoadjuvant Ki67")
log("")
# Known alternative: the IMPACT trial data might be under GSE111563 or similar
# Also try the Edinburgh group who ran IMPACT
search_geo("Dowsett anastrozole tamoxifen breast neoadjuvant gene expression")

# ═══════════════════════════════════════════════════════════════════
# Search for TransNEOS / correct letrozole neoadjuvant
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Searching for TransNEOS letrozole neoadjuvant")
log("=" * 70)

search_geo("TransNEOS letrozole breast neoadjuvant")
log("")
search_geo("neoadjuvant letrozole breast cancer gene expression response")
log("")
# GSE59515 is a known TransNEOS-related dataset
search_geo("letrozole ER-positive breast neoadjuvant RNA clinical response")

# ═══════════════════════════════════════════════════════════════════
# Also search for additional endocrine/targeted datasets we might add
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Searching for additional endocrine therapy response datasets")
log("=" * 70)

search_geo("neoadjuvant endocrine therapy breast cancer pCR Ki67 gene expression", max_results=15)
log("")
search_geo("aromatase inhibitor breast cancer response gene expression profile", max_results=15)

# ═══════════════════════════════════════════════════════════════════
# Search for PARP inhibitor response datasets beyond BrighTNess
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Searching for additional PARP inhibitor datasets")
log("=" * 70)

search_geo("PARP inhibitor breast cancer response gene expression", max_results=10)
log("")
search_geo("olaparib breast cancer gene expression response")

# ═══════════════════════════════════════════════════════════════════
# Search for immunotherapy breast cancer response
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("Searching for immunotherapy breast cancer datasets")
log("=" * 70)

search_geo("pembrolizumab breast cancer gene expression response", max_results=10)
log("")
search_geo("atezolizumab breast cancer gene expression response")
log("")
search_geo("KEYNOTE-522 breast cancer gene expression")

log("\nDONE")
