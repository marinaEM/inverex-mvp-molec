"""
Map top-ranked drugs to active breast cancer clinical trials
using the ClinicalTrials.gov API v2.
"""
import logging
from typing import Optional

import requests

from src.config import CTGOV_API

logger = logging.getLogger(__name__)


def search_breast_trials(drug_name: str, max_results: int = 5) -> list[dict]:
    """Search for active breast cancer trials involving a specific drug."""
    params = {
        "query.cond": "breast cancer",
        "query.intr": drug_name,
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING",
        "fields": "NCTId,BriefTitle,Condition,InterventionName,Phase,OverallStatus,LocationCity,EligibilityCriteria",
        "pageSize": max_results,
        "format": "json",
    }
    try:
        resp = requests.get(f"{CTGOV_API}/studies", params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"CT.gov API returned {resp.status_code} for '{drug_name}'")
            return []
    except requests.RequestException as e:
        logger.warning(f"CT.gov API request failed for '{drug_name}': {e}")
        return []

    data = resp.json()
    studies = data.get("studies", [])
    results = []
    for s in studies:
        proto = s.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        arms = proto.get("armsInterventionsModule", {})
        eligibility = proto.get("eligibilityModule", {})
        results.append({
            "nct_id": ident.get("nctId"),
            "title": ident.get("briefTitle"),
            "phase": design.get("phases", [None])[0] if design.get("phases") else None,
            "status": status.get("overallStatus"),
            "interventions": [i.get("name") for i in arms.get("interventions", [])],
            "eligibility_snippet": (eligibility.get("eligibilityCriteria") or "")[:500],
        })
    return results


def map_trials_for_rankings(
    rankings_df,
    top_n: int = 5,
    max_trials_per_drug: int = 3,
) -> dict[str, list[dict]]:
    """
    Map clinical trials for the top-N drugs in a patient's ranking.

    Returns dict: drug_name -> list of trial dicts
    """
    trial_map = {}
    filtered = rankings_df.copy()
    if "excluded_flag" in filtered.columns:
        filtered = filtered[~filtered["excluded_flag"].fillna(False)]

    for _, row in filtered.head(top_n).iterrows():
        drug = row["drug_name"]
        query_term = row.get("trial_query_term", drug)
        trials = search_breast_trials(query_term, max_results=max_trials_per_drug)
        trial_map[drug] = trials
        if trials:
            logger.info(f"  {drug}: {len(trials)} active trial(s)")
        else:
            logger.info(f"  {drug}: no active breast cancer trials found")
    return trial_map
