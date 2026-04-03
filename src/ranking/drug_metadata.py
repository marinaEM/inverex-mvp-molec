"""
Curated breast-cancer drug metadata used by the personalized ranker.

The current MVP ranks only compounds that already exist in the cached
fingerprint table / matched training set. This module adds a compact
actionability layer on top of that fixed compound universe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED

DEFAULT_METADATA_PATH = DATA_PROCESSED / "drug_metadata_breast.tsv"


def normalize_drug_name(name: str) -> str:
    """Normalize a drug name for dictionary lookups."""
    return str(name).strip().lower().replace("_", "-")


STATUS_SCORES = {
    "approved_breast": 1.0,
    "approved_oncology": 0.65,
    "approved_nononcology": -0.35,
    "investigational_breast": 0.45,
    "investigational_oncology": 0.2,
    "preclinical": -0.2,
    "tool": -1.0,
}


STANDARD_OF_CARE_BREAST = {
    "tamoxifen",
    "fulvestrant",
    "palbociclib",
    "lapatinib",
    "olaparib",
    "paclitaxel",
    "docetaxel",
    "vinorelbine",
    "epirubicin",
    "cyclophosphamide",
    "fluorouracil",
    "gemcitabine",
    "methotrexate",
}

TOOL_COMPOUNDS = {
    "agk-2",
    "azd-5438",
    "azd-7762",
    "bib r-1532",
    "bibr-1532",
    "bms-345541",
    "bms-536924",
    "gw-441756",
    "ku-55933",
    "mg-132",
    "mirin",
    "pci-34051",
    "podophyllotoxin",
    "ro-3306",
    "sb-216763",
    "sb-590885",
    "staurosporine",
    "xav-939",
    "zm-447439",
}

APPROVED_BREAST = {
    "docetaxel",
    "epirubicin",
    "fluorouracil",
    "fulvestrant",
    "gemcitabine",
    "lapatinib",
    "methotrexate",
    "olaparib",
    "paclitaxel",
    "palbociclib",
    "tamoxifen",
    "vinorelbine",
}

APPROVED_ONCOLOGY = {
    "afatinib",
    "axitinib",
    "azacitidine",
    "bortezomib",
    "bosutinib",
    "cisplatin",
    "crizotinib",
    "cyclophosphamide",
    "cytarabine",
    "dacarbazine",
    "dasatinib",
    "erlotinib",
    "fludarabine",
    "gefitinib",
    "ibrutinib",
    "irinotecan",
    "lenalidomide",
    "methotrexate",
    "mitoxantrone",
    "nilotinib",
    "olaparib",
    "rucaparib",
    "ruxolitinib",
    "sorafenib",
    "temozolomide",
    "temsirolimus",
    "topotecan",
    "trametinib",
    "vinblastine",
    "vorinostat",
}

APPROVED_NONONCOLOGY = {
    "acetylcysteine",
    "ascorbic-acid",
    "leflunomide",
    "mycophenolic-acid",
    "sirolimus",
}

INVESTIGATIONAL_BREAST = {
    "buparlisib",
    "entinostat",
    "gdc-0941",
    "mk-2206",
    "navitoclax",
    "nvp-auy922",
    "nvp-bez235",
    "selumetinib",
    "sirolimus",
    "tanespimycin",
    "veliparib",
}

INVESTIGATIONAL_ONCOLOGY = {
    "abt-737",
    "alisertib",
    "azd-6482",
    "azd-8055",
    "bms-754807",
    "cediranib",
    "foretinib",
    "lestaurtinib",
    "linsitinib",
    "motesanib",
    "nvp-bez235",
    "obatoclax",
    "osi-027",
    "pevonedistat",
    "prima-1-met",
    "tozasertib",
}


CURATED_OVERRIDES = {
    "lapatinib": {
        "aliases": "GW572016",
        "target_pathway": "ERBB2/HER2, EGFR",
        "drug_class": "HER2/EGFR tyrosine kinase inhibitor",
        "breast_cancer_relevance": 3,
        "her2_relevance": 3,
        "trial_query_term": "lapatinib",
        "notes": "Clinically plausible HER2-directed agent for HER2-positive breast cancer.",
    },
    "afatinib": {
        "target_pathway": "ERBB family",
        "drug_class": "pan-ERBB tyrosine kinase inhibitor",
        "breast_cancer_relevance": 1,
        "her2_relevance": 2,
        "trial_query_term": "afatinib",
        "notes": "Approved outside breast cancer; possible HER-family relevance but weaker breast-cancer precedence than lapatinib.",
    },
    "gefitinib": {
        "target_pathway": "EGFR",
        "drug_class": "EGFR tyrosine kinase inhibitor",
        "breast_cancer_relevance": 1,
        "her2_relevance": 1,
        "trial_query_term": "gefitinib",
    },
    "erlotinib": {
        "target_pathway": "EGFR",
        "drug_class": "EGFR tyrosine kinase inhibitor",
        "breast_cancer_relevance": 1,
        "her2_relevance": 1,
        "trial_query_term": "erlotinib",
    },
    "tanespimycin": {
        "aliases": "17-AAG",
        "target_pathway": "HSP90 client proteins, HER2 stabilization",
        "drug_class": "HSP90 inhibitor",
        "breast_cancer_relevance": 2,
        "her2_relevance": 2,
        "hsp90_relevance": 3,
        "trial_query_term": "tanespimycin",
        "notes": "Historically explored in HER2-positive breast cancer; not standard of care.",
    },
    "nvp-auy922": {
        "aliases": "luminespib",
        "target_pathway": "HSP90 client proteins, HER2 stabilization",
        "drug_class": "HSP90 inhibitor",
        "breast_cancer_relevance": 2,
        "her2_relevance": 2,
        "hsp90_relevance": 3,
        "trial_query_term": "luminespib",
        "notes": "Investigational HSP90 inhibitor with HER2-pathway plausibility.",
    },
    "gdc-0941": {
        "aliases": "pictilisib",
        "target_pathway": "PI3K",
        "drug_class": "PI3K inhibitor",
        "breast_cancer_relevance": 2,
        "pi3k_mtor_relevance": 3,
        "trial_query_term": "pictilisib",
        "notes": "PI3K-pathway inhibitor with breast-cancer trial history.",
    },
    "nvp-bez235": {
        "aliases": "dactolisib",
        "target_pathway": "PI3K/mTOR",
        "drug_class": "dual PI3K/mTOR inhibitor",
        "breast_cancer_relevance": 2,
        "pi3k_mtor_relevance": 3,
        "trial_query_term": "dactolisib",
        "notes": "Dual PI3K/mTOR inhibitor; plausible for PIK3CA-altered disease but not standard of care.",
    },
    "buparlisib": {
        "aliases": "BKM120",
        "target_pathway": "PI3K",
        "drug_class": "pan-PI3K inhibitor",
        "breast_cancer_relevance": 2,
        "pi3k_mtor_relevance": 3,
        "trial_query_term": "buparlisib",
    },
    "mk-2206": {
        "target_pathway": "AKT",
        "drug_class": "AKT inhibitor",
        "breast_cancer_relevance": 2,
        "pi3k_mtor_relevance": 2,
        "trial_query_term": "MK-2206",
    },
    "azd-6482": {
        "target_pathway": "PI3K beta",
        "drug_class": "PI3K inhibitor",
        "breast_cancer_relevance": 1,
        "pi3k_mtor_relevance": 2,
    },
    "azd-8055": {
        "target_pathway": "mTOR",
        "drug_class": "mTOR kinase inhibitor",
        "breast_cancer_relevance": 1,
        "pi3k_mtor_relevance": 2,
    },
    "osi-027": {
        "target_pathway": "mTOR",
        "drug_class": "mTOR kinase inhibitor",
        "breast_cancer_relevance": 1,
        "pi3k_mtor_relevance": 2,
    },
    "sirolimus": {
        "aliases": "rapamycin",
        "target_pathway": "mTOR",
        "drug_class": "mTOR inhibitor",
        "breast_cancer_relevance": 1,
        "pi3k_mtor_relevance": 2,
        "trial_query_term": "sirolimus",
        "notes": "Mechanistically plausible via mTOR but not a standard breast-cancer recommendation.",
    },
    "temsirolimus": {
        "target_pathway": "mTOR",
        "drug_class": "mTOR inhibitor",
        "breast_cancer_relevance": 1,
        "pi3k_mtor_relevance": 2,
        "trial_query_term": "temsirolimus",
        "notes": "Mechanistically plausible via mTOR; clinically weaker breast-cancer rationale than HER2-directed therapy in HER2-positive disease.",
    },
    "tamoxifen": {
        "target_pathway": "ESR1",
        "drug_class": "selective estrogen receptor modulator",
        "breast_cancer_relevance": 3,
        "endocrine_relevance": 3,
        "luminal_relevance": 3,
        "trial_query_term": "tamoxifen",
        "notes": "Standard endocrine therapy class for ER-positive breast cancer.",
    },
    "fulvestrant": {
        "target_pathway": "ESR1",
        "drug_class": "selective estrogen receptor degrader",
        "breast_cancer_relevance": 3,
        "endocrine_relevance": 3,
        "luminal_relevance": 3,
        "trial_query_term": "fulvestrant",
        "notes": "Standard endocrine therapy class for ER-positive breast cancer.",
    },
    "palbociclib": {
        "target_pathway": "CDK4/6",
        "drug_class": "CDK4/6 inhibitor",
        "breast_cancer_relevance": 3,
        "cdk46_relevance": 3,
        "luminal_relevance": 3,
        "trial_query_term": "palbociclib",
        "notes": "Strongly clinically relevant in HR-positive / luminal breast cancer.",
    },
    "olaparib": {
        "target_pathway": "PARP",
        "drug_class": "PARP inhibitor",
        "breast_cancer_relevance": 3,
        "parp_relevance": 3,
        "dna_damage_relevance": 3,
        "tnbc_relevance": 2,
        "trial_query_term": "olaparib",
        "notes": "Strong clinical rationale in germline BRCA-altered breast cancer.",
    },
    "rucaparib": {
        "target_pathway": "PARP",
        "drug_class": "PARP inhibitor",
        "breast_cancer_relevance": 2,
        "parp_relevance": 3,
        "dna_damage_relevance": 3,
        "trial_query_term": "rucaparib",
    },
    "veliparib": {
        "target_pathway": "PARP",
        "drug_class": "PARP inhibitor",
        "breast_cancer_relevance": 2,
        "parp_relevance": 2,
        "dna_damage_relevance": 2,
        "trial_query_term": "veliparib",
    },
    "cisplatin": {
        "target_pathway": "DNA damage",
        "drug_class": "platinum chemotherapy",
        "breast_cancer_relevance": 2,
        "dna_damage_relevance": 2,
        "tnbc_relevance": 2,
        "trial_query_term": "cisplatin",
    },
    "docetaxel": {
        "target_pathway": "microtubule",
        "drug_class": "taxane chemotherapy",
        "breast_cancer_relevance": 3,
        "microtubule_relevance": 3,
        "tnbc_relevance": 2,
        "trial_query_term": "docetaxel",
        "notes": "Standard breast-cancer chemotherapy class.",
    },
    "paclitaxel": {
        "target_pathway": "microtubule",
        "drug_class": "taxane chemotherapy",
        "breast_cancer_relevance": 3,
        "microtubule_relevance": 3,
        "tnbc_relevance": 2,
        "trial_query_term": "paclitaxel",
        "notes": "Standard breast-cancer chemotherapy class.",
    },
    "vinorelbine": {
        "target_pathway": "microtubule",
        "drug_class": "vinca alkaloid chemotherapy",
        "breast_cancer_relevance": 3,
        "microtubule_relevance": 2,
        "trial_query_term": "vinorelbine",
    },
    "vinblastine": {
        "target_pathway": "microtubule",
        "drug_class": "vinca alkaloid chemotherapy",
        "breast_cancer_relevance": 1,
        "microtubule_relevance": 2,
        "trial_query_term": "vinblastine",
    },
    "epirubicin": {
        "target_pathway": "DNA damage/topoisomerase",
        "drug_class": "anthracycline chemotherapy",
        "breast_cancer_relevance": 3,
        "dna_damage_relevance": 2,
        "trial_query_term": "epirubicin",
    },
    "cyclophosphamide": {
        "target_pathway": "DNA alkylation",
        "drug_class": "alkylating chemotherapy",
        "breast_cancer_relevance": 3,
        "dna_damage_relevance": 2,
        "trial_query_term": "cyclophosphamide",
    },
    "fluorouracil": {
        "aliases": "5-FU",
        "target_pathway": "antimetabolite",
        "drug_class": "antimetabolite chemotherapy",
        "breast_cancer_relevance": 3,
        "dna_damage_relevance": 1,
        "trial_query_term": "fluorouracil",
    },
    "gemcitabine": {
        "target_pathway": "antimetabolite",
        "drug_class": "antimetabolite chemotherapy",
        "breast_cancer_relevance": 3,
        "dna_damage_relevance": 1,
        "trial_query_term": "gemcitabine",
    },
    "methotrexate": {
        "target_pathway": "DHFR",
        "drug_class": "antimetabolite chemotherapy",
        "breast_cancer_relevance": 2,
        "trial_query_term": "methotrexate",
    },
    "irinotecan": {
        "target_pathway": "TOP1",
        "drug_class": "topoisomerase inhibitor",
        "breast_cancer_relevance": 1,
        "trial_query_term": "irinotecan",
    },
    "sn-38": {
        "target_pathway": "TOP1",
        "drug_class": "topoisomerase inhibitor metabolite",
        "breast_cancer_relevance": 1,
        "trial_query_term": "SN-38",
    },
    "topotecan": {
        "target_pathway": "TOP1",
        "drug_class": "topoisomerase inhibitor",
        "breast_cancer_relevance": 1,
        "trial_query_term": "topotecan",
    },
    "camptothecin": {
        "target_pathway": "TOP1",
        "drug_class": "topoisomerase inhibitor",
        "breast_cancer_relevance": 0,
        "trial_mappable": False,
        "notes": "Natural-product scaffold rather than practical breast-cancer recommendation.",
    },
    "dasatinib": {
        "target_pathway": "SRC/ABL",
        "drug_class": "SRC/ABL kinase inhibitor",
        "breast_cancer_relevance": 1,
        "trial_query_term": "dasatinib",
    },
    "trametinib": {
        "target_pathway": "MEK",
        "drug_class": "MEK inhibitor",
        "breast_cancer_relevance": 1,
        "trial_query_term": "trametinib",
    },
    "selumetinib": {
        "target_pathway": "MEK",
        "drug_class": "MEK inhibitor",
        "breast_cancer_relevance": 1,
        "trial_query_term": "selumetinib",
    },
    "entinostat": {
        "target_pathway": "HDAC",
        "drug_class": "HDAC inhibitor",
        "breast_cancer_relevance": 2,
        "luminal_relevance": 1,
        "trial_query_term": "entinostat",
    },
    "vorinostat": {
        "target_pathway": "HDAC",
        "drug_class": "HDAC inhibitor",
        "breast_cancer_relevance": 1,
        "trial_query_term": "vorinostat",
    },
    "navitoclax": {
        "target_pathway": "BCL2 family",
        "drug_class": "BCL2 family inhibitor",
        "breast_cancer_relevance": 1,
        "trial_query_term": "navitoclax",
    },
    "lestaurtinib": {
        "target_pathway": "FLT3/JAK family",
        "drug_class": "multi-kinase inhibitor",
        "breast_cancer_relevance": 0,
        "trial_query_term": "lestaurtinib",
        "notes": "Weak breast-cancer rationale; should not outrank biomarker-matched breast agents.",
    },
    "mg-132": {
        "target_pathway": "proteasome",
        "drug_class": "proteasome inhibitor tool compound",
        "breast_cancer_relevance": 0,
        "tool_compound": 1,
        "trial_mappable": False,
        "notes": "Laboratory proteasome inhibitor tool compound; exclude from recommendation outputs.",
    },
    "staurosporine": {
        "target_pathway": "broad kinase inhibition",
        "drug_class": "pan-kinase tool compound",
        "breast_cancer_relevance": 0,
        "tool_compound": 1,
        "trial_mappable": False,
        "notes": "Research tool compound; exclude from recommendation outputs.",
    },
}


def _base_row(drug_name: str) -> dict:
    """Return a conservative default metadata row."""
    return {
        "drug_name": drug_name,
        "aliases": "",
        "target_pathway": "Unknown",
        "drug_class": "Unclassified",
        "clinical_status": "preclinical",
        "status_score": STATUS_SCORES["preclinical"],
        "breast_cancer_relevance": 0,
        "her2_relevance": 0,
        "pi3k_mtor_relevance": 0,
        "endocrine_relevance": 0,
        "cdk46_relevance": 0,
        "parp_relevance": 0,
        "dna_damage_relevance": 0,
        "microtubule_relevance": 0,
        "tnbc_relevance": 0,
        "luminal_relevance": 0,
        "hsp90_relevance": 0,
        "tool_compound": 0,
        "standard_of_care_breast": 0,
        "trial_mappable": 1,
        "trial_query_term": drug_name,
        "notes": "",
    }


def _infer_status(norm_name: str) -> str:
    if norm_name in TOOL_COMPOUNDS:
        return "tool"
    if norm_name in APPROVED_BREAST:
        return "approved_breast"
    if norm_name in APPROVED_ONCOLOGY:
        return "approved_oncology"
    if norm_name in APPROVED_NONONCOLOGY:
        return "approved_nononcology"
    if norm_name in INVESTIGATIONAL_BREAST:
        return "investigational_breast"
    if norm_name in INVESTIGATIONAL_ONCOLOGY:
        return "investigational_oncology"
    return "preclinical"


def _apply_heuristics(row: dict) -> dict:
    """Infer broad class-based metadata for common breast-cancer classes."""
    norm_name = normalize_drug_name(row["drug_name"])
    row["clinical_status"] = _infer_status(norm_name)
    row["status_score"] = STATUS_SCORES[row["clinical_status"]]
    row["standard_of_care_breast"] = int(norm_name in STANDARD_OF_CARE_BREAST)

    endocrine = {"tamoxifen", "fulvestrant"}
    cdk46 = {"palbociclib"}
    parp = {"olaparib", "rucaparib", "veliparib"}
    microtubule = {"docetaxel", "paclitaxel", "vinorelbine", "vinblastine"}
    dna_damage = {
        "camptothecin",
        "cisplatin",
        "cyclophosphamide",
        "cytarabine",
        "dacarbazine",
        "dactinomycin",
        "epirubicin",
        "fluorouracil",
        "gemcitabine",
        "irinotecan",
        "methotrexate",
        "mitoxantrone",
        "sn-38",
        "temozolomide",
        "teniposide",
        "topotecan",
    }
    pi3k_mtor = {
        "azd-6482",
        "azd-8055",
        "buparlisib",
        "gdc-0941",
        "mk-2206",
        "nvp-bez235",
        "osi-027",
        "sirolimus",
        "temsirolimus",
    }
    her2 = {"afatinib", "erlotinib", "gefitinib", "lapatinib"}
    hsp90 = {"nvp-auy922", "tanespimycin"}
    luminal = endocrine | cdk46
    tnbc = {"cisplatin", "docetaxel", "gemcitabine", "olaparib", "paclitaxel", "vinorelbine"}

    if norm_name in endocrine:
        row["target_pathway"] = "ESR1"
        row["drug_class"] = "Endocrine therapy"
        row["endocrine_relevance"] = 3
        row["luminal_relevance"] = 3
        row["breast_cancer_relevance"] = max(row["breast_cancer_relevance"], 3)

    if norm_name in cdk46:
        row["target_pathway"] = "CDK4/6"
        row["drug_class"] = "CDK4/6 inhibitor"
        row["cdk46_relevance"] = 3
        row["luminal_relevance"] = 3
        row["breast_cancer_relevance"] = max(row["breast_cancer_relevance"], 3)

    if norm_name in parp:
        row["target_pathway"] = "PARP / homologous recombination"
        row["drug_class"] = "PARP inhibitor"
        row["parp_relevance"] = 3
        row["dna_damage_relevance"] = max(row["dna_damage_relevance"], 2)
        row["breast_cancer_relevance"] = max(row["breast_cancer_relevance"], 2)

    if norm_name in microtubule:
        row["target_pathway"] = "Microtubule dynamics"
        row["drug_class"] = "Microtubule-targeting chemotherapy"
        row["microtubule_relevance"] = 3 if norm_name in {"docetaxel", "paclitaxel"} else 2
        row["breast_cancer_relevance"] = max(row["breast_cancer_relevance"], 2)

    if norm_name in dna_damage:
        if row["target_pathway"] == "Unknown":
            row["target_pathway"] = "DNA damage / cytotoxic stress"
        if row["drug_class"] == "Unclassified":
            row["drug_class"] = "Cytotoxic chemotherapy"
        row["dna_damage_relevance"] = max(row["dna_damage_relevance"], 1)
        row["breast_cancer_relevance"] = max(row["breast_cancer_relevance"], 1)

    if norm_name in pi3k_mtor:
        row["pi3k_mtor_relevance"] = max(row["pi3k_mtor_relevance"], 2)
        row["breast_cancer_relevance"] = max(row["breast_cancer_relevance"], 1)

    if norm_name in her2:
        row["her2_relevance"] = max(row["her2_relevance"], 1)

    if norm_name in hsp90:
        row["hsp90_relevance"] = max(row["hsp90_relevance"], 2)
        row["her2_relevance"] = max(row["her2_relevance"], 1)

    if norm_name in luminal:
        row["luminal_relevance"] = max(row["luminal_relevance"], 2)

    if norm_name in tnbc:
        row["tnbc_relevance"] = max(row["tnbc_relevance"], 2)

    if row["clinical_status"] == "tool":
        row["tool_compound"] = 1
        row["trial_mappable"] = 0
        row["breast_cancer_relevance"] = 0

    if row["clinical_status"] == "approved_nononcology":
        row["trial_mappable"] = 0

    return row


def build_drug_metadata(compounds: list[str]) -> pd.DataFrame:
    """Build a compact curated metadata table for the supplied compounds."""
    rows = []
    for drug_name in sorted(set(compounds)):
        norm_name = normalize_drug_name(drug_name)
        row = _apply_heuristics(_base_row(drug_name))
        overrides = CURATED_OVERRIDES.get(norm_name, {})
        row.update(overrides)
        row["clinical_status"] = overrides.get("clinical_status", row["clinical_status"])
        row["status_score"] = STATUS_SCORES[row["clinical_status"]]
        row["tool_compound"] = int(overrides.get("tool_compound", row["tool_compound"]))
        row["standard_of_care_breast"] = int(
            overrides.get("standard_of_care_breast", row["standard_of_care_breast"])
        )
        row["trial_mappable"] = int(overrides.get("trial_mappable", row["trial_mappable"]))
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_cols = ["tool_compound", "breast_cancer_relevance", "drug_name"]
    ascending = [True, False, True]
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def ensure_drug_metadata(
    compounds: list[str],
    output_path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Create or refresh the breast-cancer metadata TSV for the current compound set."""
    df = build_drug_metadata(compounds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    return df


def load_drug_metadata(
    compounds: list[str] | None = None,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Load metadata if present, otherwise build it from the provided compounds."""
    if metadata_path.exists():
        df = pd.read_csv(metadata_path, sep="\t")
        if compounds is None:
            return df

        missing = sorted(set(compounds) - set(df["drug_name"]))
        if not missing:
            return df[df["drug_name"].isin(compounds)].reset_index(drop=True)

        refreshed = ensure_drug_metadata(sorted(set(df["drug_name"]).union(compounds)), metadata_path)
        return refreshed[refreshed["drug_name"].isin(compounds)].reset_index(drop=True)

    if compounds is None:
        raise FileNotFoundError(f"No metadata file found at {metadata_path}")

    return ensure_drug_metadata(compounds, metadata_path)
