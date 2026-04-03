"""
Drug fingerprint computation (SMILES → ECFP4).

Strategy:
    scTherapy used PubChem + RDKit to generate ECFP4 fingerprints from SMILES.
    We replicate this: for each compound in the LINCS/PharmacoDB overlap,
    fetch SMILES from PubChem PUG-REST, then compute ECFP4 with RDKit.

    ECFP4 = Extended Connectivity Fingerprint with radius 2, bit vector.
    This captures local chemical environments around each atom.
"""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import DATA_CACHE, ECFP_NBITS, ECFP_RADIUS

logger = logging.getLogger(__name__)

PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_smiles_from_pubchem(
    compound_name: str,
    timeout: int = 10,
) -> Optional[str]:
    """
    Fetch canonical SMILES for a compound from PubChem by name.

    Returns SMILES string or None if not found.
    """
    url = f"{PUBCHEM_REST}/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                # PubChem may return CanonicalSMILES or ConnectivitySMILES
                return (
                    props[0].get("CanonicalSMILES")
                    or props[0].get("ConnectivitySMILES")
                )
        return None
    except (requests.RequestException, ValueError):
        return None


def batch_fetch_smiles(
    compound_names: list[str],
    cache_dir: Path = DATA_CACHE,
    delay: float = 0.25,
) -> dict[str, str]:
    """
    Fetch SMILES for a list of compound names, with caching.

    Returns dict: compound_name → SMILES
    """
    cache_path = cache_dir / "compound_smiles_cache.parquet"

    # Load existing cache
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        smiles_map = dict(zip(cached["compound_name"], cached["smiles"]))
    else:
        smiles_map = {}

    # Find what's missing
    missing = [name for name in compound_names if name not in smiles_map]
    if missing:
        logger.info(f"Fetching SMILES for {len(missing)} compounds from PubChem...")
        for i, name in enumerate(missing):
            smiles = fetch_smiles_from_pubchem(name)
            if smiles:
                smiles_map[name] = smiles
            if (i + 1) % 50 == 0:
                logger.info(f"  ... fetched {i + 1}/{len(missing)}")
            time.sleep(delay)  # rate limiting

        # Update cache
        cache_df = pd.DataFrame([
            {"compound_name": k, "smiles": v} for k, v in smiles_map.items()
        ])
        cache_df.to_parquet(cache_path, index=False)
        logger.info(
            f"SMILES cache updated: {len(smiles_map)} compounds "
            f"({len(missing)} newly fetched)"
        )

    return smiles_map


def smiles_to_ecfp4(
    smiles: str,
    radius: int = ECFP_RADIUS,
    n_bits: int = ECFP_NBITS,
) -> Optional[np.ndarray]:
    """
    Convert a SMILES string to an ECFP4 bit vector using RDKit.

    Returns numpy array of shape (n_bits,) with 0/1 values, or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        return arr

    except ImportError:
        logger.error("RDKit not installed. Install with: pip install rdkit-pypi")
        raise


def build_fingerprint_matrix(
    compound_names: list[str],
    cache_dir: Path = DATA_CACHE,
) -> pd.DataFrame:
    """
    Build a fingerprint matrix for a list of compounds.

    Returns DataFrame: rows = compounds, columns = ECFP4 bit indices.
    Compounds without valid SMILES are excluded.
    """
    cache_path = cache_dir / "drug_fingerprints.parquet"
    if cache_path.exists():
        logger.info("Loading cached drug fingerprints...")
        cached = pd.read_parquet(cache_path)
        # Check if all requested compounds are cached
        if set(compound_names).issubset(set(cached["compound_name"])):
            return cached[cached["compound_name"].isin(compound_names)]

    # Fetch SMILES
    smiles_map = batch_fetch_smiles(compound_names, cache_dir)

    # Compute fingerprints
    rows = []
    for name in compound_names:
        smiles = smiles_map.get(name)
        if not smiles:
            continue
        fp = smiles_to_ecfp4(smiles)
        if fp is None:
            continue
        row = {"compound_name": name, "smiles": smiles}
        for i, bit in enumerate(fp):
            row[f"ecfp_{i}"] = bit
        rows.append(row)

    if not rows:
        logger.warning("No fingerprints computed. Check compound names and RDKit.")
        return pd.DataFrame()

    fp_df = pd.DataFrame(rows)
    fp_df.to_parquet(cache_path, index=False)
    logger.info(f"Built fingerprint matrix: {fp_df.shape}")
    return fp_df


def build_demo_fingerprints(
    compound_names: list[str],
    cache_dir: Path = DATA_CACHE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build synthetic fingerprints for demo/testing when RDKit/PubChem
    aren't available.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name in compound_names:
        row = {"compound_name": name, "smiles": "DEMO"}
        # Sparse random bits (typical ECFP density ~5-15%)
        bits = rng.random(ECFP_NBITS) < 0.1
        for i, bit in enumerate(bits):
            row[f"ecfp_{i}"] = int(bit)
        rows.append(row)

    fp_df = pd.DataFrame(rows)
    cache_path = cache_dir / "drug_fingerprints.parquet"
    fp_df.to_parquet(cache_path, index=False)
    logger.info(f"Built demo fingerprints: {fp_df.shape}")
    return fp_df
