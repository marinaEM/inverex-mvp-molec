"""
Shared utilities for data ingestion.
"""
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 8192,
    timeout: int = 60,
    max_retries: int = 3,
) -> Path:
    """
    Download a file with progress bar and retry logic.
    Returns the destination path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {url} → {dest} (attempt {attempt})")
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

            logger.info(f"Downloaded {dest.name} ({dest.stat().st_size:,} bytes)")
            return dest

        except (requests.RequestException, IOError) as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise

    return dest  # unreachable, but makes type checker happy


def fetch_json(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 30,
) -> dict:
    """Fetch JSON from a URL with error handling."""
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
