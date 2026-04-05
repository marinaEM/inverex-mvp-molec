"""
Agent C Full Pipeline Runner
=============================
Runs ALL stages sequentially:
1. Main pretraining (medium, hybrid, all objectives)
2. LODO evaluation
3. Ablation: input encoding (raw, rank, hybrid)
4. Ablation: objectives (mgp-only, multi-no-dav, multi+dav)
5. Ablation: model size (medium vs full)
6. Ablation: frozen vs finetuned
7. Embedding analysis
8. Comparison vs baselines

Usage: pixi run python src/models/foundation_full/run_all.py
"""

from __future__ import annotations

import os, sys, time, logging, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results" / "agent_c_full"
RESULTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS / "run_all.log"),
    ],
)

logger = logging.getLogger(__name__)

from models.foundation_full.evaluate import run_full_pipeline

if __name__ == "__main__":
    t0 = time.time()
    logger.info("Starting Agent C Full Pipeline")
    results = run_full_pipeline()
    total = time.time() - t0
    logger.info("Full pipeline complete in %.1f hours", total / 3600)
