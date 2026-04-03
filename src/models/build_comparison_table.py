"""
Build a unified comparison table merging results from all validation
steps of the INVEREX patient validation pipeline.

Sources:
  1. Cell-line model validation on CTR-DB patients
     (results/ctrdb_validation_results.csv)
  2. LINCS vs CDS-DB signature comparison
     (results/lincs_vs_cdsdb_comparison.csv)
  3. Patient model LODO-CV results
     (results/patient_model_lodo_results.csv)
  4. Original cell-line model metrics
     (results/lightgbm_metrics.json)

Output: results/model_comparison_table.csv
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS

logger = logging.getLogger(__name__)


def build_comparison_table(
    results_dir: Path = RESULTS,
    output_path: Path = RESULTS / "model_comparison_table.csv",
) -> pd.DataFrame:
    """
    Merge all validation results into a single comparison table.

    Columns:
        model, evaluation, dataset, n_patients, metric_name, metric_value, notes
    """
    rows = []

    # 1. Original cell-line model metrics
    rows.extend(_load_cellline_model_metrics(results_dir))

    # 2. Cell-line model on CTR-DB patients
    rows.extend(_load_ctrdb_validation(results_dir))

    # 3. LINCS vs CDS-DB comparison
    rows.extend(_load_signature_comparison(results_dir))

    # 4. Patient model LODO-CV
    rows.extend(_load_patient_model_lodo(results_dir))

    if not rows:
        logger.warning("No results found to merge")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort for readability
    model_order = {
        "CellLine-LightGBM": 0,
        "CellLine-LightGBM-on-patients": 1,
        "LINCS-reversal": 2,
        "CDS-DB-reversal": 3,
        "Patient-LightGBM-LODO": 4,
    }
    df["_sort"] = df["model"].map(model_order).fillna(99)
    df = df.sort_values(["_sort", "dataset", "metric_name"]).drop(
        columns="_sort"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Comparison table saved to {output_path} ({len(df)} rows)")

    return df


def _load_cellline_model_metrics(results_dir: Path) -> list[dict]:
    """Load original cell-line model CV metrics."""
    rows = []
    metrics_path = results_dir / "lightgbm_metrics.json"
    if not metrics_path.exists():
        logger.info("No cell-line model metrics found")
        return rows

    with open(metrics_path) as f:
        metrics = json.load(f)

    rows.append({
        "model": "CellLine-LightGBM",
        "evaluation": "10-fold CV (cell lines)",
        "dataset": "LINCS x GDSC2",
        "n_patients": metrics.get("n_samples", 0),
        "metric_name": "RMSE",
        "metric_value": round(metrics.get("cv_rmse_mean", 0), 4),
        "notes": f"std={metrics.get('cv_rmse_std', 0):.4f}",
    })

    return rows


def _load_ctrdb_validation(results_dir: Path) -> list[dict]:
    """Load cell-line model validation on CTR-DB patient data."""
    rows = []
    path = results_dir / "ctrdb_validation_results.csv"
    if not path.exists():
        logger.info("No CTR-DB validation results found")
        return rows

    df = pd.read_csv(path)
    for _, r in df.iterrows():
        base = {
            "model": "CellLine-LightGBM-on-patients",
            "evaluation": "CTR-DB patient validation",
            "dataset": r.get("geo_id", "unknown"),
            "n_patients": r.get("n_patients", 0),
        }
        rows.append({
            **base,
            "metric_name": "AUC",
            "metric_value": r.get("auc", np.nan),
            "notes": f"drug={r.get('drug', 'unknown')}",
        })
        rows.append({
            **base,
            "metric_name": "Wilcoxon p-value",
            "metric_value": r.get("wilcoxon_pvalue", np.nan),
            "notes": "",
        })
        rows.append({
            **base,
            "metric_name": "Cohen's d",
            "metric_value": r.get("cohens_d", np.nan),
            "notes": (
                f"mean_R={r.get('mean_score_responders', 0):.2f}, "
                f"mean_NR={r.get('mean_score_nonresponders', 0):.2f}"
            ),
        })

    # Summary row
    if len(df) > 0:
        rows.append({
            "model": "CellLine-LightGBM-on-patients",
            "evaluation": "CTR-DB patient validation (mean)",
            "dataset": "ALL",
            "n_patients": int(df["n_patients"].sum()),
            "metric_name": "Mean AUC",
            "metric_value": round(df["auc"].mean(), 4),
            "notes": f"{len(df)} datasets, std={df['auc'].std():.4f}",
        })

    return rows


def _load_signature_comparison(results_dir: Path) -> list[dict]:
    """Load LINCS vs CDS-DB comparison results."""
    rows = []
    path = results_dir / "lincs_vs_cdsdb_comparison.csv"
    if not path.exists():
        logger.info("No signature comparison results found")
        return rows

    df = pd.read_csv(path)
    for _, r in df.iterrows():
        source = r.get("source", "")
        if source == "LINCS":
            model = "LINCS-reversal"
        elif source == "CDS-DB":
            model = "CDS-DB-reversal"
        else:
            model = f"Signature-{source}"

        if pd.isna(r.get("auc")):
            # Summary/info row
            rows.append({
                "model": model,
                "evaluation": "Signature comparison",
                "dataset": r.get("geo_id", ""),
                "n_patients": r.get("n_patients", 0),
                "metric_name": "info",
                "metric_value": np.nan,
                "notes": r.get("note", ""),
            })
        else:
            rows.append({
                "model": model,
                "evaluation": "Reversal score discrimination",
                "dataset": r.get("geo_id", ""),
                "n_patients": r.get("n_patients", 0),
                "metric_name": "AUC",
                "metric_value": r.get("auc", np.nan),
                "notes": f"drug={r.get('drug', '')}, p={r.get('wilcoxon_p', '')}",
            })

    return rows


def _load_patient_model_lodo(results_dir: Path) -> list[dict]:
    """Load patient model LODO-CV results."""
    rows = []
    path = results_dir / "patient_model_lodo_results.csv"
    if not path.exists():
        logger.info("No patient model LODO results found")
        return rows

    df = pd.read_csv(path)
    for _, r in df.iterrows():
        is_mean = r.get("held_out_dataset") == "MEAN"
        dataset = "MEAN" if is_mean else r.get("held_out_dataset", "")
        evaluation = (
            "LODO-CV (mean)" if is_mean else "LODO-CV (per-fold)"
        )

        rows.append({
            "model": "Patient-LightGBM-LODO",
            "evaluation": evaluation,
            "dataset": dataset,
            "n_patients": r.get("n_test", 0),
            "metric_name": "AUC",
            "metric_value": r.get("auc", np.nan),
            "notes": (
                f"balanced_acc={r.get('balanced_accuracy', 0):.3f}, "
                f"sens={r.get('sensitivity', 0):.3f}, "
                f"spec={r.get('specificity', 0):.3f}"
            ),
        })

    return rows


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    table = build_comparison_table()
    if len(table) > 0:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TABLE")
        print("=" * 80)
        print(table.to_string(index=False))
    else:
        print("\nNo comparison data available.")
