"""
SHAP explanations for the INVEREX patient LightGBM model.

Produces:
  - Global feature importance ranking (mean |SHAP| per feature)
  - SHAP beeswarm and bar plots
  - Feature category aggregation (gene / ssGSEA / clinical)
  - Per-patient local explanations for drug ranking rationale
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def categorize_feature(name: str) -> str:
    """Assign a feature to its category."""
    if name.startswith("ssgsea_"):
        return "ssGSEA_pathway"
    elif name.startswith("chembert_"):
        return "ChemBERTa_drug"
    elif name.startswith("progeny_"):
        return "PROGENy_pathway"
    elif name.startswith("target_"):
        return "drug_target"
    elif name.startswith("gene_"):
        return "gene_expression"
    elif name in (
        "ER_status", "HER2_status", "PR_status", "age", "grade",
        "er_status", "her2_status", "pr_status",
    ):
        return "clinical"
    elif name == "dose" or name == "log_dose_um":
        return "dose"
    else:
        return "gene_expression"


def compute_global_shap(
    model,
    X_train: pd.DataFrame,
    feature_names: list[str],
    output_dir: str = "results/shap/",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute global SHAP values for the trained LightGBM model.

    Returns:
        (importance_df, shap_values_array)
    """
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)

    # Subsample if large
    if len(X_train) > 2000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), 2000, replace=False)
        X_sub = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
    else:
        X_sub = X_train

    sv = explainer.shap_values(X_sub)
    # For binary classification, shap_values may be [class_0, class_1]
    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]

    # --- Global importance ranking ---
    mean_abs_shap = np.abs(sv).mean(axis=0)
    mean_shap = sv.mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap_direction": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df["category"] = importance_df["feature"].apply(categorize_feature)
    importance_df["direction"] = importance_df["mean_shap_direction"].apply(
        lambda x: "response" if x > 0 else "resistance"
    )
    importance_df.to_csv(
        f"{output_dir}/global_feature_importance.tsv", sep="\t", index=False
    )

    # --- Beeswarm plot ---
    X_sub_df = pd.DataFrame(
        X_sub if isinstance(X_sub, np.ndarray) else X_sub.values,
        columns=feature_names,
    )
    try:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            sv, X_sub_df, feature_names=feature_names, show=False, max_display=30
        )
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/shap_beeswarm_top30.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        logger.info("Saved beeswarm plot")
    except Exception as e:
        logger.warning(f"Beeswarm plot failed: {e}")

    # --- Bar plot ---
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv, X_sub_df,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=30,
        )
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/shap_bar_top30.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        logger.info("Saved bar plot")
    except Exception as e:
        logger.warning(f"Bar plot failed: {e}")

    # --- Category-level aggregation ---
    category_importance = (
        importance_df.groupby("category")["mean_abs_shap"]
        .agg(["sum", "mean", "count"])
        .sort_values("sum", ascending=False)
    )
    category_importance.to_csv(f"{output_dir}/shap_by_category.tsv", sep="\t")

    logger.info(f"Top 15 features by SHAP importance:")
    for _, r in importance_df.head(15).iterrows():
        logger.info(
            f"  {r['feature']:30s}  |SHAP|={r['mean_abs_shap']:.4f}  "
            f"({r['category']}, {r['direction']})"
        )

    logger.info(f"Importance by category:")
    for cat, row in category_importance.iterrows():
        logger.info(f"  {cat:20s}  sum={row['sum']:.3f}  n={int(row['count'])}")

    return importance_df, sv


def compute_interaction_shap(
    model,
    X_train: pd.DataFrame,
    feature_names: list[str],
    output_dir: str = "results/shap/",
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute SHAP interaction values for top feature pairs.
    WARNING: O(n_features^2), slow. Uses a subsample.
    """
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)

    if len(X_train) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), max_samples, replace=False)
        X_sub = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
    else:
        X_sub = X_train

    logger.info(f"Computing SHAP interactions on {len(X_sub)} samples...")
    iv = explainer.shap_interaction_values(X_sub)
    if isinstance(iv, list) and len(iv) == 2:
        iv = iv[1]

    mean_interaction = np.abs(iv).mean(axis=0)
    np.fill_diagonal(mean_interaction, 0)

    n_feat = mean_interaction.shape[0]
    pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            pairs.append({
                "feature_1": feature_names[i],
                "feature_2": feature_names[j],
                "mean_abs_interaction": mean_interaction[i, j],
            })

    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values("mean_abs_interaction", ascending=False)
    )
    pairs_df.head(30).to_csv(
        f"{output_dir}/shap_top_interactions.tsv", sep="\t", index=False
    )

    logger.info("Top 10 feature interactions:")
    for _, r in pairs_df.head(10).iterrows():
        logger.info(
            f"  {r['feature_1']:25s} × {r['feature_2']:25s}  "
            f"interaction={r['mean_abs_interaction']:.4f}"
        )

    return pairs_df


def explain_single_sample(
    model, explainer, sample_features: np.ndarray, feature_names: list[str]
) -> dict:
    """SHAP explanation for a single (patient, drug) prediction."""
    sample_2d = sample_features.reshape(1, -1)
    sv = explainer.shap_values(sample_2d)
    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]
    sv = sv[0]

    pred = model.predict_proba(sample_2d)[:, 1][0]

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "abs_shap": np.abs(sv),
        "feature_value": sample_features,
    }).sort_values("abs_shap", ascending=False)

    top_positive = (
        feat_df[feat_df["shap_value"] > 0]
        .head(5)[["feature", "shap_value", "feature_value"]]
        .to_dict("records")
    )
    top_negative = (
        feat_df[feat_df["shap_value"] < 0]
        .head(3)[["feature", "shap_value", "feature_value"]]
        .to_dict("records")
    )

    return {
        "predicted_response_prob": pred,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
    }


def generate_patient_report(
    patient_id: str,
    rankings: list[dict],
    output_dir: str = "results/shap/",
) -> str:
    """Generate human-readable report for a single patient's drug ranking."""
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        f"Patient: {patient_id}",
        "=" * 55,
        "DISCLAIMER: This is a research model, NOT a clinical tool.",
        "",
    ]

    for r in rankings[:10]:
        lines.append(
            f"#{r['rank']}  {r['drug']}  "
            f"(predicted response: {r['predicted_response_prob']*100:.1f}%)"
        )

        pos_strs = []
        for f in r.get("top_positive_features", [])[:3]:
            pos_strs.append(f"{f['feature']} (+{f['shap_value']:.3f})")
        if pos_strs:
            lines.append(f"    WHY: {', '.join(pos_strs)}")

        neg_strs = []
        for f in r.get("top_negative_features", [])[:2]:
            neg_strs.append(f"{f['feature']} ({f['shap_value']:.3f})")
        if neg_strs:
            lines.append(f"    RISK: {', '.join(neg_strs)}")

        lines.append("")

    report_text = "\n".join(lines)

    safe_id = patient_id.replace("/", "_").replace(" ", "_")
    with open(f"{output_dir}/patient_report_{safe_id}.txt", "w") as fh:
        fh.write(report_text)

    return report_text
