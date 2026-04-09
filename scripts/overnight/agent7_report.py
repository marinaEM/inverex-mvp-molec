#!/usr/bin/env python
"""
Agent 7: Final Report and Morning Takeaway
===========================================

Aggregates results from all 6 agents into:
  - summaries/all_runs.csv             (already built incrementally)
  - summaries/best_by_method.csv
  - summaries/best_by_stratum.csv
  - reports/final_overnight_summary.md
  - reports/morning_takeaway.txt
  - plots/model_comparison.png
"""

import os, sys, json, time, warnings
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.overnight._shared import BENCH, setup_logger

AGENT = "agent7_report"
log = setup_logger(AGENT)

log.info("=" * 70)
log.info("AGENT 7: Final Report Aggregator")
log.info("=" * 70)


# =========================================================================
# Load all results
# =========================================================================
all_runs_path = BENCH / "summaries" / "all_runs.csv"
if not all_runs_path.exists():
    log.warning(f"No all_runs.csv at {all_runs_path}")
    runs_df = pd.DataFrame()
else:
    runs_df = pd.read_csv(all_runs_path)
    log.info(f"Loaded {len(runs_df)} runs from all_runs.csv")

# Diagnostics summary
diag_path = BENCH / "diagnostics" / "diagnostics_summary.json"
diagnostics = {}
if diag_path.exists():
    with open(diag_path) as f:
        diagnostics = json.load(f)
    log.info(f"Loaded diagnostics: {len(diagnostics)} metrics")

# Stratified summary
strat_path = BENCH / "summaries" / "stratified_summary.csv"
strat_df = pd.read_csv(strat_path) if strat_path.exists() else pd.DataFrame()
log.info(f"Loaded {len(strat_df)} stratified runs")


# =========================================================================
# Best by method
# =========================================================================
log.info("\n=== Building best_by_method ===")
best_by_method = pd.DataFrame()
if not runs_df.empty and "agent" in runs_df.columns:
    grouped = runs_df.groupby("agent").apply(
        lambda g: g.sort_values("mean_auroc", ascending=False).iloc[0]
    ).reset_index(drop=True)
    best_by_method = grouped.sort_values("mean_auroc", ascending=False)
    best_by_method.to_csv(BENCH / "summaries" / "best_by_method.csv", index=False)
    log.info(f"Best by method ({len(best_by_method)} agents):")
    for _, r in best_by_method.iterrows():
        log.info(f"  {r['agent']:25s}  AUROC={r.get('mean_auroc',0):.4f}  "
                 f"MCC={r.get('mean_mcc',0):.4f}  run={r.get('run_id','?')}")


# =========================================================================
# Best by stratum
# =========================================================================
if not strat_df.empty:
    best_by_stratum = strat_df.sort_values("mean_auroc", ascending=False).head(20)
    best_by_stratum.to_csv(BENCH / "summaries" / "best_by_stratum.csv", index=False)


# =========================================================================
# Plot: model comparison
# =========================================================================
if not best_by_method.empty:
    log.info("\nGenerating model comparison plot...")
    fig, ax = plt.subplots(figsize=(11, 6))
    bm = best_by_method.copy()
    bm["label"] = bm["agent"].str.replace("agent[0-9]+_", "", regex=True)
    bars = ax.barh(bm["label"], bm["mean_auroc"], color="steelblue", edgecolor="black")
    ax.axvline(0.5, color="grey", linestyle=":", label="Random")
    ax.axvline(0.602, color="darkred", linestyle="--", label="Production baseline (0.602)")
    ax.set_xlabel("Mean LODO AUROC (best config per method)")
    ax.set_title("Best AUROC by Method")
    ax.set_xlim(0.45, max(0.75, bm["mean_auroc"].max() + 0.02))
    ax.legend(loc="lower right")
    for bar, val in zip(bars, bm["mean_auroc"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(BENCH / "plots" / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved model_comparison.png")


# =========================================================================
# Stratified plot
# =========================================================================
if not strat_df.empty:
    fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(strat_df))))
    sd = strat_df.sort_values("mean_auroc")
    sd["label"] = sd["stratum_name"].astype(str) + "=" + sd["stratum_value"].astype(str)
    colors = sd["stratum_name"].map({"drug_class": "steelblue", "cancer_type": "darkorange", "technology": "seagreen"})
    ax.barh(sd["label"], sd["mean_auroc"], color=colors)
    ax.axvline(0.602, color="darkred", linestyle="--", label="Pooled baseline (0.602)")
    ax.set_xlabel("Mean within-stratum LODO AUROC")
    ax.set_title("Stratified Modeling Performance")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(BENCH / "plots" / "stratified_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()


# =========================================================================
# Final markdown report
# =========================================================================
log.info("\nWriting final markdown report...")

BASELINE_AUROC = 0.602

def gain_label(value, baseline=BASELINE_AUROC):
    delta = value - baseline
    if delta < -0.005: return f"WORSE ({delta:+.3f})"
    if abs(delta) < 0.005: return f"no meaningful change ({delta:+.3f})"
    if delta < 0.01: return f"suggestive gain ({delta:+.3f})"
    if delta < 0.03: return f"small gain ({delta:+.3f})"
    return f"meaningful gain ({delta:+.3f})"


report_lines = []
report_lines.append("# INVEREX Overnight Model vs Signal Benchmark — Final Report\n")
report_lines.append(f"Generated: {datetime.now().isoformat()}\n")
report_lines.append("---\n")

# Executive summary
report_lines.append("## 1. Executive Summary\n")
if not best_by_method.empty:
    overall_best = best_by_method.iloc[0]
    report_lines.append(f"- Production baseline: **AUROC = {BASELINE_AUROC:.3f}**, MCC ≈ 0.157 "
                         f"(LightGBM, depth 5, 918 genes + singscore + REO).")
    report_lines.append(f"- Best single configuration across all agents: "
                         f"**{overall_best.get('agent','?')}/{overall_best.get('run_id','?')} "
                         f"AUROC = {overall_best.get('mean_auroc',0):.4f}, "
                         f"MCC = {overall_best.get('mean_mcc',0):.4f}** "
                         f"({gain_label(overall_best.get('mean_auroc',0))}).\n")
else:
    report_lines.append("- No runs aggregated. Check raw_metrics/.\n")

if diagnostics:
    sil_resp = diagnostics.get("silhouette_response", float("nan"))
    sil_ds = diagnostics.get("silhouette_dataset", float("nan"))
    sil_tech = diagnostics.get("silhouette_technology", float("nan"))
    n_robust = diagnostics.get("n_robust_consistent_genes", 0)
    report_lines.append("- Diagnostics:")
    report_lines.append(f"  - Silhouette by **response**: `{sil_resp:+.3f}` "
                         f"({'no separation' if abs(sil_resp) < 0.05 else 'weak separation'}).")
    report_lines.append(f"  - Silhouette by **dataset**:  `{sil_ds:+.3f}`.")
    report_lines.append(f"  - Silhouette by **technology**: `{sil_tech:+.3f}`.")
    report_lines.append(f"  - Genes with consistent direction across ≥20 datasets at ≥75% concordance: **{n_robust}**.")
    report_lines.append(f"  - Significant pathways at p<0.05 / Bonferroni: "
                         f"**{diagnostics.get('n_significant_pathways_p05',0)} / "
                         f"{diagnostics.get('n_significant_pathways_bonferroni',0)}**.")
    report_lines.append("")

# What was run
report_lines.append("## 2. What Was Run\n")
if not runs_df.empty:
    n_per_agent = runs_df.groupby("agent").size().to_dict()
    for agent, n in sorted(n_per_agent.items()):
        report_lines.append(f"- `{agent}`: {n} run(s)")
    report_lines.append("")

# Main benchmark results
report_lines.append("## 3. Main Benchmark Results\n")
if not best_by_method.empty:
    report_lines.append("| Method | Best AUROC | AUPRC | MCC | Bal Acc | vs Baseline |")
    report_lines.append("|---|---|---|---|---|---|")
    for _, r in best_by_method.iterrows():
        report_lines.append(
            f"| `{r.get('agent','?')}` | "
            f"{r.get('mean_auroc',0):.4f} | "
            f"{r.get('mean_auprc',0):.4f} | "
            f"{r.get('mean_mcc',0):.4f} | "
            f"{r.get('mean_bal_acc',0):.4f} | "
            f"{gain_label(r.get('mean_auroc',0))} |"
        )
    report_lines.append("")

# Specific questions
report_lines.append("## 4. Did Deeper LightGBM Help?\n")
lgbm_runs = runs_df[runs_df["agent"].str.contains("lgbm", case=False, na=False)] if not runs_df.empty else pd.DataFrame()
if not lgbm_runs.empty:
    best_lgbm = lgbm_runs["mean_auroc"].max()
    delta = best_lgbm - BASELINE_AUROC
    report_lines.append(f"- Best LightGBM (any depth): AUROC = **{best_lgbm:.4f}**")
    report_lines.append(f"- Delta vs depth-5 baseline: **{delta:+.4f}** ({gain_label(best_lgbm)})")
    report_lines.append("")
else:
    report_lines.append("- No LightGBM sweep results available.\n")

report_lines.append("## 5. Did CatBoost Help?\n")
cat_runs = runs_df[runs_df["agent"].str.contains("catboost", case=False, na=False)] if not runs_df.empty else pd.DataFrame()
if not cat_runs.empty:
    best_cat = cat_runs["mean_auroc"].max()
    delta = best_cat - BASELINE_AUROC
    report_lines.append(f"- Best CatBoost: AUROC = **{best_cat:.4f}**")
    report_lines.append(f"- Delta vs LightGBM baseline: **{delta:+.4f}** ({gain_label(best_cat)})")
    report_lines.append("")
else:
    report_lines.append("- No CatBoost results available.\n")

report_lines.append("## 6. Did Pathway Features Help?\n")
path_runs = runs_df[runs_df["agent"].str.contains("pathway", case=False, na=False)] if not runs_df.empty else pd.DataFrame()
if not path_runs.empty:
    report_lines.append("| Feature set | AUROC | MCC |")
    report_lines.append("|---|---|---|")
    for _, r in path_runs.sort_values("mean_auroc", ascending=False).iterrows():
        report_lines.append(f"| {r.get('run_id','?')} | {r.get('mean_auroc',0):.4f} | {r.get('mean_mcc',0):.4f} |")
    report_lines.append("")
else:
    report_lines.append("- No pathway comparison results available.\n")

report_lines.append("## 7. Did Stratification Help?\n")
if not strat_df.empty:
    above = strat_df[strat_df["mean_auroc"] > BASELINE_AUROC + 0.02]
    report_lines.append(f"- {len(strat_df)} strata evaluated.")
    report_lines.append(f"- {len(above)} strata exceed pooled baseline by >0.02 AUROC.")
    if len(above) > 0:
        report_lines.append("\n**Strata where stratification helps:**\n")
        report_lines.append("| Stratum | Value | n_folds | AUROC | MCC |")
        report_lines.append("|---|---|---|---|---|")
        for _, r in above.sort_values("mean_auroc", ascending=False).iterrows():
            report_lines.append(
                f"| {r['stratum_name']} | {r['stratum_value']} | {int(r.get('n_folds',0))} | "
                f"{r.get('mean_auroc',0):.4f} | {r.get('mean_mcc',0):.4f} |"
            )
    report_lines.append("")
else:
    report_lines.append("- No stratified results available.\n")

# Diagnostics
report_lines.append("## 8. Signal and Heterogeneity Diagnostics\n")
if diagnostics:
    report_lines.append("```")
    for k, v in diagnostics.items():
        report_lines.append(f"{k}: {v}")
    report_lines.append("```\n")

    sil_resp = diagnostics.get("silhouette_response", 0)
    sil_ds = diagnostics.get("silhouette_dataset", 0)
    n_robust = diagnostics.get("n_robust_consistent_genes", 0)
    n_sig_path = diagnostics.get("n_significant_pathways_bonferroni", 0)
    enrichment = diagnostics.get("mean_per_dataset_sig_enrichment", 0)

    report_lines.append("**Interpretation:**\n")
    if abs(sil_resp) < 0.05 and abs(sil_ds) > 0.1:
        report_lines.append("- PCA structure is dominated by **dataset/technology**, not response.")
    if n_robust < 5:
        report_lines.append(f"- Only **{n_robust} genes** show consistent direction across ≥20 datasets — "
                             f"response signal is **highly heterogeneous**.")
    if n_sig_path < 3:
        report_lines.append(f"- Only **{n_sig_path} pathways** are Bonferroni-significant in pooled analysis.")
    if enrichment < 1.5:
        report_lines.append(f"- Per-dataset significance enrichment vs random: **{enrichment:.2f}x** "
                             f"(near 1 = no real signal beyond noise).")

    report_lines.append("")

# Bottom line
report_lines.append("## 10. Bottom-Line Conclusion\n")

if not best_by_method.empty:
    overall_best = best_by_method["mean_auroc"].max()
    delta_overall = overall_best - BASELINE_AUROC

    if abs(delta_overall) < 0.01:
        report_lines.append(
            f"**Performance appears signal-limited rather than model-limited.** "
            f"All tabular learners (LightGBM, CatBoost, XGBoost, RF, ElasticNet) converge to "
            f"AUROC ≈ {BASELINE_AUROC:.2f} regardless of architecture or capacity. "
            f"The best run is only {delta_overall:+.4f} above baseline."
        )
    elif delta_overall > 0.03:
        report_lines.append(
            f"**Meaningful improvement achievable.** Best run exceeds baseline by "
            f"{delta_overall:+.4f} AUROC. Investigate the winning configuration."
        )
    else:
        report_lines.append(
            f"**Small but real improvement possible.** Best run is {delta_overall:+.4f} above baseline — "
            f"worth incorporating but not transformative."
        )
    report_lines.append("")

# Recommendations
report_lines.append("## 11. Recommended Next Steps\n")

if diagnostics:
    n_robust = diagnostics.get("n_robust_consistent_genes", 0)
    sil_resp = diagnostics.get("silhouette_response", 0)
    if n_robust < 5 and abs(sil_resp) < 0.05:
        report_lines.append(
            "1. **Stop chasing model architecture.** Diagnostics show no consistent response signal "
            "across cohorts. The ~0.60 ceiling is in the data, not the learner.\n"
            "2. **Curate a subset of internally consistent datasets** (single drug, single endpoint, "
            "≥200 patients each) and retrain on those. Pan-cohort training is being defeated by "
            "endpoint heterogeneity.\n"
            "3. **Reframe the prediction task** as subtype-conditional response (e.g., HER2+ patients "
            "treated with trastuzumab) rather than pan-cohort response.\n"
            "4. **Investigate stratified models** in any subset where signal is stronger (see section 7)."
        )
    else:
        report_lines.append(
            "1. Investigate which strata or feature sets gave meaningful gains.\n"
            "2. Validate the best configuration with bootstrap confidence intervals.\n"
            "3. Test the best configuration on cross-technology holdout (e.g., RNA-seq only)."
        )

# Failures
report_lines.append("\n## 9. Failures and Caveats\n")
if not runs_df.empty:
    report_lines.append(f"- Total runs aggregated: {len(runs_df)}")
report_lines.append("- All evaluations use LODO with frozen MCC-optimal threshold from training fold.")
report_lines.append("- ssGSEA scores are computed once on raw rank-normalized expression "
                     "(per-sample, no cross-sample fitting).")
report_lines.append("- Excluded 2 datasets from training: GSE9782 (102 genes, targeted panel) "
                     "and GSE61676 (BeadArray, 0 L1000 gene matches).")

# Write report
report_path = BENCH / "reports" / "final_overnight_summary.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
log.info(f"Saved {report_path}")


# =========================================================================
# Morning takeaway (concise)
# =========================================================================
takeaway_lines = []
takeaway_lines.append("INVEREX OVERNIGHT BENCHMARK — MORNING TAKEAWAY")
takeaway_lines.append("=" * 50)
takeaway_lines.append(f"Generated: {datetime.now().isoformat()}\n")

if not best_by_method.empty:
    overall_best = best_by_method["mean_auroc"].max()
    delta = overall_best - BASELINE_AUROC
    takeaway_lines.append(f"• Best AUROC across all 6 methods: {overall_best:.4f} "
                          f"(baseline {BASELINE_AUROC:.4f}, delta {delta:+.4f})")

if not lgbm_runs.empty:
    best_lgbm = lgbm_runs["mean_auroc"].max()
    takeaway_lines.append(f"• Deeper LightGBM (50-trial sweep): best = {best_lgbm:.4f} "
                          f"({gain_label(best_lgbm)})")

if not cat_runs.empty:
    best_cat = cat_runs["mean_auroc"].max()
    takeaway_lines.append(f"• CatBoost: best = {best_cat:.4f} "
                          f"({gain_label(best_cat)})")

if not path_runs.empty:
    best_path = path_runs["mean_auroc"].max()
    takeaway_lines.append(f"• Best pathway/gene feature combo: {best_path:.4f}")

if not strat_df.empty:
    above = strat_df[strat_df["mean_auroc"] > BASELINE_AUROC + 0.02]
    takeaway_lines.append(f"• Stratified modeling: {len(above)}/{len(strat_df)} strata exceed baseline")

if diagnostics:
    n_robust = diagnostics.get("n_robust_consistent_genes", 0)
    sil_resp = diagnostics.get("silhouette_response", 0)
    n_sig_path = diagnostics.get("n_significant_pathways_bonferroni", 0)
    takeaway_lines.append(f"• Genes with consistent response direction across ≥20 datasets: {n_robust}")
    takeaway_lines.append(f"• PCA silhouette by response: {sil_resp:+.3f}  (near 0 = no separation)")
    takeaway_lines.append(f"• Bonferroni-significant pathways: {n_sig_path}")

# Bottom line bullet
if not best_by_method.empty:
    delta_overall = best_by_method["mean_auroc"].max() - BASELINE_AUROC
    if abs(delta_overall) < 0.01:
        takeaway_lines.append("\n• BOTTOM LINE: signal-limited, not model-limited. ~0.60 is the ceiling.")
    elif delta_overall > 0.03:
        takeaway_lines.append("\n• BOTTOM LINE: meaningful improvement found — see final_overnight_summary.md.")
    else:
        takeaway_lines.append("\n• BOTTOM LINE: small improvement possible. See final_overnight_summary.md.")

takeaway_lines.append("\nFull report: results/overnight_model_signal_benchmark/reports/final_overnight_summary.md")

takeaway_path = BENCH / "reports" / "morning_takeaway.txt"
with open(takeaway_path, "w") as f:
    f.write("\n".join(takeaway_lines))
log.info(f"Saved {takeaway_path}")

log.info("\nDone.")
