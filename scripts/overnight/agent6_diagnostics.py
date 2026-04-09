#!/usr/bin/env python
"""
Agent 6: Signal and Heterogeneity Diagnostics
==============================================

Quantify directly whether response-linked expression structure exists.

A. Per-gene response association (within-dataset Mann-Whitney U)
B. Cross-dataset reproducibility (sign concordance of effects)
C. Variance partitioning (response vs dataset vs technology)
D. PCA + silhouette by response and dataset
E. Pathway-response association
"""

import os, sys, json, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.overnight._shared import (
    BENCH, load_baseline_data, setup_logger,
    within_sample_rank_inv_norm, compute_singscore,
    PLATFORM_MAP, DATASET_DRUG_MAP, DATASET_CANCER_MAP,
)

AGENT = "agent6_diagnostics"
log = setup_logger(AGENT)

log.info("=" * 70)
log.info("AGENT 6: Signal and Heterogeneity Diagnostics")
log.info("=" * 70)

data = load_baseline_data()
log.info(f"{len(data['datasets'])} datasets, {len(data['pooled_labels'])} patients, {len(data['common_genes'])} genes")

pooled_expr = data["pooled_expr"]
pooled_labels = data["pooled_labels"]
batch_series = data["batch_series"]
tech_series = data["tech_series"]
common_genes = data["common_genes"]
s2d = data["s2d"]


# =========================================================================
# A. Per-gene within-dataset response association
# =========================================================================
log.info("\n=== A. Within-dataset gene-response associations ===")

per_gene_per_dataset = []
for did in sorted(set(s2d.values())):
    samples = [s for s in pooled_expr.index if s2d[s] == did]
    if len(samples) < 20:
        continue
    sub_expr = pooled_expr.loc[samples]
    sub_y = pooled_labels.loc[samples]
    if sub_y.nunique() < 2:
        continue

    pos_mask = sub_y == 1
    neg_mask = sub_y == 0

    for gene in common_genes:
        try:
            pos_vals = sub_expr.loc[pos_mask, gene].dropna()
            neg_vals = sub_expr.loc[neg_mask, gene].dropna()
            if len(pos_vals) < 3 or len(neg_vals) < 3:
                continue
            stat, p = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            # Effect size: rank-biserial correlation
            n1, n2 = len(pos_vals), len(neg_vals)
            rbc = 2 * stat / (n1 * n2) - 1
            per_gene_per_dataset.append({
                "dataset": did, "gene": gene, "p_value": p, "effect_size": rbc,
                "n_pos": n1, "n_neg": n2,
            })
        except Exception:
            continue

per_gene_df = pd.DataFrame(per_gene_per_dataset)
per_gene_df.to_csv(BENCH / "diagnostics" / "per_gene_per_dataset.tsv", sep="\t", index=False)
log.info(f"Computed {len(per_gene_df)} (gene, dataset) tests")

# Significant gene counts per dataset
sig_per_dataset = per_gene_df.groupby("dataset").apply(
    lambda g: pd.Series({
        "n_tests": len(g),
        "n_sig_p05": (g["p_value"] < 0.05).sum(),
        "n_sig_p01": (g["p_value"] < 0.01).sum(),
        "frac_sig_p05": (g["p_value"] < 0.05).mean(),
        "median_abs_effect": g["effect_size"].abs().median(),
        "max_abs_effect": g["effect_size"].abs().max(),
    })
).reset_index()
sig_per_dataset.to_csv(BENCH / "diagnostics" / "signal_strength_per_dataset.tsv", sep="\t", index=False)

log.info(f"\nSignal strength per dataset (top 10 by frac_sig_p05):")
for _, r in sig_per_dataset.sort_values("frac_sig_p05", ascending=False).head(10).iterrows():
    log.info(f"  {r['dataset']:15s}  n_tests={int(r['n_tests'])}  "
             f"sig@p05={int(r['n_sig_p05'])} ({r['frac_sig_p05']:.1%})  "
             f"median|eff|={r['median_abs_effect']:.3f}")

# Expected at random: 5% of tests significant at p<0.05
expected_sig_rate = 0.05
sig_per_dataset["enrichment_vs_random"] = sig_per_dataset["frac_sig_p05"] / expected_sig_rate
log.info(f"\nMean significance enrichment vs random: {sig_per_dataset['enrichment_vs_random'].mean():.2f}x")
log.info(f"Datasets with > 2x enrichment: {(sig_per_dataset['enrichment_vs_random'] > 2).sum()}/{len(sig_per_dataset)}")


# =========================================================================
# B. Cross-dataset reproducibility (sign concordance)
# =========================================================================
log.info("\n=== B. Cross-dataset sign concordance ===")

# For each gene, what fraction of datasets show effect in the same direction?
sign_per_gene = per_gene_df.groupby("gene").apply(
    lambda g: pd.Series({
        "n_datasets": len(g),
        "n_pos_effect": (g["effect_size"] > 0).sum(),
        "n_neg_effect": (g["effect_size"] < 0).sum(),
        "max_concordance": max((g["effect_size"] > 0).sum(), (g["effect_size"] < 0).sum()) / len(g),
        "mean_effect": g["effect_size"].mean(),
        "median_p": g["p_value"].median(),
    })
).reset_index()
sign_per_gene.to_csv(BENCH / "diagnostics" / "gene_sign_concordance.tsv", sep="\t", index=False)

# Genes with high concordance (>= 75% same direction across at least 20 datasets)
robust = sign_per_gene[(sign_per_gene["n_datasets"] >= 20) & (sign_per_gene["max_concordance"] >= 0.75)]
log.info(f"Genes with consistent direction across >=20 datasets at >=75% concordance: {len(robust)}")
if len(robust) > 0:
    log.info("Top 15:")
    for _, r in robust.sort_values("max_concordance", ascending=False).head(15).iterrows():
        log.info(f"  {r['gene']:12s}  n_ds={int(r['n_datasets'])}  concord={r['max_concordance']:.2%}  "
                 f"mean_effect={r['mean_effect']:+.3f}")

# Expected at random: any binary outcome should have ~50% concordance
log.info(f"\nMean cross-dataset concordance: {sign_per_gene['max_concordance'].mean():.3f}")
log.info(f"Expected at random: 0.500")


# =========================================================================
# C. Variance partitioning (PCA-based)
# =========================================================================
log.info("\n=== C. Variance partitioning ===")

# Use rank-normalized expression
log.info("Rank-normalizing for diagnostics...")
rank_expr = within_sample_rank_inv_norm(pooled_expr)

log.info("Computing PCA (50 components)...")
pca = PCA(n_components=50, random_state=42)
pca_coords = pca.fit_transform(rank_expr.values)
log.info(f"Cumulative variance explained by PC1-PC10: {pca.explained_variance_ratio_[:10].cumsum()[-1]:.3f}")

# For each PC, correlate with response, dataset, technology (one-hot)
log.info("Correlating PCs with structural variables...")
le_dataset = LabelEncoder().fit_transform(batch_series.values)
le_tech = LabelEncoder().fit_transform(tech_series.values)
y_arr = pooled_labels.values.astype(float)

pc_correlations = []
for pc_i in range(min(20, pca_coords.shape[1])):
    pc_vals = pca_coords[:, pc_i]
    # R^2 with dataset (categorical → use ANOVA-style F)
    from scipy.stats import f_oneway
    groups_ds = [pc_vals[le_dataset == ds_id] for ds_id in np.unique(le_dataset)]
    f_ds, p_ds = f_oneway(*groups_ds) if len(groups_ds) > 1 else (np.nan, np.nan)
    groups_tech = [pc_vals[le_tech == t] for t in np.unique(le_tech)]
    f_tech, p_tech = f_oneway(*groups_tech) if len(groups_tech) > 1 else (np.nan, np.nan)
    # Correlation with response
    r_resp, p_resp = pearsonr(pc_vals, y_arr)
    pc_correlations.append({
        "PC": pc_i + 1,
        "var_explained": pca.explained_variance_ratio_[pc_i],
        "response_pearson_r": r_resp, "response_p": p_resp,
        "dataset_F": f_ds, "dataset_p": p_ds,
        "tech_F": f_tech, "tech_p": p_tech,
    })

pc_df = pd.DataFrame(pc_correlations)
pc_df.to_csv(BENCH / "diagnostics" / "pc_structural_correlations.tsv", sep="\t", index=False)
log.info(f"\nTop 10 PCs:")
log.info(f"  {'PC':>4} {'var_exp':>10} {'resp_r':>10} {'ds_F':>10} {'tech_F':>10}")
for _, r in pc_df.head(10).iterrows():
    log.info(f"  {int(r['PC']):>4} {r['var_explained']:>10.4f} {r['response_pearson_r']:>+10.4f} "
             f"{r['dataset_F']:>10.1f} {r['tech_F']:>10.1f}")


# =========================================================================
# D. Silhouette analysis
# =========================================================================
log.info("\n=== D. Silhouette by response vs dataset ===")

# Use first 50 PCs (computational efficiency)
sub_n = min(2000, len(pca_coords))
rng = np.random.default_rng(42)
sub_idx = rng.choice(len(pca_coords), sub_n, replace=False)
sub_coords = pca_coords[sub_idx]

sil_response = silhouette_score(sub_coords, pooled_labels.values[sub_idx], metric="euclidean")
sil_dataset = silhouette_score(sub_coords, le_dataset[sub_idx], metric="euclidean")
sil_tech = silhouette_score(sub_coords, le_tech[sub_idx], metric="euclidean")

log.info(f"Silhouette score (response):    {sil_response:+.4f}")
log.info(f"Silhouette score (dataset):     {sil_dataset:+.4f}")
log.info(f"Silhouette score (technology):  {sil_tech:+.4f}")
log.info(f"Higher = better-separated clusters; near 0 = no structure")

silhouette_df = pd.DataFrame([
    {"variable": "response", "silhouette": sil_response},
    {"variable": "dataset", "silhouette": sil_dataset},
    {"variable": "technology", "silhouette": sil_tech},
])
silhouette_df.to_csv(BENCH / "diagnostics" / "silhouette_scores.tsv", sep="\t", index=False)


# =========================================================================
# E. Pathway-response association
# =========================================================================
log.info("\n=== E. Pathway-response association ===")
singscore = compute_singscore(pooled_expr, common_genes)

pathway_assoc = []
for col in singscore.columns:
    pos = singscore.loc[pooled_labels == 1, col].dropna()
    neg = singscore.loc[pooled_labels == 0, col].dropna()
    if len(pos) < 30 or len(neg) < 30:
        continue
    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
    n1, n2 = len(pos), len(neg)
    rbc = 2 * stat / (n1 * n2) - 1
    pathway_assoc.append({
        "pathway": col, "p_value": p, "effect_size": rbc,
    })

pathway_df = pd.DataFrame(pathway_assoc).sort_values("p_value")
pathway_df.to_csv(BENCH / "diagnostics" / "pathway_response_assoc.tsv", sep="\t", index=False)

log.info(f"Pathways tested: {len(pathway_df)}")
log.info(f"Significant at p<0.05: {(pathway_df['p_value'] < 0.05).sum()}")
log.info(f"Significant at Bonferroni p<{0.05/len(pathway_df):.4f}: {(pathway_df['p_value'] < 0.05/len(pathway_df)).sum()}")
log.info(f"\nTop 10 pathways:")
for _, r in pathway_df.head(10).iterrows():
    log.info(f"  {r['pathway']:50s}  p={r['p_value']:.2e}  eff={r['effect_size']:+.3f}")


# =========================================================================
# F. PCA visualization plots
# =========================================================================
log.info("\n=== F. Generating diagnostic plots ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_idx = rng.choice(len(pca_coords), min(3000, len(pca_coords)), replace=False)
plot_coords = pca_coords[plot_idx]

# By response
axes[0].scatter(plot_coords[:, 0], plot_coords[:, 1],
                c=pooled_labels.values[plot_idx], cmap="coolwarm", s=8, alpha=0.6)
axes[0].set_title(f"PCA colored by response (silhouette={sil_response:+.3f})")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

# By dataset
axes[1].scatter(plot_coords[:, 0], plot_coords[:, 1],
                c=le_dataset[plot_idx], cmap="tab20", s=8, alpha=0.6)
axes[1].set_title(f"PCA colored by dataset (silhouette={sil_dataset:+.3f})")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

# By technology
axes[2].scatter(plot_coords[:, 0], plot_coords[:, 1],
                c=le_tech[plot_idx], cmap="Set1", s=8, alpha=0.6)
axes[2].set_title(f"PCA colored by technology (silhouette={sil_tech:+.3f})")
axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")

plt.tight_layout()
plt.savefig(BENCH / "plots" / "pca_diagnostics.png", dpi=120, bbox_inches="tight")
plt.close()
log.info(f"Saved PCA plot: {BENCH / 'plots' / 'pca_diagnostics.png'}")


# =========================================================================
# G. Save aggregate diagnostics summary
# =========================================================================
diagnostics_summary = {
    "n_datasets": len(set(s2d.values())),
    "n_patients": len(pooled_labels),
    "n_genes": len(common_genes),
    "mean_per_dataset_sig_enrichment": float(sig_per_dataset["enrichment_vs_random"].mean()),
    "datasets_with_2x_enrichment": int((sig_per_dataset["enrichment_vs_random"] > 2).sum()),
    "mean_cross_dataset_concordance": float(sign_per_gene["max_concordance"].mean()),
    "n_robust_consistent_genes": len(robust),
    "silhouette_response": float(sil_response),
    "silhouette_dataset": float(sil_dataset),
    "silhouette_technology": float(sil_tech),
    "n_significant_pathways_p05": int((pathway_df["p_value"] < 0.05).sum()),
    "n_significant_pathways_bonferroni": int((pathway_df["p_value"] < 0.05/len(pathway_df)).sum()),
    "max_pc_response_correlation": float(pc_df["response_pearson_r"].abs().max()),
    "max_pc_dataset_F": float(pc_df["dataset_F"].max()),
}

with open(BENCH / "diagnostics" / "diagnostics_summary.json", "w") as f:
    json.dump(diagnostics_summary, f, indent=2)

log.info(f"\n=== DIAGNOSTICS SUMMARY ===")
for k, v in diagnostics_summary.items():
    log.info(f"  {k}: {v}")

log.info(f"\nDone.")
