#!/usr/bin/env python3
"""Generate analysis plots for the distributional BA heads experiment.

Reads all_results.json (test metrics per condition) and per-epoch summaries
from /tmp/dist-ba-results/c*_summary.json, then produces:

1. Heatmap of test AUROC by head_type x assay_mode (best across MAX/K/sigma)
2. Heatmap of test Spearman (same layout)
3. Per-epoch val_loss curves for representative conditions
4. Per-epoch val_spearman curves for representative conditions
5. Bar chart of all 32 conditions sorted by test Spearman, colored by head_type
6. D1-affine vs D2-logit paired scatter plot of AUROC
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parent
ALL_RESULTS = EXP_DIR / "all_results.json"
SUMMARY_DIR = Path("/tmp/dist-ba-results")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(ALL_RESULTS) as f:
    results = json.load(f)

# Load per-epoch summaries keyed by cond_id
epoch_data = {}
for p in sorted(SUMMARY_DIR.glob("c*_summary.json")):
    with open(p) as f:
        d = json.load(f)
    cid = d["config"]["cond_id"]
    epoch_data[cid] = d["epoch_summaries"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
HEAD_ORDER = ["mhcflurry", "log_mse", "twohot", "hlgauss"]
ASSAY_ORDER = ["affine", "additive", "d1_affine", "d2_logit"]

HEAD_COLORS = {
    "mhcflurry": "#1f77b4",
    "log_mse": "#ff7f0e",
    "twohot": "#2ca02c",
    "hlgauss": "#d62728",
}

REPRESENTATIVE_CONDS = {
    "c01 mhcflurry affine": 1,
    "c03 log_mse affine": 3,
    "c13 twohot d2_logit": 13,
    "c17 hlgauss d2_logit": 17,
    "c05 twohot d1_affine (fail)": 5,
    "c29 mhcflurry additive": 29,
    "c31 log_mse additive": 31,
}


def _build_best_matrix(results, metric):
    """Build a 2D matrix (head x assay) with the best value for `metric`."""
    best = {}
    for r in results:
        key = (r["head"], r["assay"])
        val = r[metric]
        if key not in best or val > best[key]:
            best[key] = val
    mat = np.full((len(HEAD_ORDER), len(ASSAY_ORDER)), np.nan)
    for i, h in enumerate(HEAD_ORDER):
        for j, a in enumerate(ASSAY_ORDER):
            if (h, a) in best:
                mat[i, j] = best[(h, a)]
    return mat


def _make_heatmap(ax, mat, title, fmt=".3f", cmap="YlOrRd"):
    """Render an annotated heatmap on `ax`."""
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(ASSAY_ORDER)))
    ax.set_xticklabels(ASSAY_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(HEAD_ORDER)))
    ax.set_yticklabels(HEAD_ORDER)
    ax.set_title(title, fontsize=12, fontweight="bold")
    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")
            else:
                color = "white" if v < (np.nanmin(mat) + np.nanmax(mat)) / 2 else "black"
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def _cond_label(r):
    """Short human-readable label for a condition."""
    parts = [f"c{r['cond_id']:02d}", r["head"], r["assay"]]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Plot 1 & 2: Heatmaps
# ---------------------------------------------------------------------------
def plot_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    auroc_mat = _build_best_matrix(results, "test_auroc")
    spear_mat = _build_best_matrix(results, "test_spearman")
    _make_heatmap(axes[0], auroc_mat, "Best Test AUROC\n(head_type x assay_mode)")
    _make_heatmap(axes[1], spear_mat, "Best Test Spearman\n(head_type x assay_mode)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 & 4: Per-epoch curves
# ---------------------------------------------------------------------------
def plot_epoch_curves(metric_key, ylabel, title_suffix):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, cid in REPRESENTATIVE_CONDS.items():
        if cid not in epoch_data:
            continue
        epochs_list = epoch_data[cid]
        xs = [e["epoch"] for e in epochs_list]
        ys = [e.get(metric_key, np.nan) for e in epochs_list]
        linestyle = "--" if "fail" in label else "-"
        ax.plot(xs, ys, marker="o", markersize=3, label=label, linestyle=linestyle)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-Epoch {title_suffix} — Representative Conditions", fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 5: Bar chart sorted by test Spearman
# ---------------------------------------------------------------------------
def plot_bar_spearman():
    sorted_r = sorted(results, key=lambda r: r["test_spearman"], reverse=True)
    labels = [_cond_label(r) for r in sorted_r]
    values = [r["test_spearman"] for r in sorted_r]
    colors = [HEAD_COLORS[r["head"]] for r in sorted_r]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Test Spearman")
    ax.set_title("All 32 Conditions Sorted by Test Spearman", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Legend for head types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=HEAD_COLORS[h], label=h) for h in HEAD_ORDER]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 6: D1-affine vs D2-logit paired scatter (AUROC)
# ---------------------------------------------------------------------------
def plot_d1_vs_d2_scatter():
    # Build lookup: (head, K, max_nM, sigma) -> {assay: auroc}
    lookup = {}
    for r in results:
        key = (r["head"], r.get("K"), r.get("max_nM"), r.get("sigma"))
        if key not in lookup:
            lookup[key] = {}
        lookup[key][r["assay"]] = r["test_auroc"]

    # Collect pairs where both d1_affine and d2_logit exist
    pairs = []  # (head, d1_auroc, d2_auroc)
    for key, assays in lookup.items():
        if "d1_affine" in assays and "d2_logit" in assays:
            pairs.append((key[0], assays["d1_affine"], assays["d2_logit"]))

    if not pairs:
        # Fallback: nothing to plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, "No paired d1_affine / d2_logit conditions found",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(7, 7))
    for head, d1, d2 in pairs:
        ax.scatter(d1, d2, color=HEAD_COLORS.get(head, "gray"), s=60,
                   edgecolors="black", linewidths=0.5, zorder=3)

    # Diagonal reference
    all_vals = [v for _, v1, v2 in pairs for v in (v1, v2)]
    lo, hi = min(all_vals) - 0.02, max(all_vals) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("D1-affine Test AUROC")
    ax.set_ylabel("D2-logit Test AUROC")
    ax.set_title("D1-affine vs D2-logit: Paired AUROC Comparison", fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=HEAD_COLORS[h], label=h) for h in HEAD_ORDER
                       if any(p[0] == h for p in pairs)]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main: generate all plots and save
# ---------------------------------------------------------------------------
def main():
    output_files = []

    # 1 & 2: Heatmaps
    fig_hm = plot_heatmaps()
    p = EXP_DIR / "heatmap_auroc_spearman.png"
    fig_hm.savefig(p, dpi=150)
    output_files.append(p)
    print(f"Saved: {p}")

    # 3: Val loss curves
    fig_vl = plot_epoch_curves("val_loss", "Val Loss", "Validation Loss")
    p = EXP_DIR / "epoch_val_loss.png"
    fig_vl.savefig(p, dpi=150)
    output_files.append(p)
    print(f"Saved: {p}")

    # 4: Val spearman curves
    fig_vs = plot_epoch_curves("val_spearman", "Val Spearman", "Validation Spearman")
    p = EXP_DIR / "epoch_val_spearman.png"
    fig_vs.savefig(p, dpi=150)
    output_files.append(p)
    print(f"Saved: {p}")

    # 5: Bar chart
    fig_bar = plot_bar_spearman()
    p = EXP_DIR / "bar_spearman_all_conditions.png"
    fig_bar.savefig(p, dpi=150)
    output_files.append(p)
    print(f"Saved: {p}")

    # 6: D1 vs D2 scatter
    fig_sc = plot_d1_vs_d2_scatter()
    p = EXP_DIR / "scatter_d1_vs_d2_auroc.png"
    fig_sc.savefig(p, dpi=150)
    output_files.append(p)
    print(f"Saved: {p}")

    # Combined PDF
    pdf_path = EXP_DIR / "all_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in [fig_hm, fig_vl, fig_vs, fig_bar, fig_sc]:
            pdf.savefig(fig)
    output_files.append(pdf_path)
    print(f"Saved: {pdf_path}")

    plt.close("all")

    print("\n=== All generated files ===")
    for f in output_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
