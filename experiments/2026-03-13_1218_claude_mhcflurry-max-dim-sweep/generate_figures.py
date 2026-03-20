#!/usr/bin/env python
"""Generate comprehensive figures for v3 MHCflurry MAX x embed_dim sweep."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MAX_VALUES = [25_000, 50_000, 75_000, 100_000, 125_000, 150_000]
EMBED_DIMS = [128, 256, 384, 512]
MAX_LABELS = ["25k", "50k", "75k", "100k", "125k", "150k"]

METRICS = ["spearman", "pearson", "auroc", "auprc", "f1", "balanced_accuracy", "rmse_log10"]
METRIC_LABELS = {
    "spearman": "Spearman r",
    "pearson": "Pearson r",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1 (500 nM)",
    "balanced_accuracy": "Balanced Accuracy",
    "rmse_log10": "RMSE (log10 nM)",
}
# For RMSE, lower is better
HIGHER_BETTER = {m: True for m in METRICS}
HIGHER_BETTER["rmse_log10"] = False


def load_all_summaries():
    """Load all 24 summary files, return list of dicts."""
    summaries = []
    for cid in range(1, 25):
        path = RESULTS_DIR / f"c{cid:02d}_summary.json"
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def build_grid(summaries, metric_key, source="test"):
    """Build (4 x 6) grid: rows=embed_dim, cols=max_nM."""
    grid = np.zeros((len(EMBED_DIMS), len(MAX_VALUES)))
    for s in summaries:
        cfg = s["config"]
        edim = cfg["embed_dim"]
        maxn = cfg["max_nM"]
        ri = EMBED_DIMS.index(edim)
        ci = MAX_VALUES.index(maxn)
        if source == "test":
            grid[ri, ci] = s["test_metrics"][metric_key]
        else:
            # best val epoch
            best = max(s["epoch_summaries"], key=lambda e: e.get(f"val_{metric_key}", -999))
            grid[ri, ci] = best[f"val_{metric_key}"]
    return grid


def plot_heatmaps(summaries):
    """Heatmaps for each metric over the MAX x embed_dim grid."""
    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, source, title_suffix in zip(axes, ["test", "val"], ["Test", "Best Validation"]):
            grid = build_grid(summaries, metric, source)
            if HIGHER_BETTER[metric]:
                cmap = "YlOrRd"
            else:
                cmap = "YlOrRd_r"
            im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="upper")
            ax.set_xticks(range(len(MAX_VALUES)))
            ax.set_xticklabels(MAX_LABELS)
            ax.set_yticks(range(len(EMBED_DIMS)))
            ax.set_yticklabels([str(d) for d in EMBED_DIMS])
            ax.set_xlabel("MAX (nM)")
            ax.set_ylabel("Encoder embed_dim")
            ax.set_title(f"{METRIC_LABELS[metric]} — {title_suffix}")
            # Annotate cells
            for i in range(len(EMBED_DIMS)):
                for j in range(len(MAX_VALUES)):
                    val = grid[i, j]
                    fmt = f"{val:.3f}" if abs(val) < 10 else f"{val:.2f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=9,
                            color="white" if grid[i, j] > grid.mean() + 0.5 * grid.std() else "black")
            plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"heatmap_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {len(METRICS)} heatmap figures")


def plot_line_by_max(summaries):
    """Line plots: metric vs MAX, one line per embed_dim."""
    for metric in ["spearman", "auroc", "f1", "rmse_log10"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, source, title_suffix in zip(axes, ["test", "val"], ["Test", "Best Val"]):
            grid = build_grid(summaries, metric, source)
            for ri, edim in enumerate(EMBED_DIMS):
                ax.plot(MAX_LABELS, grid[ri, :], "o-", label=f"d={edim}", linewidth=2, markersize=6)
            ax.set_xlabel("MAX (nM)")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{METRIC_LABELS[metric]} vs MAX — {title_suffix}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"line_by_max_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  Saved line_by_max figures")


def plot_line_by_dim(summaries):
    """Line plots: metric vs embed_dim, one line per MAX."""
    for metric in ["spearman", "auroc", "f1", "rmse_log10"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, source, title_suffix in zip(axes, ["test", "val"], ["Test", "Best Val"]):
            grid = build_grid(summaries, metric, source)
            for ci, maxn in enumerate(MAX_VALUES):
                ax.plot([str(d) for d in EMBED_DIMS], grid[:, ci], "o-",
                        label=f"MAX={MAX_LABELS[ci]}", linewidth=2, markersize=6)
            ax.set_xlabel("Encoder embed_dim")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{METRIC_LABELS[metric]} vs embed_dim — {title_suffix}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"line_by_dim_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  Saved line_by_dim figures")


def plot_bar_ranking(summaries):
    """Bar chart ranking all 24 conditions by test Spearman."""
    labels = []
    spearman_vals = []
    auroc_vals = []
    for s in summaries:
        cfg = s["config"]
        label = f"d{cfg['embed_dim']}_max{cfg['max_nM']//1000}k"
        labels.append(label)
        spearman_vals.append(s["test_metrics"]["spearman"])
        auroc_vals.append(s["test_metrics"]["auroc"])

    # Sort by spearman
    order = np.argsort(spearman_vals)[::-1]
    labels = [labels[i] for i in order]
    spearman_vals = [spearman_vals[i] for i in order]
    auroc_vals = [auroc_vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels))
    w = 0.35
    ax.barh(x - w/2, spearman_vals, w, label="Spearman", color="#2196F3")
    ax.barh(x + w/2, auroc_vals, w, label="AUROC", color="#FF9800")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title("All 24 Conditions Ranked by Test Spearman")
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    # Annotate best
    ax.axhline(y=0, color="green", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "bar_ranking_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved bar_ranking_all.png")


def plot_overfitting_analysis(summaries):
    """Train loss and val Spearman curves by epoch, grouped by embed_dim."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Overfitting Analysis: Training Curves by Encoder Dimension", fontsize=14)

    for ax, edim in zip(axes.flat, EMBED_DIMS):
        relevant = [s for s in summaries if s["config"]["embed_dim"] == edim]
        for s in relevant:
            cfg = s["config"]
            epochs = [e["epoch"] for e in s["epoch_summaries"]]
            train_loss = [e["train_loss"] for e in s["epoch_summaries"]]
            val_spearman = [e["val_spearman"] for e in s["epoch_summaries"]]

            label = f"MAX={cfg['max_nM']//1000}k"
            ax.plot(epochs, val_spearman, "o-", label=label, markersize=3, linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Spearman")
        ax.set_title(f"embed_dim = {edim} ({relevant[0]['config']['n_params']:,} params)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "overfitting_val_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also train loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Loss Curves by Encoder Dimension", fontsize=14)

    for ax, edim in zip(axes.flat, EMBED_DIMS):
        relevant = [s for s in summaries if s["config"]["embed_dim"] == edim]
        for s in relevant:
            cfg = s["config"]
            epochs = [e["epoch"] for e in s["epoch_summaries"]]
            train_loss = [e["train_loss"] for e in s["epoch_summaries"]]

            label = f"MAX={cfg['max_nM']//1000}k"
            ax.plot(epochs, train_loss, "o-", label=label, markersize=3, linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.set_title(f"embed_dim = {edim}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "overfitting_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved overfitting analysis figures")


def plot_train_val_gap(summaries):
    """Plot train-val gap (overfitting severity) as a function of embed_dim."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: final train loss vs best val loss
    ax = axes[0]
    for ci, maxn in enumerate(MAX_VALUES):
        final_train = []
        best_val = []
        for edim in EMBED_DIMS:
            s = [s for s in summaries if s["config"]["embed_dim"] == edim and s["config"]["max_nM"] == maxn][0]
            final_train.append(s["epoch_summaries"][-1]["train_loss"])
            best_val.append(min(e["val_loss"] for e in s["epoch_summaries"]))
        ax.plot([str(d) for d in EMBED_DIMS], [v - t for t, v in zip(final_train, best_val)],
                "o-", label=f"MAX={MAX_LABELS[ci]}", linewidth=2)
    ax.set_xlabel("embed_dim")
    ax.set_ylabel("Val Loss - Train Loss (gap)")
    ax.set_title("Overfitting Gap: Val Loss minus Train Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: best val spearman - test spearman (generalization gap)
    ax = axes[1]
    for ci, maxn in enumerate(MAX_VALUES):
        gen_gap = []
        for edim in EMBED_DIMS:
            s = [s for s in summaries if s["config"]["embed_dim"] == edim and s["config"]["max_nM"] == maxn][0]
            best_val_sp = max(e["val_spearman"] for e in s["epoch_summaries"])
            test_sp = s["test_metrics"]["spearman"]
            gen_gap.append(best_val_sp - test_sp)
        ax.plot([str(d) for d in EMBED_DIMS], gen_gap, "o-",
                label=f"MAX={MAX_LABELS[ci]}", linewidth=2)
    ax.set_xlabel("embed_dim")
    ax.set_ylabel("Best Val Spearman - Test Spearman")
    ax.set_title("Generalization Gap (Val vs Test Spearman)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "train_val_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved train_val_gap.png")


def plot_param_efficiency(summaries):
    """Scatter: n_params vs test Spearman, colored by MAX."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(MAX_VALUES)))
    for ci, maxn in enumerate(MAX_VALUES):
        for s in summaries:
            cfg = s["config"]
            if cfg["max_nM"] != maxn:
                continue
            ax.scatter(cfg["n_params"], s["test_metrics"]["spearman"],
                       color=colors[ci], s=100, edgecolors="black", linewidth=0.5, zorder=3)
            ax.annotate(f"d{cfg['embed_dim']}", (cfg["n_params"], s["test_metrics"]["spearman"]),
                        fontsize=7, ha="center", va="bottom", textcoords="offset points", xytext=(0, 5))
    # Legend
    for ci, maxn in enumerate(MAX_VALUES):
        ax.scatter([], [], color=colors[ci], s=100, edgecolors="black", linewidth=0.5,
                   label=f"MAX={MAX_LABELS[ci]}")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Test Spearman")
    ax.set_title("Parameter Efficiency: Test Spearman vs Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "param_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved param_efficiency.png")


def plot_grad_norm_analysis(summaries):
    """Gradient norm evolution by embed_dim — are bigger models unstable?"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Output Layer Gradient Norm by Encoder Dimension", fontsize=14)

    for ax, edim in zip(axes.flat, EMBED_DIMS):
        relevant = [s for s in summaries if s["config"]["embed_dim"] == edim]
        for s in relevant:
            cfg = s["config"]
            epochs = [e["epoch"] for e in s["epoch_summaries"]]
            grad_norms = [e["train_grad_norm_output"] for e in s["epoch_summaries"]]
            label = f"MAX={cfg['max_nM']//1000}k"
            ax.plot(epochs, grad_norms, "o-", label=label, markersize=3, linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Grad Norm (output layer)")
        ax.set_title(f"embed_dim = {edim}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "grad_norm_by_dim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved grad_norm_by_dim.png")


def plot_best_epoch_heatmap(summaries):
    """When does each condition reach its best val Spearman?"""
    grid = np.zeros((len(EMBED_DIMS), len(MAX_VALUES)))
    for s in summaries:
        cfg = s["config"]
        ri = EMBED_DIMS.index(cfg["embed_dim"])
        ci = MAX_VALUES.index(cfg["max_nM"])
        best_epoch = max(s["epoch_summaries"], key=lambda e: e["val_spearman"])["epoch"]
        grid[ri, ci] = best_epoch

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, cmap="coolwarm", aspect="auto", origin="upper", vmin=1, vmax=20)
    ax.set_xticks(range(len(MAX_VALUES)))
    ax.set_xticklabels(MAX_LABELS)
    ax.set_yticks(range(len(EMBED_DIMS)))
    ax.set_yticklabels([str(d) for d in EMBED_DIMS])
    ax.set_xlabel("MAX (nM)")
    ax.set_ylabel("Encoder embed_dim")
    ax.set_title("Best Validation Epoch (earlier = faster convergence)")
    for i in range(len(EMBED_DIMS)):
        for j in range(len(MAX_VALUES)):
            ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Epoch")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "best_epoch_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved best_epoch_heatmap.png")


def plot_epoch_time_vs_dim(summaries):
    """Wall-clock time per epoch vs embed_dim."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, maxn in enumerate(MAX_VALUES):
        times = []
        for edim in EMBED_DIMS:
            s = [s for s in summaries if s["config"]["embed_dim"] == edim and s["config"]["max_nM"] == maxn][0]
            avg_time = np.mean([e["epoch_time_s"] for e in s["epoch_summaries"]])
            times.append(avg_time)
        ax.plot([str(d) for d in EMBED_DIMS], times, "o-",
                label=f"MAX={MAX_LABELS[ci]}", linewidth=2, markersize=6)
    ax.set_xlabel("embed_dim")
    ax.set_ylabel("Average Epoch Time (s)")
    ax.set_title("Training Time per Epoch vs Model Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "epoch_time_vs_dim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved epoch_time_vs_dim.png")


def plot_metric_correlation(summaries):
    """Scatter matrix: Spearman vs AUROC vs F1 vs RMSE (test set)."""
    metrics_to_compare = ["spearman", "auroc", "f1", "rmse_log10"]
    n = len(metrics_to_compare)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    fig.suptitle("Test Metric Correlations Across 24 Conditions", fontsize=14)

    edim_colors = {128: "#1f77b4", 256: "#ff7f0e", 384: "#2ca02c", 512: "#d62728"}

    for i, mi in enumerate(metrics_to_compare):
        for j, mj in enumerate(metrics_to_compare):
            ax = axes[i][j]
            if i == j:
                # Histogram
                vals = [s["test_metrics"][mi] for s in summaries]
                ax.hist(vals, bins=12, color="#888888", alpha=0.7)
                ax.set_title(METRIC_LABELS[mi] if i == 0 else "")
            else:
                for s in summaries:
                    cfg = s["config"]
                    ax.scatter(s["test_metrics"][mj], s["test_metrics"][mi],
                               color=edim_colors[cfg["embed_dim"]], s=40, alpha=0.8)
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[mi], fontsize=9)
            if i == n - 1:
                ax.set_xlabel(METRIC_LABELS[mj], fontsize=9)
            ax.tick_params(labelsize=7)

    # Legend in corner
    for edim, color in edim_colors.items():
        axes[0][n-1].scatter([], [], color=color, s=40, label=f"d={edim}")
    axes[0][n-1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "metric_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved metric_correlations.png")


def plot_summary_dashboard(summaries):
    """Single-page dashboard with the key findings."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Heatmap: test Spearman
    ax1 = fig.add_subplot(gs[0, 0])
    grid = build_grid(summaries, "spearman", "test")
    im = ax1.imshow(grid, cmap="YlOrRd", aspect="auto", origin="upper")
    ax1.set_xticks(range(len(MAX_VALUES)))
    ax1.set_xticklabels(MAX_LABELS, fontsize=8)
    ax1.set_yticks(range(len(EMBED_DIMS)))
    ax1.set_yticklabels([str(d) for d in EMBED_DIMS])
    ax1.set_xlabel("MAX (nM)")
    ax1.set_ylabel("embed_dim")
    ax1.set_title("Test Spearman")
    for i in range(len(EMBED_DIMS)):
        for j in range(len(MAX_VALUES)):
            ax1.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax1, shrink=0.7)

    # 2. Heatmap: test AUROC
    ax2 = fig.add_subplot(gs[0, 1])
    grid_auroc = build_grid(summaries, "auroc", "test")
    im2 = ax2.imshow(grid_auroc, cmap="YlOrRd", aspect="auto", origin="upper")
    ax2.set_xticks(range(len(MAX_VALUES)))
    ax2.set_xticklabels(MAX_LABELS, fontsize=8)
    ax2.set_yticks(range(len(EMBED_DIMS)))
    ax2.set_yticklabels([str(d) for d in EMBED_DIMS])
    ax2.set_xlabel("MAX (nM)")
    ax2.set_title("Test AUROC")
    for i in range(len(EMBED_DIMS)):
        for j in range(len(MAX_VALUES)):
            ax2.text(j, i, f"{grid_auroc[i,j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    # 3. Parameter efficiency scatter
    ax3 = fig.add_subplot(gs[0, 2])
    edim_colors = {128: "#1f77b4", 256: "#ff7f0e", 384: "#2ca02c", 512: "#d62728"}
    for s in summaries:
        cfg = s["config"]
        ax3.scatter(cfg["n_params"] / 1e6, s["test_metrics"]["spearman"],
                    color=edim_colors[cfg["embed_dim"]], s=60, edgecolors="black", linewidth=0.3)
    for edim, color in edim_colors.items():
        ax3.scatter([], [], color=color, s=60, edgecolors="black", linewidth=0.3, label=f"d={edim}")
    ax3.set_xlabel("Parameters (M)")
    ax3.set_ylabel("Test Spearman")
    ax3.set_title("Param Efficiency")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Spearman vs MAX by dim
    ax4 = fig.add_subplot(gs[1, 0])
    grid_sp = build_grid(summaries, "spearman", "test")
    for ri, edim in enumerate(EMBED_DIMS):
        ax4.plot(MAX_LABELS, grid_sp[ri, :], "o-", label=f"d={edim}",
                 color=edim_colors[edim], linewidth=2)
    ax4.set_xlabel("MAX (nM)")
    ax4.set_ylabel("Test Spearman")
    ax4.set_title("Spearman vs MAX")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Spearman vs embed_dim
    ax5 = fig.add_subplot(gs[1, 1])
    for ci, maxn in enumerate(MAX_VALUES):
        ax5.plot([str(d) for d in EMBED_DIMS], grid_sp[:, ci], "o-",
                 label=f"MAX={MAX_LABELS[ci]}", linewidth=1.5)
    ax5.set_xlabel("embed_dim")
    ax5.set_ylabel("Test Spearman")
    ax5.set_title("Spearman vs embed_dim")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # 6. Best epoch heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    grid_epoch = np.zeros((len(EMBED_DIMS), len(MAX_VALUES)))
    for s in summaries:
        cfg = s["config"]
        ri = EMBED_DIMS.index(cfg["embed_dim"])
        ci = MAX_VALUES.index(cfg["max_nM"])
        grid_epoch[ri, ci] = max(s["epoch_summaries"], key=lambda e: e["val_spearman"])["epoch"]
    im6 = ax6.imshow(grid_epoch, cmap="coolwarm", aspect="auto", origin="upper", vmin=1, vmax=20)
    ax6.set_xticks(range(len(MAX_VALUES)))
    ax6.set_xticklabels(MAX_LABELS, fontsize=8)
    ax6.set_yticks(range(len(EMBED_DIMS)))
    ax6.set_yticklabels([str(d) for d in EMBED_DIMS])
    ax6.set_xlabel("MAX (nM)")
    ax6.set_title("Best Val Epoch")
    for i in range(len(EMBED_DIMS)):
        for j in range(len(MAX_VALUES)):
            ax6.text(j, i, f"{int(grid_epoch[i,j])}", ha="center", va="center", fontsize=9, fontweight="bold")
    plt.colorbar(im6, ax=ax6, shrink=0.7, label="Epoch")

    fig.suptitle("V3 MHCflurry Additive: MAX x embed_dim Sweep (24 conditions, 20 epochs)", fontsize=16, y=1.02)
    fig.savefig(FIGURES_DIR / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved dashboard.png")


def write_summary_table(summaries):
    """Write a summary CSV and print a leaderboard."""
    import csv
    rows = []
    for s in summaries:
        cfg = s["config"]
        tm = s["test_metrics"]
        best_val = max(s["epoch_summaries"], key=lambda e: e["val_spearman"])
        rows.append({
            "cond_id": cfg["cond_id"],
            "embed_dim": cfg["embed_dim"],
            "max_nM": cfg["max_nM"],
            "n_params": cfg["n_params"],
            "test_spearman": tm["spearman"],
            "test_pearson": tm["pearson"],
            "test_auroc": tm["auroc"],
            "test_auprc": tm["auprc"],
            "test_f1": tm["f1"],
            "test_bal_acc": tm["balanced_accuracy"],
            "test_rmse_log10": tm["rmse_log10"],
            "best_val_spearman": best_val["val_spearman"],
            "best_val_epoch": best_val["epoch"],
            "final_train_loss": s["epoch_summaries"][-1]["train_loss"],
        })
    rows.sort(key=lambda r: r["test_spearman"], reverse=True)

    csv_path = EXP_DIR / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Also write the all_results.json
    all_results_path = EXP_DIR / "all_results.json"
    with open(all_results_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n  Summary table: {csv_path}")
    print(f"  All results: {all_results_path}")

    # Print leaderboard
    print("\n  === LEADERBOARD (by Test Spearman) ===")
    print(f"  {'Rank':>4} {'Cond':>4} {'embed_dim':>9} {'MAX':>6} {'Params':>9} {'Spearman':>8} {'AUROC':>7} {'F1':>6} {'RMSE':>6} {'BestEp':>6}")
    print("  " + "-" * 80)
    for rank, r in enumerate(rows, 1):
        print(f"  {rank:>4} c{r['cond_id']:02d}  d={r['embed_dim']:>4}  {r['max_nM']//1000:>3}k  {r['n_params']:>8,}  {r['test_spearman']:.4f}  {r['test_auroc']:.4f}  {r['test_f1']:.3f}  {r['test_rmse_log10']:.3f}  {r['best_val_epoch']:>4}")


def main():
    summaries = load_all_summaries()
    print(f"Loaded {len(summaries)} condition summaries\n")

    print("Generating figures...")
    plot_heatmaps(summaries)
    plot_line_by_max(summaries)
    plot_line_by_dim(summaries)
    plot_bar_ranking(summaries)
    plot_overfitting_analysis(summaries)
    plot_train_val_gap(summaries)
    plot_param_efficiency(summaries)
    plot_grad_norm_analysis(summaries)
    plot_best_epoch_heatmap(summaries)
    plot_epoch_time_vs_dim(summaries)
    plot_metric_correlation(summaries)
    plot_summary_dashboard(summaries)

    write_summary_table(summaries)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
