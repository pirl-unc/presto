#!/usr/bin/env python
"""Generate comprehensive figures for v4 fine-grained embed_dim sweep (50 epochs)."""

import json
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

EMBED_DIMS = [32, 64, 96, 128, 192, 256]
DIM_COLORS = {32: "#1f77b4", 64: "#ff7f0e", 96: "#2ca02c", 128: "#d62728", 192: "#9467bd", 256: "#8c564b"}


def load_all():
    summaries = []
    for cid in range(1, 7):
        with open(RESULTS_DIR / f"c{cid:02d}_summary.json") as f:
            summaries.append(json.load(f))
    return summaries


def plot_val_spearman_curves(summaries):
    """Val Spearman over 50 epochs for all dims — the key plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in summaries:
        edim = s["config"]["embed_dim"]
        epochs = [e["epoch"] for e in s["epoch_summaries"]]
        vals = [e["val_spearman"] for e in s["epoch_summaries"]]
        ax.plot(epochs, vals, "-", label=f"d={edim} ({s['config']['n_params']:,} params)",
                color=DIM_COLORS[edim], linewidth=2, alpha=0.85)
        # Mark best epoch
        best = max(s["epoch_summaries"], key=lambda e: e["val_spearman"])
        ax.plot(best["epoch"], best["val_spearman"], "o", color=DIM_COLORS[edim],
                markersize=8, markeredgecolor="black", markeredgewidth=1, zorder=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Spearman", fontsize=12)
    ax.set_title("Val Spearman Over 50 Epochs by Encoder Dimension", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 51)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "val_spearman_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved val_spearman_curves.png")


def plot_train_loss_curves(summaries):
    """Train loss over 50 epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in summaries:
        edim = s["config"]["embed_dim"]
        epochs = [e["epoch"] for e in s["epoch_summaries"]]
        losses = [e["train_loss"] for e in s["epoch_summaries"]]
        ax.plot(epochs, losses, "-", label=f"d={edim}",
                color=DIM_COLORS[edim], linewidth=2, alpha=0.85)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title("Training Loss Over 50 Epochs by Encoder Dimension", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 51)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "train_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved train_loss_curves.png")


def plot_train_val_gap(summaries):
    """Train loss vs val loss to show overfitting."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Train vs Val Loss by Encoder Dimension (Overfitting Diagnostic)", fontsize=14)
    for ax, s in zip(axes.flat, summaries):
        edim = s["config"]["embed_dim"]
        epochs = [e["epoch"] for e in s["epoch_summaries"]]
        train_loss = [e["train_loss"] for e in s["epoch_summaries"]]
        val_loss = [e["val_loss"] for e in s["epoch_summaries"]]
        ax.plot(epochs, train_loss, "-", label="Train", color="#1f77b4", linewidth=1.5)
        ax.plot(epochs, val_loss, "-", label="Val", color="#d62728", linewidth=1.5)
        ax.fill_between(epochs, train_loss, val_loss, alpha=0.15, color="#d62728")
        ax.set_title(f"d={edim} ({s['config']['n_params']:,} params)", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "train_val_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved train_val_gap.png")


def plot_param_efficiency(summaries):
    """Test Spearman vs params — the efficiency curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    params = [s["config"]["n_params"] for s in summaries]
    spearman = [s["test_metrics"]["spearman"] for s in summaries]
    dims = [s["config"]["embed_dim"] for s in summaries]

    for p, sp, d in zip(params, spearman, dims):
        ax.scatter(p, sp, s=150, color=DIM_COLORS[d], edgecolors="black", linewidth=1, zorder=5)
        ax.annotate(f"d={d}", (p, sp), fontsize=10, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 8), fontweight="bold")

    ax.plot(params, spearman, "--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Test Spearman", fontsize=12)
    ax.set_title("Parameter Efficiency: Test Spearman vs Model Size (50 epochs)", fontsize=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
    ax.grid(True, alpha=0.3)

    # Add horizontal line at best
    best_sp = max(spearman)
    ax.axhline(y=best_sp, color="green", linestyle=":", alpha=0.5, label=f"Best: {best_sp:.4f}")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "param_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved param_efficiency.png")


def plot_bar_chart(summaries):
    """Bar chart of test metrics by embed_dim."""
    dims = [s["config"]["embed_dim"] for s in summaries]
    metrics = {
        "Spearman": [s["test_metrics"]["spearman"] for s in summaries],
        "AUROC": [s["test_metrics"]["auroc"] for s in summaries],
        "F1": [s["test_metrics"]["f1"] for s in summaries],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, vals) in zip(axes, metrics.items()):
        colors = [DIM_COLORS[d] for d in dims]
        bars = ax.bar([f"d={d}" for d in dims], vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f"Test {name} by embed_dim", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        # Set y range
        ax.set_ylim(min(vals) - 0.02, max(vals) + 0.015)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "bar_test_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved bar_test_metrics.png")


def plot_best_epoch_bar(summaries):
    """When each dimension reaches its best val Spearman."""
    dims = [s["config"]["embed_dim"] for s in summaries]
    best_epochs = [max(s["epoch_summaries"], key=lambda e: e["val_spearman"])["epoch"] for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [DIM_COLORS[d] for d in dims]
    bars = ax.bar([f"d={d}" for d in dims], best_epochs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, ep in zip(bars, best_epochs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(ep), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Best Validation Epoch", fontsize=12)
    ax.set_title("When Does Each Model Peak? (out of 50 epochs)", fontsize=14)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Epoch limit")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "best_epoch_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved best_epoch_bar.png")


def plot_grad_norms(summaries):
    """Gradient norm evolution."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in summaries:
        edim = s["config"]["embed_dim"]
        epochs = [e["epoch"] for e in s["epoch_summaries"]]
        grads = [e["train_grad_norm_output"] for e in s["epoch_summaries"]]
        ax.plot(epochs, grads, "-", label=f"d={edim}", color=DIM_COLORS[edim], linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Output Layer Grad Norm", fontsize=12)
    ax.set_title("Gradient Norm by Encoder Dimension (50 epochs)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "grad_norms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved grad_norms.png")


def plot_convergence_rate(summaries):
    """How quickly each model reaches 95% and 99% of its best val Spearman."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in summaries:
        edim = s["config"]["embed_dim"]
        best_sp = max(e["val_spearman"] for e in s["epoch_summaries"])
        vals = [e["val_spearman"] for e in s["epoch_summaries"]]
        # Normalize to fraction of best
        frac = [v / best_sp for v in vals]
        ax.plot(range(1, 51), frac, "-", label=f"d={edim}", color=DIM_COLORS[edim], linewidth=2)
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, label="99% of best")
    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95% of best")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Fraction of Best Val Spearman", fontsize=12)
    ax.set_title("Convergence Rate: How Quickly Does Each Model Reach Its Peak?", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.82, 1.005)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "convergence_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved convergence_rate.png")


def plot_dashboard(summaries):
    """Single-page dashboard."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Val Spearman curves
    ax1 = fig.add_subplot(gs[0, 0:2])
    for s in summaries:
        edim = s["config"]["embed_dim"]
        epochs = [e["epoch"] for e in s["epoch_summaries"]]
        vals = [e["val_spearman"] for e in s["epoch_summaries"]]
        ax1.plot(epochs, vals, "-", label=f"d={edim}", color=DIM_COLORS[edim], linewidth=1.5)
        best = max(s["epoch_summaries"], key=lambda e: e["val_spearman"])
        ax1.plot(best["epoch"], best["val_spearman"], "o", color=DIM_COLORS[edim], markersize=6,
                 markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Spearman")
    ax1.set_title("Learning Curves (circles = peak)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # 2. Param efficiency
    ax2 = fig.add_subplot(gs[0, 2])
    for s in summaries:
        edim = s["config"]["embed_dim"]
        ax2.scatter(s["config"]["n_params"] / 1e3, s["test_metrics"]["spearman"],
                    s=120, color=DIM_COLORS[edim], edgecolors="black", linewidth=0.5, zorder=5)
        ax2.annotate(f"d={edim}", (s["config"]["n_params"]/1e3, s["test_metrics"]["spearman"]),
                     fontsize=8, ha="center", va="bottom", textcoords="offset points", xytext=(0, 6))
    ax2.set_xlabel("Parameters (k)")
    ax2.set_ylabel("Test Spearman")
    ax2.set_title("Param Efficiency")
    ax2.grid(True, alpha=0.3)

    # 3. Bar: test Spearman
    ax3 = fig.add_subplot(gs[1, 0])
    dims = [s["config"]["embed_dim"] for s in summaries]
    sp_vals = [s["test_metrics"]["spearman"] for s in summaries]
    colors = [DIM_COLORS[d] for d in dims]
    bars = ax3.bar([f"d={d}" for d in dims], sp_vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, sp_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("Test Spearman")
    ax3.set_title("Test Spearman")
    ax3.set_ylim(min(sp_vals) - 0.005, max(sp_vals) + 0.005)
    ax3.grid(True, axis="y", alpha=0.3)

    # 4. Best epoch
    ax4 = fig.add_subplot(gs[1, 1])
    best_epochs = [max(s["epoch_summaries"], key=lambda e: e["val_spearman"])["epoch"] for s in summaries]
    bars4 = ax4.bar([f"d={d}" for d in dims], best_epochs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, ep in zip(bars4, best_epochs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(ep), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Best Epoch")
    ax4.set_title("Peak Epoch (of 50)")
    ax4.axhline(y=50, color="red", linestyle="--", alpha=0.3)
    ax4.grid(True, axis="y", alpha=0.3)

    # 5. Train vs Val gap at epoch 50
    ax5 = fig.add_subplot(gs[1, 2])
    train_final = [s["epoch_summaries"][-1]["train_loss"] for s in summaries]
    val_final = [s["epoch_summaries"][-1]["val_loss"] for s in summaries]
    gap = [v - t for t, v in zip(train_final, val_final)]
    bars5 = ax5.bar([f"d={d}" for d in dims], gap, color=colors, edgecolor="black", linewidth=0.5)
    for bar, g in zip(bars5, gap):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                 f"{g:.4f}", ha="center", va="bottom", fontsize=8)
    ax5.set_ylabel("Val - Train Loss")
    ax5.set_title("Overfit Gap at Epoch 50")
    ax5.grid(True, axis="y", alpha=0.3)

    fig.suptitle("V4: Fine-Grained Encoder Dimension Sweep (6 dims, 50 epochs, MAX=50k)", fontsize=15, y=1.02)
    fig.savefig(FIGURES_DIR / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved dashboard.png")


def plot_v3_v4_comparison(summaries):
    """Compare v3 (20ep) and v4 (50ep) for overlapping dims (128, 256) at MAX=50k."""
    # Load v3 data
    v3_dir = EXP_DIR.parent / "2026-03-13_1218_claude_mhcflurry-max-dim-sweep" / "all_results.json"
    if not v3_dir.exists():
        print("  Skipping v3/v4 comparison (v3 results not found)")
        return

    v3_data = json.load(open(v3_dir))
    # Find v3 conditions with MAX=50k
    v3_50k = {r["embed_dim"]: r for r in v3_data if r["max_nM"] == 50000}

    fig, ax = plt.subplots(figsize=(10, 6))

    # V3 (20 epochs)
    v3_dims = sorted(v3_50k.keys())
    v3_sp = [v3_50k[d]["test_spearman"] for d in v3_dims]
    ax.plot(v3_dims, v3_sp, "s--", label="v3 (20 epochs)", color="#888888",
            linewidth=2, markersize=8, markeredgecolor="black")

    # V4 (50 epochs)
    v4_dims = [s["config"]["embed_dim"] for s in summaries]
    v4_sp = [s["test_metrics"]["spearman"] for s in summaries]
    ax.plot(v4_dims, v4_sp, "o-", label="v4 (50 epochs)", color="#1f77b4",
            linewidth=2, markersize=8, markeredgecolor="black")

    # Annotate improvements for shared dims
    shared = set(v3_dims) & set(v4_dims)
    for d in shared:
        v3_val = v3_50k[d]["test_spearman"]
        v4_val = next(s["test_metrics"]["spearman"] for s in summaries if s["config"]["embed_dim"] == d)
        delta = v4_val - v3_val
        y_mid = (v3_val + v4_val) / 2
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"{sign}{delta:.4f}", (d, y_mid), fontsize=9, ha="left",
                    textcoords="offset points", xytext=(8, 0), color="green" if delta > 0 else "red")

    ax.set_xlabel("embed_dim", fontsize=12)
    ax.set_ylabel("Test Spearman", fontsize=12)
    ax.set_title("Effect of Training Longer: v3 (20ep) vs v4 (50ep) at MAX=50k", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(set(v3_dims) | set(v4_dims)))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "v3_v4_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved v3_v4_comparison.png")


def write_summary(summaries):
    """Write summary table and all_results.json."""
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

    with open(EXP_DIR / "all_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n  === V4 LEADERBOARD (by Test Spearman) ===")
    print(f"  {'Rank':>4} {'Cond':>4} {'dim':>5} {'Params':>9} {'Spearman':>8} {'AUROC':>7} {'F1':>6} {'RMSE':>6} {'BestEp':>6}")
    print("  " + "-" * 65)
    for rank, r in enumerate(rows, 1):
        print(f"  {rank:>4} c{r['cond_id']:02d}  d={r['embed_dim']:>3}  {r['n_params']:>8,}  {r['test_spearman']:.4f}  {r['test_auroc']:.4f}  {r['test_f1']:.3f}  {r['test_rmse_log10']:.3f}  {r['best_val_epoch']:>4}")


def main():
    summaries = load_all()
    print(f"Loaded {len(summaries)} conditions\n")
    print("Generating figures...")
    plot_val_spearman_curves(summaries)
    plot_train_loss_curves(summaries)
    plot_train_val_gap(summaries)
    plot_param_efficiency(summaries)
    plot_bar_chart(summaries)
    plot_best_epoch_bar(summaries)
    plot_grad_norms(summaries)
    plot_convergence_rate(summaries)
    plot_dashboard(summaries)
    plot_v3_v4_comparison(summaries)
    write_summary(summaries)
    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
