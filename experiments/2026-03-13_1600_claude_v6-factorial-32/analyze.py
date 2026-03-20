#!/usr/bin/env python3
"""Comprehensive analysis of v6 factorial sweep (32 conditions).

Produces:
  - plots/probe_trajectories.png — per-peptide IC50 trajectories across epochs
  - plots/probe_final_heatmap.png — final-epoch IC50 heatmap (peptide × allele × condition)
  - plots/test_metrics_comparison.png — test-set metric comparison (Spearman, AUROC, etc.)
  - plots/training_curves.png — val loss / val spearman over epochs
  - plots/cc_delta_by_head.png — content-conditioning delta by head type
  - tables/test_metrics_full.csv — all test metrics for all 32 conditions
  - tables/probe_final.csv — final-epoch probe IC50s for all conditions
  - tables/probe_trajectories.csv — all probe data across epochs
  - tables/binary_metrics.csv — <=500nM binary classification metrics
  - tables/correlation_metrics.csv — Spearman, Pearson, RMSE for all conditions
"""

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"
TABLES_DIR = Path(__file__).parent / "tables"

# v6 condition specs: cond_id -> (embed_dim, head_type, max_nM)
CONDITIONS = {}
cid = 0
for embed_dim in (32, 64, 128, 256):
    for head_type in ("mhcflurry", "hlgauss"):
        for max_nM in (50_000, 100_000):
            cid += 1
            CONDITIONS[cid] = {
                "embed_dim": embed_dim,
                "head_type": head_type,
                "max_nM": max_nM,
            }

# Map run directories to (cond_id, cc)
def discover_runs():
    """Find all v6 run directories and map them to conditions."""
    runs = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("distributional_ba_v6_c"):
            continue
        m = re.match(r"distributional_ba_v6_c(\d+)_(cc[01])_", d.name)
        if not m:
            continue
        cond_id = int(m.group(1))
        cc = int(m.group(2)[-1])
        runs[(cond_id, cc)] = d
    return runs


def load_summary(run_dir):
    with open(run_dir / "summary.json") as f:
        return json.load(f)


def load_probes(run_dir):
    rows = []
    with open(run_dir / "probes.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_metrics(run_dir):
    rows = []
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def make_condition_label(cond_id, cc):
    c = CONDITIONS[cond_id]
    cc_str = "cc1" if cc else "cc0"
    max_k = c["max_nM"] // 1000
    return f"c{cond_id:02d}_{cc_str}_d{c['embed_dim']}_{c['head_type']}_{max_k}k"


def short_label(cond_id, cc):
    c = CONDITIONS[cond_id]
    cc_str = "cc1" if cc else "cc0"
    max_k = c["max_nM"] // 1000
    h = "mcf" if c["head_type"] == "mhcflurry" else "hlg"
    return f"d{c['embed_dim']}_{h}_{max_k}k_{cc_str}"


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    runs = discover_runs()
    print(f"Found {len(runs)} runs")

    # ── 1. Collect test metrics ──────────────────────────────────
    test_rows = []
    for (cond_id, cc), run_dir in sorted(runs.items()):
        summary = load_summary(run_dir)
        tm = summary["test_metrics"]
        c = CONDITIONS[cond_id]
        row = {
            "cond_id": cond_id,
            "cc": cc,
            "embed_dim": c["embed_dim"],
            "head_type": c["head_type"],
            "max_nM": c["max_nM"],
            "label": make_condition_label(cond_id, cc),
            **{k: v for k, v in tm.items() if k != "n_samples"},
            "n_samples": tm.get("n_samples", 0),
        }
        # Get best epoch
        epochs = summary.get("epoch_summaries", [])
        if epochs:
            best_ep = max(epochs, key=lambda e: e.get("val_spearman", 0))
            row["best_epoch"] = best_ep["epoch"]
        test_rows.append(row)

    # Write full test metrics CSV
    if test_rows:
        keys = list(test_rows[0].keys())
        with open(TABLES_DIR / "test_metrics_full.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in test_rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"Wrote {TABLES_DIR / 'test_metrics_full.csv'}")

    # ── 2. Correlation metrics table ─────────────────────────────
    corr_keys = ["spearman", "pearson", "rmse_log10"]
    with open(TABLES_DIR / "correlation_metrics.csv", "w") as f:
        header = ["cond_id", "cc", "embed_dim", "head_type", "max_nM", "label"] + corr_keys
        f.write(",".join(header) + "\n")
        for r in test_rows:
            vals = [str(r.get(k, "")) for k in header]
            f.write(",".join(vals) + "\n")
    print(f"Wrote {TABLES_DIR / 'correlation_metrics.csv'}")

    # ── 3. Binary classification metrics table ───────────────────
    bin_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    with open(TABLES_DIR / "binary_metrics.csv", "w") as f:
        header = ["cond_id", "cc", "embed_dim", "head_type", "max_nM", "label"] + bin_keys
        f.write(",".join(header) + "\n")
        for r in test_rows:
            vals = [str(r.get(k, "")) for k in header]
            f.write(",".join(vals) + "\n")
    print(f"Wrote {TABLES_DIR / 'binary_metrics.csv'}")

    # ── 4. Probe data ───────────────────────────────────────────
    all_probe_rows = []
    final_probe_rows = []
    for (cond_id, cc), run_dir in sorted(runs.items()):
        probes = load_probes(run_dir)
        c = CONDITIONS[cond_id]
        label = make_condition_label(cond_id, cc)
        max_epoch = max(p["epoch"] for p in probes)
        for p in probes:
            row = {
                "cond_id": cond_id,
                "cc": cc,
                "embed_dim": c["embed_dim"],
                "head_type": c["head_type"],
                "max_nM": c["max_nM"],
                "label": label,
                "epoch": p["epoch"],
                "peptide": p["peptide"],
                "allele": p["allele"],
                "ic50_nM": p["ic50_nM"],
                "ic50_log10": p["ic50_log10"],
            }
            if "entropy" in p:
                row["entropy"] = p["entropy"]
            all_probe_rows.append(row)
            if p["epoch"] == max_epoch:
                final_probe_rows.append(row)

    # Write probe trajectories
    if all_probe_rows:
        keys = list(all_probe_rows[0].keys())
        # Ensure all rows have same keys
        all_keys = set()
        for r in all_probe_rows:
            all_keys.update(r.keys())
        keys = sorted(all_keys)
        with open(TABLES_DIR / "probe_trajectories.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in all_probe_rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"Wrote {TABLES_DIR / 'probe_trajectories.csv'}")

    # Write final probe table
    if final_probe_rows:
        keys = sorted(set().union(*(r.keys() for r in final_probe_rows)))
        with open(TABLES_DIR / "probe_final.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in final_probe_rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"Wrote {TABLES_DIR / 'probe_final.csv'}")

    # ── 5. Plots ────────────────────────────────────────────────

    # --- 5a. Test metrics comparison: grouped bar chart ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_to_plot = ["spearman", "pearson", "auroc", "auprc", "f1", "rmse_log10"]
    metric_labels = ["Spearman ↑", "Pearson ↑", "AUROC ↑", "AUPRC ↑", "F1 ↑", "RMSE log10 ↓"]

    for ax, metric, mlabel in zip(axes.flat, metrics_to_plot, metric_labels):
        # Group by (cond_id): cc0 vs cc1
        cond_ids = sorted(set(r["cond_id"] for r in test_rows))
        cc0_vals = []
        cc1_vals = []
        xlabels = []
        for cid in cond_ids:
            c = CONDITIONS[cid]
            h = "mcf" if c["head_type"] == "mhcflurry" else "hlg"
            max_k = c["max_nM"] // 1000
            xlabels.append(f"d{c['embed_dim']}\n{h}\n{max_k}k")
            v0 = [r[metric] for r in test_rows if r["cond_id"] == cid and r["cc"] == 0]
            v1 = [r[metric] for r in test_rows if r["cond_id"] == cid and r["cc"] == 1]
            cc0_vals.append(v0[0] if v0 else 0)
            cc1_vals.append(v1[0] if v1 else 0)

        x = np.arange(len(cond_ids))
        w = 0.35
        ax.bar(x - w/2, cc0_vals, w, label="cc0", color="#4477AA", alpha=0.8)
        ax.bar(x + w/2, cc1_vals, w, label="cc1", color="#EE6677", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=6)
        ax.set_title(mlabel, fontsize=11)
        ax.legend(fontsize=8)
        # Tight y-axis
        all_vals = cc0_vals + cc1_vals
        if metric != "rmse_log10":
            ymin = min(all_vals) - 0.02
            ymax = max(all_vals) + 0.01
        else:
            ymin = min(all_vals) - 0.02
            ymax = max(all_vals) + 0.02
        ax.set_ylim(ymin, ymax)

    fig.suptitle("V6 Factorial: Test Metrics (cc0 vs cc1)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "test_metrics_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / 'test_metrics_comparison.png'}")

    # --- 5b. CC delta by head type ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for head_type, color, marker in [("mhcflurry", "#4477AA", "o"), ("hlgauss", "#EE6677", "s")]:
        deltas = []
        labels = []
        for cid in sorted(CONDITIONS.keys()):
            c = CONDITIONS[cid]
            if c["head_type"] != head_type:
                continue
            v0 = [r["spearman"] for r in test_rows if r["cond_id"] == cid and r["cc"] == 0]
            v1 = [r["spearman"] for r in test_rows if r["cond_id"] == cid and r["cc"] == 1]
            if v0 and v1:
                deltas.append(v1[0] - v0[0])
                max_k = c["max_nM"] // 1000
                labels.append(f"d{c['embed_dim']}/{max_k}k")
        ax.scatter(range(len(deltas)), deltas, c=color, marker=marker, s=80, label=head_type, zorder=5)
        for i, (d, l) in enumerate(zip(deltas, labels)):
            ax.annotate(l, (i, d), fontsize=7, ha="center", va="bottom" if d > 0 else "top")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Spearman delta (cc1 − cc0)")
    ax.set_title("Content-Conditioning Effect by Head Type")
    ax.legend()
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cc_delta_by_head.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / 'cc_delta_by_head.png'}")

    # --- 5c. Probe trajectories ---
    peptides = sorted(set(r["peptide"] for r in all_probe_rows))
    alleles = sorted(set(r["allele"] for r in all_probe_rows))

    fig, axes = plt.subplots(len(peptides), len(alleles), figsize=(14, 4 * len(peptides)),
                              squeeze=False)

    # Color by (head_type, cc): 4 combos
    style_map = {
        ("mhcflurry", 0): ("#4477AA", "-"),
        ("mhcflurry", 1): ("#4477AA", "--"),
        ("hlgauss", 0): ("#EE6677", "-"),
        ("hlgauss", 1): ("#EE6677", "--"),
    }

    for pi, peptide in enumerate(peptides):
        for ai, allele in enumerate(alleles):
            ax = axes[pi][ai]
            # Group by condition
            for (cond_id, cc), run_dir in sorted(runs.items()):
                c = CONDITIONS[cond_id]
                color, ls = style_map[(c["head_type"], cc)]
                # Thin alpha for non-d32 to reduce clutter
                alpha = 0.9 if c["embed_dim"] == 32 else 0.3
                lw = 1.5 if c["embed_dim"] == 32 else 0.7

                epochs_data = [(r["epoch"], r["ic50_nM"])
                               for r in all_probe_rows
                               if r["cond_id"] == cond_id and r["cc"] == cc
                               and r["peptide"] == peptide and r["allele"] == allele]
                if not epochs_data:
                    continue
                epochs_data.sort()
                ep, ic50 = zip(*epochs_data)
                label = None
                if c["embed_dim"] == 32 and c["max_nM"] == 50_000:
                    h = "mcf" if c["head_type"] == "mhcflurry" else "hlg"
                    cc_s = "cc1" if cc else "cc0"
                    label = f"{h} {cc_s}"
                ax.plot(ep, ic50, color=color, linestyle=ls, alpha=alpha, linewidth=lw, label=label)

            ax.set_yscale("log")
            ax.set_title(f"{peptide} / {allele.split('*')[1]}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("IC50 (nM)")
            ax.axhline(500, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
            if pi == 0 and ai == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Probe IC50 Trajectories (bold=d32/50k, faint=other dims)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "probe_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / 'probe_trajectories.png'}")

    # --- 5d. Final-epoch probe heatmap ---
    # Rows: conditions, Columns: peptide × allele
    col_labels = [f"{p[:3]}/{a.split('*')[1]}" for p in peptides for a in alleles]
    row_labels = [short_label(cid, cc) for (cid, cc) in sorted(runs.keys())]

    matrix = np.zeros((len(runs), len(col_labels)))
    for ri, (cond_id, cc) in enumerate(sorted(runs.keys())):
        for ci, (p, a) in enumerate([(p, a) for p in peptides for a in alleles]):
            vals = [r["ic50_nM"] for r in final_probe_rows
                    if r["cond_id"] == cond_id and r["cc"] == cc
                    and r["peptide"] == p and r["allele"] == a]
            matrix[ri, ci] = vals[0] if vals else np.nan

    fig, ax = plt.subplots(figsize=(10, 16))
    im = ax.imshow(np.log10(matrix + 1), aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    # Annotate cells with IC50 values
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            v = matrix[ri, ci]
            if not np.isnan(v):
                txt = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
                ax.text(ci, ri, txt, ha="center", va="center", fontsize=5,
                        color="white" if np.log10(v + 1) > 3 else "black")
    ax.set_title("Final-Epoch Probe IC50 (nM)", fontsize=12)
    plt.colorbar(im, ax=ax, label="log10(IC50 + 1)", shrink=0.6)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "probe_final_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / 'probe_final_heatmap.png'}")

    # --- 5e. Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for (cond_id, cc), run_dir in sorted(runs.items()):
        metrics = load_metrics(run_dir)
        c = CONDITIONS[cond_id]
        color, ls = style_map[(c["head_type"], cc)]
        alpha = 0.9 if c["embed_dim"] == 32 else 0.25
        lw = 1.5 if c["embed_dim"] == 32 else 0.6

        epochs = [m["epoch"] for m in metrics]
        val_loss = [m["val_loss"] for m in metrics]
        val_spearman = [m["val_spearman"] for m in metrics]

        label = None
        if c["embed_dim"] == 32 and c["max_nM"] == 50_000:
            h = "mcf" if c["head_type"] == "mhcflurry" else "hlg"
            cc_s = "cc1" if cc else "cc0"
            label = f"{h} {cc_s}"

        axes[0].plot(epochs, val_loss, color=color, linestyle=ls, alpha=alpha, linewidth=lw, label=label)
        axes[1].plot(epochs, val_spearman, color=color, linestyle=ls, alpha=alpha, linewidth=lw, label=label)

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)

    axes[1].set_title("Validation Spearman")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Spearman")
    axes[1].legend(fontsize=7)

    fig.suptitle("V6 Training Curves (bold=d32/50k, faint=other)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / 'training_curves.png'}")

    # ── 6. Print summary tables to stdout ───────────────────────

    print("\n" + "=" * 80)
    print("FINAL-EPOCH PROBE IC50 (nM) — SLLQHLIGL")
    print("=" * 80)
    print(f"{'condition':<35s} {'A*02:01':>10s} {'A*24:02':>10s}")
    print("-" * 60)
    for (cond_id, cc) in sorted(runs.keys()):
        label = short_label(cond_id, cc)
        for allele in alleles:
            vals = [r["ic50_nM"] for r in final_probe_rows
                    if r["cond_id"] == cond_id and r["cc"] == cc
                    and r["peptide"] == "SLLQHLIGL" and r["allele"] == allele]
            if allele == alleles[0]:
                a02 = vals[0] if vals else float("nan")
            else:
                a24 = vals[0] if vals else float("nan")
        print(f"{label:<35s} {a02:>10.1f} {a24:>10.1f}")

    print("\n" + "=" * 80)
    print("TEST-SET BINARY METRICS (<=500nM threshold)")
    print("=" * 80)
    print(f"{'condition':<35s} {'acc':>7s} {'bal_acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'auroc':>7s} {'auprc':>7s}")
    print("-" * 90)
    for r in test_rows:
        label = short_label(r["cond_id"], r["cc"])
        print(f"{label:<35s} {r['accuracy']:>7.4f} {r['balanced_accuracy']:>7.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['auroc']:>7.4f} {r['auprc']:>7.4f}")

    print("\n" + "=" * 80)
    print("TEST-SET CORRELATION METRICS")
    print("=" * 80)
    print(f"{'condition':<35s} {'spearman':>10s} {'pearson':>10s} {'rmse_lg10':>10s}")
    print("-" * 70)
    for r in test_rows:
        label = short_label(r["cond_id"], r["cc"])
        print(f"{label:<35s} {r['spearman']:>10.4f} {r['pearson']:>10.4f} {r['rmse_log10']:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
