"""Render the fixed comparison after verification; no fitting or selection."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = Path(__file__).resolve().parents[1]


def main():
    verified = json.loads((FAMILY / "results/independent-verification.json").read_text())
    assert verified["status"] == "passed"
    summary = json.loads((FAMILY / "results/evaluation/summary.json").read_text())
    rows = summary["metrics"]
    columns = [
        "variant",
        "split",
        "data_loss",
        "exact_rmse",
        "exact_mae",
        "exact_r2",
        "exact_pearson_pooled",
        "exact_spearman_pooled",
        "prediction_sd_pooled",
        "bound_satisfaction_weighted",
        "bound_violation_rmse",
    ]
    with (FAMILY / "results/comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    labels = {
        "no_change": "No change",
        "exact_train_mean": "Exact training mean",
        "censored_constant": "Censored constant",
        "condition": "Condition only",
        "sequence_shared": "Direct shared sequence",
        "sequence_compound": "Direct compound × sequence",
        "uniform_exact": "Uniform cleavage · exact",
        "uniform_hybrid": "Uniform cleavage · sampled",
        "sequence_exact": "Sequence cleavage · exact",
        "sequence_hybrid": "Sequence cleavage · sampled",
    }
    variants = [v for v in labels if any(r["variant"] == v for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, layout="constrained")
    for i, split in enumerate(("val", "test")):
        for j, (metric, title) in enumerate(
            (
                ("data_loss", "Exact + bounded loss"),
                ("exact_rmse", "Exact-value RMSE (log10 ratio)"),
            )
        ):
            ax = axes[i, j]
            values = [
                next(r[metric] for r in rows if r["variant"] == v and r["split"] == split)
                for v in variants
            ]
            colors = [
                "#2166ac"
                if v.endswith(("_exact", "_hybrid"))
                else "#d69d3c"
                if v == "condition"
                else "#aeb7c1"
                for v in variants
            ]
            ax.barh(range(len(variants)), values, color=colors)
            ax.set_yticks(range(len(variants)), [labels[v] for v in variants])
            ax.set_title(f"{'Validation' if split == 'val' else 'Test'} · {title}")
            ax.set_xlim(0, max(values) * 1.17)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            for k, value in enumerate(values):
                ax.text(value + max(values) * 0.012, k, f"{value:.3f}", va="center", fontsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    axes[0, 0].invert_yaxis()
    fig.suptitle(
        "Native-cell peptide treatment response: fixed held-out-parent comparison\nLower is better; this is not APC concentration prediction",
        fontsize=13,
    )
    fig.savefig(FAMILY / "results/comparison.png", dpi=180)
    fig.savefig(FAMILY / "results/comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
