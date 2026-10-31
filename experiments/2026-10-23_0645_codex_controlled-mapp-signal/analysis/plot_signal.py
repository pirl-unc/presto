"""Scientific report figure; all outcomes are training-component diagnostics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "results/signal-summary.json").read_text())
values = json.loads((ROOT / "results/training-complete-case-values.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
fig.suptitle("MAPP inhibitor panel: selected training peptides only", fontsize=15)
ax = axes[0]
totals = np.array(summary["complete_case_signal_sum_by_arm_run"]) / 1e9
for i, label in enumerate(("Untreated", "Epoxomicin")):
    ax.scatter(
        np.full(3, i) + np.array([-0.12, 0, 0.12]),
        totals[i],
        s=50,
        label=label,
        color=["#226e9c", "#b66322"][i],
    )
    ax.plot([i - 0.2, i + 0.2], [totals[i].mean()] * 2, color="black")
ax.set_xticks([0, 1], ["Untreated", "Epoxomicin"])
ax.set_ylabel("Sum of measured signal (billions, arbitrary units)")
ax.set_title("59-peptide fixed support")
ax = axes[1]
ax.hist(
    [r["raw_mean_log10_shift"] for r in values],
    bins=15,
    color="#226e9c",
    edgecolor="white",
)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Mean log10(epoxo / untreated)")
ax.set_ylabel("Number of peptides")
ax.set_title("Selected peptide responses vary")
ax = axes[2]
folds = summary["raw_signal"]["folds"]
for field, label, color in (
    ("common_rmse_log10", "Common shift", "#b66322"),
    ("specific_rmse_log10", "Peptide-specific means", "#226e9c"),
):
    ax.plot(range(1, 10), [f[field] for f in folds], "o-", label=label, color=color)
ax.set_xlabel("Leave-one-run-per-arm-out comparison")
ax.set_ylabel("RMSE (log10 signal shift)")
ax.set_title("Acquisition repeatability")
ax.set_xticks([1, 3, 5, 7, 9])
ax.legend(frameon=False, fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(ROOT / "results/training-repeatability.png", dpi=160)
fig.savefig(ROOT / "results/training-repeatability.pdf")
