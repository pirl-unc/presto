"""Plot all seed outcomes without rerunning models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
summary = json.loads((FAMILY / "results/run/summary.json").read_text())
methods = ["ce", "l2", "multiset16", "multiset64", "naive"]
labels = ["Exact CE", "Exact L2", "Multiset 16", "Multiset 64", "Naive MSE"]
colors = ["#276b8e", "#a44a28"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for row, mechanism in enumerate(("positive", "blind")):
    for column, kind in enumerate(("observed", "latent")):
        ax = axes[row, column]
        for budget, offset, color in zip((32, 256), (-0.13, 0.13), colors):
            for index, method in enumerate(methods):
                values = [
                    r["metrics"]["test"][kind]["tv"]
                    for r in summary["conditions"]
                    if r["config"]["name"] == f"{mechanism}-n{budget}-{method}"
                ]
                ax.scatter(
                    np.array([index + offset] * 3) + np.linspace(-0.035, 0.035, 3),
                    values,
                    s=18,
                    color=color,
                    alpha=0.75,
                    label=f"{budget} observed counts" if index == 0 else None,
                )
                ax.plot(
                    [index + offset - 0.09, index + offset + 0.09],
                    [np.mean(values)] * 2,
                    color=color,
                    linewidth=2,
                )
        ax.set_title(f"{mechanism.capitalize()} detection · {kind} distribution")
        ax.set_xticks(range(5), labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, 0.13)
        if column == 0:
            ax.set_ylabel("Test total variation ↓")
        if row == 0 and column == 0:
            ax.legend(frameon=False, fontsize=9)
fig.suptitle(
    "Finite observations: exact losses outperform sampled multiset losses here", fontsize=14
)
fig.text(
    0.5,
    0.01,
    "Dots: three initialization seeds; bars: means. "
    "Generic integer-symbol simulation, not biological validation.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.035, 1, 0.96))
fig.savefig(FAMILY / "results/comparison.png", dpi=160)
plt.close(fig)
