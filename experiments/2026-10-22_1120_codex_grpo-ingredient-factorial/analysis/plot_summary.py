"""Readable summary figure from the independently checked condition table."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY = Path(__file__).resolve().parents[1]
rows = json.loads((FAMILY / "results/condition-means.json").read_text())
lookup = {row["condition"]: row for row in rows}
labels = {
    "exact-b0": "Exact",
    "original-reinforce8": "REINFORCE 8",
    "hybrid-starts2": "Hybrid 2 starts",
    "hybrid-starts8": "Hybrid 8 starts",
    "query-matched-c0-r4": "Reuse 4, same queries",
    "query-matched-c1-r4": "Reuse 4 + clip, same queries",
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
for index, name in enumerate(labels):
    row = lookup[name]
    axes[0].scatter(np.full(3, index), row["test_tv_original_seeds"], color="#175d9a", alpha=0.7)
axes[0].set(
    xticks=range(len(labels)),
    xticklabels=list(labels.values()),
    ylabel="Held-out total variation",
    title="Reference, hybrid and query-matched reuse",
)
axes[0].tick_params(axis="x", rotation=55, labelsize=8)
for std in (0, 1):
    for clip in (0, 1):
        values = [
            lookup[f"factorial-c{clip}-s{std}-r{reuse}-b0"]["test_tv_original"] for reuse in (1, 4)
        ]
        axes[1].plot(
            [1, 4],
            values,
            marker="o",
            linestyle="--" if clip else "-",
            color="#b96c15" if std else "#17629d",
            label=f"Std {'on' if std else 'off'}, clip {'on' if clip else 'off'}",
        )
axes[1].set(
    xticks=[1, 4],
    xlabel="Updates per collected batch",
    ylabel="Mean held-out total variation",
    title="Factorial at equal optimizer updates",
)
axes[1].legend(fontsize=8)
fig.savefig(FAMILY / "results/comparison-summary.png", dpi=170)
plt.close(fig)
