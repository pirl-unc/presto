"""Compact scientific figure from fixed, evaluated predictions; no selection."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data/measured_ms1_grouped_observer/2026-10-24_1702_canonical"
OUTPUT = ROOT / "results/evaluation"
summary = json.loads((OUTPUT / "summary.json").read_text())
shutil.copy2(__file__, ROOT / "reproduce/source/plot_results.py")
selected = summary["selection"]
models = [
    "zero",
    selected["primary_length"],
    selected["primary_sequence"],
    "nonlinear_ensemble",
]
labels = ["Zero", "Length ridge", "Sequence ridge", "Nonlinear ensemble"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
for index, split in enumerate(("validation", "test")):
    values = [summary["overall"][split][m]["rmse"] for m in models]
    axes[0].bar(
        np.arange(4) + (index - 0.5) * 0.38, values, width=0.38, label=split.title()
    )
axes[0].set_xticks(range(4), labels, rotation=25, ha="right")
axes[0].set_ylabel("RMSE: centered log10 measured MS1 signal")
axes[0].set_title("Frozen pool-held-out comparison")
axes[0].legend(frameon=False)
targets = np.array(
    [
        json.loads(line)["target"]
        for line in (DATA / "test-targets.jsonl").read_text().splitlines()
    ]
)
predictions = np.load(ROOT / "results/fit/test-predictions.npz", allow_pickle=False)
limits = [min(-2, float(targets.min())), max(2, float(targets.max()))]
for ax, name, label in zip(axes[1:], models[2:], labels[2:], strict=True):
    ax.hexbin(
        targets, predictions[name], gridsize=45, mincnt=1, bins="log", cmap="viridis"
    )
    ax.plot(limits, limits, color="gray", linestyle="--", linewidth=1)
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Measured centered log10 signal",
        ylabel="Predicted centered log10 signal",
    )
    scores = summary["overall"]["test"][name]
    ax.set_title(
        f"{label}: test\nR²={scores['r2']:.3f}, Spearman={scores['spearman']:.3f}"
    )
fig.suptitle(
    "PXD021013: conditional response in synthesized peptide pools\nUnknown purity/loading and synthesis effects remain confounded",
    fontsize=12,
)
fig.savefig(OUTPUT / "comparison.png", dpi=170)
fig.savefig(OUTPUT / "comparison.pdf")
plt.close(fig)
