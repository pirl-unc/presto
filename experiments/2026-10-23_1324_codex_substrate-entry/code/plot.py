"""Render frozen development outcomes without fitting or selecting a model."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]


def main():
    names = [
        "immediate_constant",
        "entry_constant",
        "immediate_sequence",
        "entry_sequence",
        "entry_sampled",
    ]
    labels = [
        "One rate",
        "One rate\n+ entry",
        "Sequence",
        "Sequence\n+ entry",
        "Sampled\n+ entry",
    ]
    scores = json.loads((FAMILY / "results/all-metrics.json").read_text())
    fits = {
        v: json.loads((FAMILY / f"results/fit-{v}.json").read_text()) for v in names
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    positions = np.arange(len(names))
    for offset, split, color, label in [
        (-0.19, "train", "#3986a7", "Training: 6 parents"),
        (0.19, "val", "#c47d45", "Development: 3 parents"),
    ]:
        axes[0].bar(
            positions + offset,
            [scores[v][split]["rmse"] for v in names],
            0.38,
            color=color,
            label=label,
        )
    axes[0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Equal-parent RMSE (log10 temporal ratio)")
    axes[0].set_title("Fixed-budget development comparison")
    axes[0].legend(frameon=False, fontsize=8)
    for v, color, label in [
        ("immediate_sequence", "#999999", "Immediate sequence"),
        ("entry_sequence", "#3986a7", "Entry: differentiable"),
        ("entry_sampled", "#c47d45", "Entry: sampled hybrid"),
    ]:
        h = fits[v]["history"]
        axes[1].plot(
            [r["step"] for r in h],
            [r["objective"] for r in h],
            color=color,
            label=label,
        )
    axes[1].set_xlabel("Optimizer updates")
    axes[1].set_ylabel("Training MSE + coefficient penalty")
    axes[1].set_title("Same objective; stochastic updates retain noise")
    axes[1].legend(frameon=False, fontsize=8)
    for v, color, label in [
        ("entry_constant", "#999999", "One rate + entry"),
        ("entry_sequence", "#3986a7", "Sequence + entry"),
        ("entry_sampled", "#c47d45", "Sampled sequence + entry"),
    ]:
        checkpoints = fits[v]["checkpoints"]
        steps = sorted(map(int, checkpoints))
        axes[2].plot(
            steps,
            [3600 / np.exp(checkpoints[str(i)][-1]) for i in steps],
            marker="o",
            color=color,
            label=label,
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Optimizer updates")
    axes[2].set_ylabel("Mean entry delay (seconds)")
    axes[2].set_title("Effective entry timescale")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Substrate-entry comparison — development only; previous test substrates excluded",
        fontsize=13,
    )
    fig.savefig(FAMILY / "results/comparison.png", dpi=160)
    fig.savefig(FAMILY / "results/comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
