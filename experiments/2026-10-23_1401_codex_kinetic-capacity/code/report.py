"""Separate fitted training errors from necessary-constraint lower bounds."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = Path(__file__).resolve().parents[1]


def main():
    fits = json.loads(
        (FAMILY / "results/independent-fit-verification.json").read_text()
    )["results"]
    shapes = json.loads((FAMILY / "results/shape-metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    axes[0].bar(
        [
            "Regularized\nsequence reference",
            "Unregularized\nsequence",
            "Free condition\nrates",
        ],
        [
            0.9025656732693761,
            fits["unregularized_sequence"]["metrics"]["rmse"],
            fits["free_condition_rates"]["metrics"]["rmse"],
        ],
        color=["#999999", "#3986a7", "#c47d45"],
    )
    axes[0].set_title("Fitted processes: two new fits not converged")
    axes[1].bar(
        [
            "Immediate-cut\nshape constraints",
            "Entry-model\nshape constraints",
            "Replicate/time\nmeans",
        ],
        [
            shapes[v]["rmse"]
            for v in ("immediate_shape", "entry_shape", "replicate_mean")
        ],
        color=["#3986a7", "#c47d45", "#999999"],
    )
    axes[1].set_title("Per-curve lower bounds: not predictive models")
    for ax in axes:
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Training RMSE (log10 temporal ratio)")
        ax.spines[["top", "right"]].set_visible(False)
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=10,
            )
    fig.suptitle(
        "Training-capacity diagnostic — both previous held-out strata excluded",
        fontsize=13,
    )
    fig.savefig(FAMILY / "results/comparison.png", dpi=160)
    fig.savefig(FAMILY / "results/comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
