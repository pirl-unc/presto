"""Generate comparison artifacts from independently checked saved predictions."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    verified = json.loads((ROOT / "results/independent-verification.json").read_text())
    assert verified["status"] == "passed"
    summary = json.loads((ROOT / "results/fits/summary.json").read_text())
    rows = []
    for name, populations in verified["metrics"].items():
        for population, metrics in populations.items():
            rows.append({"model": name, "population": population, **metrics})
    with (ROOT / "results/comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    methods = [
        "calibrated_precursor",
        "uniform_exact",
        "residue_exact",
        "sequence_0.1",
        "augmented_exact",
        "augmented_sampled",
    ]
    names = [
        "Condition + calibrated precursor",
        "Uniform trimming",
        "Residue-specific trimming",
        "Frozen sequence baseline",
        "Sequence + exact trimming",
        "Sequence + sampled trimming",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    positions = np.arange(len(methods))
    for pop, offset, color in [("train", -0.16, "#9bbbd0"), ("inner_cv", 0.16, "#225b7d")]:
        axes[0].barh(
            positions + offset,
            [verified["metrics"][m][pop]["rmse"] for m in methods],
            height=0.29,
            label="Training" if pop == "train" else "Grouped inner CV",
            color=color,
        )
    axes[0].set_yticks(positions, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (log10 KO / WT elution signal)")
    axes[0].set_xlim(0, 0.76)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    exact = summary["arms"]["augmented_exact"]["fits"]
    sampled = summary["arms"]["augmented_sampled"]["fits"]
    axes[1].bar(
        ["Exact", "Sampled\nK = 512"],
        [sum(r["seconds"] for r in records) for records in [exact, sampled]],
        color=["#225b7d", "#be783a"],
    )
    axes[1].set_ylabel("Fit seconds: full training + five folds")
    axes[1].set_title("Same augmented model and objective")
    fig.suptitle("ERAP1: trimming does not improve the sequence baseline")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "results/comparison.png", dpi=170)
    fig.savefig(ROOT / "results/comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
