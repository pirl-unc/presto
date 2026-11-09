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
        "reference_uniform_exact",
        "length_exact",
        "reference_residue_exact",
        "residue_length_exact",
        "sequence_0.1",
        "reference_augmented_exact",
        "augmented_exact",
    ]
    names = [
        "Old uniform trimming",
        "Length-dependent trimming",
        "Old residue-dependent trimming",
        "Residue + length trimming",
        "Frozen sequence baseline",
        "Sequence + old trimming",
        "Sequence + length trimming",
    ]
    if "augmented_sampled" in summary["arms"]:
        methods.append("augmented_sampled")
        names.append("Sequence + sampled length trimming")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
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
    sampled = summary["arms"].get("augmented_sampled", {}).get("fits", [])
    axes[1].bar(
        ["Exact", "Sampled\nK = 512"],
        [sum(r["seconds"] for r in records) for records in [exact, sampled]],
        color=["#225b7d", "#be783a"],
    )
    axes[1].set_ylabel("Fit seconds: full training + five folds")
    axes[1].set_title("Same augmented model and objective")
    fig.suptitle("ERAP1: remaining-length dependence on the fixed development cohort")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "results/comparison.png", dpi=170)
    fig.savefig(ROOT / "results/comparison.pdf")
    plt.close(fig)
    parameters = []
    for arm, result in summary["arms"].items():
        for fit in result["fits"]:
            p = fit["parameter"]
            parameters.append(
                {
                    "arm": arm,
                    "fold": fit["fold"],
                    "short_log_rate_modifier": p[-3],
                    "long_log_rate_modifier": p[-2],
                    "effective_ko_scale": float(1 / (1 + np.exp(-p[-1]))),
                    "n_changed_parameters": fit["n_changed_parameters"],
                    "exact_gradient_inf": fit["exact_gradient_inf"],
                    "seconds": fit["seconds"],
                }
            )
    with (ROOT / "results/parameter-summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameters[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(parameters)


if __name__ == "__main__":
    main()
