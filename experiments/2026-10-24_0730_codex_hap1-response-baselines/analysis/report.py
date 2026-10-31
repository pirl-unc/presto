"""Render the fixed comparison from saved outputs without fitting or selecting new models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment / "results"
    summary = json.loads((root / "fits/summary.json").read_text())
    strata = json.loads((root / "fits/strata.json").read_text())
    variants = list(summary["metrics"])
    labels = {
        "zero": "No change",
        "condition": "Condition",
        "fixed_precursor": "Condition + precursor (fixed 1)",
        "calibrated_precursor": "Condition + precursor (calibrated)",
        "sequence_0.001": "Sequence ridge 0.001",
        "sequence_0.01": "Sequence ridge 0.01",
        "sequence_0.1": "Sequence ridge 0.1",
    }
    rows = []
    for variant in variants:
        for population, values in summary["metrics"][variant].items():
            if isinstance(values, dict):
                rows.append({"variant": variant, "population": population, **values})
    with (root / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    train = [summary["metrics"][v]["train"]["rmse"] for v in variants]
    cv = [summary["metrics"][v]["inner_cv"]["rmse"] for v in variants]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    positions = np.arange(len(variants))
    axes[0].barh(positions - 0.17, train, 0.3, label="Training", color="#aab6bd")
    axes[0].barh(positions + 0.17, cv, 0.3, label="Grouped inner CV", color="#1976a3")
    axes[0].set_yticks(positions, [labels[v] for v in variants])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (log10 knockout / wildtype)")
    axes[0].set_title("Fixed biological baselines")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    a, b = summary["best_control"], summary["best_sequence"]
    values = {
        r["variant"] + r["stratum"]: r["rmse"]
        for r in strata
        if r["population"] == "inner_cv" and r["stratum"].startswith("condition:")
    }
    conditions = sorted(
        key.removeprefix("condition:") for key in summary["sequence_within_condition_sd"]
    )
    for i, condition in enumerate(conditions):
        left, right = values[a + "condition:" + condition], values[b + "condition:" + condition]
        axes[1].plot([left, right], [i, i], color="#b3b3b3", zorder=1)
        axes[1].scatter(
            left, i, color="#666666", marker="s", label="Best control" if i == 0 else None
        )
        axes[1].scatter(right, i, color="#1976a3", label="Best sequence" if i == 0 else None)
    axes[1].set_yticks(
        range(len(conditions)),
        [c.removeprefix("hap1_").removesuffix("_ko").upper() for c in conditions],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Inner-CV RMSE (log10 ratio)")
    axes[1].set_title("Across ten machinery knockouts")
    axes[1].legend(frameon=False)
    fig.suptitle("HAP1 eluted peptides — development only; outer validation/test remain closed")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.savefig(root / "comparison.png", dpi=160)
    fig.savefig(root / "comparison.pdf")
    print("Wrote metrics.csv and comparison.png/pdf")


if __name__ == "__main__":
    main()
