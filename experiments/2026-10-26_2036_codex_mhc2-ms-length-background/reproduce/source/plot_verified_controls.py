"""Plot saved independently verified metrics; no new scoring or selection."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = Path(__file__).resolve().parents[1]


def main():
    result = json.loads(
        (FAMILY / "results/logistic-controls/independent-verification.json").read_text()
    )
    assert result["status"] == "passed"
    keys = ["binding", "sequence", "combined", "native101_diagnostic"]
    labels = ["Binding", "Sequence", "Combined", "Native, seed 101"]
    colors = ["#7a8fa6", "#9eafbf", "#244765", "#c1663b"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for axis, partition, title in zip(
        axes, ["test", "external"], ["DP test", "DQ8 external"], strict=True
    ):
        values = [result["metrics"][key][partition]["auroc"] for key in keys]
        axis.barh(labels, values, color=colors)
        axis.set_ylim(len(labels) - 0.5, -0.5)
        axis.set_xlim(0, 1)
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("AUROC")
        axis.axvline(0.5, color="#555555", linestyle="--", linewidth=0.8)
        for row, value in enumerate(values):
            axis.text(value + 0.015, row, f"{value:.3f}", va="center")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("First native seed transfers worse than the combined control", fontsize=14)
    figure.text(
        0.5,
        0.025,
        "Observed positives vs synthetic backgrounds; second seed and permutation still pending",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.92))
    for extension in ("png", "svg"):
        destination = FAMILY / f"results/logistic-controls/verified-auroc.{extension}"
        figure.savefig(destination, dpi=180)
        if extension == "svg":
            destination.write_text(
                "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
            )
    plt.close(figure)


if __name__ == "__main__":
    main()
