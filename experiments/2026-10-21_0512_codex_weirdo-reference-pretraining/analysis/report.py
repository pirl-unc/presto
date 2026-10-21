"""Render saved comparison results without evaluating any model or changing selection."""

import csv
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
RESULTS = FAMILY / "results"
report = json.loads((RESULTS / "comparison.json").read_text())
raw = ROOT / "artifacts" / FAMILY.name / "fit"
metrics = report["metrics"]
rows = []
for condition, splits in metrics.items():
    for split, split_metrics in splits.items():
        for category, values in split_metrics["categories"].items():
            rows.append(
                {
                    "condition": condition,
                    "split": split,
                    "category": category,
                    **{
                        key: values[key]
                        for key in (
                            "n",
                            "positives",
                            "negatives",
                            "prevalence",
                            "ce",
                            "accuracy",
                            "balanced_accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "auroc",
                            "auprc",
                            "brier",
                            "ece_10_bins",
                            "probability_sd",
                            "tp",
                            "tn",
                            "fp",
                            "fn",
                        )
                    },
                }
            )
with (RESULTS / "category-metrics.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
with (RESULTS / "overall-metrics.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["condition", "split", "n", "macro_ce"])
    for condition, splits in metrics.items():
        for split, values in splits.items():
            writer.writerow([condition, split, values["n"], values["macro_ce"]])
for name in (
    "history.json",
    "coverage.json",
    "summary.json",
    "resolved-config.json",
    "admission.json",
):
    shutil.copyfile(raw / name, RESULTS / f"network-{name}")
shutil.copyfile(FAMILY / "reproduce/fit-launch.json", FAMILY / "reproduce/launch.json")
history = json.loads((raw / "history.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
ax = axes[0]
ax.plot(
    [x["epoch"] for x in history], [x["training_loss"] for x in history], label="Training batches"
)
ax.plot(
    [x["epoch"] for x in history],
    [sum(y["loss"] for y in x["validation"].values()) / len(x["validation"]) for x in history],
    label="Validation",
)
ax.axvline(
    report["network_training"]["selected_epoch"],
    linestyle=":",
    color="black",
    label="Selected epoch",
)
ax.set(title="Network optimization", xlabel="Complete epoch", ylabel="Mean binary cross entropy")
ax.legend(frameon=False)
labels = {
    "training_prevalence": "Prevalence",
    "composition_logistic": "Composition",
    "modular_peptide_source": "Network",
}
ax = axes[1]
for condition, values in metrics.items():
    ax.plot(
        [0, 1],
        [values[s]["macro_ce"] for s in ("validation", "test")],
        marker="o",
        label=labels[condition],
    )
ax.set(
    xticks=[0, 1],
    xticklabels=["Validation", "Test"],
    ylabel="Mean binary cross entropy",
    title="Held-out reference memberships",
)
ax.legend(frameon=False)
fig.suptitle("Network improvement over composition was not established", fontsize=13)
fig.savefig(RESULTS / "comparison.png", dpi=180)
plt.close(fig)
