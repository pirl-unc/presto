"""Render the qualified fixed-fit comparison; no fitting or test access."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
results = FAMILY / "results"
summary = json.loads((results / "qualified-v2/summary.json").read_text())
verification = json.loads((results / "independent-verification.json").read_text())
assert verification["status"] == "passed"
names = [f["variant"] for f in summary["fits"]]
labels = ["No change", "Exact train mean", "Censored constant", "Condition only", "Shared sequence", "Compound × sequence"]
by_key = {(r["variant"], r["split"]): r for r in summary["metrics"]}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
position = np.arange(len(names))
for ax, field, ylabel in zip(axes, ("data_loss", "exact_rmse"), ("Exact/censored weighted loss", "Exact-value RMSE (log10 ratio)")):
    for shift, split, label, color in ((-.18, "train", "Train", "#718aab"), (.18, "val", "Validation", "#c76b32")):
        ax.bar(position + shift, [by_key[n, split][field] for n in names], width=.34, label=label, color=color)
    ax.set_xticks(position, labels, rotation=36, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False)
fig.suptitle("Native-cell treatment response: sequence models fit train better, generalize worse")
fig.savefig(results / "comparison.png", dpi=170)
plt.close(fig)
with (results / "qualified-v2/metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in summary["metrics"]))))
    writer.writeheader()
    writer.writerows(summary["metrics"])
strata = json.loads((results / "qualified-v2/stratified-metrics.json").read_text())
counts = {}
for variant in ("sequence_shared", "sequence_compound"):
    counts[variant] = {}
    for field in ("condition_id", "component_id"):
        control = {r["stratum_id"]: r["data_loss"] for r in strata if r["variant"] == "condition" and r["split"] == "val" and r["stratum"] == field}
        scores = {r["stratum_id"]: r["data_loss"] for r in strata if r["variant"] == variant and r["split"] == "val" and r["stratum"] == field}
        counts[variant][field] = {"better": sum(scores[k] < v for k, v in control.items()), "worse": sum(scores[k] > v for k, v in control.items()), "tie": sum(scores[k] == v for k, v in control.items())}
(results / "qualified-v2/stratum-comparison.json").write_text(json.dumps(counts, indent=2) + "\n")
print(json.dumps({"verification": verification, "validation": [by_key[n, "val"] for n in names], "stratum_comparison": counts}, indent=2))
