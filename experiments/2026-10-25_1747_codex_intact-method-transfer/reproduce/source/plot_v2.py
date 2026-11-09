"""Visualize frozen primary transfer; all numerical results already archived."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

root = Path(__file__).resolve().parents[1]
assert (
    json.loads((root / "results/score-independent-replay.json").read_text())["status"]
    == "passed"
)
metrics = json.loads((root / "results/metrics.json").read_text())
primary = {
    r["arm"]: r for r in metrics if r["split"] == "test" and r["stratum"] == "pooled"
}
summary = json.loads((root / "results/score-summary.json").read_text())
rows = [
    r
    for r in pq.read_table(root / "results/predictions.parquet").to_pylist()
    if r["arm"] == "sequence" and r["split"] == "test"
]
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.6, 4.7),
    gridspec_kw={"width_ratios": [1.15, 1]},
    layout="constrained",
)
arms = [
    "zero",
    "sequence",
    "length",
    "shared_slope",
    "physical_uniform",
    "physical_residue_penalized",
    "physical_residue_unpenalized",
]
labels = [
    "No change",
    "Sequence + time",
    "Length + time",
    "Shared time slope",
    "Physical: uniform",
    "Physical: residue, penalty",
    "Physical: residue, no penalty",
]
for i, arm in enumerate(arms):
    value = primary[arm]["rmse"]
    lo, hi = summary["bootstrap"]["percentile_95_intervals"][arm]
    axes[0].plot([lo, hi], [i, i], color="#7a8693", lw=2)
    axes[0].scatter(
        [value],
        [i],
        color="#0b7285"
        if arm == "zero"
        else "#c04e35"
        if arm == "sequence"
        else "#46556a",
        s=38,
        zorder=3,
    )
axes[0].set_yticks(range(len(arms)), labels)
axes[0].invert_yaxis()
axes[0].set_xlabel("Primary RMSE in log10 ratio change (lower is better)")
axes[0].set_title("Frozen models fail to beat no change", loc="left", fontweight="bold")
axes[0].grid(axis="x", alpha=0.2)
colors = {"Brain": "#206c9e", "Liver": "#b65c2f", "Spleen": "#587b44"}
for organ, color in colors.items():
    chosen = [r for r in rows if r["organ_label"] == organ]
    axes[1].scatter(
        [r["target_change"] for r in chosen],
        [r["prediction"] for r in chosen],
        s=22,
        alpha=0.65,
        label=organ,
        color=color,
        edgecolors="none",
    )
values = [r[k] for r in rows for k in ("target_change", "prediction")]
lo = min(values) - 0.05
hi = max(values) + 0.05
axes[1].plot([lo, hi], [lo, hi], ls="--", color="#6c757d", lw=1)
axes[1].axhline(0, color="#adb5bd", lw=0.7)
axes[1].axvline(0, color="#adb5bd", lw=0.7)
axes[1].set(
    xlim=(lo, hi),
    ylim=(lo, hi),
    xlabel="Observed change from supplied zero-time baseline",
    ylabel="Sequence model predicted change",
)
axes[1].set_title(
    "Sequence variation, weak transfer", loc="left", fontweight="bold"
)
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle(
    "PXD058121 organ transfer: 236 cells at 30/60 minutes; fixed predictors",
    fontsize=13,
)
fig.savefig(root / "results/primary-transfer-v2.png", dpi=180)
fig.savefig(root / "results/primary-transfer-v2.pdf")
plt.close(fig)
print(
    json.dumps(
        {
            "status": "passed",
            "figures": ["primary-transfer-v2.png", "primary-transfer-v2.pdf"],
            "source": "verified frozen primary predictions",
            "new_fits": 0,
        }
    )
)
