"""Plot all training parents; preserve the negative LEDGF comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
rows = json.loads((root / "results/metrics.json").read_text())["by_parent"]
labels = [
    "Ffh (2 ratios)"
    if len(r["conditions"]) == 2
    else r["conditions"][0]["condition_id"].split("aSPIRE_")[-1]
    for r in rows
]
x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(8.8, 4.5), layout="constrained")
ax.bar(
    x - 0.18,
    [np.sqrt(r["no_change_mse"]) for r in rows],
    0.36,
    label="No temporal change",
    color="#a5afb8",
)
ax.bar(
    x + 0.18,
    [np.sqrt(r["cross_replicate_mse"]) for r in rows],
    0.36,
    label="Other biological label",
    color="#2c75a5",
)
ax.set_xticks(x, labels)
ax.set_ylabel("Temporal response RMSE (log10 ratio)")
ax.set_title("Training-only digestion repeatability: 5 of 6 parents improve")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.savefig(root / "results/comparison.png", dpi=160)
fig.savefig(root / "results/comparison.pdf")
