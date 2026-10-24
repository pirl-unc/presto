"""Summaries and figures from completed fits; no fitting or model selection."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
metrics = json.loads((R / "metrics.json").read_text())
models = json.loads((R / "models.json").read_text())
with (R / "comparison.csv").open("w", newline="") as stream:
    fields = [
        "variant",
        "split",
        "n_peptides",
        "n_components",
        "reverse_kl",
        "forward_kl",
        "total_variation",
        "fold_change_rmse",
        "fold_change_pearson",
        "fold_change_spearman",
        "fold_change_prediction_sd",
        "intensity_rmse",
        "intensity_pearson",
        "intensity_spearman",
    ]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for m in metrics:
        row = {k: m[k] for k in fields[:7]}
        for target in ("fold_change", "intensity"):
            for name in ("rmse", "pearson", "spearman", "prediction_sd"):
                key = target + "_" + name
                if key in fields:
                    row[key] = m["log10_" + target][name]
        writer.writerow(row)
groups = ["common_shift", "exact", "reinforce32", "reinforce128", "ridge"]
aggregates = {}
for group in groups:
    aggregate = {}
    for split in ("train", "val", "test"):
        rows = [
            m
            for m in metrics
            if m["split"] == split
            and (m["variant"] == group or m["variant"].startswith(group + "_"))
        ]
        aggregate[split] = {
            "n_runs": len(rows),
            **{
                field: {
                    "mean": float(np.mean(values)),
                    "seed_population_sd": float(np.std(values)),
                }
                for field, values in {
                    "reverse_kl": [m["reverse_kl"] for m in rows],
                    "fold_change_rmse": [m["log10_fold_change"]["rmse"] for m in rows],
                    "prediction_sd": [
                        m["log10_fold_change"]["prediction_sd"] for m in rows
                    ],
                }.items()
            },
        }
    aggregates[group] = aggregate
(R / "aggregate-metrics.json").write_text(json.dumps(aggregates, indent=2) + "\n")
rows = [
    r
    for r in json.loads((R / "contrast-examples.json").read_text())
    if r["split"] == "train"
]
source = {r["peptide_row_id"]: r for r in rows}
scaling = json.loads((R / "feature-scaling.json").read_text())
import gzip

with gzip.open(R / "predictions.jsonl.gz", "rt") as stream:
    train = [json.loads(line) for line in stream if '"split": "train"' in line]
train = [r for r in train if r["variant"] == "exact"]
x = np.array([r["features_standardized"] for r in train])
u = np.array([r["untreated_mean_input"] for r in train])
e = np.array([r["treated_mean_target"] for r in train])
model = next(m for m in models if m["variant"] == "exact")
history = json.loads((R / "runs/exact/history.json").read_text())
curves = []
for h in history:
    logp = np.log(u / u.sum()) + x @ np.array(h["weights"])
    logp -= np.logaddexp.reduce(logp)
    pred = (np.log(u.sum()) + model["beta"] + logp) / np.log(10)
    curves.append(
        {
            "update": h["update"],
            "objective": h["objective"],
            "reverse_kl": h["train_reverse_kl"],
            "training_log10_response_rmse": float(
                np.sqrt(np.mean((pred - np.log10(e)) ** 2))
            ),
        }
    )
q = e / e.sum()
(R / "training-objective-diagnostic.json").write_text(
    json.dumps(
        {
            "scope": "posthoc training-only interpretation of saved history; no new fit or selection",
            "n_training_peptides": len(e),
            "target_intensity_effective_support_1_over_sum_squared": float(1 / (q @ q)),
            "top_target_peptide_mass": float(q.max()),
            "top_five_target_peptide_mass": float(np.sort(q)[-5:].sum()),
            "curves": curves,
        },
        indent=2,
    )
    + "\n"
)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
labels = ["Common\nshift", "Exact", "REINFORCE\n32", "REINFORCE\n128", "Ridge"]
colors = ["#888888", "#277da8", "#5a9e75", "#7fb994", "#ce8932"]
for ax, split, label in zip(
    axes[0],
    ("val", "test"),
    ("Validation: 14 peptides / 5 components", "Test: 14 peptides / 9 components"),
):
    values = [aggregates[g][split]["fold_change_rmse"]["mean"] for g in groups]
    sd = [
        aggregates[g][split]["fold_change_rmse"]["seed_population_sd"] for g in groups
    ]
    ax.bar(range(len(groups)), values, yerr=sd, color=colors, capsize=3)
    ax.set_xticks(range(len(groups)), labels)
    ax.set_title(label)
    ax.set_ylabel("Response RMSE (log10 units; lower is better)")
    ax.set_ylim(0, 1.25)
axes[1, 0].plot(
    [c["update"] for c in curves], [c["objective"] for c in curves], color="#277da8"
)
axes[1, 0].set_title("Exact training objective improves")
axes[1, 0].set_ylabel("Reverse KL + coefficient penalty")
axes[1, 1].plot(
    [c["update"] for c in curves],
    [c["training_log10_response_rmse"] for c in curves],
    color="#277da8",
)
axes[1, 1].axhline(
    curves[0]["training_log10_response_rmse"],
    color="#888888",
    linestyle="--",
    label="Common shift",
)
axes[1, 1].set_title("Training response error worsens")
axes[1, 1].set_ylabel("Response RMSE (log10 units)")
axes[1, 1].legend(frameon=False)
for ax in axes[1]:
    ax.set_xlabel("Optimizer update")
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "Controlled MAPP response: no model passes both held-out gates", fontsize=15
)
fig.savefig(R / "comparison.png", dpi=160)
fig.savefig(R / "comparison.pdf")
