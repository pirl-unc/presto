"""Close fixed outcomes with comparison tables and descriptive diagnostics."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((FAMILY / name).read_text())


def dump(name, value):
    (FAMILY / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    summary = read("results/metrics.json")
    summary["conditional_sequence_response"] = read(
        "results/conditional-fit-metrics.json"
    )
    order = [
        "constant_signal",
        "response_only",
        "time_only",
        "fixed_ratio",
        "fixed_ratio_with_response",
        "constant_response",
        "sequence_response",
        "conditional_sequence_response",
    ]
    rows = []
    within = []
    ranges = []
    per_time = []
    for name in order:
        for split, metrics in summary[name].items():
            rows.append(dict(variant=name, split=split, **metrics))
        conditions = defaultdict(list)
        with gzip.open(FAMILY / f"results/{name}-predictions.jsonl.gz", "rt") as stream:
            for line in stream:
                row = json.loads(line)
                conditions[row["condition_id"]].append(row)
        for condition, values in conditions.items():
            p = np.array([r["centered_prediction"] for r in values])
            y = np.array([r["centered_target"] for r in values])
            w = np.array([r["weight"] for r in values])
            w /= w.sum()
            rank = None if np.ptp(p) < 1e-12 else float(spearmanr(p, y).statistic)
            correlation = None if np.ptp(p) < 1e-12 else float(pearsonr(p, y).statistic)
            within.append(
                dict(
                    variant=name,
                    condition=condition,
                    split=values[0]["split"],
                    n=len(values),
                    shape_rmse=float(np.sqrt(w @ (p - y) ** 2)),
                    pearson=correlation,
                    spearman=rank,
                )
            )
            for timepoint in sorted({r["time"] for r in values}):
                selected = [r for r in values if r["time"] == timepoint]
                a = np.array([r["raw_prediction"] for r in selected])
                b = np.array([r["target"] for r in selected])
                per_time.append(
                    dict(
                        variant=name,
                        condition=condition,
                        split=values[0]["split"],
                        time=timepoint,
                        n=len(selected),
                        n_peptide_sequences=len(
                            {r["peptide_sequence"] for r in selected}
                        ),
                        pearson=None
                        if np.ptp(a) < 1e-12
                        else float(pearsonr(a, b).statistic),
                        spearman=None
                        if np.ptp(a) < 1e-12
                        else float(spearmanr(a, b).statistic),
                    )
                )
            for field in ("log10_amount", "log10_response"):
                data = np.array([r[field] for r in values if r[field] is not None])
                if len(data):
                    ranges.append(
                        dict(
                            variant=name,
                            condition=condition,
                            split=values[0]["split"],
                            field=field,
                            quantiles=dict(
                                zip(
                                    ["min", "p01", "median", "p99", "max"],
                                    np.quantile(data, [0, 0.01, 0.5, 0.99, 1]).tolist(),
                                )
                            ),
                            centered_range=float(np.ptp(data)),
                        )
                    )
    for filename, content in (
        ("comparison.csv", rows),
        ("within-condition.csv", within),
        ("within-condition-time-ranks.csv", per_time),
    ):
        with (FAMILY / "results" / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    dump("results/component-ranges.json", ranges)
    dump("results/all-metrics.json", summary)
    conditional = read("results/conditional-gradient-probes.json")
    dump(
        "results/conditional-probe-summary.json",
        [
            {
                k: r[k]
                for k in (
                    "state",
                    "kind",
                    "count",
                    "mean_relative_error",
                    "rms_relative_error",
                    "zero_span_time_fraction",
                    "seconds",
                    "gate_pass",
                )
            }
            for r in conditional
        ],
    )
    labels = [
        "Constant",
        "Response only",
        "Time only",
        "Old ratio fit",
        "Old + response",
        "Kinetics",
        "Kinetics + response",
        "Conditional hybrid",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(order))
    axes[0].bar(
        x - 0.18,
        [summary[n]["train"]["shape_rmse"] for n in order],
        width=0.36,
        label="Training",
    )
    axes[0].bar(
        x + 0.18,
        [summary[n]["val"]["shape_rmse"] for n in order],
        width=0.36,
        label="Development",
    )
    axes[0].set_ylabel("Profiled log10 MS-signal RMSE")
    axes[0].set_xticks(x, labels, rotation=65, ha="right")
    axes[0].legend()
    for name, label in (
        ("constant_response", "Exact kinetics"),
        ("sequence_response", "Exact + response"),
        ("conditional_sequence_response", "Conditional hybrid"),
    ):
        fit = read(f"results/{name}-fit.json")
        axes[1].plot(
            [r["step"] for r in fit["history"]],
            [r["preupdate_loss"] for r in fit["history"]],
            label=label,
        )
    axes[1].set_xlabel("Optimizer update")
    axes[1].set_ylabel("Training loss incl. penalty")
    axes[1].legend(fontsize=8)
    for state, label in (
        ("uniform_0p01", "Uniform-rate state"),
        ("saved_ratio_initialization", "Ratio-fit state"),
    ):
        selected = [
            r
            for r in conditional
            if r["state"] == state and r["kind"] == "gradient_mixture_90_10"
        ]
        axes[2].plot(
            [r["count"] for r in selected],
            [r["rms_relative_error"] for r in selected],
            marker="o",
            label=label,
        )
    axes[2].axhline(0.5, color="gray", linestyle="--", label="Fixed RMS gate at K512")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Partitions per condition/time")
    axes[2].set_ylabel("Conditional gradient RMS relative error")
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Shared MS observation: modest development gain; conditional sampling qualifies"
    )
    fig.tight_layout()
    fig.savefig(FAMILY / "results/comparison.png", dpi=170)
    plt.close(fig)
    diagnostic = dict(
        total_predictions=sum(r["n_observations"] for r in rows),
        development_mse_reduction_vs_constant=1
        - summary["sequence_response"]["val"]["shape_mse"]
        / summary["constant_signal"]["val"]["shape_mse"],
        development_mse_reduction_vs_kinetics=1
        - summary["sequence_response"]["val"]["shape_mse"]
        / summary["constant_response"]["val"]["shape_mse"],
        original_ratio_rmse_change=summary["sequence_response"]["val"]["ratio_rmse"]
        - summary["fixed_ratio"]["val"]["ratio_rmse"],
        note="Within-condition correlations pool peptides and times; they are not per-time peptide-only rankings. All development offsets use labels to define a scale-invariant metric; no calibrated held-out intensity prediction.",
    )
    dump("results/report-summary.json", diagnostic)
    print(json.dumps(diagnostic), flush=True)


if __name__ == "__main__":
    main()
