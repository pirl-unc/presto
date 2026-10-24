"""Close the biological comparison without selecting another test winner."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    result = read(ROOT / "results/evaluation/metrics.json")["human_native"]
    states = read(ROOT / "results/fit/human_native/states.json")
    verify = read(ROOT / "results/independent-predictions.json")
    assert verify["status"] == "passed"
    with (ROOT / "results/summary.csv").open("w") as f:
        fields = [
            "model",
            "split",
            "mse",
            "rmse",
            "mae",
            "r2",
            "pearson",
            "spearman",
            "reverse_kl",
            "forward_kl",
            "total_variation",
            "prediction_sd",
            "target_sd",
            "n_examples",
            "n_components",
            "n_groups",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, splits in result.items():
            for split, values in splits.items():
                writer.writerow(
                    {
                        "model": name,
                        "split": split,
                        **{key: values[key] for key in fields[2:]},
                    }
                )
    names = [
        "zero",
        "length_1",
        "sequence_1",
        "observer_fixed",
        "observer_calibrated",
        "uniform_alone_mse",
        "uniform_observer_mse",
        "sequence_alone_mse",
        "sequence_observer_mse",
    ] + [
        f"sequence_observer_{kind}_{seed}"
        for kind in ("exact", "reinforce")
        for seed in (42, 43, 44)
    ]
    rows = []
    for name in names:
        values = result[name]
        fmt = lambda x: "undefined" if x is None else f"{x:.5f}"
        rows.append(
            "| "
            + name
            + " | "
            + " | ".join(
                fmt(x)
                for x in (
                    values["train"]["rmse"],
                    values["val"]["rmse"],
                    values["test"]["rmse"],
                    values["test"]["reverse_kl"],
                    values["test"]["r2"],
                    values["test"]["spearman"],
                )
            )
            + " |"
        )
    (ROOT / "results/comparison-table.md").write_text(
        "| Model | Train RMSE | Validation RMSE | Test RMSE | Test reverse KL | Test R² | Test Spearman |\n|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n"
    )
    report = {
        "states": {
            name: {
                k: r[k]
                for k in (
                    "n_allocated_parameters",
                    "n_changed_parameters",
                    "n_gradient_supported_parameters",
                    "runtime_seconds",
                    "n_candidate_scores",
                    "n_sampled_peptides",
                )
            }
            for name, r in states.items()
            if "theta" in r
        },
        "matched": {
            kind: {
                metric: {
                    "mean": float(
                        np.mean(
                            [
                                result[f"sequence_observer_{kind}_{seed}"]["test"][
                                    metric
                                ]
                                for seed in (42, 43, 44)
                            ]
                        )
                    ),
                    "population_seed_sd": float(
                        np.std(
                            [
                                result[f"sequence_observer_{kind}_{seed}"]["test"][
                                    metric
                                ]
                                for seed in (42, 43, 44)
                            ]
                        )
                    ),
                }
                for metric in ("rmse", "reverse_kl", "spearman")
            }
            for kind in ("exact", "reinforce")
        },
    }
    (ROOT / "results/matched-summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    selected = [
        "zero",
        "length_1",
        "sequence_1",
        "observer_fixed",
        "observer_calibrated",
        "sequence_alone_mse",
        "sequence_observer_mse",
    ]
    labels = [
        "Uniform",
        "Length",
        "Sequence",
        "Frozen MS",
        "Calibrated MS",
        "Processing",
        "Processing + MS",
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    ax = axes[0]
    positions = np.arange(len(selected))
    ax.bar(
        positions - 0.18,
        [result[n]["val"]["rmse"] for n in selected],
        width=0.36,
        color="#8199b8",
        label="Validation",
    )
    ax.bar(
        positions + 0.18,
        [result[n]["test"]["rmse"] for n in selected],
        width=0.36,
        color="#24496e",
        label="Test",
    )
    ax.set_xticks(positions, labels, rotation=30, ha="right")
    ax.set_ylabel("RMSE in centered log10 area")
    ax.legend(frameon=False)
    ax.set_title("Neither primary processing nor MS transfer passes")
    ax = axes[1]
    for i, kind in enumerate(("exact", "reinforce")):
        values = [
            result[f"sequence_observer_{kind}_{seed}"]["test"]["reverse_kl"]
            for seed in (42, 43, 44)
        ]
        ax.scatter(
            np.array([-0.08, 0, 0.08]) + i,
            values,
            s=55,
            color="#24496e" if kind == "exact" else "#b15c36",
            zorder=3,
        )
    ax.axhline(
        result["zero"]["test"]["reverse_kl"],
        color="#555555",
        ls="--",
        label="Uniform reference",
    )
    ax.set_xticks([0, 1], ["Exact gradient", "REINFORCE64"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Test reverse KL (lower is better)")
    ax.legend(frameon=False)
    ax.set_title("Matched objective, three fixed seeds")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Human native lymph: observed positive feature composition", fontsize=14
    )
    fig.text(
        0.5,
        0.01,
        "Test: 74 peptide features, 14 parents, 12 dependence components • Same-study holdout; no concentration claim",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(ROOT / "results/comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report["matched"], indent=2))


if __name__ == "__main__":
    main()
