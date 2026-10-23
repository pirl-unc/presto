"""Artifact-only numerical/runtime tables and a scientific comparison plot."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    results = FAMILY / "results"
    cfg = read(FAMILY / "reproduce/launch.json")
    prior = FAMILY.parents[1] / cfg["prior_family"]
    original = read(prior / "results/runtime-summary.json")["cases"]
    runtime = read(results / "runtime-summary.json")["cases"]
    verified = read(results / "independent-verification.json")
    assert verified["all_replay_checks_pass"]
    fields = [
        "case_id",
        "condition",
        "n_residues",
        "eta",
        "order",
        "substeps",
        "status",
        "seconds",
        "peak_rss_bytes",
        "mass_error",
        "negative_count",
        "minimum_count",
        "log_admission_gate",
    ]
    write_csv(
        results / "runtime-comparison.csv",
        [{k: r.get(k) for k in fields} for r in runtime],
    )
    grid = read(results / "exponential-grid.json")
    fields = [
        "size",
        "eta",
        "scale",
        "time",
        "order",
        "count_error",
        "gradient_error",
        "mass_error",
        "minimum_count",
        "negative_count",
        "maximum_relative_error_above_1e_12",
        "numerical_gate",
        "log_admission_gate",
        "seconds",
    ]
    write_csv(
        results / "numerical-comparison.csv", [{k: r[k] for k in fields} for r in grid]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(7)
    labels = ["Ffh12.5", "Ffh6.25", "H2A", "H2B", "IL37b", "LEDGF", "hTau"]
    for offset, label, color in [
        (-0.27, "Uniformization", "#666666"),
        (0, "CRAM16", "#a74b22"),
        (0.27, "CRAM48", "#1b6b93"),
    ]:
        if label == "Uniformization":
            rows = [r for r in original if r["eta"] == 1 and r["backend"] == "exact"]
        else:
            rows = [
                r
                for r in runtime
                if r["eta"] == 1 and r["order"] == int(label[4:]) and r["substeps"] == 1
            ]
        axes[0].bar(
            x + offset,
            [r["seconds"] for r in rows],
            width=0.25,
            label=label,
            color=color,
        )
        if label == "Uniformization":
            for i, r in enumerate(rows):
                if r["status"] != "passed":
                    axes[0].text(i + offset, 64, "stop", ha="center", fontsize=7)
    axes[0].set(
        yscale="log",
        ylabel="Forward + backward and artifact runtime (s)",
        title="Positive length effect on the same full proteins",
    )
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].legend(fontsize=8)
    for order, color in [(16, "#a74b22"), (48, "#1b6b93")]:
        y = sorted(
            r["count_error"] for r in grid if r["order"] == order and r["time"] > 0
        )
        axes[1].plot(np.arange(1, len(y) + 1), y, color=color, label=f"CRAM{order}")
    axes[1].axhline(
        1e-10,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="frozen accuracy gate",
    )
    axes[1].set(
        yscale="log",
        xlabel="Sorted nonzero-time numerical cases",
        ylabel="Maximum absolute expected-count error",
        title="Accuracy passes; strict positivity still fails",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(results / "solver-comparison.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "runtime_rows": len(runtime),
                "numerical_rows": len(grid),
                "scope": "Frozen artifact summaries only",
            }
        )
    )


if __name__ == "__main__":
    main()
