"""Descriptive tables and figures for the closed original-training diagnostic."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]


def main():
    results = FAMILY / "results"
    closure = json.loads((results / "independent-verification.json").read_text())
    assert closure["status"] == "passed"
    summary = json.loads((results / "final-summary.json").read_text())
    keys = [
        "variant",
        "phase",
        "data_loss",
        "exact_rmse",
        "exact_mae",
        "exact_r2",
        "exact_pearson_pooled",
        "exact_spearman_pooled",
        "exact_direction_accuracy",
        "prediction_sd_pooled",
        "exact_target_sd_pooled",
    ]
    with (results / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in keys} for r in summary["metrics"])
    variants = [r["variant"] for r in summary["metrics"] if r["phase"] == "full_train"]
    labels = [
        "No change",
        "Condition only",
        "Fixed precursor coefficient = 1",
        "Calibrated precursor",
        "Shared preferences; penalty .01",
        "Shared preferences; no penalty",
        "Condition-specific; penalty .01",
        "Condition-specific; no penalty",
    ]
    y = np.arange(len(variants))
    train = np.array(
        [
            next(
                r["exact_rmse"]
                for r in summary["metrics"]
                if r["variant"] == v and r["phase"] == "full_train"
            )
            for v in variants
        ]
    )
    cv = np.array(
        [
            next(
                r["exact_rmse"]
                for r in summary["metrics"]
                if r["variant"] == v and r["phase"] == "inner_cv"
            )
            for v in variants
        ]
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.6), layout="constrained")
    for yi, a, b in zip(y, train, cv):
        ax.plot([a, b], [yi, yi], color="#A6A9AD", lw=2, zorder=1)
    ax.scatter(train, y, color="#277DA8", label="Full training fit", s=45, zorder=2)
    ax.scatter(cv, y, color="#DB8B2D", label="Component-grouped inner CV", s=45, zorder=3)
    for yi, value in zip(y, cv):
        ax.text(value + 0.001, yi, f"{value:.4f}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.09, 0.153)
    ax.set_xlabel("Weighted RMSE of log10 response; lower is better")
    ax.set_title(
        "APC training diagnostic: added flexibility overfits\n492 original-training peptides; five component-grouped folds",
        loc="left",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(results / "comparison.png", dpi=170)
    fig.savefig(results / "comparison.pdf")
    plt.close(fig)
    records = []
    for p in sorted(results.glob("*/*/status.json")):
        r = json.loads(p.read_text())
        records.append({**r, "path": str(p.relative_to(FAMILY))})
    runtime = {
        "fits": records,
        "exact_fit_seconds_sum": sum(r["seconds"] for r in records),
        "peak_fit_rss_bytes": max(r["peak_rss_bytes"] for r in records),
        "scope": "sum of recorded per-fit loops/final outputs; excludes source setup, Jacobian diagnostics and independent closure",
    }
    (results / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(
        json.dumps(
            {
                "plots_written": True,
                "n_exact_fits": len(records),
                "fit_seconds": runtime["exact_fit_seconds_sum"],
            }
        )
    )


if __name__ == "__main__":
    main()
