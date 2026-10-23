"""Summarize frozen numerical/runtime outputs without fitting or target access."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = Path(__file__).resolve().parents[1]
RESULTS = FAMILY / "results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def table(name, rows):
    with (RESULTS / name).open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    verification = read("independent-verification.json")
    assert verification["all_checks_pass"]
    runtime = read("runtime-summary.json")
    assert runtime["complete"]
    rows = []
    for row in runtime["cases"]:
        rows.append(
            {
                key: row[key]
                for key in (
                    "case_id", "condition", "n_residues", "eta", "backend",
                    "status", "seconds", "peak_rss_bytes",
                )
            }
        )
    table("runtime-comparison.csv", rows)
    probes = verification["toy"]["probes"]
    table("path-gradient-comparison.csv", probes)
    table("runtime-precision.csv", verification["runtime"]["comparisons"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for backend, marker, color in [("exact", "o", "#1b6b93"), ("sampled", "s", "#a74b22")]:
        for eta, alpha in [(-1, 0.35), (0, 0.65), (1, 1.0)]:
            selected = [r for r in rows if r["backend"] == backend and r["eta"] == eta]
            axes[0].scatter(
                [r["n_residues"] for r in selected],
                [r["seconds"] for r in selected],
                color=color, marker=marker, alpha=alpha,
                label=f"{backend}, eta={eta:+d}",
            )
    axes[0].axhline(60, color="black", linestyle="--", linewidth=0.8)
    axes[0].text(140, 64, "Four exact cases stopped at 60 s", fontsize=8)
    axes[0].set(xlabel="Parent length (residues)", ylabel="Forward/backward or sampling runtime (s)", yscale="log")
    axes[0].legend(fontsize=7, ncol=2, loc="lower right")
    axes[0].set_title("Actual training inputs; no measurement targets")
    for eta, color in [(-1, "#1b6b93"), (0, "#666666"), (1, "#a74b22")]:
        for time in [0.5, 2, 8]:
            selected = [r for r in probes if r["eta"] == eta and r["time"] == time]
            axes[1].plot([r["count"] for r in selected], [r["rms_relative_error"] for r in selected], marker="o", color=color, alpha=0.7)
    axes[1].set(xlabel="Trajectories per group", ylabel="Relative RMS gradient error across 16 groups", xscale="log", yscale="log")
    axes[1].set_title("Five-bond fixture; all nine primary gates pass")
    fig.savefig(RESULTS / "numerical-runtime.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"runtime_rows": len(rows), "path_conditions": len(probes), "precision_rows": len(verification["runtime"]["comparisons"])}))


if __name__ == "__main__":
    main()
