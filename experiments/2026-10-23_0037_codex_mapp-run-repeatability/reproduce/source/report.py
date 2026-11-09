"""Summaries and portable artifacts for completed run/linear-model comparisons."""

import csv
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
RESULTS = FAMILY / "results"


def read(path):
    return json.loads(Path(path).read_text())


def archive(source, name):
    path = RESULTS / (name + ".tar.gz")
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
        tarfile.open(fileobj=compressed, mode="w") as tar,
    ):
        for p in sorted(source.rglob("*")):
            if p.is_file():
                info = tar.gettarinfo(p, arcname=str(p.relative_to(source)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with p.open("rb") as handle:
                    tar.addfile(info, handle)
    return {
        "path": str(path.relative_to(FAMILY)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "original_directory": str(source),
    }


def main():
    assert (
        read(RESULTS / "independent-analysis.json")["passed"]
        and read(RESULTS / "independent-linear-verification.json")["passed"]
    )
    a = read(RESULTS / "repeatability-summary.json")
    b = read(RESULTS / "linear-summary.json")
    baseline = read(
        ROOT
        / "experiments/2026-10-22_2330_codex_mapp-biological-comparison/results/uniform-metrics.json"
    )
    lines = [
        "| Scope | Parents | Composition RMSE: other runs / uniform | Composition Pearson | Change RMSE: other runs / zero | Change Pearson |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope in ("all", "train", "validation", "test"):
        c = a["by_task"][scope + ":same_condition_composition"]
        d = a["by_task"][scope + ":condition_change"]
        lines.append(
            f"| {scope} | {a['by_scope'][scope]['n_parents']} | {c['other_run_mean']['rmse']:.4f} / {c['zero_baseline']['rmse']:.4f} | {c['other_run_mean']['pearson']:.4f} | {d['other_run_mean']['rmse']:.4f} / {d['zero_baseline']['rmse']:.4f} | {d['other_run_mean']['pearson']:.4f} |"
        )
    (RESULTS / "repeatability-table.md").write_text("\n".join(lines) + "\n")
    lines = [
        "| Model | Parameters | L2 / selected update | Validation KL | Development-test KL | Test log10 RMSE | Test Spearman | Change RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = []
    for r in b["selected"]:
        m = r["metrics"]["test"]
        label = f"{r['architecture']} / {r['method']}" + (
            f" / seed{r['seed']}" if r["method"] != "exact" else ""
        )
        lines.append(
            f"| {label} | {r['n_parameters']} | {r['l2']:g} / {r['selected_step']} | {r['validation_reverse_kl']:.4f} | {m['reverse_kl']:.4f} | {m['centered_log10_rmse']:.4f} | {m['centered_log10_spearman']:.4f} | {m['condition_change_log10_rmse']:.4f} |"
        )
        for split, metrics in r["metrics"].items():
            csv_rows.append(
                {
                    "name": r["name"],
                    "split": split,
                    "l2": r["l2"],
                    "selected_step": r["selected_step"],
                    **metrics,
                }
            )
    u = baseline["test"]
    lines.append(
        f"| Uniform | 0 | — | {baseline['validation']['reverse_kl']:.4f} | {u['reverse_kl']:.4f} | {u['centered_log10_rmse']:.4f} | undefined | {u['condition_change_log10_rmse']:.4f} |"
    )
    (RESULTS / "model-table.md").write_text("\n".join(lines) + "\n")
    with (RESULTS / "model-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    with gzip.open(RESULTS / "run-predictions.jsonl.gz", "rt") as handle:
        predictions = [json.loads(line) for line in handle]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, task, title in zip(
        axes,
        ("same_condition_composition", "condition_change"),
        ("Baseline composition repeats", "Treatment differences repeat weakly"),
    ):
        selected = [r for r in predictions if r["task"] == task]
        p = np.concatenate([r["prediction"] for r in selected])
        t = np.concatenate([r["target"] for r in selected])
        ax.scatter(p, t, s=5, alpha=0.14, color="#286b8c", rasterized=True)
        lo = min(p.min(), t.min())
        hi = max(p.max(), t.max())
        ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1)
        ax.set(
            xlabel="Prediction from other runs (centered log10)",
            ylabel="Held-out run value / difference",
            title=title,
        )
        ax.text(
            0.04,
            0.95,
            f"Pearson r = {a['by_task']['all:' + task]['other_run_mean']['pearson']:.3f}",
            transform=ax.transAxes,
            va="top",
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "repeatability.png", dpi=180)
    plt.close(fig)
    receipts = [
        archive(ROOT / "artifacts" / FAMILY.name / "linear-fits", "linear-artifacts"),
        archive(
            ROOT
            / "data/availability_recipes/2026-10-23-mapp-complete-run-composition-v1",
            "complete-run-recipe",
        ),
    ]
    (RESULTS / "artifact-receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    for name in ("verify_linear.py", "report.py"):
        shutil.copy2(FAMILY / "code" / name, FAMILY / "reproduce/source" / name)
    for p in (FAMILY / "launch_logs").glob("*.log"):
        p.with_suffix(".log.gz").write_bytes(gzip.compress(p.read_bytes(), mtime=0))
    print("\n".join(lines))
    print("Archive sizes:", [r["size_bytes"] for r in receipts])


if __name__ == "__main__":
    main()
