"""Summarize already completed, independently replayed fits; never train."""

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
ARTIFACTS = ROOT / "artifacts" / FAMILY.name


def read(path):
    return json.loads(path.read_text())


def main():
    assert read(RESULTS / "independent-results-verification.json")["passed"]
    summary = read(RESULTS / "training-summary.json")
    uniform = read(RESULTS / "uniform-metrics.json")
    variants = ("exact-context", "reinforce64-context", "exact-no-context")
    aggregates, records = {}, []
    for variant in variants:
        runs = [r for r in summary["runs"] if r["variant"] == variant]
        aggregates[variant] = {}
        for split in runs[0]["metrics"]:
            aggregates[variant][split] = {}
            for metric in runs[0]["metrics"][split]:
                values = [r["metrics"][split][metric] for r in runs]
                if any(v is None for v in values):
                    continue
                aggregates[variant][split][metric] = {
                    "mean": float(np.mean(values)),
                    "seed_sd": float(np.std(values)),
                    "min": min(values),
                    "max": max(values),
                }
        aggregates[variant]["mean_seconds"] = float(
            np.mean([r["seconds"] for r in runs])
        )
        for r in runs:
            for split, values in r["metrics"].items():
                records.append(
                    {
                        "variant": variant,
                        "seed": r["seed"],
                        "split": split,
                        "selected_epoch": r["selected_epoch"],
                        "seconds": r["seconds"],
                        **values,
                    }
                )
    for split, values in uniform.items():
        row = {
            "variant": "uniform",
            "seed": None,
            "split": split,
            "selected_epoch": None,
            "seconds": None,
            **values,
        }
        # Numerical centering of identical values leaves ~1e-16 residuals for some
        # set sizes. Preserve raw outputs, but do not give them biological ranks.
        row["centered_log10_pearson"] = row["centered_log10_spearman"] = None
        row["prediction_centered_log10_sd"] = 0.0
        records.append(row)
    with (RESULTS / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    (RESULTS / "aggregate-metrics.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "| Method | Validation KL | Test KL | Test TV | Test log10 RMSE | Test Spearman | Condition-change RMSE | Seconds/fit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in variants:
        a = aggregates[name]

        def value(split, metric, group=a):
            x = group[split][metric]
            return f"{x['mean']:.4f} ± {x['seed_sd']:.4f}"

        lines.append(
            f"| {name} | {value('validation', 'reverse_kl')} | {value('test', 'reverse_kl')} | {value('test', 'total_variation')} | {value('test', 'centered_log10_rmse')} | {value('test', 'centered_log10_spearman')} | {value('test', 'condition_change_log10_rmse')} | {a['mean_seconds']:.2f} |"
        )
    u = uniform["test"]
    lines.append(
        f"| Uniform | {uniform['validation']['reverse_kl']:.4f} | {u['reverse_kl']:.4f} | {u['total_variation']:.4f} | {u['centered_log10_rmse']:.4f} | undefined | {u['condition_change_log10_rmse']:.4f} | unfitted |"
    )
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n")
    notes = {
        "uniform_correlations": "Undefined for a constant composition contrast. Original raw metrics preserve tiny rounding-derived values; report/CSV render them null and SD zero.",
        "spread": "Population SD across three random seeds, not a confidence interval or biological replicate uncertainty.",
        "aggregation": "KL/TV and squared composition error macro-average protein-condition distributions; RMSE is root macro MSE. Correlations/SD pool centered peptide values.",
        "holdouts": "15 validation and 14 test parent sequences; exact-source components do not establish homology or specimen independence.",
    }
    (RESULTS / "metric-notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = ("#2c6da4", "#c46f22", "#5c8651")
    labels = ("Exact + context", "REINFORCE64 + context", "Exact, no context")
    for i, (variant, color, label) in enumerate(zip(variants, colors, labels)):
        runs = [r for r in summary["runs"] if r["variant"] == variant]
        histories = [
            read(ARTIFACTS / f"{variant}-seed{r['seed']}" / "history.json")
            for r in runs
        ]
        y = np.array(
            [[h["validation"]["reverse_kl"] for h in history] for history in histories]
        )
        x = np.arange(1, 41)
        axes[0].plot(x, y.mean(0), color=color, label=label)
        axes[0].fill_between(x, y.min(0), y.max(0), alpha=0.12, color=color)
        vals = [r["metrics"]["test"]["centered_log10_rmse"] for r in runs]
        axes[1].scatter(i + np.array([-0.08, 0, 0.08]), vals, color=color, s=35)
        axes[1].plot(
            [i - 0.15, i + 0.15], [np.mean(vals)] * 2, color=color, linewidth=2
        )
    axes[0].axhline(
        uniform["validation"]["reverse_kl"],
        color="#444444",
        linestyle="--",
        label="Uniform",
    )
    axes[1].axhline(
        u["centered_log10_rmse"], color="#444444", linestyle="--", label="Uniform"
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Validation reverse KL",
        title="Later epochs worsen validation",
    )
    axes[1].set(
        xticks=range(3),
        xticklabels=["Exact\ncontext", "REINFORCE64\ncontext", "Exact\nno context"],
        ylabel="Test centered log10 RMSE",
        title="No fitted model improves on uniform",
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "comparison.png", dpi=180)
    plt.close(fig)
    archive = RESULTS / "training-artifacts.tar.gz"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped,
        tarfile.open(fileobj=zipped, mode="w") as tar,
    ):
        for path in sorted(ARTIFACTS.rglob("*")):
            if path.is_file():
                info = tar.gettarinfo(path, arcname=str(path.relative_to(ARTIFACTS)))
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
    (RESULTS / "training-artifacts-receipt.json").write_text(
        json.dumps(
            {
                "path": str(archive.relative_to(FAMILY)),
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "size_bytes": archive.stat().st_size,
                "original_path": str(ARTIFACTS),
            },
            indent=2,
        )
        + "\n"
    )
    for name in ("verify_results.py", "report.py"):
        shutil.copy2(FAMILY / "code" / name, FAMILY / "reproduce/source" / name)
    for path in (FAMILY / "launch_logs").glob("*.log"):
        path.with_suffix(".log.gz").write_bytes(
            gzip.compress(path.read_bytes(), mtime=0)
        )
    print("\n".join(lines))
    print(
        json.dumps(
            {
                "training_seconds": summary["seconds"],
                "peak_rss_bytes": summary["peak_process_rss_bytes"],
                "archive_bytes": archive.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
