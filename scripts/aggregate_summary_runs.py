from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEST_METRICS = [
    "spearman",
    "pearson",
    "auroc",
    "auprc",
    "f1",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "rmse_log10",
    "loss",
]
METRIC_LABELS = {
    "spearman": "Test Spearman",
    "pearson": "Test Pearson",
    "auroc": "Test AUROC",
    "auprc": "Test AUPRC",
    "f1": "Test F1",
    "balanced_accuracy": "Test Balanced Acc",
    "accuracy": "Test Accuracy",
    "precision": "Test Precision",
    "recall": "Test Recall",
    "rmse_log10": "Test RMSE log10",
    "loss": "Test Loss",
}


def jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return json.dumps(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def run_dirs(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "summary.json").exists()
    )


def summarize_run(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((run_dir / "summary.json").read_text())
    config = summary.get("config", {})
    epoch_df = pd.DataFrame(summary.get("epoch_summaries", []))
    probe_path = run_dir / "probes.jsonl"
    probe_rows = []
    if probe_path.exists():
        with probe_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    probe_rows.append(json.loads(line))
    probe_df = pd.DataFrame(probe_rows)

    row = {
        "run_id": run_dir.name,
        "label": config.get("label", run_dir.name),
    }
    for key, value in config.items():
        row[key] = jsonable(value)

    if not epoch_df.empty:
        row["final_epoch"] = int(epoch_df["epoch"].max())
        if "epoch_time_s" in epoch_df:
            row["mean_epoch_s"] = float(epoch_df["epoch_time_s"].mean())
        if "val_loss" in epoch_df:
            best_loss_idx = epoch_df["val_loss"].astype(float).idxmin()
            row["best_val_loss"] = float(epoch_df.loc[best_loss_idx, "val_loss"])
            row["best_val_loss_epoch"] = int(epoch_df.loc[best_loss_idx, "epoch"])
            row["final_val_loss"] = float(epoch_df["val_loss"].iloc[-1])
        if "val_spearman" in epoch_df:
            best_spr_idx = epoch_df["val_spearman"].astype(float).idxmax()
            row["best_val_spearman"] = float(epoch_df.loc[best_spr_idx, "val_spearman"])
            row["best_val_spearman_epoch"] = int(epoch_df.loc[best_spr_idx, "epoch"])
            row["final_val_spearman"] = float(epoch_df["val_spearman"].iloc[-1])

    test_metrics = summary.get("test_metrics", {})
    for key in TEST_METRICS:
        if key in test_metrics:
            row[f"test_{key}"] = float(test_metrics[key])

    if probe_df.empty:
        final_probe_df = probe_df
    else:
        probe_df = probe_df.copy()
        probe_df["epoch"] = probe_df["epoch"].astype(int)
        final_epoch = int(probe_df["epoch"].max())
        final_probe_df = probe_df.loc[probe_df["epoch"] == final_epoch].copy()
        final_probe_df.insert(0, "run_id", run_dir.name)
        final_probe_df.insert(1, "label", row["label"])

    if not epoch_df.empty:
        epoch_df = epoch_df.copy()
        epoch_df.insert(0, "run_id", run_dir.name)
        epoch_df.insert(1, "label", row["label"])

    return row, epoch_df, final_probe_df


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plot_metric_ranking(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty or "test_spearman" not in summary_df:
        return
    plot_df = summary_df.sort_values("test_spearman", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(plot_df))))
    ax.barh(plot_df["display_label"], plot_df["test_spearman"], color="#35618f")
    ax.set_xlabel("Test Spearman")
    ax.set_ylabel("Condition")
    ax.set_title("Held-out Test Spearman by Condition")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_grid(summary_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [metric for metric in ["test_spearman", "test_auroc", "test_f1", "test_rmse_log10"] if metric in summary_df]
    if not metrics:
        return
    plot_df = summary_df.sort_values(metrics[0], ascending=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        ax.bar(plot_df["display_label"], plot_df[metric], color="#7c9dc4")
        ax.set_title(METRIC_LABELS[metric.removeprefix("test_")])
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(metrics) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_curves(epoch_df: pd.DataFrame, out_path: Path) -> None:
    if epoch_df.empty:
        return
    plot_metrics = [metric for metric in ["val_loss", "val_spearman"] if metric in epoch_df]
    if not plot_metrics:
        return
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(7 * len(plot_metrics), 5))
    if len(plot_metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, plot_metrics):
        for display_label, group in epoch_df.groupby("display_label", sort=False):
            ax.plot(group["epoch"], group[metric], label=display_label, linewidth=1.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_probe_heatmap(probe_df: pd.DataFrame, out_path: Path) -> None:
    if probe_df.empty:
        return
    probe_df = probe_df.copy()
    probe_df["probe_id"] = probe_df["allele"] + " | " + probe_df["peptide"]
    pivot = (
        probe_df.pivot_table(index="display_label", columns="probe_id", values="ic50_nM", aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return
    values = np.log10(pivot.astype(float).clip(lower=1e-6))
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * values.shape[1]), max(4, 0.5 * values.shape[0])))
    image = ax.imshow(values.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(values.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(values.index, fontsize=8)
    ax.set_title("Final Probe Predictions (log10 nM)")
    fig.colorbar(image, ax=ax, label="log10(IC50 nM)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate summary.json-based experiment runs.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory containing results/runs.")
    parser.add_argument("--runs-subdir", default="results/runs", help="Run directory relative to experiment dir.")
    parser.add_argument("--output-subdir", default="results", help="Output directory relative to experiment dir.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    runs_root = experiment_dir / args.runs_subdir
    output_dir = experiment_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    epoch_frames = []
    probe_frames = []
    for run_dir in run_dirs(runs_root):
        row, epoch_df, probe_df = summarize_run(run_dir)
        rows.append(row)
        if not epoch_df.empty:
            epoch_frames.append(epoch_df)
        if not probe_df.empty:
            probe_frames.append(probe_df)

    if not rows:
        raise SystemExit(f"No run directories with summary.json found under {runs_root}")

    summary_df = pd.DataFrame(rows)
    sort_metric = "test_spearman" if "test_spearman" in summary_df else "label"
    summary_df["display_label"] = summary_df["label"]
    duplicate_mask = summary_df["label"].duplicated(keep=False)
    summary_df.loc[duplicate_mask, "display_label"] = summary_df.loc[duplicate_mask].apply(
        lambda row: f"{row['label']} [{row['run_id']}]", axis=1
    )
    summary_df = summary_df.sort_values(sort_metric, ascending=False if sort_metric != "label" else True)
    epoch_df = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()
    probe_df = pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame()
    if not epoch_df.empty:
        epoch_df["display_label"] = epoch_df["run_id"].map(summary_df.set_index("run_id")["display_label"])
    if not probe_df.empty:
        probe_df["display_label"] = probe_df["run_id"].map(summary_df.set_index("run_id")["display_label"])

    summary_df.to_csv(output_dir / "condition_summary.csv", index=False)
    epoch_df.to_csv(output_dir / "epoch_summary.csv", index=False)
    probe_df.to_csv(output_dir / "final_probe_predictions.csv", index=False)
    write_json(output_dir / "condition_summary.json", summary_df.to_dict(orient="records"))

    best_row = summary_df.iloc[0].to_dict()
    summary_bundle = {
        "experiment_dir": str(experiment_dir),
        "n_conditions": int(len(summary_df)),
        "best_by_test_spearman": best_row,
    }
    write_json(output_dir / "summary_bundle.json", summary_bundle)

    plot_metric_ranking(summary_df, output_dir / "test_spearman_ranking.png")
    plot_metric_grid(summary_df, output_dir / "test_metric_grid.png")
    plot_training_curves(epoch_df, output_dir / "training_curves.png")
    plot_probe_heatmap(probe_df, output_dir / "final_probe_heatmap.png")


if __name__ == "__main__":
    main()
