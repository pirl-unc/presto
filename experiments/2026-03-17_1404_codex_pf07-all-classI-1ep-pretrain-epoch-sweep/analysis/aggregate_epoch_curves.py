#!/usr/bin/env python
"""Aggregate per-epoch metric curves for the all-class-I PF07 epoch sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_METRICS = (
    ("val_spearman", "Validation Spearman"),
    ("val_auroc", "Validation AUROC"),
    ("val_auprc", "Validation AUPRC"),
    ("val_rmse_log10", "Validation RMSE log10"),
    ("val_loss", "Validation Loss"),
)

PROBE_OUTPUT_COLUMNS = (
    "kd_nM",
    "ic50_nM",
    "ec50_nM",
    "kd_proxy_ic50_nM",
    "kd_proxy_ec50_nM",
    "probe_kd_nM",
)


def _display_label(row: dict[str, object]) -> str:
    condition = str(row.get("condition_key", "")).strip()
    epochs = row.get("epoch_budget")
    if condition and epochs is not None:
        return f"{condition}_e{int(epochs):03d}"
    if condition:
        return condition
    return str(row.get("run_id", ""))


def _manifest_lookup(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    lookup: dict[str, dict[str, object]] = {}
    for row in payload:
        run_id = str(row.get("run_id", "")).strip()
        if run_id:
            lookup[run_id] = row
    return lookup


def _plot_metric(df: pd.DataFrame, metric: str, label: str, out_path: Path, order: list[str]) -> None:
    if metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in order:
        group = df.loc[df["display_label"] == name]
        if group.empty:
            continue
        ax.plot(group["epoch"], group[metric], linewidth=2.0, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.set_title(f"{label} Over Epochs")
    ax.grid(alpha=0.25)
    if metric in {"val_spearman", "val_auroc", "val_auprc"}:
        ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_grid(df: pd.DataFrame, out_path: Path, order: list[str]) -> None:
    metrics = [(metric, label) for metric, label in PLOT_METRICS if metric in df.columns]
    if not metrics:
        return
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes_list = list(axes.flat)
    for ax, (metric, label) in zip(axes_list, metrics):
        for name in order:
            group = df.loc[df["display_label"] == name]
            if group.empty:
                continue
            ax.plot(group["epoch"], group[metric], linewidth=1.8, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.25)
        if metric in {"val_spearman", "val_auroc", "val_auprc"}:
            ax.set_ylim(0.0, 1.0)
    for ax in axes_list[len(metrics):]:
        ax.axis("off")
    axes_list[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_probe_frames(runs_root: Path, manifest_lookup: dict[str, dict[str, object]], order: list[str]) -> list[pd.DataFrame]:
    frames = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        probe_path = run_dir / "probe_affinity_over_epochs.csv"
        if not probe_path.exists():
            continue
        df = pd.read_csv(probe_path)
        if df.empty:
            continue
        meta = manifest_lookup.get(run_dir.name, {})
        display_label = _display_label({"run_id": run_dir.name, **meta})
        if display_label not in order:
            order.append(display_label)
        df.insert(0, "run_id", run_dir.name)
        df.insert(1, "display_label", display_label)
        frames.append(df)
    return frames


def _long_probe_outputs(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        base = {
            "run_id": row["run_id"],
            "display_label": row["display_label"],
            "epoch": int(row["epoch"]),
            "peptide": row["peptide"],
            "allele": row["allele"],
        }
        for output_name in PROBE_OUTPUT_COLUMNS:
            if output_name not in row.index:
                continue
            value = row[output_name]
            if pd.isna(value):
                continue
            rows.append({**base, "output_name": output_name, "prediction_nM": float(value)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-epoch metric curves for an all-class-I PF07 sweep.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--runs-subdir", default="results/runs")
    parser.add_argument("--output-subdir", default="results")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    runs_root = experiment_dir / args.runs_subdir
    output_dir = experiment_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lookup = _manifest_lookup(experiment_dir / "manifest.json")
    frames = []
    order: list[str] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        epoch_path = run_dir / "epoch_metrics.csv"
        if not epoch_path.exists():
            continue
        df = pd.read_csv(epoch_path)
        if df.empty:
            continue
        meta = manifest_lookup.get(run_dir.name, {})
        display_label = _display_label({"run_id": run_dir.name, **meta})
        if display_label not in order:
            order.append(display_label)
        df.insert(0, "run_id", run_dir.name)
        df.insert(1, "display_label", display_label)
        frames.append(df)

    if not frames:
        raise SystemExit(f"No epoch_metrics.csv files found under {runs_root}")

    epoch_df = pd.concat(frames, ignore_index=True)
    epoch_df.to_csv(output_dir / "epoch_metrics_by_condition.csv", index=False)
    (output_dir / "epoch_metrics_by_condition.json").write_text(
        epoch_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    for metric, label in PLOT_METRICS:
        _plot_metric(epoch_df, metric, label, output_dir / f"{metric}_over_epochs.png", order)
    _plot_metric_grid(epoch_df, output_dir / "val_metric_curves_over_epochs.png", order)

    probe_frames = _load_probe_frames(runs_root, manifest_lookup, order)
    if not probe_frames:
        return

    probe_df = pd.concat(probe_frames, ignore_index=True)
    probe_df.to_csv(output_dir / "probe_affinity_by_condition.csv", index=False)
    (output_dir / "probe_affinity_by_condition.json").write_text(
        probe_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    probe_long = _long_probe_outputs(probe_df)
    if probe_long.empty:
        return

    probe_long.to_csv(output_dir / "probe_affinity_by_condition_long.csv", index=False)
    (output_dir / "probe_affinity_by_condition_long.json").write_text(
        probe_long.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    final_probe = (
        probe_long.sort_values(["display_label", "peptide", "allele", "output_name", "epoch"])
        .groupby(["display_label", "peptide", "allele", "output_name"], as_index=False)
        .tail(1)
        .sort_values(["display_label", "peptide", "allele", "output_name"])
        .reset_index(drop=True)
    )
    final_probe.to_csv(output_dir / "final_probe_predictions.csv", index=False)
    (output_dir / "final_probe_predictions.json").write_text(
        final_probe.to_json(orient="records", indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
