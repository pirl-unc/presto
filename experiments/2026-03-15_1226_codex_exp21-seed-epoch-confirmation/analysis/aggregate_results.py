#!/usr/bin/env python
"""Summarize the 36-run EXP-21 seed/epoch confirmation sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _display_model(row: pd.Series) -> str:
    backbone = str(row["encoder_backbone"])
    cond_id = int(row["cond_id"])
    if backbone == "historical_ablation" and cond_id == 2:
        return "historical c02"
    if backbone == "groove" and cond_id == 1:
        return "groove c01"
    if backbone == "groove" and cond_id == 2:
        return "groove c02"
    return f"{backbone} c{cond_id:02d}"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plot_seed_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    order = ["historical c02", "groove c02", "groove c01"]
    groups = [df.loc[df["model_name"] == model, "test_spearman"].tolist() for model in order if model in set(df["model_name"])]
    labels = [model for model in order if model in set(df["model_name"])]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel("Test Spearman")
    ax.set_title("Seed Distribution by Model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_epoch_lines(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    summary = (
        df.groupby(["model_name", "epochs"], as_index=False)
        .agg(
            mean_test_spearman=("test_spearman", "mean"),
            mean_test_auroc=("test_auroc", "mean"),
            mean_test_rmse_log10=("test_rmse_log10", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("mean_test_spearman", "Mean Test Spearman"),
        ("mean_test_auroc", "Mean Test AUROC"),
        ("mean_test_rmse_log10", "Mean Test RMSE log10"),
    ]
    colors = {
        "historical c02": "#5c5c5c",
        "groove c02": "#3a6ea5",
        "groove c01": "#b05028",
    }
    for ax, (metric, title) in zip(axes, metrics):
        for model_name, group in summary.groupby("model_name", sort=False):
            group = group.sort_values("epochs")
            ax.plot(
                group["epochs"],
                group[metric],
                marker="o",
                linewidth=2,
                color=colors.get(model_name),
                label=model_name,
            )
        ax.set_xlabel("Epochs")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the EXP-21 seed/epoch confirmation sweep.")
    parser.add_argument(
        "--experiment-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Owning experiment directory. Defaults to the parent of this analysis script.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    results_dir = experiment_dir / "results"
    summary_path = results_dir / "condition_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}; run aggregate_summary_runs.py first.")

    df = pd.read_csv(summary_path).copy()
    df["model_name"] = df.apply(_display_model, axis=1)
    df["epochs"] = df["epochs"].astype(int)
    df["seed"] = df["seed"].astype(int)

    by_model_epoch = (
        df.groupby(["model_name", "epochs"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            mean_test_spearman=("test_spearman", "mean"),
            std_test_spearman=("test_spearman", "std"),
            mean_test_auroc=("test_auroc", "mean"),
            mean_test_auprc=("test_auprc", "mean"),
            mean_test_rmse_log10=("test_rmse_log10", "mean"),
        )
        .sort_values(["epochs", "mean_test_spearman"], ascending=[True, False])
    )
    by_model_seed = (
        df.groupby(["model_name", "seed"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            mean_test_spearman=("test_spearman", "mean"),
            mean_test_auroc=("test_auroc", "mean"),
            mean_test_rmse_log10=("test_rmse_log10", "mean"),
        )
        .sort_values(["seed", "mean_test_spearman"], ascending=[True, False])
    )
    by_model = (
        df.groupby("model_name", as_index=False)
        .agg(
            runs=("run_id", "count"),
            mean_test_spearman=("test_spearman", "mean"),
            max_test_spearman=("test_spearman", "max"),
            mean_test_auroc=("test_auroc", "mean"),
            mean_test_auprc=("test_auprc", "mean"),
            mean_test_rmse_log10=("test_rmse_log10", "mean"),
        )
        .sort_values("mean_test_spearman", ascending=False)
    )

    by_model.to_csv(results_dir / "model_summary.csv", index=False)
    by_model_epoch.to_csv(results_dir / "model_epoch_summary.csv", index=False)
    by_model_seed.to_csv(results_dir / "model_seed_summary.csv", index=False)
    write_json(
        results_dir / "model_summary.json",
        {
            "winner_by_mean_test_spearman": by_model.iloc[0].to_dict(),
            "model_summary": by_model.to_dict(orient="records"),
            "model_epoch_summary": by_model_epoch.to_dict(orient="records"),
        },
    )

    plot_seed_boxplot(df, results_dir / "seed_spearman_boxplot.png")
    plot_epoch_lines(df, results_dir / "epoch_budget_comparison.png")


if __name__ == "__main__":
    main()
