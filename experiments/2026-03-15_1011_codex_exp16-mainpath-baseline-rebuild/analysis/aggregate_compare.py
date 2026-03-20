from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plot_backend_heatmap(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty or "test_spearman" not in summary_df:
        return
    heat_df = (
        summary_df.assign(
            cc_tag=summary_df["content_conditioned"].astype(str).map({"True": "cc1", "False": "cc0"}),
            cond_key=lambda df: "c" + df["cond_id"].astype(int).astype(str).str.zfill(2) + "_" + df["cc_tag"],
        )
        .pivot_table(index="encoder_backbone", columns="cond_key", values="test_spearman", aggfunc="first")
        .sort_index(axis=1)
    )
    if heat_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(12, 0.5 * heat_df.shape[1]), 3.5))
    image = ax.imshow(heat_df.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(heat_df.shape[1]))
    ax.set_xticklabels(heat_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(heat_df.shape[0]))
    ax.set_yticklabels(heat_df.index)
    ax.set_title("Test Spearman by Condition and Encoder Backbone")
    fig.colorbar(image, ax=ax, label="Test Spearman")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_backend_metric_bars(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    backend_df = (
        summary_df.groupby("encoder_backbone", as_index=False)
        .agg(
            mean_test_spearman=("test_spearman", "mean"),
            mean_test_auroc=("test_auroc", "mean"),
            mean_test_rmse_log10=("test_rmse_log10", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("mean_test_spearman", "Mean Test Spearman"),
        ("mean_test_auroc", "Mean Test AUROC"),
        ("mean_test_rmse_log10", "Mean Test RMSE log10"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(backend_df["encoder_backbone"], backend_df[metric], color=["#40698e", "#a0632b"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EXP-16 backbone comparison results.")
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    results_dir = experiment_dir / "results"
    summary_path = results_dir / "condition_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}; run aggregate_summary_runs.py first.")

    summary_df = pd.read_csv(summary_path)
    summary_df["content_conditioned"] = summary_df["content_conditioned"].astype(str)

    backend_summary = (
        summary_df.groupby("encoder_backbone", as_index=False)
        .agg(
            mean_test_spearman=("test_spearman", "mean"),
            max_test_spearman=("test_spearman", "max"),
            mean_test_auroc=("test_auroc", "mean"),
            mean_test_rmse_log10=("test_rmse_log10", "mean"),
            n_runs=("run_id", "count"),
        )
        .sort_values("mean_test_spearman", ascending=False)
    )
    backend_summary.to_csv(results_dir / "backend_summary.csv", index=False)

    condition_compare = (
        summary_df.sort_values(["cond_id", "content_conditioned", "encoder_backbone"])
        .loc[
            :,
            [
                "run_id",
                "label",
                "cond_id",
                "content_conditioned",
                "encoder_backbone",
                "test_spearman",
                "test_auroc",
                "test_auprc",
                "test_f1",
                "test_rmse_log10",
            ],
        ]
    )
    condition_compare.to_csv(results_dir / "backend_condition_comparison.csv", index=False)

    winner = summary_df.sort_values("test_spearman", ascending=False).iloc[0].to_dict()
    write_json(
        results_dir / "backend_summary.json",
        {
            "experiment_dir": str(experiment_dir),
            "winner": winner,
            "backend_summary": backend_summary.to_dict(orient="records"),
        },
    )

    plot_backend_heatmap(summary_df, results_dir / "backend_condition_heatmap.png")
    plot_backend_metric_bars(summary_df, results_dir / "backend_metric_bars.png")


if __name__ == "__main__":
    main()
