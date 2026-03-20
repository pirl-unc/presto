from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
THRESHOLD_NM = 500.0
VERIFY_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "pearson",
    "spearman",
    "rmse_log10",
]
VERIFY_TOLERANCE = {
    "accuracy": 1e-5,
    "balanced_accuracy": 1e-5,
    "precision": 1e-5,
    "recall": 1e-5,
    "f1": 1e-5,
    "auroc": 1e-3,
    "auprc": 1e-3,
    "pearson": 1e-4,
    "spearman": 1e-4,
    "rmse_log10": 1e-4,
}
EXPECTED_FILES = [
    "summary.json",
    "metrics.jsonl",
    "probes.jsonl",
    "step_log.jsonl",
    "val_predictions.csv",
    "test_predictions.csv",
    "model.pt",
    "splits/train_records.csv",
    "splits/val_records.csv",
    "splits/test_records.csv",
]
HEAD_COLORS = {
    "mhcflurry": "#1f77b4",
    "log_mse": "#ff7f0e",
    "twohot": "#2ca02c",
    "hlgauss": "#d62728",
}
PROBE_TARGETS = [
    ("HLA-A*02:01", "SLLQHLIGL"),
    ("HLA-A*02:01", "FLRYLLFGI"),
    ("HLA-A*24:02", "NFLIKFLLI"),
]


def load_manifest() -> list[dict]:
    manifest_path = EXPERIMENT_DIR / "manifest.json"
    return json.loads(manifest_path.read_text())


def exact_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["is_exact"].astype(str).str.lower() == "true"
    out = df.loc[mask].copy()
    out["true_ic50_nM"] = out["true_ic50_nM"].astype(float)
    out["true_ic50_log10"] = out["true_ic50_log10"].astype(float)
    out["pred_ic50_nM"] = out["pred_ic50_nM"].astype(float)
    out["pred_ic50_log10"] = out["pred_ic50_log10"].astype(float)
    return out


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    exact = exact_rows(df)
    y_true = (exact["true_ic50_nM"] <= THRESHOLD_NM).astype(int)
    y_pred = (exact["pred_ic50_nM"] <= THRESHOLD_NM).astype(int)
    y_score = -exact["pred_ic50_log10"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "pearson": float(exact["true_ic50_log10"].corr(exact["pred_ic50_log10"])),
        "spearman": float(
            exact["true_ic50_log10"].rank(method="average").corr(
                exact["pred_ic50_log10"].rank(method="average")
            )
        ),
        "rmse_log10": float(
            np.sqrt(
                np.mean(
                    (exact["true_ic50_log10"] - exact["pred_ic50_log10"]) ** 2
                )
            )
        ),
        "n_exact": int(len(exact)),
    }


def verify_metrics(
    label: str, split: str, summary_metrics: dict, recomputed_metrics: dict
) -> list[dict]:
    rows: list[dict] = []
    for key in VERIFY_KEYS:
        summary_value = float(summary_metrics[key])
        recomputed_value = float(recomputed_metrics[key])
        abs_diff = abs(summary_value - recomputed_value)
        rows.append(
            {
                "label": label,
                "split": split,
                "metric": key,
                "summary_value": summary_value,
                "recomputed_value": recomputed_value,
                "abs_diff": abs_diff,
                "within_tolerance": abs_diff <= VERIFY_TOLERANCE[key],
            }
        )
        if abs_diff > VERIFY_TOLERANCE[key]:
            raise ValueError(
                f"{label} {split} {key} mismatch: "
                f"summary={summary_value} recomputed={recomputed_value}"
            )
    return rows


def per_allele_metrics(df: pd.DataFrame, split: str, label: str) -> list[dict]:
    exact = exact_rows(df)
    rows: list[dict] = []
    for allele, allele_df in exact.groupby("allele", sort=True):
        metrics = compute_metrics(allele_df)
        row = {"label": label, "split": split, "allele": allele}
        row.update(metrics)
        rows.append(row)
    return rows


def require_expected_files(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for rel_path in EXPECTED_FILES:
        full_path = run_dir / rel_path
        rows.append(
            {
                "run_id": run_dir.name,
                "relative_path": rel_path,
                "present": full_path.exists(),
            }
        )
        if not full_path.exists():
            raise FileNotFoundError(f"Missing expected file: {full_path}")
    return rows


def metric_or_nan(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return math.nan
    return float(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df.loc[:, columns].copy()
    headers = [str(col) for col in columns]
    rows = []
    for _, row in table.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    cells.append("nan")
                else:
                    cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append(cells)
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))
    ]
    out = []
    out.append(
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    )
    out.append(
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    )
    for row in rows:
        out.append(
            "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        )
    return "\n".join(out)


def plot_test_spearman(condition_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = condition_df.sort_values("test_spearman", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        plot_df["label"],
        plot_df["test_spearman"],
        color=[HEAD_COLORS.get(head, "#444444") for head in plot_df["head_type"]],
    )
    ax.set_xlabel("Test Spearman")
    ax.set_ylabel("Condition")
    ax.set_title("Held-out Test Spearman by Condition")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_val_spearman_curves(epoch_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, run_df in epoch_df.groupby("label", sort=False):
        head_type = run_df["head_type"].iloc[0]
        ax.plot(
            run_df["epoch"],
            run_df["val_spearman"],
            label=label,
            color=HEAD_COLORS.get(head_type, "#444444"),
            alpha=0.8,
            linewidth=1.5,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Spearman")
    ax.set_title("Validation Spearman Curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_probe_heatmap(probe_df: pd.DataFrame, out_path: Path) -> None:
    selected = []
    column_labels = []
    for allele, peptide in PROBE_TARGETS:
        column_labels.append(f"{allele.split('*')[1]}/{peptide[:3]}")
        mask = (probe_df["allele"] == allele) & (probe_df["peptide"] == peptide)
        subset = probe_df.loc[mask, ["label", "ic50_nM"]].copy()
        subset["value"] = np.log10(subset["ic50_nM"].clip(lower=1e-6))
        selected.append(subset.set_index("label")["value"])
    matrix = pd.concat(selected, axis=1)
    matrix.columns = column_labels
    matrix = matrix.sort_index()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(matrix.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Final Probe Predictions (log10 nM)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10 predicted nM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    manifest = load_manifest()
    condition_rows: list[dict] = []
    epoch_rows: list[dict] = []
    probe_rows: list[dict] = []
    per_allele_rows: list[dict] = []
    verification_rows: list[dict] = []
    file_rows: list[dict] = []

    for entry in manifest:
        run_id = entry["run_id"]
        run_dir = RUNS_DIR / run_id
        file_rows.extend(require_expected_files(run_dir))

        summary = json.loads((run_dir / "summary.json").read_text())
        val_predictions = pd.read_csv(run_dir / "val_predictions.csv")
        test_predictions = pd.read_csv(run_dir / "test_predictions.csv")
        recomputed_val = compute_metrics(val_predictions)
        recomputed_test = compute_metrics(test_predictions)
        verification_rows.extend(
            verify_metrics(entry["label"], "val", summary["val_metrics"], recomputed_val)
        )
        verification_rows.extend(
            verify_metrics(entry["label"], "test", summary["test_metrics"], recomputed_test)
        )
        per_allele_rows.extend(per_allele_metrics(val_predictions, "val", entry["label"]))
        per_allele_rows.extend(per_allele_metrics(test_predictions, "test", entry["label"]))

        probe_df = pd.read_json(run_dir / "probes.jsonl", lines=True)
        final_epoch = int(probe_df["epoch"].max())
        final_probe_df = probe_df.loc[probe_df["epoch"] == final_epoch].copy()
        final_probe_df.insert(0, "label", entry["label"])
        final_probe_df.insert(1, "run_id", run_id)
        probe_rows.extend(final_probe_df.to_dict(orient="records"))

        epoch_summaries = summary["epoch_summaries"]
        for epoch_summary in epoch_summaries:
            row = {
                "run_id": run_id,
                "label": entry["label"],
                "head_type": entry["head_type"],
                "assay_mode": entry["assay_mode"],
                "max_nM": entry["max_nM"],
                "n_bins": entry["n_bins"],
                "epoch": epoch_summary["epoch"],
                "epoch_time_s": epoch_summary["epoch_time_s"],
                "train_loss": epoch_summary["train_loss"],
                "val_loss": epoch_summary["val_loss"],
                "val_spearman": epoch_summary["val_spearman"],
                "val_pearson": epoch_summary["val_pearson"],
                "val_rmse_log10": epoch_summary["val_rmse_log10"],
                "val_accuracy": epoch_summary["val_accuracy"],
                "val_auroc": epoch_summary["val_auroc"],
                "val_auprc": epoch_summary["val_auprc"],
            }
            epoch_rows.append(row)

        condition_row = {
            "run_id": run_id,
            "app_id": entry["app_id"],
            "label": entry["label"],
            "head_type": entry["head_type"],
            "assay_mode": entry["assay_mode"],
            "max_nM": entry["max_nM"],
            "n_bins": entry["n_bins"],
            "sigma_mult": entry["sigma_mult"],
            "best_epoch": summary["best_epoch"],
            "best_val_loss": summary["best_val_loss"],
            "n_params": summary["config"]["n_params"],
            "train_rows": summary["config"]["train_rows"],
            "val_rows": summary["config"]["val_rows"],
            "test_rows": summary["config"]["test_rows"],
            "mean_epoch_time_s": float(
                np.mean([epoch["epoch_time_s"] for epoch in epoch_summaries])
            ),
            "total_epoch_time_s": float(
                np.sum([epoch["epoch_time_s"] for epoch in epoch_summaries])
            ),
        }
        for split_name, metrics in (
            ("val", summary["val_metrics"]),
            ("test", summary["test_metrics"]),
        ):
            for key, value in metrics.items():
                if key == "split":
                    continue
                condition_row[f"{split_name}_{key}"] = value
        for allele, peptide in PROBE_TARGETS:
            mask = (final_probe_df["allele"] == allele) & (final_probe_df["peptide"] == peptide)
            column_name = f"probe_{allele.replace('-', '').replace('*', '').replace(':', '')}_{peptide}_nM"
            condition_row[column_name] = float(final_probe_df.loc[mask, "ic50_nM"].iloc[0])
        condition_rows.append(condition_row)

    condition_df = pd.DataFrame(condition_rows).sort_values(
        ["test_spearman", "test_auroc"], ascending=[False, False]
    )
    epoch_df = pd.DataFrame(epoch_rows).sort_values(["label", "epoch"])
    probe_df = pd.DataFrame(probe_rows).sort_values(["label", "epoch", "allele", "peptide"])
    per_allele_df = pd.DataFrame(per_allele_rows).sort_values(["split", "label", "allele"])
    verification_df = pd.DataFrame(verification_rows).sort_values(["label", "split", "metric"])
    file_df = pd.DataFrame(file_rows).sort_values(["run_id", "relative_path"])

    family_summary_df = (
        condition_df.groupby("head_type", sort=False)[
            ["test_spearman", "test_rmse_log10", "test_auroc", "test_auprc", "mean_epoch_time_s"]
        ]
        .mean()
        .reset_index()
        .sort_values("test_spearman", ascending=False)
    )
    cap_summary_df = (
        condition_df.groupby(["head_type", "max_nM"], sort=False)[
            ["test_spearman", "test_rmse_log10", "test_auroc", "test_auprc"]
        ]
        .mean()
        .reset_index()
        .sort_values(["head_type", "max_nM"])
    )
    best_by_family_df = (
        condition_df.sort_values(["head_type", "test_spearman", "test_auroc"], ascending=[True, False, False])
        .groupby("head_type", sort=False)
        .head(1)
        .reset_index(drop=True)
    )

    condition_df.to_csv(RESULTS_DIR / "condition_summary.csv", index=False)
    condition_df.to_json(RESULTS_DIR / "condition_summary.json", orient="records", indent=2)
    epoch_df.to_csv(RESULTS_DIR / "epoch_summary.csv", index=False)
    probe_df.to_csv(RESULTS_DIR / "final_probe_predictions.csv", index=False)
    per_allele_df.to_csv(RESULTS_DIR / "per_allele_metrics.csv", index=False)
    verification_df.to_csv(RESULTS_DIR / "metric_verification.csv", index=False)
    file_df.to_csv(RESULTS_DIR / "artifact_inventory.csv", index=False)
    family_summary_df.to_csv(RESULTS_DIR / "family_summary.csv", index=False)
    cap_summary_df.to_csv(RESULTS_DIR / "cap_summary.csv", index=False)
    best_by_family_df.to_csv(RESULTS_DIR / "best_by_family.csv", index=False)

    topline_md = markdown_table(
        condition_df,
        [
            "label",
            "head_type",
            "max_nM",
            "n_bins",
            "test_spearman",
            "test_rmse_log10",
            "test_auroc",
            "test_auprc",
            "mean_epoch_time_s",
        ],
    )
    family_md = markdown_table(
        family_summary_df,
        ["head_type", "test_spearman", "test_rmse_log10", "test_auroc", "test_auprc", "mean_epoch_time_s"],
    )
    (RESULTS_DIR / "topline_test_ranking.md").write_text(topline_md + "\n")
    (RESULTS_DIR / "family_summary.md").write_text(family_md + "\n")

    plot_test_spearman(condition_df, RESULTS_DIR / "test_spearman_ranking.png")
    plot_val_spearman_curves(epoch_df, RESULTS_DIR / "val_spearman_curves.png")
    plot_probe_heatmap(probe_df, RESULTS_DIR / "final_probe_heatmap.png")

    summary_bundle = {
        "winner_by_test_spearman": condition_df.iloc[0]["label"],
        "winner_test_spearman": float(condition_df.iloc[0]["test_spearman"]),
        "winner_test_auroc": float(condition_df.iloc[0]["test_auroc"]),
        "winner_test_auprc": float(condition_df.iloc[0]["test_auprc"]),
        "winner_test_rmse_log10": float(condition_df.iloc[0]["test_rmse_log10"]),
        "best_by_family": best_by_family_df.to_dict(orient="records"),
        "family_summary": family_summary_df.to_dict(orient="records"),
        "cap_summary": cap_summary_df.to_dict(orient="records"),
        "max_metric_verification_abs_diff": float(verification_df["abs_diff"].max()),
        "all_expected_files_present": bool(file_df["present"].all()),
    }
    (RESULTS_DIR / "summary_bundle.json").write_text(json.dumps(summary_bundle, indent=2))


if __name__ == "__main__":
    main()
