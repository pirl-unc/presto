"""Deterministic figures for versioned Presto performance reports.

The functions in this module consume only validated manifest records. They do
not discover datasets, checkpoints, tools, or Git state.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "svg.hashsalt": "presto-performance-report-v1",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(
        png,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "Presto performance report"},
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "Presto performance report", "Date": None},
    )
    return [png, svg]


def render_comparator_overview(metrics: list[dict], output_dir: Path) -> list[Path]:
    """Render primary matched-cohort method estimates and their intervals."""
    primary = sorted(
        (row for row in metrics if row["primary"]),
        key=lambda row: (
            row["task"],
            row["output_id"],
            row["split"],
            row["metric"],
            row["method"],
        ),
    )
    if not primary:
        raise ValueError("at least one primary metric is required for the overview")

    plt = _pyplot()
    height = max(3.0, 0.38 * len(primary) + 1.2)
    fig, ax = plt.subplots(figsize=(9.5, height))
    colors = defaultdict(lambda: f"C{len(colors) % 10}")
    labels = []
    values = []
    lows = []
    highs = []
    methods = []
    for row in primary:
        labels.append(f"{row['task']} · {row['output_id']} · {row['split']} · {row['metric']}")
        values.append(float(row["value"]))
        lows.append(float(row["value"] if row.get("ci_low") is None else row["ci_low"]))
        highs.append(float(row["value"] if row.get("ci_high") is None else row["ci_high"]))
        methods.append(row["method"])
    y = list(reversed(range(len(primary))))
    for index, ypos in enumerate(y):
        value = values[index]
        ax.errorbar(
            value,
            ypos,
            xerr=[[value - lows[index]], [highs[index] - value]],
            fmt="o",
            color=colors[methods[index]],
            capsize=3,
            markersize=5,
            label=methods[index],
        )
    method_labels = [
        f"{row['method']} [{row.get('method_output_id', row['output_id'])}]" for row in primary
    ]
    ax.set_yticks(y, [f"{label}\n{method}" for label, method in zip(labels, method_labels)])
    ax.set_xlabel("Metric value (see labels; scales are not pooled)")
    ax.set_title("Primary matched-cohort comparator overview")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7)
    handles, legend_labels = ax.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", frameon=False)
    fig.subplots_adjust(left=0.38, right=0.98, top=0.93, bottom=0.12)
    paths = _save(fig, output_dir, "comparator_overview")
    plt.close(fig)
    return paths


def render_data_composition(datasets: list[dict], output_dir: Path) -> list[Path]:
    """Render frozen dataset row counts by role without collapsing source identity."""
    rows = sorted(datasets, key=lambda row: (row["role"], row["id"]))
    if not rows:
        raise ValueError("data composition requires at least one dataset")
    if any(not math.isfinite(float(row["rows"])) or int(row["rows"]) < 0 for row in rows):
        raise ValueError("dataset rows must be nonnegative finite integers")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.34 * len(rows) + 1.2)))
    roles = sorted({row["role"] for row in rows})
    colors = {role: f"C{index % 10}" for index, role in enumerate(roles)}
    y = list(reversed(range(len(rows))))
    for row, ypos in zip(rows, y):
        ax.barh(ypos, int(row["rows"]), color=colors[row["role"]], height=0.68)
        ax.text(int(row["rows"]), ypos, f" {int(row['rows']):,}", va="center", fontsize=8)
    ax.set_yticks(y, [f"{row['id']}\n{row['role']}" for row in rows])
    ax.set_xlabel("Rows")
    ax.set_title("Frozen data composition")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7)
    handles = [plt.Line2D([0], [0], color=colors[role], linewidth=6, label=role) for role in roles]
    ax.legend(handles=handles, loc="best", frameon=False)
    fig.subplots_adjust(left=0.32, right=0.96, top=0.91, bottom=0.13)
    paths = _save(fig, output_dir, "data_composition")
    plt.close(fig)
    return paths


def _finite_records(records: list[dict], required: tuple[str, ...]) -> list[dict]:
    """Return validated numeric figure rows without silently dropping data."""
    if not records:
        raise ValueError("figure input is empty")
    result = []
    for index, row in enumerate(records):
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"figure row {index} lacks {missing}")
        for name in required:
            if name in {"method", "mhc_class", "study", "task", "metric"}:
                if not isinstance(row[name], str) or not row[name]:
                    raise ValueError(f"figure row {index} has invalid {name}")
            elif not math.isfinite(float(row[name])):
                raise ValueError(f"figure row {index} has non-finite {name}")
        result.append(row)
    return result


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        raise ValueError("ROC requires both classes")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    true_positive = np.cumsum(sorted_labels)[ends]
    false_positive = (ends + 1) - true_positive
    return np.r_[0.0, false_positive / negatives], np.r_[0.0, true_positive / positives]


def _pr_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if not positives or positives == len(labels):
        raise ValueError("precision-recall requires both classes")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    hits = np.cumsum(sorted_labels)[ends]
    recall = hits / positives
    precision = hits / (ends + 1)
    return np.r_[0.0, recall], np.r_[1.0, precision]


def render_affinity_scatter(records: list[dict], output_dir: Path) -> list[Path]:
    """Plot exact affinity predictions on matched rows, separated by MHC class."""
    rows = _finite_records(
        records,
        ("method", "mhc_class", "target_log10_nM", "prediction_log10_nM"),
    )
    classes = sorted({row["mhc_class"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    if not classes or not methods:
        raise ValueError("affinity scatter requires classes and methods")
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(classes), figsize=(5.0 * len(classes), 4.4), squeeze=False)
    colors = {method: f"C{index % 10}" for index, method in enumerate(methods)}
    for axis, mhc_class in zip(axes[0], classes, strict=True):
        class_rows = [row for row in rows if row["mhc_class"] == mhc_class]
        bounds = [
            float(row[name])
            for row in class_rows
            for name in ("target_log10_nM", "prediction_log10_nM")
        ]
        low, high = min(bounds), max(bounds)
        padding = max(0.1, 0.03 * (high - low))
        axis.plot(
            [low - padding, high + padding],
            [low - padding, high + padding],
            "--",
            color="#777777",
        )
        for method in methods:
            selected = [row for row in class_rows if row["method"] == method]
            if not selected:
                continue
            truth = np.asarray([row["target_log10_nM"] for row in selected], dtype=float)
            prediction = np.asarray([row["prediction_log10_nM"] for row in selected], dtype=float)
            rho = _correlation(_ranks(truth), _ranks(prediction))
            label = f"{method} (ρ={rho:.3f})" if rho is not None else f"{method} (ρ=NA)"
            axis.scatter(
                truth,
                prediction,
                s=12,
                alpha=0.45,
                color=colors[method],
                edgecolors="none",
                label=label,
            )
        axis.set_title(f"MHC-{mhc_class} exact affinity")
        axis.set_xlabel("Observed log10(nM)")
        axis.set_ylabel("Predicted log10(nM)")
        axis.grid(color="#e1e1e1", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Matched exact-affinity predictions")
    fig.tight_layout()
    paths = _save(fig, output_dir, "affinity_scatter")
    plt.close(fig)
    return paths


def render_affinity_calibration(records: list[dict], output_dir: Path) -> list[Path]:
    """Render decile calibration and residual spread for exact affinity rows."""
    rows = _finite_records(
        records,
        ("method", "mhc_class", "target_log10_nM", "prediction_log10_nM"),
    )
    classes = sorted({row["mhc_class"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    plt = _pyplot()
    fig, axes = plt.subplots(2, len(classes), figsize=(5.0 * len(classes), 7.6), squeeze=False)
    colors = {method: f"C{index % 10}" for index, method in enumerate(methods)}
    for column, mhc_class in enumerate(classes):
        class_rows = [row for row in rows if row["mhc_class"] == mhc_class]
        all_values = [
            float(row[name])
            for row in class_rows
            for name in ("target_log10_nM", "prediction_log10_nM")
        ]
        low, high = min(all_values), max(all_values)
        axes[0, column].plot([low, high], [low, high], "--", color="#777777")
        for method in methods:
            selected = [row for row in class_rows if row["method"] == method]
            if not selected:
                continue
            truth = np.asarray([row["target_log10_nM"] for row in selected], dtype=float)
            prediction = np.asarray([row["prediction_log10_nM"] for row in selected], dtype=float)
            order = np.argsort(prediction, kind="mergesort")
            bins = [chunk for chunk in np.array_split(order, min(10, len(order))) if len(chunk)]
            axes[0, column].plot(
                [float(prediction[chunk].mean()) for chunk in bins],
                [float(truth[chunk].mean()) for chunk in bins],
                marker="o",
                markersize=4,
                color=colors[method],
                label=method,
            )
            axes[1, column].scatter(
                prediction,
                prediction - truth,
                s=10,
                alpha=0.35,
                color=colors[method],
                edgecolors="none",
                label=method,
            )
        axes[0, column].set_title(f"MHC-{mhc_class} decile calibration")
        axes[0, column].set_xlabel("Mean predicted log10(nM)")
        axes[0, column].set_ylabel("Mean observed log10(nM)")
        axes[1, column].axhline(0, linestyle="--", color="#777777")
        axes[1, column].set_title(f"MHC-{mhc_class} residuals")
        axes[1, column].set_xlabel("Predicted log10(nM)")
        axes[1, column].set_ylabel("Predicted − observed")
        for axis in axes[:, column]:
            axis.grid(color="#e1e1e1", linewidth=0.6)
            axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = _save(fig, output_dir, "affinity_calibration")
    plt.close(fig)
    return paths


def render_discrimination_curves(
    records: list[dict], output_dir: Path, *, stem: str, title: str
) -> list[Path]:
    """Render matched ROC and precision-recall curves for binary outputs."""
    rows = _finite_records(records, ("method", "target", "score"))
    methods = sorted({row["method"] for row in rows})
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for index, method in enumerate(methods):
        selected = [row for row in rows if row["method"] == method]
        labels = np.asarray([row["target"] for row in selected], dtype=float)
        scores = np.asarray([row["score"] for row in selected], dtype=float)
        false_positive, true_positive = _roc_curve(labels, scores)
        recall, precision = _pr_curve(labels, scores)
        roc_area = float(np.trapezoid(true_positive, false_positive))
        # Average precision uses precision at the achieved recall thresholds.
        ap = float(np.sum(np.diff(recall) * precision[1:]))
        color = f"C{index % 10}"
        axes[0].plot(false_positive, true_positive, color=color, label=f"{method} ({roc_area:.3f})")
        axes[1].plot(recall, precision, color=color, label=f"{method} ({ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#888888")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    prevalence = float(np.mean([row["target"] for row in rows if row["method"] == methods[0]]))
    axes[1].axhline(prevalence, linestyle="--", color="#888888")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(color="#e1e1e1", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    paths = _save(fig, output_dir, stem)
    plt.close(fig)
    return paths


def render_study_deltas(
    records: list[dict], output_dir: Path, *, stem: str, title: str
) -> list[Path]:
    """Render per-study paired metric differences and their pooled interval."""
    rows = _finite_records(records, ("study", "metric", "delta", "ci_low", "ci_high"))
    rows = sorted(rows, key=lambda row: (row["metric"], float(row["delta"]), row["study"]))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.2, max(3.2, 0.28 * len(rows) + 1.3)))
    y = list(reversed(range(len(rows))))
    for position, row in zip(y, rows, strict=True):
        value = float(row["delta"])
        ax.errorbar(
            value,
            position,
            xerr=[[value - float(row["ci_low"])], [float(row["ci_high"]) - value]],
            fmt="o",
            capsize=2,
            color="#285f8f" if row["study"] != "pooled_bootstrap" else "#b33b2e",
        )
    ax.axvline(0, linestyle="--", color="#777777")
    ax.set_yticks(y, [f"{row['study']} · {row['metric']}" for row in rows])
    ax.set_xlabel("Paired metric difference (first method − second method)")
    ax.set_title(title)
    ax.grid(axis="x", color="#e1e1e1", linewidth=0.6)
    fig.tight_layout()
    paths = _save(fig, output_dir, stem)
    plt.close(fig)
    return paths
