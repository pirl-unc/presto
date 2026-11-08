"""Deterministic figures for versioned Presto performance reports.

The functions in this module consume only validated manifest records. They do
not discover datasets, checkpoints, tools, or Git state.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path


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
