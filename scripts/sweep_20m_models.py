#!/usr/bin/env python
"""Sweep ~20M-parameter model configs and rank early loss-drop speed.

This script:
1) enumerates architecture candidates near a target parameter budget,
2) launches short `presto train unified` runs for each candidate,
3) parses run `metrics.csv`,
4) ranks candidates by loss-drop speed, and
5) writes summary tables + plots.

Example:
    python -m presto.scripts.sweep_20m_models \
      --merged-tsv /data/merged_canonical_large_20260224a.tsv \
      --index-csv /data/mhc_index.csv \
      --epochs 3 --max-candidates 6
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from presto.models.presto import Presto


@dataclass(frozen=True)
class ModelCandidate:
    """One architecture candidate."""

    d_model: int
    n_layers: int
    n_heads: int
    n_params: int

    @property
    def tag(self) -> str:
        return f"d{self.d_model}_l{self.n_layers}_h{self.n_heads}_p{self.n_params}"


@dataclass
class CandidateRun:
    """Runtime + metrics for one candidate training run."""

    candidate: ModelCandidate
    run_dir: Path
    log_path: Path
    command: List[str]
    status: str
    return_code: int
    duration_sec: float
    train_loss_series: List[Tuple[int, float]]
    val_loss_series: List[Tuple[int, float]]
    summary: Dict[str, float]


def parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"Expected at least one integer in list: {raw!r}")
    return values


def trainable_param_count(d_model: int, n_layers: int, n_heads: int) -> int:
    model = Presto(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_candidates(
    *,
    target_params: int,
    min_params: int,
    max_params: int,
    d_models: Sequence[int],
    layer_min: int,
    layer_max: int,
    heads: Sequence[int],
    max_candidates: int,
) -> List[ModelCandidate]:
    """Generate candidate model configs near target parameter count."""
    if layer_min > layer_max:
        raise ValueError(f"layer_min must be <= layer_max, got {layer_min} > {layer_max}")

    param_cache: Dict[Tuple[int, int], int] = {}
    candidates: List[ModelCandidate] = []
    for d_model in sorted(set(int(v) for v in d_models)):
        valid_heads = sorted({h for h in heads if h > 0 and d_model % h == 0})
        if not valid_heads:
            continue
        for n_layers in range(int(layer_min), int(layer_max) + 1):
            cache_key = (d_model, n_layers)
            if cache_key not in param_cache:
                param_cache[cache_key] = trainable_param_count(
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=valid_heads[0],
                )
            n_params = int(param_cache[cache_key])
            if n_params < min_params or n_params > max_params:
                continue
            for n_heads in valid_heads:
                candidates.append(
                    ModelCandidate(
                        d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        n_params=n_params,
                    )
                )

    candidates.sort(
        key=lambda c: (
            abs(c.n_params - target_params),
            c.n_params,
            c.d_model,
            c.n_layers,
            c.n_heads,
        )
    )
    return candidates[: max(1, int(max_candidates))]


def read_metric_series(metrics_csv: Path, split: str, metric: str) -> List[Tuple[int, float]]:
    """Read one metric trajectory from RunLogger metrics.csv."""
    if not metrics_csv.exists() or metrics_csv.stat().st_size == 0:
        return []

    rows: List[Tuple[int, float]] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue
            if row.get("metric") != metric:
                continue
            try:
                step = int(float(str(row.get("step", "")).strip()))
                value = float(str(row.get("value", "")).strip())
            except (TypeError, ValueError):
                continue
            rows.append((step, value))

    rows.sort(key=lambda pair: pair[0])
    return rows


def summarize_loss_series(series: Sequence[Tuple[int, float]], prefix: str) -> Dict[str, float]:
    """Summarize speed/shape of a loss trajectory."""
    if not series:
        return {
            f"{prefix}_first_loss": float("nan"),
            f"{prefix}_last_loss": float("nan"),
            f"{prefix}_best_loss": float("nan"),
            f"{prefix}_best_step": float("nan"),
            f"{prefix}_drop_abs": float("nan"),
            f"{prefix}_drop_per_epoch": float("nan"),
            f"{prefix}_slope": float("nan"),
            f"{prefix}_speed": float("nan"),
            f"{prefix}_num_points": 0.0,
        }

    steps = [int(x) for x, _ in series]
    values = [float(y) for _, y in series]
    first = values[0]
    last = values[-1]
    best_idx = min(range(len(values)), key=lambda i: values[i])
    best = values[best_idx]
    best_step = float(steps[best_idx])
    drop_abs = first - last
    step_span = steps[-1] - steps[0]
    drop_per_epoch = drop_abs / float(step_span) if step_span > 0 else float("nan")

    n = len(steps)
    x_mean = sum(steps) / float(n)
    y_mean = sum(values) / float(n)
    denom = sum((x - x_mean) ** 2 for x in steps)
    if denom <= 0:
        slope = float("nan")
    else:
        numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(steps, values))
        slope = numer / denom
    speed = -slope if slope == slope else float("nan")  # speed: positive means descending loss

    return {
        f"{prefix}_first_loss": float(first),
        f"{prefix}_last_loss": float(last),
        f"{prefix}_best_loss": float(best),
        f"{prefix}_best_step": float(best_step),
        f"{prefix}_drop_abs": float(drop_abs),
        f"{prefix}_drop_per_epoch": float(drop_per_epoch),
        f"{prefix}_slope": float(slope),
        f"{prefix}_speed": float(speed),
        f"{prefix}_num_points": float(len(series)),
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ModelCandidate):
        return {
            "d_model": value.d_model,
            "n_layers": value.n_layers,
            "n_heads": value.n_heads,
            "n_params": value.n_params,
            "tag": value.tag,
        }
    return str(value)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_results(
    *,
    runs: Sequence[CandidateRun],
    summary_rows: Sequence[Dict[str, object]],
    out_curves: Path,
    out_speed: Path,
    out_scatter: Path,
    ranking_metric: str,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"matplotlib unavailable: {exc}"

    successful = [r for r in runs if r.status == "ok"]
    if successful:
        plt.figure(figsize=(11, 7))
        for run in successful:
            series = run.val_loss_series or run.train_loss_series
            if not series:
                continue
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            label = (
                f"{run.candidate.tag} "
                f"(val_drop={run.summary.get('val_drop_per_epoch', float('nan')):.3f})"
            )
            plt.plot(xs, ys, marker="o", linewidth=1.4, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves for ~20M Candidate Models (val preferred, train fallback)")
        plt.grid(alpha=0.2, linestyle="--")
        plt.legend(fontsize=8)
        out_curves.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_curves, dpi=180)
        plt.close()

    ranked_rows = [row for row in summary_rows if isinstance(row.get(ranking_metric), float)]
    ranked_rows = [row for row in ranked_rows if row[ranking_metric] == row[ranking_metric]]
    ranked_rows = sorted(ranked_rows, key=lambda row: float(row[ranking_metric]), reverse=True)
    if ranked_rows:
        labels = [str(row["tag"]) for row in ranked_rows]
        values = [float(row[ranking_metric]) for row in ranked_rows]
        plt.figure(figsize=(11, 7))
        plt.barh(labels[::-1], values[::-1])
        plt.xlabel(ranking_metric)
        plt.title(f"Loss-Drop Speed Ranking ({ranking_metric})")
        plt.grid(axis="x", alpha=0.2, linestyle="--")
        out_speed.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_speed, dpi=180)
        plt.close()

        xs = [float(row["n_params"]) for row in ranked_rows]
        ys = values
        plt.figure(figsize=(9, 6))
        plt.scatter(xs, ys, s=60)
        for row, x, y in zip(ranked_rows, xs, ys):
            plt.annotate(str(row["tag"]), (x, y), fontsize=8, xytext=(5, 4), textcoords="offset points")
        plt.xlabel("Trainable parameters")
        plt.ylabel(ranking_metric)
        plt.title("Parameter Count vs Loss-Drop Speed")
        plt.grid(alpha=0.2, linestyle="--")
        out_scatter.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_scatter, dpi=180)
        plt.close()
    return None


def _build_train_command(
    *,
    args: argparse.Namespace,
    candidate: ModelCandidate,
    run_dir: Path,
    checkpoint: Path,
) -> List[str]:
    cmd = [sys.executable, "-m", "presto", "train", "unified"]
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    cmd.extend(
        [
            "--data-dir",
            args.data_dir,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--d_model",
            str(candidate.d_model),
            "--n_layers",
            str(candidate.n_layers),
            "--n_heads",
            str(candidate.n_heads),
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint),
        ]
    )
    if args.merged_tsv:
        cmd.extend(["--merged-tsv", args.merged_tsv])
    if args.index_csv:
        cmd.extend(["--index-csv", args.index_csv])
    if args.no_balanced_batches:
        cmd.append("--no-balanced-batches")
    if not args.synthetic_negatives:
        cmd.extend(
            [
                "--synthetic-pmhc-negative-ratio",
                "0",
                "--synthetic-class-i-no-mhc-beta-negative-ratio",
                "0",
                "--synthetic-processing-negative-ratio",
                "0",
            ]
        )
    if args.track_probe_affinity:
        cmd.append("--track-probe-affinity")
        if args.probe_peptide:
            cmd.extend(["--probe-peptide", args.probe_peptide])
        if args.probe_alleles:
            cmd.extend(["--probe-alleles", args.probe_alleles])
    else:
        cmd.append("--no-track-probe-affinity")
    return cmd


def run_candidate(args: argparse.Namespace, candidate: ModelCandidate, sweep_dir: Path) -> CandidateRun:
    run_dir = sweep_dir / candidate.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    checkpoint = run_dir / "model.pt"
    cmd = _build_train_command(
        args=args,
        candidate=candidate,
        run_dir=run_dir,
        checkpoint=checkpoint,
    )

    if args.dry_run:
        return CandidateRun(
            candidate=candidate,
            run_dir=run_dir,
            log_path=log_path,
            command=cmd,
            status="dry_run",
            return_code=0,
            duration_sec=0.0,
            train_loss_series=[],
            val_loss_series=[],
            summary={},
        )

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("$ " + shlex.join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    duration = time.perf_counter() - t0

    metrics_csv = run_dir / "metrics.csv"
    train_series = read_metric_series(metrics_csv, split="train", metric="loss")
    val_series = read_metric_series(metrics_csv, split="val", metric="loss")
    summary: Dict[str, float] = {}
    summary.update(summarize_loss_series(train_series, "train"))
    summary.update(summarize_loss_series(val_series, "val"))

    return CandidateRun(
        candidate=candidate,
        run_dir=run_dir,
        log_path=log_path,
        command=cmd,
        status="ok" if proc.returncode == 0 else "failed",
        return_code=int(proc.returncode),
        duration_sec=float(duration),
        train_loss_series=train_series,
        val_loss_series=val_series,
        summary=summary,
    )


def _resolve_sweep_dir(path_arg: Optional[str]) -> Path:
    if path_arg:
        return Path(path_arg)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "sweeps" / f"20m_loss_drop_{ts}"


def _to_summary_row(run: CandidateRun, ranking_metric: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "tag": run.candidate.tag,
        "d_model": run.candidate.d_model,
        "n_layers": run.candidate.n_layers,
        "n_heads": run.candidate.n_heads,
        "n_params": run.candidate.n_params,
        "status": run.status,
        "return_code": run.return_code,
        "duration_sec": run.duration_sec,
        "run_dir": str(run.run_dir),
        "log_path": str(run.log_path),
    }
    for key, value in run.summary.items():
        row[key] = value
    row["ranking_metric"] = ranking_metric
    row["ranking_value"] = float(run.summary.get(ranking_metric, float("nan")))
    return row


def _print_ranked_table(summary_rows: Sequence[Dict[str, object]], ranking_metric: str) -> None:
    usable = [row for row in summary_rows if isinstance(row.get("ranking_value"), float)]
    usable = [row for row in usable if row["ranking_value"] == row["ranking_value"]]
    usable = sorted(usable, key=lambda row: float(row["ranking_value"]), reverse=True)
    print("\nTop candidates by", ranking_metric)
    for idx, row in enumerate(usable, start=1):
        print(
            f"{idx:>2}. {row['tag']}: "
            f"{ranking_metric}={float(row['ranking_value']):.6f}, "
            f"val_first={float(row.get('val_first_loss', float('nan'))):.4f}, "
            f"val_last={float(row.get('val_last_loss', float('nan'))):.4f}, "
            f"duration={float(row.get('duration_sec', float('nan'))):.1f}s"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep near-20M Presto model hyperparameters and rank loss-drop speed."
    )
    parser.add_argument("--sweep-dir", type=str, default=None, help="Output directory for sweep artifacts")
    parser.add_argument("--target-params", type=int, default=20_000_000, help="Target trainable parameter count")
    parser.add_argument("--min-params", type=int, default=18_000_000, help="Minimum trainable parameter count")
    parser.add_argument("--max-params", type=int, default=22_000_000, help="Maximum trainable parameter count")
    parser.add_argument(
        "--d-models",
        type=str,
        default="160,192,224,256,288,320",
        help="Comma-separated d_model candidates",
    )
    parser.add_argument("--layer-min", type=int, default=2, help="Minimum n_layers to consider")
    parser.add_argument("--layer-max", type=int, default=6, help="Maximum n_layers to consider")
    parser.add_argument(
        "--heads",
        type=str,
        default="4,8,10,12,16",
        help="Comma-separated n_heads candidates (filtered by d_model divisibility)",
    )
    parser.add_argument("--max-candidates", type=int, default=6, help="Max candidate configs to run")

    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory for unified training")
    parser.add_argument("--merged-tsv", type=str, default=None, help="Merged TSV path")
    parser.add_argument("--index-csv", type=str, default=None, help="MHC index CSV path")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per candidate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per candidate")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate per candidate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay per candidate")
    parser.add_argument(
        "--synthetic-negatives",
        dest="synthetic_negatives",
        action="store_true",
        default=True,
        help="Enable synthetic negatives in candidate runs (default: true)",
    )
    parser.add_argument(
        "--no-synthetic-negatives",
        dest="synthetic_negatives",
        action="store_false",
        help="Disable synthetic negatives for cleaner architecture comparison",
    )
    parser.add_argument(
        "--no-balanced-batches",
        action="store_true",
        default=False,
        help="Disable balanced batch sampler in candidate runs",
    )
    parser.add_argument(
        "--track-probe-affinity",
        action="store_true",
        default=False,
        help="Enable per-epoch probe affinity logging during sweep runs",
    )
    parser.add_argument("--probe-peptide", type=str, default="SLLQHLIGL", help="Probe peptide when probe tracking is enabled")
    parser.add_argument("--probe-alleles", type=str, default="HLA-A*02:01,HLA-A*24:02", help="Probe allele CSV when probe tracking is enabled")
    parser.add_argument(
        "--ranking-metric",
        type=str,
        default="val_drop_per_epoch",
        choices=["val_drop_per_epoch", "val_speed", "train_drop_per_epoch", "train_speed"],
        help="Metric used to rank candidate loss-drop speed",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra raw arguments appended to `presto train unified`",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print/run planning only; do not launch training")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="Continue sweep if a candidate run fails (default: true)",
    )
    parser.add_argument(
        "--fail-fast",
        dest="keep_going",
        action="store_false",
        help="Abort sweep at first failed candidate run",
    )
    args = parser.parse_args(argv)

    sweep_dir = _resolve_sweep_dir(args.sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    d_models = parse_int_list(args.d_models)
    heads = parse_int_list(args.heads)
    candidates = generate_candidates(
        target_params=args.target_params,
        min_params=args.min_params,
        max_params=args.max_params,
        d_models=d_models,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        heads=heads,
        max_candidates=args.max_candidates,
    )
    if not candidates:
        raise RuntimeError(
            "No candidate architectures found in requested parameter band. "
            "Try widening --min-params/--max-params or adjusting d-models/layer/head ranges."
        )

    print(f"Sweep dir: {sweep_dir}")
    print("Candidates:")
    for c in candidates:
        print(f"  - {c.tag}")

    runs: List[CandidateRun] = []
    for i, candidate in enumerate(candidates, start=1):
        print(f"\n[{i}/{len(candidates)}] Running {candidate.tag}")
        run = run_candidate(args, candidate, sweep_dir)
        runs.append(run)
        print(
            f"  status={run.status}, return_code={run.return_code}, "
            f"duration_sec={run.duration_sec:.1f}, run_dir={run.run_dir}"
        )
        if run.status == "failed" and not args.keep_going:
            print("Aborting sweep due to --fail-fast")
            break

    summary_rows = [_to_summary_row(run, args.ranking_metric) for run in runs]
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda row: (
            row.get("status") != "ok",
            -float(row.get("ranking_value", float("-inf")))
            if float(row.get("ranking_value", float("nan")))
            == float(row.get("ranking_value", float("nan")))
            else float("inf"),
        ),
    )

    summary_csv = sweep_dir / "summary.csv"
    write_summary_csv(summary_csv, summary_rows_sorted)

    runs_json = sweep_dir / "runs.json"
    runs_payload = []
    for run in runs:
        runs_payload.append(
            {
                "candidate": run.candidate,
                "run_dir": run.run_dir,
                "log_path": run.log_path,
                "command": run.command,
                "status": run.status,
                "return_code": run.return_code,
                "duration_sec": run.duration_sec,
                "train_loss_series": run.train_loss_series,
                "val_loss_series": run.val_loss_series,
                "summary": run.summary,
            }
        )
    runs_json.write_text(
        json.dumps(runs_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    curves_png = sweep_dir / "val_loss_curves.png"
    speed_png = sweep_dir / "loss_drop_speed.png"
    scatter_png = sweep_dir / "params_vs_speed.png"
    plot_error = _plot_results(
        runs=runs,
        summary_rows=summary_rows_sorted,
        out_curves=curves_png,
        out_speed=speed_png,
        out_scatter=scatter_png,
        ranking_metric=args.ranking_metric,
    )

    _print_ranked_table(summary_rows_sorted, args.ranking_metric)
    print("\nArtifacts:")
    print(f"  - summary CSV: {summary_csv}")
    print(f"  - run records: {runs_json}")
    if plot_error is None:
        for label, path in (
            ("curves plot", curves_png),
            ("speed plot", speed_png),
            ("params/speed scatter", scatter_png),
        ):
            if path.exists():
                print(f"  - {label}: {path}")
            else:
                print(f"  - {label}: not generated")
    else:
        print(f"  - plots skipped: {plot_error}")


if __name__ == "__main__":
    main()
