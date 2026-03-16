#!/usr/bin/env python
"""Launch and collect fixed-epoch runtime variants on the 44k multi-allele contract."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from experiment_registry import default_agent_label, initialize_experiment_dir


BASE_ALLELES = (
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02",
)
BASE_EXTRA_ARGS = [
    "--source", "iedb",
    "--alleles", ",".join(BASE_ALLELES),
    "--probe-peptide", "SLLQHLIGL",
    "--extra-probe-peptides", "FLRYLLFGI,NFLIKFLLI",
    "--measurement-profile", "all_binding_rows",
    "--qualifier-filter", "all",
    "--groove-pos-mode", "triple",
    "--binding-core-lengths", "8,9,10,11",
    "--binding-core-refinement", "shared",
    "--affinity-loss-mode", "assay_heads_only",
    "--binding-contrastive-weight", "1.0",
    "--binding-peptide-contrastive-weight", "0.5",
    "--init-checkpoint", "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt",
]


@dataclass(frozen=True)
class Variant:
    variant_id: str
    description: str
    extra_args: Tuple[str, ...]


VARIANTS: Sequence[Variant] = (
    Variant("R00", "nw=0 pin=0 tf32=0", ("--no-persistent-workers",)),
    Variant("R01", "nw=0 pin=1 tf32=0", ("--pin-memory", "--no-persistent-workers")),
    Variant("R02", "nw=0 pin=0 tf32=1", ("--allow-tf32", "--matmul-precision", "high", "--no-persistent-workers")),
    Variant("R03", "nw=0 pin=1 tf32=1", ("--pin-memory", "--allow-tf32", "--matmul-precision", "high", "--no-persistent-workers")),
    Variant("R04", "nw=2 pin=1 persist=0 p2 tf32=0", ("--num-workers", "2", "--pin-memory", "--no-persistent-workers", "--prefetch-factor", "2")),
    Variant("R05", "nw=4 pin=1 persist=0 p2 tf32=0", ("--num-workers", "4", "--pin-memory", "--no-persistent-workers", "--prefetch-factor", "2")),
    Variant("R06", "nw=8 pin=1 persist=0 p2 tf32=0", ("--num-workers", "8", "--pin-memory", "--no-persistent-workers", "--prefetch-factor", "2")),
    Variant("R07", "nw=2 pin=1 persist=1 p2 tf32=0", ("--num-workers", "2", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2")),
    Variant("R08", "nw=4 pin=1 persist=1 p2 tf32=0", ("--num-workers", "4", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2")),
    Variant("R09", "nw=8 pin=1 persist=1 p2 tf32=0", ("--num-workers", "8", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2")),
    Variant("R10", "nw=2 pin=1 persist=0 p2 tf32=1", ("--num-workers", "2", "--pin-memory", "--no-persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high")),
    Variant("R11", "nw=4 pin=1 persist=0 p2 tf32=1", ("--num-workers", "4", "--pin-memory", "--no-persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high")),
    Variant("R12", "nw=8 pin=1 persist=0 p2 tf32=1", ("--num-workers", "8", "--pin-memory", "--no-persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high")),
    Variant("R13", "nw=2 pin=1 persist=1 p2 tf32=1", ("--num-workers", "2", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high")),
    Variant("R14", "nw=4 pin=1 persist=1 p2 tf32=1", ("--num-workers", "4", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high")),
    Variant("R15", "nw=8 pin=1 persist=1 p2 tf32=1", ("--num-workers", "8", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high")),
)


def _build_extra_args(variant: Variant) -> str:
    args = list(BASE_EXTRA_ARGS)
    args.extend(["--design-id", variant.variant_id])
    args.extend(list(variant.extra_args))
    return " ".join(args)


def _write_manifest(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Multi-Allele Runtime Variants",
        "",
        "| variant | description | run_id | app_id | url |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant_id']}` | {row['description']} | `{row['run_id']}` | `{row.get('app_id', '')}` | {row.get('url', '')} |"
        )
    (output_dir / "variants.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _launch_variant(
    *,
    variant: Variant,
    epochs: int,
    batch_size: int,
    output_dir: Path,
    stamp: str,
) -> Dict[str, Any]:
    run_id = f"runtime44k-{variant.variant_id.lower()}-{stamp}"
    extra_args = _build_extra_args(variant)
    launch_log = output_dir / f"{run_id}.launch.log"
    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::focused_binding_run",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--run-id",
        run_id,
        "--extra-args",
        extra_args,
    ]
    with launch_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=output_dir.parent.parent,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            close_fds=True,
        )
    return {
        "variant_id": variant.variant_id,
        "description": variant.description,
        "run_id": run_id,
        "app_id": "",
        "url": "",
        "extra_args": extra_args,
        "launch_pid": str(proc.pid),
        "launch_log": str(launch_log),
        "status": "launching",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_artifact(run_id: str, remote_relpath: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "modal",
        "volume",
        "get",
        "--force",
        "presto-checkpoints",
        f"{run_id}/{remote_relpath}",
        str(dest_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and dest_path.exists()


def _probe_rows_by_epoch(path: Path) -> Dict[int, List[Mapping[str, Any]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        epoch = int(row.get("epoch", 0))
        grouped.setdefault(epoch, []).append(row)
    return grouped


def _probe_value(rows: Sequence[Mapping[str, Any]], peptide: str, allele: str) -> Optional[float]:
    for row in rows:
        if str(row.get("peptide")) == peptide and str(row.get("allele")) == allele:
            value = row.get("ic50_nM")
            if value is not None:
                return float(value)
            value = row.get("kd_nM")
            if value is not None:
                return float(value)
    return None


def _safe_ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom is None or denom <= 0.0:
        return None
    return float(numer / denom)


def _mean_metric(epochs: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(epoch[key]) for epoch in epochs if epoch.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _max_metric(epochs: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(epoch[key]) for epoch in epochs if epoch.get(key) is not None]
    if not values:
        return None
    return float(max(values))


def _best_epoch(summary: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    epochs = summary.get("epochs", [])
    if not isinstance(epochs, list) or not epochs:
        return None
    valid = [epoch for epoch in epochs if epoch.get("val_loss") is not None]
    if not valid:
        return None
    return min(valid, key=lambda epoch: float(epoch["val_loss"]))


def _effectiveness_row(
    *,
    manifest_row: Mapping[str, Any],
    summary: Mapping[str, Any],
    probe_by_epoch: Mapping[int, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    epochs = list(summary.get("epochs", []))
    best_epoch = _best_epoch(summary)
    best_epoch_idx = int(best_epoch["epoch"]) if best_epoch is not None else None
    probe_rows = probe_by_epoch.get(best_epoch_idx or 0, [])
    sll_a02 = _probe_value(probe_rows, "SLLQHLIGL", "HLA-A*02:01")
    sll_a24 = _probe_value(probe_rows, "SLLQHLIGL", "HLA-A*24:02")
    flr_a02 = _probe_value(probe_rows, "FLRYLLFGI", "HLA-A*02:01")
    flr_a24 = _probe_value(probe_rows, "FLRYLLFGI", "HLA-A*24:02")
    nfl_a02 = _probe_value(probe_rows, "NFLIKFLLI", "HLA-A*02:01")
    nfl_a24 = _probe_value(probe_rows, "NFLIKFLLI", "HLA-A*24:02")
    correct = 0
    if sll_a02 is not None and sll_a24 is not None and sll_a24 > sll_a02:
        correct += 1
    if flr_a02 is not None and flr_a24 is not None and flr_a24 > flr_a02:
        correct += 1
    if nfl_a02 is not None and nfl_a24 is not None and nfl_a02 > nfl_a24:
        correct += 1
    sll_ratio = _safe_ratio(sll_a24, sll_a02)
    flr_ratio = _safe_ratio(flr_a24, flr_a02)
    nfl_ratio = _safe_ratio(nfl_a02, nfl_a24)
    return {
        "variant_id": manifest_row["variant_id"],
        "description": manifest_row["description"],
        "run_id": manifest_row["run_id"],
        "best_epoch": best_epoch_idx,
        "setup_wall_s": float(summary.get("setup_wall_s", 0.0) or 0.0),
        "epoch_wall_s_mean": _mean_metric(epochs, "epoch_wall_s"),
        "train_data_wait_s_mean": _mean_metric(epochs, "train_data_wait_s"),
        "train_forward_loss_s_mean": _mean_metric(epochs, "train_forward_loss_s"),
        "train_backward_s_mean": _mean_metric(epochs, "train_backward_s"),
        "train_optimizer_s_mean": _mean_metric(epochs, "train_optimizer_s"),
        "gpu_busy_wall_s_mean": _mean_metric(epochs, "gpu_busy_wall_s"),
        "gpu_util_mean_pct_mean": _mean_metric(epochs, "gpu_util_mean_pct"),
        "gpu_util_peak_pct_max": _max_metric(epochs, "gpu_util_peak_pct"),
        "gpu_mem_util_mean_pct_mean": _mean_metric(epochs, "gpu_mem_util_mean_pct"),
        "gpu_peak_allocated_gib_max": _max_metric(epochs, "gpu_peak_allocated_gib"),
        "gpu_peak_reserved_gib_max": _max_metric(epochs, "gpu_peak_reserved_gib"),
        "best_val_loss": (float(best_epoch["val_loss"]) if best_epoch is not None else None),
        "sll_a02_ic50_nM": sll_a02,
        "sll_a24_ic50_nM": sll_a24,
        "sll_ratio_a24_over_a02": sll_ratio,
        "flr_a02_ic50_nM": flr_a02,
        "flr_a24_ic50_nM": flr_a24,
        "flr_ratio_a24_over_a02": flr_ratio,
        "nfl_a02_ic50_nM": nfl_a02,
        "nfl_a24_ic50_nM": nfl_a24,
        "nfl_ratio_a02_over_a24": nfl_ratio,
        "correct_probe_orders": correct,
    }


def _write_summary(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    lines = [
        "# Multi-Allele Runtime Benchmark",
        "",
        "| variant | epoch wall s | setup s | gpu util mean % | val loss | `SLLQHLIGL` A02 / A24 | ratio | correct probes |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        epoch_wall = row.get("epoch_wall_s_mean")
        setup_wall = row.get("setup_wall_s")
        gpu_util = row.get("gpu_util_mean_pct_mean")
        val_loss = row.get("best_val_loss")
        sll = f"{row.get('sll_a02_ic50_nM', None)} / {row.get('sll_a24_ic50_nM', None)}"
        ratio = row.get("sll_ratio_a24_over_a02")
        epoch_wall_text = f"{float(epoch_wall):.2f}" if isinstance(epoch_wall, (int, float)) else ""
        setup_wall_text = f"{float(setup_wall):.2f}" if isinstance(setup_wall, (int, float)) else ""
        gpu_util_text = f"{float(gpu_util):.1f}" if isinstance(gpu_util, (int, float)) else ""
        val_loss_text = f"{float(val_loss):.4f}" if isinstance(val_loss, (int, float)) else ""
        ratio_text = f"{ratio:.1f}x" if isinstance(ratio, (int, float)) and math.isfinite(ratio) else ""
        lines.append(
            f"| `{row['variant_id']}` | "
            f"{epoch_wall_text} | "
            f"{setup_wall_text} | "
            f"{gpu_util_text} | "
            f"{val_loss_text} | "
            f"`{sll}` | "
            f"{ratio_text} | "
            f"{int(row.get('correct_probe_orders', 0))} |"
        )
    (output_dir / "options_vs_perf.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch(args: argparse.Namespace) -> None:
    output_dir = initialize_experiment_dir(
        out_dir=str(args.output_dir),
        slug="runtime-multiallele-44k",
        title="Runtime Multi-Allele 44k Benchmark",
        source_script="scripts/benchmark_runtime_multiallele.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": list(BASE_ALLELES),
                "measurement_profile": "all_binding_rows",
                "qualifier_filter": "all",
                "loss": "censored KD/IC50/EC50",
                "ranking": "on",
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "variants": [variant.variant_id for variant in VARIANTS],
            },
            "tested": [variant.variant_id for variant in VARIANTS],
        },
    )
    stamp = str(args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        row = _launch_variant(
            variant=variant,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            output_dir=output_dir,
            stamp=stamp,
        )
        rows.append(row)
        _write_manifest(output_dir, rows)
        print(json.dumps(row, sort_keys=True), flush=True)


def collect(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    result_rows: List[Dict[str, Any]] = []
    for row in manifest:
        run_id = str(row["run_id"])
        run_dir = output_dir / run_id
        summary_path = run_dir / "summary.json"
        probe_path = run_dir / "probe_affinity_over_epochs.json"
        fetched_summary = _fetch_artifact(run_id, "summary.json", summary_path)
        fetched_probe = _fetch_artifact(run_id, "probe_affinity_over_epochs.json", probe_path)
        if not (fetched_summary and fetched_probe):
            result_rows.append(
                {
                    "variant_id": row["variant_id"],
                    "description": row["description"],
                    "run_id": run_id,
                    "status": "missing_artifacts",
                }
            )
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        probe_by_epoch = _probe_rows_by_epoch(probe_path)
        result = _effectiveness_row(
            manifest_row=row,
            summary=summary,
            probe_by_epoch=probe_by_epoch,
        )
        result["status"] = "ok"
        result_rows.append(result)
    ok_rows = [row for row in result_rows if row.get("status") == "ok"]
    ok_rows_sorted = sorted(
        ok_rows,
        key=lambda row: (
            -int(row.get("correct_probe_orders", 0)),
            -(float(row.get("sll_ratio_a24_over_a02") or 0.0)),
            float(row.get("best_val_loss") or float("inf")),
            float(row.get("epoch_wall_s_mean") or float("inf")),
        ),
    )
    _write_summary(output_dir, ok_rows_sorted + [row for row in result_rows if row.get("status") != "ok"])
    print(json.dumps({"rows": result_rows, "top": ok_rows_sorted[:3]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch/collect the 44k multi-allele runtime benchmark")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--epochs", type=int, default=3)
    launch_parser.add_argument("--batch-size", type=int, default=140)
    launch_parser.add_argument("--agent-label", type=str, default=default_agent_label())
    launch_parser.add_argument("--output-dir", type=str, default="")
    launch_parser.add_argument("--stamp", type=str, default="")
    launch_parser.set_defaults(func=launch)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output-dir", type=str, default="modal_runs/runtime_multiallele_44k")
    collect_parser.set_defaults(func=collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
