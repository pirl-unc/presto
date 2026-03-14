#!/usr/bin/env python
"""Launch and summarize Stage-A register-design benchmarks on Modal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from experiment_registry import default_agent_label, initialize_experiment_dir


DEFAULT_ALLELES = (
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02",
)
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI")
DEFAULT_SEEDS = (41, 42, 43)
DEFAULT_WARM_START = "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")
PROBE_EXPECTATIONS = {
    "SLLQHLIGL": ("HLA-A*02:01", "HLA-A*24:02"),
    "FLRYLLFGI": ("HLA-A*02:01", "HLA-A*24:02"),
    "NFLIKFLLI": ("HLA-A*24:02", "HLA-A*02:01"),
}


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    groove_pos_mode: str
    binding_core_lengths: str
    binding_core_refinement: str


DESIGNS: Tuple[DesignSpec, ...] = (
    DesignSpec(
        design_id="M0",
        groove_pos_mode="sequential",
        binding_core_lengths="9",
        binding_core_refinement="shared",
    ),
    DesignSpec(
        design_id="M1",
        groove_pos_mode="triple",
        binding_core_lengths="8,9,10,11",
        binding_core_refinement="shared",
    ),
    DesignSpec(
        design_id="M2",
        groove_pos_mode="triple",
        binding_core_lengths="8,9,10,11",
        binding_core_refinement="class_specific",
    ),
)


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _run_id(prefix: str, design_id: str, seed: int) -> str:
    return f"{prefix}-{design_id.lower()}-seed{seed}"


def _build_extra_args(
    *,
    design: DesignSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
) -> str:
    args = [
        "--alleles",
        ",".join(alleles),
        "--probe-peptide",
        probes[0],
        "--extra-probe-peptides",
        ",".join(probes[1:]),
        "--design-id",
        design.design_id,
        "--groove-pos-mode",
        design.groove_pos_mode,
        "--binding-core-lengths",
        design.binding_core_lengths,
        "--binding-core-refinement",
        design.binding_core_refinement,
        "--measurement-profile",
        "direct_affinity_only",
        "--measurement-type-filter",
        "ic50",
        "--qualifier-filter",
        "exact",
        "--affinity-loss-mode",
        "full",
        "--binding-peptide-contrastive-weight",
        "0.5",
        "--binding-contrastive-weight",
        "0.0",
        "--no-synthetic-negatives",
        "--balanced-batches",
        "--init-checkpoint",
        warm_start,
    ]
    return " ".join(args)


def _call(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        check=check,
        text=True,
        capture_output=True,
    )


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _launch_run(
    *,
    run_id: str,
    design: DesignSpec,
    seed: int,
    epochs: int,
    batch_size: int,
    warm_start: str,
    alleles: Sequence[str],
    probes: Sequence[str],
) -> Dict[str, Any]:
    extra_args = _build_extra_args(
        design=design,
        alleles=alleles,
        probes=probes,
        warm_start=warm_start,
    )
    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::focused_binding_run",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--data-dir",
        "/data",
        "--run-id",
        run_id,
        "--extra-args",
        extra_args,
    ]
    timeout_sec = 40.0
    output = ""
    timed_out = False
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = "\n".join(
            part
            for part in (_coerce_text(exc.stdout), _coerce_text(exc.stderr))
            if part
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            part
            for part in (_coerce_text(exc.stdout), _coerce_text(exc.stderr))
            if part
        )
        raise RuntimeError(f"Failed to launch run {run_id}:\n{output}") from exc
    match = APP_ID_PATTERN.search(output)
    if match is None:
        raise RuntimeError(f"Detached Modal launch for {run_id} did not emit an app id:\n{output}")
    return {
        "run_id": run_id,
        "design_id": design.design_id,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "app_id": match.group(0) if match else None,
        "launch_cmd": cmd,
        "launch_output": output.strip(),
        "status": "launched" if not timed_out else "launched_detached",
    }


def _fetch_volume_file(*, run_id: str, remote_name: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "modal",
        "volume",
        "get",
        "--force",
        "presto-checkpoints",
        f"{run_id}/{remote_name}",
        str(local_path),
    ]
    result = _call(cmd, check=False)
    return result.returncode == 0 and local_path.exists()


def _load_probe_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _best_epoch(summary: Mapping[str, Any]) -> Optional[int]:
    epochs = list(summary.get("epochs", []))
    if not epochs:
        return None
    best = min(
        (
            epoch
            for epoch in epochs
            if isinstance(epoch, Mapping) and epoch.get("val_loss") is not None
        ),
        key=lambda row: float(row["val_loss"]),
        default=None,
    )
    if not best:
        return None
    return int(best["epoch"])


def _probe_value_at_epoch(
    rows: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    peptide: str,
    allele: str,
) -> Optional[float]:
    for row in rows:
        if (
            int(row.get("epoch", -1)) == int(epoch)
            and str(row.get("peptide", "")).strip().upper() == peptide
            and str(row.get("allele", "")).strip() == allele
        ):
            raw = row.get("ic50_nM") or row.get("kd_nM")
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
    return None


def _probe_metrics(summary: Mapping[str, Any], probe_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    best_epoch = _best_epoch(summary)
    if best_epoch is None:
        return {}
    correct_ge_1p5 = 0
    correct_any = 0
    margins: List[float] = []
    probes: Dict[str, Any] = {}
    for peptide, (stronger, weaker) in PROBE_EXPECTATIONS.items():
        strong_v = _probe_value_at_epoch(probe_rows, epoch=best_epoch, peptide=peptide, allele=stronger)
        weak_v = _probe_value_at_epoch(probe_rows, epoch=best_epoch, peptide=peptide, allele=weaker)
        if strong_v is None or weak_v is None or strong_v <= 0.0 or weak_v <= 0.0:
            continue
        ratio = weak_v / strong_v
        margin = math.log10(ratio)
        correct = bool(ratio > 1.0)
        strong = bool(ratio >= 1.5)
        if correct:
            correct_any += 1
        if strong:
            correct_ge_1p5 += 1
        margins.append(margin)
        probes[peptide] = {
            "best_epoch": int(best_epoch),
            "expected_stronger": stronger,
            "expected_weaker": weaker,
            "stronger_nM": strong_v,
            "weaker_nM": weak_v,
            "ratio": ratio,
            "log10_margin": margin,
            "correct": correct,
            "correct_ge_1p5": strong,
        }
    best_val_loss = None
    for row in summary.get("epochs", []):
        if int(row.get("epoch", -1)) == int(best_epoch):
            try:
                best_val_loss = float(row["val_loss"])
            except (TypeError, ValueError, KeyError):
                best_val_loss = None
            break
    return {
        "best_epoch": int(best_epoch),
        "best_val_loss": best_val_loss,
        "probe_correct_any": int(correct_any),
        "probe_correct_ge_1p5": int(correct_ge_1p5),
        "probe_mean_log10_margin": (sum(margins) / len(margins)) if margins else None,
        "probes": probes,
    }


def _score_tuple(metrics: Mapping[str, Any]) -> Tuple[float, float, float]:
    correct = float(metrics.get("probe_correct_ge_1p5", -1))
    margin = float(metrics.get("probe_mean_log10_margin", float("-inf")) or float("-inf"))
    val_loss = float(metrics.get("best_val_loss", float("inf")) or float("inf"))
    return (correct, margin, -val_loss)


def _refresh_run(local_root: Path, run: Mapping[str, Any]) -> Dict[str, Any]:
    run_id = str(run["run_id"])
    run_root = local_root / run_id
    summary_path = run_root / "summary.json"
    probe_csv_path = run_root / "probe_affinity_over_epochs.csv"
    fetched_summary = _fetch_volume_file(run_id=run_id, remote_name="summary.json", local_path=summary_path)
    fetched_probe = _fetch_volume_file(
        run_id=run_id,
        remote_name="probe_affinity_over_epochs.csv",
        local_path=probe_csv_path,
    )

    updated = dict(run)
    updated["local_root"] = str(run_root)
    updated["summary_present"] = bool(fetched_summary)
    updated["probe_csv_present"] = bool(fetched_probe)
    if not summary_path.exists():
        return updated

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    probe_rows = _load_probe_rows(probe_csv_path)
    metrics = _probe_metrics(summary, probe_rows)
    updated["status"] = "running"
    updated["metrics"] = metrics
    updated["summary_path"] = str(summary_path)
    updated["probe_csv_path"] = str(probe_csv_path)
    completed_epochs = len(list(summary.get("epochs", [])))
    updated["completed_epochs"] = completed_epochs
    if completed_epochs >= int(run.get("epochs", 0)):
        updated["status"] = "complete"
    return updated


def _design_rollup(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for run in runs:
        metrics = run.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        grouped.setdefault(str(run["design_id"]), []).append(run)

    rows: List[Dict[str, Any]] = []
    for design_id, items in grouped.items():
        count = len(items)
        rows.append(
            {
                "design_id": design_id,
                "runs_reported": count,
                "avg_probe_correct_ge_1p5": sum(
                    float(run["metrics"].get("probe_correct_ge_1p5", 0.0)) for run in items
                )
                / count,
                "avg_probe_correct_any": sum(
                    float(run["metrics"].get("probe_correct_any", 0.0)) for run in items
                )
                / count,
                "avg_probe_mean_log10_margin": sum(
                    float(run["metrics"].get("probe_mean_log10_margin", 0.0) or 0.0) for run in items
                )
                / count,
                "avg_best_val_loss": sum(
                    float(run["metrics"].get("best_val_loss", 0.0) or 0.0) for run in items
                )
                / count,
                "best_run_id": max(items, key=lambda run: _score_tuple(run["metrics"]))["run_id"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["avg_probe_correct_ge_1p5"],
            row["avg_probe_mean_log10_margin"],
            -row["avg_best_val_loss"],
        ),
        reverse=True,
    )
    return rows


def _write_leaderboard(output_dir: Path, runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    design_rows = _design_rollup(runs)
    ranked_runs = [
        run for run in runs if isinstance(run.get("metrics"), Mapping)
    ]
    ranked_runs.sort(key=lambda run: _score_tuple(run["metrics"]), reverse=True)
    payload = {
        "generated_at_unix": time.time(),
        "leader_by_run": ranked_runs[0] if ranked_runs else None,
        "leader_by_design": design_rows[0] if design_rows else None,
        "design_rows": design_rows,
        "runs": list(ranked_runs),
    }
    (output_dir / "leaderboard.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Stage-A register benchmark leaderboard",
        "",
    ]
    if design_rows:
        lines += [
            "## Design means",
            "",
            "| design | runs | >=1.5x correct | any correct | mean log10 margin | mean best val loss | best run |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in design_rows:
            lines.append(
                "| {design_id} | {runs_reported} | {avg_probe_correct_ge_1p5:.2f} | {avg_probe_correct_any:.2f} | {avg_probe_mean_log10_margin:.3f} | {avg_best_val_loss:.4f} | {best_run_id} |".format(
                    **row
                )
            )
        lines.append("")
    if ranked_runs:
        lines += [
            "## Runs",
            "",
            "| run | design | seed | status | best epoch | >=1.5x correct | mean log10 margin | best val loss |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for run in ranked_runs:
            metrics = run["metrics"]
            lines.append(
                f"| {run['run_id']} | {run['design_id']} | {run['seed']} | {run['status']} | "
                f"{metrics.get('best_epoch', '')} | {metrics.get('probe_correct_ge_1p5', '')} | "
                f"{(metrics.get('probe_mean_log10_margin') if metrics.get('probe_mean_log10_margin') is not None else float('nan')):.3f} | "
                f"{(metrics.get('best_val_loss') if metrics.get('best_val_loss') is not None else float('nan')):.4f} |"
            )
    (output_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _iter_target_runs(
    *,
    prefix: str,
    seeds: Iterable[int],
) -> Iterable[Tuple[str, DesignSpec, int]]:
    for design in DESIGNS:
        for seed in seeds:
            yield _run_id(prefix, design.design_id, seed), design, int(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch and poll Stage-A register design benchmarks")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--prefix", type=str, default="register-stagea-20260308")
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-interval-sec", type=float, default=180.0)
    parser.add_argument("--max-polls", type=int, default=1)
    args = parser.parse_args()

    output_dir = initialize_experiment_dir(
        out_dir=str(args.output_dir),
        slug="register-stage-a",
        title="Register Design Stage A",
        source_script="scripts/benchmark_register_designs.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": DEFAULT_ALLELES,
                "measurement_profile": "direct_affinity_only",
                "measurement_type_filter": "ic50",
                "qualifier_filter": "exact",
                "warm_start": str(args.warm_start),
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "seeds": [int(seed) for seed in _parse_csv(str(args.seeds))],
            },
            "tested": [design.design_id for design in DESIGNS],
        },
    )
    manifest_path = output_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    known_by_run = {str(item["run_id"]): dict(item) for item in manifest.get("runs", [])}

    seeds = [int(seed) for seed in _parse_csv(str(args.seeds))]
    alleles = _parse_csv(str(args.alleles))
    probes = [probe.strip().upper() for probe in _parse_csv(str(args.probes))]
    if len(probes) < 3:
        raise ValueError("Expected at least three probe peptides")

    if args.launch:
        for run_id, design, seed in _iter_target_runs(prefix=str(args.prefix), seeds=seeds):
            if run_id in known_by_run:
                continue
            launched = _launch_run(
                run_id=run_id,
                design=design,
                seed=seed,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                warm_start=str(args.warm_start),
                alleles=alleles,
                probes=probes,
            )
            known_by_run[run_id] = launched
            _write_manifest(manifest_path, {"runs": list(known_by_run.values())})
            print(
                json.dumps(
                    {
                        "event": "launched",
                        "run_id": run_id,
                        "design_id": design.design_id,
                        "seed": seed,
                        "app_id": launched.get("app_id"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if args.poll:
        last_leader_key: Optional[Tuple[Any, ...]] = None
        for poll_idx in range(max(int(args.max_polls), 1)):
            refreshed: List[Dict[str, Any]] = []
            for run in known_by_run.values():
                refreshed.append(_refresh_run(output_dir, run))
            known_by_run = {str(run["run_id"]): run for run in refreshed}
            _write_manifest(manifest_path, {"runs": list(known_by_run.values())})
            leaderboard = _write_leaderboard(output_dir, list(known_by_run.values()))
            leader = leaderboard.get("leader_by_run")
            if isinstance(leader, Mapping):
                metrics = leader.get("metrics", {})
                key = (
                    leader.get("run_id"),
                    metrics.get("probe_correct_ge_1p5"),
                    metrics.get("probe_mean_log10_margin"),
                    metrics.get("best_val_loss"),
                )
                if key != last_leader_key:
                    print(
                        json.dumps(
                            {
                                "event": "leader_update",
                                "run_id": leader.get("run_id"),
                                "design_id": leader.get("design_id"),
                                "seed": leader.get("seed"),
                                "status": leader.get("status"),
                                "metrics": metrics,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    last_leader_key = key
            if poll_idx + 1 < int(args.max_polls):
                time.sleep(max(float(args.poll_interval_sec), 5.0))


if __name__ == "__main__":
    main()
