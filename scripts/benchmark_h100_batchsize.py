#!/usr/bin/env python
"""Launch a fixed-contract H100! batch-size sweep for broad class-I binding."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from experiment_registry import default_agent_label, initialize_experiment_dir
from benchmark_broad_frontier_5ep import DEFAULT_ALLELES, DEFAULT_PROBES, DEFAULT_WARM_START, DESIGNS


APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")
DESIGN_IDS = ("A03", "A05", "A06", "A07")
BATCH_SIZES = (64, 128, 192, 256)
GPU = "H100!"


@dataclass(frozen=True)
class BatchRun:
    design_id: str
    batch_size: int


def _design_map() -> Dict[str, Any]:
    return {d.design_id: d for d in DESIGNS if d.design_id in DESIGN_IDS}


def _run_id(prefix: str, design_id: str, batch_size: int) -> str:
    return f"{prefix}-{design_id.lower()}-bs{batch_size}-20260312a"


def _common_args(*, alleles: Sequence[str], probes: Sequence[str]) -> List[str]:
    return [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--probe-plot-frequency", "off",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--design-id", "batchsize",
        "--affinity-loss-mode", "assay_heads_only",
    ]


def _build_extra_args(design: Any, *, alleles: Sequence[str], probes: Sequence[str], warm_start: str) -> List[str]:
    args = _common_args(alleles=alleles, probes=probes)
    if design.warm_start:
        args.extend(["--init-checkpoint", warm_start])
    args.extend(list(design.extra_args))
    return args


def _write_manifest(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(json.dumps(list(runs), indent=2, sort_keys=True), encoding="utf-8")


def _write_variants(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# H100 Batch-Size Variants",
        "",
        "| design | batch size | app id | run id | description |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for run in runs:
        lines.append(
            f"| `{run['design_id']}` | `{run['batch_size']}` | `{run.get('app_id','')}` | `{run['run_id']}` | {run['description']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _launch(
    run: BatchRun,
    *,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
    epochs: int,
    prefix: str,
    out_dir: Path,
    timeout_s: float,
    retries: int,
) -> Dict[str, Any]:
    design = _design_map()[run.design_id]
    run_id = _run_id(prefix, design.design_id, run.batch_size)
    extra_args = _build_extra_args(design, alleles=alleles, probes=probes, warm_start=warm_start)
    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::focused_binding_run",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(run.batch_size),
        "--run-id",
        run_id,
        "--extra-args",
        " ".join(extra_args),
    ]
    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    for attempt in range(1, retries + 1):
        env = os.environ.copy()
        env["PRESTO_MODAL_GPU"] = GPU
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                proc = subprocess.Popen(
                    cmd,
                    text=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
            start = time.time()
            app_id = ""
            while True:
                if time.time() - start > timeout_s:
                    existing = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
                    raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s, output=existing)
                if log_path.exists():
                    output = log_path.read_text(encoding="utf-8", errors="replace")
                    m = APP_ID_PATTERN.search(output)
                    if m:
                        app_id = m.group(0)
                        break
                if proc.poll() is not None and not app_id:
                    break
                time.sleep(0.5)
            if not app_id:
                output = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
                raise RuntimeError(f"No app id in detached output for {run_id}:\n{output}")
            return {
                "design_id": design.design_id,
                "family": design.family,
                "description": design.description,
                "requested_gpu": GPU,
                "run_id": run_id,
                "app_id": app_id,
                "batch_size": run.batch_size,
                "epochs": epochs,
                "extra_args": extra_args,
                "launch_log": str(log_path),
            }
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to launch {run_id}:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch fixed-contract H100! batch-size sweep")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="h100-bssweep-r1")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--launch-timeout-s", type=float, default=240.0)
    parser.add_argument("--launch-retries", type=int, default=3)
    args = parser.parse_args()

    alleles = [a.strip() for a in args.alleles.split(",") if a.strip()]
    probes = [p.strip().upper() for p in args.probes.split(",") if p.strip()]
    metadata = {
        "dataset_contract": {
            "alleles": alleles,
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "assays": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"],
        },
        "training": {
            "epochs": args.epochs,
            "warm_start": args.warm_start,
            "synthetics": False,
            "ranking": False,
            "requested_gpu": GPU,
            "batch_sizes": list(BATCH_SIZES),
        },
        "tested": [
            {"design_id": d, "batch_size": bs, "gpu": GPU}
            for d in DESIGN_IDS
            for bs in BATCH_SIZES
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=args.out_dir,
        slug="h100-batchsize-bakeoff",
        title="H100! Batch-Size Bakeoff",
        source_script="scripts/benchmark_h100_batchsize.py",
        agent_label=args.agent_label,
        metadata=metadata,
    )

    runs: List[Dict[str, Any]] = []
    for design_id in DESIGN_IDS:
        for batch_size in BATCH_SIZES:
            record = _launch(
                BatchRun(design_id=design_id, batch_size=batch_size),
                alleles=alleles,
                probes=probes,
                warm_start=args.warm_start,
                epochs=args.epochs,
                prefix=args.prefix,
                out_dir=out_dir,
                timeout_s=args.launch_timeout_s,
                retries=args.launch_retries,
            )
            runs.append(record)
            _write_manifest(out_dir / "manifest.json", runs)
            _write_variants(out_dir / "variants.md", runs)


if __name__ == "__main__":
    main()
