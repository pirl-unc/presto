#!/usr/bin/env python
"""Launch the broad-contract A03/A07 target-space bakeoff on Modal."""

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
DEFAULT_WARM_START = "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
GPU = "H100!"
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    description: str
    extra_args: Sequence[str]


TARGET_SPACES = (
    ("log10_50k", "log10", 50000),
    ("log10_100k", "log10", 100000),
    ("mhcflurry_50k", "mhcflurry", 50000),
    ("mhcflurry_100k", "mhcflurry", 100000),
)

DESIGNS = (
    DesignSpec(
        "A03",
        "shared_base_segment_residual + split_kd_proxy on P04 positional base",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--affinity-assay-residual-mode", "shared_base_segment_residual",
            "--kd-grouping-mode", "split_kd_proxy",
        ),
    ),
    DesignSpec(
        "A07",
        "shared_base_factorized_context_plus_segment_residual + split_kd_proxy on P04 positional base",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--affinity-assay-residual-mode", "shared_base_factorized_context_plus_segment_residual",
            "--kd-grouping-mode", "split_kd_proxy",
        ),
    ),
)


def _run_id(prefix: str, design_id: str, target_label: str) -> str:
    return f"{prefix}-{design_id.lower()}-{target_label}-20260312a"


def _common_args(*, alleles: Sequence[str], probes: Sequence[str], warm_start: str) -> List[str]:
    return [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--affinity-loss-mode", "assay_heads_only",
        "--init-checkpoint", warm_start,
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--probe-plot-frequency", "off",
    ]


def _write_manifest(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(json.dumps(list(runs), indent=2, sort_keys=True), encoding="utf-8")


def _write_variants(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Broad Target-Space Variants",
        "",
        "| design | target space | app id | run id | description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        lines.append(
            f"| `{run['design_id']}` | `{run['target_label']}` | `{run.get('app_id','')}` | `{run['run_id']}` | {run['description']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _launch(
    *,
    design: DesignSpec,
    target_label: str,
    target_encoding: str,
    max_affinity_nm: int,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
    epochs: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
    timeout_s: float,
    retries: int,
) -> Dict[str, Any]:
    run_id = _run_id(prefix, design.design_id, target_label)
    extra_args = _common_args(alleles=alleles, probes=probes, warm_start=warm_start)
    extra_args.extend(list(design.extra_args))
    extra_args.extend(
        [
            "--design-id", f"{design.design_id}_{target_label}",
            "--affinity-target-encoding", target_encoding,
            "--max-affinity-nm", str(max_affinity_nm),
        ]
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
                "description": design.description,
                "target_label": target_label,
                "affinity_target_encoding": target_encoding,
                "max_affinity_nM": max_affinity_nm,
                "requested_gpu": GPU,
                "run_id": run_id,
                "app_id": app_id,
                "batch_size": batch_size,
                "epochs": epochs,
                "extra_args": extra_args,
                "launch_log": str(log_path),
            }
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to launch {run_id}:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch broad-contract target-space bakeoff")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="broad-target-r1")
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
            "batch_size": args.batch_size,
            "warm_start": args.warm_start,
            "synthetics": False,
            "ranking": False,
            "requested_gpu": GPU,
        },
        "tested": [
            {
                "design_id": design.design_id,
                "target_label": target_label,
                "affinity_target_encoding": target_encoding,
                "max_affinity_nM": max_affinity_nm,
            }
            for design in DESIGNS
            for target_label, target_encoding, max_affinity_nm in TARGET_SPACES
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=args.out_dir,
        slug="broad-target-space-round1",
        title="Broad Target-Space Bakeoff",
        source_script="scripts/benchmark_target_space_bakeoff.py",
        agent_label=args.agent_label,
        metadata=metadata,
    )

    runs: List[Dict[str, Any]] = []
    for design in DESIGNS:
        for target_label, target_encoding, max_affinity_nm in TARGET_SPACES:
            record = _launch(
                design=design,
                target_label=target_label,
                target_encoding=target_encoding,
                max_affinity_nm=max_affinity_nm,
                alleles=alleles,
                probes=probes,
                warm_start=args.warm_start,
                epochs=args.epochs,
                batch_size=args.batch_size,
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
