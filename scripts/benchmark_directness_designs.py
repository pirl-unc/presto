#!/usr/bin/env python
"""Launch a directness-focused architecture bake-off on Modal."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    kind: str  # "presto" or "groove"
    extra_args: Tuple[str, ...]


DESIGNS: Tuple[DesignSpec, ...] = (
    DesignSpec(
        design_id="G0",
        kind="groove",
        extra_args=(
            "--model-variant", "mlp",
            "--embed-dim", "64",
            "--hidden-dim", "128",
        ),
    ),
    DesignSpec(
        design_id="G1",
        kind="groove",
        extra_args=(
            "--model-variant", "transformer",
            "--embed-dim", "128",
            "--hidden-dim", "256",
            "--n-layers", "2",
            "--n-heads", "4",
        ),
    ),
    DesignSpec(
        design_id="P0",
        kind="presto",
        extra_args=(
            "--d-model", "256",
            "--groove-pos-mode", "triple",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "off",
        ),
    ),
    DesignSpec(
        design_id="P1",
        kind="presto",
        extra_args=(
            "--d-model", "256",
            "--groove-pos-mode", "triple",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "affinity_residual",
        ),
    ),
    DesignSpec(
        design_id="P2",
        kind="presto",
        extra_args=(
            "--d-model", "256",
            "--groove-pos-mode", "triple",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "affinity_stability_residual",
        ),
    ),
    DesignSpec(
        design_id="P3",
        kind="presto",
        extra_args=(
            "--d-model", "256",
            "--groove-pos-mode", "triple",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "gated_affinity",
        ),
    ),
    DesignSpec(
        design_id="P4",
        kind="presto",
        extra_args=(
            "--d-model", "128",
            "--groove-pos-mode", "triple",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "affinity_residual",
        ),
    ),
    DesignSpec(
        design_id="P5",
        kind="presto",
        extra_args=(
            "--d-model", "128",
            "--groove-pos-mode", "triple",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "off",
        ),
    ),
)


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260310a"


def _common_args(*, alleles: Sequence[str], probes: Sequence[str]) -> List[str]:
    return [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "direct_affinity_only",
        "--measurement-type-filter", "ic50",
        "--qualifier-filter", "exact",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--probe-plot-frequency", "off",
    ]


def _build_extra_args(
    *,
    design: DesignSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
) -> List[str]:
    args = _common_args(alleles=alleles, probes=probes)
    if design.kind == "presto":
        args.extend([
            "--affinity-loss-mode", "ic50_only",
            "--init-checkpoint", warm_start,
        ])
    args.extend(design.extra_args)
    return args


def _launch_design(
    *,
    design: DesignSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
    epochs: int,
    batch_size: int,
    prefix: str,
) -> Dict[str, Any]:
    run_id = _run_id(prefix, design.design_id)
    extra_args = _build_extra_args(
        design=design,
        alleles=alleles,
        probes=probes,
        warm_start=warm_start,
    )
    target = (
        "scripts/train_modal.py::groove_baseline_run"
        if design.kind == "groove"
        else "scripts/train_modal.py::focused_binding_run"
    )
    cmd = [
        "modal", "run", "--detach", target,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--run-id", run_id,
        "--extra-args", " ".join(extra_args),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    match = APP_ID_PATTERN.search(output)
    if match is None:
        raise RuntimeError(f"Detached launch for {run_id} did not emit app id:\n{output}")
    return {
        "run_id": run_id,
        "design_id": design.design_id,
        "kind": design.kind,
        "app_id": match.group(0),
        "command": cmd,
        "extra_args": extra_args,
        "launch_output": output.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch directness bake-off designs on Modal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size-presto", type=int, default=140)
    parser.add_argument("--batch-size-groove", type=int, default=256)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="directness-c0")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    alleles = [part.strip() for part in str(args.alleles).split(",") if part.strip()]
    probes = [part.strip() for part in str(args.probes).split(",") if part.strip()]
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="directness-designs",
        title="Directness Design Bakeoff",
        source_script="scripts/benchmark_directness_designs.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": alleles,
                "measurement_profile": "direct_affinity_only",
                "measurement_type_filter": "ic50",
                "qualifier_filter": "exact",
                "warm_start": str(args.warm_start),
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size_presto": int(args.batch_size_presto),
                "batch_size_groove": int(args.batch_size_groove),
                "prefix": str(args.prefix),
            },
            "tested": [design.design_id for design in DESIGNS],
        },
    )

    launched: List[Dict[str, Any]] = []
    for design in DESIGNS:
        launched.append(
            _launch_design(
                design=design,
                alleles=alleles,
                probes=probes,
                warm_start=str(args.warm_start),
                epochs=int(args.epochs),
                batch_size=(
                    int(args.batch_size_groove)
                    if design.kind == "groove"
                    else int(args.batch_size_presto)
                ),
                prefix=str(args.prefix),
            )
        )

    manifest = {
        "epochs": int(args.epochs),
        "alleles": alleles,
        "probes": probes,
        "warm_start": str(args.warm_start),
        "runs": launched,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    lines = ["| design | kind | run_id | app_id |", "| --- | --- | --- | --- |"]
    for row in launched:
        lines.append(
            f"| {row['design_id']} | {row['kind']} | {row['run_id']} | {row['app_id']} |"
        )
    (out_dir / "manifest.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
