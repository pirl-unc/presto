#!/usr/bin/env python
"""Launch broad-contract positional/encoding round-3 sweeps on Modal."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    description: str
    extra_args: Tuple[str, ...]


POSITIONAL_SWEEP: Tuple[DesignSpec, ...] = tuple(
    DesignSpec(
        design_id=f"Q{idx:02d}",
        description=f"pep={pep_pos}, groove={groove_pos}, residual=shared_base_segment_residual",
        extra_args=(
            "--d-model", "128",
            "--peptide-pos-mode", pep_pos,
            "--groove-pos-mode", groove_pos,
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--affinity-assay-residual-mode", "shared_base_segment_residual",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "off",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    )
    for idx, (pep_pos, groove_pos) in enumerate(
        (
            ("triple", "triple"),
            ("triple", "abs_only"),
            ("triple", "triple_plus_abs"),
            ("abs_only", "triple"),
            ("abs_only", "abs_only"),
            ("abs_only", "triple_plus_abs"),
            ("triple_plus_abs", "triple"),
            ("triple_plus_abs", "abs_only"),
            ("triple_plus_abs", "triple_plus_abs"),
        )
    )
)

ENCODING_SWEEP: Tuple[DesignSpec, ...] = (
    DesignSpec(
        design_id="E00",
        description="P03 positional config, log10 target, 50k cap",
        extra_args=(
            "--d-model", "128",
            "--peptide-pos-mode", "triple",
            "--groove-pos-mode", "triple_plus_abs",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--affinity-assay-residual-mode", "shared_base_segment_residual",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "off",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        design_id="E01",
        description="P03 positional config, log10 target, 100k cap",
        extra_args=(
            "--d-model", "128",
            "--peptide-pos-mode", "triple",
            "--groove-pos-mode", "triple_plus_abs",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--affinity-assay-residual-mode", "shared_base_segment_residual",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "off",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "100000",
        ),
    ),
    DesignSpec(
        design_id="E02",
        description="P03 positional config, mhcflurry target, 50k cap",
        extra_args=(
            "--d-model", "128",
            "--peptide-pos-mode", "triple",
            "--groove-pos-mode", "triple_plus_abs",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--affinity-assay-residual-mode", "shared_base_segment_residual",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "off",
            "--affinity-target-encoding", "mhcflurry",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        design_id="E03",
        description="P03 positional config, mhcflurry target, 100k cap",
        extra_args=(
            "--d-model", "128",
            "--peptide-pos-mode", "triple",
            "--groove-pos-mode", "triple_plus_abs",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--affinity-assay-residual-mode", "shared_base_segment_residual",
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "off",
            "--affinity-target-encoding", "mhcflurry",
            "--max-affinity-nm", "100000",
        ),
    ),
)

DESIGNS: Tuple[DesignSpec, ...] = POSITIONAL_SWEEP + ENCODING_SWEEP


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260311a"


def _common_args(*, alleles: Sequence[str], probes: Sequence[str], warm_start: str) -> List[str]:
    return [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--affinity-loss-mode", "full",
        "--init-checkpoint", warm_start,
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
    args = _common_args(alleles=alleles, probes=probes, warm_start=warm_start)
    args.extend(design.extra_args)
    args.extend(["--design-id", design.design_id])
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
    out_dir: Path,
) -> Dict[str, Any]:
    run_id = _run_id(prefix, design.design_id)
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
        "--run-id",
        run_id,
        "--extra-args",
        " ".join(extra_args),
    ]
    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )
    app_id: str | None = None
    output = ""
    for _ in range(20):
        try:
            output = log_path.read_text()
        except FileNotFoundError:
            output = ""
        match = APP_ID_PATTERN.search(output)
        if match is not None:
            app_id = match.group(0)
            break
        time.sleep(0.5)
    return {
        "run_id": run_id,
        "design_id": design.design_id,
        "description": design.description,
        "app_id": app_id,
        "command": cmd,
        "extra_args": extra_args,
        "launch_output": output.strip(),
        "launch_log": str(log_path),
        "launcher_pid": proc.pid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch broad-contract round-3 Presto sweeps on Modal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=140)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="directness-r3")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--design-ids",
        type=str,
        default="",
        help="Optional comma-separated subset of design IDs to launch.",
    )
    args = parser.parse_args()

    alleles = [part.strip() for part in str(args.alleles).split(",") if part.strip()]
    probes = [part.strip() for part in str(args.probes).split(",") if part.strip()]
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="directness-round3",
        title="Directness Bakeoff Round 3",
        source_script="scripts/benchmark_design_round3.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": alleles,
                "measurement_profile": "numeric_no_qualitative",
                "qualifier_filter": "all",
                "warm_start": str(args.warm_start),
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "prefix": str(args.prefix),
            },
            "tested": [design.design_id for design in designs],
        },
    )
    selected_design_ids = {
        part.strip().upper()
        for part in str(args.design_ids).split(",")
        if part.strip()
    }
    designs = tuple(
        design for design in DESIGNS
        if not selected_design_ids or design.design_id.upper() in selected_design_ids
    )
    if not designs:
        raise SystemExit("No design IDs selected")

    manifest: List[Dict[str, Any]] = []
    for design in designs:
        entry = _launch_design(
            design=design,
            alleles=alleles,
            probes=probes,
            warm_start=str(args.warm_start),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            prefix=str(args.prefix),
            out_dir=out_dir,
        )
        manifest.append(entry)
        print(
            json.dumps(
                {
                    "design_id": design.design_id,
                    "run_id": entry["run_id"],
                    "app_id": entry["app_id"],
                    "description": design.description,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Round-3 Launches", ""]
    for entry in manifest:
        lines.append(f"- `{entry['design_id']}` -> `{entry['run_id']}` ({entry['app_id'] or 'pending'})")
        lines.append(f"  - {entry['description']}")
    (out_dir / "variants.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
