#!/usr/bin/env python
"""Launch the round-2 positional/assay bake-off on Modal."""

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
    kind: str  # "presto" or "groove"
    extra_args: Tuple[str, ...]


PRESTO_FACTORIAL: Tuple[DesignSpec, ...] = tuple(
    DesignSpec(
        design_id=f"P{idx:02d}",
        kind="presto",
        extra_args=(
            "--d-model", "128",
            "--peptide-pos-mode", peptide_pos,
            "--groove-pos-mode", groove_pos,
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-mode", "legacy",
            "--affinity-assay-residual-mode", residual_mode,
            "--binding-kinetic-input-mode", "affinity_vec",
            "--binding-direct-segment-mode", "off",
        ),
    )
    for idx, (peptide_pos, groove_pos, residual_mode) in enumerate(
        (
            ("triple", "triple", "legacy"),
            ("triple", "triple", "shared_base_segment_residual"),
            ("triple", "triple_plus_abs", "legacy"),
            ("triple", "triple_plus_abs", "shared_base_segment_residual"),
            ("triple_plus_abs", "triple", "legacy"),
            ("triple_plus_abs", "triple", "shared_base_segment_residual"),
            ("triple_plus_abs", "triple_plus_abs", "legacy"),
            ("triple_plus_abs", "triple_plus_abs", "shared_base_segment_residual"),
        )
    )
)

GROOVE_CONTROLS: Tuple[DesignSpec, ...] = (
    DesignSpec(
        design_id="G0",
        kind="groove",
        extra_args=(
            "--model-variant", "mlp",
            "--embed-dim", "64",
            "--hidden-dim", "128",
            "--binding-contrastive-weight", "0",
            "--binding-peptide-contrastive-weight", "0",
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
            "--binding-contrastive-weight", "0",
            "--binding-peptide-contrastive-weight", "0",
        ),
    ),
    DesignSpec(
        design_id="G0R",
        kind="groove",
        extra_args=(
            "--model-variant", "mlp",
            "--embed-dim", "64",
            "--hidden-dim", "128",
            "--binding-contrastive-weight", "1.0",
            "--binding-peptide-contrastive-weight", "0.5",
        ),
    ),
    DesignSpec(
        design_id="G1R",
        kind="groove",
        extra_args=(
            "--model-variant", "transformer",
            "--embed-dim", "128",
            "--hidden-dim", "256",
            "--n-layers", "2",
            "--n-heads", "4",
            "--binding-contrastive-weight", "1.0",
            "--binding-peptide-contrastive-weight", "0.5",
        ),
    ),
)

DESIGNS: Tuple[DesignSpec, ...] = PRESTO_FACTORIAL + GROOVE_CONTROLS


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260310a"


def _common_broad_args(*, alleles: Sequence[str], probes: Sequence[str]) -> List[str]:
    return [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--no-synthetic-negatives",
        "--probe-plot-frequency", "off",
    ]


def _build_extra_args(
    *,
    design: DesignSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
) -> List[str]:
    args = _common_broad_args(alleles=alleles, probes=probes)
    if design.kind == "presto":
        args.extend(
            [
                "--affinity-loss-mode", "full",
                "--init-checkpoint", warm_start,
                "--binding-contrastive-weight", "0",
                "--binding-peptide-contrastive-weight", "0",
            ]
        )
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
    target = (
        "scripts/train_modal.py::groove_baseline_run"
        if design.kind == "groove"
        else "scripts/train_modal.py::focused_binding_run"
    )
    cmd = [
        "modal",
        "run",
        "--detach",
        target,
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
        "kind": design.kind,
        "app_id": app_id,
        "command": cmd,
        "extra_args": extra_args,
        "launch_output": output.strip(),
        "launch_log": str(log_path),
        "launcher_pid": proc.pid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch round-2 directness bake-off on Modal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size-presto", type=int, default=140)
    parser.add_argument("--batch-size-groove", type=int, default=256)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="directness-r2")
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
        slug="directness-round2",
        title="Directness Bakeoff Round 2",
        source_script="scripts/benchmark_design_round2.py",
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
                "batch_size_presto": int(args.batch_size_presto),
                "batch_size_groove": int(args.batch_size_groove),
                "prefix": str(args.prefix),
            },
            "tested": [design.design_id for design in selected_designs],
        },
    )
    selected_design_ids = {
        part.strip().upper()
        for part in str(args.design_ids).split(",")
        if part.strip()
    }
    selected_designs = tuple(
        design
        for design in DESIGNS
        if not selected_design_ids or design.design_id.upper() in selected_design_ids
    )
    if selected_design_ids and not selected_designs:
        raise ValueError(f"No designs matched --design-ids={args.design_ids!r}")

    launched: List[Dict[str, Any]] = []
    for design in selected_designs:
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
                out_dir=out_dir,
            )
        )
        (out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "epochs": int(args.epochs),
                    "alleles": alleles,
                    "probes": probes,
                    "warm_start": str(args.warm_start),
                    "design_ids": sorted(selected_design_ids),
                    "launches": launched,
                },
                indent=2,
                sort_keys=True,
            )
        )

    manifest = {
        "epochs": int(args.epochs),
        "alleles": alleles,
        "probes": probes,
        "warm_start": str(args.warm_start),
        "design_ids": sorted(selected_design_ids),
        "launches": launched,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out_dir / "variants.md").write_text(
        "\n".join(
            [
                "# Round-2 Bake-Off",
                "",
                *[
                    f"- `{entry['design_id']}` `{entry['kind']}` -> "
                    f"{entry['run_id']} ({entry['app_id']})"
                    for entry in launched
                ],
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
