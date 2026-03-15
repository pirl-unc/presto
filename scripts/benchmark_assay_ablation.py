#!/usr/bin/env python
"""Launch assay head ablation experiments (A1-A8) on Modal."""

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
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    design_id: str
    extra_args: Tuple[str, ...]


ABLATION_DESIGNS: Tuple[AblationSpec, ...] = tuple(
    AblationSpec(
        variant=f"a{i}",
        design_id=f"A{i}",
        extra_args=(
            "--embed-dim", "128",
            "--hidden-dim", "128",
        ),
    )
    for i in range(1, 9)
)


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260311a"


def _common_args(*, alleles: Sequence[str], probes: Sequence[str]) -> List[str]:
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
    design: AblationSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
) -> List[str]:
    args = _common_args(alleles=alleles, probes=probes)
    args.extend(["--variant", design.variant])
    args.extend(design.extra_args)
    return args


def _launch_design(
    *,
    design: AblationSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
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
    )
    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::assay_ablation_run",
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
        "variant": design.variant,
        "app_id": app_id,
        "command": cmd,
        "extra_args": extra_args,
        "launch_output": output.strip(),
        "launch_log": str(log_path),
        "launcher_pid": proc.pid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch assay ablation experiments on Modal")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="assay-ablate")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--design-ids",
        type=str,
        default="",
        help="Optional comma-separated subset of design IDs to launch (e.g. A1,A3).",
    )
    args = parser.parse_args()

    alleles = [part.strip() for part in str(args.alleles).split(",") if part.strip()]
    probes = [part.strip() for part in str(args.probes).split(",") if part.strip()]
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="assay-ablation",
        title="Assay Head Ablation",
        source_script="scripts/benchmark_assay_ablation.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": alleles,
                "measurement_profile": "numeric_no_qualitative",
                "qualifier_filter": "all",
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
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
        for design in ABLATION_DESIGNS
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
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                prefix=str(args.prefix),
                out_dir=out_dir,
            )
        )
        # Write manifest incrementally so partial launches are recoverable
        (out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "alleles": alleles,
                    "probes": probes,
                    "design_ids": sorted(selected_design_ids),
                    "launches": launched,
                },
                indent=2,
                sort_keys=True,
            )
        )

    manifest = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "alleles": alleles,
        "probes": probes,
        "design_ids": sorted(selected_design_ids),
        "launches": launched,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out_dir / "variants.md").write_text(
        "\n".join(
            [
                "# Assay Head Ablation",
                "",
                *[
                    f"- `{entry['design_id']}` ({entry['variant']}) -> "
                    f"{entry['run_id']} ({entry['app_id']})"
                    for entry in launched
                ],
                "",
            ]
        )
    )
    print(f"\nLaunched {len(launched)} ablation variants.")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
