#!/usr/bin/env python
"""Launch fixed-epoch runtime variants for the focused legacy_m1 benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    "--measurement-profile", "direct_affinity_only",
    "--measurement-type-filter", "ic50",
    "--qualifier-filter", "exact",
    "--groove-pos-mode", "triple",
    "--binding-core-lengths", "8,9,10,11",
    "--binding-core-refinement", "shared",
    "--affinity-assay-mode", "legacy",
    "--binding-contrastive-weight", "0",
    "--binding-peptide-contrastive-weight", "0",
    "--init-checkpoint", "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt",
]


@dataclass(frozen=True)
class Variant:
    variant_id: str
    description: str
    extra_args: List[str]


VARIANTS = [
    Variant("V00", "baseline workers=0 pin=0", ["--no-persistent-workers"]),
    Variant("V01", "nw=2 pin persist p2", ["--num-workers", "2", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2"]),
    Variant("V02", "nw=4 pin persist p2", ["--num-workers", "4", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2"]),
    Variant("V03", "nw=8 pin persist p2", ["--num-workers", "8", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2"]),
    Variant("V04", "nw=4 pin persist p4", ["--num-workers", "4", "--pin-memory", "--persistent-workers", "--prefetch-factor", "4"]),
    Variant("V05", "nw=8 pin persist p4", ["--num-workers", "8", "--pin-memory", "--persistent-workers", "--prefetch-factor", "4"]),
    Variant("V06", "nw=4 pin persist p2 tf32 high", ["--num-workers", "4", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high"]),
    Variant("V07", "nw=8 pin persist p2 tf32 high", ["--num-workers", "8", "--pin-memory", "--persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high"]),
    Variant("V08", "nw=4 pin persist p4 tf32 high", ["--num-workers", "4", "--pin-memory", "--persistent-workers", "--prefetch-factor", "4", "--allow-tf32", "--matmul-precision", "high"]),
    Variant("V09", "nw=8 pin persist p4 tf32 high", ["--num-workers", "8", "--pin-memory", "--persistent-workers", "--prefetch-factor", "4", "--allow-tf32", "--matmul-precision", "high"]),
    Variant("V10", "nw=4 pin no-persist p2 tf32 high", ["--num-workers", "4", "--pin-memory", "--no-persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high"]),
    Variant("V11", "nw=8 pin no-persist p2 tf32 high", ["--num-workers", "8", "--pin-memory", "--no-persistent-workers", "--prefetch-factor", "2", "--allow-tf32", "--matmul-precision", "high"]),
]
def _build_extra_args(variant: Variant) -> str:
    args = list(BASE_EXTRA_ARGS)
    args.extend(["--design-id", variant.variant_id])
    args.extend(variant.extra_args)
    return " ".join(args)


def _write_manifest(output_dir: Path, rows: List[Dict[str, str]]) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Runtime Variants",
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
) -> Dict[str, str]:
    run_id = f"runtime-m1-{variant.variant_id.lower()}-{stamp}"
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch runtime-only legacy_m1 Modal benchmark variants")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=140)
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    args = parser.parse_args()

    output_dir = initialize_experiment_dir(
        out_dir=str(args.output_dir),
        slug="runtime-m1-bench",
        title="Runtime M1 Benchmark",
        source_script="scripts/benchmark_runtime_variants.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": list(BASE_ALLELES),
                "measurement_profile": "direct_affinity_only",
                "measurement_type_filter": "ic50",
                "qualifier_filter": "exact",
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
            },
            "tested": [variant.variant_id for variant in VARIANTS],
        },
    )
    stamp = str(args.stamp or datetime.now().strftime("%Y%m%d%H%M%S"))

    rows: List[Dict[str, str]] = []
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


if __name__ == "__main__":
    main()
