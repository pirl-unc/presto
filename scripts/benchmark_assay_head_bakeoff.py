#!/usr/bin/env python
"""Launch the broad-contract assay-head / KD-grouping bakeoff on Modal."""

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


DESIGNS: Tuple[DesignSpec, ...] = (
    DesignSpec(
        "A00",
        "P04 positional base + pooled_single_output + merged_kd",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-residual-mode", "pooled_single_output",
            "--kd-grouping-mode", "merged_kd",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        "A01",
        "P04 positional base + pooled_single_output + split_kd_proxy",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-residual-mode", "pooled_single_output",
            "--kd-grouping-mode", "split_kd_proxy",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        "A02",
        "P04 positional base + shared_base_segment_residual + merged_kd",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-residual-mode", "shared_base_segment_residual",
            "--kd-grouping-mode", "merged_kd",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        "A03",
        "P04 positional base + shared_base_segment_residual + split_kd_proxy",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-residual-mode", "shared_base_segment_residual",
            "--kd-grouping-mode", "split_kd_proxy",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        "A04",
        "P04 positional base + factorized_context_residual + merged_kd",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-residual-mode", "shared_base_factorized_context_residual",
            "--kd-grouping-mode", "merged_kd",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        "A05",
        "P04 positional base + factorized_context_residual + split_kd_proxy",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-residual-mode", "shared_base_factorized_context_residual",
            "--kd-grouping-mode", "split_kd_proxy",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        "A06",
        "P04 positional base + factorized_context_plus_segment_residual + merged_kd",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-residual-mode", "shared_base_factorized_context_plus_segment_residual",
            "--kd-grouping-mode", "merged_kd",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
    DesignSpec(
        "A07",
        "P04 positional base + factorized_context_plus_segment_residual + split_kd_proxy",
        (
            "--d-model", "128",
            "--peptide-pos-mode", "concat_start_end_frac",
            "--groove-pos-mode", "concat_start_end_frac",
            "--binding-core-lengths", "8,9,10,11",
            "--binding-core-refinement", "shared",
            "--affinity-assay-residual-mode", "shared_base_factorized_context_plus_segment_residual",
            "--kd-grouping-mode", "split_kd_proxy",
            "--affinity-target-encoding", "log10",
            "--max-affinity-nm", "50000",
        ),
    ),
)


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260311a"


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
    extra_args = _common_args(alleles=alleles, probes=probes, warm_start=warm_start)
    extra_args.extend(design.extra_args)
    extra_args.extend(["--design-id", design.design_id])
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
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    log_path.write_text(output, encoding="utf-8")
    match = APP_ID_PATTERN.search(output)
    return {
        "run_id": run_id,
        "design_id": design.design_id,
        "description": design.description,
        "app_id": match.group(0) if match else None,
        "command": cmd,
        "extra_args": extra_args,
        "returncode": completed.returncode,
        "launch_output": output.strip(),
        "launch_log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch assay-head / KD-grouping bakeoff on Modal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=140)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="assay-head-r1")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    alleles = [x.strip() for x in str(args.alleles).split(",") if x.strip()]
    probes = [x.strip().upper() for x in str(args.probes).split(",") if x.strip()]
    metadata = {
        "dataset_contract": {
            "alleles": alleles,
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "probe_peptides": probes,
            "broad_numeric_families": [
                "IC50",
                "KD",
                "KD (~IC50)",
                "KD (~EC50)",
                "EC50",
            ],
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "warm_start": str(args.warm_start),
            "synthetic_negatives": False,
            "ranking_losses": False,
            "affinity_loss_mode": "assay_heads_only",
        },
        "tested": [
            {
                "design_id": design.design_id,
                "description": design.description,
                "extra_args": list(design.extra_args),
            }
            for design in DESIGNS
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="assay-head-round1",
        title="Assay Head / KD Grouping Bakeoff Round 1",
        source_script="scripts/benchmark_assay_head_bakeoff.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    launches: List[Dict[str, Any]] = []
    for design in DESIGNS:
        launches.append(
            _launch_design(
                design=design,
                alleles=alleles,
                probes=probes,
                warm_start=str(args.warm_start),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                prefix=str(args.prefix),
                out_dir=out_dir,
            )
        )
        time.sleep(0.2)

    manifest = {
        "experiment_dir": str(out_dir),
        "alleles": alleles,
        "probes": probes,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "warm_start": str(args.warm_start),
        "designs": launches,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Assay Head / KD Grouping Round 1", ""]
    for row in launches:
        lines.extend(
            [
                f"## {row['design_id']}",
                "",
                f"- App: `{row.get('app_id')}`",
                f"- Run: `{row['run_id']}`",
                f"- Description: {row['description']}",
                f"- Launch log: `{row['launch_log']}`",
                "",
            ]
        )
    (out_dir / "variants.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "assay_head_bakeoff_launched", "out_dir": str(out_dir), "n_designs": len(launches)}, sort_keys=True))


if __name__ == "__main__":
    main()
