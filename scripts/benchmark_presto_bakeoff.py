#!/usr/bin/env python
"""Presto vs Groove fair head-to-head sweep on Modal (bakeoff v4).

5 conditions: 3 Groove complexity ladder + 2 Presto variants.
Fixes v3 confounds: 7-allele panel (not 2), same batch size everywhere,
shorter training (10 ep), and includes the EXP-07 Groove winner (A5).

Condition matrix:
  G1:  Groove baseline — single IC50 head (~106K params)
  G2:  Groove A2 multi-head — type-routed IC50/KD/EC50 (~376K params)
  G5:  Groove A5 context conditioning — type+method embeddings (~284K params)
  PA:  Presto assay_heads_only — warm, segres, multicore, lr=2.8e-4
  PF:  Presto full loss — warm, segres, multicore, lr=2.8e-4
"""

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
    "HLA-A*02:01", "HLA-A*24:02", "HLA-A*03:01", "HLA-A*11:01",
    "HLA-A*01:01", "HLA-B*07:02", "HLA-B*44:02",
)
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI")
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")


@dataclass(frozen=True)
class BakeoffSpec:
    condition_id: str
    name: str
    modal_function: str
    extra_args: Tuple[str, ...]


def _common_data_args(*, alleles: Sequence[str], probes: Sequence[str]) -> List[str]:
    return [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--no-synthetic-negatives",
        "--probe-plot-frequency", "off",
    ]


WARM_CHECKPOINT = "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"


def _groove_args() -> Tuple[str, ...]:
    return ("--embed-dim", "128", "--hidden-dim", "128", "--lr", "1e-3")


def _presto_base(*, loss_mode: str) -> Tuple[str, ...]:
    """Build Presto extra_args tuple. Both v4 variants share everything except loss_mode."""
    return (
        "--d-model", "128",
        "--n-layers", "2",
        "--n-heads", "4",
        "--lr", "2.8e-4",
        "--peptide-pos-mode", "concat_start_end_frac",
        "--groove-pos-mode", "concat_start_end_frac",
        "--affinity-loss-mode", loss_mode,
        "--affinity-assay-mode", "legacy",
        "--affinity-assay-residual-mode", "shared_base_segment_residual",
        "--binding-core-lengths", "8,9,10,11",
        "--binding-direct-segment-mode", "off",
        "--init-checkpoint", WARM_CHECKPOINT,
    )


BAKEOFF_CONDITIONS: Tuple[BakeoffSpec, ...] = (
    # G1: Groove baseline — single IC50 head (~106K params)
    BakeoffSpec(
        condition_id="G1",
        name="Groove baseline",
        modal_function="groove_baseline_run",
        extra_args=_groove_args(),
    ),
    # G2: Groove A2 multi-head — type-routed IC50/KD/EC50 (~376K params)
    BakeoffSpec(
        condition_id="G2",
        name="Groove A2 multi-head",
        modal_function="assay_ablation_run",
        extra_args=("--variant", "a2") + _groove_args(),
    ),
    # G5: Groove A5 context conditioning — type+method embeddings (~284K params)
    BakeoffSpec(
        condition_id="G5",
        name="Groove A5 context conditioning",
        modal_function="assay_ablation_run",
        extra_args=("--variant", "a5") + _groove_args(),
    ),
    # PA: Presto assay_heads_only — v3 winner (P7)
    BakeoffSpec(
        condition_id="PA",
        name="Presto assay_heads_only warm segres multicore",
        modal_function="focused_binding_run",
        extra_args=_presto_base(loss_mode="assay_heads_only"),
    ),
    # PF: Presto full loss — v3 P6 for comparison
    BakeoffSpec(
        condition_id="PF",
        name="Presto full loss warm segres multicore",
        modal_function="focused_binding_run",
        extra_args=_presto_base(loss_mode="full"),
    ),
)


def _run_id(prefix: str, condition_id: str) -> str:
    return f"{prefix}-{condition_id.lower()}"


def _build_extra_args(
    *,
    spec: BakeoffSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
) -> List[str]:
    args = _common_data_args(alleles=alleles, probes=probes)
    args.extend(spec.extra_args)
    return args


def _launch_condition(
    *,
    spec: BakeoffSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    epochs: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
) -> Dict[str, Any]:
    run_id = _run_id(prefix, spec.condition_id)
    extra_args = _build_extra_args(spec=spec, alleles=alleles, probes=probes)
    cmd = [
        "modal",
        "run",
        "--detach",
        f"scripts/train_modal.py::{spec.modal_function}",
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
        "condition_id": spec.condition_id,
        "name": spec.name,
        "modal_function": spec.modal_function,
        "app_id": app_id,
        "command": cmd,
        "extra_args": extra_args,
        "launch_output": output.strip(),
        "launch_log": str(log_path),
        "launcher_pid": proc.pid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Presto vs Groove bakeoff experiments on Modal",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="bakeoff-v4")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--condition-ids",
        type=str,
        default="",
        help="Optional comma-separated subset of condition IDs to launch (e.g. G1,PA).",
    )
    args = parser.parse_args()

    alleles = [part.strip() for part in str(args.alleles).split(",") if part.strip()]
    probes = [part.strip() for part in str(args.probes).split(",") if part.strip()]

    selected_condition_ids = {
        part.strip().upper()
        for part in str(args.condition_ids).split(",")
        if part.strip()
    }
    selected_conditions = tuple(
        spec
        for spec in BAKEOFF_CONDITIONS
        if not selected_condition_ids or spec.condition_id.upper() in selected_condition_ids
    )
    if selected_condition_ids and not selected_conditions:
        raise ValueError(f"No conditions matched --condition-ids={args.condition_ids!r}")

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="presto-v-groove-sweep",
        title="Presto vs Groove Fair Head-to-Head (v4)",
        source_script="scripts/benchmark_presto_bakeoff.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": alleles,
                "measurement_profile": "numeric_no_qualitative",
                "qualifier_filter": "all",
                "synthetics": False,
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "groove_lr": "1e-3",
                "presto_lr": "2.8e-4",
                "warm_start_checkpoint": WARM_CHECKPOINT,
                "seed": 42,
            },
            "tested": [spec.condition_id for spec in selected_conditions],
        },
    )

    launched: List[Dict[str, Any]] = []
    for spec in selected_conditions:
        launched.append(
            _launch_condition(
                spec=spec,
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
                    "condition_ids": sorted(selected_condition_ids),
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
        "condition_ids": sorted(selected_condition_ids),
        "launches": launched,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out_dir / "conditions.md").write_text(
        "\n".join(
            [
                "# Presto vs Groove Fair Head-to-Head (v4)",
                "",
                "| ID | Name | Modal Function | App ID |",
                "|-----|------|----------------|--------|",
                *[
                    f"| {entry['condition_id']} | {entry['name']} "
                    f"| `{entry['modal_function']}` | `{entry['app_id']}` |"
                    for entry in launched
                ],
                "",
            ]
        )
    )
    print(f"\nLaunched {len(launched)} bakeoff conditions.")
    print(f"Experiment dir: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
