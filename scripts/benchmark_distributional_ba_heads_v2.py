#!/usr/bin/env python
"""Launch 28-condition v2 distributional BA head experiment on Modal.

V2 adds Gaussian and Quantile heads, sweeps bins (8,16,32,64),
and uses 50k/250k nM limits.

Usage:
    python scripts/benchmark_distributional_ba_heads_v2.py
    python scripts/benchmark_distributional_ba_heads_v2.py --cond-ids 1,5,7
    python scripts/benchmark_distributional_ba_heads_v2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from experiment_registry import default_agent_label, initialize_experiment_dir
from presto.scripts.distributional_ba.config_v2 import CONDITIONS_V2, ConditionSpec

DEFAULT_ALLELES = (
    "HLA-A*02:01", "HLA-A*24:02", "HLA-A*03:01", "HLA-A*11:01",
    "HLA-A*01:01", "HLA-B*07:02", "HLA-B*44:02",
)
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI")
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")


def _launch_condition(
    *,
    spec: ConditionSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    epochs: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
) -> Dict[str, Any]:
    run_id = f"{prefix}-c{spec.cond_id:02d}"
    extra_args_parts = [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--qualifier-filter", "all",
    ]

    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::distributional_ba_v2_run",
        "--cond-id",
        str(spec.cond_id),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--run-id",
        run_id,
        "--extra-args",
        " ".join(extra_args_parts),
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
        "cond_id": spec.cond_id,
        "label": spec.label,
        "head_type": spec.head_type,
        "assay_mode": spec.assay_mode,
        "max_nM": spec.max_nM,
        "n_bins": spec.n_bins,
        "sigma_mult": spec.sigma_mult,
        "app_id": app_id,
        "command": cmd,
        "launch_log": str(log_path),
        "launcher_pid": proc.pid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch 28-condition v2 distributional BA head experiments on Modal",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="dist-ba-v2")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--cond-ids",
        type=str,
        default="",
        help="Optional comma-separated subset of condition IDs (1-28).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alleles = [part.strip() for part in str(args.alleles).split(",") if part.strip()]
    probes = [part.strip() for part in str(args.probes).split(",") if part.strip()]

    selected_ids = set()
    if str(args.cond_ids).strip():
        for part in str(args.cond_ids).split(","):
            part = part.strip()
            if part:
                selected_ids.add(int(part))

    selected_conditions = [
        spec for spec in CONDITIONS_V2
        if not selected_ids or spec.cond_id in selected_ids
    ]
    if selected_ids and not selected_conditions:
        raise ValueError(f"No conditions matched --cond-ids={args.cond_ids!r}")

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="distributional-ba-heads-v2",
        title="Distributional BA Heads V2: Bin Sweep + Uncertainty Heads (28 conditions)",
        source_script="scripts/benchmark_distributional_ba_heads_v2.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": list(alleles),
                "measurement_profile": "numeric_no_qualitative",
                "qualifier_filter": "all",
                "split": "peptide-stratified 80/10/10",
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": "1e-3",
                "weight_decay": 0.01,
                "seed": 42,
                "encoder": "GrooveTransformerModel(embed=128, layers=2, heads=4)",
            },
            "tested": [
                {
                    "cond_id": s.cond_id,
                    "head_type": s.head_type,
                    "assay_mode": s.assay_mode,
                    "max_nM": s.max_nM,
                    "n_bins": s.n_bins,
                    "sigma_mult": s.sigma_mult,
                }
                for s in selected_conditions
            ],
        },
    )

    # --- Dry run ---
    if args.dry_run:
        manifest = {
            "conditions": len(selected_conditions),
            "condition_list": [
                {"cond_id": s.cond_id, "label": s.label}
                for s in selected_conditions
            ],
            "out_dir": str(out_dir),
        }
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    # --- Launch ---
    manifest: List[Dict[str, Any]] = []
    for spec in selected_conditions:
        result = _launch_condition(
            spec=spec,
            alleles=alleles,
            probes=probes,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            prefix=str(args.prefix),
            out_dir=out_dir,
        )
        manifest.append(result)
        print(json.dumps({
            "event": "launched",
            "cond_id": spec.cond_id,
            "label": spec.label,
            "app_id": result.get("app_id"),
            "run_id": result["run_id"],
        }, sort_keys=True), flush=True)

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nManifest: {manifest_path}")
    print(f"Conditions launched: {len(manifest)}")


if __name__ == "__main__":
    main()
