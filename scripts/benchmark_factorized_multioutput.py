#!/usr/bin/env python
"""Launch factorized multi-output A07 experiment on Modal (7-allele, d=32, 50 epochs).

Runs the A07 architecture (factorized_context + segment_residual + split_kd_proxy)
with actually wired-up factorized assay embeddings on the 7-allele / 44K-row contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


def _run_id(prefix: str) -> str:
    return f"{prefix}-factorized-multioutput"


def _build_extra_args(
    *,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
) -> List[str]:
    return [
        "--d-model", "32",
        "--n-layers", "2",
        "--n-heads", "4",
        "--lr", "1e-3",
        "--lr-schedule", "constant",
        "--weight-decay", "0.01",
        "--affinity-target-encoding", "mhcflurry",
        "--max-affinity-nm", "100000",
        "--affinity-assay-residual-mode", "shared_base_factorized_context_plus_segment_residual",
        "--kd-grouping-mode", "split_kd_proxy",
        "--affinity-loss-mode", "assay_heads_only",
        "--alleles", ",".join(alleles),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--peptide-pos-mode", "concat_start_end_frac",
        "--groove-pos-mode", "concat_start_end_frac",
        "--binding-core-lengths", "8,9,10,11",
        "--binding-core-refinement", "shared",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--probe-peptide", probes[0],
        "--probe-plot-frequency", "off",
        "--init-checkpoint", warm_start,
        "--design-id", "factorized-multioutput-a07",
    ]
    # Note: extra-probe-peptides passed via the probes[1:] if available


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch factorized multi-output A07 experiment on Modal"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="fac-multiout")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="Print command without launching")
    args = parser.parse_args()

    alleles = [x.strip() for x in str(args.alleles).split(",") if x.strip()]
    probes = [x.strip().upper() for x in str(args.probes).split(",") if x.strip()]
    run_id = _run_id(args.prefix)

    extra_args = _build_extra_args(
        alleles=alleles,
        probes=probes,
        warm_start=str(args.warm_start),
    )
    if len(probes) > 1:
        extra_args.extend(["--extra-probe-peptides", ",".join(probes[1:])])

    metadata = {
        "dataset_contract": {
            "alleles": alleles,
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "probe_peptides": probes,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "warm_start": str(args.warm_start),
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "lr": 1e-3,
            "lr_schedule": "constant",
            "weight_decay": 0.01,
            "affinity_target_encoding": "mhcflurry",
            "max_affinity_nM": 100000,
            "synthetic_negatives": False,
            "ranking_losses": False,
            "affinity_loss_mode": "assay_heads_only",
            "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
            "kd_grouping_mode": "split_kd_proxy",
        },
        "eval_metrics": [
            "spearman", "auroc",
            "{family}_spearman", "{family}_auroc", "{family}_n_samples",
            "coverage_weighted_spearman",
            "probe_head_rank_corr",
        ],
    }

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="factorized-multioutput-a07",
        title="Factorized Multi-Output A07 (d=32, 7-allele, 50ep)",
        source_script="scripts/benchmark_factorized_multioutput.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::focused_binding_run",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--run-id",
        run_id,
        "--extra-args",
        " ".join(extra_args),
    ]

    print(f"Experiment dir: {out_dir}")
    print(f"Run ID: {run_id}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("\n[DRY RUN] Command not executed.")
        (out_dir / "launch.json").write_text(
            json.dumps({"cmd": cmd, "extra_args": extra_args, "dry_run": True}, indent=2),
            encoding="utf-8",
        )
        return

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

    launch_result: Dict[str, Any] = {
        "run_id": run_id,
        "app_id": match.group(0) if match else None,
        "command": cmd,
        "extra_args": extra_args,
        "returncode": completed.returncode,
        "launch_output": output.strip(),
        "launch_log": str(log_path),
    }
    (out_dir / "launch.json").write_text(
        json.dumps(launch_result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(launch_result, indent=2))
    if completed.returncode != 0:
        print(f"\nWARNING: modal run exited with code {completed.returncode}")


if __name__ == "__main__":
    main()
