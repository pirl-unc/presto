#!/usr/bin/env python
"""Launch a seq-only Presto main-path replication of the current best BA config."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CODE_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_registry import default_agent_label, initialize_experiment_dir


GPU = "H100!"
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")
DEFAULT_ALLELES = ("HLA-A*02:01", "HLA-A*24:02")
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI")


def _build_extra_args(*, alleles: list[str], probes: list[str], seed: int) -> list[str]:
    return [
        "--design-id", "presto_mainpath_seqonly_c02",
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--val-fraction", "0.1",
        "--test-fraction", "0.1",
        "--seed", str(seed),
        "--d-model", "32",
        "--n-layers", "2",
        "--n-heads", "4",
        "--peptide-pos-mode", "triple",
        "--groove-pos-mode", "sequential",
        "--binding-core-lengths", "9",
        "--binding-core-refinement", "shared",
        "--lr", "1e-3",
        "--weight-decay", "0.01",
        "--affinity-loss-mode", "assay_heads_only",
        "--affinity-target-encoding", "mhcflurry",
        "--max-affinity-nm", "100000",
        "--affinity-assay-residual-mode", "pooled_single_output",
        "--kd-grouping-mode", "merged_kd",
        "--binding-kinetic-input-mode", "affinity_vec",
        "--binding-direct-segment-mode", "off",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--no-synthetic-negatives",
        "--probe-plot-frequency", "off",
    ]


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _launch(
    *,
    out_dir: Path,
    run_id: str,
    epochs: int,
    batch_size: int,
    extra_args: list[str],
    launch_timeout_s: float,
) -> dict[str, Any]:
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
    env = os.environ.copy()
    env["PRESTO_MODAL_GPU"] = GPU
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    start = time.time()
    app_id = ""
    while True:
        if time.time() - start > launch_timeout_s:
            output = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Timed out waiting for Modal app id for {run_id}:\n{output}")
        if log_path.exists():
            output = log_path.read_text(encoding="utf-8", errors="replace")
            match = APP_ID_PATTERN.search(output)
            if match:
                app_id = match.group(0)
                break
        if proc.poll() is not None and not app_id:
            output = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Detached launch exited before app id for {run_id}:\n{output}")
        time.sleep(0.5)
    return {
        "design_id": "presto_mainpath_seqonly_c02",
        "description": (
            "Main Presto seq-only replication of the EXP-21 groove c02 broad-numeric winner "
            "using only peptide/nflank/cflank/mhc_a/mhc_b as inputs"
        ),
        "required_files": [
            "summary.json",
            "val_predictions.csv",
            "test_predictions.csv",
        ],
        "run_id": run_id,
        "app_id": app_id,
        "requested_gpu": GPU,
        "epochs": epochs,
        "batch_size": batch_size,
        "extra_args": extra_args,
        "launch_log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch seq-only Presto main-path affinity replication")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="presto-mainpath-affinity-seqonly")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--launch-timeout-s", type=float, default=240.0)
    args = parser.parse_args()

    alleles = [token.strip() for token in str(args.alleles).split(",") if token.strip()]
    probes = [token.strip().upper() for token in str(args.probes).split(",") if token.strip()]
    if len(probes) < 1:
        raise SystemExit("At least one probe peptide is required")

    metadata = {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "alleles": alleles,
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "split": {
                "policy": "peptide_group",
                "seed": int(args.seed),
                "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            },
            "input_fields": ["peptide", "nflank", "cflank", "mhc_a", "mhc_b"],
            "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"],
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "synthetic_negatives": False,
            "binding_ranking_losses": False,
            "requested_gpu": GPU,
        },
        "tested": [
            {
                "design_id": "presto_mainpath_seqonly_c02",
                "description": (
                    "Presto main path with seq-only inputs, d=32/layers=2/heads=4, "
                    "mhcflurry target encoding, max_nM=100k, no assay input context"
                ),
            }
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="presto-mainpath-affinity-seqonly-replication",
        title="Presto Main-Path Seq-Only Affinity Replication",
        source_script="experiments/2026-03-16_1010_codex_presto-mainpath-affinity-seqonly-replication/code/launch.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    extra_args = _build_extra_args(alleles=alleles, probes=probes, seed=int(args.seed))
    run_id = f"{args.prefix}-e{int(args.epochs):03d}-s{int(args.seed)}"
    result = _launch(
        out_dir=out_dir,
        run_id=run_id,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        extra_args=extra_args,
        launch_timeout_s=float(args.launch_timeout_s),
    )
    manifest = [result]
    _write_manifest(out_dir / "manifest.json", manifest)
    print(json.dumps({"experiment_dir": str(out_dir), "manifest": str(out_dir / 'manifest.json'), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
