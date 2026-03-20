#!/usr/bin/env python
"""Rerun the current EXP-21 winner through the canonical experiment-local launcher path."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CODE_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_registry import default_agent_label, initialize_experiment_dir


APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")
CHECKPOINT_VOLUME = "presto-checkpoints"
ENCODER_BACKBONE = "groove"
COND_ID = 2
CONTENT_CONDITIONED = False
EPOCHS = 50
BATCH_SIZE = 256
SEED = 43
GPU = "H100!"
ALLELES = ("HLA-A*02:01", "HLA-A*24:02")
RUN_PREFIX = "dist-ba-v6-rerun"


def _write_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remote_artifacts(run_id: str) -> list[str]:
    result = subprocess.run(
        ["modal", "volume", "ls", CHECKPOINT_VOLUME, run_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _spawn_background_launch(*, cmd: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
            close_fds=True,
        )
    return proc.pid


def _launch_run(*, out_dir: Path, prefix: str) -> dict[str, Any]:
    cc_tag = "cc1" if CONTENT_CONDITIONED else "cc0"
    run_id = f"{prefix}-groove_c02-c{COND_ID:02d}-{cc_tag}-e{EPOCHS:03d}-s{SEED}"
    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::distributional_ba_v6_run",
        "--cond-id",
        str(COND_ID),
        "--encoder-backbone",
        ENCODER_BACKBONE,
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--run-id",
        run_id,
        "--extra-args",
        (
            "--measurement-profile numeric_no_qualitative "
            "--alleles HLA-A*02:01,HLA-A*24:02 "
            f"--seed {SEED}"
        ),
    ]
    if CONTENT_CONDITIONED:
        cmd.append("--content-conditioned")

    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    remote_artifacts = _remote_artifacts(run_id)
    if remote_artifacts:
        return {
            "run_id": run_id,
            "cond_id": COND_ID,
            "encoder_backbone": ENCODER_BACKBONE,
            "content_conditioned": CONTENT_CONDITIONED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "required_files": ["summary.json", "probes.jsonl", "metrics.jsonl", "step_log.jsonl"],
            "command": cmd,
            "launch_log": str(log_path),
            "launcher_pid": None,
            "launch_status": "remote_artifacts_already_present",
            "remote_artifacts": remote_artifacts,
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    launcher_pid = _spawn_background_launch(cmd=cmd, log_path=log_path)
    return {
        "run_id": run_id,
        "cond_id": COND_ID,
        "encoder_backbone": ENCODER_BACKBONE,
        "content_conditioned": CONTENT_CONDITIONED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "required_files": ["summary.json", "probes.jsonl", "metrics.jsonl", "step_log.jsonl"],
        "command": cmd,
        "launch_log": str(log_path),
        "launcher_pid": launcher_pid,
        "launch_status": "spawned_background",
        "remote_artifacts": [],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun the EXP-21 groove c02 50-epoch winner.")
    parser.add_argument("--prefix", type=str, default=RUN_PREFIX)
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "panel": list(ALLELES),
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "split": "peptide_group_80_10_10_seed42",
            "assay_families": ["IC50", "KD", "KD (~IC50)", "KD (~EC50)", "EC50"],
            "source_baseline": "experiments/2026-03-15_1226_codex_exp21-seed-epoch-confirmation",
        },
        "training": {
            "config_version": "v6",
            "encoder_backbone": ENCODER_BACKBONE,
            "cond_id": COND_ID,
            "content_conditioned": CONTENT_CONDITIONED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": "1e-3",
            "weight_decay": 0.01,
            "seed": SEED,
            "warm_start": False,
            "gpu": os.environ.get("PRESTO_MODAL_GPU", GPU),
        },
        "tested": [
            {
                "model_key": "groove_c02",
                "encoder_backbone": ENCODER_BACKBONE,
                "cond_id": COND_ID,
                "content_conditioned": CONTENT_CONDITIONED,
                "epochs": EPOCHS,
                "seed": SEED,
                "purpose": "Exact structure-check rerun of the current robust baseline",
            }
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="exp21-winner-rerun",
        title="EXP-21 Winner Rerun via Canonical Launcher Layout",
        source_script="experiments/2026-03-16_1333_codex_exp21-winner-rerun/code/launch.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "tested": metadata["tested"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    manifest = [_launch_run(out_dir=out_dir, prefix=str(args.prefix))]
    _write_manifest(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "launched", "experiment_dir": str(out_dir), **manifest[0]}, sort_keys=True))


if __name__ == "__main__":
    main()
