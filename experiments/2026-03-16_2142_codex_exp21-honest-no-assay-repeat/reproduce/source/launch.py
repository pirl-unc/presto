#!/usr/bin/env python
"""Run an honest no-assay-input repeat of the legacy EXP-21 benchmark family."""

from __future__ import annotations

import argparse
import json
import os
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


CHECKPOINT_VOLUME = "presto-checkpoints"
GPU = "H100!"
ALLELES = ("HLA-A*02:01", "HLA-A*24:02")
EPOCHS = 50
BATCH_SIZE = 256
SEED = 43
RUN_PREFIX = "dist-ba-v6-honest"
MODELS = (
    {
        "model_key": "groove_c02",
        "encoder_backbone": "groove",
        "cond_id": 2,
        "description": "Old EXP-21 best single run repeated with assay inputs disabled.",
    },
    {
        "model_key": "groove_c01",
        "encoder_backbone": "groove",
        "cond_id": 1,
        "description": "Closest groove competitor repeated with assay inputs disabled.",
    },
    {
        "model_key": "historical_c02",
        "encoder_backbone": "historical_ablation",
        "cond_id": 2,
        "description": "Historical positive control repeated with assay inputs disabled.",
    },
)


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


def _launch_condition(
    *,
    out_dir: Path,
    prefix: str,
    model_key: str,
    encoder_backbone: str,
    cond_id: int,
    description: str,
) -> dict[str, Any]:
    run_id = f"{prefix}-{model_key}-c{cond_id:02d}-ai0-e{EPOCHS:03d}-s{SEED}"
    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::distributional_ba_v6_run",
        "--cond-id",
        str(cond_id),
        "--encoder-backbone",
        encoder_backbone,
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
            f"--seed {SEED} "
            "--assay-input-mode none"
        ),
    ]
    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    remote_artifacts = _remote_artifacts(run_id)
    if remote_artifacts:
        return {
            "run_id": run_id,
            "model_key": model_key,
            "encoder_backbone": encoder_backbone,
            "cond_id": cond_id,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "assay_input_mode": "none",
            "description": description,
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
        "model_key": model_key,
        "encoder_backbone": encoder_backbone,
        "cond_id": cond_id,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "assay_input_mode": "none",
        "description": description,
        "required_files": ["summary.json", "probes.jsonl", "metrics.jsonl", "step_log.jsonl"],
        "command": cmd,
        "launch_log": str(log_path),
        "launcher_pid": launcher_pid,
        "launch_status": "spawned_background",
        "remote_artifacts": [],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat the EXP-21 benchmark family with assay inputs disabled."
    )
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
            "assay_families": ["IC50", "KD", "KD (~IC50)", "KD (~EC50)", "EC50"],
            "assay_selector_inputs_forbidden": True,
            "split_seed": SEED,
            "legacy_benchmark_contract": True,
        },
        "training": {
            "config_version": "v6",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "seed": SEED,
            "content_conditioned": False,
            "assay_input_mode": "none",
            "requested_gpu": os.environ.get("PRESTO_MODAL_GPU", GPU),
        },
        "tested": [
            {
                "model_key": item["model_key"],
                "encoder_backbone": item["encoder_backbone"],
                "cond_id": item["cond_id"],
                "epochs": EPOCHS,
                "seed": SEED,
                "assay_input_mode": "none",
                "description": item["description"],
            }
            for item in MODELS
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="exp21-honest-no-assay-repeat",
        title="EXP-21 Honest Repeat Without Assay Inputs",
        source_script="experiments/2026-03-16_2142_codex_exp21-honest-no-assay-repeat/code/launch.py",
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

    manifest = [
        _launch_condition(
            out_dir=out_dir,
            prefix=str(args.prefix),
            model_key=item["model_key"],
            encoder_backbone=item["encoder_backbone"],
            cond_id=int(item["cond_id"]),
            description=str(item["description"]),
        )
        for item in MODELS
    ]
    _write_manifest(out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "launched",
                "experiment_dir": str(out_dir),
                "n_runs": len(manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
