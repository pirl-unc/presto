#!/usr/bin/env python
"""Launch a fresh 36-run seed/epoch confirmation sweep for the EXP-20 winner."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment_registry import default_agent_label, initialize_experiment_dir


APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")
MODELS = (
    {
        "model_key": "groove_c01",
        "encoder_backbone": "groove",
        "cond_id": 1,
        "content_conditioned": False,
    },
    {
        "model_key": "groove_c02",
        "encoder_backbone": "groove",
        "cond_id": 2,
        "content_conditioned": False,
    },
    {
        "model_key": "historical_c02",
        "encoder_backbone": "historical_ablation",
        "cond_id": 2,
        "content_conditioned": False,
    },
)
SEEDS = (42, 43, 44, 45)
EPOCHS = (50, 100, 200)
CHECKPOINT_VOLUME = "presto-checkpoints"


def _write_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _remote_artifacts(run_id: str) -> list[str]:
    result = subprocess.run(
        [
            "modal",
            "volume",
            "ls",
            CHECKPOINT_VOLUME,
            run_id,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _extract_app_id(output: str) -> str | None:
    match = APP_ID_PATTERN.search(output)
    if match is None:
        return None
    return match.group(0)


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _spawn_background_launch(*, cmd: list[str], log_path: Path) -> int:
    with log_path.open("w") as log_file:
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
    model_key: str,
    encoder_backbone: str,
    cond_id: int,
    content_conditioned: bool,
    epochs: int,
    seed: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
) -> dict[str, Any]:
    cc_tag = "cc1" if content_conditioned else "cc0"
    run_id = f"{prefix}-{model_key}-c{cond_id:02d}-{cc_tag}-e{epochs:03d}-s{seed}"
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
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--run-id",
        run_id,
        "--extra-args",
        (
            "--measurement-profile numeric_no_qualitative "
            "--alleles HLA-A*02:01,HLA-A*24:02 "
            f"--seed {seed}"
        ),
    ]
    if content_conditioned:
        cmd.append("--content-conditioned")

    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing_artifacts = _remote_artifacts(run_id)
    if existing_artifacts:
        return {
            "run_id": run_id,
            "model_key": model_key,
            "cond_id": cond_id,
            "content_conditioned": content_conditioned,
            "encoder_backbone": encoder_backbone,
            "epochs": epochs,
            "seed": seed,
            "app_id": None,
            "command": cmd,
            "launch_log": str(log_path),
            "launcher_pid": None,
            "launch_status": "remote_artifacts_already_present",
            "remote_artifacts": existing_artifacts,
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    launcher_pid = _spawn_background_launch(cmd=cmd, log_path=log_path)

    return {
        "run_id": run_id,
        "model_key": model_key,
        "cond_id": cond_id,
        "content_conditioned": content_conditioned,
        "encoder_backbone": encoder_backbone,
        "epochs": epochs,
        "seed": seed,
        "app_id": None,
        "command": cmd,
        "launch_log": str(log_path),
        "launcher_pid": launcher_pid,
        "launch_status": "spawned_background",
        "remote_artifacts": [],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the 36-run seed/epoch confirmation sweep for EXP-20 winners."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prefix", type=str, default="dist-ba-v6-confirm")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tested: list[dict[str, Any]] = []
    for model in MODELS:
        for epochs in EPOCHS:
            for seed in SEEDS:
                tested.append(
                    {
                        **model,
                        "epochs": epochs,
                        "seed": seed,
                    }
                )

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="exp21-seed-epoch-confirmation",
        title="EXP-21 Seed + Epoch Confirmation Sweep",
        source_script="scripts/benchmark_distributional_ba_v6_seed_epoch_confirmation.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": ["HLA-A*02:01", "HLA-A*24:02"],
                "measurement_profile": "numeric_no_qualitative",
                "qualifier_filter": "all",
                "split": "peptide_group_80_10_10_seed42",
                "source": "data/merged_deduped.tsv",
                "assay_families": ["IC50", "KD", "KD (~IC50)", "KD (~EC50)", "EC50"],
            },
            "training": {
                "config_version": "v6",
                "batch_size": int(args.batch_size),
                "epochs": list(EPOCHS),
                "lr": "1e-3",
                "weight_decay": 0.01,
                "seeds": list(SEEDS),
                "warm_start": False,
                "gpu": os.environ.get("PRESTO_MODAL_GPU", "H100!"),
            },
            "tested": tested,
        },
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "conditions": len(tested),
                    "out_dir": str(out_dir),
                    "tested": tested[:6],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    manifest: list[dict[str, Any]] = []
    manifest_path = out_dir / "manifest.json"
    for model in MODELS:
        for epochs in EPOCHS:
            for seed in SEEDS:
                result = _launch_condition(
                    model_key=str(model["model_key"]),
                    encoder_backbone=str(model["encoder_backbone"]),
                    cond_id=int(model["cond_id"]),
                    content_conditioned=bool(model["content_conditioned"]),
                    epochs=epochs,
                    seed=seed,
                    batch_size=int(args.batch_size),
                    prefix=str(args.prefix),
                    out_dir=out_dir,
                )
                manifest.append(result)
                _write_manifest(manifest_path, manifest)
                print(
                    json.dumps(
                        {
                            "event": "spawned",
                            "run_id": result["run_id"],
                            "app_id": result.get("app_id"),
                            "model_key": result["model_key"],
                            "epochs": result["epochs"],
                            "seed": result["seed"],
                            "launch_status": result.get("launch_status"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    print(f"\nManifest: {manifest_path}")
    print(f"Conditions launched: {len(manifest)}")


if __name__ == "__main__":
    main()
