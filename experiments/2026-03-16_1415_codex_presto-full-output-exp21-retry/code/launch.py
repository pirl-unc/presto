#!/usr/bin/env python
"""Launch a careful full-Presto retry on the exact EXP-21 data contract."""

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


GPU = "H100!"
CHECKPOINT_VOLUME = "presto-checkpoints"
DEFAULT_ALLELES = ("HLA-A*02:01", "HLA-A*24:02")
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI")
DEFAULT_PREFIX = "presto-exp21-full-20260316a"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 43
DEFAULT_SPLIT_SEED = 42


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remote_artifacts(run_id: str) -> list[str]:
    result = subprocess.run(
        ["modal", "volume", "ls", CHECKPOINT_VOLUME, run_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _spawn_background_launch(*, cmd: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["PRESTO_MODAL_GPU"] = env.get("PRESTO_MODAL_GPU", GPU)
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    return int(proc.pid)


def _build_extra_args(
    *,
    alleles: list[str],
    probes: list[str],
    design_id: str,
    affinity_assay_residual_mode: str,
    affinity_target_encoding: str,
    max_affinity_nm: int,
    seed: int,
    split_seed: int,
) -> list[str]:
    return [
        "--design-id", design_id,
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--val-fraction", "0.1",
        "--test-fraction", "0.1",
        "--seed", str(seed),
        "--split-seed", str(split_seed),
        "--d-model", "32",
        "--n-layers", "2",
        "--n-heads", "4",
        "--peptide-pos-mode", "concat_start_end_frac",
        "--groove-pos-mode", "concat_start_end_frac",
        "--binding-core-lengths", "8,9,10,11",
        "--binding-core-refinement", "shared",
        "--lr", "1e-3",
        "--weight-decay", "0.01",
        "--affinity-loss-mode", "full",
        "--affinity-target-encoding", affinity_target_encoding,
        "--max-affinity-nm", str(max_affinity_nm),
        "--affinity-assay-residual-mode", affinity_assay_residual_mode,
        "--kd-grouping-mode", "split_kd_proxy",
        "--binding-kinetic-input-mode", "affinity_vec",
        "--binding-direct-segment-mode", "off",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--probe-plot-frequency", "off",
    ]


def _condition_rows(
    *,
    alleles: list[str],
    probes: list[str],
    seed: int,
    split_seed: int,
) -> list[dict[str, Any]]:
    return [
        {
            "condition_key": "PF03_log10_100k_full",
            "design_id": "presto_pf03_log10_100k_full",
            "description": (
                "Presto full loss, A03-style shared_base_segment_residual, split KD proxy, "
                "log10 100k target space"
            ),
            "affinity_assay_residual_mode": "shared_base_segment_residual",
            "affinity_target_encoding": "log10",
            "max_affinity_nM": 100000,
            "extra_args": _build_extra_args(
                alleles=alleles,
                probes=probes,
                design_id="presto_pf03_log10_100k_full",
                affinity_assay_residual_mode="shared_base_segment_residual",
                affinity_target_encoding="log10",
                max_affinity_nm=100000,
                seed=seed,
                split_seed=split_seed,
            ),
        },
        {
            "condition_key": "PF07_mhcflurry_100k_full",
            "design_id": "presto_pf07_mhcflurry_100k_full",
            "description": (
                "Presto full loss, A07-style factorized_context_plus_segment residual, split KD proxy, "
                "mhcflurry 100k target space"
            ),
            "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
            "affinity_target_encoding": "mhcflurry",
            "max_affinity_nM": 100000,
            "extra_args": _build_extra_args(
                alleles=alleles,
                probes=probes,
                design_id="presto_pf07_mhcflurry_100k_full",
                affinity_assay_residual_mode="shared_base_factorized_context_plus_segment_residual",
                affinity_target_encoding="mhcflurry",
                max_affinity_nm=100000,
                seed=seed,
                split_seed=split_seed,
            ),
        },
    ]


def _launch_condition(
    *,
    out_dir: Path,
    prefix: str,
    condition: dict[str, Any],
    epochs: int,
    batch_size: int,
    seed: int,
    split_seed: int,
) -> dict[str, Any]:
    run_id = f"{prefix}-{condition['condition_key'].lower()}-e{epochs:03d}-s{seed}"
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
        " ".join(condition["extra_args"]),
    ]
    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    remote_artifacts = _remote_artifacts(run_id)
    launcher_pid: int | None = None
    launch_status = "remote_artifacts_already_present"
    if not remote_artifacts:
        launcher_pid = _spawn_background_launch(cmd=cmd, log_path=log_path)
        launch_status = "spawned_background"

    return {
        "condition_key": condition["condition_key"],
        "design_id": condition["design_id"],
        "description": condition["description"],
        "affinity_loss_mode": "full",
        "affinity_assay_residual_mode": condition["affinity_assay_residual_mode"],
        "kd_grouping_mode": "split_kd_proxy",
        "affinity_target_encoding": condition["affinity_target_encoding"],
        "max_affinity_nM": condition["max_affinity_nM"],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "required_files": [
            "summary.json",
            "probe_affinity_over_epochs.json",
            "probe_affinity_over_epochs.csv",
            "val_predictions.csv",
            "test_predictions.csv",
        ],
        "run_id": run_id,
        "command": cmd,
        "extra_args": list(condition["extra_args"]),
        "launch_log": str(log_path),
        "launcher_pid": launcher_pid,
        "launch_status": launch_status,
        "remote_artifacts": remote_artifacts,
        "requested_gpu": os.environ.get("PRESTO_MODAL_GPU", GPU),
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch full-Presto retry on the EXP-21 data contract.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alleles = [token.strip() for token in str(args.alleles).split(",") if token.strip()]
    probes = [token.strip().upper() for token in str(args.probes).split(",") if token.strip()]
    if not probes:
        raise SystemExit("At least one probe peptide is required")

    conditions = _condition_rows(
        alleles=alleles,
        probes=probes,
        seed=int(args.seed),
        split_seed=int(args.split_seed),
    )
    metadata = {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "alleles": alleles,
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "split_policy": "peptide_group_80_10_10",
            "split_seed": int(args.split_seed),
            "train_seed": int(args.seed),
            "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"],
            "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"],
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "synthetic_negatives": False,
            "ranking_losses": False,
            "requested_gpu": os.environ.get("PRESTO_MODAL_GPU", GPU),
            "warm_start": "",
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "peptide_pos_mode": "concat_start_end_frac",
            "groove_pos_mode": "concat_start_end_frac",
            "binding_core_lengths": [8, 9, 10, 11],
            "binding_core_refinement": "shared",
            "binding_kinetic_input_mode": "affinity_vec",
            "binding_direct_segment_mode": "off",
        },
        "tested": [
            {
                "condition_key": row["condition_key"],
                "description": row["description"],
                "affinity_loss_mode": "full",
                "affinity_assay_residual_mode": row["affinity_assay_residual_mode"],
                "kd_grouping_mode": "split_kd_proxy",
                "affinity_target_encoding": row["affinity_target_encoding"],
                "max_affinity_nM": row["max_affinity_nM"],
            }
            for row in conditions
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="presto-full-output-exp21-retry",
        title="Presto Full-Output Retry on EXP-21 Contract",
        source_script="experiments/2026-03-16_1415_codex_presto-full-output-exp21-retry/code/launch.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_dir": str(out_dir),
                    "conditions": conditions,
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
            condition=condition,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            split_seed=int(args.split_seed),
        )
        for condition in conditions
    ]
    _write_manifest(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "launched", "experiment_dir": str(out_dir), "runs": manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
