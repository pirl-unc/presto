#!/usr/bin/env python
"""Launch a staged PF07 MHC-pretrain impact sweep on the main Presto path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL")

DEFAULT_FT_PREFIX = "presto-pf07-mhcpre-20260317a"
DEFAULT_PRETRAIN_PREFIX = "mhc-pretrain-d32-20260317a"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 43
DEFAULT_SPLIT_SEED = 42

DEFAULT_PRETRAIN_BATCH_SIZE = 192
DEFAULT_PRETRAIN_SEED = 42
DEFAULT_PRETRAIN_CHECKPOINT_NAME = "mhc_pretrain.pt"


MODEL_VARIANTS = (
    {
        "condition_key": "pf07_control_constant",
        "design_id": "presto_pf07_control_constant",
        "description": "Flat honest PF07 control with constant LR",
        "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    },
    {
        "condition_key": "pf07_dag_method_leaf_constant",
        "design_id": "presto_pf07_dag_method_leaf_constant",
        "description": "Method-leaf DAG with constant LR",
        "affinity_assay_residual_mode": "dag_method_leaf",
    },
    {
        "condition_key": "pf07_dag_prep_readout_leaf_constant",
        "design_id": "presto_pf07_dag_prep_readout_leaf_constant",
        "description": "Prep/readout-leaf DAG with constant LR",
        "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    },
)

PRETRAIN_VARIANTS = (
    {
        "pretrain_key": "pretrain_0ep",
        "description": "No warm start",
        "epochs": 0,
    },
    {
        "pretrain_key": "pretrain_1ep",
        "description": "Fresh d32 MHC class/species pretrain for 1 epoch",
        "epochs": 1,
    },
    {
        "pretrain_key": "pretrain_2ep",
        "description": "Fresh d32 MHC class/species pretrain for 2 epochs",
        "epochs": 2,
    },
)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remote_artifacts(volume: str, run_id: str) -> list[str]:
    result = subprocess.run(
        ["modal", "volume", "ls", volume, run_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _remote_ready(remote_artifacts: list[str], required_files: list[str]) -> bool:
    present = {Path(item).name for item in remote_artifacts}
    return all(name in present for name in required_files)


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


def _wait_for_remote_files(
    *,
    run_id: str,
    required_files: list[str],
    timeout_sec: float,
    poll_interval_sec: float,
) -> list[str]:
    deadline = time.time() + float(timeout_sec)
    while True:
        remote_artifacts = _remote_artifacts(CHECKPOINT_VOLUME, run_id)
        if _remote_ready(remote_artifacts, required_files):
            return remote_artifacts
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {run_id} to materialize {required_files} on {CHECKPOINT_VOLUME}"
            )
        time.sleep(float(poll_interval_sec))


def _pretrain_run_id(prefix: str, epochs: int) -> str:
    return f"{prefix}-e{epochs:02d}"


def _pretrain_remote_checkpoint(run_id: str) -> str:
    return f"/checkpoints/{run_id}/{DEFAULT_PRETRAIN_CHECKPOINT_NAME}"


def _build_pretrain_command(
    *,
    run_id: str,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[str]:
    extra_args = [
        "--d-model", "32",
        "--n-layers", "2",
        "--n-heads", "4",
        "--seed", str(seed),
    ]
    return [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py",
        "--mode",
        "mhc_pretrain",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--run-id",
        run_id,
        "--checkpoint-name",
        DEFAULT_PRETRAIN_CHECKPOINT_NAME,
        "--extra-args",
        " ".join(extra_args),
    ]


def _build_finetune_extra_args(
    *,
    alleles: list[str],
    probes: list[str],
    design_id: str,
    seed: int,
    split_seed: int,
    residual_mode: str,
    init_checkpoint: str,
) -> list[str]:
    args = [
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
        "--lr-schedule", "constant",
        "--weight-decay", "0.01",
        "--affinity-loss-mode", "full",
        "--affinity-target-encoding", "mhcflurry",
        "--max-affinity-nm", "100000",
        "--affinity-assay-residual-mode", residual_mode,
        "--kd-grouping-mode", "split_kd_proxy",
        "--binding-kinetic-input-mode", "affinity_vec",
        "--binding-direct-segment-mode", "off",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--binding-kd-family-consistency-weight", "0",
        "--binding-proxy-cross-consistency-weight", "0",
        "--binding-output-consistency-beta", "0.25",
        "--probe-plot-frequency", "off",
    ]
    if init_checkpoint:
        args.extend(["--init-checkpoint", init_checkpoint])
    return args


def _pretrain_rows(
    *,
    prefix: str,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in PRETRAIN_VARIANTS:
        epochs = int(variant["epochs"])
        if epochs <= 0:
            continue
        run_id = _pretrain_run_id(prefix, epochs)
        rows.append(
            {
                "phase": "pretrain",
                "pretrain_key": variant["pretrain_key"],
                "pretrain_epochs": epochs,
                "description": variant["description"],
                "run_id": run_id,
                "checkpoint_name": DEFAULT_PRETRAIN_CHECKPOINT_NAME,
                "checkpoint_path": _pretrain_remote_checkpoint(run_id),
                "required_files": ["summary.json", DEFAULT_PRETRAIN_CHECKPOINT_NAME],
                "batch_size": int(batch_size),
                "seed": int(seed),
                "command": _build_pretrain_command(
                    run_id=run_id,
                    epochs=epochs,
                    batch_size=int(batch_size),
                    seed=int(seed),
                ),
                "requested_gpu": os.environ.get("PRESTO_MODAL_GPU", GPU),
            }
        )
    return rows


def _finetune_rows(
    *,
    alleles: list[str],
    probes: list[str],
    prefix: str,
    epochs: int,
    batch_size: int,
    seed: int,
    split_seed: int,
    pretrain_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_variant in MODEL_VARIANTS:
        for pretrain_variant in PRETRAIN_VARIANTS:
            pretrain_epochs = int(pretrain_variant["epochs"])
            init_checkpoint = (
                _pretrain_remote_checkpoint(_pretrain_run_id(pretrain_prefix, pretrain_epochs))
                if pretrain_epochs > 0
                else ""
            )
            run_id = (
                f"{prefix}-{model_variant['condition_key']}-{pretrain_variant['pretrain_key']}"
                f"-e{epochs:03d}-s{seed}"
            )
            extra_args = _build_finetune_extra_args(
                alleles=alleles,
                probes=probes,
                design_id=str(model_variant["design_id"]),
                seed=seed,
                split_seed=split_seed,
                residual_mode=str(model_variant["affinity_assay_residual_mode"]),
                init_checkpoint=init_checkpoint,
            )
            rows.append(
                {
                    "phase": "finetune",
                    "condition_key": model_variant["condition_key"],
                    "design_id": model_variant["design_id"],
                    "description": model_variant["description"],
                    "affinity_assay_residual_mode": model_variant["affinity_assay_residual_mode"],
                    "pretrain_key": pretrain_variant["pretrain_key"],
                    "pretrain_epochs": pretrain_epochs,
                    "init_checkpoint": init_checkpoint,
                    "run_id": run_id,
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "seed": int(seed),
                    "split_seed": int(split_seed),
                    "lr": 1e-3,
                    "lr_schedule": "constant",
                    "affinity_loss_mode": "full",
                    "affinity_target_encoding": "mhcflurry",
                    "kd_grouping_mode": "split_kd_proxy",
                    "max_affinity_nM": 100000,
                    "required_files": [
                        "summary.json",
                        "probe_affinity_over_epochs.json",
                        "probe_affinity_over_epochs.csv",
                        "val_predictions.csv",
                        "test_predictions.csv",
                    ],
                    "command": [
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
                    ],
                    "extra_args": extra_args,
                    "requested_gpu": os.environ.get("PRESTO_MODAL_GPU", GPU),
                }
            )
    return rows


def _launch_rows(
    *,
    rows: list[dict[str, Any]],
    log_root: Path,
) -> list[dict[str, Any]]:
    launched: list[dict[str, Any]] = []
    log_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        run_id = str(row["run_id"])
        remote_artifacts = _remote_artifacts(CHECKPOINT_VOLUME, run_id)
        launcher_pid: int | None = None
        launch_status = "remote_artifacts_already_present"
        log_path = log_root / f"{run_id}.log"
        if not remote_artifacts:
            launcher_pid = _spawn_background_launch(cmd=list(row["command"]), log_path=log_path)
            launch_status = "spawned_background"
        launched.append(
            {
                **row,
                "launch_log": str(log_path),
                "launcher_pid": launcher_pid,
                "launch_status": launch_status,
                "remote_artifacts": remote_artifacts,
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    return launched


def _metadata(
    *,
    alleles: list[str],
    probes: list[str],
    ft_epochs: int,
    ft_batch_size: int,
    ft_seed: int,
    split_seed: int,
    pretrain_batch_size: int,
    pretrain_seed: int,
    tested_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "alleles": alleles,
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "split_policy": "peptide_group_80_10_10",
            "split_seed": int(split_seed),
            "train_seed": int(ft_seed),
            "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"],
            "assay_selector_inputs_forbidden": True,
            "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"],
            "probe_peptides": probes,
        },
        "training": {
            "pretraining": {
                "mode": "mhc_pretrain",
                "targets": ["chain_type", "species", "class"],
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "batch_size": int(pretrain_batch_size),
                "seed": int(pretrain_seed),
                "pretrain_epochs": [1, 2],
                "checkpoint_name": DEFAULT_PRETRAIN_CHECKPOINT_NAME,
            },
            "downstream": {
                "epochs": int(ft_epochs),
                "batch_size": int(ft_batch_size),
                "optimizer": "AdamW",
                "weight_decay": 0.01,
                "synthetic_negatives": False,
                "requested_gpu": os.environ.get("PRESTO_MODAL_GPU", GPU),
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "peptide_pos_mode": "concat_start_end_frac",
                "groove_pos_mode": "concat_start_end_frac",
                "binding_core_lengths": [8, 9, 10, 11],
                "binding_core_refinement": "shared",
                "binding_kinetic_input_mode": "affinity_vec",
                "binding_direct_segment_mode": "off",
                "affinity_loss_mode": "full",
                "affinity_target_encoding": "mhcflurry",
                "kd_grouping_mode": "split_kd_proxy",
                "max_affinity_nM": 100000,
            },
        },
        "tested": tested_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a staged PF07 MHC-pretrain impact sweep on the honest main Presto path."
    )
    parser.add_argument("--phase", choices=("pretrain", "finetune", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--pretrain-batch-size", type=int, default=DEFAULT_PRETRAIN_BATCH_SIZE)
    parser.add_argument("--pretrain-seed", type=int, default=DEFAULT_PRETRAIN_SEED)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--ft-prefix", type=str, default=DEFAULT_FT_PREFIX)
    parser.add_argument("--pretrain-prefix", type=str, default=DEFAULT_PRETRAIN_PREFIX)
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--wait-for-pretrains", action="store_true")
    parser.add_argument("--wait-timeout-sec", type=float, default=7200.0)
    parser.add_argument("--poll-interval-sec", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alleles = [token.strip() for token in str(args.alleles).split(",") if token.strip()]
    probes = [token.strip().upper() for token in str(args.probes).split(",") if token.strip()]
    if not probes:
        raise SystemExit("At least one probe peptide is required")

    pretrain_rows = _pretrain_rows(
        prefix=str(args.pretrain_prefix),
        batch_size=int(args.pretrain_batch_size),
        seed=int(args.pretrain_seed),
    )
    finetune_rows = _finetune_rows(
        alleles=alleles,
        probes=probes,
        prefix=str(args.ft_prefix),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        split_seed=int(args.split_seed),
        pretrain_prefix=str(args.pretrain_prefix),
    )
    tested_rows = [
        {
            "condition_key": row["condition_key"],
            "description": row["description"],
            "pretrain_key": row["pretrain_key"],
            "pretrain_epochs": row["pretrain_epochs"],
            "lr": row["lr"],
            "lr_schedule": row["lr_schedule"],
            "affinity_loss_mode": row["affinity_loss_mode"],
            "affinity_target_encoding": row["affinity_target_encoding"],
            "affinity_assay_residual_mode": row["affinity_assay_residual_mode"],
            "kd_grouping_mode": row["kd_grouping_mode"],
            "max_affinity_nM": row["max_affinity_nM"],
            "init_checkpoint": row["init_checkpoint"],
        }
        for row in finetune_rows
    ]
    metadata = _metadata(
        alleles=alleles,
        probes=probes,
        ft_epochs=int(args.epochs),
        ft_batch_size=int(args.batch_size),
        ft_seed=int(args.seed),
        split_seed=int(args.split_seed),
        pretrain_batch_size=int(args.pretrain_batch_size),
        pretrain_seed=int(args.pretrain_seed),
        tested_rows=tested_rows,
    )
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="pf07-mhc-pretrain-impact-sweep",
        title="PF07 MHC Pretrain Impact Sweep",
        source_script="experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/code/launch.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_dir": str(out_dir),
                    "phase": str(args.phase),
                    "pretrain_manifest": pretrain_rows,
                    "finetune_manifest": finetune_rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.phase in {"pretrain", "all"}:
        launched_pretrains = _launch_rows(
            rows=pretrain_rows,
            log_root=out_dir / "launch_logs" / "pretrain",
        )
        _write_manifest(out_dir / "manifest_pretrain.json", launched_pretrains)
    else:
        launched_pretrains = json.loads((out_dir / "manifest_pretrain.json").read_text())

    if args.phase == "pretrain":
        return

    if args.wait_for_pretrains:
        refreshed_pretrains: list[dict[str, Any]] = []
        for row in launched_pretrains:
            remote_artifacts = _wait_for_remote_files(
                run_id=str(row["run_id"]),
                required_files=list(row["required_files"]),
                timeout_sec=float(args.wait_timeout_sec),
                poll_interval_sec=float(args.poll_interval_sec),
            )
            refreshed_pretrains.append(
                {
                    **row,
                    "launch_status": "remote_ready",
                    "remote_artifacts": remote_artifacts,
                    "remote_ready_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        launched_pretrains = refreshed_pretrains
        _write_manifest(out_dir / "manifest_pretrain.json", launched_pretrains)
    else:
        missing = []
        for row in launched_pretrains:
            remote_artifacts = _remote_artifacts(CHECKPOINT_VOLUME, str(row["run_id"]))
            if not _remote_ready(remote_artifacts, list(row["required_files"])):
                missing.append(str(row["run_id"]))
        if missing:
            raise SystemExit(
                "Pretrain checkpoints are not ready yet. Re-run with --wait-for-pretrains "
                f"or launch finetune phase later. Missing: {', '.join(missing)}"
            )

    launched_finetunes = _launch_rows(
        rows=finetune_rows,
        log_root=out_dir / "launch_logs" / "finetune",
    )
    _write_manifest(out_dir / "manifest.json", launched_finetunes)


if __name__ == "__main__":
    main()
