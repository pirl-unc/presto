#!/usr/bin/env python
"""Launch a PF07 DAG epoch-curve rerun with richer per-epoch artifacts."""

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
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL")
DEFAULT_PREFIX = "presto-pf07-dagcurve-20260317a"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 43
DEFAULT_SPLIT_SEED = 42

VARIANTS = (
    {
        "condition_key": "pf07_control_constant",
        "design_id": "presto_pf07_control_constant",
        "description": "Flat honest PF07 control rerun: 1e-3 constant",
        "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
        "lr": 1e-3,
        "lr_schedule": "constant",
    },
    {
        "condition_key": "pf07_dag_family_constant",
        "design_id": "presto_pf07_dag_family_constant",
        "description": "Family-anchor DAG rerun: 1e-3 constant",
        "affinity_assay_residual_mode": "dag_family",
        "lr": 1e-3,
        "lr_schedule": "constant",
    },
    {
        "condition_key": "pf07_dag_method_leaf_constant",
        "design_id": "presto_pf07_dag_method_leaf_constant",
        "description": "Method-leaf DAG rerun: 1e-3 constant",
        "affinity_assay_residual_mode": "dag_method_leaf",
        "lr": 1e-3,
        "lr_schedule": "constant",
    },
    {
        "condition_key": "pf07_dag_prep_readout_leaf_constant",
        "design_id": "presto_pf07_dag_prep_readout_leaf_constant",
        "description": "Prep/readout-leaf DAG rerun: 1e-3 constant",
        "affinity_assay_residual_mode": "dag_prep_readout_leaf",
        "lr": 1e-3,
        "lr_schedule": "constant",
    },
    {
        "condition_key": "pf07_dag_method_leaf_warmup_cosine",
        "design_id": "presto_pf07_dag_method_leaf_warmup_cosine",
        "description": "Method-leaf DAG schedule variant: 3e-4 warmup_cosine",
        "affinity_assay_residual_mode": "dag_method_leaf",
        "lr": 3e-4,
        "lr_schedule": "warmup_cosine",
    },
    {
        "condition_key": "pf07_dag_prep_readout_leaf_warmup_cosine",
        "design_id": "presto_pf07_dag_prep_readout_leaf_warmup_cosine",
        "description": "Prep/readout-leaf DAG schedule variant: 3e-4 warmup_cosine",
        "affinity_assay_residual_mode": "dag_prep_readout_leaf",
        "lr": 3e-4,
        "lr_schedule": "warmup_cosine",
    },
)


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
    seed: int,
    split_seed: int,
    residual_mode: str,
    lr: float,
    lr_schedule: str,
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
        "--lr", str(lr),
        "--lr-schedule", lr_schedule,
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
        "--probe-plot-frequency", "final",
        "--epoch-val-metrics-frequency", "1",
    ]


def _condition_rows(
    *,
    alleles: list[str],
    probes: list[str],
    seed: int,
    split_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        row = dict(variant)
        row["extra_args"] = _build_extra_args(
            alleles=alleles,
            probes=probes,
            design_id=str(row["design_id"]),
            seed=seed,
            split_seed=split_seed,
            residual_mode=str(row["affinity_assay_residual_mode"]),
            lr=float(row["lr"]),
            lr_schedule=str(row["lr_schedule"]),
        )
        rows.append(row)
    return rows


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
    run_id = f"{prefix}-{condition['condition_key']}-e{epochs:03d}-s{seed}"
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
        "affinity_target_encoding": "mhcflurry",
        "max_affinity_nM": 100000,
        "lr": float(condition["lr"]),
        "lr_schedule": condition["lr_schedule"],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "required_files": [
            "summary.json",
            "epoch_metrics.csv",
            "epoch_metrics.json",
            "val_metrics_over_epochs.png",
            "probe_affinity_over_epochs.json",
            "probe_affinity_over_epochs.csv",
            "probe_affinity_over_epochs.png",
            "probe_affinity_all_outputs_over_epochs.png",
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
    parser = argparse.ArgumentParser(
        description="Launch PF07 DAG epoch-curve rerun on the main Presto affinity path."
    )
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
            "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"],
            "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"],
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "split_policy": "peptide_group_80_10_10",
            "split_seed": int(args.split_seed),
            "train_seed": int(args.seed),
            "probe_peptides": probes,
            "assay_selector_inputs_forbidden": True,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "peptide_pos_mode": "concat_start_end_frac",
            "groove_pos_mode": "concat_start_end_frac",
            "binding_core_lengths": [8, 9, 10, 11],
            "binding_core_refinement": "shared",
            "affinity_loss_mode": "full",
            "affinity_target_encoding": "mhcflurry",
            "kd_grouping_mode": "split_kd_proxy",
            "max_affinity_nM": 100000,
            "binding_kinetic_input_mode": "affinity_vec",
            "binding_direct_segment_mode": "off",
            "requested_gpu": os.environ.get("PRESTO_MODAL_GPU", GPU),
            "epoch_val_metrics_frequency": 1,
            "probe_plot_frequency": "final",
            "probe_artifact_schema": [
                "KD_nM",
                "IC50_nM",
                "EC50_nM",
                "KD_proxy_ic50_nM",
                "KD_proxy_ec50_nM",
                "binding_affinity_probe_kd",
            ],
        },
        "tested": [
            {
                "condition_key": condition["condition_key"],
                "description": condition["description"],
                "affinity_assay_residual_mode": condition["affinity_assay_residual_mode"],
                "affinity_loss_mode": "full",
                "affinity_target_encoding": "mhcflurry",
                "kd_grouping_mode": "split_kd_proxy",
                "lr": float(condition["lr"]),
                "lr_schedule": str(condition["lr_schedule"]),
                "max_affinity_nM": 100000,
            }
            for condition in conditions
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="pf07-dag-epoch-curve-rerun",
        title="PF07 DAG Epoch-Curve Rerun",
        source_script="experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun/code/launch.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    manifest_path = out_dir / "manifest.json"
    if args.dry_run:
        dry_manifest = []
        for condition in conditions:
            dry_manifest.append(
                {
                    "condition_key": condition["condition_key"],
                    "design_id": condition["design_id"],
                    "description": condition["description"],
                    "lr": float(condition["lr"]),
                    "lr_schedule": str(condition["lr_schedule"]),
                    "run_id": f"{args.prefix}-{condition['condition_key']}-e{int(args.epochs):03d}-s{int(args.seed)}",
                    "extra_args": list(condition["extra_args"]),
                }
            )
        _write_manifest(manifest_path, dry_manifest)
        print(json.dumps(dry_manifest, indent=2))
        return

    launched = [
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
    _write_manifest(manifest_path, launched)
    print(json.dumps(launched, indent=2))


if __name__ == "__main__":
    main()
