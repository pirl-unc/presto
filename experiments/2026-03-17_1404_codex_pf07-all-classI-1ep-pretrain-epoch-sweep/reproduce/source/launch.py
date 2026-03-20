#!/usr/bin/env python
"""Launch an all-class-I PF07 epoch sweep on the rebuilt canonical dataset."""

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
DEFAULT_PREFIX = "presto-pf07-allclass1-20260317a"
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 43
DEFAULT_SPLIT_SEED = 42
DEFAULT_EPOCH_GRID = (10, 25, 50)
DEFAULT_INIT_CHECKPOINT = "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"
DEFAULT_INIT_CHECKPOINT_RUN_ID = "mhc-pretrain-d32-20260317a-e01"
DEFAULT_INIT_CHECKPOINT_NAME = "mhc_pretrain.pt"

VARIANTS = (
    {
        "condition_key": "pf07_control_constant",
        "design_id": "presto_pf07_control_constant",
        "description": "Flat honest PF07 control on rebuilt all-class-I numeric data",
        "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    },
    {
        "condition_key": "pf07_dag_method_leaf_constant",
        "design_id": "presto_pf07_dag_method_leaf_constant",
        "description": "Method-leaf DAG on rebuilt all-class-I numeric data",
        "affinity_assay_residual_mode": "dag_method_leaf",
    },
    {
        "condition_key": "pf07_dag_prep_readout_leaf_constant",
        "design_id": "presto_pf07_dag_prep_readout_leaf_constant",
        "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data",
        "affinity_assay_residual_mode": "dag_prep_readout_leaf",
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


def _require_remote_checkpoint(init_checkpoint: str) -> None:
    run_id = DEFAULT_INIT_CHECKPOINT_RUN_ID
    artifacts = _remote_artifacts(CHECKPOINT_VOLUME, run_id)
    names = {Path(item).name for item in artifacts}
    if DEFAULT_INIT_CHECKPOINT_NAME not in names:
        print(
            "WARN: warm-start checkpoint precheck did not see "
            f"{DEFAULT_INIT_CHECKPOINT_NAME} under {run_id}; continuing because "
            "Modal CLI listing can be inconsistent across launch contexts."
        )


def _run_detached_launch(*, cmd: list[str], log_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PRESTO_MODAL_GPU"] = env.get("PRESTO_MODAL_GPU", GPU)
    result = subprocess.run(
        cmd,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    output = result.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    return {
        "returncode": int(result.returncode),
        "output": output,
    }


def _parse_epoch_grid(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Epochs must be positive; got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one epoch budget is required")
    return values


def _build_extra_args(
    *,
    probe_alleles: list[str],
    probes: list[str],
    design_id: str,
    seed: int,
    split_seed: int,
    residual_mode: str,
    init_checkpoint: str,
) -> list[str]:
    args = [
        "--source", "iedb",
        "--design-id", design_id,
        "--alleles", ",".join(probe_alleles),
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
        "--train-mhc-class-filter", "I",
        "--train-all-alleles",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--binding-kd-family-consistency-weight", "0",
        "--binding-proxy-cross-consistency-weight", "0",
        "--binding-output-consistency-beta", "0.25",
        "--probe-plot-frequency", "final",
        "--epoch-val-metrics-frequency", "1",
    ]
    if init_checkpoint:
        args.extend(["--init-checkpoint", init_checkpoint])
    return args


def _condition_rows(
    *,
    probe_alleles: list[str],
    probes: list[str],
    prefix: str,
    epoch_grid: list[int],
    batch_size: int,
    seed: int,
    split_seed: int,
    init_checkpoint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for epochs in epoch_grid:
            run_id = f"{prefix}-{variant['condition_key']}-e{epochs:03d}-s{seed}"
            extra_args = _build_extra_args(
                probe_alleles=probe_alleles,
                probes=probes,
                design_id=str(variant["design_id"]),
                seed=seed,
                split_seed=split_seed,
                residual_mode=str(variant["affinity_assay_residual_mode"]),
                init_checkpoint=init_checkpoint,
            )
            rows.append(
                {
                    "condition_key": variant["condition_key"],
                    "design_id": variant["design_id"],
                    "description": variant["description"],
                    "affinity_assay_residual_mode": variant["affinity_assay_residual_mode"],
                    "epoch_budget": int(epochs),
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
                    "init_checkpoint": init_checkpoint,
                    "required_files": [
                        "summary.json",
                        "epoch_metrics.csv",
                        "epoch_metrics.json",
                        "probe_affinity_over_epochs.csv",
                        "probe_affinity_over_epochs.json",
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


def _metadata(
    *,
    probe_alleles: list[str],
    probes: list[str],
    epoch_grid: list[int],
    batch_size: int,
    seed: int,
    split_seed: int,
    init_checkpoint: str,
    tested_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "source_refresh": "canonical rebuild 2026-03-17",
            "sequence_resolution": "mhcseqs_first_with_index_fallback",
            "source_filter": "iedb",
            "train_all_alleles": True,
            "train_mhc_class_filter": "I",
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "split_policy": "peptide_group_80_10_10",
            "split_seed": int(split_seed),
            "train_seed": int(seed),
            "probe_alleles": probe_alleles,
            "probe_peptides": probes,
            "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"],
            "assay_selector_inputs_forbidden": True,
            "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"],
        },
        "training": {
            "pretraining": {
                "mode": "mhc_pretrain",
                "warm_start_checkpoint": init_checkpoint,
                "warm_start_epochs": 1,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
            },
            "downstream": {
                "epoch_grid": epoch_grid,
                "batch_size": int(batch_size),
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
        description="Launch an honest all-class-I PF07 epoch sweep on the rebuilt canonical dataset."
    )
    parser.add_argument("--epochs-grid", type=str, default=",".join(str(v) for v in DEFAULT_EPOCH_GRID))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--ft-prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--init-checkpoint", type=str, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    epoch_grid = _parse_epoch_grid(str(args.epochs_grid))
    probe_alleles = [token.strip() for token in str(args.alleles).split(",") if token.strip()]
    probes = [token.strip().upper() for token in str(args.probes).split(",") if token.strip()]
    if not probe_alleles:
        raise SystemExit("At least one probe allele is required")
    if not probes:
        raise SystemExit("At least one probe peptide is required")

    rows = _condition_rows(
        probe_alleles=probe_alleles,
        probes=probes,
        prefix=str(args.ft_prefix),
        epoch_grid=epoch_grid,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        split_seed=int(args.split_seed),
        init_checkpoint=str(args.init_checkpoint),
    )
    tested_rows = [
        {
            "condition_key": row["condition_key"],
            "description": row["description"],
            "epoch_budget": row["epoch_budget"],
            "lr": row["lr"],
            "lr_schedule": row["lr_schedule"],
            "affinity_loss_mode": row["affinity_loss_mode"],
            "affinity_target_encoding": row["affinity_target_encoding"],
            "affinity_assay_residual_mode": row["affinity_assay_residual_mode"],
            "kd_grouping_mode": row["kd_grouping_mode"],
            "max_affinity_nM": row["max_affinity_nM"],
            "init_checkpoint": row["init_checkpoint"],
        }
        for row in rows
    ]
    metadata = _metadata(
        probe_alleles=probe_alleles,
        probes=probes,
        epoch_grid=epoch_grid,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        split_seed=int(args.split_seed),
        init_checkpoint=str(args.init_checkpoint),
        tested_rows=tested_rows,
    )

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="pf07-all-classI-1ep-pretrain-epoch-sweep",
        title="PF07 All-Class-I 1ep-Pretrain Epoch Sweep",
        source_script="experiments/2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep/code/launch.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )
    (out_dir / "results" / "runs").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if args.dry_run:
        _write_manifest(manifest_path, rows)
        print(json.dumps({"out_dir": str(out_dir), "n_conditions": len(rows), "first_run": rows[0] if rows else None}, indent=2))
        return

    _require_remote_checkpoint(str(args.init_checkpoint))
    launched: list[dict[str, Any]] = []
    log_root = out_dir / "launch_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    launch_failures: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row["run_id"])
        remote_artifacts = _remote_artifacts(CHECKPOINT_VOLUME, run_id)
        launch_status = "remote_artifacts_already_present"
        log_path = log_root / f"{run_id}.log"
        if not remote_artifacts:
            launch_result = _run_detached_launch(cmd=list(row["command"]), log_path=log_path)
            if int(launch_result["returncode"]) == 0:
                launch_status = "submitted_detached"
            else:
                launch_status = "submit_failed"
                launch_failures.append(
                    {
                        "run_id": run_id,
                        "log_path": str(log_path),
                        "returncode": int(launch_result["returncode"]),
                    }
                )
        launched.append(
            {
                **row,
                "launch_log": str(log_path),
                "launch_status": launch_status,
                "remote_artifacts": remote_artifacts,
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    _write_manifest(manifest_path, launched)
    if launch_failures:
        raise SystemExit(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "n_launched": len(launched) - len(launch_failures),
                    "n_failed": len(launch_failures),
                    "failures": launch_failures,
                    "manifest": str(manifest_path),
                },
                indent=2,
            )
        )
    print(json.dumps({"out_dir": str(out_dir), "n_launched": len(launched), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
