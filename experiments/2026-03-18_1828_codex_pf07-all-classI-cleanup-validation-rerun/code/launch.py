#!/usr/bin/env python
"""Launch the all-class-I PF07 cleanup-validation rerun."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


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
DEFAULT_PREFIX = "presto-pf07-allclass1-cleanup-20260318a"
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 43
DEFAULT_SPLIT_SEED = 42
DEFAULT_EPOCH_GRID = (10, 25, 50)
DEFAULT_INIT_CHECKPOINT = "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"
DEFAULT_INIT_CHECKPOINT_RUN_ID = "mhc-pretrain-d32-20260317a-e01"
DEFAULT_INIT_CHECKPOINT_NAME = "mhc_pretrain.pt"
DEFAULT_LOCAL_INIT_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep"
    / "results"
    / "pretrains"
    / "mhc-pretrain-d32-20260317a-e01"
    / "mhc_pretrain.pt"
)
PRETRAIN_SOURCE_EXPERIMENT = "2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep"

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


def _require_local_checkpoint(path: str) -> None:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Local warm-start checkpoint not found: "
            f"{checkpoint_path}"
        )


def _run_logged_command(
    *,
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
    cwd: Path = REPO_ROOT,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return {"returncode": int(result.returncode)}


def _required_files_present(run_dir: Path, required_files: Sequence[str]) -> bool:
    return all((run_dir / name).exists() for name in required_files)


def _aggregate_local_results(out_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_summary_runs.py",
            "--experiment-dir",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _reuse_existing_experiment_dir(path: Path) -> bool:
    return path.exists() and (path / "README.md").exists() and (path / "code" / "launch.py").exists()


def _portable_path_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


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


def _parse_condition_keys(raw: str, *, valid_keys: tuple[str, ...]) -> list[str]:
    requested = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not requested:
        return list(valid_keys)
    unknown = [token for token in requested if token not in valid_keys]
    if unknown:
        raise ValueError(
            f"Unknown condition key(s): {', '.join(sorted(unknown))}; "
            f"valid keys are: {', '.join(valid_keys)}"
        )
    return requested


def _build_extra_args(
    *,
    probe_alleles: list[str],
    probes: list[str],
    design_id: str,
    seed: int,
    split_seed: int,
    residual_mode: str,
    init_checkpoint: str,
    max_records: Optional[int],
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
    if max_records is not None:
        args.extend(["--max-records", str(int(max_records))])
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
    local_init_checkpoint: str,
    selected_condition_keys: list[str],
    max_records: Optional[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        if str(variant["condition_key"]) not in selected_condition_keys:
            continue
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
                max_records=max_records,
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
                    "max_records": None if max_records is None else int(max_records),
                    "lr": 1e-3,
                    "lr_schedule": "constant",
                    "affinity_loss_mode": "full",
                    "affinity_target_encoding": "mhcflurry",
                    "kd_grouping_mode": "split_kd_proxy",
                    "max_affinity_nM": 100000,
                    "init_checkpoint": init_checkpoint,
                    "local_init_checkpoint": _portable_path_str(Path(local_init_checkpoint)),
                    "required_files": [
                        "summary.json",
                        "epoch_metrics.csv",
                        "epoch_metrics.json",
                        "probe_affinity_over_epochs.csv",
                        "probe_affinity_over_epochs.json",
                        "val_predictions.csv",
                        "test_predictions.csv",
                    ],
                    "modal_command": [
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
    local_init_checkpoint: str,
    tested_rows: list[dict[str, Any]],
    max_records: Optional[int],
) -> dict[str, Any]:
    return {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "source_refresh": "canonical rebuild 2026-03-17",
            "sequence_resolution": "mhcseqs_first_with_index_fallback",
            "validation_purpose": "post_mhcseqs_cleanup_rerun",
            "comparison_target": "2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep",
            "source_filter": "iedb",
            "train_all_alleles": True,
            "train_mhc_class_filter": "I",
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "max_records": None if max_records is None else int(max_records),
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
                "warm_start_checkpoint_modal": init_checkpoint,
                "warm_start_checkpoint_local": _portable_path_str(Path(local_init_checkpoint)),
                "warm_start_checkpoint_source_experiment": PRETRAIN_SOURCE_EXPERIMENT,
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


def _build_local_command(
    *,
    row: Mapping[str, Any],
    run_dir: Path,
    data_dir: Path,
    device: str,
    mps_safe_mode: str,
) -> list[str]:
    python_cmd = os.environ.get("PRESTO_LOCAL_PYTHON", "python")
    cmd = [
        python_cmd,
        "-m",
        "presto.scripts.focused_binding_probe",
        "--data-dir",
        _portable_path_str(data_dir),
        "--out-dir",
        _portable_path_str(run_dir),
        "--epochs",
        str(row["epochs"]),
        "--batch-size",
        str(row["batch_size"]),
        "--device",
        str(device),
        "--mps-safe-mode",
        str(mps_safe_mode),
        "--num-workers",
        "0",
        "--no-pin-memory",
        "--no-persistent-workers",
    ]
    extra_args = list(row.get("extra_args", []))
    local_checkpoint = str(row.get("local_init_checkpoint", "")).strip()
    if local_checkpoint:
        local_checkpoint_str = _portable_path_str(Path(local_checkpoint))
        filtered_args: list[str] = []
        skip_next = False
        for idx, token in enumerate(extra_args):
            if skip_next:
                skip_next = False
                continue
            if token == "--init-checkpoint":
                skip_next = True
                continue
            filtered_args.append(token)
        extra_args = filtered_args + ["--init-checkpoint", local_checkpoint_str]
    cmd.extend(extra_args)
    return cmd


def _ensure_experiment_dir(
    *,
    out_dir: Path,
    agent_label: str,
    metadata: Mapping[str, Any],
) -> Path:
    if _reuse_existing_experiment_dir(out_dir):
        return out_dir
    return initialize_experiment_dir(
        out_dir=str(out_dir),
        slug="pf07-all-classI-cleanup-validation-rerun",
        title="PF07 All-Class-I Cleanup Validation Rerun",
        source_script="experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py",
        agent_label=agent_label,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the post-cleanup rerun of the all-class-I PF07 epoch sweep."
    )
    parser.add_argument("--backend", choices=("modal", "local"), default="modal")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--mps-safe-mode",
        choices=("auto", "off", "manual_dropout", "zero_dropout"),
        default="auto",
    )
    parser.add_argument("--local-data-dir", type=str, default="data")
    parser.add_argument("--epochs-grid", type=str, default=",".join(str(v) for v in DEFAULT_EPOCH_GRID))
    parser.add_argument(
        "--condition-keys",
        type=str,
        default=",".join(str(item["condition_key"]) for item in VARIANTS),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap for reduced local validation runs; <=0 means no cap.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--ft-prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--init-checkpoint", type=str, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--local-init-checkpoint", type=str, default=str(DEFAULT_LOCAL_INIT_CHECKPOINT))
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    epoch_grid = _parse_epoch_grid(str(args.epochs_grid))
    selected_condition_keys = _parse_condition_keys(
        str(args.condition_keys),
        valid_keys=tuple(str(item["condition_key"]) for item in VARIANTS),
    )
    max_records = None if int(args.max_records) <= 0 else int(args.max_records)
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
        local_init_checkpoint=str(args.local_init_checkpoint),
        selected_condition_keys=selected_condition_keys,
        max_records=max_records,
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
            "max_records": row["max_records"],
            "init_checkpoint_modal": row["init_checkpoint"],
            "init_checkpoint_local": _portable_path_str(Path(str(row["local_init_checkpoint"]))),
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
        local_init_checkpoint=str(args.local_init_checkpoint),
        tested_rows=tested_rows,
        max_records=max_records,
    )

    out_dir = _ensure_experiment_dir(
        out_dir=Path(args.out_dir),
        agent_label=str(args.agent_label),
        metadata=metadata,
    )
    results_root = out_dir / "results" / "runs"
    results_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    local_data_dir = Path(str(args.local_data_dir)).expanduser()
    if not local_data_dir.is_absolute():
        local_data_dir = (REPO_ROOT / local_data_dir).resolve()

    if args.dry_run:
        preview_rows = list(rows)
        if preview_rows:
            preview_rows[0] = {
                **preview_rows[0],
                "local_command": _build_local_command(
                    row=preview_rows[0],
                    run_dir=results_root / str(preview_rows[0]["run_id"]),
                    data_dir=local_data_dir,
                    device=str(args.device),
                    mps_safe_mode=str(args.mps_safe_mode),
                ),
            }
        _write_manifest(manifest_path, preview_rows)
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "backend": str(args.backend),
                    "device": str(args.device),
                    "mps_safe_mode": str(args.mps_safe_mode),
                    "condition_keys": selected_condition_keys,
                    "max_records": max_records,
                    "n_conditions": len(preview_rows),
                    "first_run": preview_rows[0] if preview_rows else None,
                },
                indent=2,
            )
        )
        return

    if str(args.backend) == "modal":
        _require_remote_checkpoint(str(args.init_checkpoint))
    else:
        _require_local_checkpoint(str(args.local_init_checkpoint))
    launched: list[dict[str, Any]] = []
    log_root = out_dir / "launch_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    launch_failures: list[dict[str, Any]] = []
    modal_env = os.environ.copy()
    modal_env["PRESTO_MODAL_GPU"] = modal_env.get("PRESTO_MODAL_GPU", GPU)
    local_env = os.environ.copy()
    local_env.setdefault("PYTHONUNBUFFERED", "1")
    for row in rows:
        run_id = str(row["run_id"])
        run_dir = results_root / run_id
        required_files = tuple(str(name) for name in row["required_files"])
        remote_artifacts: list[str] = []
        log_path = log_root / f"{run_id}.log"
        local_command = _build_local_command(
            row=row,
            run_dir=run_dir,
            data_dir=local_data_dir,
            device=str(args.device),
            mps_safe_mode=str(args.mps_safe_mode),
        )
        launch_status = ""
        launch_command: list[str] = []
        if str(args.backend) == "modal":
            remote_artifacts = _remote_artifacts(CHECKPOINT_VOLUME, run_id)
            launch_command = list(row["modal_command"])
            if remote_artifacts:
                launch_status = "remote_artifacts_already_present"
            else:
                launch_result = _run_logged_command(
                    cmd=launch_command,
                    log_path=log_path,
                    env=modal_env,
                )
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
        else:
            launch_command = local_command
            if _required_files_present(run_dir, required_files):
                launch_status = "local_results_already_present"
            else:
                launch_result = _run_logged_command(
                    cmd=launch_command,
                    log_path=log_path,
                    env=local_env,
                )
                if int(launch_result["returncode"]) == 0 and _required_files_present(run_dir, required_files):
                    launch_status = "local_completed"
                else:
                    launch_status = "local_failed"
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
                "launch_backend": str(args.backend),
                "launch_command": launch_command,
                "local_command": local_command,
                "launch_log": str(log_path),
                "launch_status": launch_status,
                "remote_artifacts": remote_artifacts,
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    _write_manifest(manifest_path, launched)
    if str(args.backend) == "local" and not launch_failures:
        _aggregate_local_results(out_dir)
    if launch_failures:
        raise SystemExit(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "backend": str(args.backend),
                    "n_launched": len(launched) - len(launch_failures),
                    "n_failed": len(launch_failures),
                    "failures": launch_failures,
                    "manifest": str(manifest_path),
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "backend": str(args.backend),
                "device": str(args.device),
                "n_launched": len(launched),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
