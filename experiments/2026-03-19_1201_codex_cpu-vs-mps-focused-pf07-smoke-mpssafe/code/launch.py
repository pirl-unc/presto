#!/usr/bin/env python
"""Launch a matched local CPU vs MPS focused PF07 smoke comparison."""

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


DEFAULT_ALLELES = ("HLA-A*02:01", "HLA-A*24:02")
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI")
DEFAULT_PREFIX = "presto-focused-device-smoke-20260319b"
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 1
DEFAULT_MAX_RECORDS = 200
DEFAULT_SEED = 43
DEFAULT_SPLIT_SEED = 42
DEFAULT_LOCAL_INIT_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep"
    / "results"
    / "pretrains"
    / "mhc-pretrain-d32-20260317a-e01"
    / "mhc_pretrain.pt"
)

VARIANTS = (
    {
        "condition_key": "cpu",
        "description": "Matched tiny focused PF07 smoke run on CPU",
        "device": "cpu",
    },
    {
        "condition_key": "mps",
        "description": "Matched tiny focused PF07 smoke run on Apple Silicon MPS",
        "device": "mps",
    },
)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _portable_path_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _build_command(
    *,
    run_dir: Path,
    run_id: str,
    device: str,
    data_dir: Path,
    init_checkpoint: Path,
    batch_size: int,
    epochs: int,
    max_records: int,
    seed: int,
    split_seed: int,
    probe_alleles: list[str],
    probes: list[str],
) -> list[str]:
    python_cmd = os.environ.get("PRESTO_LOCAL_PYTHON", "python")
    return [
        python_cmd,
        "-m",
        "presto.scripts.focused_binding_probe",
        "--data-dir",
        _portable_path_str(data_dir),
        "--out-dir",
        _portable_path_str(run_dir),
        "--source",
        "iedb",
        "--design-id",
        f"focused_device_smoke_{device}",
        "--alleles",
        ",".join(probe_alleles),
        "--probe-peptide",
        probes[0],
        "--extra-probe-peptides",
        ",".join(probes[1:]),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        str(device),
        "--mps-safe-mode",
        "auto",
        "--num-workers",
        "0",
        "--no-pin-memory",
        "--no-persistent-workers",
        "--matmul-precision",
        "default",
        "--measurement-profile",
        "numeric_no_qualitative",
        "--qualifier-filter",
        "all",
        "--val-fraction",
        "0.1",
        "--test-fraction",
        "0.1",
        "--max-records",
        str(max_records),
        "--seed",
        str(seed),
        "--split-seed",
        str(split_seed),
        "--d-model",
        "32",
        "--n-layers",
        "2",
        "--n-heads",
        "4",
        "--peptide-pos-mode",
        "concat_start_end_frac",
        "--groove-pos-mode",
        "concat_start_end_frac",
        "--binding-core-lengths",
        "8,9,10,11",
        "--binding-core-refinement",
        "shared",
        "--lr",
        "1e-3",
        "--lr-schedule",
        "constant",
        "--weight-decay",
        "0.01",
        "--affinity-loss-mode",
        "full",
        "--affinity-target-encoding",
        "mhcflurry",
        "--max-affinity-nm",
        "100000",
        "--affinity-assay-residual-mode",
        "dag_prep_readout_leaf",
        "--kd-grouping-mode",
        "split_kd_proxy",
        "--binding-kinetic-input-mode",
        "affinity_vec",
        "--binding-direct-segment-mode",
        "off",
        "--train-mhc-class-filter",
        "I",
        "--train-all-alleles",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight",
        "0",
        "--binding-peptide-contrastive-weight",
        "0",
        "--binding-kd-family-consistency-weight",
        "0",
        "--binding-proxy-cross-consistency-weight",
        "0",
        "--binding-output-consistency-beta",
        "0.25",
        "--probe-plot-frequency",
        "off",
        "--epoch-val-metrics-frequency",
        "1",
        "--init-checkpoint",
        _portable_path_str(init_checkpoint),
    ]


def _run_logged_command(*, cmd: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return int(completed.returncode)


def _required_files_present(run_dir: Path, required_files: list[str]) -> bool:
    return all((run_dir / name).exists() for name in required_files)


def _summary_diverged(run_dir: Path) -> bool | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        return None
    return bool(payload.get("diverged"))


def _aggregate_results(out_dir: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched local CPU vs MPS focused PF07 smoke comparison.")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--init-checkpoint", type=str, default=str(DEFAULT_LOCAL_INIT_CHECKPOINT))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--ft-prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    probe_alleles = [token.strip() for token in str(args.alleles).split(",") if token.strip()]
    probes = [token.strip().upper() for token in str(args.probes).split(",") if token.strip()]
    if not probe_alleles or not probes:
        raise SystemExit("At least one allele and one probe peptide are required")

    data_dir = Path(str(args.data_dir)).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    init_checkpoint = Path(str(args.init_checkpoint)).expanduser()
    if not init_checkpoint.is_absolute():
        init_checkpoint = (REPO_ROOT / init_checkpoint).resolve()
    if not init_checkpoint.exists():
        raise SystemExit(f"Missing init checkpoint: {init_checkpoint}")

    tested_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    required_files = [
        "summary.json",
        "epoch_metrics.csv",
        "epoch_metrics.json",
    ]
    for variant in VARIANTS:
        run_id = f"{args.ft_prefix}-{variant['condition_key']}-e{int(args.epochs):03d}-s{int(args.seed)}"
        run_dir = Path(str(args.out_dir)) / "results" / "runs" / run_id
        command = _build_command(
            run_dir=run_dir,
            run_id=run_id,
            device=str(variant["device"]),
            data_dir=data_dir,
            init_checkpoint=init_checkpoint,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            max_records=int(args.max_records),
            seed=int(args.seed),
            split_seed=int(args.split_seed),
            probe_alleles=probe_alleles,
            probes=probes,
        )
        tested = {
            "condition_key": variant["condition_key"],
            "description": variant["description"],
            "device": variant["device"],
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_records": int(args.max_records),
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "init_checkpoint": _portable_path_str(init_checkpoint),
        }
        tested_rows.append(tested)
        rows.append(
            {
                **tested,
                "run_id": run_id,
                "required_files": required_files,
                "local_command": command,
            }
        )

    metadata = {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "source_filter": "iedb",
            "train_mhc_class_filter": "I",
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "max_records": int(args.max_records),
            "split_policy": "peptide_group_80_10_10",
            "split_seed": int(args.split_seed),
            "train_seed": int(args.seed),
            "probe_alleles": probe_alleles,
            "probe_peptides": probes,
            "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"],
            "assay_selector_inputs_forbidden": True,
        },
        "training": {
            "pretraining": {
                "warm_start_checkpoint_local": _portable_path_str(init_checkpoint),
            },
            "downstream": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "devices": [variant["device"] for variant in VARIANTS],
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "affinity_assay_residual_mode": "dag_prep_readout_leaf",
                "affinity_target_encoding": "mhcflurry",
                "kd_grouping_mode": "split_kd_proxy",
                "matmul_precision": "default",
            },
        },
        "tested": tested_rows,
    }

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="cpu-vs-mps-focused-pf07-smoke",
        title="CPU vs MPS Focused PF07 Smoke Compare",
        source_script="experiments/2026-03-19_1007_codex_cpu-vs-mps-focused-pf07-smoke/code/launch.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )
    results_root = out_dir / "results" / "runs"
    results_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if args.dry_run:
        _write_manifest(manifest_path, rows)
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "n_conditions": len(rows),
                    "first_run": rows[0] if rows else None,
                },
                indent=2,
            )
        )
        return

    launch_failures: list[dict[str, Any]] = []
    launched_rows: list[dict[str, Any]] = []
    log_root = out_dir / "launch_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        run_dir = results_root / str(row["run_id"])
        log_path = log_root / f"{row['run_id']}.log"
        if _required_files_present(run_dir, list(row["required_files"])):
            launch_status = "local_diverged" if _summary_diverged(run_dir) else "local_results_already_present"
            returncode = 0
        else:
            returncode = _run_logged_command(cmd=list(row["local_command"]), log_path=log_path)
            if returncode == 0 and _required_files_present(run_dir, list(row["required_files"])):
                launch_status = "local_diverged" if _summary_diverged(run_dir) else "local_completed"
            else:
                launch_status = "local_failed"
        launched_rows.append(
            {
                **row,
                "launch_status": launch_status,
                "launch_log": str(log_path),
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        if launch_status == "local_failed":
            launch_failures.append(
                {
                    "run_id": row["run_id"],
                    "log_path": str(log_path),
                    "returncode": int(returncode),
                }
            )

    _write_manifest(manifest_path, launched_rows)
    _aggregate_results(out_dir)
    if launch_failures:
        raise SystemExit(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "n_failed": len(launch_failures),
                    "failures": launch_failures,
                    "manifest": str(manifest_path),
                },
                indent=2,
            )
        )
    print(json.dumps({"out_dir": str(out_dir), "n_completed": len(launched_rows), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
