#!/usr/bin/env python
"""Launch a PF07 output-tying weight sweep on the main Presto affinity path."""

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
DEFAULT_PREFIX = "presto-pf07-tiewt-20260316a"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 43
DEFAULT_SPLIT_SEED = 42
DEFAULT_KD_WEIGHTS = (0.0, 0.0025, 0.01, 0.04)
DEFAULT_PROXY_WEIGHTS = (0.0, 0.001, 0.004)
DEFAULT_BETA = 0.25


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


def _weight_token(value: float) -> str:
    token = f"{value:.4f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def _build_extra_args(
    *,
    alleles: list[str],
    probes: list[str],
    design_id: str,
    seed: int,
    split_seed: int,
    kd_weight: float,
    proxy_weight: float,
    beta: float,
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
        "--lr-schedule", "constant",
        "--weight-decay", "0.01",
        "--affinity-loss-mode", "full",
        "--affinity-target-encoding", "mhcflurry",
        "--max-affinity-nm", "100000",
        "--affinity-assay-residual-mode", "shared_base_factorized_context_plus_segment_residual",
        "--kd-grouping-mode", "split_kd_proxy",
        "--binding-kinetic-input-mode", "affinity_vec",
        "--binding-direct-segment-mode", "off",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--binding-kd-family-consistency-weight", str(kd_weight),
        "--binding-proxy-cross-consistency-weight", str(proxy_weight),
        "--binding-output-consistency-beta", str(beta),
        "--probe-plot-frequency", "off",
    ]


def _condition_rows(
    *,
    alleles: list[str],
    probes: list[str],
    seed: int,
    split_seed: int,
    kd_weights: list[float],
    proxy_weights: list[float],
    beta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kd_weight in kd_weights:
        for proxy_weight in proxy_weights:
            kd_token = _weight_token(kd_weight)
            proxy_token = _weight_token(proxy_weight)
            condition_key = f"pf07_kd{kd_token}_cross{proxy_token}"
            design_id = f"presto_pf07_tiewt_kd{kd_token}_cross{proxy_token}"
            rows.append(
                {
                    "condition_key": condition_key,
                    "design_id": design_id,
                    "description": (
                        "PF07 control with output tying "
                        f"(kd_family={kd_weight:g}, proxy_cross={proxy_weight:g}, beta={beta:g})"
                    ),
                    "lr": 1e-3,
                    "lr_schedule": "constant",
                    "binding_kd_family_consistency_weight": kd_weight,
                    "binding_proxy_cross_consistency_weight": proxy_weight,
                    "binding_output_consistency_beta": beta,
                }
            )
    for row in rows:
        row["extra_args"] = _build_extra_args(
            alleles=alleles,
            probes=probes,
            design_id=str(row["design_id"]),
            seed=seed,
            split_seed=split_seed,
            kd_weight=float(row["binding_kd_family_consistency_weight"]),
            proxy_weight=float(row["binding_proxy_cross_consistency_weight"]),
            beta=float(row["binding_output_consistency_beta"]),
        )
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
        "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
        "kd_grouping_mode": "split_kd_proxy",
        "affinity_target_encoding": "mhcflurry",
        "max_affinity_nM": 100000,
        "binding_kd_family_consistency_weight": float(condition["binding_kd_family_consistency_weight"]),
        "binding_proxy_cross_consistency_weight": float(condition["binding_proxy_cross_consistency_weight"]),
        "binding_output_consistency_beta": float(condition["binding_output_consistency_beta"]),
        "lr": float(condition["lr"]),
        "lr_schedule": condition["lr_schedule"],
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
    parser = argparse.ArgumentParser(
        description="Launch PF07 output-tying weight sweep on the main Presto affinity path."
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
    parser.add_argument(
        "--kd-family-weights",
        type=str,
        default=",".join(str(value) for value in DEFAULT_KD_WEIGHTS),
    )
    parser.add_argument(
        "--proxy-cross-weights",
        type=str,
        default=",".join(str(value) for value in DEFAULT_PROXY_WEIGHTS),
    )
    parser.add_argument("--consistency-beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alleles = [token.strip() for token in str(args.alleles).split(",") if token.strip()]
    probes = [token.strip().upper() for token in str(args.probes).split(",") if token.strip()]
    kd_weights = [float(token.strip()) for token in str(args.kd_family_weights).split(",") if token.strip()]
    proxy_weights = [float(token.strip()) for token in str(args.proxy_cross_weights).split(",") if token.strip()]
    if not probes:
        raise SystemExit("At least one probe peptide is required")
    if not kd_weights:
        raise SystemExit("At least one KD-family weight is required")
    if not proxy_weights:
        raise SystemExit("At least one proxy-cross weight is required")

    conditions = _condition_rows(
        alleles=alleles,
        probes=probes,
        seed=int(args.seed),
        split_seed=int(args.split_seed),
        kd_weights=kd_weights,
        proxy_weights=proxy_weights,
        beta=float(args.consistency_beta),
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
            "assay_selector_inputs_forbidden": True,
            "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"],
            "probe_peptides": probes,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "ranking_losses": False,
            "synthetic_negatives": False,
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
            "affinity_loss_mode": "full",
            "affinity_target_encoding": "mhcflurry",
            "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
            "kd_grouping_mode": "split_kd_proxy",
            "max_affinity_nM": 100000,
            "binding_output_consistency_beta": float(args.consistency_beta),
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
                "condition_key": row["condition_key"],
                "description": row["description"],
                "lr": row["lr"],
                "lr_schedule": row["lr_schedule"],
                "affinity_loss_mode": "full",
                "affinity_target_encoding": "mhcflurry",
                "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
                "kd_grouping_mode": "split_kd_proxy",
                "max_affinity_nM": 100000,
                "binding_kd_family_consistency_weight": row["binding_kd_family_consistency_weight"],
                "binding_proxy_cross_consistency_weight": row["binding_proxy_cross_consistency_weight"],
                "binding_output_consistency_beta": row["binding_output_consistency_beta"],
            }
            for row in conditions
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="pf07-output-tying-weight-sweep",
        title="PF07 Output-Tying Weight Sweep",
        source_script="experiments/2026-03-16_1621_codex_pf07-output-tying-weight-sweep/code/launch.py",
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
