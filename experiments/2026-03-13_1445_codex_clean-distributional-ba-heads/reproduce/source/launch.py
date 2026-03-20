#!/usr/bin/env python
"""Launch the self-contained clean distributional BA benchmark on Modal."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CODE_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(1, str(SCRIPTS_DIR))

from experiment_registry import default_agent_label, initialize_experiment_dir  # noqa: E402
from distributional_ba.config import CONDITIONS, ConditionSpec  # noqa: E402


DEFAULT_ALLELES = (
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02",
)
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI")
DEFAULT_GPU = "H100!"
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")


def _metadata(
    *,
    alleles: Sequence[str],
    probes: Sequence[str],
    epochs: int,
    batch_size: int,
    max_records: int,
    selected_conditions: Sequence[ConditionSpec],
) -> Dict[str, Any]:
    return {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "panel": list(alleles),
            "measurement_profile": "numeric_no_qualitative",
            "assay_families": ["IC50", "KD", "KD (~IC50)", "KD (~EC50)", "EC50"],
            "qualifier_filter": "all",
            "split": "peptide_group_80_10_10_seed42",
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": "1e-4",
            "schedule": "warmup_cosine",
            "weight_decay": 0.01,
            "seed": 42,
            "gpu": DEFAULT_GPU,
            "encoder": "FixedBackbone(embed=128,layers=2,heads=4,ff=128)",
            "probes": list(probes),
            "max_records": int(max_records),
        },
        "tested": [
            {
                "cond_id": spec.cond_id,
                "label": spec.label,
                "head_type": spec.head_type,
                "assay_mode": spec.assay_mode,
                "max_nM": spec.max_nM,
                "n_bins": spec.n_bins,
                "sigma_mult": spec.sigma_mult,
            }
            for spec in selected_conditions
        ],
    }


def _launch_condition(
    *,
    spec: ConditionSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    epochs: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
    max_records: int,
) -> Dict[str, Any]:
    run_id = f"{prefix}-c{spec.cond_id:02d}"
    extra_args_parts: List[str] = [
        "--alleles",
        ",".join(alleles),
        "--probe-peptide",
        probes[0],
        "--extra-probe-peptides",
        ",".join(probes[1:]),
        "--qualifier-filter",
        "all",
        "--embed-dim",
        "128",
        "--n-heads",
        "4",
        "--n-layers",
        "2",
        "--ff-dim",
        "128",
        "--lr",
        "1e-4",
        "--lr-schedule",
        "warmup_cosine",
        "--weight-decay",
        "0.01",
        "--seed",
        "42",
        "--warmup-fraction",
        "0.1",
        "--min-lr-scale",
        "0.1",
    ]
    if int(max_records) > 0:
        extra_args_parts.extend(["--max-records", str(int(max_records))])

    cmd = [
        "modal",
        "run",
        "--detach",
        "scripts/train_modal.py::distributional_ba_clean_run",
        "--cond-id",
        str(spec.cond_id),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--run-id",
        run_id,
        "--extra-args",
        " ".join(extra_args_parts),
    ]

    env = dict(os.environ)
    env.setdefault("PRESTO_MODAL_GPU", DEFAULT_GPU)

    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    app_id: str | None = None
    for _ in range(20):
        output = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        match = APP_ID_PATTERN.search(output)
        if match is not None:
            app_id = match.group(0)
            break
        time.sleep(0.5)

    return {
        "run_id": run_id,
        "cond_id": spec.cond_id,
        "label": spec.label,
        "head_type": spec.head_type,
        "assay_mode": spec.assay_mode,
        "max_nM": spec.max_nM,
        "n_bins": spec.n_bins,
        "sigma_mult": spec.sigma_mult,
        "command": cmd,
        "launch_log": str(log_path),
        "launcher_pid": proc.pid,
        "app_id": app_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the clean 12-condition BA head sweep on Modal")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="dist-ba-clean")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--cond-ids", type=str, default="", help="Optional comma-separated subset of condition IDs.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional dataset cap for smoke launches.")
    parser.add_argument("--smoke", action="store_true", help="Launch only condition 1 with 1 epoch and a small record cap.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alleles = [part.strip() for part in str(args.alleles).split(",") if part.strip()]
    probes = [part.strip() for part in str(args.probes).split(",") if part.strip()]

    selected_ids = set()
    if str(args.cond_ids).strip():
        for part in str(args.cond_ids).split(","):
            part = part.strip()
            if part:
                selected_ids.add(int(part))
    if args.smoke and not selected_ids:
        selected_ids = {1}

    selected_conditions = [spec for spec in CONDITIONS if not selected_ids or spec.cond_id in selected_ids]
    if selected_ids and not selected_conditions:
        raise ValueError(f"No conditions matched --cond-ids={args.cond_ids!r}")

    resolved_epochs = 1 if args.smoke else int(args.epochs)
    resolved_max_records = 4096 if args.smoke and int(args.max_records) <= 0 else int(args.max_records)
    resolved_prefix = f"{args.prefix}-smoke" if args.smoke else str(args.prefix)

    source_script = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="clean-distributional-ba-heads",
        title="Clean Distributional vs Regression BA Heads",
        source_script=source_script,
        agent_label=str(args.agent_label),
        metadata=_metadata(
            alleles=alleles,
            probes=probes,
            epochs=resolved_epochs,
            batch_size=int(args.batch_size),
            max_records=resolved_max_records,
            selected_conditions=selected_conditions,
        ),
    )

    if args.dry_run:
        manifest = {
            "conditions": len(selected_conditions),
            "condition_list": [{"cond_id": spec.cond_id, "label": spec.label} for spec in selected_conditions],
            "epochs": resolved_epochs,
            "batch_size": int(args.batch_size),
            "max_records": resolved_max_records,
            "out_dir": str(out_dir),
        }
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    manifest: List[Dict[str, Any]] = []
    for spec in selected_conditions:
        launched = _launch_condition(
            spec=spec,
            alleles=alleles,
            probes=probes,
            epochs=resolved_epochs,
            batch_size=int(args.batch_size),
            prefix=resolved_prefix,
            out_dir=out_dir,
            max_records=resolved_max_records,
        )
        manifest.append(launched)
        print(
            json.dumps(
                {
                    "event": "launched",
                    "cond_id": spec.cond_id,
                    "label": spec.label,
                    "run_id": launched["run_id"],
                    "app_id": launched.get("app_id"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"Conditions launched: {len(manifest)}")


if __name__ == "__main__":
    main()
