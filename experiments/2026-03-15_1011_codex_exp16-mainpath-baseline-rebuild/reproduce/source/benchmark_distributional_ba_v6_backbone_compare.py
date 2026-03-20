#!/usr/bin/env python
"""Launch the v6 factorial with an encoder-backbone comparison axis.

This reruns the actual executable EXP-16 contract on the shared main path:
2 alleles, broad numeric measurement profile, no warm start, 50 epochs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from experiment_registry import default_agent_label, initialize_experiment_dir


APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")
BACKBONES = ("historical_ablation", "groove")


def _launch_condition(
    *,
    cond_id: int,
    content_conditioned: bool,
    encoder_backbone: str,
    epochs: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
) -> Dict[str, Any]:
    cc_tag = "cc1" if content_conditioned else "cc0"
    short_backend = "ablation" if encoder_backbone == "historical_ablation" else encoder_backbone
    run_id = f"{prefix}-{short_backend}-c{cond_id:02d}-{cc_tag}"
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
        "--measurement-profile numeric_no_qualitative --alleles HLA-A*02:01,HLA-A*24:02",
    ]
    if content_conditioned:
        cmd.append("--content-conditioned")

    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )

    app_id: str | None = None
    output = ""
    for _ in range(20):
        try:
            output = log_path.read_text()
        except FileNotFoundError:
            output = ""
        match = APP_ID_PATTERN.search(output)
        if match is not None:
            app_id = match.group(0)
            break
        time.sleep(0.5)

    return {
        "run_id": run_id,
        "cond_id": cond_id,
        "content_conditioned": content_conditioned,
        "encoder_backbone": encoder_backbone,
        "app_id": app_id,
        "command": cmd,
        "launch_log": str(log_path),
        "launcher_pid": proc.pid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch v6 factorial with historical-ablation vs groove backbones",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prefix", type=str, default="dist-ba-v6-mainpath")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tested = []
    for encoder_backbone in BACKBONES:
        for content_conditioned in (False, True):
            for cond_id in range(1, 17):
                tested.append({
                    "cond_id": cond_id,
                    "content_conditioned": content_conditioned,
                    "encoder_backbone": encoder_backbone,
                })

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="exp16-mainpath-baseline-rebuild",
        title="EXP-16 Main-Path Baseline Rebuild",
        source_script="scripts/benchmark_distributional_ba_v6_backbone_compare.py",
        agent_label=str(args.agent_label),
        metadata={
            "dataset_contract": {
                "panel": ["HLA-A*02:01", "HLA-A*24:02"],
                "measurement_profile": "numeric_no_qualitative",
                "qualifier_filter": "all",
                "split": "peptide_group_80_10_10_seed42",
                "source": "data/merged_deduped.tsv",
            },
            "training": {
                "config_version": "v6",
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": "1e-3",
                "weight_decay": 0.01,
                "seed": 42,
                "warm_start": False,
                "gpu": os.environ.get("PRESTO_MODAL_GPU", "H100!"),
            },
            "tested": tested,
        },
    )

    if args.dry_run:
        print(json.dumps({
            "conditions": len(tested),
            "out_dir": str(out_dir),
            "tested": tested[:4],
        }, indent=2, sort_keys=True))
        return

    manifest: List[Dict[str, Any]] = []
    for encoder_backbone in BACKBONES:
        for content_conditioned in (False, True):
            for cond_id in range(1, 17):
                result = _launch_condition(
                    cond_id=cond_id,
                    content_conditioned=content_conditioned,
                    encoder_backbone=encoder_backbone,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    prefix=str(args.prefix),
                    out_dir=out_dir,
                )
                manifest.append(result)
                print(json.dumps({
                    "event": "launched",
                    "cond_id": cond_id,
                    "content_conditioned": content_conditioned,
                    "encoder_backbone": encoder_backbone,
                    "app_id": result.get("app_id"),
                    "run_id": result["run_id"],
                }, sort_keys=True), flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nManifest: {manifest_path}")
    print(f"Conditions launched: {len(manifest)}")


if __name__ == "__main__":
    main()
