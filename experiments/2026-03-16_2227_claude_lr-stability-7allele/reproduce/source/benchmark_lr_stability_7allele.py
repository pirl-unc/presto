#!/usr/bin/env python
"""Launch lr/schedule sweep to stabilize d=128 + full loss on 7-allele panel.

Follow-up to the factorized ablation (2026-03-16_1454) which showed that
lr=1e-3 + d=128 + full loss diverges. Tests lower lr and warmup schedules.

Usage:
    python scripts/benchmark_lr_stability_7allele.py --dry-run
    python scripts/benchmark_lr_stability_7allele.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from experiment_registry import default_agent_label, initialize_experiment_dir


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
DEFAULT_WARM_START = "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")

A07 = "shared_base_factorized_context_plus_segment_residual"
A03 = "shared_base_segment_residual"


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    description: str
    extra_args: Tuple[str, ...]
    uses_pretrain: bool


# --------------------------------------------------------------------------- #
#  6 conditions — stabilize d=128 + full loss on 7-allele
# --------------------------------------------------------------------------- #

DESIGNS: Tuple[DesignSpec, ...] = (
    # S1: Main hypothesis — lr=3e-4, warmup_cosine should stabilize full loss
    DesignSpec(
        "S1",
        "d128 A07 full pretrain mhcflurry lr=3e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
    # S2: lr=3e-4 + onecycle
    DesignSpec(
        "S2",
        "d128 A07 full pretrain mhcflurry lr=3e-4 onecycle",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
            "--lr", "3e-4",
            "--lr-schedule", "onecycle",
        ),
        uses_pretrain=True,
    ),
    # S3: Lower lr — lr=1e-4, warmup_cosine
    DesignSpec(
        "S3",
        "d128 A07 full pretrain mhcflurry lr=1e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
            "--lr", "1e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
    # S4: Schedule-only fix — lr=1e-3 + warmup_cosine (does schedule alone fix it?)
    DesignSpec(
        "S4",
        "d128 A07 full pretrain mhcflurry lr=1e-3 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
            "--lr", "1e-3",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
    # S5: C4 control — assay_heads_only at lr=1e-3 constant (known stable baseline)
    DesignSpec(
        "S5",
        "d128 A07 heads_only pretrain mhcflurry lr=1e-3 constant [C4 ctrl]",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "assay_heads_only",
            "--affinity-target-encoding", "mhcflurry",
            "--lr", "1e-3",
            "--lr-schedule", "constant",
        ),
        uses_pretrain=True,
    ),
    # S6: A03 architecture at lr=3e-4 warmup_cosine (does architecture matter once stable?)
    DesignSpec(
        "S6",
        "d128 A03 full pretrain mhcflurry lr=3e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A03,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
)


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260316b"


def _common_args(*, alleles: Sequence[str], probes: Sequence[str]) -> List[str]:
    args = [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--peptide-pos-mode", "concat_start_end_frac",
        "--groove-pos-mode", "concat_start_end_frac",
        "--binding-core-lengths", "8,9,10,11",
        "--binding-core-refinement", "shared",
        "--kd-grouping-mode", "split_kd_proxy",
        "--max-affinity-nm", "100000",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
        "--probe-plot-frequency", "off",
        "--weight-decay", "0.01",
    ]
    if len(probes) > 1:
        args.extend(["--extra-probe-peptides", ",".join(probes[1:])])
    return args


def _launch_design(
    *,
    design: DesignSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
    epochs: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    run_id = _run_id(prefix, design.design_id)
    extra_args = _common_args(alleles=alleles, probes=probes)
    extra_args.extend(design.extra_args)
    if design.uses_pretrain:
        extra_args.extend(["--init-checkpoint", warm_start])
    extra_args.extend(["--design-id", design.design_id])

    cmd = [
        "modal", "run", "--detach",
        "scripts/train_modal.py::focused_binding_run",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--run-id", run_id,
        "--extra-args", " ".join(extra_args),
    ]

    print(f"\n{'='*60}")
    print(f"  {design.design_id}: {design.description}")
    print(f"  run_id: {run_id}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN]")
        return {
            "run_id": run_id, "design_id": design.design_id,
            "description": design.description, "app_id": None,
            "command": cmd, "extra_args": extra_args,
            "returncode": None, "launch_output": "[DRY RUN]",
            "launch_log": None, "uses_pretrain": design.uses_pretrain,
        }

    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, text=True, capture_output=True, env=os.environ.copy(), check=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    log_path.write_text(output, encoding="utf-8")
    match = APP_ID_PATTERN.search(output)
    result = {
        "run_id": run_id, "design_id": design.design_id,
        "description": design.description,
        "app_id": match.group(0) if match else None,
        "command": cmd, "extra_args": extra_args,
        "returncode": completed.returncode,
        "launch_output": output.strip(),
        "launch_log": str(log_path),
        "uses_pretrain": design.uses_pretrain,
    }
    if completed.returncode != 0:
        print(f"  WARNING: modal exited with code {completed.returncode}")
    else:
        print(f"  app_id: {result['app_id']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch lr/schedule sweep to stabilize d=128 + full loss on 7-allele panel"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="lr-stab")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alleles = [x.strip() for x in str(args.alleles).split(",") if x.strip()]
    probes = [x.strip().upper() for x in str(args.probes).split(",") if x.strip()]

    metadata = {
        "dataset_contract": {
            "alleles": alleles,
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "probe_peptides": probes,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "warm_start": str(args.warm_start),
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "weight_decay": 0.01,
            "max_affinity_nM": 100000,
            "kd_grouping_mode": "split_kd_proxy",
            "synthetic_negatives": False,
        },
        "tested": [
            {"design_id": d.design_id, "description": d.description,
             "extra_args": list(d.extra_args), "uses_pretrain": d.uses_pretrain}
            for d in DESIGNS
        ],
        "questions": {
            "lr_3e4_warmup_stabilizes": "S1 vs C1 (diverged at ep49)",
            "onecycle_vs_warmup": "S1 vs S2",
            "lr_1e4_too_conservative": "S3 vs S1",
            "schedule_alone_fixes": "S4 vs C1",
            "full_vs_heads_only_at_stability": "S1 vs S5",
            "A07_vs_A03_once_stable": "S1 vs S6",
        },
        "modal_gpu": "H100!",
        "prior_experiment": "2026-03-16_1454_claude_factorized-ablation-7allele",
    }

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="lr-stability-7allele",
        title="LR/Schedule Stability Sweep (d=128 full loss, 7-allele, 50ep)",
        source_script="scripts/benchmark_lr_stability_7allele.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    print(f"Experiment dir: {out_dir}")
    print(f"Conditions: {len(DESIGNS)}")
    if args.dry_run:
        print("[DRY RUN MODE]")

    launches: List[Dict[str, Any]] = []
    for design in DESIGNS:
        launches.append(_launch_design(
            design=design, alleles=alleles, probes=probes,
            warm_start=str(args.warm_start), epochs=int(args.epochs),
            batch_size=int(args.batch_size), prefix=str(args.prefix),
            out_dir=out_dir, dry_run=bool(args.dry_run),
        ))
        if not args.dry_run:
            time.sleep(0.2)

    manifest = {
        "experiment_dir": str(out_dir), "alleles": alleles, "probes": probes,
        "epochs": int(args.epochs), "batch_size": int(args.batch_size),
        "warm_start": str(args.warm_start), "modal_gpu": "H100!",
        "designs": launches,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# LR/Schedule Stability Sweep — 6 Conditions", ""]
    for row in launches:
        lines.extend([
            f"## {row['design_id']}: {row['description']}", "",
            f"- App: `{row.get('app_id')}`",
            f"- Run: `{row['run_id']}`",
            f"- Launch log: `{row.get('launch_log')}`", "",
        ])
    (out_dir / "variants.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "event": "lr_stability_launched", "out_dir": str(out_dir),
        "n_designs": len(launches), "dry_run": bool(args.dry_run),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
