#!/usr/bin/env python
"""Launch 8-condition factorized multi-output ablation on 7-allele panel.

Tests factorized assay embeddings, pretraining, capacity, loss mode,
target encoding, and pooled_single_output (negative control) on the
7-allele / ~44K-row contract.

Usage:
    python scripts/benchmark_factorized_ablation.py --dry-run
    python scripts/benchmark_factorized_ablation.py
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

# Short aliases for residual modes
A07 = "shared_base_factorized_context_plus_segment_residual"
A03 = "shared_base_segment_residual"
POOLED = "pooled_single_output"


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    description: str
    extra_args: Tuple[str, ...]
    uses_pretrain: bool


# --------------------------------------------------------------------------- #
#  8 conditions — see plan for rationale and pairwise comparisons
# --------------------------------------------------------------------------- #

DESIGNS: Tuple[DesignSpec, ...] = (
    # C1: Expected winner — full-featured d=128, A07, full loss, pretrained, mhcflurry
    DesignSpec(
        "C1",
        "d128 A07(fac+seg) full pretrain=yes enc=mhcflurry",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
        ),
        uses_pretrain=True,
    ),
    # C2: Factorized ablation (C1 vs C2) — segment-only, no factorized context
    DesignSpec(
        "C2",
        "d128 A03(seg only) full pretrain=yes enc=mhcflurry",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A03,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
        ),
        uses_pretrain=True,
    ),
    # C3: Pretraining ablation (C1 vs C3) — same as C1 but cold-start
    DesignSpec(
        "C3",
        "d128 A07(fac+seg) full pretrain=no enc=mhcflurry",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
        ),
        uses_pretrain=False,
    ),
    # C4: Loss mode ablation (C1 vs C4) — assay_heads_only
    DesignSpec(
        "C4",
        "d128 A07(fac+seg) assay_heads_only pretrain=yes enc=mhcflurry",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "assay_heads_only",
            "--affinity-target-encoding", "mhcflurry",
        ),
        uses_pretrain=True,
    ),
    # C5: Encoding ablation (C1 vs C5) — log10 target encoding
    DesignSpec(
        "C5",
        "d128 A07(fac+seg) full pretrain=yes enc=log10",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "log10",
        ),
        uses_pretrain=True,
    ),
    # C6: Capacity ablation (C1 vs C6) — d=32 cold-start (pretrain inert at d=32)
    DesignSpec(
        "C6",
        "d32 A07(fac+seg) full pretrain=no enc=mhcflurry",
        (
            "--d-model", "32",
            "--n-layers", "2",
            "--n-heads", "4",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
        ),
        uses_pretrain=False,
    ),
    # C7: Small + no factorized (C6 vs C7)
    DesignSpec(
        "C7",
        "d32 A03(seg only) full pretrain=no enc=mhcflurry",
        (
            "--d-model", "32",
            "--n-layers", "2",
            "--n-heads", "4",
            "--affinity-assay-residual-mode", A03,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
        ),
        uses_pretrain=False,
    ),
    # C8: Negative control — pooled_single expected to collapse
    DesignSpec(
        "C8",
        "d128 pooled_single full pretrain=yes enc=mhcflurry [NEG CTRL]",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", POOLED,
            "--affinity-loss-mode", "full",
            "--affinity-target-encoding", "mhcflurry",
        ),
        uses_pretrain=True,
    ),
)


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260316a"


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
        "--lr", "1e-3",
        "--lr-schedule", "constant",
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
    ]

    print(f"\n{'='*60}")
    print(f"  {design.design_id}: {design.description}")
    print(f"  run_id: {run_id}")
    print(f"  pretrain: {design.uses_pretrain}")
    print(f"{'='*60}")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        return {
            "run_id": run_id,
            "design_id": design.design_id,
            "description": design.description,
            "app_id": None,
            "command": cmd,
            "extra_args": extra_args,
            "returncode": None,
            "launch_output": "[DRY RUN]",
            "launch_log": None,
            "uses_pretrain": design.uses_pretrain,
        }

    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    log_path.write_text(output, encoding="utf-8")
    match = APP_ID_PATTERN.search(output)
    result = {
        "run_id": run_id,
        "design_id": design.design_id,
        "description": design.description,
        "app_id": match.group(0) if match else None,
        "command": cmd,
        "extra_args": extra_args,
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
        description="Launch 8-condition factorized ablation on 7-allele panel"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="fac-ablation")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching")
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
            "d_model": "varies (128 or 32)",
            "n_layers": 2,
            "n_heads": 4,
            "lr": 1e-3,
            "lr_schedule": "constant",
            "weight_decay": 0.01,
            "max_affinity_nM": 100000,
            "kd_grouping_mode": "split_kd_proxy",
            "synthetic_negatives": False,
            "ranking_losses": False,
        },
        "tested": [
            {
                "design_id": d.design_id,
                "description": d.description,
                "extra_args": list(d.extra_args),
                "uses_pretrain": d.uses_pretrain,
            }
            for d in DESIGNS
        ],
        "questions": {
            "factorized_helps": "C1 vs C2, C6 vs C7",
            "pretraining_helps": "C1 vs C3",
            "capacity_d128_vs_d32": "C1 vs C6",
            "full_vs_assay_heads_only": "C1 vs C4",
            "mhcflurry_vs_log10": "C1 vs C5",
            "pooled_collapses": "C8 vs all",
        },
        "modal_gpu": "H100!",
    }

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="factorized-ablation-7allele",
        title="Factorized Multi-Output Ablation (8 conditions, 7-allele, 50ep)",
        source_script="scripts/benchmark_factorized_ablation.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    print(f"Experiment dir: {out_dir}")
    print(f"Conditions: {len(DESIGNS)}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    if args.dry_run:
        print("[DRY RUN MODE]")

    launches: List[Dict[str, Any]] = []
    for design in DESIGNS:
        launches.append(
            _launch_design(
                design=design,
                alleles=alleles,
                probes=probes,
                warm_start=str(args.warm_start),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                prefix=str(args.prefix),
                out_dir=out_dir,
                dry_run=bool(args.dry_run),
            )
        )
        if not args.dry_run:
            time.sleep(0.2)

    manifest = {
        "experiment_dir": str(out_dir),
        "alleles": alleles,
        "probes": probes,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "warm_start": str(args.warm_start),
        "modal_gpu": "H100!",
        "designs": launches,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write variants summary
    lines = ["# Factorized Ablation — 8 Conditions", ""]
    for row in launches:
        lines.extend([
            f"## {row['design_id']}: {row['description']}",
            "",
            f"- App: `{row.get('app_id')}`",
            f"- Run: `{row['run_id']}`",
            f"- Pretrain: {row.get('uses_pretrain')}",
            f"- Launch log: `{row.get('launch_log')}`",
            "",
        ])
    lines.extend([
        "## Pairwise Comparisons",
        "",
        "| Question | Comparison | Expected |",
        "|----------|------------|----------|",
        "| Factorized helps? | C1 vs C2, C6 vs C7 | Modest improvement |",
        "| Pretraining helps? | C1 vs C3 | Yes, ~0.01-0.03 Spearman |",
        "| d=128 vs d=32? | C1 vs C6 | d=128 wins |",
        "| full vs assay_heads_only? | C1 vs C4 | full slightly better |",
        "| mhcflurry vs log10? | C1 vs C5 | mhcflurry slightly better |",
        "| pooled collapses? | C8 vs all | Yes, Spearman ~0.02 |",
        "",
    ])
    (out_dir / "variants.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "event": "factorized_ablation_launched",
        "out_dir": str(out_dir),
        "n_designs": len(launches),
        "dry_run": bool(args.dry_run),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
