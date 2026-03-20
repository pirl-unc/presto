#!/usr/bin/env python
"""Launch 13-condition x 3-seed bakeoff on 7-allele panel.

Systematically compares residual modes (A07, A03, dag_prep_readout_leaf),
loss modes (full, assay_heads_only), capacity (d128, d32), pretraining,
and lr/schedule combinations. Every condition uses a known-stable recipe.

Usage:
    python experiments/2026-03-17_1042_claude_7allele-bakeoff/launch.py --dry-run
    python experiments/2026-03-17_1042_claude_7allele-bakeoff/launch.py
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
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
DAG = "dag_prep_readout_leaf"


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    description: str
    extra_args: Tuple[str, ...]
    uses_pretrain: bool


# --------------------------------------------------------------------------- #
#  13 conditions — see plan for rationale and pairwise comparisons
# --------------------------------------------------------------------------- #

DESIGNS: Tuple[DesignSpec, ...] = (
    # B1: Control (=S1) — d128 A07 full pretrain lr=3e-4 warmup_cosine
    DesignSpec(
        "B1",
        "d128 A07 full pretrain lr=3e-4 warmup_cosine [ctrl=S1]",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
    # B2: Control (=S5/C4) — d128 A07 heads_only pretrain lr=1e-3 constant
    DesignSpec(
        "B2",
        "d128 A07 heads_only pretrain lr=1e-3 constant [ctrl=S5/C4]",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "assay_heads_only",
            "--lr", "1e-3",
            "--lr-schedule", "constant",
        ),
        uses_pretrain=True,
    ),
    # B3: Control (=C7) — d32 A03 full no-pretrain lr=1e-3 constant
    DesignSpec(
        "B3",
        "d32 A03 full no-pretrain lr=1e-3 constant [ctrl=C7]",
        (
            "--d-model", "32",
            "--n-layers", "2",
            "--n-heads", "4",
            "--affinity-assay-residual-mode", A03,
            "--affinity-loss-mode", "full",
            "--lr", "1e-3",
            "--lr-schedule", "constant",
        ),
        uses_pretrain=False,
    ),
    # D1: DAG variant of B1
    DesignSpec(
        "D1",
        "d128 DAG full pretrain lr=3e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "full",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
    # D2: DAG variant of B2
    DesignSpec(
        "D2",
        "d128 DAG heads_only pretrain lr=1e-3 constant",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "assay_heads_only",
            "--lr", "1e-3",
            "--lr-schedule", "constant",
        ),
        uses_pretrain=True,
    ),
    # D3: DAG variant of B3 (d32)
    DesignSpec(
        "D3",
        "d32 DAG full no-pretrain lr=1e-3 constant",
        (
            "--d-model", "32",
            "--n-layers", "2",
            "--n-heads", "4",
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "full",
            "--lr", "1e-3",
            "--lr-schedule", "constant",
        ),
        uses_pretrain=False,
    ),
    # A1: A03 variant of B1
    DesignSpec(
        "A1",
        "d128 A03 full pretrain lr=3e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A03,
            "--affinity-loss-mode", "full",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
    # A2: A03 variant of B2
    DesignSpec(
        "A2",
        "d128 A03 heads_only pretrain lr=1e-3 constant",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A03,
            "--affinity-loss-mode", "assay_heads_only",
            "--lr", "1e-3",
            "--lr-schedule", "constant",
        ),
        uses_pretrain=True,
    ),
    # L1: LR sweep — heads_only with lr=3e-4 warmup_cosine (vs B2 at lr=1e-3 constant)
    DesignSpec(
        "L1",
        "d128 A07 heads_only pretrain lr=3e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "assay_heads_only",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
    # L2: LR sweep — DAG heads_only with lr=3e-4 warmup_cosine (vs D2 at lr=1e-3 constant)
    DesignSpec(
        "L2",
        "d128 DAG heads_only pretrain lr=3e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "assay_heads_only",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=True,
    ),
    # P1: No pretrain variant of B1
    DesignSpec(
        "P1",
        "d128 A07 full no-pretrain lr=3e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=False,
    ),
    # P2: No pretrain variant of D1
    DesignSpec(
        "P2",
        "d128 DAG full no-pretrain lr=3e-4 warmup_cosine",
        (
            "--d-model", "128",
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "full",
            "--lr", "3e-4",
            "--lr-schedule", "warmup_cosine",
        ),
        uses_pretrain=False,
    ),
    # C1: A07 variant of B3 (d32)
    DesignSpec(
        "C1",
        "d32 A07 full no-pretrain lr=1e-3 constant",
        (
            "--d-model", "32",
            "--n-layers", "2",
            "--n-heads", "4",
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "full",
            "--lr", "1e-3",
            "--lr-schedule", "constant",
        ),
        uses_pretrain=False,
    ),
)


def _run_id(prefix: str, design_id: str, seed: int) -> str:
    return f"{prefix}-{design_id.lower()}-s{seed}-20260317a"


def _common_args(*, alleles: Sequence[str], probes: Sequence[str]) -> List[str]:
    args = [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--affinity-target-encoding", "mhcflurry",
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
    seed: int,
    split_seed: int,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
    epochs: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    run_id = _run_id(prefix, design.design_id, seed)
    extra_args = _common_args(alleles=alleles, probes=probes)
    extra_args.extend(design.extra_args)
    if design.uses_pretrain:
        extra_args.extend(["--init-checkpoint", warm_start])
    extra_args.extend([
        "--seed", str(seed),
        "--split-seed", str(split_seed),
        "--design-id", f"{design.design_id}-s{seed}",
    ])

    cmd = [
        "modal", "run", "--detach",
        "scripts/train_modal.py::focused_binding_run",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--run-id", run_id,
        "--extra-args", " ".join(extra_args),
    ]

    print(f"\n{'='*60}")
    print(f"  {design.design_id} seed={seed}: {design.description}")
    print(f"  run_id: {run_id}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] cmd: {' '.join(cmd)}")
        return {
            "run_id": run_id, "design_id": design.design_id, "seed": seed,
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
        "run_id": run_id, "design_id": design.design_id, "seed": seed,
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
        description="Launch 13-condition x 3-seed bakeoff on 7-allele panel"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated train seeds (split_seed always 42)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Fixed dataset split seed (default: 42)")
    parser.add_argument("--prefix", type=str, default="bakeoff")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    parser.add_argument("--skip-launched", action="store_true",
                        help="Skip runs that already have an app_id in manifest.json")
    args = parser.parse_args()

    alleles = [x.strip() for x in str(args.alleles).split(",") if x.strip()]
    probes = [x.strip().upper() for x in str(args.probes).split(",") if x.strip()]
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    metadata = {
        "dataset_contract": {
            "alleles": alleles,
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "probe_peptides": probes,
            "split_seed": int(args.split_seed),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "warm_start": str(args.warm_start),
            "d_model": "varies (128 or 32)",
            "n_layers": 2,
            "n_heads": 4,
            "weight_decay": 0.01,
            "max_affinity_nM": 100000,
            "kd_grouping_mode": "split_kd_proxy",
            "affinity_target_encoding": "mhcflurry",
            "synthetic_negatives": False,
            "contrastive": False,
            "binding_core_lengths": [8, 9, 10, 11],
            "binding_core_refinement": "shared",
            "seeds": seeds,
            "split_seed": int(args.split_seed),
        },
        "tested": [
            {"design_id": d.design_id, "description": d.description,
             "extra_args": list(d.extra_args), "uses_pretrain": d.uses_pretrain}
            for d in DESIGNS
        ],
        "questions": {
            "Q1_dag_vs_a07": "D1 vs B1 (d128 full), D2 vs B2, D3 vs C1",
            "Q2_full_vs_heads_only": "B1 vs B2, D1 vs D2, A1 vs A2",
            "Q3_d128_vs_d32": "B1 vs C1, D1 vs D3",
            "Q4_a07_vs_a03_vs_dag": "B1 vs A1 vs D1 (d128 full), B2 vs A2 vs D2",
            "Q5_pretraining_effect": "B1 vs P1, D1 vs P2",
            "Q6_lr_sweep_heads_only": "L1 vs B2, L2 vs D2",
        },
        "modal_gpu": "H100!",
        "prior_experiments": [
            "2026-03-16_1454_claude_factorized-ablation-7allele",
            "2026-03-16_1813_claude_lr-stability-7allele",
            "2026-03-16_2355_codex_pf07-assay-structured-dag-sweep",
        ],
    }

    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug="7allele-bakeoff",
        title="7-Allele Model Bakeoff (13 conditions x 3 seeds = 39 runs)",
        source_script="experiments/2026-03-17_1042_claude_7allele-bakeoff/launch.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    # Load previously launched runs if resuming
    already_launched: set = set()
    if args.skip_launched:
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            for d in prev.get("designs", []):
                if d.get("app_id"):
                    already_launched.add(d["run_id"])
            print(f"Skipping {len(already_launched)} already-launched runs")

    n_total = len(DESIGNS) * len(seeds)
    print(f"Experiment dir: {out_dir}")
    print(f"Conditions: {len(DESIGNS)}, Seeds: {seeds}, Total runs: {n_total}")
    if args.dry_run:
        print("[DRY RUN MODE]")

    launches: List[Dict[str, Any]] = []
    for design in DESIGNS:
        for seed in seeds:
            run_id = _run_id(str(args.prefix), design.design_id, seed)
            if run_id in already_launched:
                # Preserve the previous successful entry
                prev_entry = next(
                    d for d in prev["designs"]
                    if d["run_id"] == run_id
                )
                launches.append(prev_entry)
                print(f"  SKIP {run_id} (already launched: {prev_entry['app_id']})")
                continue
            launches.append(_launch_design(
                design=design, seed=seed, split_seed=int(args.split_seed),
                alleles=alleles, probes=probes,
                warm_start=str(args.warm_start), epochs=int(args.epochs),
                batch_size=int(args.batch_size), prefix=str(args.prefix),
                out_dir=out_dir, dry_run=bool(args.dry_run),
            ))
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
        "seeds": seeds,
        "split_seed": int(args.split_seed),
        "n_conditions": len(DESIGNS),
        "n_total_runs": n_total,
        "designs": launches,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write variants summary
    lines = [f"# 7-Allele Bakeoff — {len(DESIGNS)} Conditions x {len(seeds)} Seeds = {n_total} Runs", ""]
    for design in DESIGNS:
        design_launches = [r for r in launches if r["design_id"] == design.design_id]
        lines.extend([
            f"## {design.design_id}: {design.description}", "",
        ])
        for row in design_launches:
            lines.append(
                f"- seed={row['seed']}: `{row['run_id']}` app=`{row.get('app_id')}`"
            )
        lines.append("")

    lines.extend([
        "## Pairwise Comparisons", "",
        "| # | Question | Primary Comparison | Secondary |",
        "|---|----------|-------------------|-----------|",
        "| Q1 | DAG vs A07 on 7-allele? | D1 vs B1 (d128, full) | D2 vs B2, D3 vs C1 |",
        "| Q2 | full vs heads_only on test metrics? | B1 vs B2, D1 vs D2 | A1 vs A2 |",
        "| Q3 | d128 vs d32? | B1 vs C1, D1 vs D3 | |",
        "| Q4 | A07 vs A03 vs DAG (3-way)? | B1 vs A1 vs D1 (d128 full) | B2 vs A2 vs D2 |",
        "| Q5 | Pretraining effect? | B1 vs P1, D1 vs P2 | |",
        "| Q6 | lr sweep for heads_only? | L1 vs B2, L2 vs D2 | |",
        "",
    ])
    (out_dir / "variants.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "event": "bakeoff_launched",
        "out_dir": str(out_dir),
        "n_conditions": len(DESIGNS),
        "n_seeds": len(seeds),
        "n_total": n_total,
        "dry_run": bool(args.dry_run),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
