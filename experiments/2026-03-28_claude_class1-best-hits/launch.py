#!/usr/bin/env python
"""Full Class I best-hits bakeoff: 6 conditions x 3 seeds = 18 runs.

Tests the top architectural variants from the 7-allele bakeoff at full
MHC class I scale (~105 HLA alleles, ~250K+ rows). All conditions use
the proven stable lr=3e-4 warmup_cosine recipe. mhcseqs groove sequences
are used by default (installed on the Modal image).

Questions:
  F1 vs F2: Does DAG still beat A07 at full scale?
  F1 vs F3: Does full loss work better with more data?
  F1 vs F4: Does pretraining matter at 105 alleles?
  F5:       Cold-start + full loss stress test
  F1 vs F6: 3-way A07/A03/DAG architecture comparison

Usage:
    python experiments/2026-03-28_claude_class1-best-hits/launch.py --dry-run
    python experiments/2026-03-28_claude_class1-best-hits/launch.py
    python experiments/2026-03-28_claude_class1-best-hits/launch.py --skip-launched
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment_registry import default_agent_label, initialize_experiment_dir


PROBE_ALLELES = (
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02",
)
PROBE_PEPTIDES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI")
WARM_START = "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")

DAG = "dag_prep_readout_leaf"
A07 = "shared_base_factorized_context_plus_segment_residual"
A03 = "shared_base_segment_residual"


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    description: str
    extra_args: Tuple[str, ...]
    uses_pretrain: bool


DESIGNS: Tuple[DesignSpec, ...] = (
    # F1: Control — L2 (7-allele bakeoff winner) at full scale
    DesignSpec(
        "F1",
        "d128 DAG heads_only pretrain lr=3e-4 warmup [ctrl=L2]",
        (
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "assay_heads_only",
        ),
        uses_pretrain=True,
    ),
    # F2: DAG vs A07 at full scale
    DesignSpec(
        "F2",
        "d128 A07 heads_only pretrain lr=3e-4 warmup",
        (
            "--affinity-assay-residual-mode", A07,
            "--affinity-loss-mode", "assay_heads_only",
        ),
        uses_pretrain=True,
    ),
    # F3: full loss with more data — does probe degradation still happen?
    DesignSpec(
        "F3",
        "d128 DAG full pretrain lr=3e-4 warmup",
        (
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "full",
        ),
        uses_pretrain=True,
    ),
    # F4: pretraining effect at 105 alleles
    DesignSpec(
        "F4",
        "d128 DAG heads_only no-pretrain lr=3e-4 warmup",
        (
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "assay_heads_only",
        ),
        uses_pretrain=False,
    ),
    # F5: cold-start + full loss stress test
    DesignSpec(
        "F5",
        "d128 DAG full no-pretrain lr=3e-4 warmup",
        (
            "--affinity-assay-residual-mode", DAG,
            "--affinity-loss-mode", "full",
        ),
        uses_pretrain=False,
    ),
    # F6: A03 for 3-way architecture comparison
    DesignSpec(
        "F6",
        "d128 A03 heads_only pretrain lr=3e-4 warmup",
        (
            "--affinity-assay-residual-mode", A03,
            "--affinity-loss-mode", "assay_heads_only",
        ),
        uses_pretrain=True,
    ),
)


def _run_id(prefix: str, design_id: str, seed: int) -> str:
    return f"{prefix}-{design_id.lower()}-s{seed}-20260328a"


def _common_args() -> List[str]:
    return [
        "--alleles", ",".join(PROBE_ALLELES),
        "--train-all-alleles",
        "--train-mhc-class-filter", "I",
        "--probe-peptide", PROBE_PEPTIDES[0],
        "--extra-probe-peptides", ",".join(PROBE_PEPTIDES[1:]),
        "--d-model", "128",
        "--affinity-target-encoding", "mhcflurry",
        "--lr", "3e-4",
        "--lr-schedule", "warmup_cosine",
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


def _launch(
    *,
    design: DesignSpec,
    seed: int,
    split_seed: int,
    warm_start: str,
    epochs: int,
    batch_size: int,
    prefix: str,
    out_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    run_id = _run_id(prefix, design.design_id, seed)
    extra_args = _common_args()
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
        description="Launch full class I best-hits bakeoff (6 conditions x 3 seeds)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warm-start", type=str, default=WARM_START)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="class1")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-launched", action="store_true",
                        help="Skip runs with app_id in existing manifest")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    metadata = {
        "dataset_contract": {
            "probe_alleles": list(PROBE_ALLELES),
            "train_all_alleles": True,
            "train_mhc_class_filter": "I",
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "probe_peptides": list(PROBE_PEPTIDES),
            "split_seed": int(args.split_seed),
            "sequence_resolution": "mhcseqs_first_with_index_fallback",
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "warm_start": str(args.warm_start),
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "lr": "3e-4",
            "lr_schedule": "warmup_cosine",
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
            "Q1_dag_vs_a07_at_scale": "F1 vs F2",
            "Q2_full_vs_heads_only_at_scale": "F1 vs F3",
            "Q3_pretraining_at_105_alleles": "F1 vs F4",
            "Q4_cold_start_stress_test": "F5",
            "Q5_3way_architecture": "F1 vs F2 vs F6 (DAG vs A07 vs A03)",
        },
        "modal_gpu": "H100!",
        "prior_experiment": "2026-03-17_1042_claude_7allele-bakeoff",
    }

    out_dir = Path(args.out_dir)
    if not (out_dir / "README.md").exists():
        initialize_experiment_dir(
            out_dir=str(out_dir),
            slug="class1-best-hits",
            title="Full Class I Best Hits (6 conditions x 3 seeds, mhcseqs grooves)",
            source_script="experiments/2026-03-28_claude_class1-best-hits/launch.py",
            agent_label=str(args.agent_label),
            metadata=metadata,
        )

    # Load previously launched runs if resuming
    already_launched: dict = {}
    manifest_path = out_dir / "manifest.json"
    prev_designs: list = []
    if args.skip_launched and manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        prev_designs = prev.get("designs", [])
        for d in prev_designs:
            if d.get("app_id"):
                already_launched[d["run_id"]] = d
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
                launches.append(already_launched[run_id])
                print(f"  SKIP {run_id} (already launched: {already_launched[run_id]['app_id']})")
                continue
            launches.append(_launch(
                design=design, seed=seed, split_seed=int(args.split_seed),
                warm_start=str(args.warm_start), epochs=int(args.epochs),
                batch_size=int(args.batch_size), prefix=str(args.prefix),
                out_dir=out_dir, dry_run=bool(args.dry_run),
            ))
            if not args.dry_run:
                time.sleep(0.2)

    manifest = {
        "experiment_dir": str(out_dir),
        "probe_alleles": list(PROBE_ALLELES),
        "probe_peptides": list(PROBE_PEPTIDES),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "warm_start": str(args.warm_start),
        "modal_gpu": "H100!",
        "seeds": seeds,
        "split_seed": int(args.split_seed),
        "n_conditions": len(DESIGNS),
        "n_total_runs": n_total,
        "sequence_resolution": "mhcseqs_first_with_index_fallback",
        "designs": launches,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write variants summary
    lines = [f"# Class I Best Hits — {len(DESIGNS)} Conditions x {len(seeds)} Seeds = {n_total} Runs", ""]
    for design in DESIGNS:
        design_launches = [r for r in launches if r["design_id"] == design.design_id]
        lines.extend([f"## {design.design_id}: {design.description}", ""])
        for row in design_launches:
            lines.append(f"- seed={row['seed']}: `{row['run_id']}` app=`{row.get('app_id')}`")
        lines.append("")

    lines.extend([
        "## Questions", "",
        "| # | Question | Comparison |",
        "|---|----------|------------|",
        "| Q1 | DAG vs A07 at full scale? | F1 vs F2 |",
        "| Q2 | full vs heads_only with more data? | F1 vs F3 |",
        "| Q3 | Pretraining effect at 105 alleles? | F1 vs F4 |",
        "| Q4 | Cold-start + full loss stress test | F5 |",
        "| Q5 | 3-way architecture (DAG vs A07 vs A03) | F1 vs F2 vs F6 |",
        "",
    ])
    (out_dir / "variants.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "event": "class1_best_hits_launched",
        "out_dir": str(out_dir),
        "n_conditions": len(DESIGNS),
        "n_seeds": len(seeds),
        "n_total": n_total,
        "dry_run": bool(args.dry_run),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
