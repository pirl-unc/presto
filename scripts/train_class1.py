#!/usr/bin/env python
"""Train full MHC Class I binding model using the L2 recipe.

Unified launcher for Modal (GPU), MPS (Apple Silicon), or CPU.

L2 = dag_prep_readout_leaf + assay_heads_only + lr=3e-4 warmup_cosine
     + d128 + pretrained (7-allele bakeoff winner)

Usage:
    # Local MPS (Apple Silicon)
    python scripts/train_class1.py --backend mps

    # Local CPU
    python scripts/train_class1.py --backend cpu --epochs 3

    # Modal H100 (detached)
    python scripts/train_class1.py --backend modal

    # Dry run (print command only)
    python scripts/train_class1.py --backend mps --dry-run

    # Override defaults
    python scripts/train_class1.py --backend mps --epochs 10 --seed 43 --no-pretrain
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

# L2 recipe defaults
DEFAULTS = {
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "residual_mode": "dag_prep_readout_leaf",
    "loss_mode": "assay_heads_only",
    "target_encoding": "mhcflurry",
    "lr": "3e-4",
    "lr_schedule": "warmup_cosine",
    "weight_decay": 0.01,
    "epochs": 50,
    "batch_size": 256,
    "seed": 42,
    "split_seed": 42,
}

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

# Pretrain checkpoint locations (checked in order)
CHECKPOINT_PATHS = [
    REPO_ROOT / "modal_runs" / "pulls" / "mhc-pretrain-20260308b" / "mhc_pretrain.pt",
    REPO_ROOT / "modal_runs" / "mhc-pretrain-20260308b" / "mhc_pretrain.pt",
    Path("/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"),  # Modal volume
]

APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")


def _find_checkpoint() -> Optional[Path]:
    for p in CHECKPOINT_PATHS:
        if p.exists():
            return p
    return None


def _build_extra_args(args: argparse.Namespace) -> List[str]:
    """Build the training args common to all backends."""
    extra = [
        "--alleles", ",".join(PROBE_ALLELES),
        "--train-all-alleles",
        "--train-mhc-class-filter", "I",
        "--probe-peptide", PROBE_PEPTIDES[0],
        "--extra-probe-peptides", ",".join(PROBE_PEPTIDES[1:]),
        "--d-model", str(args.d_model),
        "--n-layers", str(DEFAULTS["n_layers"]),
        "--n-heads", str(DEFAULTS["n_heads"]),
        "--affinity-assay-residual-mode", args.residual_mode,
        "--affinity-loss-mode", args.loss_mode,
        "--affinity-target-encoding", DEFAULTS["target_encoding"],
        "--lr", str(args.lr),
        "--lr-schedule", args.lr_schedule,
        "--weight-decay", str(DEFAULTS["weight_decay"]),
        "--seed", str(args.seed),
        "--split-seed", str(args.split_seed),
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
        "--probe-plot-frequency", str(args.probe_plot_frequency),
        "--design-id", args.design_id,
    ]

    # Checkpoint
    if not args.no_pretrain:
        if args.backend == "modal":
            extra.extend(["--init-checkpoint", "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"])
        else:
            ckpt = _find_checkpoint()
            if ckpt:
                extra.extend(["--init-checkpoint", str(ckpt)])
                print(f"Using pretrain checkpoint: {ckpt}")
            else:
                print("WARNING: Pretrain checkpoint not found — running cold start")
                print(f"  Searched: {[str(p) for p in CHECKPOINT_PATHS[:2]]}")

    return extra


def _launch_local(args: argparse.Namespace, extra_args: List[str]) -> None:
    """Launch training locally (MPS or CPU)."""
    merged_tsv = REPO_ROOT / "data" / "merged_deduped.tsv"
    index_csv = REPO_ROOT / "data" / "mhc_index.csv"
    if not merged_tsv.exists():
        print(f"ERROR: {merged_tsv} not found. Run: python -m presto data merge --datadir data")
        sys.exit(1)
    if not index_csv.exists():
        print(f"ERROR: {index_csv} not found.")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "focused_binding_probe.py"),
        "--device", args.backend,
        "--mps-safe-mode", "auto",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--merged-tsv", str(merged_tsv),
        "--index-csv", str(index_csv),
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"  Class I Training — {args.backend.upper()}")
    print(f"  epochs={args.epochs}  seed={args.seed}  d_model={args.d_model}")
    print(f"  lr={args.lr}  schedule={args.lr_schedule}")
    print(f"  residual={args.residual_mode}")
    print(f"  loss={args.loss_mode}")
    print(f"  pretrain={'no' if args.no_pretrain else 'yes'}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("  [DRY RUN] " + " ".join(cmd))
        return

    os.execv(sys.executable, cmd)


def _launch_modal(args: argparse.Namespace, extra_args: List[str]) -> None:
    """Launch training on Modal."""
    run_id = args.run_id or f"class1-L2-e{args.epochs}-s{args.seed}"

    cmd = [
        "modal", "run", "--detach",
        "scripts/train_modal.py::focused_binding_run",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--run-id", run_id,
        "--extra-args", " ".join(extra_args),
    ]

    print(f"\n{'='*60}")
    print(f"  Class I Training — Modal (H100)")
    print(f"  run_id={run_id}")
    print(f"  epochs={args.epochs}  seed={args.seed}  d_model={args.d_model}")
    print(f"  lr={args.lr}  schedule={args.lr_schedule}")
    print(f"  residual={args.residual_mode}")
    print(f"  loss={args.loss_mode}")
    print(f"  pretrain={'no' if args.no_pretrain else 'yes'}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("  [DRY RUN] " + " ".join(cmd))
        return

    result = subprocess.run(cmd, text=True, capture_output=True, env=os.environ.copy(), check=False)
    output = (result.stdout or "") + (result.stderr or "")
    print(output)
    match = APP_ID_PATTERN.search(output)
    if match:
        print(f"  app_id: {match.group(0)}")
    if result.returncode != 0:
        print(f"  WARNING: modal exited with code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train full MHC Class I binding model (L2 recipe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=["mps", "cpu", "modal"], required=True,
                        help="Training backend: mps (Apple Silicon), cpu, or modal (H100)")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--split-seed", type=int, default=DEFAULTS["split_seed"])
    parser.add_argument("--d-model", type=int, default=DEFAULTS["d_model"])
    parser.add_argument("--lr", type=str, default=DEFAULTS["lr"])
    parser.add_argument("--lr-schedule", type=str, default=DEFAULTS["lr_schedule"])
    parser.add_argument("--residual-mode", type=str, default=DEFAULTS["residual_mode"])
    parser.add_argument("--loss-mode", type=str, default=DEFAULTS["loss_mode"])
    parser.add_argument("--no-pretrain", action="store_true", help="Skip pretrain checkpoint")
    parser.add_argument("--run-id", type=str, default="", help="Run ID (modal only)")
    parser.add_argument("--design-id", type=str, default="L2-class1",
                        help="Design ID tag for the run")
    parser.add_argument("--probe-plot-frequency", type=str, default="off")
    parser.add_argument("--dry-run", action="store_true", help="Print command without launching")
    args = parser.parse_args()

    extra_args = _build_extra_args(args)

    if args.backend == "modal":
        _launch_modal(args, extra_args)
    else:
        _launch_local(args, extra_args)


if __name__ == "__main__":
    main()
