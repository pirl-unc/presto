#!/usr/bin/env python
"""Main training loop for distributional vs regression BA head experiment.

Usage:
    python -m presto.scripts.distributional_ba.train --cond-id 1 [options]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from presto.data import PrestoCollator, PrestoDataset, create_dataloader
from presto.data.tokenizer import Tokenizer
from presto.models.affinity import DEFAULT_MAX_AFFINITY_NM, normalize_binding_target_log10
from presto.scripts.focused_binding_probe import (
    DEFAULT_ALLELES,
    DEFAULT_PROBE_PEPTIDE,
    _keep_binding_qualifier,
    _keep_measurement_type,
    _load_binding_records_from_merged_tsv,
    _resolve_allele_sequences,
    _select_fit_supported_probe_peptides,
    _split_csv,
    _split_records_by_peptide,
    MEASUREMENT_PROFILE_NUMERIC,
    QUALIFIER_FILTERS,
)
from presto.scripts.groove_baseline_probe import _verify_groove_representations
from presto.scripts.train_iedb import resolve_mhc_sequences_from_index

from .config import CONDITIONS_BY_ID, ConditionSpec, DistributionalModel, build_model
from .evaluate import evaluate_held_out, evaluate_probe_panel

# Lazy import for v2 conditions — avoids import cost when using v1
_CONDITIONS_BY_VERSION = {
    "v1": lambda: CONDITIONS_BY_ID,
}


def _get_conditions_lookup(version: str):
    if version == "v2":
        from .config_v2 import CONDITIONS_V2_BY_ID
        return CONDITIONS_V2_BY_ID
    if version == "v3":
        from .config_v3 import CONDITIONS_V3_BY_ID
        return CONDITIONS_V3_BY_ID
    return CONDITIONS_BY_ID


# ---------------------------------------------------------------------------
# 3-way split
# ---------------------------------------------------------------------------

def _split_records_three_way(
    records: list,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    alleles: Optional[Sequence[str]] = None,
) -> Tuple[list, list, list, Dict[str, Any]]:
    """Peptide-group stratified 3-way split.

    First split off test (test_frac of total), then split remaining into
    train/val using adjusted fractions.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    # First: split off test set
    remaining, test_records, split1_stats = _split_records_by_peptide(
        records, val_fraction=test_frac, seed=seed + 100, alleles=alleles,
    )
    # Then: split remaining into train/val
    adjusted_val = val_frac / (train_frac + val_frac)
    train_records, val_records, split2_stats = _split_records_by_peptide(
        remaining, val_fraction=adjusted_val, seed=seed, alleles=alleles,
    )

    stats = {
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "test_rows": len(test_records),
        "split1": split1_stats,
        "split2": split2_stats,
    }
    return train_records, val_records, test_records, stats


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _train_step(
    model: DistributionalModel,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Single training step. Returns (loss_value, metrics_dict)."""
    batch = batch.to(device)

    h = model.encoder(batch.pep_tok, batch.mhc_a_tok, batch.mhc_b_tok)

    binding_ctx = getattr(batch, "binding_context", {})
    assay_emb = model.assay_ctx(
        binding_ctx.get("assay_type_idx", torch.zeros(h.shape[0], dtype=torch.long, device=device)),
        binding_ctx.get("assay_prep_idx", torch.zeros(h.shape[0], dtype=torch.long, device=device)),
        binding_ctx.get("assay_geometry_idx", torch.zeros(h.shape[0], dtype=torch.long, device=device)),
        binding_ctx.get("assay_readout_idx", torch.zeros(h.shape[0], dtype=torch.long, device=device)),
    )

    # Get raw IC50 nM targets
    bind_target = getattr(batch, "bind_target", None)
    bind_mask = getattr(batch, "bind_mask", None)
    bind_qual = getattr(batch, "bind_qual", None)
    if bind_target is None or bind_mask is None:
        raise RuntimeError("Batch missing binding target/mask")

    ic50_nM = bind_target.float().reshape(-1).to(device)
    mask = bind_mask.float().reshape(-1).to(device)
    qual = (bind_qual.long().reshape(-1).to(device) if bind_qual is not None
            else torch.zeros_like(mask, dtype=torch.long))

    loss, metrics = model.head.compute_loss(h, assay_emb, ic50_nM, qual, mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Track output-layer grad norm
    output_params = [p for p in model.head.parameters() if p.grad is not None]
    if output_params:
        grad_norm = torch.nn.utils.clip_grad_norm_(output_params, max_norm=float("inf"))
        metrics["grad_norm_output"] = float(grad_norm)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    return float(loss.detach()), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Distributional BA head training")
    parser.add_argument("--cond-id", type=int, required=True, help="Condition ID (1-32)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probe-peptide", type=str, default=DEFAULT_PROBE_PEPTIDE)
    parser.add_argument("--extra-probe-peptides", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qualifier-filter", type=str, default="all",
                        choices=sorted(QUALIFIER_FILTERS))
    parser.add_argument("--train-all-alleles", action="store_true")
    parser.add_argument("--config-version", type=str, default="v1",
                        choices=["v1", "v2", "v3"], help="Condition matrix version")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conditions_lookup = _get_conditions_lookup(str(args.config_version))
    cond_id = int(args.cond_id)
    if cond_id not in conditions_lookup:
        valid_ids = sorted(conditions_lookup.keys())
        raise ValueError(f"Unknown condition ID {cond_id} for {args.config_version}. Valid: {valid_ids}")
    spec = conditions_lookup[cond_id]

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    random.seed(int(args.seed))

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"artifacts/distributional_ba/{spec.label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    merged_tsv = data_dir / "merged_deduped.tsv"
    index_csv = data_dir / "mhc_index.csv"
    if not merged_tsv.exists():
        raise FileNotFoundError(f"Merged TSV not found: {merged_tsv}")
    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")

    probe_alleles = _split_csv(args.alleles)
    training_alleles = [] if bool(args.train_all_alleles) else list(probe_alleles)

    # --- Load data ---
    records, subset_stats = _load_binding_records_from_merged_tsv(
        merged_tsv,
        alleles=training_alleles,
        max_records=None,
        sampling_seed=int(args.seed) + 17,
    )
    records = [
        r for r in records
        if _keep_measurement_type(r.measurement_type, MEASUREMENT_PROFILE_NUMERIC)
    ]
    records = [
        r for r in records
        if _keep_binding_qualifier(getattr(r, "qualifier", 0), str(args.qualifier_filter))
    ]

    # --- 3-way split ---
    train_records, val_records, test_records, split_stats = _split_records_three_way(
        records, seed=int(args.seed), alleles=probe_alleles,
    )
    if not train_records or not val_records:
        raise RuntimeError("Split must produce train and val records")

    mhc_sequences, mhc_stats = resolve_mhc_sequences_from_index(
        index_csv=str(index_csv),
        alleles=sorted({
            str(r.mhc_allele or "").strip()
            for r in (train_records + val_records + test_records)
            if str(r.mhc_allele or "").strip()
        }),
    )

    # --- Build datasets ---
    collator = PrestoCollator()
    train_ds = PrestoDataset(
        binding_records=train_records, mhc_sequences=mhc_sequences, strict_mhc_resolution=False,
    )
    val_ds = PrestoDataset(
        binding_records=val_records, mhc_sequences=mhc_sequences, strict_mhc_resolution=False,
    )
    test_ds = PrestoDataset(
        binding_records=test_records, mhc_sequences=mhc_sequences, strict_mhc_resolution=False,
    )

    train_loader = create_dataloader(
        train_ds, batch_size=int(args.batch_size), shuffle=True,
        collator=collator, balanced=False, seed=int(args.seed),
    )
    val_loader = create_dataloader(
        val_ds, batch_size=int(args.batch_size), shuffle=False,
        collator=collator, balanced=False, seed=int(args.seed),
    )
    test_loader = create_dataloader(
        test_ds, batch_size=int(args.batch_size), shuffle=False,
        collator=collator, balanced=False, seed=int(args.seed),
    )

    # --- Build model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_embed_dim = getattr(spec, 'embed_dim', int(args.embed_dim))
    model = build_model(
        spec, embed_dim=actual_embed_dim,
        n_heads=int(args.n_heads), n_layers=int(args.n_layers),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay),
    )

    # --- Probe setup ---
    tokenizer = Tokenizer()
    allele_sequences = _resolve_allele_sequences(index_csv)
    fit_probe_peptides = _select_fit_supported_probe_peptides(records, probe_alleles)
    probe_peptides = [str(args.probe_peptide).strip().upper()]
    for pep in _split_csv(str(args.extra_probe_peptides or "")):
        pep_norm = pep.strip().upper()
        if pep_norm and pep_norm not in probe_peptides:
            probe_peptides.append(pep_norm)
    for pep in fit_probe_peptides:
        if pep not in probe_peptides:
            probe_peptides.append(pep)

    # --- Dry run ---
    if args.dry_run:
        batch = next(iter(train_loader))
        loss_val, metrics = _train_step(model, batch, optimizer, device)
        probe_eval = evaluate_probe_panel(
            model, tokenizer, allele_sequences, probe_peptides[:1], probe_alleles, device,
        )
        print(json.dumps({
            "event": "dry_run",
            "cond_id": cond_id,
            "label": spec.label,
            "n_params": n_params,
            "loss": loss_val,
            "metrics": metrics,
            "probe_sample": probe_eval[:2],
        }, sort_keys=True), flush=True)
        return

    # --- Setup summary ---
    config_dict = {
        "cond_id": cond_id,
        "label": spec.label,
        "head_type": spec.head_type,
        "assay_mode": spec.assay_mode,
        "max_nM": spec.max_nM,
        "n_bins": spec.n_bins,
        "sigma_mult": spec.sigma_mult,
        "n_params": n_params,
        "embed_dim": actual_embed_dim,
        "n_heads": int(args.n_heads),
        "n_layers": int(args.n_layers),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "probe_alleles": probe_alleles,
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "test_rows": len(test_records),
        "device": device,
    }
    print(json.dumps({"event": "setup", **config_dict}, sort_keys=True), flush=True)

    # --- Training loop ---
    step_log: List[Dict[str, Any]] = []
    epoch_summaries: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []

    global_step = 0
    model.train()

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_batches = 0
        epoch_metrics: Dict[str, float] = defaultdict(float)

        for batch in train_loader:
            global_step += 1
            loss_val, batch_metrics = _train_step(model, batch, optimizer, device)
            train_loss_sum += loss_val
            train_batches += 1
            for k, v in batch_metrics.items():
                epoch_metrics[k] += v

            step_log.append({
                "step": global_step,
                "epoch": epoch,
                "train_loss": loss_val,
                **{k: v for k, v in batch_metrics.items()},
            })

        train_loss = train_loss_sum / max(train_batches, 1)
        for k in epoch_metrics:
            epoch_metrics[k] /= max(train_batches, 1)

        # --- Validation ---
        val_result = evaluate_held_out(model, val_loader, device)
        epoch_time = time.perf_counter() - epoch_start

        # --- Probe ---
        probe_eval = evaluate_probe_panel(
            model, tokenizer, allele_sequences, probe_peptides, probe_alleles, device,
        )
        for row in probe_eval:
            probe_rows.append({"epoch": epoch, **row})

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_s": round(epoch_time, 2),
            **{f"train_{k}": v for k, v in epoch_metrics.items()},
            **{f"val_{k}": v for k, v in val_result.items()},
        }
        epoch_summaries.append(epoch_summary)

        print(json.dumps({"event": "epoch", **epoch_summary}, sort_keys=True), flush=True)

    # --- Test-set evaluation ---
    test_result = evaluate_held_out(model, test_loader, device)
    print(json.dumps({"event": "test", **test_result}, sort_keys=True), flush=True)

    # --- Save artifacts ---
    _write_jsonl(out_dir / "step_log.jsonl", step_log)
    _write_jsonl(out_dir / "metrics.jsonl", epoch_summaries)
    _write_jsonl(out_dir / "probes.jsonl", probe_rows)

    summary = {
        "config": config_dict,
        "epoch_summaries": epoch_summaries,
        "test_metrics": test_result,
        "split_stats": split_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    # Save model checkpoint
    torch.save(model.state_dict(), out_dir / "model.pt")

    print(json.dumps({
        "event": "done",
        "cond_id": cond_id,
        "label": spec.label,
        "final_train_loss": epoch_summaries[-1]["train_loss"] if epoch_summaries else None,
        "final_val_loss": val_result.get("loss"),
        "test_loss": test_result.get("loss"),
        "test_spearman": test_result.get("spearman"),
        "test_auroc": test_result.get("auroc"),
        "out_dir": str(out_dir),
    }, sort_keys=True), flush=True)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
