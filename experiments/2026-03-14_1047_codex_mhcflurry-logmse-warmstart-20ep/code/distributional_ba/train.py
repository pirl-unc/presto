#!/usr/bin/env python
"""Self-contained trainer for the clean distributional BA head benchmark.

Usage:
    PYTHONPATH=<experiment_code>:<repo_parent> python -m distributional_ba.train --cond-id 1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from presto.data import BindingRecord, PrestoCollator, PrestoDataset, create_dataloader
from presto.data.tokenizer import Tokenizer
from presto.scripts.focused_binding_probe import (
    DEFAULT_ALLELES,
    DEFAULT_PROBE_PEPTIDE,
    MEASUREMENT_PROFILE_NUMERIC,
    QUALIFIER_FILTERS,
    _keep_binding_qualifier,
    _keep_measurement_type,
    _load_binding_records_from_merged_tsv,
    _resolve_allele_sequences,
    _select_fit_supported_probe_peptides,
    _split_csv,
    _split_records_by_peptide,
)
from presto.scripts.train_iedb import resolve_mhc_sequences_from_index

from .config import CONDITIONS_BY_ID, ConditionSpec, DistributionalModel, build_model
from .evaluate import evaluate_held_out, evaluate_probe_panel


def _split_records_three_way(
    records: Sequence[BindingRecord],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    alleles: Optional[Sequence[str]] = None,
) -> Tuple[List[BindingRecord], List[BindingRecord], List[BindingRecord], Dict[str, Any]]:
    """Deterministic peptide-group train/val/test split."""

    if abs(float(train_frac) + float(val_frac) + float(test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    remaining, test_records, split1_stats = _split_records_by_peptide(
        list(records),
        val_fraction=float(test_frac),
        seed=int(seed) + 100,
        alleles=alleles,
    )
    adjusted_val = float(val_frac) / max(float(train_frac) + float(val_frac), 1e-8)
    train_records, val_records, split2_stats = _split_records_by_peptide(
        remaining,
        val_fraction=adjusted_val,
        seed=int(seed),
        alleles=alleles,
    )
    stats = {
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "test_rows": len(test_records),
        "split1": split1_stats,
        "split2": split2_stats,
    }
    return list(train_records), list(val_records), list(test_records), stats


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    base_lr: float,
    steps_per_epoch: int,
    epochs: int,
    warmup_fraction: float,
    min_lr_scale: float,
) -> Optional[SequentialLR]:
    schedule = str(schedule).strip().lower()
    if schedule == "constant":
        return None
    if schedule != "warmup_cosine":
        raise ValueError(f"Unsupported lr schedule: {schedule!r}")

    total_steps = max(1, int(steps_per_epoch) * int(epochs))
    if total_steps <= 1:
        return None
    warmup_steps = int(round(total_steps * float(warmup_fraction)))
    warmup_steps = max(1, min(total_steps - 1, warmup_steps))
    start_factor = min(1.0, max(1e-4, 1e-6 / max(float(base_lr), 1e-12)))
    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=float(base_lr) * float(min_lr_scale),
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def _load_checkpoint_payload(checkpoint_path: str) -> Dict[str, Any]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def _filter_backbone_warm_start_state(
    model: DistributionalModel,
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    encoder_state = model.encoder.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    loaded_pairs: List[Dict[str, str]] = []
    skipped_missing: List[str] = []
    skipped_shape: List[Dict[str, Any]] = []

    for src_key, value in state_dict.items():
        candidate_keys: List[str] = []
        if src_key == "aa_embedding.weight":
            candidate_keys.append("aa_embedding.weight")
        if src_key.startswith("stream_encoder."):
            candidate_keys.append("encoder." + src_key[len("stream_encoder.") :])
        if not candidate_keys:
            continue

        matched = False
        for dst_key in candidate_keys:
            target = encoder_state.get(dst_key)
            if target is None:
                continue
            matched = True
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target.shape):
                skipped_shape.append(
                    {
                        "src": src_key,
                        "dst": dst_key,
                        "src_shape": tuple(value.shape) if isinstance(value, torch.Tensor) else None,
                        "dst_shape": tuple(target.shape),
                    }
                )
                break
            compatible[dst_key] = value
            loaded_pairs.append({"src": src_key, "dst": dst_key})
            break
        if not matched:
            skipped_missing.append(src_key)

    return compatible, {
        "loaded_pairs": loaded_pairs,
        "loaded_count": len(compatible),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
    }


def _maybe_apply_warm_start(
    model: DistributionalModel,
    checkpoint_path: str,
) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(checkpoint_path)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid model state_dict")
    compatible_state_dict, warm_start_stats = _filter_backbone_warm_start_state(model, state_dict)
    model.encoder.load_state_dict(compatible_state_dict, strict=False)
    return {
        "used": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_format": payload.get("checkpoint_format", ""),
        "checkpoint_model_class": payload.get("model_class", ""),
        **warm_start_stats,
    }


def _train_step(
    model: DistributionalModel,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    scheduler: Optional[SequentialLR] = None,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Single optimization step."""

    batch = batch.to(device)
    h = model.encoder(batch.pep_tok, batch.mhc_a_tok, batch.mhc_b_tok)
    binding_ctx = getattr(batch, "binding_context", {})
    assay_emb = model._assay_embedding(
        batch_size=h.shape[0],
        device=h.device,
        assay_type_idx=binding_ctx.get("assay_type_idx"),
        assay_prep_idx=binding_ctx.get("assay_prep_idx"),
        assay_geometry_idx=binding_ctx.get("assay_geometry_idx"),
        assay_readout_idx=binding_ctx.get("assay_readout_idx"),
    )

    bind_target = getattr(batch, "bind_target", None)
    bind_mask = getattr(batch, "bind_mask", None)
    bind_qual = getattr(batch, "bind_qual", None)
    if bind_target is None or bind_mask is None:
        raise RuntimeError("Batch missing binding target/mask")

    ic50_nM = bind_target.float().reshape(-1).to(device)
    mask = bind_mask.float().reshape(-1).to(device)
    qual = (
        bind_qual.long().reshape(-1).to(device)
        if bind_qual is not None
        else torch.zeros_like(mask, dtype=torch.long)
    )

    optimizer.zero_grad(set_to_none=True)
    loss, metrics = model.head.compute_loss(h, assay_emb, ic50_nM, qual, mask)
    loss.backward()

    output_params = [param for param in model.head.parameters() if param.grad is not None]
    if output_params:
        grad_norm = torch.nn.utils.clip_grad_norm_(output_params, max_norm=float("inf"))
        metrics["grad_norm_output"] = float(grad_norm)
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
    metrics["grad_norm_total"] = float(total_grad_norm)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    metrics["lr"] = float(optimizer.param_groups[0]["lr"])
    return float(loss.detach()), metrics


def _record_to_row(split: str, index: int, record: BindingRecord) -> Dict[str, Any]:
    return {
        "split": split,
        "split_index": int(index),
        "peptide": str(record.peptide),
        "mhc_allele": str(record.mhc_allele),
        "value_nM": float(record.value),
        "qualifier": int(record.qualifier),
        "measurement_type": str(record.measurement_type),
        "assay_type": str(record.assay_type or ""),
        "assay_method": str(record.assay_method or ""),
        "effector_culture_condition": str(record.effector_culture_condition or ""),
        "apc_culture_condition": str(record.apc_culture_condition or ""),
        "mhc_class": str(record.mhc_class or ""),
        "species": str(record.species or ""),
        "antigen_species": str(record.antigen_species or ""),
        "source": str(record.source or ""),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_split_manifests(
    out_dir: Path,
    *,
    train_records: Sequence[BindingRecord],
    val_records: Sequence[BindingRecord],
    test_records: Sequence[BindingRecord],
) -> None:
    split_dir = out_dir / "splits"
    _write_csv(
        split_dir / "train_records.csv",
        [_record_to_row("train", idx, record) for idx, record in enumerate(train_records)],
    )
    _write_csv(
        split_dir / "val_records.csv",
        [_record_to_row("val", idx, record) for idx, record in enumerate(val_records)],
    )
    _write_csv(
        split_dir / "test_records.csv",
        [_record_to_row("test", idx, record) for idx, record in enumerate(test_records)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean distributional BA head trainer")
    parser.add_argument("--cond-id", type=int, required=True, help="Condition ID (1-4)")
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
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", type=str, choices=("constant", "warmup_cosine"), default="warmup_cosine")
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qualifier-filter", type=str, default="all", choices=sorted(QUALIFIER_FILTERS))
    parser.add_argument("--max-records", type=int, default=0, help="Optional dataset cap for smoke runs.")
    parser.add_argument("--train-all-alleles", action="store_true")
    parser.add_argument("--threshold-nm", type=float, default=500.0)
    parser.add_argument("--init-checkpoint", type=str, default="", help="Required for warm-start conditions.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cond_id = int(args.cond_id)
    if cond_id not in CONDITIONS_BY_ID:
        raise ValueError(f"Unknown condition ID {cond_id}. Valid: {sorted(CONDITIONS_BY_ID)}")
    spec: ConditionSpec = CONDITIONS_BY_ID[cond_id]
    init_checkpoint = str(args.init_checkpoint or "").strip()
    if spec.warm_start and not init_checkpoint:
        raise ValueError(f"{spec.label} requires --init-checkpoint")
    if not spec.warm_start and init_checkpoint:
        raise ValueError(f"{spec.label} is a cold-start condition and must not receive --init-checkpoint")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    random.seed(int(args.seed))

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else Path(f"artifacts/distributional_ba/{spec.label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    merged_tsv = data_dir / "merged_deduped.tsv"
    index_csv = data_dir / "mhc_index.csv"
    if not merged_tsv.exists():
        raise FileNotFoundError(f"Merged TSV not found: {merged_tsv}")
    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")

    probe_alleles = _split_csv(str(args.alleles))
    training_alleles = [] if bool(args.train_all_alleles) else list(probe_alleles)

    records, subset_stats = _load_binding_records_from_merged_tsv(
        merged_tsv,
        alleles=training_alleles,
        max_records=(None if int(args.max_records) <= 0 else int(args.max_records)),
        sampling_seed=int(args.seed) + 17,
    )
    records = [record for record in records if _keep_measurement_type(record.measurement_type, MEASUREMENT_PROFILE_NUMERIC)]
    records = [
        record
        for record in records
        if _keep_binding_qualifier(getattr(record, "qualifier", 0), str(args.qualifier_filter))
    ]

    train_records, val_records, test_records, split_stats = _split_records_three_way(
        records,
        seed=int(args.seed),
        alleles=probe_alleles,
    )
    if not train_records or not val_records or not test_records:
        raise RuntimeError("Deterministic split must produce non-empty train/val/test partitions")

    mhc_sequences, mhc_stats = resolve_mhc_sequences_from_index(
        index_csv=str(index_csv),
        alleles=sorted(
            {
                str(record.mhc_allele or "").strip()
                for record in (train_records + val_records + test_records)
                if str(record.mhc_allele or "").strip()
            }
        ),
    )

    collator = PrestoCollator()
    train_ds = PrestoDataset(binding_records=train_records, mhc_sequences=mhc_sequences, strict_mhc_resolution=False)
    val_ds = PrestoDataset(binding_records=val_records, mhc_sequences=mhc_sequences, strict_mhc_resolution=False)
    test_ds = PrestoDataset(binding_records=test_records, mhc_sequences=mhc_sequences, strict_mhc_resolution=False)
    train_loader = create_dataloader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collator=collator,
        balanced=False,
        seed=int(args.seed),
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collator=collator,
        balanced=False,
        seed=int(args.seed),
    )
    test_loader = create_dataloader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collator=collator,
        balanced=False,
        seed=int(args.seed),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_embed_dim = getattr(spec, "embed_dim", int(args.embed_dim))
    model = build_model(
        spec,
        embed_dim=actual_embed_dim,
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        ff_dim=int(args.ff_dim),
    )
    warm_start_stats = {
        "used": False,
        "checkpoint_path": "",
        "checkpoint_format": "",
        "checkpoint_model_class": "",
        "loaded_count": 0,
        "loaded_pairs": [],
        "skipped_missing": [],
        "skipped_shape": [],
    }
    if init_checkpoint:
        warm_start_stats = _maybe_apply_warm_start(model, init_checkpoint)
    model = model.to(device)
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = _build_scheduler(
        optimizer,
        schedule=str(args.lr_schedule),
        base_lr=float(args.lr),
        steps_per_epoch=max(len(train_loader), 1),
        epochs=int(args.epochs),
        warmup_fraction=float(args.warmup_fraction),
        min_lr_scale=float(args.min_lr_scale),
    )

    tokenizer = Tokenizer()
    allele_sequences = _resolve_allele_sequences(index_csv)
    fit_probe_peptides = _select_fit_supported_probe_peptides(records, probe_alleles)
    probe_peptides = [str(args.probe_peptide).strip().upper()]
    for peptide in _split_csv(str(args.extra_probe_peptides or "")):
        normalized = peptide.strip().upper()
        if normalized and normalized not in probe_peptides:
            probe_peptides.append(normalized)
    for peptide in fit_probe_peptides:
        if peptide not in probe_peptides:
            probe_peptides.append(peptide)

    if args.dry_run:
        batch = next(iter(train_loader))
        loss_value, batch_metrics = _train_step(
            model,
            batch,
            optimizer,
            device,
            scheduler=scheduler,
        )
        probe_eval = evaluate_probe_panel(
            model,
            tokenizer,
            allele_sequences,
            probe_peptides[:1],
            probe_alleles,
            device,
        )
        print(
            json.dumps(
                {
                    "event": "dry_run",
                    "cond_id": cond_id,
                    "label": spec.label,
                    "n_params": n_params,
                    "warm_start": warm_start_stats,
                    "loss": loss_value,
                    "metrics": batch_metrics,
                    "probe_sample": probe_eval[:2],
                    "train_rows": len(train_records),
                    "val_rows": len(val_records),
                    "test_rows": len(test_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

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
        "ff_dim": int(args.ff_dim),
        "lr": float(args.lr),
        "lr_schedule": str(args.lr_schedule),
        "warmup_fraction": float(args.warmup_fraction),
        "min_lr_scale": float(args.min_lr_scale),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "qualifier_filter": str(args.qualifier_filter),
        "max_records": int(args.max_records),
        "probe_alleles": probe_alleles,
        "probe_peptides": probe_peptides,
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "test_rows": len(test_records),
        "device": device,
        "warm_start_requested": bool(spec.warm_start),
        "warm_start_checkpoint": init_checkpoint,
        "warm_start_loaded_count": int(warm_start_stats.get("loaded_count", 0)),
    }
    print(json.dumps({"event": "setup", **config_dict}, sort_keys=True), flush=True)
    if warm_start_stats.get("used"):
        print(json.dumps({"event": "warm_start", **warm_start_stats}, sort_keys=True), flush=True)

    _write_split_manifests(
        out_dir,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )

    step_log: List[Dict[str, Any]] = []
    epoch_summaries: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []

    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    model.train()

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_batches = 0
        epoch_metrics: Dict[str, float] = defaultdict(float)

        for batch in train_loader:
            global_step += 1
            loss_value, batch_metrics = _train_step(
                model,
                batch,
                optimizer,
                device,
                scheduler=scheduler,
            )
            train_loss_sum += loss_value
            train_batches += 1
            for key, value in batch_metrics.items():
                epoch_metrics[key] += float(value)

            step_log.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": loss_value,
                    **{key: float(value) for key, value in batch_metrics.items()},
                }
            )

        train_loss = train_loss_sum / max(train_batches, 1)
        for key in list(epoch_metrics.keys()):
            epoch_metrics[key] /= max(train_batches, 1)

        val_metrics, _ = evaluate_held_out(
            model,
            val_loader,
            device,
            split_name="val",
            threshold_nM=float(args.threshold_nm),
        )
        probe_eval = evaluate_probe_panel(
            model,
            tokenizer,
            allele_sequences,
            probe_peptides,
            probe_alleles,
            device,
        )
        for row in probe_eval:
            probe_rows.append({"epoch": epoch, **row})

        epoch_time = time.perf_counter() - epoch_start
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_s": round(epoch_time, 3),
            "lr_end": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": float(value) for key, value in epoch_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        epoch_summaries.append(epoch_summary)
        if float(val_metrics.get("loss", float("inf"))) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
        print(json.dumps({"event": "epoch", **epoch_summary}, sort_keys=True), flush=True)

    val_metrics, val_predictions = evaluate_held_out(
        model,
        val_loader,
        device,
        split_name="val",
        threshold_nM=float(args.threshold_nm),
    )
    test_metrics, test_predictions = evaluate_held_out(
        model,
        test_loader,
        device,
        split_name="test",
        threshold_nM=float(args.threshold_nm),
    )
    print(json.dumps({"event": "val_final", **val_metrics}, sort_keys=True), flush=True)
    print(json.dumps({"event": "test", **test_metrics}, sort_keys=True), flush=True)

    _write_jsonl(out_dir / "step_log.jsonl", step_log)
    _write_jsonl(out_dir / "metrics.jsonl", epoch_summaries)
    _write_jsonl(out_dir / "probes.jsonl", probe_rows)
    _write_csv(out_dir / "val_predictions.csv", val_predictions)
    _write_csv(out_dir / "test_predictions.csv", test_predictions)

    summary = {
        "config": config_dict,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epoch_summaries": epoch_summaries,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "split_stats": split_stats,
        "subset_stats": subset_stats,
        "mhc_stats": mhc_stats,
        "warm_start": warm_start_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "model.pt")

    print(
        json.dumps(
            {
                "event": "done",
                "cond_id": cond_id,
                "label": spec.label,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "final_val_loss": val_metrics.get("loss"),
                "test_loss": test_metrics.get("loss"),
                "test_spearman": test_metrics.get("spearman"),
                "test_auroc": test_metrics.get("auroc"),
                "out_dir": str(out_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
