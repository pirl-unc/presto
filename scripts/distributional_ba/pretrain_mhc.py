#!/usr/bin/env python
"""MHC species + class pretraining for GrooveTransformerModel encoder.

Trains the same encoder architecture used by distributional_ba to predict
MHC chain class (I vs II) and species of origin from groove sequences.
The resulting checkpoint can be loaded via --init-checkpoint in train.py
to warm-start the binding affinity experiment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from presto.data.tokenizer import Tokenizer
from presto.data.vocab import CHAIN_SPECIES_TO_IDX
from presto.scripts.groove_baseline_probe import GrooveTransformerModel
from presto.scripts.pretrain_mhc_encoder import (
    GroupBalancedBatchSampler,
    MHCWarmStartDataset,
    MHCWarmStartSample,
    _collate_samples,
    _derived_class_logits,
    _derived_species_logits,
    _masked_ce,
    _split_samples,
    build_mhc_warmstart_samples,
)

N_SPECIES = len(CHAIN_SPECIES_TO_IDX)
N_CHAIN_TYPES = 5  # matches MHC_CHAIN_FINE_TO_IDX size


class MHCPretrainWrapper(nn.Module):
    """GrooveTransformerModel + classification heads for MHC pretraining."""

    def __init__(self, encoder: GrooveTransformerModel):
        super().__init__()
        self.encoder = encoder
        d = encoder.embed_dim
        self.type_a_head = nn.Linear(d, N_CHAIN_TYPES)
        self.type_b_head = nn.Linear(d, N_CHAIN_TYPES)
        self.species_a_head = nn.Linear(d, N_SPECIES)
        self.species_b_head = nn.Linear(d, N_SPECIES)

    def forward(
        self, mhc_a_tok: torch.Tensor, mhc_b_tok: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mhc_a_vec = self.encoder._encode_segment(
            mhc_a_tok, pos_mode=self.encoder.groove_pos_mode,
        )
        mhc_b_vec = self.encoder._encode_segment(
            mhc_b_tok, pos_mode=self.encoder.groove_pos_mode,
        )
        return {
            "mhc_a_type_logits": self.type_a_head(mhc_a_vec),
            "mhc_b_type_logits": self.type_b_head(mhc_b_vec),
            "mhc_a_species_logits": self.species_a_head(mhc_a_vec),
            "mhc_b_species_logits": self.species_b_head(mhc_b_vec),
        }


def _epoch_pass(
    wrapper: MHCPretrainWrapper,
    loader: DataLoader,
    tokenizer: Tokenizer,
    *,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    wrapper.train(mode=train_mode)
    loss_sum = 0.0
    batches = 0
    type_correct = type_total = 0
    species_correct = species_total = 0
    class_correct = class_total = 0

    for samples in loader:
        batch = _collate_samples(samples, tokenizer, device=device)
        outputs = wrapper(batch["mhc_a_tok"], batch["mhc_b_tok"])
        losses: List[torch.Tensor] = []

        for side in ("a", "b"):
            loss, c, t = _masked_ce(
                outputs[f"mhc_{side}_type_logits"],
                batch[f"type_{side}_target"],
                batch[f"type_{side}_mask"],
            )
            if loss is not None:
                losses.append(loss)
            type_correct += c
            type_total += t

            loss, c, t = _masked_ce(
                outputs[f"mhc_{side}_species_logits"],
                batch["species_target"],
                batch[f"species_{side}_mask"],
            )
            if loss is not None:
                losses.append(loss)
            species_correct += c
            species_total += t

        class_logits = _derived_class_logits(
            outputs["mhc_a_type_logits"],
            outputs["mhc_b_type_logits"],
            batch["type_a_mask"],
            batch["type_b_mask"],
        )
        class_loss = F.cross_entropy(class_logits, batch["class_target"])
        losses.append(class_loss)
        class_preds = torch.argmax(class_logits, dim=-1)
        class_correct += int((class_preds == batch["class_target"]).sum().item())
        class_total += int(batch["class_target"].numel())

        species_logits = _derived_species_logits(
            outputs["mhc_a_species_logits"],
            outputs["mhc_b_species_logits"],
            batch["species_a_mask"],
            batch["species_b_mask"],
        )
        losses.append(F.cross_entropy(species_logits, batch["species_target"]))

        total_loss = torch.stack(losses).mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
            optimizer.step()

        loss_sum += float(total_loss.detach().item())
        batches += 1

    return {
        "loss": loss_sum / max(batches, 1),
        "type_acc": float(type_correct / max(type_total, 1)),
        "species_acc": float(species_correct / max(species_total, 1)),
        "class_acc": float(class_correct / max(class_total, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MHC species+class pretraining for GrooveTransformerModel encoder",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--index-csv", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="artifacts/mhc_pretrain_groove")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    index_csv = Path(args.index_csv) if args.index_csv.strip() else data_dir / "mhc_index.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")

    samples, dataset_stats = build_mhc_warmstart_samples(
        index_csv, max_samples=args.max_samples, seed=args.seed,
    )
    if not samples:
        raise RuntimeError("No MHC samples built from the index")

    train_samples, val_samples, split_stats = _split_samples(
        samples, val_fraction=args.val_fraction, seed=args.seed,
    )

    train_ds = MHCWarmStartDataset(train_samples)
    val_ds = MHCWarmStartDataset(val_samples)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=GroupBalancedBatchSampler(
            train_ds, batch_size=args.batch_size, seed=args.seed,
        ),
        collate_fn=lambda items: items,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda items: items, num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = GrooveTransformerModel(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.embed_dim,
        hidden_dim=args.embed_dim,
    )
    wrapper = MHCPretrainWrapper(encoder).to(device)
    optimizer = torch.optim.AdamW(
        wrapper.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    tokenizer = Tokenizer()

    print(json.dumps({
        "event": "mhc_pretrain_groove_setup",
        "rows": len(samples),
        "train_rows": len(train_samples),
        "val_rows": len(val_samples),
        "embed_dim": args.embed_dim,
        "device": device,
        "dataset_stats": dataset_stats,
    }, sort_keys=True), flush=True)

    epochs_log: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch_pass(
            wrapper, train_loader, tokenizer, device=device, optimizer=optimizer,
        )
        val_metrics = _epoch_pass(
            wrapper, val_loader, tokenizer, device=device, optimizer=None,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        epochs_log.append(row)
        print(json.dumps({"event": "mhc_pretrain_epoch", **row}, sort_keys=True), flush=True)

    # Save only the encoder state dict (not classification heads)
    encoder_path = out_dir / "encoder.pt"
    torch.save(encoder.state_dict(), encoder_path)

    summary = {
        "config": {
            "embed_dim": args.embed_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "max_samples": args.max_samples or None,
            "index_csv": str(index_csv),
        },
        "dataset_stats": dataset_stats,
        "split_stats": split_stats,
        "epochs": epochs_log,
        "encoder_checkpoint": str(encoder_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"event": "mhc_pretrain_done", "encoder_path": str(encoder_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
