#!/usr/bin/env python
"""MHC-only warm-start pretraining for the shared groove encoder.

Trains on indexed MHC groove sequences to predict:
- chain class/type (via existing MHC chain heads)
- species category used by the network

The resulting checkpoint is intended only as a short warm start for focused
affinity training, not as a standalone final model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from presto.data.allele_resolver import normalize_species_label
from presto.data.groove import parse_class_i, parse_class_ii_alpha, parse_class_ii_beta
from presto.data.loaders import MHC_ALLOWED_AA
from presto.data.mhc_index import infer_fine_chain_type, load_mhc_index
from presto.data.tokenizer import Tokenizer
from presto.data.vocab import CHAIN_SPECIES_TO_IDX, MHC_CHAIN_FINE_TO_IDX
from presto.models.presto import Presto
from presto.training.checkpointing import save_model_checkpoint


DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 192
DEFAULT_D_MODEL = 128
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 4
DEFAULT_LR = 2.0e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42

TYPE_I = int(MHC_CHAIN_FINE_TO_IDX["MHC_I"])
TYPE_IIA = int(MHC_CHAIN_FINE_TO_IDX["MHC_IIa"])
TYPE_IIB = int(MHC_CHAIN_FINE_TO_IDX["MHC_IIb"])


@dataclass(frozen=True)
class MHCWarmStartSample:
    allele: str
    mhc_a: str
    mhc_b: str
    class_target: int
    species_target: int
    type_a_target: int
    type_b_target: int
    type_a_mask: float
    type_b_mask: float
    species_a_mask: float
    species_b_mask: float
    group_key: str


class MHCWarmStartDataset(Dataset):
    def __init__(self, samples: Sequence[MHCWarmStartSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MHCWarmStartSample:
        return self.samples[idx]


class GroupBalancedBatchSampler(Sampler[List[int]]):
    """Strictly balances batches across active label groups."""

    def __init__(
        self,
        dataset: MHCWarmStartDataset,
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            groups[str(sample.group_key)].append(idx)
        self.groups = {key: indices for key, indices in groups.items() if indices}
        if not self.groups:
            raise ValueError("Cannot build group-balanced sampler on an empty dataset")

        self.group_names = sorted(self.groups.keys())
        base = self.batch_size // len(self.group_names)
        extra = self.batch_size % len(self.group_names)
        if base <= 0:
            raise ValueError(
                f"batch_size={self.batch_size} too small for {len(self.group_names)} groups"
            )
        self.slots_by_group = {
            group: base + (1 if i < extra else 0)
            for i, group in enumerate(self.group_names)
        }
        self.num_batches = max(
            math.ceil(len(self.groups[group]) / float(self.slots_by_group[group]))
            for group in self.group_names
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        pools = {group: list(indices) for group, indices in self.groups.items()}
        cursors = {group: 0 for group in self.group_names}
        for group in self.group_names:
            rng.shuffle(pools[group])

        for _ in range(self.num_batches):
            batch: List[int] = []
            for group in self.group_names:
                slots = self.slots_by_group[group]
                pool = pools[group]
                picks: List[int] = []
                while len(picks) < slots:
                    remaining = len(pool) - cursors[group]
                    take = min(slots - len(picks), remaining)
                    if take > 0:
                        picks.extend(pool[cursors[group] : cursors[group] + take])
                        cursors[group] += take
                    if len(picks) < slots:
                        rng.shuffle(pool)
                        cursors[group] = 0
                batch.extend(picks)
            rng.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
        self._epoch += 1


def _record_groove_halves(record: Any, chain_type: str) -> Tuple[str, str]:
    half_1 = str(getattr(record, "groove_half_1", "") or "").strip().upper()
    half_2 = str(getattr(record, "groove_half_2", "") or "").strip().upper()
    sequence = str(getattr(record, "sequence", "") or "").strip().upper()
    if chain_type == "MHC_I":
        if half_1 and half_2:
            return half_1, half_2
        parsed = parse_class_i(sequence)
        if parsed.ok:
            return parsed.groove_half_1, parsed.groove_half_2
        return "", ""
    if chain_type == "MHC_IIa":
        if half_1:
            return half_1, ""
        parsed = parse_class_ii_alpha(sequence)
        if parsed.ok:
            return parsed.groove_half_1, ""
        return "", ""
    if chain_type == "MHC_IIb":
        if half_2:
            return "", half_2
        parsed = parse_class_ii_beta(sequence)
        if parsed.ok:
            return "", parsed.groove_half_2
        return "", ""
    return "", ""


def _is_valid_mhc_segment(sequence: str) -> bool:
    seq = str(sequence or "").strip().upper()
    return bool(seq) and set(seq) <= MHC_ALLOWED_AA


def build_mhc_warmstart_samples(
    index_csv: Path,
    *,
    max_samples: int = 0,
    seed: int = 42,
) -> Tuple[List[MHCWarmStartSample], Dict[str, Any]]:
    records = load_mhc_index(str(index_csv))
    samples: List[MHCWarmStartSample] = []
    stats: Counter[str] = Counter()
    by_group: Counter[str] = Counter()
    by_chain_type: Counter[str] = Counter()
    by_species: Counter[str] = Counter()

    for record in records.values():
        stats["rows_seen"] += 1
        if getattr(record, "is_functional", None) is False:
            stats["skip_nonfunctional"] += 1
            continue
        chain_type = infer_fine_chain_type(record.gene, record.mhc_class, record.seq_len)
        if chain_type not in {"MHC_I", "MHC_IIa", "MHC_IIb"}:
            stats["skip_chain_type"] += 1
            continue
        species_bucket = normalize_species_label(record.species)
        if species_bucket is None:
            stats["skip_species"] += 1
            continue
        half_1, half_2 = _record_groove_halves(record, chain_type)
        if chain_type == "MHC_I" and (not half_1 or not half_2):
            stats["skip_missing_groove"] += 1
            continue
        if chain_type == "MHC_IIa" and not half_1:
            stats["skip_missing_groove"] += 1
            continue
        if chain_type == "MHC_IIb" and not half_2:
            stats["skip_missing_groove"] += 1
            continue
        if chain_type == "MHC_I" and (not _is_valid_mhc_segment(half_1) or not _is_valid_mhc_segment(half_2)):
            stats["skip_invalid_groove_chars"] += 1
            continue
        if chain_type == "MHC_IIa" and not _is_valid_mhc_segment(half_1):
            stats["skip_invalid_groove_chars"] += 1
            continue
        if chain_type == "MHC_IIb" and not _is_valid_mhc_segment(half_2):
            stats["skip_invalid_groove_chars"] += 1
            continue

        species_target = int(CHAIN_SPECIES_TO_IDX[species_bucket])
        class_target = 0 if chain_type == "MHC_I" else 1
        if chain_type == "MHC_I":
            type_a_target, type_b_target = TYPE_I, TYPE_I
            type_a_mask, type_b_mask = 1.0, 1.0
            species_a_mask, species_b_mask = 1.0, 1.0
        elif chain_type == "MHC_IIa":
            type_a_target, type_b_target = TYPE_IIA, 0
            type_a_mask, type_b_mask = 1.0, 0.0
            species_a_mask, species_b_mask = 1.0, 0.0
        else:
            type_a_target, type_b_target = 0, TYPE_IIB
            type_a_mask, type_b_mask = 0.0, 1.0
            species_a_mask, species_b_mask = 0.0, 1.0

        group_key = f"class{class_target}:{species_bucket}"
        sample = MHCWarmStartSample(
            allele=str(record.normalized or record.allele_raw),
            mhc_a=half_1,
            mhc_b=half_2,
            class_target=class_target,
            species_target=species_target,
            type_a_target=type_a_target,
            type_b_target=type_b_target,
            type_a_mask=type_a_mask,
            type_b_mask=type_b_mask,
            species_a_mask=species_a_mask,
            species_b_mask=species_b_mask,
            group_key=group_key,
        )
        samples.append(sample)
        stats["kept"] += 1
        by_group[group_key] += 1
        by_chain_type[chain_type] += 1
        by_species[species_bucket] += 1

    if max_samples > 0 and len(samples) > max_samples:
        rng = random.Random(int(seed))
        rng.shuffle(samples)
        samples = samples[: int(max_samples)]
        stats["capped"] = len(samples)

    return samples, {
        **stats,
        "by_group": dict(by_group),
        "by_chain_type": dict(by_chain_type),
        "by_species": dict(by_species),
        "rows_final": len(samples),
    }


def _split_samples(
    samples: Sequence[MHCWarmStartSample],
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[List[MHCWarmStartSample], List[MHCWarmStartSample], Dict[str, Any]]:
    grouped: Dict[str, List[MHCWarmStartSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_key].append(sample)
    rng = random.Random(int(seed))
    train: List[MHCWarmStartSample] = []
    val: List[MHCWarmStartSample] = []
    for group, rows in grouped.items():
        rows = list(rows)
        rng.shuffle(rows)
        n_val = int(round(len(rows) * float(val_fraction)))
        n_val = max(1, n_val) if len(rows) > 1 else 0
        if n_val >= len(rows):
            n_val = len(rows) - 1
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val, {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_groups": dict(Counter(s.group_key for s in train)),
        "val_groups": dict(Counter(s.group_key for s in val)),
    }


def _collate_samples(
    samples: Sequence[MHCWarmStartSample],
    tokenizer: Tokenizer,
    *,
    device: str,
) -> Dict[str, torch.Tensor]:
    mhc_a_tok = tokenizer.batch_encode([s.mhc_a for s in samples], max_len=120).to(device)
    mhc_b_tok = tokenizer.batch_encode([s.mhc_b for s in samples], max_len=120).to(device)
    return {
        "mhc_a_tok": mhc_a_tok,
        "mhc_b_tok": mhc_b_tok,
        "class_target": torch.tensor([s.class_target for s in samples], dtype=torch.long, device=device),
        "species_target": torch.tensor([s.species_target for s in samples], dtype=torch.long, device=device),
        "type_a_target": torch.tensor([s.type_a_target for s in samples], dtype=torch.long, device=device),
        "type_b_target": torch.tensor([s.type_b_target for s in samples], dtype=torch.long, device=device),
        "type_a_mask": torch.tensor([s.type_a_mask for s in samples], dtype=torch.float32, device=device),
        "type_b_mask": torch.tensor([s.type_b_mask for s in samples], dtype=torch.float32, device=device),
        "species_a_mask": torch.tensor([s.species_a_mask for s in samples], dtype=torch.float32, device=device),
        "species_b_mask": torch.tensor([s.species_b_mask for s in samples], dtype=torch.float32, device=device),
    }


def _derived_class_logits(
    mhc_a_type_logits: torch.Tensor,
    mhc_b_type_logits: torch.Tensor,
    type_a_mask: torch.Tensor,
    type_b_mask: torch.Tensor,
) -> torch.Tensor:
    a = torch.stack(
        [mhc_a_type_logits[:, TYPE_I], torch.logsumexp(mhc_a_type_logits[:, TYPE_IIA:TYPE_IIB + 1], dim=-1)],
        dim=-1,
    )
    b = torch.stack(
        [mhc_b_type_logits[:, TYPE_I], torch.logsumexp(mhc_b_type_logits[:, TYPE_IIA:TYPE_IIB + 1], dim=-1)],
        dim=-1,
    )
    mask_a = type_a_mask.unsqueeze(-1)
    mask_b = type_b_mask.unsqueeze(-1)
    return (a * mask_a + b * mask_b) / (mask_a + mask_b).clamp(min=1.0)


def _derived_species_logits(
    mhc_a_species_logits: torch.Tensor,
    mhc_b_species_logits: torch.Tensor,
    species_a_mask: torch.Tensor,
    species_b_mask: torch.Tensor,
) -> torch.Tensor:
    mask_a = species_a_mask.unsqueeze(-1)
    mask_b = species_b_mask.unsqueeze(-1)
    return (mhc_a_species_logits * mask_a + mhc_b_species_logits * mask_b) / (
        mask_a + mask_b
    ).clamp(min=1.0)


def _masked_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], int, int]:
    active = mask > 0
    if not bool(active.any().item()):
        return None, 0, 0
    loss = F.cross_entropy(logits[active], target[active])
    preds = torch.argmax(logits[active], dim=-1)
    correct = int((preds == target[active]).sum().item())
    total = int(active.sum().item())
    return loss, correct, total


def _epoch_pass(
    model: Presto,
    loader: DataLoader,
    tokenizer: Tokenizer,
    *,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    loss_sum = 0.0
    batches = 0
    type_correct = 0
    type_total = 0
    species_correct = 0
    species_total = 0
    class_correct = 0
    class_total = 0

    for samples in loader:
        batch = _collate_samples(samples, tokenizer, device=device)
        outputs = model.forward_mhc_only(
            mhc_a_tok=batch["mhc_a_tok"],
            mhc_b_tok=batch["mhc_b_tok"],
        )
        losses: List[torch.Tensor] = []

        loss, correct, total = _masked_ce(
            outputs["mhc_a_type_logits"], batch["type_a_target"], batch["type_a_mask"]
        )
        if loss is not None:
            losses.append(loss)
        type_correct += correct
        type_total += total

        loss, correct, total = _masked_ce(
            outputs["mhc_b_type_logits"], batch["type_b_target"], batch["type_b_mask"]
        )
        if loss is not None:
            losses.append(loss)
        type_correct += correct
        type_total += total

        loss, correct, total = _masked_ce(
            outputs["mhc_a_species_logits"], batch["species_target"], batch["species_a_mask"]
        )
        if loss is not None:
            losses.append(loss)
        species_correct += correct
        species_total += total

        loss, correct, total = _masked_ce(
            outputs["mhc_b_species_logits"], batch["species_target"], batch["species_b_mask"]
        )
        if loss is not None:
            losses.append(loss)
        species_correct += correct
        species_total += total

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
        species_cls_loss = F.cross_entropy(species_logits, batch["species_target"])
        losses.append(species_cls_loss)

        total_loss = torch.stack(losses).mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    parser = argparse.ArgumentParser(description="MHC-only warm-start pretraining")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--index-csv", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="artifacts/mhc_pretrain")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint-name", type=str, default="mhc_pretrain.pt")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    data_dir = Path(args.data_dir)
    index_csv = Path(args.index_csv) if str(args.index_csv).strip() else data_dir / "mhc_index.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")

    samples, dataset_stats = build_mhc_warmstart_samples(
        index_csv,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    if not samples:
        raise RuntimeError("No MHC warm-start samples were built from the index")
    train_samples, val_samples, split_stats = _split_samples(
        samples,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )

    train_dataset = MHCWarmStartDataset(train_samples)
    val_dataset = MHCWarmStartDataset(val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=GroupBalancedBatchSampler(
            train_dataset,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        ),
        collate_fn=lambda items: items,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=lambda items: items,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Presto(
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        use_pmhc_interaction_block=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    tokenizer = Tokenizer()

    print(
        json.dumps(
            {
                "event": "mhc_pretrain_setup",
                "rows": len(samples),
                "train_rows": len(train_samples),
                "val_rows": len(val_samples),
                "device": device,
                "checkpoint_name": str(args.checkpoint_name),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    epochs: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _epoch_pass(
            model, train_loader, tokenizer, device=device, optimizer=optimizer
        )
        val_metrics = _epoch_pass(
            model, val_loader, tokenizer, device=device, optimizer=None
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_type_acc": train_metrics["type_acc"],
            "val_type_acc": val_metrics["type_acc"],
            "train_species_acc": train_metrics["species_acc"],
            "val_species_acc": val_metrics["species_acc"],
            "train_class_acc": train_metrics["class_acc"],
            "val_class_acc": val_metrics["class_acc"],
        }
        epochs.append(epoch_row)
        print(json.dumps({"event": "mhc_pretrain_epoch", **epoch_row}, sort_keys=True), flush=True)

    summary = {
        "config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "d_model": int(args.d_model),
            "n_layers": int(args.n_layers),
            "n_heads": int(args.n_heads),
            "seed": int(args.seed),
            "max_samples": None if int(args.max_samples) <= 0 else int(args.max_samples),
        },
        "dataset_stats": dataset_stats,
        "split_stats": split_stats,
        "epochs": epochs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_model_checkpoint(
        out_dir / str(args.checkpoint_name),
        model=model,
        epoch=int(args.epochs),
        metrics=epochs[-1] if epochs else {},
        run_config=summary["config"],
    )


if __name__ == "__main__":
    main()
