#!/usr/bin/env python
"""Focused allele-panel binding diagnostic.

Trains Presto on a restricted binding-only subset built from the merged TSV
using the same binding-record helper path as the main trainer, then tracks a
small probe panel each epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
import pickle
import random
import shutil
import subprocess
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Sampler

from presto.data import BindingRecord, PrestoCollator, PrestoDataset, create_dataloader
from presto.data.allele_resolver import infer_mhc_class_optional, normalize_mhc_class
from presto.data.cross_source_dedup import UnifiedRecord, classify_assay_type
from presto.data.groove import prepare_mhc_input
from presto.data.mhc_index import build_mhc_sequence_lookup, load_mhc_index
from presto.data.tokenizer import Tokenizer
from presto.models.affinity import (
    AFFINITY_TARGET_ENCODINGS,
    DEFAULT_MAX_AFFINITY_NM,
    affinity_log10_to_target,
    affinity_nm_to_log10,
    binding_prob_from_kd_log10,
    max_log10_nM,
    normalize_binding_target_log10,
    qualifier_for_target_encoding,
)
from presto.models.presto import Presto
from presto.scripts.train_iedb import (
    _normalize_required_aa_sequence,
    ALL_SYNTHETIC_MODES,
    augment_binding_records_with_synthetic_negatives,
    resolve_mhc_sequences_from_index,
)
from presto.training.losses import censor_aware_loss


DEFAULT_ALLELES = ["HLA-A*02:01", "HLA-A*24:02"]
DEFAULT_PROBE_PEPTIDE = "SLLQHLIGL"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 512
DEFAULT_D_MODEL = 128
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 4
DEFAULT_LR = 2.8e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42

MEASUREMENT_PROFILE_ALL = "all_binding_rows"
MEASUREMENT_PROFILE_NUMERIC = "numeric_no_qualitative"
MEASUREMENT_PROFILE_DIRECT = "direct_affinity_only"
MEASUREMENT_PROFILES = {
    MEASUREMENT_PROFILE_ALL,
    MEASUREMENT_PROFILE_NUMERIC,
    MEASUREMENT_PROFILE_DIRECT,
}

DIRECT_MEASUREMENT_TYPES = {
    "dissociation constant kd",
    "half maximal inhibitory concentration (ic50)",
    "half maximal effective concentration (ec50)",
}
NON_QUANTITATIVE_MEASUREMENT_TYPES = {
    "qualitative binding",
    "3d structure",
}
NORMALIZED_MEASUREMENT_FILTERS = {"", "ic50", "kd", "ec50"}
QUALIFIER_FILTERS = {"all", "exact"}
AFFINITY_LOSS_MODES = {"full", "probe_only", "ic50_only", "assay_heads_only"}
LR_SCHEDULE_MODES = {"constant", "warmup_cosine", "onecycle"}
PEPTIDE_POS_MODES = {
    "triple",
    "triple_baseline",
    "abs_only",
    "triple_plus_abs",
    "start_only",
    "end_only",
    "start_plus_end",
    "concat_start_end",
    "concat_start_end_frac",
    "mlp_start_end",
    "mlp_start_end_frac",
}
GROOVE_POS_MODES = {
    "sequential",
    "triple",
    "triple_baseline",
    "abs_only",
    "triple_plus_abs",
    "start_only",
    "end_only",
    "start_plus_end",
    "concat_start_end",
    "concat_start_end_frac",
    "mlp_start_end",
    "mlp_start_end_frac",
}
CORE_REFINEMENT_MODES = {"shared", "class_specific"}
AFFINITY_ASSAY_RESIDUAL_MODES = {
    "legacy",
    "pooled_single_output",
    "shared_base_segment_residual",
    "shared_base_factorized_context_residual",
    "shared_base_factorized_context_plus_segment_residual",
}
DEFAULT_GROOVE_AUDIT_CAP_PER_SOURCE = 128
FOCUSED_DATASET_CACHE_VERSION = 1


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    base_lr: float,
    steps_per_epoch: int,
    epochs: int,
    warmup_fraction: float,
    min_lr_scale: float,
    onecycle_pct_start: float,
):
    schedule = str(schedule).strip().lower()
    if schedule == "constant":
        return None
    total_steps = max(1, int(steps_per_epoch) * int(epochs))
    if schedule == "warmup_cosine":
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
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    if schedule == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=float(base_lr),
            total_steps=total_steps,
            pct_start=float(onecycle_pct_start),
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1_000.0,
        )
    raise ValueError(f"Unsupported lr schedule: {schedule!r}")


def _gradients_are_finite(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            return False
    return True


class _GpuTelemetrySampler:
    """Lightweight GPU utilization sampler using nvidia-smi when available."""

    def __init__(self, *, enabled: bool, interval_s: float = 1.0) -> None:
        self.enabled = bool(enabled) and shutil.which("nvidia-smi") is not None
        self.interval_s = max(float(interval_s), 0.25)
        self._samples: List[Tuple[float, float, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2.0))
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            timestamp = time.perf_counter()
            gpu_util, mem_util = self._sample_once()
            if gpu_util is not None and mem_util is not None:
                with self._lock:
                    self._samples.append((timestamp, gpu_util, mem_util))
            self._stop.wait(self.interval_s)

    @staticmethod
    def _sample_once() -> Tuple[Optional[float], Optional[float]]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return None, None
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            return None, None
        try:
            gpu_str, mem_str = [token.strip() for token in lines[0].split(",", 1)]
            return float(gpu_str), float(mem_str)
        except (TypeError, ValueError):
            return None, None

    def summarize_window(self, start_s: float, end_s: float) -> Dict[str, float]:
        if not self.enabled:
            return {}
        with self._lock:
            window = [
                (gpu_util, mem_util)
                for timestamp, gpu_util, mem_util in self._samples
                if start_s <= timestamp <= end_s
            ]
        if not window:
            return {
                "gpu_util_mean_pct": 0.0,
                "gpu_util_peak_pct": 0.0,
                "gpu_mem_util_mean_pct": 0.0,
                "gpu_mem_util_peak_pct": 0.0,
                "gpu_sample_count": 0.0,
            }
        gpu_vals = [row[0] for row in window]
        mem_vals = [row[1] for row in window]
        return {
            "gpu_util_mean_pct": float(sum(gpu_vals) / len(gpu_vals)),
            "gpu_util_peak_pct": float(max(gpu_vals)),
            "gpu_mem_util_mean_pct": float(sum(mem_vals) / len(mem_vals)),
            "gpu_mem_util_peak_pct": float(max(mem_vals)),
            "gpu_sample_count": float(len(window)),
        }


class StrictAlleleBalancedBatchSampler(Sampler[List[int]]):
    """Strict per-batch allele balancer for focused allele-panel experiments.

    Uses all rows from the largest target-allele pool each epoch and cycles the
    smaller pools with reshuffling so every batch has balanced target-allele
    counts.
    """

    def __init__(
        self,
        dataset: Dataset,
        alleles: Sequence[str],
        batch_size: int,
        *,
        synthetic_fraction: float = 0.0,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.alleles = [str(a).strip() for a in alleles if str(a).strip()]
        if len(self.alleles) < 2:
            raise ValueError("Strict allele-balanced batching requires at least two alleles")
        self.batch_size = max(1, int(batch_size))
        self.synthetic_fraction = max(0.0, min(1.0, float(synthetic_fraction)))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        self._index_by_allele: Dict[str, List[int]] = {allele: [] for allele in self.alleles}
        self._real_index_by_allele: Dict[str, List[int]] = {allele: [] for allele in self.alleles}
        self._synthetic_index_by_allele: Dict[str, List[int]] = {allele: [] for allele in self.alleles}
        for idx, sample in enumerate(getattr(dataset, "samples", [])):
            allele = str(getattr(sample, "primary_allele", "") or "").strip()
            if allele in self._index_by_allele:
                self._index_by_allele[allele].append(idx)
                if str(getattr(sample, "synthetic_kind", "") or "").strip():
                    self._synthetic_index_by_allele[allele].append(idx)
                else:
                    self._real_index_by_allele[allele].append(idx)
        missing = [allele for allele, indices in self._index_by_allele.items() if not indices]
        if missing:
            raise ValueError(
                "Strict allele-balanced batching requires rows for every target allele: "
                + ", ".join(missing)
            )

        base = self.batch_size // len(self.alleles)
        extra = self.batch_size % len(self.alleles)
        if base <= 0:
            raise ValueError(
                f"batch_size={self.batch_size} too small for {len(self.alleles)} target alleles"
            )
        self._slots_by_allele = {
            allele: base + (1 if i < extra else 0)
            for i, allele in enumerate(self.alleles)
        }
        self._synthetic_slots_by_allele = {
            allele: min(
                self._slots_by_allele[allele],
                int(round(self._slots_by_allele[allele] * self.synthetic_fraction)),
            )
            for allele in self.alleles
        }
        self._real_slots_by_allele = {
            allele: self._slots_by_allele[allele] - self._synthetic_slots_by_allele[allele]
            for allele in self.alleles
        }
        self._use_synthetic_balance = any(
            self._synthetic_slots_by_allele[allele] > 0 for allele in self.alleles
        )
        if self._use_synthetic_balance:
            missing_real = [
                allele
                for allele in self.alleles
                if self._real_slots_by_allele[allele] > 0 and not self._real_index_by_allele[allele]
            ]
            missing_synth = [
                allele
                for allele in self.alleles
                if self._synthetic_slots_by_allele[allele] > 0 and not self._synthetic_index_by_allele[allele]
            ]
            if missing_real:
                raise ValueError(
                    "Strict allele-balanced batching requires real rows for every target allele: "
                    + ", ".join(missing_real)
                )
            if missing_synth:
                raise ValueError(
                    "Strict allele-balanced batching requires synthetic rows for every target allele "
                    "when synthetic_fraction > 0: "
                    + ", ".join(missing_synth)
                )
            self._num_batches = max(
                max(
                    math.ceil(
                        len(self._real_index_by_allele[allele])
                        / float(max(self._real_slots_by_allele[allele], 1))
                    )
                    if self._real_slots_by_allele[allele] > 0
                    else 0,
                    math.ceil(
                        len(self._synthetic_index_by_allele[allele])
                        / float(max(self._synthetic_slots_by_allele[allele], 1))
                    )
                    if self._synthetic_slots_by_allele[allele] > 0
                    else 0,
                )
                for allele in self.alleles
            )
        else:
            self._num_batches = max(
                math.ceil(len(self._index_by_allele[allele]) / float(self._slots_by_allele[allele]))
                for allele in self.alleles
            )

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        pools = {allele: list(indices) for allele, indices in self._index_by_allele.items()}
        cursors = {allele: 0 for allele in self.alleles}
        real_pools = {allele: list(indices) for allele, indices in self._real_index_by_allele.items()}
        synth_pools = {
            allele: list(indices) for allele, indices in self._synthetic_index_by_allele.items()
        }
        real_cursors = {allele: 0 for allele in self.alleles}
        synth_cursors = {allele: 0 for allele in self.alleles}
        for allele in self.alleles:
            rng.shuffle(pools[allele])
            rng.shuffle(real_pools[allele])
            rng.shuffle(synth_pools[allele])

        def _draw(
            pool: List[int],
            *,
            slots: int,
            cursor_key: str,
            cursor_map: Dict[str, int],
        ) -> List[int]:
            picks: List[int] = []
            if slots <= 0:
                return picks
            while len(picks) < slots:
                remaining = len(pool) - cursor_map[cursor_key]
                take = min(slots - len(picks), remaining)
                if take > 0:
                    picks.extend(pool[cursor_map[cursor_key] : cursor_map[cursor_key] + take])
                    cursor_map[cursor_key] += take
                if len(picks) < slots:
                    rng.shuffle(pool)
                    cursor_map[cursor_key] = 0
            return picks

        for _ in range(self._num_batches):
            batch: List[int] = []
            for allele in self.alleles:
                if self._use_synthetic_balance:
                    picks = []
                    picks.extend(
                        _draw(
                            real_pools[allele],
                            slots=self._real_slots_by_allele[allele],
                            cursor_key=allele,
                            cursor_map=real_cursors,
                        )
                    )
                    picks.extend(
                        _draw(
                            synth_pools[allele],
                            slots=self._synthetic_slots_by_allele[allele],
                            cursor_key=allele,
                            cursor_map=synth_cursors,
                        )
                    )
                else:
                    picks = _draw(
                        pools[allele],
                        slots=self._slots_by_allele[allele],
                        cursor_key=allele,
                        cursor_map=cursors,
                    )
                batch.extend(picks)
            rng.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
        self._epoch += 1


class CombinedSampleDataset(Dataset):
    """Concatenate fixed real samples with per-epoch synthetic samples."""

    def __init__(self, *datasets: PrestoDataset) -> None:
        self.datasets = [dataset for dataset in datasets if dataset is not None]
        self.samples: List[Any] = []
        self._ranges: List[Tuple[int, int, Dataset]] = []
        cursor = 0
        for dataset in self.datasets:
            size = len(dataset)
            if size <= 0:
                continue
            self.samples.extend(getattr(dataset, "samples", []))
            self._ranges.append((cursor, cursor + size, dataset))
            cursor += size

    def __len__(self) -> int:
        return sum(end - start for start, end, _ in self._ranges)

    def __getitem__(self, idx: int) -> Any:
        for start, end, dataset in self._ranges:
            if start <= idx < end:
                return dataset[idx - start]
        raise IndexError(idx)


@dataclass
class EpochTrainState:
    train_records: List[Any]
    synthetic_records: List[Any]
    train_dataset: Dataset
    train_loader: DataLoader
    synthetic_stats: Dict[str, Any]
    record_groove_audit: Dict[str, Any]
    dataset_groove_audit: Dict[str, Any]


@dataclass(frozen=True)
class DatasetContract:
    source: str
    probe_alleles: Tuple[str, ...]
    training_alleles: Tuple[str, ...]
    train_all_alleles: bool
    train_mhc_class_filter: str
    max_records: int
    sampling_seed: int
    measurement_profile: str
    measurement_type_filter: str
    qualifier_filter: str
    shared_peptides_only: bool
    max_per_allele: int
    split_seed: int
    explicit_probe_peptides: Tuple[str, ...]


@dataclass
class PreparedBindingState:
    real_records: List[BindingRecord]
    real_train_records: List[BindingRecord]
    real_val_records: List[BindingRecord]
    subset_stats: Dict[str, Any]
    shared_peptide_stats: Dict[str, Any]
    probe_allele_counts_after_filter: Dict[str, int]
    balance_stats: Dict[str, Any]
    split_stats: Dict[str, Any]
    mhc_sequences: Dict[str, str]
    mhc_stats: Dict[str, Any]
    probe_support: Dict[str, Any]
    cache_hit: bool = False
    cache_key: str = ""


def _normalize_binding_measurement(text: Optional[str]) -> str:
    key = _measurement_type_key(text)
    if "ic50" in key or "inhibitory concentration" in key:
        return "ic50"
    if "ec50" in key or "effective concentration" in key:
        return "ec50"
    if "kd" in key or "dissociation constant" in key:
        return "kd"
    return "unknown"


def _keep_binding_qualifier(qualifier: Any, mode: str) -> bool:
    if mode == "all":
        return True
    qual = int(qualifier or 0)
    if mode == "exact":
        return qual == 0
    raise ValueError(f"Unsupported qualifier filter: {mode!r}")


def _balance_alleles(
    records: List[Any],
    alleles: List[str],
    max_per_allele: int,
    rng_seed: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Downsample target alleles while preserving shared peptide families."""
    if max_per_allele < 0:
        return list(records), {"skipped": True}

    target_set = {str(a or "").strip() for a in alleles if str(a or "").strip()}
    by_allele: Dict[str, List[Any]] = defaultdict(list)
    other: List[Any] = []
    for rec in records:
        a = str(rec.mhc_allele or "").strip()
        if a in target_set:
            by_allele[a].append(rec)
        else:
            other.append(rec)

    counts_before = {a: len(recs) for a, recs in by_allele.items()}
    unique_peptides_before = {
        allele: len({str(getattr(rec, "peptide", "") or "").strip().upper() for rec in recs})
        for allele, recs in by_allele.items()
    }
    peptide_to_alleles: Dict[str, set[str]] = defaultdict(set)
    for allele, recs in by_allele.items():
        for rec in recs:
            peptide = str(getattr(rec, "peptide", "") or "").strip().upper()
            if peptide:
                peptide_to_alleles[peptide].add(allele)
    shared_peptides = {
        peptide for peptide, peptide_alleles in peptide_to_alleles.items()
        if len(peptide_alleles) >= 2
    }

    if max_per_allele == 0:
        cap = min(len(recs) for recs in by_allele.values()) if by_allele else 0
    else:
        cap = max_per_allele

    rng = random.Random(rng_seed + 31)
    result = list(other)
    counts_after: Dict[str, int] = {}
    unique_peptides_after: Dict[str, int] = {}
    selected_shared_peptides = sorted(shared_peptides)
    if cap > 0 and len(selected_shared_peptides) > cap:
        selected_shared_peptides = sorted(rng.sample(selected_shared_peptides, cap))
    selected_shared_set = set(selected_shared_peptides)

    for allele, recs in by_allele.items():
        if cap == 0:
            continue
        by_peptide: Dict[str, List[Any]] = defaultdict(list)
        for rec in recs:
            peptide = str(getattr(rec, "peptide", "") or "").strip().upper()
            by_peptide[peptide].append(rec)

        guaranteed: List[Any] = []
        guaranteed_ids: set[int] = set()
        for peptide in selected_shared_peptides:
            options = by_peptide.get(peptide)
            if not options:
                continue
            chosen = rng.choice(options)
            guaranteed.append(chosen)
            guaranteed_ids.add(id(chosen))

        remaining_cap = max(0, cap - len(guaranteed))
        extra_pool = [rec for rec in recs if id(rec) not in guaranteed_ids]
        extras = _sample_records_stratified(extra_pool, remaining_cap, rng)
        final_records = guaranteed + extras
        result.extend(final_records)
        counts_after[allele] = len(final_records)
        unique_peptides_after[allele] = len(
            {str(getattr(rec, "peptide", "") or "").strip().upper() for rec in final_records}
        )

    return result, {
        "cap": cap,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "unique_peptides_before": unique_peptides_before,
        "unique_peptides_after": unique_peptides_after,
        "shared_peptides_before": len(shared_peptides),
        "shared_peptides_selected": len(selected_shared_set),
        "shared_rows_guaranteed_per_allele": min(len(selected_shared_set), cap),
        "skipped": False,
    }


def _record_affinity_bin(record: Any) -> str:
    value = getattr(record, "value", None)
    if value is None:
        return "missing"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "missing"
    if numeric <= 50.0:
        return "le_50"
    if numeric <= 500.0:
        return "50_500"
    if numeric <= 5000.0:
        return "500_5k"
    if numeric <= DEFAULT_MAX_AFFINITY_NM:
        return "5k_50k"
    return "gt_50k"


def _sample_records_stratified(
    records: Sequence[Any],
    cap: int,
    rng: random.Random,
) -> List[Any]:
    if cap <= 0 or not records:
        return []
    if cap >= len(records):
        return list(records)

    strata: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    for rec in records:
        key = (
            _measurement_type_key(getattr(rec, "measurement_type", None)),
            _record_affinity_bin(rec),
        )
        strata[key].append(rec)

    total = float(len(records))
    selected: List[Any] = []
    leftovers: List[Any] = []
    quota_by_key: Dict[Tuple[str, str], int] = {}
    fractional: List[Tuple[float, Tuple[str, str]]] = []

    for key, group in sorted(strata.items()):
        raw = len(group) * cap / total
        quota = min(len(group), int(raw))
        quota_by_key[key] = quota
        fractional.append((raw - quota, key))

    allocated = sum(quota_by_key.values())
    remaining = cap - allocated
    for _, key in sorted(fractional, key=lambda item: (item[0], item[1]), reverse=True):
        if remaining <= 0:
            break
        if quota_by_key[key] >= len(strata[key]):
            continue
        quota_by_key[key] += 1
        remaining -= 1

    for key, group in sorted(strata.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        quota = quota_by_key[key]
        selected.extend(shuffled[:quota])
        leftovers.extend(shuffled[quota:])

    if len(selected) < cap and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: cap - len(selected)])
    return selected[:cap]


def _split_records_by_peptide(
    records: Sequence[Any],
    *,
    val_fraction: float,
    seed: int,
    alleles: Optional[Sequence[str]] = None,
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    if not records:
        return [], [], {"train_rows": 0, "val_rows": 0, "shared_peptides_total": 0}

    target_set = {str(a or "").strip() for a in (alleles or []) if str(a or "").strip()}
    peptide_groups: Dict[str, List[Any]] = defaultdict(list)
    peptide_target_alleles: Dict[str, set[str]] = defaultdict(set)
    use_all_alleles = not target_set
    for rec in records:
        peptide = str(getattr(rec, "peptide", "") or "").strip().upper()
        peptide_groups[peptide].append(rec)
        allele = str(getattr(rec, "mhc_allele", "") or "").strip()
        if allele and (use_all_alleles or allele in target_set):
            peptide_target_alleles[peptide].add(allele)

    grouped_keys: Dict[str, List[str]] = defaultdict(list)
    for peptide, allele_set in peptide_target_alleles.items():
        if len(allele_set) >= 2:
            category = "shared"
        elif allele_set:
            category = ",".join(sorted(allele_set))
        else:
            category = "other"
        grouped_keys[category].append(peptide)

    rng = random.Random(seed + 53)
    val_peptides: set[str] = set()
    for category, peptides in grouped_keys.items():
        bucket = list(peptides)
        rng.shuffle(bucket)
        if len(bucket) <= 1:
            continue
        n_val = int(round(len(bucket) * float(val_fraction)))
        n_val = max(1, min(len(bucket) - 1, n_val))
        val_peptides.update(bucket[:n_val])

    if not val_peptides and len(peptide_groups) > 1:
        peptides = sorted(peptide_groups)
        rng.shuffle(peptides)
        val_peptides.add(peptides[0])

    train_records: List[Any] = []
    val_records: List[Any] = []
    for peptide, group in peptide_groups.items():
        target = val_records if peptide in val_peptides else train_records
        target.extend(group)

    train_shared = sum(
        1 for peptide, allele_set in peptide_target_alleles.items()
        if len(allele_set) >= 2 and peptide not in val_peptides
    )
    val_shared = sum(
        1 for peptide, allele_set in peptide_target_alleles.items()
        if len(allele_set) >= 2 and peptide in val_peptides
    )
    return train_records, val_records, {
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "train_peptides": len({str(getattr(r, 'peptide', '') or '').strip().upper() for r in train_records}),
        "val_peptides": len({str(getattr(r, 'peptide', '') or '').strip().upper() for r in val_records}),
        "shared_peptides_total": sum(1 for s in peptide_target_alleles.values() if len(s) >= 2),
        "shared_peptides_train": train_shared,
        "shared_peptides_val": val_shared,
    }


def _filter_shared_peptides_only(
    records: Sequence[Any],
    alleles: Sequence[str],
) -> Tuple[List[Any], Dict[str, Any]]:
    target_set = {str(a or "").strip() for a in alleles if str(a or "").strip()}
    peptide_alleles: Dict[str, set[str]] = defaultdict(set)
    for rec in records:
        allele = str(getattr(rec, "mhc_allele", "") or "").strip()
        peptide = str(getattr(rec, "peptide", "") or "").strip().upper()
        if allele in target_set and peptide:
            peptide_alleles[peptide].add(allele)

    shared_peptides = {
        peptide
        for peptide, seen in peptide_alleles.items()
        if target_set and target_set.issubset(seen)
    }
    filtered = [
        rec for rec in records
        if str(getattr(rec, "peptide", "") or "").strip().upper() in shared_peptides
    ]
    return filtered, {
        "shared_peptides": len(shared_peptides),
        "rows_before": len(records),
        "rows_after": len(filtered),
    }


def _split_csv(text: str) -> List[str]:
    return [token.strip() for token in str(text or "").split(",") if token.strip()]


def _file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _default_cache_dir(data_dir: Path) -> Path:
    return data_dir / ".cache" / "focused_binding"


def _contract_cache_key(
    *,
    contract: DatasetContract,
    merged_tsv: Path,
    index_csv: Path,
) -> str:
    payload = {
        "version": FOCUSED_DATASET_CACHE_VERSION,
        "contract": asdict(contract),
        "merged_tsv": _file_signature(merged_tsv),
        "index_csv": _file_signature(index_csv),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _cache_paths(cache_dir: Path, cache_key: str) -> Tuple[Path, Path]:
    return cache_dir / f"{cache_key}.pkl", cache_dir / f"{cache_key}.meta.json"


def _load_prepared_binding_state_from_cache(
    *,
    cache_dir: Path,
    cache_key: str,
) -> Optional[PreparedBindingState]:
    payload_path, meta_path = _cache_paths(cache_dir, cache_key)
    if not payload_path.exists() or not meta_path.exists():
        return None
    try:
        with payload_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (pickle.PickleError, EOFError, OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        state = PreparedBindingState(
            real_records=list(payload["real_records"]),
            real_train_records=list(payload["real_train_records"]),
            real_val_records=list(payload["real_val_records"]),
            subset_stats=dict(payload["subset_stats"]),
            shared_peptide_stats=dict(payload["shared_peptide_stats"]),
            probe_allele_counts_after_filter=dict(payload["probe_allele_counts_after_filter"]),
            balance_stats=dict(payload["balance_stats"]),
            split_stats=dict(payload["split_stats"]),
            mhc_sequences=dict(payload["mhc_sequences"]),
            mhc_stats=dict(payload["mhc_stats"]),
            probe_support=dict(payload["probe_support"]),
            cache_hit=True,
            cache_key=str(cache_key),
        )
    except KeyError:
        return None
    return state


def _write_prepared_binding_state_to_cache(
    *,
    cache_dir: Path,
    cache_key: str,
    contract: DatasetContract,
    merged_tsv: Path,
    index_csv: Path,
    state: PreparedBindingState,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload_path, meta_path = _cache_paths(cache_dir, cache_key)
    payload = {
        "real_records": list(state.real_records),
        "real_train_records": list(state.real_train_records),
        "real_val_records": list(state.real_val_records),
        "subset_stats": dict(state.subset_stats),
        "shared_peptide_stats": dict(state.shared_peptide_stats),
        "probe_allele_counts_after_filter": dict(state.probe_allele_counts_after_filter),
        "balance_stats": dict(state.balance_stats),
        "split_stats": dict(state.split_stats),
        "mhc_sequences": dict(state.mhc_sequences),
        "mhc_stats": dict(state.mhc_stats),
        "probe_support": dict(state.probe_support),
    }
    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "version": FOCUSED_DATASET_CACHE_VERSION,
        "cache_key": cache_key,
        "contract": asdict(contract),
        "merged_tsv": _file_signature(merged_tsv),
        "index_csv": _file_signature(index_csv),
        "rows": {
            "real": len(state.real_records),
            "train": len(state.real_train_records),
            "val": len(state.real_val_records),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _parse_synthetic_modes(text: str) -> Optional[Tuple[str, ...]]:
    tokens = [token.strip() for token in _split_csv(text) if token.strip()]
    if not tokens:
        return None
    normalized = [token.lower() for token in tokens]
    if normalized == ["all"]:
        return None
    unknown = sorted(set(normalized) - set(ALL_SYNTHETIC_MODES))
    if unknown:
        raise ValueError(f"Unknown synthetic modes: {', '.join(unknown)}")
    return tuple(normalized)


def _resolve_batch_synthetic_fraction(
    *,
    synthetic_negatives: bool,
    negative_ratio: float,
    explicit_fraction: float,
) -> float:
    if not synthetic_negatives:
        return 0.0
    if explicit_fraction >= 0.0:
        return max(0.0, min(1.0, float(explicit_fraction)))
    ratio = max(0.0, float(negative_ratio))
    if ratio <= 0.0:
        return 0.0
    return ratio / (1.0 + ratio)


def _parse_int_csv(text: str, *, default: Sequence[int]) -> Tuple[int, ...]:
    tokens = [token.strip() for token in str(text or "").split(",") if token.strip()]
    if not tokens:
        return tuple(int(v) for v in default)
    values = sorted({max(1, int(token)) for token in tokens})
    return tuple(values)


def _measurement_type_key(text: Optional[str]) -> str:
    return str(text or "").strip().lower()


def _epoch_synthetic_seed(base_seed: int, epoch: int, *, refresh_each_epoch: bool) -> int:
    if not refresh_each_epoch:
        return int(base_seed)
    return int(base_seed) + 1009 * int(epoch)


def _planned_strict_batch_contract(
    *,
    alleles: Sequence[str],
    batch_size: int,
    synthetic_fraction: float,
) -> Dict[str, Any]:
    normalized_alleles = [str(a).strip() for a in alleles if str(a).strip()]
    if not normalized_alleles:
        return {}
    base = int(batch_size) // len(normalized_alleles)
    extra = int(batch_size) % len(normalized_alleles)
    per_allele_slots = {
        allele: base + (1 if idx < extra else 0)
        for idx, allele in enumerate(normalized_alleles)
    }
    synthetic_slots = {
        allele: min(slots, int(round(slots * float(synthetic_fraction))))
        for allele, slots in per_allele_slots.items()
    }
    real_slots = {
        allele: per_allele_slots[allele] - synthetic_slots[allele]
        for allele in normalized_alleles
    }
    return {
        "batch_size": int(batch_size),
        "synthetic_fraction": float(synthetic_fraction),
        "per_allele_slots": per_allele_slots,
        "per_allele_real_slots": real_slots,
        "per_allele_synthetic_slots": synthetic_slots,
    }


def _keep_measurement_type(measurement_type: Optional[str], profile: str) -> bool:
    key = _measurement_type_key(measurement_type)
    if profile == MEASUREMENT_PROFILE_ALL:
        return True
    if profile == MEASUREMENT_PROFILE_NUMERIC:
        return key not in NON_QUANTITATIVE_MEASUREMENT_TYPES
    if profile == MEASUREMENT_PROFILE_DIRECT:
        return key in DIRECT_MEASUREMENT_TYPES
    raise ValueError(f"Unsupported measurement profile: {profile!r}")


def _load_binding_records_from_merged_tsv(
    merged_tsv: Path,
    *,
    alleles: Optional[Sequence[str]] = None,
    mhc_class_filter: Optional[str] = None,
    max_records: Optional[int] = None,
    sampling_seed: int = 42,
) -> Tuple[List[BindingRecord], Dict[str, Any]]:
    target_alleles = {
        str(allele or "").strip()
        for allele in (alleles or ())
        if str(allele or "").strip()
    }
    class_filter = normalize_mhc_class(mhc_class_filter, default=None)
    limit = None if max_records is None or int(max_records) <= 0 else int(max_records)
    rng = random.Random(int(sampling_seed) + 97)

    def _append_record(pool: List[BindingRecord], record: BindingRecord, seen: int) -> int:
        if limit is None:
            pool.append(record)
            return seen
        next_seen = seen + 1
        if len(pool) < limit:
            pool.append(record)
            return next_seen
        replace_idx = rng.randrange(next_seen)
        if replace_idx < limit:
            pool[replace_idx] = record
        return next_seen

    records: List[BindingRecord] = []
    rows_scanned = 0
    seen = 0
    rows_by_allele: Counter[str] = Counter()
    measurement_type_counts: Counter[str] = Counter()
    assay_method_counts: Counter[str] = Counter()

    with merged_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            allele = str(row.get("mhc_allele") or "").strip()
            peptide = _normalize_required_aa_sequence(row.get("peptide"))
            if not allele or not peptide:
                continue
            if target_alleles and allele not in target_alleles:
                continue
            try:
                value = float(str(row.get("value") or "").strip())
            except (TypeError, ValueError):
                continue

            mhc_class = normalize_mhc_class(
                row.get("mhc_class"),
                default=infer_mhc_class_optional(allele),
            )
            if class_filter is not None and mhc_class != class_filter:
                continue

            value_type = str(row.get("value_type") or "").strip()
            source = str(row.get("source") or "").strip() or "unknown"
            unified = UnifiedRecord(
                peptide=peptide,
                mhc_allele=allele,
                mhc_class=mhc_class,
                source=source,
                record_type=str(row.get("record_type") or "").strip(),
                value=value,
                value_type=value_type,
                qualifier=int(str(row.get("qualifier") or "0").strip() or 0),
                response=str(row.get("response") or "").strip(),
                assay_type=str(row.get("assay_type") or "").strip() or None,
                assay_method=str(row.get("assay_method") or "").strip() or None,
                apc_name=str(row.get("apc_name") or "").strip() or None,
                effector_culture_condition=str(row.get("effector_culture_condition") or "").strip() or None,
                apc_culture_condition=str(row.get("apc_culture_condition") or "").strip() or None,
                in_vitro_process_type=str(row.get("in_vitro_process_type") or "").strip() or None,
                in_vitro_responder_cell=str(row.get("in_vitro_responder_cell") or "").strip() or None,
                in_vitro_stimulator_cell=str(row.get("in_vitro_stimulator_cell") or "").strip() or None,
                cdr3_alpha=None,
                cdr3_beta=None,
                trav=None,
                trbv=None,
                species=str(row.get("species") or "").strip() or None,
                antigen_species=str(row.get("antigen_species") or "").strip() or None,
            )
            if classify_assay_type(unified) != "binding_affinity":
                continue

            record = BindingRecord(
                peptide=peptide,
                mhc_allele=allele,
                value=value,
                qualifier=int(str(row.get("qualifier") or "0").strip() or 0),
                measurement_type=value_type or "IC50",
                assay_type=str(row.get("assay_type") or "").strip() or None,
                assay_method=str(row.get("assay_method") or "").strip() or None,
                effector_culture_condition=str(row.get("effector_culture_condition") or "").strip() or None,
                apc_culture_condition=str(row.get("apc_culture_condition") or "").strip() or None,
                mhc_class=mhc_class,
                species=str(row.get("species") or "").strip() or None,
                antigen_species=str(row.get("antigen_species") or "").strip() or None,
                source=source,
            )
            rows_scanned += 1
            seen = _append_record(records, record, seen)
            rows_by_allele[allele] += 1
            measurement_type_counts[record.measurement_type] += 1
            assay_method_counts[str(record.assay_method or "<blank>")] += 1

    unique_peptides_by_allele: Dict[str, set[str]] = defaultdict(set)
    for rec in records:
        unique_peptides_by_allele[str(rec.mhc_allele)].add(str(rec.peptide))
    return records, {
        "rows_scanned": rows_scanned,
        "rows_selected": len(records),
        "allele_filter_count": len(target_alleles),
        "mhc_class_filter": class_filter,
        "rows_by_allele_top": dict(rows_by_allele.most_common(25)),
        "n_alleles": len(rows_by_allele),
        "unique_peptides_by_allele_top": {
            allele: len(peptides)
            for allele, peptides in list(
                sorted(unique_peptides_by_allele.items(), key=lambda item: (-len(item[1]), item[0]))
            )[:25]
        },
        "measurement_type_counts": dict(measurement_type_counts),
        "assay_method_counts_top": dict(assay_method_counts.most_common(20)),
        "cap_sampling": ("reservoir" if limit is not None else "none"),
    }


def _prepare_real_binding_state(
    *,
    merged_tsv: Path,
    index_csv: Path,
    source_filter: str,
    probe_alleles: Sequence[str],
    training_alleles: Sequence[str],
    train_all_alleles: bool,
    train_class_filter: Optional[str],
    max_records: int,
    measurement_profile: str,
    measurement_type_filter: str,
    qualifier_filter: str,
    shared_peptides_only: bool,
    max_per_allele: int,
    seed: int,
    explicit_probe_peptides: Sequence[str],
    cache_dir: Optional[Path],
    use_cache: bool,
) -> PreparedBindingState:
    contract = DatasetContract(
        source=str(source_filter or "").strip().lower(),
        probe_alleles=tuple(str(a).strip() for a in probe_alleles if str(a).strip()),
        training_alleles=tuple(str(a).strip() for a in training_alleles if str(a).strip()),
        train_all_alleles=bool(train_all_alleles),
        train_mhc_class_filter=str(train_class_filter or "all"),
        max_records=int(max_records),
        sampling_seed=int(seed) + 17,
        measurement_profile=str(measurement_profile),
        measurement_type_filter=str(measurement_type_filter or ""),
        qualifier_filter=str(qualifier_filter),
        shared_peptides_only=bool(shared_peptides_only),
        max_per_allele=int(max_per_allele),
        split_seed=int(seed),
        explicit_probe_peptides=tuple(
            str(p).strip().upper() for p in explicit_probe_peptides if str(p).strip()
        ),
    )
    cache_key = _contract_cache_key(contract=contract, merged_tsv=merged_tsv, index_csv=index_csv)
    if use_cache and cache_dir is not None:
        cached = _load_prepared_binding_state_from_cache(cache_dir=cache_dir, cache_key=cache_key)
        if cached is not None:
            return cached

    records, subset_stats = _load_binding_records_from_merged_tsv(
        merged_tsv,
        alleles=training_alleles,
        mhc_class_filter=train_class_filter,
        max_records=(None if int(max_records) <= 0 else int(max_records)),
        sampling_seed=int(seed) + 17,
    )
    if source_filter:
        records = [rec for rec in records if str(rec.source or "").strip().lower() == source_filter]
    records = [rec for rec in records if _keep_measurement_type(rec.measurement_type, measurement_profile)]
    if measurement_type_filter:
        records = [
            rec for rec in records
            if _normalize_binding_measurement(rec.measurement_type) == str(measurement_type_filter)
        ]
    records = [
        rec for rec in records
        if _keep_binding_qualifier(getattr(rec, "qualifier", 0), str(qualifier_filter))
    ]

    real_records = list(records)
    shared_peptide_stats: Dict[str, Any] = {}
    if shared_peptides_only:
        real_records, shared_peptide_stats = _filter_shared_peptides_only(real_records, probe_alleles)
    probe_allele_counts_after_filter = _require_target_allele_coverage(real_records, probe_alleles)
    balance_stats: Dict[str, Any] = {}
    if max_per_allele >= 0:
        real_records, balance_stats = _balance_alleles(
            real_records,
            list(probe_alleles),
            int(max_per_allele),
            rng_seed=int(seed),
        )
        probe_allele_counts_after_filter = _require_target_allele_coverage(real_records, probe_alleles)
    train_records, val_records, split_stats = _split_records_by_peptide(
        real_records,
        val_fraction=0.2,
        seed=int(seed),
        alleles=(probe_alleles if not train_all_alleles else None),
    )
    if not train_records or not val_records:
        raise RuntimeError("Focused binding split must produce both train and val records")

    mhc_sequences, mhc_stats = resolve_mhc_sequences_from_index(
        index_csv=str(index_csv),
        alleles=sorted(
            {
                str(rec.mhc_allele or "").strip()
                for rec in (train_records + val_records)
                if str(rec.mhc_allele or "").strip()
            }
        ),
    )
    probe_support = {
        peptide: _audit_probe_support(merged_tsv, peptide, alleles_of_interest=probe_alleles)
        for peptide in explicit_probe_peptides
    }

    state = PreparedBindingState(
        real_records=list(real_records),
        real_train_records=list(train_records),
        real_val_records=list(val_records),
        subset_stats=subset_stats,
        shared_peptide_stats=shared_peptide_stats,
        probe_allele_counts_after_filter=probe_allele_counts_after_filter,
        balance_stats=balance_stats,
        split_stats=split_stats,
        mhc_sequences=mhc_sequences,
        mhc_stats=mhc_stats,
        probe_support=probe_support,
        cache_hit=False,
        cache_key=cache_key,
    )
    if use_cache and cache_dir is not None:
        _write_prepared_binding_state_to_cache(
            cache_dir=cache_dir,
            cache_key=cache_key,
            contract=contract,
            merged_tsv=merged_tsv,
            index_csv=index_csv,
            state=state,
        )
    return state


def _summarize_binding_records(records: Sequence[Any]) -> Dict[str, Any]:
    by_allele_total: Counter[str] = Counter()
    by_allele_le_500: Counter[str] = Counter()
    measurement_type_counter: Counter[str] = Counter()
    qualifier_counter: Counter[int] = Counter()
    for rec in records:
        allele = str(rec.mhc_allele or "").strip()
        by_allele_total[allele] += 1
        value = getattr(rec, "value", None)
        qualifier = getattr(rec, "qualifier", None)
        if (
            value is not None
            and qualifier is not None
            and float(value) <= 500.0
            and int(qualifier) in (-1, 0)
        ):
            by_allele_le_500[allele] += 1
        measurement_type_counter[str(rec.measurement_type or "")] += 1
        qualifier_counter[int(qualifier or 0)] += 1
    return {
        "rows": len(records),
        "rows_by_allele": dict(by_allele_total),
        "fraction_le_500_by_allele": {
            allele: (by_allele_le_500[allele] / count) if count else 0.0
            for allele, count in by_allele_total.items()
        },
        "measurement_type_counts": dict(measurement_type_counter),
        "qualifier_counts": {str(k): v for k, v in qualifier_counter.items()},
    }


def _require_target_allele_coverage(
    records: Sequence[Any],
    alleles: Sequence[str],
) -> Dict[str, int]:
    counts = {
        str(allele): sum(
            1
            for rec in records
            if str(getattr(rec, "mhc_allele", "") or "").strip() == str(allele)
        )
        for allele in alleles
    }
    missing = [allele for allele, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(
            "Focused binding subset has no retained rows for target alleles: "
            + ", ".join(missing)
        )
    return counts


def _generate_synthetic_train_records(
    *,
    train_records: Sequence[Any],
    mhc_sequences: Mapping[str, str],
    negative_ratio: float,
    seed: int,
    class_i_anchor_strategy: str,
    modes: Optional[Sequence[str]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    train_augmented, train_stats = augment_binding_records_with_synthetic_negatives(
        binding_records=train_records,
        mhc_sequences=dict(mhc_sequences),
        negative_ratio=float(negative_ratio),
        weak_value_min_nM=DEFAULT_MAX_AFFINITY_NM,
        weak_value_max_nM=DEFAULT_MAX_AFFINITY_NM,
        seed=int(seed),
        class_i_no_mhc_beta_ratio=0.0,
        class_i_anchor_strategy=str(class_i_anchor_strategy),
        modes=modes,
    )
    synthetic_records = list(train_augmented[len(train_records) :])
    return synthetic_records, {
        "train": train_stats,
        "val": {"added": 0, "reason": "validation_real_only"},
    }


def _augment_train_records_only(
    *,
    train_records: Sequence[Any],
    val_records: Sequence[Any],
    mhc_sequences: Mapping[str, str],
    negative_ratio: float,
    seed: int,
    class_i_anchor_strategy: str,
    modes: Optional[Sequence[str]] = None,
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    synthetic_records, stats = _generate_synthetic_train_records(
        train_records=train_records,
        mhc_sequences=mhc_sequences,
        negative_ratio=negative_ratio,
        seed=seed,
        class_i_anchor_strategy=class_i_anchor_strategy,
        modes=modes,
    )
    return list(train_records) + list(synthetic_records), list(val_records), stats


def _record_source_bucket(record: Any) -> str:
    source = str(getattr(record, "source", "") or "").strip()
    return source if source else "real"


def _audit_record_groove_preparation(
    records: Sequence[Any],
    *,
    mhc_sequences: Mapping[str, str],
    max_per_source: int = DEFAULT_GROOVE_AUDIT_CAP_PER_SOURCE,
) -> Dict[str, Any]:
    sampled_by_source: Dict[str, int] = defaultdict(int)
    rows_by_source: Counter[str] = Counter()
    audit: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "sampled": 0,
            "missing_sequence": 0,
            "expected_missing_chain": 0,
            "used_fallback": 0,
            "status_counts": Counter(),
            "length_pair_counts": Counter(),
        }
    )
    for rec in records:
        source = _record_source_bucket(rec)
        rows_by_source[source] += 1
        bucket = audit[source]
        bucket["rows"] += 1
        if sampled_by_source[source] >= max_per_source:
            continue
        sampled_by_source[source] += 1
        bucket["sampled"] += 1

        if source in {"synthetic_negative_no_mhc_alpha", "synthetic_negative_no_mhc_beta"}:
            bucket["expected_missing_chain"] += 1
            continue

        allele = str(getattr(rec, "mhc_allele", "") or "").strip()
        mhc_class = normalize_mhc_class(
            getattr(rec, "mhc_class", None),
            default=infer_mhc_class_optional(allele),
        ) or "I"
        direct_seq = str(getattr(rec, "mhc_sequence", "") or "").strip().upper()
        if not direct_seq and allele:
            direct_seq = str(mhc_sequences.get(allele, "") or "").strip().upper()
        if not direct_seq:
            bucket["missing_sequence"] += 1
            continue
        prepared = prepare_mhc_input(
            mhc_class=mhc_class,
            mhc_a=direct_seq,
            mhc_b="",
            allow_fallback_truncation=True,
        )
        if prepared.used_fallback:
            bucket["used_fallback"] += 1
        bucket["status_counts"][
            (str(prepared.groove_status_a or ""), str(prepared.groove_status_b or ""))
        ] += 1
        bucket["length_pair_counts"][
            (len(prepared.groove_half_1), len(prepared.groove_half_2))
        ] += 1

    out: Dict[str, Any] = {}
    for source, stats in audit.items():
        out[source] = {
            "rows": int(stats["rows"]),
            "sampled": int(stats["sampled"]),
            "missing_sequence": int(stats["missing_sequence"]),
            "expected_missing_chain": int(stats["expected_missing_chain"]),
            "used_fallback": int(stats["used_fallback"]),
            "status_counts": {
                f"{a}|{b}": count
                for (a, b), count in stats["status_counts"].most_common(8)
            },
            "length_pair_counts": {
                f"{a}:{b}": count
                for (a, b), count in stats["length_pair_counts"].most_common(8)
            },
        }
    return {
        "rows_by_source": dict(rows_by_source),
        "sources": out,
        "max_per_source": int(max_per_source),
    }


def _audit_dataset_groove_inputs(
    dataset: PrestoDataset,
    *,
    max_per_source: int = DEFAULT_GROOVE_AUDIT_CAP_PER_SOURCE,
) -> Dict[str, Any]:
    sampled_by_source: Dict[str, int] = defaultdict(int)
    rows_by_source: Counter[str] = Counter()
    audit: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "sampled": 0,
            "empty_mhc_a": 0,
            "empty_mhc_b": 0,
            "class_counts": Counter(),
            "length_pair_counts": Counter(),
        }
    )
    for sample in getattr(dataset, "samples", []):
        source = str(getattr(sample, "sample_source", "") or "").strip() or "real"
        rows_by_source[source] += 1
        bucket = audit[source]
        bucket["rows"] += 1
        if sampled_by_source[source] >= max_per_source:
            continue
        sampled_by_source[source] += 1
        bucket["sampled"] += 1
        mhc_class = normalize_mhc_class(getattr(sample, "mhc_class", None), default=None) or "I"
        bucket["class_counts"][mhc_class] += 1
        mhc_a = str(getattr(sample, "mhc_a", "") or "")
        mhc_b = str(getattr(sample, "mhc_b", "") or "")
        if not mhc_a:
            bucket["empty_mhc_a"] += 1
        if not mhc_b:
            bucket["empty_mhc_b"] += 1
        bucket["length_pair_counts"][(len(mhc_a), len(mhc_b))] += 1

    out: Dict[str, Any] = {}
    for source, stats in audit.items():
        out[source] = {
            "rows": int(stats["rows"]),
            "sampled": int(stats["sampled"]),
            "empty_mhc_a": int(stats["empty_mhc_a"]),
            "empty_mhc_b": int(stats["empty_mhc_b"]),
            "class_counts": dict(stats["class_counts"]),
            "length_pair_counts": {
                f"{a}:{b}": count
                for (a, b), count in stats["length_pair_counts"].most_common(8)
            },
        }
    return {
        "rows_by_source": dict(rows_by_source),
        "sources": out,
        "max_per_source": int(max_per_source),
    }


def _build_epoch_train_state(
    *,
    real_train_records: Sequence[Any],
    real_train_dataset: PrestoDataset,
    mhc_sequences: Mapping[str, str],
    synthetic_negatives: bool,
    negative_ratio: float,
    synthetic_seed: int,
    class_i_anchor_strategy: str,
    synthetic_modes: Optional[Sequence[str]],
    batch_size: int,
    balanced: bool,
    seed: int,
    alleles: Sequence[str],
    force_global_balance: bool,
    batch_synthetic_fraction: float,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> EpochTrainState:
    if synthetic_negatives:
        synthetic_records, synthetic_stats = _generate_synthetic_train_records(
            train_records=real_train_records,
            mhc_sequences=mhc_sequences,
            negative_ratio=negative_ratio,
            seed=synthetic_seed,
            class_i_anchor_strategy=class_i_anchor_strategy,
            modes=synthetic_modes,
        )
    else:
        synthetic_records = []
        synthetic_stats = {
            "train": {"added": 0, "added_general": 0},
            "val": {"added": 0, "reason": "validation_real_only"},
        }
    if synthetic_records:
        synthetic_dataset = PrestoDataset(
            binding_records=synthetic_records,
            mhc_sequences=dict(mhc_sequences),
            strict_mhc_resolution=False,
            dataset_index_offset=len(real_train_dataset),
        )
        train_dataset: Dataset = CombinedSampleDataset(real_train_dataset, synthetic_dataset)
    else:
        train_dataset = real_train_dataset
    train_records = list(real_train_records) + list(synthetic_records)
    collator = PrestoCollator()
    train_loader = _create_focused_train_loader(
        train_dataset,
        collator=collator,
        batch_size=batch_size,
        balanced=balanced,
        seed=seed,
        alleles=alleles,
        force_global_balance=force_global_balance,
        synthetic_fraction=batch_synthetic_fraction,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    record_groove_audit = _audit_record_groove_preparation(
        train_records,
        mhc_sequences=mhc_sequences,
    )
    dataset_groove_audit = _audit_dataset_groove_inputs(train_dataset)
    return EpochTrainState(
        train_records,
        list(synthetic_records),
        train_dataset,
        train_loader,
        synthetic_stats,
        record_groove_audit,
        dataset_groove_audit,
    )


def _create_focused_train_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    collator: PrestoCollator,
    balanced: bool,
    seed: int,
    alleles: Sequence[str],
    force_global_balance: bool = False,
    synthetic_fraction: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    normalized_alleles = [str(a).strip() for a in alleles if str(a).strip()]
    if balanced and not force_global_balance and len(normalized_alleles) >= 2:
        batch_sampler = StrictAlleleBalancedBatchSampler(
            dataset=dataset,
            alleles=normalized_alleles,
            batch_size=batch_size,
            synthetic_fraction=float(synthetic_fraction),
            seed=seed,
        )
        loader_kwargs: Dict[str, Any] = {
            "num_workers": max(int(num_workers), 0),
            "pin_memory": bool(pin_memory),
        }
        if int(num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            **loader_kwargs,
        )
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collator=collator,
        balanced=balanced,
        seed=seed,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        prefetch_factor=int(prefetch_factor),
    )


def _select_fit_supported_probe_peptides(
    records: Sequence[Any],
    alleles: Sequence[str],
) -> List[str]:
    if len(alleles) < 2:
        return []
    allele_a = str(alleles[0])
    allele_b = str(alleles[1])
    values_by_peptide: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        allele = str(rec.mhc_allele or "").strip()
        if allele not in {allele_a, allele_b}:
            continue
        if int(rec.qualifier) != 0:
            continue
        values_by_peptide[str(rec.peptide)].setdefault(allele, []).append(
            float(affinity_nm_to_log10(float(rec.value)))
        )

    paired: List[Tuple[float, str]] = []
    for peptide, allele_map in values_by_peptide.items():
        vals_a = allele_map.get(allele_a)
        vals_b = allele_map.get(allele_b)
        if not vals_a or not vals_b:
            continue
        delta = float(median(vals_a) - median(vals_b))
        paired.append((delta, peptide))
    if not paired:
        return []

    paired.sort(key=lambda item: item[0])
    selected: List[str] = [paired[0][1]]
    if paired[-1][1] != selected[0]:
        selected.append(paired[-1][1])
    return selected


def _audit_probe_support(
    merged_tsv: Path,
    peptide: str,
    *,
    alleles_of_interest: Sequence[str],
) -> Dict[str, Any]:
    target = str(peptide or "").strip().upper()
    if not target:
        return {}

    by_record_type: Counter[str] = Counter()
    by_assay_bucket: Counter[str] = Counter()
    by_allele: Counter[str] = Counter()
    quantitative_binding_rows = 0
    alleles_set = {str(a or "").strip() for a in alleles_of_interest if str(a or "").strip()}

    with merged_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("peptide") or "").strip().upper() != target:
                continue
            value_raw = str(row.get("value") or "").strip()
            value = float(value_raw) if value_raw else None
            qualifier_raw = str(row.get("qualifier") or "").strip()
            qualifier = int(qualifier_raw) if qualifier_raw else 0
            rec = UnifiedRecord(
                peptide=target,
                mhc_allele=str(row.get("mhc_allele") or "").strip(),
                mhc_class=str(row.get("mhc_class") or "").strip(),
                source=str(row.get("source") or "").strip(),
                record_type=str(row.get("record_type") or "").strip(),
                value=value,
                value_type=str(row.get("value_type") or "").strip(),
                qualifier=qualifier,
                response=str(row.get("response") or "").strip(),
                assay_type=str(row.get("assay_type") or "").strip() or None,
                assay_method=str(row.get("assay_method") or "").strip() or None,
                apc_name=str(row.get("apc_name") or "").strip() or None,
                effector_culture_condition=str(row.get("effector_culture_condition") or "").strip() or None,
                apc_culture_condition=str(row.get("apc_culture_condition") or "").strip() or None,
                in_vitro_process_type=str(row.get("in_vitro_process_type") or "").strip() or None,
                in_vitro_responder_cell=str(row.get("in_vitro_responder_cell") or "").strip() or None,
                in_vitro_stimulator_cell=str(row.get("in_vitro_stimulator_cell") or "").strip() or None,
                cdr3_alpha=None,
                cdr3_beta=None,
                trav=None,
                trbv=None,
                species=str(row.get("species") or "").strip() or None,
                antigen_species=str(row.get("antigen_species") or "").strip() or None,
            )
            bucket = classify_assay_type(rec)
            allele = rec.mhc_allele or "(none)"
            by_record_type[rec.record_type] += 1
            by_assay_bucket[bucket] += 1
            by_allele[allele] += 1
            if (
                bucket == "binding_affinity"
                and rec.value is not None
                and allele in alleles_set
            ):
                quantitative_binding_rows += 1

    return {
        "peptide": target,
        "total_rows": int(sum(by_record_type.values())),
        "by_record_type": dict(by_record_type),
        "by_assay_bucket": dict(by_assay_bucket),
        "by_allele": dict(by_allele),
        "quantitative_binding_rows_for_target_alleles": quantitative_binding_rows,
    }


def _resolve_allele_sequences(index_csv: Path) -> Dict[str, str]:
    records = load_mhc_index(str(index_csv))
    return build_mhc_sequence_lookup(records)


def _prepare_probe_allele_sequences(
    *,
    probe_alleles: Sequence[str],
    prepared_mhc_sequences: Mapping[str, str],
    index_csv: Path,
) -> Dict[str, str]:
    allele_sequences = dict(prepared_mhc_sequences)
    missing = [
        allele for allele in probe_alleles
        if str(allele or "").strip() and str(allele) not in allele_sequences
    ]
    if not missing:
        return allele_sequences
    fallback_sequences = _resolve_allele_sequences(index_csv)
    for allele in missing:
        seq = fallback_sequences.get(str(allele))
        if seq:
            allele_sequences[str(allele)] = seq
    return allele_sequences


def _find_allele_sequence(allele_sequences: Mapping[str, str], allele: str) -> Optional[str]:
    for key in [allele, allele.replace("HLA-", "")]:
        if key in allele_sequences:
            return allele_sequences[key]
    for key, value in allele_sequences.items():
        if allele in key or key in allele:
            return value
    return None


def _evaluate_probe_panel(
    model: Presto,
    tokenizer: Tokenizer,
    allele_sequences: Mapping[str, str],
    peptides: Sequence[str],
    alleles: Sequence[str],
    device: str,
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    midpoint = float(getattr(model, "binding_midpoint_nM", 500.0))
    scale = float(getattr(model, "binding_log10_scale", 0.35))
    with torch.no_grad():
        for peptide in peptides:
            pep = str(peptide or "").strip().upper()
            if not pep:
                continue
            pep_tok = torch.tensor(tokenizer.encode(pep, max_len=50)).unsqueeze(0).to(device)
            for allele in alleles:
                mhc_seq = _find_allele_sequence(allele_sequences, allele)
                if not mhc_seq:
                    continue
                prepared = prepare_mhc_input(mhc_a=mhc_seq, mhc_class="I")
                mhc_a_tok = torch.tensor(
                    tokenizer.encode(prepared.groove_half_1, max_len=120)
                ).unsqueeze(0).to(device)
                mhc_b_tok = torch.tensor(
                    tokenizer.encode(prepared.groove_half_2, max_len=120)
                ).unsqueeze(0).to(device)
                outputs = model.forward_affinity_only(
                    pep_tok=pep_tok,
                    mhc_a_tok=mhc_a_tok,
                    mhc_b_tok=mhc_b_tok,
                    mhc_class="I",
                    species="human",
                )
                assays = outputs.get("assays", {})
                kd_tensor = assays.get("KD_nM") if isinstance(assays, dict) else None
                ic50_tensor = assays.get("IC50_nM") if isinstance(assays, dict) else None
                kd_log10 = float(kd_tensor[0].item()) if isinstance(kd_tensor, torch.Tensor) else None
                ic50_log10 = (
                    float(ic50_tensor[0].item())
                    if isinstance(ic50_tensor, torch.Tensor)
                    else None
                )
                probe_kd_tensor = outputs.get("binding_affinity_probe_kd")
                probe_kd_log10 = (
                    float(probe_kd_tensor[0].item())
                    if isinstance(probe_kd_tensor, torch.Tensor)
                    else None
                )
                presentation_logit = outputs.get("presentation_logit")
                processing_logit = outputs.get("processing_logit")
                rows.append(
                    {
                        "peptide": pep,
                        "allele": str(allele),
                        "kd_log10": kd_log10,
                        "kd_nM": (float(10.0 ** kd_log10) if kd_log10 is not None else None),
                        "ic50_log10": ic50_log10,
                        "ic50_nM": (
                            float(10.0 ** ic50_log10)
                            if ic50_log10 is not None
                            else None
                        ),
                        "probe_kd_log10": probe_kd_log10,
                        "probe_kd_nM": (
                            float(10.0 ** probe_kd_log10)
                            if probe_kd_log10 is not None
                            else None
                        ),
                        "binding_prob": (
                            float(
                                binding_prob_from_kd_log10(
                                    ic50_log10 if ic50_log10 is not None else kd_log10,
                                    midpoint_nM=midpoint,
                                    log10_scale=scale,
                                )
                            )
                            if (ic50_log10 is not None or kd_log10 is not None)
                            else None
                        ),
                        "presentation_prob": (
                            float(torch.sigmoid(presentation_logit)[0].item())
                            if isinstance(presentation_logit, torch.Tensor)
                            else None
                        ),
                        "processing_prob": (
                            float(torch.sigmoid(processing_logit)[0].item())
                            if isinstance(processing_logit, torch.Tensor)
                            else None
                        ),
                    }
                )
    model.train()
    return rows


def _as_float_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32).reshape(-1)


def _collect_rankable_binding_pairs(
    *,
    pair_index_tensor: Optional[torch.Tensor],
    bind_mask: Optional[torch.Tensor],
    bind_target_log10: Optional[torch.Tensor],
    bind_qual: Optional[torch.Tensor],
    target_gap_min: float,
    pair_prefix: str,
    max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
    metrics: Dict[str, float] = {
        f"{pair_prefix}_pairs": 0.0,
        f"{pair_prefix}_labeled_pairs": 0.0,
        f"{pair_prefix}_exact_pairs": 0.0,
        f"{pair_prefix}_rankable_pairs": 0.0,
    }
    if not (
        isinstance(pair_index_tensor, torch.Tensor)
        and isinstance(bind_mask, torch.Tensor)
        and isinstance(bind_target_log10, torch.Tensor)
        and isinstance(bind_qual, torch.Tensor)
        and pair_index_tensor.ndim == 2
        and pair_index_tensor.shape[-1] == 2
        and pair_index_tensor.numel() > 0
    ):
        return None, None, None, metrics

    pair_index_tensor = pair_index_tensor.long()
    metrics[f"{pair_prefix}_pairs"] = float(pair_index_tensor.shape[0])

    idx_i = pair_index_tensor[:, 0]
    idx_j = pair_index_tensor[:, 1]

    bind_mask_vec = _as_float_vector(bind_mask) > 0
    qual_vec = _as_float_vector(bind_qual).to(dtype=torch.long)
    value_vec = _as_float_vector(bind_target_log10)

    labeled_mask = bind_mask_vec[idx_i] & bind_mask_vec[idx_j]
    metrics[f"{pair_prefix}_labeled_pairs"] = float(labeled_mask.sum().item())
    if not torch.any(labeled_mask):
        return None, None, None, metrics

    idx_i = idx_i[labeled_mask]
    idx_j = idx_j[labeled_mask]
    qual_i = qual_vec[idx_i]
    qual_j = qual_vec[idx_j]
    exact_mask = (qual_i == 0) & (qual_j == 0)
    metrics[f"{pair_prefix}_exact_pairs"] = float(exact_mask.sum().item())

    value_i = value_vec[idx_i]
    value_j = value_vec[idx_j]
    min_log10 = value_vec.new_full(value_i.shape, -3.0)
    max_log10 = value_vec.new_full(value_i.shape, max_log10_nM(max_affinity_nM))

    lower_i = torch.where(qual_i < 0, min_log10, value_i)
    upper_i = torch.where(qual_i > 0, max_log10, value_i)
    lower_j = torch.where(qual_j < 0, min_log10, value_j)
    upper_j = torch.where(qual_j > 0, max_log10, value_j)

    i_stronger = upper_i + float(target_gap_min) <= lower_j
    j_stronger = upper_j + float(target_gap_min) <= lower_i
    rankable_mask = i_stronger | j_stronger
    metrics[f"{pair_prefix}_rankable_pairs"] = float(rankable_mask.sum().item())
    if not torch.any(rankable_mask):
        return None, None, None, metrics

    idx_i = idx_i[rankable_mask]
    idx_j = idx_j[rankable_mask]
    i_stronger = i_stronger[rankable_mask]
    lower_i = lower_i[rankable_mask]
    upper_i = upper_i[rankable_mask]
    lower_j = lower_j[rankable_mask]
    upper_j = upper_j[rankable_mask]

    stronger_idx = torch.where(i_stronger, idx_i, idx_j)
    weaker_idx = torch.where(i_stronger, idx_j, idx_i)
    gaps = torch.maximum(lower_j - upper_i, lower_i - upper_j)
    metrics[f"{pair_prefix}_target_gap_mean"] = float(gaps.mean().item())
    return gaps, stronger_idx, weaker_idx, metrics


def _collect_binding_contrastive_pairs(
    batch: Any,
    *,
    target_gap_min: float,
    max_pairs: int,
    max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
) -> Tuple[List[Tuple[float, int, int]], Dict[str, float]]:
    gaps, stronger_idx, weaker_idx, metrics = _collect_rankable_binding_pairs(
        pair_index_tensor=getattr(batch, "same_peptide_diff_allele_pairs", None),
        bind_mask=getattr(batch, "bind_mask", None),
        bind_target_log10=getattr(batch, "bind_target_log10", None),
        bind_qual=getattr(batch, "bind_qual", None),
        target_gap_min=target_gap_min,
        pair_prefix="out_binding_same_peptide",
        max_affinity_nM=max_affinity_nM,
    )
    if gaps is None or stronger_idx is None or weaker_idx is None or gaps.numel() == 0:
        metrics["out_binding_same_peptide_pairs_used"] = 0.0
        return [], metrics
    if max_pairs > 0 and gaps.numel() > max_pairs:
        topk = torch.topk(gaps, k=max_pairs, largest=True).indices
        gaps = gaps[topk]
        stronger_idx = stronger_idx[topk]
        weaker_idx = weaker_idx[topk]
    metrics["out_binding_same_peptide_pairs_used"] = float(gaps.numel())
    return list(
        zip(
            gaps.detach().cpu().tolist(),
            stronger_idx.detach().cpu().tolist(),
            weaker_idx.detach().cpu().tolist(),
        )
    ), metrics


def _collect_binding_peptide_ranking_pairs(
    batch: Any,
    *,
    target_gap_min: float,
    max_pairs: int,
    max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
) -> Tuple[List[Tuple[float, int, int]], Dict[str, float]]:
    gaps, stronger_idx, weaker_idx, metrics = _collect_rankable_binding_pairs(
        pair_index_tensor=getattr(batch, "same_allele_diff_peptide_pairs", None),
        bind_mask=getattr(batch, "bind_mask", None),
        bind_target_log10=getattr(batch, "bind_target_log10", None),
        bind_qual=getattr(batch, "bind_qual", None),
        target_gap_min=target_gap_min,
        pair_prefix="out_binding_same_allele",
        max_affinity_nM=max_affinity_nM,
    )
    if gaps is None or stronger_idx is None or weaker_idx is None or gaps.numel() == 0:
        metrics["out_binding_same_allele_pairs_used"] = 0.0
        return [], metrics
    if max_pairs > 0 and gaps.numel() > max_pairs:
        topk = torch.topk(gaps, k=max_pairs, largest=True).indices
        gaps = gaps[topk]
        stronger_idx = stronger_idx[topk]
        weaker_idx = weaker_idx[topk]
    metrics["out_binding_same_allele_pairs_used"] = float(gaps.numel())
    return list(
        zip(
            gaps.detach().cpu().tolist(),
            stronger_idx.detach().cpu().tolist(),
            weaker_idx.detach().cpu().tolist(),
        )
    ), metrics


def _compute_binding_contrastive_loss(
    outputs: Dict[str, object],
    batch: Any,
    regularization: Mapping[str, float],
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    gaps, stronger_idx, weaker_idx, metrics = _collect_rankable_binding_pairs(
        pair_index_tensor=getattr(batch, "same_peptide_diff_allele_pairs", None),
        bind_mask=getattr(batch, "bind_mask", None),
        bind_target_log10=getattr(batch, "bind_target_log10", None),
        bind_qual=getattr(batch, "bind_qual", None),
        target_gap_min=float(regularization.get("binding_contrastive_target_gap_min", 0.3)),
        pair_prefix="out_binding_same_peptide",
        max_affinity_nM=float(regularization.get("max_affinity_nM", DEFAULT_MAX_AFFINITY_NM)),
    )
    kd_tensor: Optional[torch.Tensor]
    if bool(regularization.get("binding_rank_use_probe", False)):
        probe_tensor = outputs.get("binding_affinity_probe_kd")
        kd_tensor = probe_tensor if isinstance(probe_tensor, torch.Tensor) else None
    else:
        assays = outputs.get("assays")
        if not isinstance(assays, dict):
            return None, metrics
        kd_tensor = assays.get("KD_nM") if isinstance(assays.get("KD_nM"), torch.Tensor) else None
    if (
        not isinstance(kd_tensor, torch.Tensor)
        or gaps is None
        or stronger_idx is None
        or weaker_idx is None
        or gaps.numel() == 0
    ):
        return None, metrics

    kd_vec = _as_float_vector(kd_tensor)
    margin = float(regularization.get("binding_contrastive_margin", 0.2))
    target_gap_cap = float(regularization.get("binding_contrastive_target_gap_cap", 2.0))
    if target_gap_cap <= 0.0:
        target_gap_cap = margin
    weight = float(regularization.get("binding_contrastive_weight", 0.0))
    pred_gaps = kd_vec[weaker_idx] - kd_vec[stronger_idx]
    required_gaps = torch.clamp(gaps, min=margin, max=target_gap_cap)
    max_pairs = int(regularization.get("binding_contrastive_max_pairs", 64))
    if max_pairs > 0 and pred_gaps.numel() > max_pairs:
        topk = torch.topk(gaps, k=max_pairs, largest=True).indices
        pred_gaps = pred_gaps[topk]
        gaps = gaps[topk]
        required_gaps = required_gaps[topk]
    metrics["out_binding_same_peptide_pairs_used"] = float(pred_gaps.numel())
    metrics["out_binding_contrastive_pred_gap_mean"] = float(pred_gaps.mean().item())
    metrics["out_binding_contrastive_target_gap_mean"] = float(gaps.mean().item())
    metrics["out_binding_contrastive_required_gap_mean"] = float(required_gaps.mean().item())
    if weight <= 0.0:
        return None, metrics
    return weight * torch.relu(required_gaps - pred_gaps).mean(), metrics


def _compute_binding_peptide_ranking_loss(
    outputs: Dict[str, object],
    batch: Any,
    regularization: Mapping[str, float],
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    gaps, stronger_idx, weaker_idx, metrics = _collect_rankable_binding_pairs(
        pair_index_tensor=getattr(batch, "same_allele_diff_peptide_pairs", None),
        bind_mask=getattr(batch, "bind_mask", None),
        bind_target_log10=getattr(batch, "bind_target_log10", None),
        bind_qual=getattr(batch, "bind_qual", None),
        target_gap_min=float(regularization.get("binding_peptide_contrastive_target_gap_min", 0.5)),
        pair_prefix="out_binding_same_allele",
        max_affinity_nM=float(regularization.get("max_affinity_nM", DEFAULT_MAX_AFFINITY_NM)),
    )
    kd_tensor: Optional[torch.Tensor]
    if bool(regularization.get("binding_rank_use_probe", False)):
        probe_tensor = outputs.get("binding_affinity_probe_kd")
        kd_tensor = probe_tensor if isinstance(probe_tensor, torch.Tensor) else None
    else:
        assays = outputs.get("assays")
        if not isinstance(assays, dict):
            return None, metrics
        kd_tensor = assays.get("KD_nM") if isinstance(assays.get("KD_nM"), torch.Tensor) else None
    if (
        not isinstance(kd_tensor, torch.Tensor)
        or gaps is None
        or stronger_idx is None
        or weaker_idx is None
        or gaps.numel() == 0
    ):
        return None, metrics

    kd_vec = _as_float_vector(kd_tensor)
    margin = float(regularization.get("binding_peptide_contrastive_margin", 0.2))
    target_gap_cap = float(regularization.get("binding_peptide_contrastive_target_gap_cap", 2.0))
    if target_gap_cap <= 0.0:
        target_gap_cap = margin
    weight = float(regularization.get("binding_peptide_contrastive_weight", 0.0))
    pred_gaps = kd_vec[weaker_idx] - kd_vec[stronger_idx]
    required_gaps = torch.clamp(gaps, min=margin, max=target_gap_cap)
    max_pairs = int(regularization.get("binding_peptide_contrastive_max_pairs", 128))
    if max_pairs > 0 and pred_gaps.numel() > max_pairs:
        topk = torch.topk(gaps, k=max_pairs, largest=True).indices
        pred_gaps = pred_gaps[topk]
        gaps = gaps[topk]
        required_gaps = required_gaps[topk]
    metrics["out_binding_same_allele_pairs_used"] = float(pred_gaps.numel())
    metrics["out_binding_same_allele_pred_gap_mean"] = float(pred_gaps.mean().item())
    metrics["out_binding_same_allele_required_gap_mean"] = float(required_gaps.mean().item())
    if weight <= 0.0:
        return None, metrics
    return weight * torch.relu(required_gaps - pred_gaps).mean(), metrics


def _mask_from_batch_mapping(
    mapping: Optional[Mapping[str, torch.Tensor]],
    key: str,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mapping is None or key not in mapping:
        return None
    return mapping[key].to(device=device)


def _grad_norm_for_prefixes(model: torch.nn.Module, prefixes: Sequence[str]) -> float:
    total = 0.0
    matched = False
    for name, param in model.named_parameters():
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if param.grad is None:
            continue
        matched = True
        grad = param.grad.detach().float()
        total += float(torch.sum(grad * grad).item())
    return (total ** 0.5) if matched else 0.0


def _affinity_only_loss(
    model: Presto,
    batch: Any,
    device: str,
    regularization: Optional[Mapping[str, float]] = None,
    loss_mode: str = "full",
    affinity_target_encoding: str = "log10",
    max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
    kd_grouping_mode: str = "merged_kd",
) -> tuple[torch.Tensor, Dict[str, float]]:
    batch = batch.to(device)
    outputs = model.forward_affinity_only(
        pep_tok=batch.pep_tok,
        mhc_a_tok=batch.mhc_a_tok,
        mhc_b_tok=batch.mhc_b_tok,
        mhc_class=batch.mhc_class,
        species=batch.processing_species,
        flank_n_tok=batch.flank_n_tok,
        flank_c_tok=batch.flank_c_tok,
        binding_context=getattr(batch, "binding_context", None),
    )

    targets = getattr(batch, "targets", {}) or {}
    target_masks = getattr(batch, "target_masks", {}) or {}
    target_quals = getattr(batch, "target_quals", {}) or {}
    losses: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}

    def _apply_censor_loss(
        name: str,
        pred: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        qual: Optional[torch.Tensor],
    ) -> None:
        if pred is None or target is None or mask is None or qual is None:
            return
        pred_log10 = _as_float_vector(pred)
        target_log10 = normalize_binding_target_log10(
            _as_float_vector(target),
            max_affinity_nM=max_affinity_nM,
            assume_log10=False,
        )
        pred_vec = affinity_log10_to_target(
            pred_log10,
            encoding=affinity_target_encoding,
            max_affinity_nM=max_affinity_nM,
        )
        target_vec = affinity_log10_to_target(
            target_log10,
            encoding=affinity_target_encoding,
            max_affinity_nM=max_affinity_nM,
        )
        mask_vec = _as_float_vector(mask).to(device=pred_vec.device)
        qual_vec = qualifier_for_target_encoding(
            _as_float_vector(qual).to(device=pred_vec.device, dtype=torch.long),
            encoding=affinity_target_encoding,
        )
        support = float(mask_vec.sum().item())
        if support <= 0.0:
            return
        loss_vec = censor_aware_loss(
            pred_vec,
            target_vec,
            qual_vec,
            reduction="none",
        )
        losses.append((loss_vec * mask_vec).sum() / (mask_vec.sum() + 1e-8))
        metrics[f"support_{name}"] = support

    if loss_mode == "probe_only":
        _apply_censor_loss(
            "binding_affinity_probe",
            outputs.get("binding_affinity_probe_kd"),
            getattr(batch, "bind_target", None),
            getattr(batch, "bind_mask", None),
            getattr(batch, "bind_qual", None),
        )
    elif loss_mode == "ic50_only":
        _apply_censor_loss(
            "binding_ic50",
            outputs.get("assays", {}).get("IC50_nM"),
            targets.get("binding_ic50"),
            _mask_from_batch_mapping(target_masks, "binding_ic50", device=batch.pep_tok.device),
            _mask_from_batch_mapping(target_quals, "binding_ic50", device=batch.pep_tok.device),
        )
    elif loss_mode == "assay_heads_only":
        if str(kd_grouping_mode) == "split_kd_proxy":
            _apply_censor_loss(
                "binding_kd_direct",
                outputs.get("assays", {}).get("KD_nM"),
                targets.get("binding_kd_direct"),
                _mask_from_batch_mapping(target_masks, "binding_kd_direct", device=batch.pep_tok.device),
                _mask_from_batch_mapping(target_quals, "binding_kd_direct", device=batch.pep_tok.device),
            )
            _apply_censor_loss(
                "binding_kd_proxy_ic50",
                outputs.get("assays", {}).get("KD_proxy_ic50_nM"),
                targets.get("binding_kd_proxy_ic50"),
                _mask_from_batch_mapping(target_masks, "binding_kd_proxy_ic50", device=batch.pep_tok.device),
                _mask_from_batch_mapping(target_quals, "binding_kd_proxy_ic50", device=batch.pep_tok.device),
            )
            _apply_censor_loss(
                "binding_kd_proxy_ec50",
                outputs.get("assays", {}).get("KD_proxy_ec50_nM"),
                targets.get("binding_kd_proxy_ec50"),
                _mask_from_batch_mapping(target_masks, "binding_kd_proxy_ec50", device=batch.pep_tok.device),
                _mask_from_batch_mapping(target_quals, "binding_kd_proxy_ec50", device=batch.pep_tok.device),
            )
        else:
            _apply_censor_loss(
                "binding_kd",
                outputs.get("assays", {}).get("KD_nM"),
                targets.get("binding_kd"),
                _mask_from_batch_mapping(target_masks, "binding_kd", device=batch.pep_tok.device),
                _mask_from_batch_mapping(target_quals, "binding_kd", device=batch.pep_tok.device),
            )
        _apply_censor_loss(
            "binding_ic50",
            outputs.get("assays", {}).get("IC50_nM"),
            targets.get("binding_ic50"),
            _mask_from_batch_mapping(target_masks, "binding_ic50", device=batch.pep_tok.device),
            _mask_from_batch_mapping(target_quals, "binding_ic50", device=batch.pep_tok.device),
        )
        _apply_censor_loss(
            "binding_ec50",
            outputs.get("assays", {}).get("EC50_nM"),
            targets.get("binding_ec50"),
            _mask_from_batch_mapping(target_masks, "binding_ec50", device=batch.pep_tok.device),
            _mask_from_batch_mapping(target_quals, "binding_ec50", device=batch.pep_tok.device),
        )
    elif loss_mode == "full":
        _apply_censor_loss(
            "binding",
            outputs.get("assays", {}).get("KD_nM"),
            getattr(batch, "bind_target", None),
            getattr(batch, "bind_mask", None),
            getattr(batch, "bind_qual", None),
        )
        _apply_censor_loss(
            "binding_affinity_probe",
            outputs.get("binding_affinity_probe_kd"),
            getattr(batch, "bind_target", None),
            getattr(batch, "bind_mask", None),
            getattr(batch, "bind_qual", None),
        )
        if str(kd_grouping_mode) == "split_kd_proxy":
            _apply_censor_loss(
                "binding_kd_direct",
                outputs.get("assays", {}).get("KD_nM"),
                targets.get("binding_kd_direct"),
                _mask_from_batch_mapping(target_masks, "binding_kd_direct", device=batch.pep_tok.device),
                _mask_from_batch_mapping(target_quals, "binding_kd_direct", device=batch.pep_tok.device),
            )
            _apply_censor_loss(
                "binding_kd_proxy_ic50",
                outputs.get("assays", {}).get("KD_proxy_ic50_nM"),
                targets.get("binding_kd_proxy_ic50"),
                _mask_from_batch_mapping(target_masks, "binding_kd_proxy_ic50", device=batch.pep_tok.device),
                _mask_from_batch_mapping(target_quals, "binding_kd_proxy_ic50", device=batch.pep_tok.device),
            )
            _apply_censor_loss(
                "binding_kd_proxy_ec50",
                outputs.get("assays", {}).get("KD_proxy_ec50_nM"),
                targets.get("binding_kd_proxy_ec50"),
                _mask_from_batch_mapping(target_masks, "binding_kd_proxy_ec50", device=batch.pep_tok.device),
                _mask_from_batch_mapping(target_quals, "binding_kd_proxy_ec50", device=batch.pep_tok.device),
            )
        else:
            _apply_censor_loss(
                "binding_kd",
                outputs.get("assays", {}).get("KD_nM"),
                targets.get("binding_kd"),
                _mask_from_batch_mapping(target_masks, "binding_kd", device=batch.pep_tok.device),
                _mask_from_batch_mapping(target_quals, "binding_kd", device=batch.pep_tok.device),
            )
        _apply_censor_loss(
            "binding_ic50",
            outputs.get("assays", {}).get("IC50_nM"),
            targets.get("binding_ic50"),
            _mask_from_batch_mapping(target_masks, "binding_ic50", device=batch.pep_tok.device),
            _mask_from_batch_mapping(target_quals, "binding_ic50", device=batch.pep_tok.device),
        )
        _apply_censor_loss(
            "binding_ec50",
            outputs.get("assays", {}).get("EC50_nM"),
            targets.get("binding_ec50"),
            _mask_from_batch_mapping(target_masks, "binding_ec50", device=batch.pep_tok.device),
            _mask_from_batch_mapping(target_quals, "binding_ec50", device=batch.pep_tok.device),
        )
    else:
        raise ValueError(f"Unsupported affinity loss mode: {loss_mode!r}")

    total = torch.stack(losses).mean() if losses else None
    contrastive_loss, contrastive_metrics = _compute_binding_contrastive_loss(
        outputs=outputs,
        batch=batch,
        regularization=regularization or {},
    )
    metrics.update(contrastive_metrics)
    if contrastive_loss is not None:
        total = contrastive_loss if total is None else (total + contrastive_loss)
    peptide_ranking_loss, peptide_ranking_metrics = _compute_binding_peptide_ranking_loss(
        outputs=outputs,
        batch=batch,
        regularization=regularization or {},
    )
    metrics.update(peptide_ranking_metrics)
    if peptide_ranking_loss is not None:
        total = peptide_ranking_loss if total is None else (total + peptide_ranking_loss)
    if total is None:
        raise RuntimeError("Affinity-only batch produced no supervised losses")
    metrics["loss_tasks"] = float(len(losses))
    return total, metrics


def _mean_affinity_loss(
    model: Presto,
    loader: torch.utils.data.DataLoader,
    device: str,
    regularization: Optional[Mapping[str, float]] = None,
    loss_mode: str = "full",
    affinity_target_encoding: str = "log10",
    max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
    kd_grouping_mode: str = "merged_kd",
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    skipped = 0
    with torch.no_grad():
        for batch in loader:
            try:
                loss, _ = _affinity_only_loss(
                    model,
                    batch,
                    device,
                    regularization=regularization,
                    loss_mode=loss_mode,
                    affinity_target_encoding=affinity_target_encoding,
                    max_affinity_nM=max_affinity_nM,
                    kd_grouping_mode=kd_grouping_mode,
                )
            except RuntimeError as exc:
                if "no supervised losses" not in str(exc).lower():
                    raise
                skipped += 1
                continue
            total += float(loss.detach().item())
            batches += 1
    model.train()
    if batches == 0:
        raise RuntimeError(
            f"Validation loader produced no supervised batches for loss_mode={loss_mode!r}, "
            f"kd_grouping_mode={kd_grouping_mode!r}; skipped={skipped}"
        )
    return total / batches


def _write_probe_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
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


def _write_probe_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped: Dict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        y_value = row.get("ic50_nM", row.get("kd_nM"))
        if y_value in (None, ""):
            continue
        grouped[(str(row["peptide"]), str(row["allele"]))].append(
            (int(row["epoch"]), float(y_value))
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    for (peptide, allele), vals in sorted(grouped.items()):
        vals = sorted(vals, key=lambda item: item[0])
        ax.plot(
            [v[0] for v in vals],
            [v[1] for v in vals],
            marker="o",
            label=f"{peptide} | {allele}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Predicted Affinity (nM)")
    ax.set_title("Focused Binding Probe Trajectory")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_summary_artifacts(
    *,
    out_dir: Path,
    summary: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    write_probe_plot: bool = True,
) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_probe_csv(out_dir / "probe_affinity_over_epochs.csv", probe_rows)
    (out_dir / "probe_affinity_over_epochs.json").write_text(
        json.dumps(list(probe_rows), indent=2),
        encoding="utf-8",
    )
    if write_probe_plot:
        _write_probe_plot(out_dir / "probe_affinity_over_epochs.png", probe_rows)


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


def _filter_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
    model_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped_missing: List[str] = []
    skipped_shape: List[str] = []

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            skipped_missing.append(key)
            continue
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target.shape):
            skipped_shape.append(key)
            continue
        compatible[key] = value

    return compatible, {
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
    }


def main() -> None:
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Focused allele-panel binding diagnostic")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="artifacts/focused_binding_probe")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Optional prepared-dataset cache directory. Default: <data-dir>/.cache/focused_binding",
    )
    parser.add_argument("--dataset-cache", dest="dataset_cache", action="store_true")
    parser.add_argument("--no-dataset-cache", dest="dataset_cache", action="store_false")
    parser.set_defaults(dataset_cache=True)
    parser.add_argument(
        "--alleles",
        type=str,
        default=",".join(DEFAULT_ALLELES),
        help="Probe/evaluation allele panel. Also used as the training panel unless --train-all-alleles is set.",
    )
    parser.add_argument("--probe-peptide", type=str, default=DEFAULT_PROBE_PEPTIDE)
    parser.add_argument(
        "--extra-probe-peptides",
        type=str,
        default="",
        help="Optional comma-separated extra probe peptides to log every epoch.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=("default", "highest", "high", "medium"),
        default="high",
        help="Optional torch float32 matmul precision override.",
    )
    parser.add_argument("--allow-tf32", dest="allow_tf32", action="store_true")
    parser.add_argument("--no-allow-tf32", dest="allow_tf32", action="store_false")
    parser.set_defaults(allow_tf32=True)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument(
        "--design-id",
        type=str,
        default="baseline",
        help="Optional benchmark design label recorded in summary artifacts.",
    )
    parser.add_argument(
        "--peptide-pos-mode",
        type=str,
        choices=sorted(PEPTIDE_POS_MODES),
        default="triple",
        help="Peptide positional encoding mode.",
    )
    parser.add_argument(
        "--groove-pos-mode",
        type=str,
        choices=sorted(GROOVE_POS_MODES),
        default="sequential",
        help="Groove positional encoding mode.",
    )
    parser.add_argument(
        "--affinity-target-encoding",
        type=str,
        choices=sorted(AFFINITY_TARGET_ENCODINGS),
        default="log10",
        help="Training-space encoding for quantitative affinity losses.",
    )
    parser.add_argument(
        "--max-affinity-nm",
        type=float,
        default=DEFAULT_MAX_AFFINITY_NM,
        help="Upper affinity cap used for normalization and bounded outputs.",
    )
    parser.add_argument(
        "--binding-core-lengths",
        type=str,
        default="9",
        help="Comma-separated candidate core lengths. Example: 8,9,10,11",
    )
    parser.add_argument(
        "--binding-core-refinement",
        type=str,
        choices=sorted(CORE_REFINEMENT_MODES),
        default="shared",
        help="Core-window refinement head mode.",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr-schedule", type=str, choices=sorted(LR_SCHEDULE_MODES), default="constant")
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", type=str, default="iedb")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--train-mhc-class-filter",
        type=str,
        choices=("all", "I", "II"),
        default="all",
        help="Optional MHC class filter applied to the training/validation binding rows.",
    )
    parser.add_argument(
        "--train-all-alleles",
        action="store_true",
        help="Train on all alleles matching --train-mhc-class-filter instead of only the probe/eval allele panel.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional Presto checkpoint used to warm-start the focused affinity run.",
    )
    parser.add_argument(
        "--measurement-profile",
        type=str,
        choices=sorted(MEASUREMENT_PROFILES),
        default=MEASUREMENT_PROFILE_NUMERIC,
    )
    parser.add_argument(
        "--measurement-type-filter",
        type=str,
        default="",
        choices=sorted(NORMALIZED_MEASUREMENT_FILTERS),
        help="Optional normalized assay-type filter after profile selection: ic50, kd, ec50.",
    )
    parser.add_argument(
        "--qualifier-filter",
        type=str,
        default="all",
        choices=sorted(QUALIFIER_FILTERS),
        help="Optional qualifier filter after assay filtering. 'exact' keeps only qualifier=0 rows.",
    )
    parser.add_argument(
        "--shared-peptides-only",
        action="store_true",
        help="Keep only peptide families observed on all target alleles after filtering.",
    )
    parser.add_argument(
        "--max-per-allele",
        type=int,
        default=-1,
        help="Cap each allele to this many records. 0 = auto-balance to minority allele count. -1 = no balancing.",
    )
    parser.add_argument(
        "--class-i-anchor-strategy",
        type=str,
        choices=("none", "property_opposite"),
        default="none",
        help="Optional class-I anchor-aware strategy for focused synthetic negatives.",
    )
    parser.add_argument("--synthetic-negatives", dest="synthetic_negatives", action="store_true")
    parser.add_argument("--no-synthetic-negatives", dest="synthetic_negatives", action="store_false")
    parser.set_defaults(synthetic_negatives=False)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--batch-synthetic-fraction",
        type=float,
        default=-1.0,
        help="Exact synthetic fraction per train batch. Negative values derive the fraction from negative_ratio.",
    )
    parser.add_argument(
        "--synthetic-modes",
        type=str,
        default="",
        help="Optional comma-separated synthetic mode subset. Empty or 'all' uses all modes.",
    )
    parser.add_argument(
        "--synthetic-refresh-each-epoch",
        dest="synthetic_refresh_each_epoch",
        action="store_true",
        help="Regenerate synthetic negatives from the fixed real-train split every epoch.",
    )
    parser.add_argument(
        "--static-synthetic-negatives",
        dest="synthetic_refresh_each_epoch",
        action="store_false",
        help="Generate train synthetics once and reuse them for all epochs.",
    )
    parser.set_defaults(synthetic_refresh_each_epoch=True)
    parser.add_argument("--balanced-batches", action="store_true", default=True)
    parser.add_argument(
        "--affinity-loss-mode",
        type=str,
        choices=sorted(AFFINITY_LOSS_MODES),
        default="full",
        help=(
            "Focused affinity loss contract. "
            "'probe_only' uses only the direct affinity head. "
            "'ic50_only' supervises only assays.IC50_nM. "
            "'assay_heads_only' supervises only KD/IC50/EC50 heads with censor-aware losses."
        ),
    )
    parser.add_argument(
        "--affinity-assay-mode",
        type=str,
        choices=("legacy", "score_context"),
        default="legacy",
        help="Assay-path mode for affinity outputs. 'legacy' restores the stronger pre-score/context assay route.",
    )
    parser.add_argument(
        "--affinity-assay-residual-mode",
        type=str,
        choices=sorted(AFFINITY_ASSAY_RESIDUAL_MODES),
        default="legacy",
        help="Residual/bias mode for KD/IC50/EC50 assay heads.",
    )
    parser.add_argument(
        "--kd-grouping-mode",
        type=str,
        choices=("merged_kd", "split_kd_proxy"),
        default="merged_kd",
        help="Whether direct KD and proxy-KD assay families share one KD output or use split proxy outputs.",
    )
    parser.add_argument(
        "--binding-kinetic-input-mode",
        type=str,
        choices=("affinity_vec", "interaction_vec", "fused"),
        default="affinity_vec",
        help=(
            "Input route for the BindingModule kinetic branch. "
            "'interaction_vec' restores the older 512-d kinetic input, "
            "'affinity_vec' keeps the cleaned path, and 'fused' combines both."
        ),
    )
    parser.add_argument(
        "--binding-direct-segment-mode",
        type=str,
        choices=("off", "affinity_residual", "affinity_stability_residual", "gated_affinity"),
        default="off",
        help=(
            "Direct pooled {peptide, groove1, groove2} branch into the canonical affinity path."
        ),
    )
    parser.add_argument(
        "--binding-contrastive-weight",
        type=float,
        default=1.0,
        help="Weight for same-peptide/different-allele binding ranking loss.",
    )
    parser.add_argument(
        "--binding-contrastive-margin",
        type=float,
        default=0.2,
        help="Required predicted log10(KD) margin for stronger-vs-weaker allele pairs.",
    )
    parser.add_argument(
        "--binding-contrastive-target-gap-min",
        type=float,
        default=0.3,
        help="Minimum observed log10(KD) gap required before a pair is used for ranking.",
    )
    parser.add_argument(
        "--binding-contrastive-target-gap-cap",
        type=float,
        default=2.0,
        help="Maximum target log10(KD) gap enforced by the binding ranking loss.",
    )
    parser.add_argument(
        "--binding-contrastive-max-pairs",
        type=int,
        default=64,
        help="Maximum same-peptide/different-allele pairs per batch used for ranking.",
    )
    parser.add_argument(
        "--binding-peptide-contrastive-weight",
        type=float,
        default=0.5,
        help="Weight for same-allele/different-peptide binding ranking loss.",
    )
    parser.add_argument(
        "--binding-peptide-contrastive-margin",
        type=float,
        default=0.2,
        help="Required predicted log10(KD) margin for stronger-vs-weaker peptide pairs within an allele.",
    )
    parser.add_argument(
        "--binding-peptide-contrastive-target-gap-min",
        type=float,
        default=0.5,
        help="Minimum observed log10(KD) gap required before a same-allele peptide pair is used.",
    )
    parser.add_argument(
        "--binding-peptide-contrastive-target-gap-cap",
        type=float,
        default=2.0,
        help="Maximum target log10(KD) gap enforced by same-allele peptide ranking.",
    )
    parser.add_argument(
        "--binding-peptide-contrastive-max-pairs",
        type=int,
        default=128,
        help="Maximum same-allele/different-peptide pairs per batch used for ranking.",
    )
    parser.add_argument(
        "--probe-plot-frequency",
        choices=("epoch", "final", "off"),
        default="final",
        help="How often to rewrite the probe trajectory PNG during training.",
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    random.seed(int(args.seed))
    synthetic_modes = _parse_synthetic_modes(str(args.synthetic_modes))
    batch_synthetic_fraction = _resolve_batch_synthetic_fraction(
        synthetic_negatives=bool(args.synthetic_negatives),
        negative_ratio=float(args.negative_ratio),
        explicit_fraction=float(args.batch_synthetic_fraction),
    )
    if str(args.matmul_precision) != "default":
        torch.set_float32_matmul_precision(str(args.matmul_precision))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_tsv = data_dir / "merged_deduped.tsv"
    index_csv = data_dir / "mhc_index.csv"
    if not merged_tsv.exists():
        raise FileNotFoundError(f"Merged TSV not found: {merged_tsv}")
    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")

    probe_alleles = _split_csv(args.alleles)
    if not probe_alleles:
        raise ValueError("At least one allele is required")
    train_class_filter = None if str(args.train_mhc_class_filter) == "all" else str(args.train_mhc_class_filter)
    if args.train_all_alleles and args.shared_peptides_only:
        raise ValueError("--shared-peptides-only is only valid for explicit allele-panel training")
    if args.train_all_alleles and int(args.max_per_allele) >= 0:
        raise ValueError("--max-per-allele is only valid for explicit allele-panel training")

    training_alleles = [] if bool(args.train_all_alleles) else list(probe_alleles)
    source_filter = str(args.source or "").strip().lower()
    explicit_probe_peptides = [str(args.probe_peptide).strip().upper()]
    for peptide in _split_csv(str(args.extra_probe_peptides)):
        peptide_key = str(peptide).strip().upper()
        if peptide_key and peptide_key not in explicit_probe_peptides:
            explicit_probe_peptides.append(peptide_key)

    cache_dir = Path(args.cache_dir).expanduser() if str(args.cache_dir).strip() else _default_cache_dir(data_dir)
    setup_stage_timings: Dict[str, float] = {}
    setup_stage_start = time.perf_counter()
    prepared_state = _prepare_real_binding_state(
        merged_tsv=merged_tsv,
        index_csv=index_csv,
        source_filter=source_filter,
        probe_alleles=probe_alleles,
        training_alleles=training_alleles,
        train_all_alleles=bool(args.train_all_alleles),
        train_class_filter=train_class_filter,
        max_records=int(args.max_records),
        measurement_profile=str(args.measurement_profile),
        measurement_type_filter=str(args.measurement_type_filter),
        qualifier_filter=str(args.qualifier_filter),
        shared_peptides_only=bool(args.shared_peptides_only),
        max_per_allele=int(args.max_per_allele),
        seed=int(args.seed),
        explicit_probe_peptides=explicit_probe_peptides,
        cache_dir=cache_dir,
        use_cache=bool(args.dataset_cache),
    )
    setup_stage_timings["prepare_real_binding_state_s"] = time.perf_counter() - setup_stage_start

    real_records = list(prepared_state.real_records)
    subset_stats = prepared_state.subset_stats
    shared_peptide_stats = prepared_state.shared_peptide_stats
    probe_allele_counts_after_filter = prepared_state.probe_allele_counts_after_filter
    balance_stats = prepared_state.balance_stats
    split_stats = prepared_state.split_stats
    mhc_sequences = prepared_state.mhc_sequences
    mhc_stats = prepared_state.mhc_stats
    real_train_records = list(prepared_state.real_train_records)
    real_val_records = list(prepared_state.real_val_records)
    if not real_train_records or not real_val_records:
        raise RuntimeError("No binding records remain after split/augmentation")

    dataset_stage_start = time.perf_counter()
    real_train_dataset = PrestoDataset(
        binding_records=real_train_records,
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=False,
    )
    val_dataset = PrestoDataset(
        binding_records=real_val_records,
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=False,
    )
    setup_stage_timings["dataset_build_s"] = time.perf_counter() - dataset_stage_start

    collator = PrestoCollator()
    loader_stage_start = time.perf_counter()
    val_loader = create_dataloader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collator=collator,
        balanced=False,
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor),
    )
    setup_stage_timings["val_loader_build_s"] = time.perf_counter() - loader_stage_start

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_sampler = _GpuTelemetrySampler(enabled=(device == "cuda"))
    gpu_sampler.start()
    binding_core_lengths = _parse_int_csv(str(args.binding_core_lengths), default=(9,))
    model_stage_start = time.perf_counter()
    model = Presto(
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        max_affinity_nM=float(args.max_affinity_nm),
        use_pmhc_interaction_block=True,
        peptide_pos_mode=str(args.peptide_pos_mode),
        groove_pos_mode=str(args.groove_pos_mode),
        core_window_lengths=binding_core_lengths,
        core_refinement_mode=str(args.binding_core_refinement),
        affinity_assay_mode=str(args.affinity_assay_mode),
        affinity_assay_residual_mode=str(args.affinity_assay_residual_mode),
        kd_grouping_mode=str(args.kd_grouping_mode),
        binding_kinetic_input_mode=str(args.binding_kinetic_input_mode),
        binding_direct_segment_mode=str(args.binding_direct_segment_mode),
    ).to(device)
    warm_start_payload: Optional[Dict[str, Any]] = None
    if str(args.init_checkpoint or "").strip():
        warm_start_payload = _load_checkpoint_payload(str(args.init_checkpoint).strip())
        state_dict = warm_start_payload.get("model_state_dict", warm_start_payload)
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid model state_dict")
        compatible_state_dict, warm_start_filter_stats = _filter_compatible_state_dict(
            model,
            state_dict,
        )
        model.load_state_dict(compatible_state_dict, strict=False)
    else:
        warm_start_filter_stats = {"skipped_missing": [], "skipped_shape": []}
    if bool(args.torch_compile):
        model = torch.compile(model, mode="reduce-overhead")
    setup_stage_timings["model_init_and_warm_start_s"] = time.perf_counter() - model_stage_start
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = None
    param_count = int(sum(p.numel() for p in model.parameters()))
    regularization_cfg = {
        "binding_contrastive_weight": float(args.binding_contrastive_weight),
        "binding_contrastive_margin": float(args.binding_contrastive_margin),
        "binding_contrastive_target_gap_min": float(args.binding_contrastive_target_gap_min),
        "binding_contrastive_target_gap_cap": float(args.binding_contrastive_target_gap_cap),
        "binding_contrastive_max_pairs": int(args.binding_contrastive_max_pairs),
        "binding_peptide_contrastive_weight": float(args.binding_peptide_contrastive_weight),
        "binding_peptide_contrastive_margin": float(args.binding_peptide_contrastive_margin),
        "binding_peptide_contrastive_target_gap_min": float(
            args.binding_peptide_contrastive_target_gap_min
        ),
        "binding_peptide_contrastive_target_gap_cap": float(
            args.binding_peptide_contrastive_target_gap_cap
        ),
        "binding_peptide_contrastive_max_pairs": int(args.binding_peptide_contrastive_max_pairs),
        "binding_rank_use_probe": bool(str(args.affinity_loss_mode) == "probe_only"),
    }

    probe_stage_start = time.perf_counter()
    tokenizer = Tokenizer()
    allele_sequences = _prepare_probe_allele_sequences(
        probe_alleles=probe_alleles,
        prepared_mhc_sequences=mhc_sequences,
        index_csv=index_csv,
    )
    fit_probe_peptides = _select_fit_supported_probe_peptides(real_records, probe_alleles)
    probe_peptides = list(explicit_probe_peptides)
    for peptide in fit_probe_peptides:
        if peptide not in probe_peptides:
            probe_peptides.append(peptide)

    probe_support = dict(prepared_state.probe_support)
    for peptide in probe_peptides:
        if peptide in probe_support:
            continue
        probe_support[peptide] = _audit_probe_support(
            merged_tsv,
            peptide,
            alleles_of_interest=probe_alleles,
        )
    setup_stage_timings["probe_setup_s"] = time.perf_counter() - probe_stage_start
    batch_contract = _planned_strict_batch_contract(
        alleles=probe_alleles,
        batch_size=int(args.batch_size),
        synthetic_fraction=float(batch_synthetic_fraction),
    )

    print(
        json.dumps(
            {
                "event": "focused_binding_setup",
                "probe_alleles": probe_alleles,
                "design_id": str(args.design_id),
                "train_all_alleles": bool(args.train_all_alleles),
                "train_mhc_class_filter": train_class_filter,
                "rows": len(real_records),
                "train_rows": len(real_train_records),
                "val_rows": len(real_val_records),
                "device": device,
                "measurement_profile": args.measurement_profile,
                "measurement_type_filter": str(args.measurement_type_filter),
                "qualifier_filter": str(args.qualifier_filter),
                "shared_peptides_only": bool(args.shared_peptides_only),
                "affinity_loss_mode": str(args.affinity_loss_mode),
                "affinity_assay_mode": str(args.affinity_assay_mode),
                "affinity_assay_residual_mode": str(args.affinity_assay_residual_mode),
                "kd_grouping_mode": str(args.kd_grouping_mode),
                "affinity_target_encoding": str(args.affinity_target_encoding),
                "max_affinity_nM": float(args.max_affinity_nm),
                "binding_kinetic_input_mode": str(args.binding_kinetic_input_mode),
                "binding_direct_segment_mode": str(args.binding_direct_segment_mode),
                "init_checkpoint": str(args.init_checkpoint or ""),
                "synthetic_negatives": bool(args.synthetic_negatives),
                "synthetic_refresh_each_epoch": bool(args.synthetic_refresh_each_epoch),
                "synthetic_modes": list(synthetic_modes or ALL_SYNTHETIC_MODES),
                "batch_synthetic_fraction": float(batch_synthetic_fraction),
                "planned_batch_contract": batch_contract,
                "runtime_config": {
                    "num_workers": int(args.num_workers),
                    "pin_memory": bool(args.pin_memory),
                    "persistent_workers": bool(args.persistent_workers),
                    "prefetch_factor": int(args.prefetch_factor),
                    "matmul_precision": str(args.matmul_precision),
                    "allow_tf32": bool(args.allow_tf32),
                    "torch_compile": bool(args.torch_compile),
                    "probe_plot_frequency": str(args.probe_plot_frequency),
                },
                "peptide_pos_mode": str(args.peptide_pos_mode),
                "groove_pos_mode": str(args.groove_pos_mode),
                "binding_core_lengths": list(binding_core_lengths),
                "binding_core_refinement": str(args.binding_core_refinement),
                "probe_allele_counts_after_filter": probe_allele_counts_after_filter,
                "probe_peptides": probe_peptides,
                "dataset_cache": {
                    "enabled": bool(args.dataset_cache),
                    "cache_dir": str(cache_dir),
                    "cache_hit": bool(prepared_state.cache_hit),
                    "cache_key": str(prepared_state.cache_key),
                },
                "setup_stage_timings": setup_stage_timings,
                "param_count": param_count,
                "setup_wall_s": time.perf_counter() - wall_start,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    epoch_summaries: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    model.train()
    setup_wall_s = time.perf_counter() - wall_start
    diverged = False
    divergence_epoch: Optional[int] = None
    divergence_reason = ""
    try:
        for epoch in range(1, int(args.epochs) + 1):
            epoch_wall_start = time.perf_counter()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            epoch_synthetic_seed = _epoch_synthetic_seed(
                int(args.seed),
                epoch,
                refresh_each_epoch=bool(args.synthetic_refresh_each_epoch),
            )
            (
                epoch_train_state
            ) = _build_epoch_train_state(
                real_train_records=real_train_records,
                real_train_dataset=real_train_dataset,
                mhc_sequences=mhc_sequences,
                synthetic_negatives=bool(args.synthetic_negatives),
                negative_ratio=float(args.negative_ratio),
                synthetic_seed=epoch_synthetic_seed,
                class_i_anchor_strategy=str(args.class_i_anchor_strategy),
                synthetic_modes=synthetic_modes,
                batch_size=int(args.batch_size),
                balanced=bool(args.balanced_batches),
                seed=int(args.seed) + epoch,
                alleles=probe_alleles,
                force_global_balance=bool(args.train_all_alleles),
                batch_synthetic_fraction=float(batch_synthetic_fraction),
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=int(args.prefetch_factor),
            )
            train_records = epoch_train_state.train_records
            train_dataset = epoch_train_state.train_dataset
            train_loader = epoch_train_state.train_loader
            if scheduler is None:
                scheduler = _build_lr_scheduler(
                    optimizer,
                    schedule=str(args.lr_schedule),
                    base_lr=float(args.lr),
                    steps_per_epoch=max(1, len(train_loader)),
                    epochs=int(args.epochs),
                    warmup_fraction=float(args.warmup_fraction),
                    min_lr_scale=float(args.min_lr_scale),
                    onecycle_pct_start=float(args.onecycle_pct_start),
                )
            epoch_synthetic_stats = epoch_train_state.synthetic_stats
            record_groove_audit = epoch_train_state.record_groove_audit
            dataset_groove_audit = epoch_train_state.dataset_groove_audit
            train_loss_sum = 0.0
            train_batches = 0
            grad_metrics_sum: Counter[str] = Counter()
            batch_metrics: Dict[str, float] = {}
            train_data_wait_s = 0.0
            train_forward_loss_s = 0.0
            train_backward_s = 0.0
            train_optimizer_s = 0.0
            skipped_train_batches = 0
            last_batch_done = time.perf_counter()
            for batch in train_loader:
                batch_start = time.perf_counter()
                train_data_wait_s += batch_start - last_batch_done
                forward_start = time.perf_counter()
                try:
                    total_loss, batch_metrics = _affinity_only_loss(
                        model,
                        batch,
                        device,
                        regularization=regularization_cfg,
                        loss_mode=str(args.affinity_loss_mode),
                        affinity_target_encoding=str(args.affinity_target_encoding),
                        max_affinity_nM=float(args.max_affinity_nm),
                        kd_grouping_mode=str(args.kd_grouping_mode),
                    )
                except RuntimeError as exc:
                    if "no supervised losses" not in str(exc).lower():
                        raise
                    skipped_train_batches += 1
                    last_batch_done = time.perf_counter()
                    continue
                if not torch.isfinite(total_loss):
                    diverged = True
                    divergence_epoch = epoch
                    divergence_reason = "non_finite_train_loss"
                    break
                train_forward_loss_s += time.perf_counter() - forward_start
                optimizer.zero_grad(set_to_none=True)
                backward_start = time.perf_counter()
                total_loss.backward()
                train_backward_s += time.perf_counter() - backward_start
                if not _gradients_are_finite(model):
                    diverged = True
                    divergence_epoch = epoch
                    divergence_reason = "non_finite_gradients"
                    break
                grad_metrics_sum["grad_norm_affinity_probe"] += _grad_norm_for_prefixes(
                    model,
                    ["affinity_predictor.binding_affinity_probe."],
                )
                grad_metrics_sum["grad_norm_binding_core"] += _grad_norm_for_prefixes(
                    model,
                    ["affinity_predictor.binding."],
                )
                grad_metrics_sum["grad_norm_trunk_other"] += _grad_norm_for_prefixes(
                    model,
                    [
                        "aa_embedding.",
                        "segment_embedding.",
                        "stream_encoder.",
                        "latent_layers.pmhc_interaction.",
                        "pmhc_interaction_token_proj.",
                        "pmhc_interaction_vec_norm.",
                        "binding_affinity_readout_proj.",
                        "core_window_fuse.",
                        "core_window_score.",
                        "core_window_prior.",
                        "core_window_vec_norm.",
                    ],
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_start = time.perf_counter()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                train_optimizer_s += time.perf_counter() - optimizer_start
                train_loss_sum += float(total_loss.detach().item())
                train_batches += 1
                last_batch_done = time.perf_counter()
            if diverged:
                epoch_wall_end = time.perf_counter()
                gpu_metrics: Dict[str, float] = gpu_sampler.summarize_window(epoch_wall_start, epoch_wall_end)
                if device == "cuda":
                    peak_allocated = float(torch.cuda.max_memory_allocated())
                    peak_reserved = float(torch.cuda.max_memory_reserved())
                    gpu_metrics.update(
                        {
                            "gpu_peak_allocated_bytes": peak_allocated,
                            "gpu_peak_reserved_bytes": peak_reserved,
                            "gpu_peak_allocated_gib": peak_allocated / float(1024**3),
                            "gpu_peak_reserved_gib": peak_reserved / float(1024**3),
                        }
                    )
                epoch_summary = {
                    "epoch": epoch,
                    "train_loss": float("nan"),
                    "val_loss": float("nan"),
                    "epoch_wall_s": epoch_wall_end - epoch_wall_start,
                    "train_data_wait_s": train_data_wait_s,
                    "train_forward_loss_s": train_forward_loss_s,
                    "train_backward_s": train_backward_s,
                    "train_optimizer_s": train_optimizer_s,
                    "val_wall_s": 0.0,
                    "probe_eval_wall_s": 0.0,
                    "current_lr": float(optimizer.param_groups[0]["lr"]),
                    "diverged": True,
                    "divergence_reason": divergence_reason,
                    "synthetic_seed": int(epoch_synthetic_seed),
                    "synthetic_stats": epoch_synthetic_stats,
                    "train_record_summary": _summarize_binding_records(train_records),
                    "record_groove_audit": record_groove_audit,
                    "dataset_groove_audit": dataset_groove_audit,
                    "observed_train_examples": int(train_batches * int(args.batch_size)),
                    "skipped_train_batches": int(skipped_train_batches),
                    "summary_write_wall_s": 0.0,
                    **gpu_metrics,
                }
                epoch_summaries.append(epoch_summary)
                break

            train_loss = train_loss_sum / max(train_batches, 1)
            val_start = time.perf_counter()
            val_loss = _mean_affinity_loss(
                model,
                val_loader,
                device,
                regularization=regularization_cfg,
                loss_mode=str(args.affinity_loss_mode),
                affinity_target_encoding=str(args.affinity_target_encoding),
                max_affinity_nM=float(args.max_affinity_nm),
                kd_grouping_mode=str(args.kd_grouping_mode),
            )
            if not math.isfinite(val_loss):
                diverged = True
                divergence_epoch = epoch
                divergence_reason = "non_finite_val_loss"
            val_wall_s = time.perf_counter() - val_start
            probe_start = time.perf_counter()
            probe_eval = _evaluate_probe_panel(
                model,
                tokenizer,
                allele_sequences,
                probe_peptides,
                probe_alleles,
                device,
            )
            probe_eval_wall_s = time.perf_counter() - probe_start
            for row in probe_eval:
                probe_rows.append({"epoch": epoch, **row})
            epoch_wall_end = time.perf_counter()
            gpu_metrics: Dict[str, float] = gpu_sampler.summarize_window(epoch_wall_start, epoch_wall_end)
            if device == "cuda":
                peak_allocated = float(torch.cuda.max_memory_allocated())
                peak_reserved = float(torch.cuda.max_memory_reserved())
                gpu_metrics.update(
                    {
                        "gpu_peak_allocated_bytes": peak_allocated,
                        "gpu_peak_reserved_bytes": peak_reserved,
                        "gpu_peak_allocated_gib": peak_allocated / float(1024**3),
                        "gpu_peak_reserved_gib": peak_reserved / float(1024**3),
                    }
                )
            epoch_summary = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_wall_s": epoch_wall_end - epoch_wall_start,
                "train_data_wait_s": train_data_wait_s,
                "train_forward_loss_s": train_forward_loss_s,
                "train_backward_s": train_backward_s,
                "train_optimizer_s": train_optimizer_s,
                "gpu_busy_wall_s": (
                    train_forward_loss_s + train_backward_s + train_optimizer_s + val_wall_s + probe_eval_wall_s
                ),
                "val_wall_s": val_wall_s,
                "probe_eval_wall_s": probe_eval_wall_s,
                "current_lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm_affinity_probe": grad_metrics_sum["grad_norm_affinity_probe"] / max(train_batches, 1),
                "grad_norm_binding_core": grad_metrics_sum["grad_norm_binding_core"] / max(train_batches, 1),
                "grad_norm_trunk_other": grad_metrics_sum["grad_norm_trunk_other"] / max(train_batches, 1),
                "diverged": bool(diverged),
                "divergence_reason": divergence_reason if diverged else "",
                "synthetic_seed": int(epoch_synthetic_seed),
                "synthetic_stats": epoch_synthetic_stats,
                "train_record_summary": _summarize_binding_records(train_records),
                "record_groove_audit": record_groove_audit,
                "dataset_groove_audit": dataset_groove_audit,
                "observed_train_examples": int(train_batches * int(args.batch_size)),
                "skipped_train_batches": int(skipped_train_batches),
                "summary_write_wall_s": 0.0,
                **gpu_metrics,
            }
            epoch_summaries.append(epoch_summary)
            if diverged:
                break
            summary = {
                "config": {
                    "design_id": str(args.design_id),
                    "probe_alleles": probe_alleles,
                    "train_all_alleles": bool(args.train_all_alleles),
                    "train_mhc_class_filter": train_class_filter,
                    "probe_peptides": probe_peptides,
                    "peptide_pos_mode": str(args.peptide_pos_mode),
                    "groove_pos_mode": str(args.groove_pos_mode),
                    "binding_core_lengths": list(binding_core_lengths),
                    "binding_core_refinement": str(args.binding_core_refinement),
                    "measurement_profile": args.measurement_profile,
                    "measurement_type_filter": str(args.measurement_type_filter),
                    "qualifier_filter": str(args.qualifier_filter),
                    "shared_peptides_only": bool(args.shared_peptides_only),
                    "affinity_loss_mode": str(args.affinity_loss_mode),
                    "affinity_assay_mode": str(args.affinity_assay_mode),
                    "affinity_assay_residual_mode": str(args.affinity_assay_residual_mode),
                    "kd_grouping_mode": str(args.kd_grouping_mode),
                    "affinity_target_encoding": str(args.affinity_target_encoding),
                    "max_affinity_nM": float(args.max_affinity_nm),
                    "binding_kinetic_input_mode": str(args.binding_kinetic_input_mode),
                    "binding_direct_segment_mode": str(args.binding_direct_segment_mode),
                    "init_checkpoint": str(args.init_checkpoint or ""),
                    "synthetic_negatives": bool(args.synthetic_negatives),
                    "negative_ratio": float(args.negative_ratio),
                    "batch_synthetic_fraction": float(batch_synthetic_fraction),
                    "synthetic_refresh_each_epoch": bool(args.synthetic_refresh_each_epoch),
                    "synthetic_modes": list(synthetic_modes or ALL_SYNTHETIC_MODES),
                    "planned_batch_contract": batch_contract,
                    "class_i_anchor_strategy": str(args.class_i_anchor_strategy),
                    "binding_contrastive_weight": float(args.binding_contrastive_weight),
                    "binding_contrastive_margin": float(args.binding_contrastive_margin),
                    "binding_contrastive_target_gap_min": float(args.binding_contrastive_target_gap_min),
                    "binding_contrastive_target_gap_cap": float(args.binding_contrastive_target_gap_cap),
                    "binding_contrastive_max_pairs": int(args.binding_contrastive_max_pairs),
                    "binding_peptide_contrastive_weight": float(args.binding_peptide_contrastive_weight),
                    "binding_peptide_contrastive_margin": float(args.binding_peptide_contrastive_margin),
                    "binding_peptide_contrastive_target_gap_min": float(
                        args.binding_peptide_contrastive_target_gap_min
                    ),
                    "binding_peptide_contrastive_target_gap_cap": float(
                        args.binding_peptide_contrastive_target_gap_cap
                    ),
                    "binding_peptide_contrastive_max_pairs": int(
                        args.binding_peptide_contrastive_max_pairs
                    ),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr_schedule": str(args.lr_schedule),
                    "warmup_fraction": float(args.warmup_fraction),
                    "min_lr_scale": float(args.min_lr_scale),
                    "onecycle_pct_start": float(args.onecycle_pct_start),
                    "seed": int(args.seed),
                    "runtime_config": {
                        "num_workers": int(args.num_workers),
                        "pin_memory": bool(args.pin_memory),
                        "persistent_workers": bool(args.persistent_workers),
                        "prefetch_factor": int(args.prefetch_factor),
                        "matmul_precision": str(args.matmul_precision),
                        "allow_tf32": bool(args.allow_tf32),
                        "torch_compile": bool(args.torch_compile),
                        "probe_plot_frequency": str(args.probe_plot_frequency),
                    },
                },
                "subset_stats": subset_stats,
                "balance_stats": balance_stats,
                "shared_peptide_stats": shared_peptide_stats,
                "split_stats": split_stats,
                "mhc_resolve_stats": mhc_stats,
                "probe_allele_counts_after_filter": probe_allele_counts_after_filter,
                "real_record_summary": _summarize_binding_records(real_records),
                "real_train_record_summary": _summarize_binding_records(real_train_records),
                "train_record_summary": _summarize_binding_records(train_records),
                "val_record_summary": _summarize_binding_records(real_val_records),
                "record_summary": _summarize_binding_records(train_records + real_val_records),
                "synthetic_stats": epoch_synthetic_stats,
                "record_groove_audit": record_groove_audit,
                "dataset_groove_audit": dataset_groove_audit,
                "dataset_size": len(train_dataset) + len(val_dataset),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "param_count": param_count,
                "setup_wall_s": setup_wall_s,
                "setup_stage_timings": setup_stage_timings,
                "diverged": bool(diverged),
                "divergence_epoch": divergence_epoch,
                "divergence_reason": divergence_reason,
                "probe_support": probe_support,
                "warm_start": {
                    "used": bool(str(args.init_checkpoint or "").strip()),
                    "checkpoint": str(args.init_checkpoint or ""),
                    "filter_stats": warm_start_filter_stats,
                    "checkpoint_epoch": (
                        int(warm_start_payload.get("epoch"))
                        if isinstance(warm_start_payload, dict)
                        and warm_start_payload.get("epoch") is not None
                        else None
                    ),
                },
                "epochs": epoch_summaries,
            }
            write_start = time.perf_counter()
            _write_summary_artifacts(
                out_dir=out_dir,
                summary=summary,
                probe_rows=probe_rows,
                write_probe_plot=(
                    str(args.probe_plot_frequency) == "epoch"
                    or (
                        str(args.probe_plot_frequency) == "final"
                        and epoch == int(args.epochs)
                    )
                ),
            )
            epoch_summary["summary_write_wall_s"] = time.perf_counter() - write_start
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "event": "focused_binding_epoch",
                        "design_id": str(args.design_id),
                        **epoch_summary,
                        "train_batch_metrics": batch_metrics,
                        "probe_rows": probe_eval,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if diverged:
                break
    finally:
        gpu_sampler.stop()


if __name__ == "__main__":
    main()
