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
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from presto.data import PrestoCollator, PrestoDataset, create_dataloader
from presto.data.cross_source_dedup import UnifiedRecord, classify_assay_type
from presto.data.groove import prepare_mhc_input
from presto.data.mhc_index import build_mhc_sequence_lookup, load_mhc_index
from presto.data.tokenizer import Tokenizer
from presto.models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    affinity_nm_to_log10,
    binding_prob_from_kd_log10,
    normalize_binding_target_log10,
)
from presto.models.presto import Presto
from presto.scripts.train_iedb import (
    augment_binding_records_with_synthetic_negatives,
    load_binding_records_for_alleles_from_merged_tsv,
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


def _split_csv(text: str) -> List[str]:
    return [token.strip() for token in str(text or "").split(",") if token.strip()]


def _measurement_type_key(text: Optional[str]) -> str:
    return str(text or "").strip().lower()


def _keep_measurement_type(measurement_type: Optional[str], profile: str) -> bool:
    key = _measurement_type_key(measurement_type)
    if profile == MEASUREMENT_PROFILE_ALL:
        return True
    if profile == MEASUREMENT_PROFILE_NUMERIC:
        return key not in NON_QUANTITATIVE_MEASUREMENT_TYPES
    if profile == MEASUREMENT_PROFILE_DIRECT:
        return key in DIRECT_MEASUREMENT_TYPES
    raise ValueError(f"Unsupported measurement profile: {profile!r}")


def _summarize_binding_records(records: Sequence[Any]) -> Dict[str, Any]:
    by_allele_total: Counter[str] = Counter()
    by_allele_le_500: Counter[str] = Counter()
    measurement_type_counter: Counter[str] = Counter()
    qualifier_counter: Counter[int] = Counter()
    for rec in records:
        allele = str(rec.mhc_allele or "").strip()
        by_allele_total[allele] += 1
        if float(rec.value) <= 500.0 and int(rec.qualifier) in (-1, 0):
            by_allele_le_500[allele] += 1
        measurement_type_counter[str(rec.measurement_type or "")] += 1
        qualifier_counter[int(rec.qualifier)] += 1
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
                kd_log10 = float(outputs["assays"]["KD_nM"][0].item())
                presentation_logit = outputs.get("presentation_logit")
                processing_logit = outputs.get("processing_logit")
                rows.append(
                    {
                        "peptide": pep,
                        "allele": str(allele),
                        "kd_log10": kd_log10,
                        "kd_nM": float(10.0 ** kd_log10),
                        "binding_prob": float(
                            binding_prob_from_kd_log10(
                                kd_log10,
                                midpoint_nM=midpoint,
                                log10_scale=scale,
                            )
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


def _mask_from_batch_mapping(
    mapping: Optional[Mapping[str, torch.Tensor]],
    key: str,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mapping is None or key not in mapping:
        return None
    return mapping[key].to(device=device)


def _affinity_only_loss(
    model: Presto,
    batch: Any,
    device: str,
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
        pred_vec = _as_float_vector(pred)
        target_vec = normalize_binding_target_log10(
            _as_float_vector(target),
            max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            assume_log10=False,
        )
        mask_vec = _as_float_vector(mask).to(device=pred_vec.device)
        qual_vec = _as_float_vector(qual).to(device=pred_vec.device, dtype=torch.long)
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

    if not losses:
        raise RuntimeError("Affinity-only batch produced no supervised losses")
    total = torch.stack(losses).mean()
    metrics["loss_tasks"] = float(len(losses))
    return total, metrics


def _mean_affinity_loss(
    model: Presto,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            loss, _ = _affinity_only_loss(model, batch, device)
            total += float(loss.detach().item())
            batches += 1
    model.train()
    return total / max(batches, 1)


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
        grouped[(str(row["peptide"]), str(row["allele"]))].append(
            (int(row["epoch"]), float(row["kd_nM"]))
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
    ax.set_ylabel("Predicted KD (nM)")
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
) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_probe_csv(out_dir / "probe_affinity_over_epochs.csv", probe_rows)
    (out_dir / "probe_affinity_over_epochs.json").write_text(
        json.dumps(list(probe_rows), indent=2),
        encoding="utf-8",
    )
    _write_probe_plot(out_dir / "probe_affinity_over_epochs.png", probe_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused allele-panel binding diagnostic")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="artifacts/focused_binding_probe")
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probe-peptide", type=str, default=DEFAULT_PROBE_PEPTIDE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", type=str, default="iedb")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--measurement-profile",
        type=str,
        choices=sorted(MEASUREMENT_PROFILES),
        default=MEASUREMENT_PROFILE_NUMERIC,
    )
    parser.add_argument("--synthetic-negatives", dest="synthetic_negatives", action="store_true")
    parser.add_argument("--no-synthetic-negatives", dest="synthetic_negatives", action="store_false")
    parser.set_defaults(synthetic_negatives=False)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--balanced-batches", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_tsv = data_dir / "merged_deduped.tsv"
    index_csv = data_dir / "mhc_index.csv"
    if not merged_tsv.exists():
        raise FileNotFoundError(f"Merged TSV not found: {merged_tsv}")
    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")

    alleles = _split_csv(args.alleles)
    if not alleles:
        raise ValueError("At least one allele is required")

    records, subset_stats = load_binding_records_for_alleles_from_merged_tsv(
        merged_tsv,
        alleles=alleles,
        max_records=(None if int(args.max_records) <= 0 else int(args.max_records)),
        cap_sampling="reservoir",
        sampling_seed=int(args.seed) + 17,
    )

    source_filter = str(args.source or "").strip().lower()
    if source_filter:
        records = [rec for rec in records if str(rec.source or "").strip().lower() == source_filter]
    records = [
        rec for rec in records if _keep_measurement_type(rec.measurement_type, args.measurement_profile)
    ]

    mhc_sequences, mhc_stats = resolve_mhc_sequences_from_index(
        index_csv=str(index_csv),
        alleles=sorted({str(rec.mhc_allele or "").strip() for rec in records if str(rec.mhc_allele or "").strip()}),
    )

    synthetic_stats: Dict[str, Any] = {}
    if args.synthetic_negatives:
        records, synthetic_stats = augment_binding_records_with_synthetic_negatives(
            binding_records=records,
            mhc_sequences=mhc_sequences,
            negative_ratio=float(args.negative_ratio),
            weak_value_min_nM=DEFAULT_MAX_AFFINITY_NM,
            weak_value_max_nM=DEFAULT_MAX_AFFINITY_NM,
            seed=int(args.seed),
            class_i_no_mhc_beta_ratio=0.0,
        )

    if not records:
        raise RuntimeError("No binding records remain after filtering")

    dataset = PrestoDataset(
        binding_records=records,
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=False,
    )
    train_size = max(1, int(0.8 * len(dataset)))
    val_size = max(1, len(dataset) - train_size)
    if train_size + val_size > len(dataset):
        train_size = len(dataset) - 1
        val_size = 1
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(args.seed)),
    )

    collator = PrestoCollator()
    train_loader = create_dataloader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collator=collator,
        balanced=bool(args.balanced_batches),
        seed=int(args.seed),
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collator=collator,
        balanced=False,
        seed=int(args.seed),
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
    allele_sequences = _resolve_allele_sequences(index_csv)
    fit_probe_peptides = _select_fit_supported_probe_peptides(records, alleles)
    probe_peptides = [str(args.probe_peptide).strip().upper()]
    for peptide in fit_probe_peptides:
        if peptide not in probe_peptides:
            probe_peptides.append(peptide)

    probe_support = {
        peptide: _audit_probe_support(merged_tsv, peptide, alleles_of_interest=alleles)
        for peptide in probe_peptides
    }

    print(
        json.dumps(
            {
                "event": "focused_binding_setup",
                "alleles": alleles,
                "rows": len(records),
                "device": device,
                "measurement_profile": args.measurement_profile,
                "synthetic_negatives": bool(args.synthetic_negatives),
                "probe_peptides": probe_peptides,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    epoch_summaries: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    model.train()
    for epoch in range(1, int(args.epochs) + 1):
        train_loss_sum = 0.0
        train_batches = 0
        for batch in train_loader:
            total_loss, batch_metrics = _affinity_only_loss(model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(total_loss.detach().item())
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = _mean_affinity_loss(model, val_loader, device)
        probe_eval = _evaluate_probe_panel(
            model,
            tokenizer,
            allele_sequences,
            probe_peptides,
            alleles,
            device,
        )
        for row in probe_eval:
            probe_rows.append({"epoch": epoch, **row})
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        epoch_summaries.append(epoch_summary)
        summary = {
            "config": {
                "alleles": alleles,
                "probe_peptides": probe_peptides,
                "measurement_profile": args.measurement_profile,
                "synthetic_negatives": bool(args.synthetic_negatives),
                "negative_ratio": float(args.negative_ratio),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
            },
            "subset_stats": subset_stats,
            "mhc_resolve_stats": mhc_stats,
            "record_summary": _summarize_binding_records(records),
            "synthetic_stats": synthetic_stats,
            "dataset_size": len(dataset),
            "train_size": train_size,
            "val_size": val_size,
            "probe_support": probe_support,
            "epochs": epoch_summaries,
        }
        _write_summary_artifacts(
            out_dir=out_dir,
            summary=summary,
            probe_rows=probe_rows,
        )
        print(
            json.dumps(
                {
                    "event": "focused_binding_epoch",
                    **epoch_summary,
                    "train_batch_metrics": batch_metrics,
                    "probe_rows": probe_eval,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
