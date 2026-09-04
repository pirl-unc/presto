"""Machine-readable train/validation/test supervision support audits."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch

from presto.data.collate import PrestoCollator


DEFAULT_BINARY_TARGETS = frozenset(
    {
        "elution",
        "excision",
        "foreignness",
        "ms_detectability",
        "processing",
        "presentation",
        "tcell",
        "tcr_evidence",
        "tcr_evidence_method",
    }
)


def _masked_values(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flatten target values selected by a broadcast-compatible mask."""
    target = target.detach().cpu()
    mask = mask.detach().cpu()
    if target.shape != mask.shape:
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(-1)
        mask = torch.broadcast_to(mask, target.shape)
    return target.reshape(-1)[mask.reshape(-1) > 0]


def audit_split_support(
    splits: Mapping[str, Iterable[Any]],
    *,
    collator: Optional[PrestoCollator] = None,
    binary_targets: Sequence[str] = tuple(DEFAULT_BINARY_TARGETS),
    chunk_size: int = 512,
) -> Dict[str, Any]:
    """Count effective target support after dataset construction and splitting.

    Counting collated masks rather than record classes makes this use the same
    routing contract as the loss. That catches, for example, a present
    ``StabilityRecord`` whose only label was lost before ``target_masks``.
    """
    collate = collator or PrestoCollator()
    binary = frozenset(binary_targets)
    split_payload: Dict[str, Any] = {}
    all_sample_ids: set[str] = set()
    duplicate_sample_ids: set[str] = set()
    lineage_issue_count = 0
    lineage_issue_examples: list[str] = []
    fake_null_sequence_count = 0
    fake_null_sequence_examples: list[str] = []
    all_sample_contract_digests: list[str] = []
    all_supervision_contract_digests: list[str] = []

    for split_name, dataset in splits.items():
        target_counts: Dict[str, Dict[str, Any]] = {}
        full_contract = hashlib.sha256()
        supervision_contract = hashlib.sha256()
        row_count = len(dataset) if hasattr(dataset, "__len__") else 0
        for start in range(0, row_count, max(1, int(chunk_size))):
            samples = [dataset[index] for index in range(start, min(row_count, start + chunk_size))]
            if not samples:
                continue
            for sample in samples:
                sample_id = str(getattr(sample, "sample_id", "") or "")
                if sample_id in all_sample_ids:
                    duplicate_sample_ids.add(sample_id)
                all_sample_ids.add(sample_id)

                evidence_row_id = str(getattr(sample, "evidence_row_id", "") or "").strip()
                mapped = int(getattr(sample, "source_mapping_n_candidates", 0) or 0) > 0
                lineage_failures: list[str] = []
                if mapped and not evidence_row_id:
                    lineage_failures.append("missing_evidence_row_id")
                if evidence_row_id:
                    has_observation_source = bool(
                        str(getattr(sample, "assay_iri", "") or "")
                        or str(getattr(sample, "reference_iri", "") or "")
                    )
                    has_selected_mapping = bool(
                        str(getattr(sample, "mapping_protein_id", "") or "")
                        and getattr(sample, "mapping_position", None) is not None
                        and str(getattr(sample, "mapping_proteome", "") or "")
                    )
                    if not has_observation_source:
                        lineage_failures.append("missing_observation_source")
                    if mapped and not has_selected_mapping:
                        lineage_failures.append("incomplete_selected_mapping")
                if lineage_failures:
                    lineage_issue_count += 1
                    if len(lineage_issue_examples) < 10:
                        identity = sample_id or evidence_row_id or "<missing-sample-id>"
                        lineage_issue_examples.append(f"{identity} ({','.join(lineage_failures)})")

                for field in ("flank_n", "flank_c"):
                    value = str(getattr(sample, field, "") or "").strip().upper()
                    no_selected_mapping = getattr(sample, "mapping_position", None) is None
                    if value == "NAN" and no_selected_mapping:
                        fake_null_sequence_count += 1
                        if len(fake_null_sequence_examples) < 10:
                            fake_null_sequence_examples.append(f"{sample_id}:{field}")

                sample_payload = asdict(sample) if is_dataclass(sample) else vars(sample)
                rendered = json.dumps(
                    sample_payload, sort_keys=True, separators=(",", ":"), default=str
                ).encode()
                full_contract.update(rendered)
                all_sample_contract_digests.append(hashlib.sha256(rendered).hexdigest())
                invariant_payload = dict(sample_payload)
                for field in (
                    "flank_n",
                    "flank_c",
                    "flank_n_is_terminus",
                    "flank_c_is_terminus",
                ):
                    invariant_payload.pop(field, None)
                invariant_rendered = json.dumps(
                    invariant_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode()
                supervision_contract.update(invariant_rendered)
                all_supervision_contract_digests.append(
                    hashlib.sha256(invariant_rendered).hexdigest()
                )
            batch = collate(samples)
            for target_name, mask in batch.target_masks.items():
                target = batch.targets.get(target_name)
                if target is None:
                    continue
                values = _masked_values(target, mask)
                entry = target_counts.setdefault(
                    target_name,
                    {
                        "count": 0,
                        "exact": 0,
                        "censored_lower": 0,
                        "censored_upper": 0,
                    },
                )
                entry["count"] += int(values.numel())

                qualifier = batch.target_quals.get(target_name)
                if qualifier is not None:
                    qualifiers = _masked_values(qualifier, mask).to(torch.int64)
                    entry["exact"] += int((qualifiers == 0).sum().item())
                    entry["censored_lower"] += int((qualifiers < 0).sum().item())
                    entry["censored_upper"] += int((qualifiers > 0).sum().item())

                if target_name in binary:
                    entry.setdefault("negative", 0)
                    entry.setdefault("positive", 0)
                    entry["negative"] += int((values <= 0.5).sum().item())
                    entry["positive"] += int((values > 0.5).sum().item())

        split_payload[split_name] = {
            "rows": row_count,
            "targets": dict(sorted(target_counts.items())),
            "sample_contract_sha256": full_contract.hexdigest(),
            "supervision_contract_sha256": supervision_contract.hexdigest(),
        }

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "binary_targets": sorted(binary),
        "splits": split_payload,
        "lineage": {
            "issue_count": lineage_issue_count,
            "issue_examples": lineage_issue_examples,
            "duplicate_sample_id_count": len(duplicate_sample_ids),
            "duplicate_sample_id_examples": sorted(duplicate_sample_ids)[:10],
        },
        "fake_null_sequences": {
            "count": fake_null_sequence_count,
            "examples": fake_null_sequence_examples,
        },
        "dataset_contract_sha256": hashlib.sha256(
            "".join(sorted(all_sample_contract_digests)).encode()
        ).hexdigest(),
        "dataset_supervision_contract_sha256": hashlib.sha256(
            "".join(sorted(all_supervision_contract_digests)).encode()
        ).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_split_support(
    audit: Mapping[str, Any],
    *,
    required_targets: Sequence[str] = (),
    require_all_active: bool = False,
    binary_balance_targets: Sequence[str] = (),
    require_all_active_binary_balance: bool = False,
    min_count: int = 1,
    require_traceable_lineage: bool = False,
    forbid_fake_null_sequences: bool = False,
) -> None:
    """Fail with all missing-support and one-class violations at once."""
    splits = audit.get("splits", {})
    active = {
        target
        for split in splits.values()
        for target, counts in split.get("targets", {}).items()
        if int(counts.get("count", 0)) > 0
    }
    required = set(required_targets)
    if require_all_active:
        required.update(active)

    binary_required = set(binary_balance_targets)
    if require_all_active_binary_balance:
        binary_required.update(active & set(audit.get("binary_targets", ())))

    failures = []
    if require_traceable_lineage:
        lineage = audit.get("lineage", {})
        if int(lineage.get("issue_count", 0)):
            failures.append(
                "source lineage is incomplete for "
                f"{int(lineage.get('issue_count', 0))} sample(s): "
                f"{lineage.get('issue_examples', [])}"
            )
        if int(lineage.get("duplicate_sample_id_count", 0)):
            failures.append(
                "stable sample IDs are duplicated for "
                f"{int(lineage.get('duplicate_sample_id_count', 0))} ID(s): "
                f"{lineage.get('duplicate_sample_id_examples', [])}"
            )
    if forbid_fake_null_sequences:
        fake_nulls = audit.get("fake_null_sequences", {})
        if int(fake_nulls.get("count", 0)):
            failures.append(
                f"found {int(fake_nulls.get('count', 0))} optional sequence(s) equal to NAN: "
                f"{fake_nulls.get('examples', [])}"
            )
    for split_name, split in splits.items():
        targets = split.get("targets", {})
        for target in sorted(required):
            count = int(targets.get(target, {}).get("count", 0))
            if count < int(min_count):
                failures.append(
                    f"{split_name}:{target} has {count} examples; requires >= {int(min_count)}"
                )
        for target in sorted(binary_required):
            counts = targets.get(target, {})
            positive = int(counts.get("positive", 0))
            negative = int(counts.get("negative", 0))
            if positive == 0 or negative == 0:
                failures.append(
                    f"{split_name}:{target} is one-class (positive={positive}, negative={negative})"
                )

    if failures:
        raise RuntimeError("Split-support preflight failed:\n- " + "\n- ".join(failures))


def write_split_support_artifacts(out_dir: Path | str, audit: Mapping[str, Any]) -> Dict[str, Path]:
    """Persist the support audit as canonical JSON and a reviewable flat CSV."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "split_support.json"
    csv_path = output / "split_support.csv"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = (
        "split",
        "rows",
        "target",
        "count",
        "exact",
        "censored_lower",
        "censored_upper",
        "negative",
        "positive",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_name, split in audit.get("splits", {}).items():
            for target_name, counts in split.get("targets", {}).items():
                writer.writerow(
                    {
                        "split": split_name,
                        "rows": split.get("rows", 0),
                        "target": target_name,
                        **{field: counts.get(field, "") for field in fields[3:]},
                    }
                )
    return {"json": json_path, "csv": csv_path}


def write_data_funnel_artifacts(out_dir: Path | str, funnel: Mapping[str, Any]) -> Dict[str, Path]:
    """Persist pre/post-curation counts and explicit drop reasons."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(funnel)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()

    json_path = output / "data_funnel.json"
    csv_path = output / "data_funnel.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("kind", "stage", "name", "count"))
        writer.writeheader()
        for kind in ("stages", "drop_reasons", "additions", "diagnostics"):
            for stage, counts in payload.get(kind, {}).items():
                if not isinstance(counts, Mapping):
                    continue
                for name, count in counts.items():
                    if isinstance(count, (int, float)) and not isinstance(count, bool):
                        writer.writerow(
                            {
                                "kind": kind,
                                "stage": stage,
                                "name": name,
                                "count": count,
                            }
                        )
    return {"json": json_path, "csv": csv_path}
