"""Source-faithful projection of public NetMHC BA training rows.

This module deliberately does not download archives or assign evaluation roles.
It converts already pinned BA rows into TRAIN observations while preserving an
unspecified assay method and applying caller-supplied identity embargoes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Callable, Iterable, Mapping


BA_TARGET_MAX_NM = 50_000.0
AA = "ACDEFGHIKLMNPQRSTVWY"
PEPTIDE = re.compile(f"[{AA}]+")


@dataclass(frozen=True)
class ArchiveBARow:
    archive: str
    member: str
    fold: int
    line_number: int
    peptide: str
    target_token: str
    target: float
    identity: str
    context: str
    row_sha256: str

    @property
    def source_key(self) -> tuple[str, str, str, str]:
        return self.archive, self.peptide, self.identity, self.context

    @property
    def exact_key(self) -> tuple[str, str, str, str, float]:
        return (*self.source_key, self.target)


@dataclass(frozen=True)
class ArchiveBAIssue:
    archive: str
    member: str
    fold: int
    line_number: int
    row_sha256: str
    reason: str


@dataclass(frozen=True)
class ProjectionResult:
    observations: tuple[dict, ...]
    dispositions: tuple[dict, ...]
    conflicts: tuple[dict, ...]
    duplicates: tuple[dict, ...]


def peptide_cores(peptide: str, length: int = 9) -> tuple[str, ...]:
    if length <= 0:
        raise ValueError("core length must be positive")
    return tuple(peptide[index : index + length] for index in range(len(peptide) - length + 1))


def parse_ba_line(
    raw: bytes,
    *,
    archive: str,
    member: str,
    fold: int,
    line_number: int,
) -> ArchiveBARow | ArchiveBAIssue:
    stripped = raw.decode("utf-8", errors="strict").strip()
    digest = hashlib.sha256(stripped.encode()).hexdigest()
    fields = stripped.split()
    if len(fields) != 4:
        return ArchiveBAIssue(archive, member, fold, line_number, digest, "malformed_column_count")
    peptide, target_token, identity, context = fields
    peptide = peptide.upper()
    if PEPTIDE.fullmatch(peptide) is None:
        return ArchiveBAIssue(archive, member, fold, line_number, digest, "noncanonical_peptide")
    try:
        target = float(target_token)
    except ValueError:
        target = math.nan
    if not math.isfinite(target) or not 0 <= target <= 1:
        return ArchiveBAIssue(
            archive, member, fold, line_number, digest, "invalid_transformed_target"
        )
    return ArchiveBARow(
        archive,
        member,
        fold,
        line_number,
        peptide,
        target_token,
        target,
        identity,
        context,
        digest,
    )


def invert_ba_target(target: float) -> tuple[float, str]:
    if not math.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("NetMHC BA target must be finite and in [0, 1]")
    if target == 0:
        return BA_TARGET_MAX_NM, ">="
    if target == 1:
        return 1.0, "<="
    return BA_TARGET_MAX_NM ** (1 - target), "="


def current_identity_sets(rows: Iterable[Mapping]) -> tuple[set[str], set[str], set[tuple]]:
    """Read only peptide/MHC identity from a frozen observation stream."""
    peptides: set[str] = set()
    cores: set[str] = set()
    pmhc: set[tuple[str, str, str]] = set()
    for row in rows:
        peptide = str(row["peptide"])
        peptides.add(peptide)
        cores.update(peptide_cores(peptide))
        for mhc in row["mhc"]:
            pmhc.add((peptide, str(mhc["groove1"]), str(mhc["groove2"])))
    return peptides, cores, pmhc


def _lineage(row: ArchiveBARow) -> dict:
    return {
        "fold": row.fold,
        "line_number": row.line_number,
        "member": row.member,
        "row_sha256": row.row_sha256,
    }


def _source_row_id(row: ArchiveBARow | ArchiveBAIssue) -> str:
    value = "\x1f".join(
        (
            row.archive,
            row.member,
            str(row.line_number),
            row.row_sha256,
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _observation_id(row: ArchiveBARow) -> str:
    value = "\x1f".join((*row.source_key, row.target.hex()))
    return f"{row.archive}:ba:{hashlib.sha256(value.encode()).hexdigest()}"


def project_ba_rows(
    rows: Iterable[ArchiveBARow | ArchiveBAIssue],
    *,
    resolve_identity: Callable[[str], Mapping | None],
    archive_hashes: Mapping[str, str],
    source_names: Mapping[str, str],
    development_peptides: set[str],
    development_cores: set[str],
    train_pmhc: set[tuple[str, str, str]],
    validate_observation: Callable[[Mapping], object] | None = None,
) -> ProjectionResult:
    """Project BA rows using a fixed, label-independent disposition order."""
    ordered = sorted(
        rows,
        key=lambda row: (row.archive, row.member, row.line_number, row.row_sha256),
    )
    valid = [row for row in ordered if isinstance(row, ArchiveBARow)]
    targets: dict[tuple[str, str, str, str], set[float]] = defaultdict(set)
    exact_rows: dict[tuple[str, str, str, str, float], list[ArchiveBARow]] = defaultdict(list)
    for row in valid:
        targets[row.source_key].add(row.target)
        exact_rows[row.exact_key].append(row)
    conflict_keys = {key for key, values in targets.items() if len(values) > 1}
    canonical = {key: values[0] for key, values in exact_rows.items()}
    resolutions: dict[str, Mapping | None] = {}
    observations: list[dict] = []
    dispositions: list[dict] = []

    def disposition(row, reason: str) -> None:
        dispositions.append(
            {
                "archive": row.archive,
                "fold": row.fold,
                "line_number": row.line_number,
                "member": row.member,
                "row_sha256": row.row_sha256,
                "source_row_id": _source_row_id(row),
                "disposition": reason,
            }
        )

    for row in ordered:
        if isinstance(row, ArchiveBAIssue):
            disposition(row, row.reason)
            continue
        if row.identity not in resolutions:
            resolutions[row.identity] = resolve_identity(row.identity)
        resolution = resolutions[row.identity]
        if resolution is None:
            disposition(row, "nonhuman_or_unresolved_mhc")
            continue
        if row.source_key in conflict_keys:
            disposition(row, "conflicting_source_targets")
            continue
        if canonical[row.exact_key] is not row:
            disposition(row, "duplicate_exact_source_row")
            continue
        if row.peptide in development_peptides:
            disposition(row, "development_exact_peptide_embargo")
            continue
        if any(core in development_cores for core in peptide_cores(row.peptide)):
            disposition(row, "development_core9_embargo")
            continue
        pmhc = (row.peptide, str(resolution["groove1"]), str(resolution["groove2"]))
        if pmhc in train_pmhc:
            disposition(row, "current_train_exact_pmhc_dedup")
            continue
        value, qualifier = invert_ba_target(row.target)
        lineage = [_lineage(value) for value in exact_rows[row.exact_key]]
        mhc_class = str(resolution["mhc_class"])
        catalog_alleles = list(resolution["catalog_alleles"])
        observation = {
            "schema": "presto.modular.observation.v1",
            "observation_id": _observation_id(row),
            "source": source_names[row.archive],
            "endpoint": "ic50",
            "peptide": row.peptide,
            "mhc": [
                {
                    "groove1": resolution["groove1"],
                    "groove2": resolution["groove2"],
                    "mhc_class": mhc_class,
                    "species1": "HomoSapiens",
                    "species2": "HomoSapiens",
                    "provenance": {
                        "archive_identity": row.identity,
                        "catalog_alleles": catalog_alleles,
                    },
                }
            ],
            "value": value,
            "unit": "nM",
            "qualifier": qualifier,
            "split": "train",
            "group_id": f"{row.archive}:ba:study-unknown",
            "assay_method": {
                "preparation": "unknown",
                "mechanism": "unknown",
                "detector": "unknown",
                "review_status": "unknown",
            },
            "provenance": {
                "archive": row.archive,
                "archive_sha256": archive_hashes[row.archive],
                "archive_modality": "ba",
                "archive_identity": row.identity,
                "archive_context": row.context,
                "archive_transformed_target": row.target,
                "archive_lineage": lineage,
                "assay_output": "unspecified_assay/IC50",
                "study_id": None,
                "study_availability": "not_available_in_archive",
                "redistribution_status": "unresolved",
                "target_mapping": "IC50_nM=50000**(1-y)",
            },
        }
        if validate_observation is not None:
            validate_observation(observation)
        observations.append(observation)
        disposition(row, "admitted_train")

    conflicts = tuple(
        {
            "archive": key[0],
            "context": key[3],
            "identity": key[2],
            "peptide": key[1],
            "row_count": sum(len(exact_rows[(*key, target)]) for target in targets[key]),
            "targets": sorted(targets[key]),
        }
        for key in sorted(conflict_keys)
    )
    duplicates = tuple(
        {
            "archive": key[0],
            "context": key[3],
            "identity": key[2],
            "lineage": [_lineage(row) for row in values],
            "peptide": key[1],
            "target": key[4],
        }
        for key, values in sorted(exact_rows.items())
        if len(values) > 1
    )
    observations.sort(key=lambda row: row["observation_id"])
    dispositions.sort(key=lambda row: row["source_row_id"])
    return ProjectionResult(tuple(observations), tuple(dispositions), conflicts, duplicates)
