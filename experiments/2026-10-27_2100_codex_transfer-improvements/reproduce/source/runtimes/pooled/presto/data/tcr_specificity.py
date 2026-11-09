"""Strict normalized receptor-specific evidence, independent of VDJdb score.

This is a Presto interchange schema, NOT a parser for an uninspected TCRvdb
release. Source conversion must preserve the author outcome/QC policy.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TCRSpecificityRecord:
    record_id: str
    study_id: str
    assay_id: str
    peptide: str
    mhc_a: str
    mhc_b: str
    mhc_class: str
    tcr_alpha: str
    tcr_beta: str
    sequence_provenance: str
    sequence_reference: str
    outcome: str
    quality: str

    @property
    def target(self) -> float:
        """Non-validation is a negative for this functional assay only."""
        if self.quality != "pass" or self.outcome not in {"validated", "not_validated"}:
            raise ValueError("Only measured, QC-passing functional outcomes have binary targets")
        return float(self.outcome == "validated")


@dataclass(frozen=True)
class TCRSpecificityData:
    records: tuple[TCRSpecificityRecord, ...]
    source_sha256: str
    exclusions: dict[str, int]


def load_tcr_specificity_csv(path: str | Path, *, max_chain_length: int = 512):
    """Read explicit paired sequences and measured, QC-passing outcomes.

    ``not_validated`` means assayed but not validating under the source policy.
    Failed QC, indeterminate and unassayed pairs do not become negative labels.
    Unknown column names/values and duplicate source IDs fail closed.
    """
    from ..models.tcr_specificity import validate_chain

    path = Path(path)
    records, excluded, seen = [], Counter(), set()
    required = set(TCRSpecificityRecord.__dataclass_fields__)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != required or len(reader.fieldnames) != len(required):
            raise ValueError(f"Expected normalized Presto columns: {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {line}")
            row = {key: value.strip() for key, value in row.items()}
            if not row["record_id"] or row["record_id"] in seen:
                raise ValueError(f"Missing/duplicate record_id at row {line}")
            seen.add(row["record_id"])
            if row["outcome"] not in {"validated", "not_validated", "indeterminate", "unassayed"}:
                raise ValueError(f"Unknown functional outcome at row {line}")
            if row["quality"] not in {"pass", "fail", "unknown"}:
                raise ValueError(f"Unknown quality status at row {line}")
            if row["quality"] != "pass":
                excluded["quality_" + row["quality"]] += 1
                continue
            if row["outcome"] in {"indeterminate", "unassayed"}:
                excluded[row["outcome"]] += 1
                continue
            for field in ("study_id", "assay_id", "mhc_a", "sequence_reference"):
                if not row[field]:
                    raise ValueError(f"Missing {field} at row {line}")
            if row["mhc_class"] not in {"I", "II"}:
                raise ValueError(f"Unknown MHC class at row {line}")
            if row["mhc_class"] == "II" and not row["mhc_b"]:
                raise ValueError(f"Class II requires both MHC chains at row {line}")
            if row["sequence_provenance"] not in {"observed_full_length", "reconstructed"}:
                raise ValueError(f"Declare observed_full_length or reconstructed at row {line}")
            for field in ("tcr_alpha", "tcr_beta"):
                row[field] = validate_chain(row[field], max_length=max_chain_length)
            row["peptide"] = validate_chain(row["peptide"], max_length=50)
            records.append(TCRSpecificityRecord(**row))
    return TCRSpecificityData(
        tuple(records), hashlib.sha256(path.read_bytes()).hexdigest(), dict(excluded)
    )
