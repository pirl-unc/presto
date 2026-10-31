"""Independent selected-peptide audit against Weirdo's pinned protein reference."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import time
import zlib

import numpy as np

from .modular import AA
from .modular_sources import file_sha256


def audit_weirdo_reference(
    observations,
    reference_path,
    *,
    reference_sha256,
    categories,
    salt,
    train_percent=70,
    validation_percent=15,
    buckets=20,
    chunk_proteins=2048,
    max_seconds=900,
    progress=None,
    reconstruct_missing_groups=False,
):
    """Audit all selected labels/occurrences; return discrepancies without relabeling.

    Prefix groups reproduce the source construction, not validated homology
    clusters. Positions are zero-based. Canonical packed windows cannot match
    noncanonical residues or separators between proteins.
    """
    if not categories or len(set(categories)) != len(categories) or len(categories) > 63:
        raise ValueError("unique declared reference categories required (at most 63)")
    if not salt or not 0 < train_percent < train_percent + validation_percent < 100:
        raise ValueError("explicit reference grouping/fold policy required")
    if chunk_proteins <= 0 or buckets <= 0 or max_seconds <= 0:
        raise ValueError("positive chunk, bucket and audit-time limits required")
    if type(reconstruct_missing_groups) is not bool:
        raise ValueError("reconstruct_missing_groups must be explicit boolean")
    path = Path(reference_path)
    before = path.stat()
    if file_sha256(path) != reference_sha256:
        raise ValueError("reference protein hash mismatch")
    observations = list(observations)
    if not observations:
        raise ValueError("selected observations required")
    lengths = {len(row.peptide) for row in observations}
    if len(lengths) != 1 or not 1 <= next(iter(lengths)) <= 12:
        raise ValueError("one canonical k-mer length in [1, 12] required")
    k = next(iter(lengths))
    if len({row.peptide for row in observations}) != len(observations):
        raise ValueError("duplicate selected reference peptide")
    if len({row.observation_id for row in observations}) != len(observations):
        raise ValueError("duplicate selected observation ID")
    codes = {aa: index for index, aa in enumerate(AA)}

    def packed(sequence):
        result = 0
        for aa in sequence:
            result = (result << 5) | codes[aa]
        return result

    for row in observations:
        row.validate()
        if row.endpoint != "identity" or row.split not in ("train", "validation", "test"):
            raise ValueError(
                "reference audit requires labeled auxiliary train/validation/test rows"
            )
        if row.provenance.get("family_split_salt") != salt:
            raise ValueError("selected reference split salt mismatch")
        if row.provenance.get("source_reference_sha256") != reference_sha256:
            raise ValueError("selected reference protein pin mismatch")
        declared_split = {"train": "train", "val": "validation", "test": "test"}.get(
            row.provenance.get("source_split")
        )
        if declared_split != row.split:
            raise ValueError("selected archive split disagrees with observation split")
        if set(row.identity_labels) != {f"peptide_source:{category}" for category in categories}:
            raise ValueError("complete declared membership labels required")
        if set(row.identity_labels.values()) - {"present", "absent"}:
            raise ValueError("unsupported reference membership label")
    order = sorted(range(len(observations)), key=lambda i: packed(observations[i].peptide))
    targets = np.asarray([packed(observations[i].peptide) for i in order], dtype=np.uint64)
    matches = [[] for _ in observations]
    lookup = np.full(256, 31, dtype=np.uint8)
    for aa, code in codes.items():
        lookup[ord(aa)] = code
    category_bits = {category: 1 << i for i, category in enumerate(categories)}
    start_time, protein_count = time.monotonic(), 0

    def scan(batch):
        nonlocal protein_count
        if time.monotonic() - start_time > max_seconds:
            raise TimeoutError("reference audit exceeded its declared wall-time ceiling")
        starts = np.concatenate(([0], np.cumsum([len(row["sequence"]) + 1 for row in batch])[:-1]))
        values = lookup[
            np.frombuffer(
                "\0".join(row["sequence"] for row in batch).encode("ascii", errors="replace"),
                dtype=np.uint8,
            )
        ]
        n = len(values) - k + 1
        if n > 0:
            windows = np.zeros(n, dtype=np.uint64)
            for offset in range(k):
                windows = (windows << np.uint64(5)) | values[offset : offset + n].astype(np.uint64)
            indices = np.searchsorted(targets, windows).clip(max=len(targets) - 1)
            positions = np.flatnonzero(targets[indices] == windows)
            proteins = np.searchsorted(starts, positions, side="right") - 1
            grouped = {}
            for position, protein in zip(positions.tolist(), proteins.tolist(), strict=True):
                selected = order[int(indices[position])]
                grouped.setdefault((selected, protein), []).append(position - int(starts[protein]))
            for (selected, protein), offsets in grouped.items():
                row = batch[protein]
                family = row["identifier"].split("_", 1)[0]
                if not family or row["category"] not in category_bits:
                    raise ValueError("invalid source identifier/category")
                bucket = zlib.crc32(f"{salt}:{family}".encode()) % 100
                fold = (
                    "train"
                    if bucket < train_percent
                    else "validation"
                    if bucket < train_percent + validation_percent
                    else "test"
                )
                matches[selected].append(
                    {
                        "reference_row": protein_count + protein,
                        "identifier": row["identifier"],
                        "category": row["category"],
                        "positions": offsets,
                        "family_key": family,
                        "fold": fold,
                        "bucket": zlib.crc32(f"{salt}:bucket:{family}".encode()) % buckets,
                    }
                )
        protein_count += len(batch)
        if progress:
            progress(protein_count)

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"identifier", "sequence", "category"} <= set(reader.fieldnames or ()):
            raise ValueError("reference protein columns missing")
        batch = []
        for row in reader:
            if row["category"] not in category_bits or not row["identifier"] or not row["sequence"]:
                raise ValueError("invalid reference protein row")
            batch.append(row)
            if len(batch) == chunk_proteins:
                scan(batch)
                batch = []
        if batch:
            scan(batch)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("reference protein file changed during audit")

    # Connect selected peptides sharing any source mnemonic-prefix group.
    parents = list(range(len(observations)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    first_family = {}
    for index, proteins in enumerate(matches):
        for protein in proteins:
            family = protein["family_key"]
            if family in first_family:
                parents[root(index)] = root(first_family[family])
            else:
                first_family[family] = index
    members = {}
    for index, row in enumerate(observations):
        members.setdefault(root(index), []).append(row.observation_id)
    group_ids = {
        key: hashlib.sha256("\n".join(sorted(value)).encode()).hexdigest()
        for key, value in members.items()
    }
    records, discrepancies = [], []
    for index, (row, proteins) in enumerate(zip(observations, matches, strict=True)):
        expected = sum(
            category_bits[category]
            for category in categories
            if row.identity_labels[f"peptide_source:{category}"] == "present"
        )
        actual = 0
        for protein in proteins:
            actual |= category_bits[protein["category"]]
        folds = sorted({protein["fold"] for protein in proteins})
        bucket = min((protein["bucket"] for protein in proteins), default=None)
        reasons = []
        if not proteins:
            reasons.append("not_in_reference")
        if expected != actual:
            reasons.append("membership_mismatch")
        if folds != [row.split]:
            reasons.append("source_fold_mismatch_or_overlap")
        supplied_bucket = row.provenance.get("source_family_bucket")
        if supplied_bucket is None and not reconstruct_missing_groups:
            reasons.append("missing_source_bucket")
        elif supplied_bucket is not None and bucket != supplied_bucket:
            reasons.append("source_bucket_mismatch")
        record = {
            "observation_id": row.observation_id,
            "peptide": row.peptide,
            "split": row.split,
            "expected_mask": expected,
            "reference_mask": actual,
            "folds": folds,
            "source_bucket": supplied_bucket,
            "reference_bucket": bucket,
            "bucket_origin": "reference_reconstruction"
            if supplied_bucket is None
            else "supplied_archive",
            "component": group_ids[root(index)],
            "proteins": proteins,
            "discrepancies": reasons,
        }
        records.append(record)
        if reasons:
            discrepancies.append({"observation_id": row.observation_id, "reasons": reasons})
    if time.monotonic() - start_time > max_seconds:
        raise TimeoutError("reference audit exceeded its declared wall-time ceiling")
    return {
        "status": "failed" if discrepancies else "passed",
        "reference_sha256": reference_sha256,
        "reference_path": str(path.resolve()),
        "reference_proteins": protein_count,
        "categories": list(categories),
        "k": k,
        "salt": salt,
        "fold_policy": [
            train_percent,
            validation_percent,
            100 - train_percent - validation_percent,
        ],
        "bucket_count": buckets,
        "selected": len(observations),
        "components": len(members),
        "supplied_bucket_comparisons": sum(
            row.provenance.get("source_family_bucket") is not None for row in observations
        ),
        "missing_buckets_reconstructed": sum(
            record["source_bucket"] is None and record["reference_bucket"] is not None
            for record in records
        )
        if reconstruct_missing_groups
        else 0,
        "elapsed_seconds": time.monotonic() - start_time,
        "records": records,
        "discrepancies": discrepancies,
        "independence": "source mnemonic-prefix groups; homology independence unverified",
    }
