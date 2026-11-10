"""Independently replay membership using a reverse candidate-core lookup."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    for path, digest in summary["inputs"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    records = pq.read_table(args.membership).to_pylist()
    peptides = {r["peptide"] for r in records}
    reverse = defaultdict(set)
    for peptide in peptides:
        for offset in range(len(peptide) - 7):
            reverse[peptide[offset : offset + 8]].add(peptide)
    hits, exact, counts = defaultdict(set), defaultdict(set), defaultdict(int)
    for source in summary["inputs"]:
        if source.endswith("shared-export.parquet"):
            continue
        atlas = source.endswith("dispositions.parquet")
        field = "fold" if atlas else "partition"
        prefix = "atlas:" if atlas else "presentation:"
        for batch in pq.ParquetFile(source).iter_batches(columns=[field, "peptide"]):
            for row in batch.to_pylist():
                group, peptide = prefix + row[field], row["peptide"]
                counts[group] += 1
                if peptide in peptides:
                    exact[group].add(peptide)
                for offset in range(len(peptide) - 7):
                    hits[group].update(reverse.get(peptide[offset : offset + 8], ()))
    protected = ["presentation:consumed_development", "atlas:validation", "atlas:test"]
    embargo = set().union(*(hits[g] | exact[g] for g in protected))
    for row in records:
        for group in counts:
            assert row[group + ":exact"] == (row["peptide"] in exact[group])
            assert row[group + ":core8"] == (row["peptide"] in hits[group])
        assert row["protected_presentation_or_atlas_overlap"] == (
            row["peptide"] in embargo
        )
        assert row["training_eligible"] is False
    for group, count in counts.items():
        expected = summary["cohorts"][group]
        assert count == expected["n_index_rows"]
        assert len(hits[group]) == expected["n_core8_overlap_candidate_peptides"]
        assert len(exact[group]) == expected["n_exact_overlap_candidate_peptides"]
    result = {
        "status": "passed",
        "n_candidate_rows_verified": len(records),
        "n_membership_fields_per_row": 12,
        "n_protected_overlap_rows": sum(r["peptide"] in embargo for r in records),
        "n_training_eligible_rows": 0,
        "outcomes_or_predictions_accessed": False,
        "input_membership_sha256": hashlib.sha256(
            args.membership.read_bytes()
        ).hexdigest(),
    }
    assert (
        result["n_protected_overlap_rows"]
        == summary["n_candidate_rows_overlapping_protected_identities"]
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
