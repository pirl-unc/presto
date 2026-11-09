"""Verify recovered chemistry and observation identities independently of admission."""

from collections import Counter
from pathlib import Path
import csv
import gzip
import hashlib
import json


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, receipt):
    root = Path(root)
    summary = receipt["summary"]
    source = root / "recovered/strazar_2023_recovered_unmodified.csv"
    evidence_path = root / "recovered/source-evidence.jsonl"
    assert sha(source) == summary["source_sha256"]
    assert sha(evidence_path) == summary["source_evidence_sha256"]
    for name, meta in summary["files"].items():
        assert sha(root / "recovered-admission" / name) == meta["sha256"]
    with source.open() as stream:
        original = list(csv.DictReader(stream))
    assert len(original) == 14187
    assert all(row["has_ptm"] == "False" for row in original)
    source_pairs = {(r["peptide"], r["mhc_restriction"]): r for r in original}
    assert len(source_pairs) == len(original)
    with evidence_path.open() as stream:
        evidence = {row["evidence_id"]: row for line in stream for row in [json.loads(line)]}
    length_counts = Counter(len(r["peptide"]) for r in original if len(r["peptide"]) < 9)
    assert length_counts == {7: 132, 8: 157}
    seen = set()
    with gzip.open(root / "recovered-admission/train.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["split"] == "train" and row["group_id"] == "pmid:37301199"
            assert row["endpoint"] == "elution_observation" and row["value"] == 1
            assert len(row["mhc"]) == 1 and row["mhc"][0]["mhc_class"] == "II"
            assert row["observation_id"] not in seen
            seen.add(row["observation_id"])
            raw = row["provenance"]["raw_record"]
            assert raw["has_ptm"] is False and raw["peptide_modifications"] == ""
            assert int(raw["pmid"]) == 37301199 and len(row["peptide"]) >= 9
            key = (row["peptide"], row["mhc"][0]["provenance"]["reported_restriction"])
            linked = source_pairs[key]
            review = row["provenance"]["peptidoform_review"]
            assert review["evidence_id"] == linked["source_evidence_id"]
            assert review["evidence_ledger_sha256"] == summary["source_evidence_sha256"]
            original_evidence = evidence[review["evidence_id"]]
            assert original_evidence["state"] == "unmodified"
            assert (original_evidence["peptide"], original_evidence["mhc_restriction"]) == key
            assert original_evidence["support"]
            for support in original_evidence["support"]:
                assert support["nterm"] == "Hydrogen" and support["cterm"] == "Free Acid"
                assert not any(support[k] for k in ("modifications", "variableSites", "cys"))
                assert support["source_row"] >= 2 and len(support["source_sha256"]) == 64
    assert len(seen) == summary["train_observations"] == 13550
    return dict(
        status="passed",
        reviewed_source_observations=len(original),
        train_observations=len(seen),
        short_observations=sum(length_counts.values()),
        short_length_counts=dict(length_counts),
        source_sha256=sha(source),
        evidence_sha256=sha(evidence_path),
    )
