"""Independent source-join and applicability replay; no Presto imports."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datacache import install_bundle


def pin(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size_bytes": p.stat().st_size}


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def verify(receipt):
    paths = {n: Path(p) for n, p in receipt["paths"].items()}
    for name, path in paths.items():
        assert pin(path) == receipt["files"][name], name
    policy = json.loads(paths["policy.json"].read_text())
    assert policy["policy"] == "explicit_preparation_and_elution_applicability_v1"
    assert (
        policy["inspection_only"]
        and not policy["new_predictive_inputs"]
        and not policy["historical_fields_changed"]
    )
    sources = install_bundle(
        Path.home() / ".hitlist/presto-assay-applicability/inputs",
        policy["source_inputs"],
        download_options={"show_progress": False},
    )
    for name in ["prior-observations.tsv", "prior-ic50.tsv"]:
        assert pin(sources[name]) == pin(paths[name])
    prior = read(paths["prior-observations.tsv"])
    new = read(paths["observations.tsv"])
    old_ic50 = read(paths["prior-ic50.tsv"])
    new_ic50 = read(paths["ic50.tsv"])
    assert len(prior) == len(new) == 9053 and len(old_ic50) == len(new_ic50) == 2614
    native = {row["evidence_row_id"]: row for row in read(paths["native-evidence.tsv"])}
    assert len(native) == 9053
    joined = set()
    for batch in pq.ParquetFile(sources["training.parquet"]).iter_batches(
        batch_size=5000, columns=["evidence_row_id", "assay_method", "evidence_kind"]
    ):
        for row in batch.to_pylist():
            identity = row["evidence_row_id"]
            if identity in native:
                assert native[identity] == {
                    k: "" if v is None else str(v) for k, v in row.items()
                }
                joined.add(identity)
    assert joined == native.keys()
    fields = {
        "source_assay_method",
        "source_evidence_kind",
        "parent_assay_contexts",
        "processing_context_applicability",
        "processing_context_applicability_basis",
        "processing_context_availability",
        "assay_context_origin",
    }
    added = {}
    coverage = Counter()
    for before, after in zip(prior, new):
        assert set(after) - set(before) == fields
        assert all(after[k] == v for k, v in before.items())
        source = native[before["evidence_row_id"]]
        method = source["assay_method"].lower()
        kind = source["evidence_kind"]
        assert before["assay_method"] == source["assay_method"]
        if ("purified mhc" in method and "cellular mhc" in method) or (
            method.startswith("purified mhc/") and kind == "ms"
        ):
            state, basis = "unknown", "conflicting_assay_preparation"
        elif method.startswith("purified mhc/"):
            state, basis = "not_applicable", "explicit_purified_mhc_preparation"
        elif method.startswith("cellular mhc/"):
            state, basis = "applicable", "explicit_cellular_mhc_preparation"
        elif kind == "ms":
            state, basis = "applicable", "native_elution_evidence"
        else:
            state, basis = "unknown", "unspecified_assay_preparation"
        availability = (
            "not_applicable"
            if state == "not_applicable"
            else "applicability_unknown"
            if state == "unknown"
            else "resolved"
            if before["processing_species_status"] == "resolved"
            else "unresolved"
        )
        expected = {
            "source_assay_method": source["assay_method"],
            "source_evidence_kind": kind,
            "parent_assay_contexts": "",
            "processing_context_applicability": state,
            "processing_context_applicability_basis": basis,
            "processing_context_availability": availability,
            "assay_context_origin": "source_observation",
        }
        assert {k: after[k] for k in fields} == expected
        added[before["evidence_row_id"]] = expected
        coverage[
            (
                before["source_partition"],
                kind,
                source["assay_method"],
                state,
                availability,
            )
        ] += 1
    ic50_counts = Counter()
    for before, after in zip(old_ic50, new_ic50):
        assert after == {**before, **added[before["evidence_row_id"]]}
        ic50_counts[after["processing_context_applicability"]] += 1
    assert dict(ic50_counts) == {"not_applicable": 2533, "applicable": 81}
    columns = [
        "source_partition",
        "source_evidence_kind",
        "source_assay_method",
        "processing_context_applicability",
        "processing_context_availability",
    ]
    observed = {
        tuple(r[c] for c in columns): int(r["n_observations"])
        for r in read(paths["coverage.tsv"])
    }
    assert observed == coverage
    contract = json.loads(paths["output-contract.json"].read_text())
    assert (
        contract["schema_version"] == 5
        and contract["assay_context_policy"] == policy["policy"]
    )
    return {
        "verified": True,
        "n_observations_with_all_original_fields_unchanged": 9053,
        "n_ic50_with_raw_and_effective_targets_unchanged": 2614,
        "n_native_source_joins": len(joined),
        "n_ic50_by_applicability": dict(ic50_counts),
        "n_artifacts_verified": len(paths),
        "method": "independent original/native source joins, all-cell comparisons and preparation/coverage classification; no Presto imports",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.receipt.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
