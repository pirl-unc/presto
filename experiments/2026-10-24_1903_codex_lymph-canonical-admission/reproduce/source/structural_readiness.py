"""Inspect canonical reference support and feature multiplicity without fitting."""

import gzip
import hashlib
import json
import resource
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from hitlist.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parents[1] / "data/lymph_source_qualification/2026-10-24_1903_canonical"


def main():
    started = time.perf_counter()
    output = ROOT / "results/structural-readiness.json"
    assert not output.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/structural_readiness.py")
    store = EvidenceStore.open(PACKAGE)
    tables = {}
    for name in ("observations", "references", "quantities"):
        physical = pq.read_table(PACKAGE / "nanaware_lymph/typed" / f"{name}.parquet")
        queried = store.query(
            "lymph_" + name,
            select=["native." + c for c in physical.column_names],
            limit=None,
        ).table
        queried = queried.rename_columns(physical.column_names)
        assert queried.equals(physical)
        tables[name] = queried.to_pylist()
    references = defaultdict(lambda: defaultdict(list))
    for row in tables["references"]:
        if row["reference_sequence"]:
            references[row["accession_raw"]][row["reference_sequence"]].append(
                row["reference_record_id"]
            )
    vectors = defaultdict(list)
    for row in tables["quantities"]:
        vectors[row["observation_id"]].append(
            (
                row["sample_id"],
                row["quantity_kind"],
                row["value_status"],
                row["quantitative_value"],
            )
        )
    feature_keys = defaultdict(list)
    counts = defaultdict(Counter)
    per_observation = []
    supported_parents = defaultdict(set)
    for row in tables["observations"]:
        block = row["block_id"]
        count = counts[block]
        count["n_source_rows"] += 1
        count["sequence_parse:" + row["sequence_parse_status"]] += 1
        count["length_status:" + row["length_status"]] += 1
        peptide = row["peptide_sequence"]
        parents = references.get(row["parent_label_raw"], {})
        mappings = []
        for sequence, reference_ids in parents.items():
            if peptide is None:
                continue
            offset = sequence.find(peptide)
            while offset >= 0:
                digest = hashlib.sha256(sequence.encode()).hexdigest()
                mappings.append(
                    {
                        "parent_sha256": digest,
                        "start_1based": offset + 1,
                        "end_1based": offset + len(peptide),
                        "reference_record_ids": reference_ids,
                    }
                )
                supported_parents[block].add(digest)
                offset = sequence.find(peptide, offset + 1)
        status = (
            "unparsed_peptide"
            if peptide is None
            else "no_exact_accession_label_match"
            if not parents
            else "no_exact_sequence_occurrence"
            if not mappings
            else "one_candidate_occurrence"
            if len(mappings) == 1
            else "multiple_candidate_occurrences"
        )
        count["mapping:" + status] += 1
        key = tuple(
            row[field]
            for field in (
                "block_id",
                "peptide_raw",
                "parent_label_raw",
                "charge_raw",
                "start_raw",
                "end_raw",
            )
        )
        vector = tuple(sorted(vectors[row["observation_id"]]))
        feature_keys[key].append((row["observation_id"], vector))
        per_observation.append(
            {
                "observation_id": row["observation_id"],
                "source_record_id": row["source_record_id"],
                "block": block,
                "peptide_sequence": peptide,
                "mapping_status": status,
                "candidate_occurrences": mappings,
            }
        )
    repeats = []
    for key, rows in feature_keys.items():
        if len(rows) > 1:
            repeats.append(
                {
                    "block": key[0],
                    "feature_key": key,
                    "n_rows": len(rows),
                    "n_distinct_quantity_vectors": len({vector for _, vector in rows}),
                    "observation_ids": [oid for oid, _ in rows],
                }
            )
            counts[key[0]]["n_repeated_feature_keys"] += 1
            counts[key[0]]["n_rows_in_repeated_feature_keys"] += len(rows)
    for block, parents in supported_parents.items():
        counts[block]["n_distinct_candidate_parent_sequences"] = len(parents)
    with gzip.GzipFile(
        filename=str(ROOT / "results/structural-observations.json.gz"),
        mode="wb",
        mtime=0,
    ) as handle:
        handle.write(
            json.dumps(
                {"observations": per_observation, "repeated_keys": repeats},
                separators=(",", ":"),
            ).encode()
        )
    result = {
        "status": "completed_source_structure_only",
        "blocks": {block: dict(count) for block, count in counts.items()},
        "exact_accession_matching_only": True,
        "candidate_mappings_are_not_measured_origins": True,
        "aggregation_or_training_admission": False,
        "model_training_or_predictions": False,
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    assert result["seconds"] <= 600 and result["peak_rss_gib"] <= 3
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
