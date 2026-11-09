"""Inventory canonical polarization composition support without predictive scoring."""

import gzip
import hashlib
import json
import math
import resource
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from hitlist.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCE = Path("/private/tmp/hitlist-v2-reference-200-final")
TYPED = SOURCE / "external/pxd057588/typed"
PREVIOUS = REPO / "experiments/2026-10-24_0449_codex_apc-precursor-response"
SALT = "pxd057588_polarization_composition_v1:"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode(), mtime=0))
    else:
        path.write_text(text)


def split_for(component):
    bucket = int(hashlib.sha256((SALT + component).encode()).hexdigest()[:16], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def main():
    started = time.perf_counter()
    assert not (ROOT / "results/inventory.json").exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/inventory.py")
    spec = json.loads((ROOT / "reproduce/launch.json").read_text())
    assert digest(SOURCE / "manifest.json") == spec["source_manifest_sha256"]
    store = EvidenceStore.open(SOURCE)
    logical = store.query(
        "external_reference_source_mapping", limit=None
    ).table.to_pylist()
    maps = pq.read_table(TYPED / "reference_source_mappings.parquet").to_pylist()
    assert len(logical) == len(maps)
    by_id = {r["peptide_row_id"]: r for r in maps}
    for row in logical:
        source = by_id[row["peptide_row_id"]]
        for key in (
            "component",
            "sequence_id",
            "table_number",
            "reference_full_sequence_count",
        ):
            assert row[key] == source[key]
        assert row["peptide_sequence"] == source["sequence_raw"]
    names = [
        "reference_source_mappings",
        "reference_occurrences",
        "reference_parents",
        "channels",
        "quantity_results",
        "quantity_measurements",
        "source_assertions",
        "peptide_rows",
        "source_cells",
    ]
    schemas = {
        name: {
            "n_rows": pq.ParquetFile(TYPED / f"{name}.parquet").metadata.num_rows,
            "fields": pq.read_schema(TYPED / f"{name}.parquet").names,
            "sha256": digest(TYPED / f"{name}.parquet"),
        }
        for name in names
    }
    write(ROOT / "results/schemas.json", schemas)
    occurrences = pq.read_table(TYPED / "reference_occurrences.parquet").to_pylist()
    parents = {
        r["accession"]: r
        for r in pq.read_table(TYPED / "reference_parents.parquet").to_pylist()
    }
    seq_occurrences, parent_components = defaultdict(list), defaultdict(set)
    for row in occurrences:
        parent = parents[row["accession"]]
        assert parent["sequence_sha256"] == row["parent_sequence_sha256"]
        assert (
            hashlib.sha256(parent["sequence"].encode()).hexdigest()
            == parent["sequence_sha256"]
        )
        assert (
            parent["sequence"][row["start1"] - 1 : row["end1"]]
            == row["peptide_sequence"]
        )
        seq_occurrences[row["sequence_id"]].append(row)
        parent_components[row["parent_sequence_sha256"]].add(row["component"])
    prior = json.loads((PREVIOUS / "results/cohort.json").read_text())
    prior_root = Path(prior["model_input_directory"])
    previous_inputs = {}
    for name in ("parents.json.gz", "dispositions.json.gz"):
        assert digest(prior_root / name) == prior["files"][name]
        previous_inputs[name] = json.loads(
            gzip.decompress((prior_root / name).read_bytes())
        )
    previous_ids = {
        r["peptide_row_id"]
        for r in previous_inputs["dispositions.json.gz"]
        if r["included"]
    }
    assert len(previous_ids) == prior["n_admitted_peptide_rows"]
    previous_parent_hashes = set(previous_inputs["parents.json.gz"])
    for sha, sequence in previous_inputs["parents.json.gz"].items():
        assert hashlib.sha256(sequence.encode()).hexdigest() == sha
    blocked = {by_id[key]["component"] for key in previous_ids}
    for sha in previous_parent_hashes:
        assert sha in parent_components
        blocked.update(parent_components[sha])
    # Canonical components precede filtering. Identical full parent sequences must
    # not reconnect an excluded component after accession aliases are collapsed.
    changed = True
    while changed:
        before = len(blocked)
        for components in parent_components.values():
            if components & blocked:
                blocked.update(components)
        changed = len(blocked) > before
    write(
        ROOT / "results/prior-exclusion.json",
        {
            "previous_source_rows": sorted(previous_ids),
            "previous_parent_hashes": sorted(previous_parent_hashes),
            "blocked_components": sorted(blocked),
            "previous_covariate_files": {
                name: prior["files"][name] for name in previous_inputs
            },
            "previous_outcomes_loaded": False,
        },
    )
    selected = [r for r in maps if r["table_number"] in (7, 8)]
    dispositions, eligible, parent_sequences = [], {}, {}
    for row in selected:
        origins = seq_occurrences[row["sequence_id"]]
        hashes = sorted({o["parent_sequence_sha256"] for o in origins})
        reasons = []
        if not row["component"]:
            reasons.append("no_source_component")
        if row["component"] in blocked:
            reasons.append("previous_predictive_component_or_parent")
        if not row["sequence_raw"] or set(row["sequence_raw"]) - set(
            "ACDEFGHIKLMNPQRSTVWY"
        ):
            reasons.append("noncanonical_sequence")
        if len(hashes) != 1:
            reasons.append("not_one_full_parent_sequence")
        if not origins:
            reasons.append("no_supported_origin")
        if len(hashes) == 1:
            proteins = {parents[o["accession"]]["sequence"] for o in origins}
            assert len(proteins) == 1
            sequence = next(iter(proteins))
            if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                reasons.append("noncanonical_parent")
        item = {
            "peptide_row_id": row["peptide_row_id"],
            "sequence_id": row["sequence_id"],
            "sequence": row["sequence_raw"],
            "table_number": row["table_number"],
            "component": row["component"],
            "split": split_for(row["component"]) if row["component"] else None,
            "parent_sequence_hashes": hashes,
            "source_parent_status": row["source_parent_status"],
            "origins": origins,
            "reasons": reasons,
            "included_before_group_support": not reasons,
        }
        dispositions.append(item)
        if not reasons:
            eligible[row["peptide_row_id"]] = item
            parent_sequences[hashes[0]] = sequence
    write(ROOT / "results/source-row-dispositions.json.gz", dispositions)
    channels = [
        r
        for r in pq.read_table(TYPED / "channels.parquet").to_pylist()
        if r["table_number"] in (7, 8)
    ]
    write(ROOT / "results/channels.json", channels)
    assertions = pq.read_table(TYPED / "source_assertions.parquet").to_pylist()
    write(ROOT / "results/context-assertions.json", assertions)
    results = {
        r["result_id"]: r
        for r in pq.read_table(
            TYPED / "quantity_results.parquet", filters=[("table_number", "in", [7, 8])]
        ).to_pylist()
    }
    grouped, quantity_dispositions, seen = defaultdict(list), [], set()
    for batch in pq.ParquetFile(TYPED / "quantity_measurements.parquet").iter_batches():
        for measurement in batch.to_pylist():
            if (
                measurement["result_id"] not in results
                or measurement["representation"] != "raw_tmt_intensity"
            ):
                continue
            result = results[measurement["result_id"]]
            assert measurement["measurement_id"] not in seen
            seen.add(measurement["measurement_id"])
            value = measurement["reported_value"]
            reasons = []
            if result["peptide_row_id"] not in eligible:
                reasons.append("source_row_excluded")
            if value is None or not math.isfinite(value) or value <= 0:
                reasons.append("not_positive_finite_measurement")
            item = {**measurement, **result, "reasons": reasons}
            quantity_dispositions.append(item)
            if reasons:
                continue
            source = eligible[result["peptide_row_id"]]
            parent = source["parent_sequence_hashes"][0]
            key = (
                source["component"],
                parent,
                result["table_number"],
                result["channel_id"],
            )
            grouped[key].append(
                {
                    **item,
                    "sequence": source["sequence"],
                    "sequence_id": source["sequence_id"],
                }
            )
    assert len(seen) == len(results), (
        "Each candidate result must have exactly one raw-intensity representation"
    )
    group_rows = []
    used_parents = {s: set() for s in ("train", "validation", "test")}
    used_components = {s: set() for s in used_parents}
    for (component, parent, table, channel), members in sorted(grouped.items()):
        sequences = {m["sequence"] for m in members}
        assert len(sequences) == len(members)
        split = split_for(component)
        included = len(sequences) >= 3
        item = {
            "component": component,
            "parent_sequence_sha256": parent,
            "table_number": table,
            "channel_id": channel,
            "split": split,
            "n_distinct_peptides": len(sequences),
            "included": included,
            "reason": None if included else "fewer_than_three_distinct_peptides",
            "measurement_ids": [m["measurement_id"] for m in members],
            "source_row_ids": [m["peptide_row_id"] for m in members],
            "peptides": sorted(sequences),
        }
        group_rows.append(item)
        if included:
            used_parents[split].add(parent)
            used_components[split].add(component)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert (
            not used_parents[a] & used_parents[b]
            and not used_components[a] & used_components[b]
        )
    write(ROOT / "results/group-support.json.gz", group_rows)
    write(ROOT / "results/quantity-dispositions.json.gz", quantity_dispositions)
    write(ROOT / "results/candidate-parents.json.gz", parent_sequences)
    gates = {
        s: len(used_parents[s]) >= threshold
        for s, threshold in (("train", 100), ("validation", 25), ("test", 25))
    }
    report = {
        "status": "support_passed_context_review_pending"
        if all(gates.values())
        else "support_gate_failed",
        "n_source_rows": len(selected),
        "n_source_rows_before_group_support": len(eligible),
        "n_prior_components_excluded": len(blocked),
        "n_prior_parent_sequences": len(previous_parent_hashes),
        "source_row_exclusion_counts": dict(
            Counter(reason for r in dispositions for reason in r["reasons"])
        ),
        "n_raw_quantity_cells": len(seen),
        "n_groups_before_minimum": len(group_rows),
        "splits": {
            s: {
                "n_distinct_parents": len(used_parents[s]),
                "n_components": len(used_components[s]),
                "n_supported_channel_groups": sum(
                    r["included"] and r["split"] == s for r in group_rows
                ),
            }
            for s in used_parents
        },
        "support_gates": gates,
        "source_manifest_sha256": digest(SOURCE / "manifest.json"),
        "source_schemas": "results/schemas.json",
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "new_artifact_bytes": sum(
            p.stat().st_size for p in ROOT.rglob("*") if p.is_file()
        ),
        "free_disk_bytes": shutil.disk_usage(REPO).free,
        "argv": sys.argv,
        "predictions_or_model_scores_computed": False,
    }
    assert report["seconds"] <= 600 and report["peak_rss_gib"] <= 3
    assert (
        report["new_artifact_bytes"] <= 200 * 1024**2
        and report["free_disk_bytes"] >= 2 * 1024**3
    )
    write(ROOT / "results/inventory.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
