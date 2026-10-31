"""Independent source-cell, exclusion-graph and prospective-support reconstruction."""

import gzip
import hashlib
import json
import math
import resource
import shutil
import time
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from xml.etree import ElementTree as ET

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/private/tmp/hitlist-v2-reference-200-final")
TYPED = SOURCE / "external/pxd057588/typed"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read(name):
    path = ROOT / "results" / name
    return json.loads(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )


def original_cells(path):
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")):
                strings.append("".join(node.text or "" for node in item.iter(NS + "t")))
        cells = {}
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        for item in root.iter(NS + "c"):
            kind = item.get("t")
            value = item.findtext(NS + "v")
            if kind == "s":
                value = strings[int(value)]
            elif kind == "inlineStr":
                value = "".join(node.text or "" for node in item.iter(NS + "t"))
            elif value is not None:
                value = float(value)
            cells[item.get("r")] = value
        return cells


def partition(component):
    raw = hashlib.sha256(
        ("pxd057588_polarization_composition_v1:" + component).encode()
    ).digest()
    bucket = int.from_bytes(raw[:8], byteorder="big") % 100
    return "train" if bucket <= 69 else "validation" if bucket <= 84 else "test"


def main():
    start = time.perf_counter()
    destination = ROOT / "results/independent-verification.json"
    assert not destination.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/verify.py")
    inventory = read("inventory.json")
    assert (
        hashlib.sha256((SOURCE / "manifest.json").read_bytes()).hexdigest()
        == inventory["source_manifest_sha256"]
    )
    for name, record in read("schemas.json").items():
        assert (
            hashlib.sha256((TYPED / f"{name}.parquet").read_bytes()).hexdigest()
            == record["sha256"]
        )
    maps = {
        r["peptide_row_id"]: r
        for r in pq.read_table(TYPED / "reference_source_mappings.parquet").to_pylist()
    }
    occurrences = pq.read_table(TYPED / "reference_occurrences.parquet").to_pylist()
    by_sequence = defaultdict(list)
    graph = defaultdict(set)
    for row in occurrences:
        by_sequence[row["sequence_id"]].append(row)
        component, parent = (
            "component:" + row["component"],
            "parent:" + row["parent_sequence_sha256"],
        )
        graph[component].add(parent)
        graph[parent].add(component)
    exclusion = read("prior-exclusion.json")
    seeds = {
        "component:" + maps[key]["component"]
        for key in exclusion["previous_source_rows"]
    }
    seeds.update("parent:" + value for value in exclusion["previous_parent_hashes"])
    reached = set(seeds)
    queue = deque(seeds)
    while queue:
        for neighbor in graph[queue.popleft()]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    blocked = {
        value.removeprefix("component:")
        for value in reached
        if value.startswith("component:")
    }
    assert blocked == set(exclusion["blocked_components"])
    parents = {
        r["sequence_sha256"]: r["sequence"]
        for r in pq.read_table(TYPED / "reference_parents.parquet").to_pylist()
        if r["sequence"]
    }
    source_rows = read("source-row-dispositions.json.gz")
    assert {r["peptide_row_id"] for r in source_rows} == {
        key for key, value in maps.items() if value["table_number"] in (7, 8)
    }
    eligible = {}
    for row in source_rows:
        source = maps[row["peptide_row_id"]]
        origins = by_sequence[source["sequence_id"]]
        hashes = {o["parent_sequence_sha256"] for o in origins}
        reasons = []
        if not source["component"]:
            reasons.append("no_source_component")
        if source["component"] in blocked:
            reasons.append("previous_predictive_component_or_parent")
        if not source["sequence_raw"] or any(
            c not in "ACDEFGHIKLMNPQRSTVWY" for c in source["sequence_raw"]
        ):
            reasons.append("noncanonical_sequence")
        if len(hashes) != 1:
            reasons.append("not_one_full_parent_sequence")
        if not origins:
            reasons.append("no_supported_origin")
        if len(hashes) == 1 and any(
            c not in "ACDEFGHIKLMNPQRSTVWY" for c in parents[next(iter(hashes))]
        ):
            reasons.append("noncanonical_parent")
        assert sorted(row["reasons"]) == sorted(reasons)
        assert row["included_before_group_support"] == (not reasons)
        assert row["parent_sequence_hashes"] == sorted(hashes)
        assert row["origins"] == origins
        assert row["sequence"] == source["sequence_raw"]
        assert row["split"] == (
            partition(source["component"]) if source["component"] else None
        )
        if not reasons:
            eligible[row["peptide_row_id"]] = row
    originals = {
        table: original_cells(
            SOURCE / f"external/pxd057588/assets/PXD057588/table{table}.xlsx"
        )
        for table in (7, 8)
    }
    channels = {r["channel_id"]: r for r in read("channels.json")}
    for row in source_rows:
        mapping = maps[row["peptide_row_id"]]
        cells = originals[row["table_number"]]
        assert cells[mapping["sequence_cell"]] == row["sequence"]
        assert cells[mapping["parent_label_cell"]] == mapping["parent_label_raw"]
    for row in channels.values():
        header = originals[row["table_number"]][row["channel_label_cell"]]
        if isinstance(header, float) and header.is_integer():
            header = int(header)
        assert str(header) == row["channel_label_raw"]
        assert (
            originals[row["table_number"]][row["sample_label_cell"]]
            == row["sample_label_raw"]
        )
    grouped = defaultdict(dict)
    quantities = read("quantity-dispositions.json.gz")
    measurement_ids = set()
    for row in quantities:
        assert row["measurement_id"] not in measurement_ids
        measurement_ids.add(row["measurement_id"])
        assert row["representation"] == "raw_tmt_intensity"
        assert (
            row["reported_value"]
            == originals[row["table_number"]][row["source_intensity_cell"]]
        )
        assert row["source_cell"] == row["source_intensity_cell"]
        reasons = []
        if row["peptide_row_id"] not in eligible:
            reasons.append("source_row_excluded")
        if (
            row["reported_value"] is None
            or not math.isfinite(row["reported_value"])
            or row["reported_value"] <= 0
        ):
            reasons.append("not_positive_finite_measurement")
        assert reasons == row["reasons"]
        if reasons:
            continue
        source = eligible[row["peptide_row_id"]]
        key = (
            source["component"],
            source["parent_sequence_hashes"][0],
            row["table_number"],
            row["channel_id"],
        )
        assert source["sequence"] not in grouped[key]
        grouped[key][source["sequence"]] = row
    assert len(quantities) == sum(
        sum(r["table_number"] == table for r in source_rows)
        * sum(r["table_number"] == table for r in channels.values())
        for table in (7, 8)
    )
    groups = read("group-support.json.gz")
    assert len(groups) == len(grouped)
    used = {s: set() for s in ("train", "validation", "test")}
    for row in groups:
        key = (
            row["component"],
            row["parent_sequence_sha256"],
            row["table_number"],
            row["channel_id"],
        )
        members = grouped[key]
        assert row["n_distinct_peptides"] == len(members)
        assert row["peptides"] == sorted(members)
        assert set(row["measurement_ids"]) == {
            r["measurement_id"] for r in members.values()
        }
        assert set(row["source_row_ids"]) == {
            r["peptide_row_id"] for r in members.values()
        }
        assert row["included"] == (len(members) >= 3)
        assert row["split"] == partition(row["component"])
        if row["included"]:
            used[row["split"]].add(row["parent_sequence_sha256"])
    for split, values in used.items():
        assert len(values) == inventory["splits"][split]["n_distinct_parents"]
        threshold = 100 if split == "train" else 25
        assert inventory["support_gates"][split] == (len(values) >= threshold)
    assert (
        not used["train"] & used["validation"]
        and not used["train"] & used["test"]
        and not used["validation"] & used["test"]
    )
    report = {
        "status": "passed",
        "original_quantity_cells_verified": len(quantities),
        "original_sequence_parent_cells_verified": 2 * len(source_rows),
        "source_row_dispositions_verified": len(source_rows),
        "candidate_groups_verified": len(groups),
        "independent_exclusion_graph": "breadth-first component/full-parent graph",
        "original_parser": "OOXML XML/shared-string decoder; no training curator",
        "n_blocked_components": len(blocked),
        "support_gates": inventory["support_gates"],
        "predictions_or_model_scores_computed": False,
        "seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    assert report["seconds"] <= 600 and report["peak_rss_gib"] <= 3
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
