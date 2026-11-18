"""Independent source, component and random-draw replay before MS predictions."""

from bisect import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import resource
import shutil
import signal
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from presto.data.modular import Observation

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
INPUT = Path("/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01")
START = time.monotonic()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def segments(peptide):
    return set(re.findall(r"(?=(.{9}))", peptide))


def guard():
    assert time.monotonic() - START < 300
    assert psutil.virtual_memory().available >= 10 * 1024**3
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024**3
    assert shutil.disk_usage(FAMILY).free >= 10 * 1024**3


def independent_components(peptides):
    neighbors = defaultdict(set)
    for peptide in peptides:
        for key in segments(peptide):
            neighbors[key].add(peptide)
    remaining, result = set(peptides), {}
    while remaining:
        start = min(remaining)
        members, stack, used = set(), [start], set()
        while stack:
            peptide = stack.pop()
            if peptide in members:
                continue
            members.add(peptide)
            for key in segments(peptide) - used:
                used.add(key)
                stack.extend(neighbors[key] - members)
        remaining.difference_update(members)
        result.update({peptide: min(members) for peptide in members})
    return result


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    signal.alarm(300)
    assert psutil.virtual_memory().available >= 13 * 1024**3, "startup RAM reserve"
    guard()
    receipt = read(FAMILY / "reproduce/launch.json")
    assert receipt["status"] == "prepared_pending_independent_verification"
    for name, expected in receipt["code_hashes"].items():
        assert sha(FAMILY / "code" / name) == expected
    for name, expected in receipt["inputs"]["admitted"].items():
        assert sha(INPUT / name) == expected
    summary = read(FAMILY / "results/cohort.json")
    affinity = read(receipt["inputs"]["affinity_manifest"]["path"])
    assert (
        sha(receipt["inputs"]["affinity_manifest"]["path"])
        == receipt["inputs"]["affinity_manifest"]["sha256"]
    )
    exposed = set()
    for item in affinity["sources"]:
        assert sha(item["path"]) == item["sha256"]
        with Path(item["path"]).open() as stream:
            for line in stream:
                exposed.update(segments(json.loads(line)["peptide"]))
    expected_units, expected_lineage, peptides = {}, [], defaultdict(set)
    with (INPUT / "observations.jsonl").open("rb") as stream:
        ordinal = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            raw = row["provenance"]["raw_record"]
            study = str(raw.get("pmid") or "")
            if study in {"35251023", "34433824"} and len(row["mhc"]) == 1:
                pair = row["mhc"][0]
                key = (study, row["peptide"], pair["groove1"], pair["groove2"])
                unit_id = "ms-unit:" + hashlib.sha256(json.dumps(key).encode()).hexdigest()
                expected_lineage.append(
                    dict(
                        unit_id=unit_id,
                        study=study,
                        observation_id=row["observation_id"],
                        admission_ordinal=ordinal,
                        admission_byte_offset=offset,
                        native_ordinal=row["provenance"]["source_ordinal"],
                        provenance_id=raw["provenance_id"],
                    )
                )
                peptides[study].add(row["peptide"])
                current = expected_units.get(key)
                if current is None or row["observation_id"] < current["id"]:
                    expected_units[key] = {
                        "id": row["observation_id"],
                        "mhc": row["mhc"],
                        "offset": offset,
                        "unit_id": unit_id,
                    }
            ordinal += 1
            if ordinal % 1000 == 0:
                guard()
    actual_lineage = pq.read_table(FAMILY / "results/source-lineage.parquet").to_pylist()
    assert actual_lineage == expected_lineage
    expected_dispositions, assignments = [], {}
    dp_components = independent_components(peptides["35251023"])
    bad_components = {
        group for peptide, group in dp_components.items() if segments(peptide) & exposed
    }
    for peptide, component in sorted(dp_components.items()):
        value = (
            int.from_bytes(
                hashlib.sha256(("presto-ms-dp-v1|" + component).encode()).digest()[:8], "big"
            )
            / 2**64
        )
        split = "train" if value < 0.70 else "validation" if value < 0.85 else "test"
        reason = "included"
        if split != "train" and component in bad_components:
            reason = "initializer_exposed_evaluation_component"
        expected_dispositions.append(
            dict(study="35251023", peptide=peptide, component=component, split=split, reason=reason)
        )
        if reason == "included":
            assignments[("35251023", peptide)] = (
                split,
                "dp:" + hashlib.sha256(component.encode()).hexdigest(),
            )
    all_dp = set().union(*(segments(p) for p in peptides["35251023"]))
    dq_components = independent_components(peptides["34433824"])
    for peptide, component in sorted(dq_components.items()):
        reason = (
            "dp_source_overlap"
            if segments(peptide) & all_dp
            else "initializer_overlap"
            if segments(peptide) & exposed
            else "included"
        )
        expected_dispositions.append(
            dict(
                study="34433824",
                peptide=peptide,
                component=component,
                split="external",
                reason=reason,
            )
        )
        if reason == "included":
            assignments[("34433824", peptide)] = (
                "external",
                "dq8:" + hashlib.sha256(component.encode()).hexdigest(),
            )
    assert (
        pq.read_table(FAMILY / "results/dispositions.parquet").to_pylist() == expected_dispositions
    )
    guard()
    counts = Counter(
        residue
        for study, peptide in assignments
        if study == "35251023" and assignments[(study, peptide)][0] == "train"
        for residue in peptide
    )
    frequencies = read(FAMILY / "results/background-frequencies.json")
    assert dict(zip(frequencies["alphabet"], frequencies["counts"])) == counts
    native = receipt["inputs"]["native"]
    assert sha(native["path"]) == native["sha256"]
    known = set()
    for batch in pq.ParquetFile(native["path"]).iter_batches(
        batch_size=50000, columns=["peptide", "is_ms_observation", "ms_outcome_status"]
    ):
        for row in batch.to_pylist():
            if row["is_ms_observation"] is True and row["ms_outcome_status"] in {
                "positive_result",
                "curated_ligand_table",
            }:
                known.add(row["peptide"])
        guard()
    with gzip.open(FAMILY / "results/generated-backgrounds.jsonl.gz", "rt") as stream:
        controls = {row["positive"]: row for line in stream if (row := json.loads(line))}
    split_by_peptide = {}
    owners = defaultdict(set)
    for (_, peptide), (split, _) in assignments.items():
        split_by_peptide[peptide] = split
        for segment in segments(peptide):
            owners[segment].add(split)
    generated = set()
    cumulative = list(itertools.accumulate(frequencies["counts"]))
    alphabet = frequencies["alphabet"]
    for peptide, split in sorted(split_by_peptide.items(), key=lambda item: (item[1], item[0])):
        stored = controls[peptide]
        seed = int.from_bytes(
            hashlib.sha256(("presto-ms-length-background-v1|" + peptide).encode()).digest(), "big"
        )
        assert stored["seed"] == str(seed) and stored["split"] == split
        rng = random.Random(seed)
        for attempt in range(1, stored["attempt"] + 1):
            candidate = "".join(
                alphabet[bisect(cumulative, rng.random() * cumulative[-1], 0, len(alphabet) - 1)]
                for _ in peptide
            )
            parts = segments(candidate)
            invalid = (
                candidate == peptide
                or candidate in known
                or candidate in generated
                or any(owners[p] - {split} for p in parts)
                or (split != "train" and bool(parts & exposed))
            )
            assert invalid == (attempt != stored["attempt"])
        assert candidate == stored["peptide"]
        generated.add(candidate)
        for part in parts:
            owners[part].add(split)
    assert set(controls) == set(split_by_peptide)
    guard()
    observed_units, observed_controls, row_counts = set(), set(), Counter()
    for source in summary["sources"]:
        assert sha(source["path"]) == source["sha256"]
        split = source["partition"]
        with Path(source["path"]).open() as stream:
            for line in stream:
                row = Observation.from_dict(json.loads(line))
                assert row.source == source["source"]
                assert row.split == ("test" if split == "external" else split)
                assert not row.context.interventions and row.context.cell_type is None
                study = row.provenance["study"]
                if row.value == 1:
                    pair = row.mhc[0]
                    key = (study, row.peptide, pair.groove1, pair.groove2)
                    original = expected_units[key]
                    assert row.observation_id == original["id"]
                    assert row.provenance["unit_id"] == original["unit_id"]
                    assert row.provenance["admitted_byte_offset"] == original["offset"]
                    assert [vars(pair)] == original["mhc"]
                    assert assignments[(study, row.peptide)] == (split, row.group_id)
                    assert key not in observed_units
                    observed_units.add(key)
                else:
                    assert row.value == 0
                    assert row.provenance["label_kind"] == "generated_control_not_measured_negative"
                    parent = row.provenance["parent_peptide"]
                    pair = row.mhc[0]
                    key = (study, parent, pair.groove1, pair.groove2)
                    assert row.peptide == controls[parent]["peptide"]
                    assert row.provenance["parent_observation_id"] == expected_units[key]["id"]
                    assert assignments[(study, parent)] == (split, row.group_id)
                    assert key not in observed_controls
                    observed_controls.add(key)
                row_counts[split] += 1
                if sum(row_counts.values()) % 1000 == 0:
                    guard()
    expected = {key for key in expected_units if key[:2] in assignments}
    assert observed_units == observed_controls == expected
    assert dict(row_counts) == summary["row_counts"]
    assert summary["native_final_steps"] == 2592
    signal.alarm(0)
    result = {
        "status": "passed",
        "utc": datetime.now(timezone.utc).isoformat(),
        "agent_model": "Codex / GPT-6",
        "code_sha256": sha(__file__),
        "row_counts": dict(row_counts),
        "positive_units": len(expected),
        "source_lineage_rows": len(expected_lineage),
        "controls": len(controls),
        "components_method": "independent segment graph flood fill",
        "control_method": "independent cumulative-frequency inverse-CDF replay",
        "elapsed_seconds": time.monotonic() - START,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "argv": list(__import__("sys").argv),
    }
    path = FAMILY / "results/independent-cohort-verification.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
