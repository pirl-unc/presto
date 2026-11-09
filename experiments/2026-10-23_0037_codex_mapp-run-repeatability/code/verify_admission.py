"""Independently replay complete-run admission from original source tables."""

import copy
import csv
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
SOURCE = Path("/private/tmp/hitlist-v2-mapp-reference-second")
BASE = ROOT / "data/availability_recipes/2026-10-22-mapp-captured-composition-min2-v1"
COHORT = ROOT / "data/availability_recipes/2026-10-23-mapp-complete-run-composition-v1"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tsv(name):
    with (SOURCE / "external/pxd009092/assets/txt" / f"{name}.txt").open() as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def main():
    tick = time.monotonic()
    assert (
        sha(BASE / "manifest.json")
        == "df78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80"
    )
    for directory in (SOURCE, BASE, COHORT):
        manifest = read(directory / "manifest.json")
        for name, record in manifest["files"].items():
            assert sha(directory / name) == record["sha256"]
    assert (
        sha(SOURCE / "manifest.json")
        == read(COHORT / "manifest.json")["source_manifest_sha256"]
    )
    base_examples = rows(BASE / "examples.jsonl")
    base_peptides = {
        c["peptide_row_id"]: (e, c)
        for e in base_examples
        if e["condition"] == "UT"
        for c in e["candidates"]
    }
    peptides = {"pxd009092:peptide:" + r["id"]: r for r in tsv("peptides")}
    acquisitions = {}
    for row in tsv("summary"):
        if row["Experiment"] and row["Experiment"].rsplit("_", 2)[1] in (
            "UT",
            "TNF+IFN",
        ):
            acquisitions["pxd009092:run:" + row["Experiment"]] = row
    assert len(acquisitions) == 6
    evidence, direct = defaultdict(list), defaultdict(list)
    for row in tsv("evidence"):
        pid, channel = (
            "pxd009092:peptide:" + row["Peptide ID"],
            "pxd009092:run:" + row["Experiment"],
        )
        if pid in base_peptides and channel in acquisitions:
            eid = "pxd009092:evidence:" + row["id"]
            evidence[pid, channel].append(eid)
            if int(row["MS/MS Count"] or 0) > 0:
                direct[pid, channel].append(eid)
    # Metadata conservation is compared against typed relations; numerical
    # quantities and run/evidence identities are separately replayed from TSV.
    typed = SOURCE / "external/pxd009092/typed"
    quantities = {
        r["measurement_id"]: r
        for r in pq.read_table(
            typed / "quantity_measurements.parquet",
            filters=[
                ("peptide_row_id", "in", list(base_peptides)),
                ("representation", "=", "bead_filtered"),
            ],
        ).to_pylist()
    }
    channels = {
        r["channel_id"]: r
        for r in pq.read_table(typed / "channels.parquet").to_pylist()
    }
    expected_values, complete = {}, set()
    for pid in base_peptides:
        for cid, channel in acquisitions.items():
            col = "Intensity " + channel["Experiment"].replace(
                "_CP-HEK-T+I_", "_CP-HEK-T+I-MAPPbeadFilt_"
            )
            expected_values[pid, cid] = float(peptides[pid][col])
        if all(
            math.isfinite(expected_values[pid, cid]) and expected_values[pid, cid] > 0
            for cid in acquisitions
        ):
            complete.add(pid)
    counts = Counter(
        base_peptides[pid][0]["parent_sequence_sha256"] for pid in complete
    )
    admitted = {
        pid
        for pid in complete
        if counts[base_peptides[pid][0]["parent_sequence_sha256"]] >= 2
    }
    parent_ids = {base_peptides[pid][0]["parent_sequence_sha256"] for pid in admitted}
    observed = rows(COHORT / "run_observations.jsonl")
    assert len(observed) == len(base_peptides) * 6
    assert {(r["peptide_row_id"], r["channel"]["channel_id"]) for r in observed} == set(
        expected_values
    )
    for row in observed:
        pid, cid = row["peptide_row_id"], row["channel"]["channel_id"]
        example, _ = base_peptides[pid]
        original = acquisitions[cid]
        assert row["channel"] == channels[cid]
        assert row["channel"]["raw_file"] == original["Raw file"]
        assert row["channel"]["experiment_raw"] == original["Experiment"]
        assert row["condition"] == original["Experiment"].rsplit("_", 2)[1]
        assert row["channel"]["replicate_label"] == int(
            original["Experiment"].rsplit("_", 1)[1]
        )
        assert row["measurement"] == quantities[row["measurement"]["measurement_id"]]
        assert row["measurement"]["value"] == expected_values[pid, cid]
        assert row["evidence_ids"] == sorted(evidence[pid, cid])
        assert row["direct_msms_evidence_ids"] == sorted(direct[pid, cid])
        assert row["complete_peptide"] == (pid in complete)
        assert row["admitted_complete_parent_set"] == (pid in admitted)
        assert row["positive_finite"] == (expected_values[pid, cid] > 0)
        for key in ("parent_sequence_sha256", "component_id", "split"):
            assert row[key] == example[key]
    dispositions = rows(BASE / "peptide_dispositions.jsonl")
    for row in dispositions:
        pid = row["peptide_row_id"]
        if row["status"] == "admitted":
            if pid not in complete:
                row["reasons"].append("incomplete_positive_filtered_runs")
            elif pid not in admitted:
                row["reasons"].append("fewer_than_two_complete_peptides_per_parent")
            row["status"] = "admitted" if pid in admitted else "excluded"
    assert dispositions == rows(COHORT / "peptide_dispositions.jsonl")
    assert [
        r
        for r in rows(BASE / "parents.jsonl")
        if r["parent_sequence_sha256"] in parent_ids
    ] == rows(COHORT / "parents.jsonl")
    assert [
        r for r in rows(BASE / "contributors.jsonl") if r["peptide_row_id"] in admitted
    ] == rows(COHORT / "contributors.jsonl")
    actual_examples = {r["example_id"]: r for r in rows(COHORT / "examples.jsonl")}
    assert len(actual_examples) == 2 * len(parent_ids)
    for old in base_examples:
        if old["parent_sequence_sha256"] not in parent_ids:
            continue
        expected = copy.deepcopy(old)
        expected["example_id"] = old["example_id"].replace(
            "mapp_captured_composition_min2_v1:", "mapp_complete_run_composition_v1:", 1
        )
        expected["candidates"] = [
            c for c in expected["candidates"] if c["peptide_row_id"] in admitted
        ]
        actual = actual_examples[expected["example_id"]]
        assert {k: v for k, v in expected.items() if k != "candidates"} == {
            k: v for k, v in actual.items() if k != "candidates"
        }
        linear = [10 ** c["source_log10_intensity"] for c in expected["candidates"]]
        mean = sum(c["source_log10_intensity"] for c in expected["candidates"]) / len(
            linear
        )
        for i, (a, b) in enumerate(zip(expected["candidates"], actual["candidates"])):
            assert {k: v for k, v in a.items() if not k.startswith("target_")} == {
                k: v for k, v in b.items() if not k.startswith("target_")
            }
            assert math.isclose(
                b["target_probability"], linear[i] / sum(linear), abs_tol=1e-14
            )
            assert math.isclose(
                b["target_log_probability"],
                math.log(linear[i] / sum(linear)),
                abs_tol=1e-14,
            )
            assert math.isclose(
                b["target_centered_log10"],
                a["source_log10_intensity"] - mean,
                abs_tol=1e-14,
            )
    result = {
        "passed": True,
        "source_manifest_sha256": sha(SOURCE / "manifest.json"),
        "base_manifest_sha256": sha(BASE / "manifest.json"),
        "recipe_manifest_sha256": sha(COHORT / "manifest.json"),
        "base_peptides": len(base_peptides),
        "run_values_and_metadata": len(observed),
        "all_source_dispositions": len(dispositions),
        "complete_peptides_before_parent_gate": len(complete),
        "admitted_peptides": len(admitted),
        "admitted_parents": len(parent_ids),
        "seconds": time.monotonic() - tick,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "verifier_sha256": sha(__file__),
        "scope": "Original TSV run/value/evidence replay, typed metadata conservation, base-recipe partitions/admission, complete eligibility and target arithmetic",
    }
    (FAMILY / "results/independent-admission.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
