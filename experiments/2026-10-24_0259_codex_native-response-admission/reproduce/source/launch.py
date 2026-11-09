"""Build the shared native source/recipe and replay original source invariants."""
import collections
import gzip
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    tick = time.perf_counter()
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    checkout = Path(config["hitlist_checkout"])
    for name, expected in config["hitlist_source_sha256"].items():
        assert digest(checkout / name) == expected, name
    sys.path.insert(0, str(checkout))
    import hitlist
    import pyarrow.parquet as pq
    from hitlist import EvidenceStore, write_tmab_peptidomics_package

    assert Path(hitlist.__file__).resolve() == checkout / "hitlist/__init__.py"
    assert not (FAMILY / "results/materialization.json").exists()
    source = Path(config["source_package"])
    prior = Path(config["old_package"])
    assert digest(prior / "manifest.json") == config["old_manifest_sha256"]
    source_manifest = write_tmab_peptidomics_package(config["input_directory"], source)
    old_rows = pq.read_table(prior / "tmab/source_rows.parquet").to_pylist()
    new_rows = pq.read_table(source / "tmab/source_rows.parquet").to_pylist()
    old_measurements = pq.read_table(prior / "tmab/measurements.parquet").to_pylist()
    new_measurements = pq.read_table(source / "tmab/measurements.parquet").to_pylist()
    old_by_id = {r["measurement_id"]: r for r in old_measurements}
    assert set(old_by_id) == {r["measurement_id"] for r in new_measurements}
    old_row_by_id = {r["source_record_id"]: r for r in old_rows}
    assert set(old_row_by_id) == {r["source_record_id"] for r in new_rows}
    n_invariants = 0
    for row in new_rows:
        old = old_row_by_id[row["source_record_id"]]
        for field, value in old.items():
            if field not in {"source_sequence_raw", "row_identity_status"}:
                assert row[field] == value, (row["source_record_id"], field)
                n_invariants += 1
        if row["source_row_disposition"] == "primary_ratio_panel":
            assert row["source_sequence_raw"] == old["source_sequence_raw"]
    mutable = {"peptide_sequence", "peptide_modifications_raw", "context_id", "lineage_group_id",
               "eligibility_status", "source_reuse_status"}
    for m in new_measurements:
        for field, value in old_by_id[m["measurement_id"]].items():
            if field not in mutable:
                assert m[field] == value, (m["measurement_id"], field)
                n_invariants += 1
    lineages = pq.read_table(source / "tmab/lineage.parquet").to_pylist()
    new_by_id = {m["measurement_id"]: m for m in new_measurements}
    memberships = collections.defaultdict(set)
    known_groups = set()
    for line in lineages:
        group = line["candidate_reuse_group_id"]
        if group:
            m = new_by_id[line["measurement_id"]]
            memberships[group].add((m["source_file"], m["source_row_index"]))
            if m["peptide_sequence_raw"]:
                known_groups.add(group)
    asset_files = {"dasgupta2014_s001": "dasgupta_s1.xlsx", "gelman2013_s004": "gelman_s1.xlsx"}
    expected_known = set()
    with gzip.open(Path(config["old_independent_audit"]) / "results/cross-paper-candidates.jsonl.gz", "rt") as stream:
        for line in stream:
            group = json.loads(line)
            expected_known.add(tuple(sorted((asset_files[r["asset"]], r["row"]) for r in group["rows"])))
    assert {tuple(sorted(memberships[g])) for g in known_groups} == expected_known
    source_result = dict(n_original_field_invariants=n_invariants,
                         n_source_rows=len(new_rows), n_measurement_slots=len(new_measurements),
                         n_known_sequence_reuse_groups=len(known_groups),
                         n_total_possible_reuse_groups=len(memberships),
                         n_unidentified_possible_reuse_groups=len(memberships)-len(known_groups),
                         n_candidate_rows=sum(map(len, memberships.values())),
                         source_manifest_sha256=digest(source_manifest))
    dump(FAMILY / "results/source-replay.json", source_result)
    dump(FAMILY / "results/source-manifest.json", json.loads(source_manifest.read_text()))
    del old_rows, old_measurements, old_by_id, old_row_by_id, new_rows, new_measurements, new_by_id, lineages
    store = EvidenceStore.open(source)
    output = Path(config["recipe_directory"])
    recipe_manifest = store.materialize(config["recipe"], output)
    manifest = json.loads(recipe_manifest.read_text())
    dump(FAMILY / "results/recipe-manifest.json", manifest)
    dump(FAMILY / "results/recipe-summary.json", manifest["coverage"])
    examples = read_gzip(output / "examples.json.gz")
    by_split = {}
    for split in ("train", "val", "test"):
        groups = [g for g in examples if g["split"] == split]
        by_split[split] = dict(
            n_response_groups=len(groups), n_components=len({g["component_id"] for g in groups}),
            n_conditions=len({g["condition_id"] for g in groups}),
            n_effective_observations=sum(len(g["observations"]) for g in groups),
            n_exact_observations=sum(o["inequality"] is None for g in groups for o in g["observations"]),
            n_bounded_observations=sum(o["inequality"] is not None for g in groups for o in g["observations"]),
            n_full_parent_supported_groups=sum(g["full_parent_model_supported"] for g in groups),
            total_weight=sum(o["weight"] for g in groups for o in g["observations"]),
        )
    dump(FAMILY / "results/split-coverage.json", by_split)
    summary = dict(hitlist_version=hitlist.__version__, source_package=str(source),
                   recipe_directory=str(output), source_manifest_sha256=digest(source_manifest),
                   recipe_manifest_sha256=digest(recipe_manifest), source_replay=source_result,
                   recipe_coverage=manifest["coverage"], split_coverage=by_split,
                   seconds=time.perf_counter()-tick, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   independent_recipe_verification="pending", biological_training=False)
    dump(FAMILY / "results/materialization.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    assert summary["seconds"] < config["seconds_cap"] and summary["peak_rss_bytes"] < config["rss_bytes_cap"]


if __name__ == "__main__":
    main()
