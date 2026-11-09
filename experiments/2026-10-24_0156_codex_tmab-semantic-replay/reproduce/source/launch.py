"""Source-preserving public TMAB build and independent semantic replay."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
RESULTS = FAMILY / "results"
ASSET_FILES = {
    "dasgupta2014_s001": "dasgupta_s1.xlsx",
    "gelman2013_s004": "gelman_s1.xlsx",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def source_identity(raw):
    """Whole-string grammar independent of the production token scanner."""
    if raw is None or not str(raw).strip():
        return None, [], "unidentified", "unresolved"
    raw = str(raw).strip()
    tentative = raw.startswith("(") and raw.endswith(")")
    body = raw[1:-1].strip() if tentative else raw
    match = re.fullmatch(r"(Ac-)?((?:Mox|[ACDEFGHIKLMNPQRSTVWY])+)", body)
    if match is None:
        return None, [], "unresolved_annotation", "unresolved_source_notation"
    tokens = re.findall(r"Mox|[ACDEFGHIKLMNPQRSTVWY]", match[2])
    modifications = ([{"name": "N_terminal_acetylation", "position": 0, "source_token": "Ac-"}]
                     if match[1] else [])
    modifications.extend({"name": "oxidation", "position": i, "source_token": "Mox"}
                         for i, token in enumerate(tokens, 1) if token == "Mox")
    return ("".join("M" if t == "Mox" else t for t in tokens), modifications,
            "tentative" if tentative else "reported_sequence",
            "supported_source_notation" if modifications else "none_reported")


def main():
    started = time.perf_counter()
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    checkout = Path(config["hitlist_checkout"])
    assert not (RESULTS / "replay.json").exists(), "Never overwrite an executed attempt"
    for name, expected in config["hitlist_source_sha256"].items():
        assert digest(checkout / name) == expected, name
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == config["hitlist_base"]
    old = Path(config["old_package"])
    new = Path(config["new_package"])
    assert digest(old / "manifest.json") == config["old_manifest_sha256"]
    sys.path.insert(0, str(checkout))
    import hitlist
    from hitlist import EvidenceStore, write_tmab_peptidomics_package
    import pyarrow.parquet as pq

    assert Path(hitlist.__file__).resolve() == checkout / "hitlist/__init__.py"
    manifest_path = write_tmab_peptidomics_package(config["source_directory"], new)
    store = EvidenceStore.open(new)
    dump(RESULTS / "old-manifest.json", json.loads((old / "manifest.json").read_text()))
    dump(RESULTS / "new-manifest.json", json.loads(manifest_path.read_text()))
    old_rows = pq.read_table(old / "tmab/source_rows.parquet").to_pylist()
    new_rows = pq.read_table(new / "tmab/source_rows.parquet").to_pylist()
    old_measurements = pq.read_table(old / "tmab/measurements.parquet").to_pylist()
    measurements = pq.read_table(new / "tmab/measurements.parquet").to_pylist()
    lineage = pq.read_table(new / "tmab/lineage.parquet").to_pylist()
    contexts = pq.read_table(new / "tmab/protocol_context.parquet").to_pylist()
    failures = []
    checks = collections.Counter()

    def check(label, actual, expected, identity=None):
        checks[label] += 1
        if actual != expected:
            failures.append(dict(check=label, identity=identity, actual=actual, expected=expected))

    def map_unique(rows, key):
        result = {r[key]: r for r in rows}
        check("unique_" + key, len(result), len(rows))
        return result

    old_row_map = map_unique(old_rows, "source_record_id")
    new_row_map = map_unique(new_rows, "source_record_id")
    old_by_id = map_unique(old_measurements, "measurement_id")
    new_by_id = map_unique(measurements, "measurement_id")
    lineage_by_id = map_unique(lineage, "measurement_id")
    context_by_id = map_unique(contexts, "context_id")
    check("source_row_ids_preserved", sorted(new_row_map), sorted(old_row_map))
    check("measurement_ids_preserved", sorted(new_by_id), sorted(old_by_id))
    check("lineage_covers_measurements", sorted(lineage_by_id), sorted(new_by_id))
    changed_rows = []
    for identity, row in new_row_map.items():
        before = old_row_map[identity]
        for key, value in before.items():
            if key not in {"source_sequence_raw", "row_identity_status"}:
                check("raw_row_field_" + key, row[key], value, identity)
        values = json.loads(row["source_row_json"])
        if row["source_row_disposition"] == "deferred_heatmap_republication":
            index = 5 if row["source_file"] == "gelman_s2.xlsx" else 4
            expected = None if values[index] is None else str(values[index]).strip() or None
            check("heatmap_sequence_column", row["source_sequence_raw"], expected, identity)
        else:
            check("primary_sequence_raw_preserved", row["source_sequence_raw"], before["source_sequence_raw"], identity)
            check("row_identification", row["row_identity_status"], source_identity(row["source_sequence_raw"])[2], identity)
        changes = {k: {"old": before.get(k), "new": v} for k, v in row.items() if before.get(k) != v}
        if changes:
            changed_rows.append(dict(source_record_id=identity, source_file=row["source_file"],
                                     source_row_index=row["source_row_index"], changes=changes))
    write_jsonl(RESULTS / "changed-source-rows.jsonl.gz", changed_rows)

    mutable = {"context_id", "lineage_group_id", "peptide_sequence", "peptide_modifications_raw",
               "eligibility_status", "source_reuse_status"}
    changes = []
    for identity, m in new_by_id.items():
        before = old_by_id[identity]
        for field, value in before.items():
            if field not in mutable:
                check("measurement_invariant_" + field, m[field], value, identity)
        seq, mods, identification, modification_status = source_identity(m["peptide_sequence_raw"])
        check("normalized_sequence", m["peptide_sequence"], seq, identity)
        check("structured_modifications", json.loads(m["peptide_modifications_json"]), mods, identity)
        check("identification_status", m["identification_status"], identification, identity)
        check("modification_status", m["modification_status"], modification_status, identity)
        v = m["quantitative_value"]
        if m["ratio_role"] != "treated_vs_control":
            expected = "deferred_control_or_derived_average"
        elif identification != "reported_sequence":
            expected = "deferred_" + identification
        elif v is None or not math.isfinite(v):
            expected = "deferred_nonquantifiable"
        elif v <= 0:
            expected = "deferred_nonpositive_ratio"
        else:
            expected = {"exact": "candidate_exact", "source_declared_bound": "candidate_censored"}.get(
                m["censoring_status"], "deferred_nonquantifiable")
        check("eligibility", m["eligibility_status"], expected, identity)
        check("denominator_identity_unresolved", m["shared_control_group_status"],
              "source_label_proxy_unresolved_denominator", identity)
        line = lineage_by_id[identity]
        check("specimen_independence_unknown", line["independent_specimen_status"], "unknown", identity)
        check("lineage_group_join", line["lineage_group_id"], m["lineage_group_id"], identity)
        check("context_join", m["context_id"] in context_by_id, True, identity)
        changed = {k: {"old": before.get(k), "new": val} for k, val in m.items() if before.get(k) != val}
        if changed:
            changes.append(dict(measurement_id=identity, source_file=m["source_file"],
                                source_cell=m["source_cell"], changes=changed))
    write_jsonl(RESULTS / "changed-measurements.jsonl.gz", changes)

    prior_family = Path(config["old_family"])
    expected_groups = read_jsonl(prior_family / "results/cross-paper-candidates.jsonl.gz")
    expected_memberships = {tuple(sorted((ASSET_FILES[r["asset"]], r["row"]) for r in g["rows"]))
                            for g in expected_groups}
    groups = collections.defaultdict(set)
    per_slot = collections.defaultdict(set)
    group_label_status = collections.defaultdict(set)
    for line in lineage:
        m = new_by_id[line["measurement_id"]]
        group = line["candidate_reuse_group_id"]
        if group is not None:
            groups[group].add((m["source_file"], m["source_row_index"]))
            per_slot[(group, m["ratio_role"], m["replicate_label"])].add(line["lineage_group_id"])
            group_label_status[group].add(line["experiment_label_agreement"])
            check("possible_reuse_status", line["lineage_status"], "possible_cross_paper_reuse", m["measurement_id"])
        else:
            check("unmatched_lineage_unknown", line["lineage_status"], "unresolved_source_lineage", m["measurement_id"])
    actual_memberships = {tuple(sorted(members)) for members in groups.values()}
    check("candidate_reuse_memberships", sorted(actual_memberships), sorted(expected_memberships))
    for key, ids in per_slot.items():
        check("shared_candidate_slot_group", len(ids), 1, key)
    source_primary = read_jsonl(prior_family / "results/primary-row-audit.jsonl.gz")
    primary_map = {(ASSET_FILES[r["asset"]], r["row"]): r for r in source_primary}
    for group, members in groups.items():
        labels = {" ".join((primary_map[member]["experiment"] or "").lower().split()) for member in members}
        expected = "equal_after_case_whitespace_normalization" if len(labels) == 1 else "different_source_labels"
        check("reuse_label_agreement", sorted(group_label_status[group]), [expected], group)
    write_jsonl(RESULTS / "candidate-memberships.jsonl.gz", [
        dict(candidate_group=group, members=sorted(members), label_agreement=sorted(group_label_status[group]))
        for group, members in sorted(groups.items())])

    # Verify protocol assertions against the frozen original row annotations.
    expected_context_members = collections.defaultdict(list)
    for m in measurements:
        expected_context_members[m["context_id"]].append(primary_map[(m["source_file"], m["source_row_index"])])
    for context in contexts:
        for r in expected_context_members[context["context_id"]]:
            check("context_literal_experiment", context["treatment_label_raw"], r["experiment"], context["context_id"])
            raw = r["average_annotation"]
            annotation = None if raw is None else str(raw).strip() or None
            check("context_literal_average", context["average_annotation_raw"], annotation, context["context_id"])
        instrument = "Waters Synapt G2" if context["source_id"] == "dasgupta_2014" else "Waters Synapt G1"
        check("publishing_instrument", context["publishing_protocol_instrument"], instrument, context["context_id"])
        name = context["treatment_name"] or ""
        reused = name.startswith(("epoxomicin", "epoximicin")) or (context["source_id"] == "dasgupta_2014" and name == "bortezomib")
        check("original_instrument_scope", context["instrument"], None if reused else instrument, context["context_id"])
        check("instrument_status", context["instrument_status"], "unresolved_reused_acquisition" if reused else "reported_study_protocol", context["context_id"])
        harvest = "present_per_study_protocol" if context["source_id"] == "dasgupta_2014" and not reused else "unresolved"
        check("harvest_presence", context["harvest_compound_status"], harvest, context["context_id"])
    # These source-specific examples verify distinct intervals rather than copying
    # the production timeline parser into the verifier.
    for source_file, row, incubation, harvest, total in [
        ("dasgupta_s1.xlsx", 9, 20 / 60, 15 / 60, 35 / 60),
        ("gelman_s1.xlsx", 10, 10 / 60, None, 30 / 60),
    ]:
        m = next(m for m in measurements if m["source_file"] == source_file and m["source_row_index"] == row)
        c = context_by_id[m["context_id"]]
        check("example_incubation", c["incubation_duration_hours"], incubation, [source_file, row])
        check("example_harvest", c["harvest_duration_hours"], harvest, [source_file, row])
        check("example_total", c["duration_hours"], total, [source_file, row])

    # Every new field is exercised via the shared public query path.
    query_specs = {
        "tmab_measurement": ["measurement_id", "peptide_sequence", "peptide_modifications_json", "identification_status", "modification_status", "eligibility_status", "shared_control_group_status"],
        "tmab_protocol_context": list(contexts[0]),
        "tmab_measurement_lineage": list(lineage[0]),
        "tmab_source_asset": ["source_asset_id", "source_asset_role"],
    }
    physical = {"tmab_measurement": measurements, "tmab_protocol_context": contexts,
                "tmab_measurement_lineage": lineage}
    for relation, fields in query_specs.items():
        rows = store.query(relation, select=fields, limit=None).table.to_pylist()
        write_jsonl(RESULTS / (relation + "-query.jsonl.gz"), rows)
        if relation in physical:
            check("public_query_" + relation, rows, [{k: r[k] for k in fields} for r in physical[relation]])
        else:
            check("public_source_assets", len(rows), 4)
            check("public_asset_role_populated", all(r["source_asset_role"] for r in rows), True)
    dump(RESULTS / "contexts.json", contexts)
    dump(RESULTS / "disagreements.json", failures)
    old_coverage = json.loads((old / "tmab/coverage.json").read_text())
    new_coverage = json.loads((new / "tmab/coverage.json").read_text())
    summary = dict(
        hitlist_base=config["hitlist_base"], hitlist_version=hitlist.__version__,
        source_snapshot="reproduce/source/hitlist", new_package=str(new),
        old_manifest_sha256=config["old_manifest_sha256"], new_manifest_sha256=digest(manifest_path),
        n_source_rows=len(new_rows), n_measurement_slots=len(measurements),
        n_candidate_cross_paper_row_groups=len(groups), n_candidate_rows=sum(map(len, groups.values())),
        n_candidate_groups_with_equal_labels=sum(s == {"equal_after_case_whitespace_normalization"} for s in group_label_status.values()),
        n_changed_source_rows=len(changed_rows), n_changed_measurement_slots=len(changes),
        old_coverage=old_coverage, new_coverage=new_coverage,
        n_disagreements=len(failures), checks=dict(checks),
        seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        independent_specimens="unresolved", training_admission=False,
        training_admission_reason="Source candidates are not an independence-qualified native-response recipe.",
    )
    dump(RESULTS / "replay.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"checks", "old_coverage", "new_coverage"}}, indent=2), flush=True)
    assert not failures, "See preserved disagreements.json"
    assert summary["seconds"] < config["seconds_cap"] and summary["peak_rss_bytes"] < config["rss_bytes_cap"]


if __name__ == "__main__":
    main()
