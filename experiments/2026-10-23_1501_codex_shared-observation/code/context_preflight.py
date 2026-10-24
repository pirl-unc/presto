"""Canonical acquisition queries and conservative aggregate applicability audit."""

import gzip
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from hitlist.evidence_store import EvidenceStore

FAMILY = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main():
    started = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    stage = json.loads((FAMILY / "reproduce/preflight.json").read_text())
    assert sha(__file__) == stage["source_sha256"]
    recipe, package = Path(cfg["recipe_directory"]), Path(cfg["source_package"])
    assert sha(recipe / "manifest.json") == cfg["recipe_manifest_sha256"]
    assert sha(package / "manifest.json") == cfg["source_package_manifest_sha256"]
    assert (recipe / "source_manifest.json").read_bytes() == (
        package / "manifest.json"
    ).read_bytes()
    assert not (FAMILY / "results/context-summary.json").exists()
    manifest = json.loads((recipe / "manifest.json").read_text())
    assert sha(recipe / "examples.json") == manifest["files"]["examples.json"]["sha256"]
    # Test labels are never added to the selected observation collection.
    examples = [
        r
        for r in json.loads((recipe / "examples.json").read_text())
        if r["split"] in cfg["allowed_splits"]
    ]
    assert len([r for r in examples if r["split"] == "train"]) == 23054
    assert (
        sum(len(r["common_positive_times"]) for r in examples if r["split"] == "train")
        == 90630
    )
    conditions = sorted({r["condition_id"] for r in examples})
    where = {"condition_id": {"op": "in", "value": conditions}}
    store = EvidenceStore.open(package)
    query = dict(
        evidence_type="digestion_raw_quantities",
        select=[
            "source_record_id",
            "condition_id",
            "peptide_sequence",
            "digestion_biological_replicate",
            "digestion_quantity_time",
            "digestion_metadata_time",
            "digestion_time_status",
            "digestion_run_id",
            "quantitative_value",
            "native.assignments_raw",
            "acquisition.instrument",
            "acquisition.source_locator",
        ],
        joins=["acquisition"],
        where=where,
        limit=None,
    )
    result = store.query(**query)
    pq.write_table(
        result.table, FAMILY / "results/raw-quantity-acquisition-query.parquet"
    )
    raw = result.table.to_pylist()
    grouped = defaultdict(list)
    for r in raw:
        grouped[
            r["condition_id"],
            r["peptide_sequence"].replace("I", "L"),
            r["digestion_biological_replicate"],
            r["digestion_quantity_time"],
        ].append(r)
    wanted = set()
    for e in examples:
        for t in e["common_positive_times"]:
            key = (
                e["condition_id"],
                e["peptide_IL_sequence"],
                e["biological_replicate_raw"],
                t,
            )
            for r in grouped[key]:
                wanted.add(
                    (
                        r["condition_id"],
                        r["peptide_sequence"].replace("I", "L"),
                        r["digestion_run_id"],
                    )
                )
    msquery = dict(
        evidence_type="digestion_ms1",
        select=[
            "condition_id",
            "peptide_sequence",
            "digestion_run_id",
            "digestion_modified_sequence_raw",
            "native.precursor_charge_raw",
            "native.assignment_id_raw",
            "digestion_run_link_status",
        ],
        where=where,
        limit=None,
    )
    forms = defaultdict(
        lambda: dict(
            forms=set(), charges=set(), assignments=set(), statuses=set(), rows=0
        )
    )
    scanned = matched = 0
    for result in store.iter_query(**msquery):
        for r in result.table.to_pylist():
            scanned += 1
            key = (
                r["condition_id"],
                r["peptide_sequence"].replace("I", "L"),
                r["digestion_run_id"],
            )
            if key not in wanted:
                continue
            matched += 1
            info = forms[key]
            info["forms"].add(r["digestion_modified_sequence_raw"])
            info["charges"].add(r["native.precursor_charge_raw"])
            info["assignments"].add(r["native.assignment_id_raw"])
            info["statuses"].add(r["digestion_run_link_status"])
            info["rows"] += 1
    counters = Counter()
    per_condition = defaultdict(Counter)
    assignment_examples = []
    with gzip.open(FAMILY / "results/observation-context.jsonl.gz", "wt") as stream:
        for e in examples:
            for t in e["common_positive_times"]:
                selected = next(o for o in e["observations"] if o["time_hours"] == t)
                rows = grouped[
                    e["condition_id"],
                    e["peptide_IL_sequence"],
                    e["biological_replicate_raw"],
                    t,
                ]
                instruments = sorted(
                    {
                        r["acquisition.instrument"]
                        for r in rows
                        if r["acquisition.instrument"]
                    }
                )
                unknown_instrument = any(not r["acquisition.instrument"] for r in rows)
                assert all(
                    r["digestion_time_status"] == "source_agreement" for r in rows
                )
                status = (
                    "unambiguous_candidate_instrument"
                    if len(instruments) == 1 and not unknown_instrument
                    else "mixed_or_unknown_candidate_instrument"
                )
                if not rows:
                    status = "no_candidate_raw_quantity"
                fm, ch, st = set(), set(), set()
                missing_forms = 0
                for r in rows:
                    key = (
                        r["condition_id"],
                        r["peptide_sequence"].replace("I", "L"),
                        r["digestion_run_id"],
                    )
                    info = forms.get(key)
                    if info is None:
                        missing_forms += 1
                    else:
                        fm.update(info["forms"])
                        ch.update(info["charges"])
                        st.update(info["statuses"])
                    if (
                        len(assignment_examples) < 5
                        and r["native.assignments_raw"] not in assignment_examples
                    ):
                        assignment_examples.append(r["native.assignments_raw"])
                record = dict(
                    curve_id=e["curve_id"],
                    condition_id=e["condition_id"],
                    split=e["split"],
                    time=t,
                    source_target=selected["target_value"],
                    source_slots=selected["source_slots"],
                    candidate_raw_quantity_ids=sorted(
                        r["source_record_id"] for r in rows
                    ),
                    candidate_run_ids=sorted({r["digestion_run_id"] for r in rows}),
                    candidate_instruments=instruments,
                    instrument_status=status,
                    instrument=instruments[0]
                    if status == "unambiguous_candidate_instrument"
                    else None,
                    candidate_modified_sequences=sorted(fm, key=lambda x: str(x)),
                    candidate_charge_labels=sorted(ch, key=lambda x: str(x)),
                    candidate_form_link_statuses=sorted(st, key=lambda x: str(x)),
                    candidate_quantities_without_ms1_forms=missing_forms,
                    contributor_scope="Condition/IL-peptide/source-biological-label/time raw candidate superset; exact retained technical assignment is unresolved. Forms are candidates, not certified aggregate-form contributions.",
                )
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                counters[(e["split"], status)] += 1
                per_condition[e["condition_id"]][status] += 1
                per_condition[e["condition_id"]]["n_observations"] += 1
                per_condition[e["condition_id"]]["multiple_candidate_forms"] += (
                    len(fm) > 1
                )
                per_condition[e["condition_id"]]["missing_ms1_form_candidates"] += (
                    missing_forms > 0
                )
                per_condition[e["condition_id"]][
                    "instrument:" + str(record["instrument"])
                ] += 1
    receipt = dict(
        query=query,
        ms1_query=msquery,
        source_manifest=store._manifest_hash(),
        acquisition_relationships=result.joined_relationships
        if result.evidence_type == "digestion_raw_quantities"
        else [],
        n_raw_quantity_rows=len(raw),
        n_ms1_rows_scanned=scanned,
        n_ms1_rows_matched=matched,
        n_source_curves=len(examples),
        counts={str(k): v for k, v in counters.items()},
        per_condition={k: dict(v) for k, v in per_condition.items()},
        assignment_raw_examples=assignment_examples,
        instrument_scope="All candidate source runs agree is sufficient for the retained subset; exact retained technical pairing remains unresolved",
        form_scope="Conservative per-run/IL-peptide candidate forms; no raw-assignment-to-retained-form equivalence asserted",
        seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    dump(FAMILY / "results/context-summary.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
