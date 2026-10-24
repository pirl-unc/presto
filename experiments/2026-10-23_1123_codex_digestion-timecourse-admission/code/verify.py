"""Rebuild admission from pinned source tables without importing Hitlist."""

import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ident(parts):
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def canon(s):
    return s.replace("I", "L")


def number(s):
    if s in ("", "NA", "NaN", "#N/A"):
        return None
    v = float(s)
    return v if math.isfinite(v) else None


def main():
    started = time.monotonic()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    out, package = Path(cfg["output_recipe"]), Path(cfg["package"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert sha(package / "manifest.json") == cfg["package_manifest_sha256"]
    for name, expected in manifest["files"].items():
        assert sha(out / name) == expected["sha256"], name
    source_manifest = json.loads((package / "manifest.json").read_text())

    def read(name, columns=None):
        relative = f"external/soh2024_digestion/typed/{name}.parquet"
        p = out / f"source_tables/{name}.parquet"
        assert (
            sha(p)
            == sha(package / relative)
            == source_manifest["files"][relative]["sha256"]
        )
        return pq.read_table(p, columns=columns).to_pylist()

    parents = {r["construct_id"]: r for r in read("constructs")}
    contexts = {r["condition_id"]: r for r in read("contexts")}
    final = read(
        "final_vectors",
        [
            "source_record_id",
            "condition_id",
            "construct_id",
            "peptide_sequence",
            "biological_replicate_raw",
            "times_raw",
            "intensities_raw",
            "product_type_raw",
            "splice_type_raw",
            "row_status",
        ],
    )
    # Independently construct the undirected source-sharing graph and BFS components.
    graph = {cid: set() for cid in parents}
    sharing = defaultdict(set)
    for cid, parent in parents.items():
        sharing["parent", parent["sequence"]].add(cid)
    for r in final:
        sharing["peptide", canon(r["peptide_sequence"])].add(r["construct_id"])
    for owners in sharing.values():
        for cid in owners:
            graph[cid].update(owners)
    remaining, components = set(parents), {}
    while remaining:
        visited, queue = set(), [min(remaining)]
        while queue:
            cid = queue.pop()
            if cid not in visited:
                visited.add(cid)
                queue.extend(graph[cid] - visited)
        remaining -= visited
        components[ident(sorted(visited))] = sorted(visited)
    assignments = {}
    for idx, (component, members) in enumerate(sorted(components.items())):
        for cid in members:
            fold = idx % 5
            assignments[cid] = dict(
                construct_id=cid,
                component_id=component,
                fold=fold,
                split={0: "test", 1: "val"}.get(fold, "train"),
            )
    actual_components = json.loads((out / "components.json").read_text())
    assert {r["construct_id"]: r for r in actual_components} == assignments
    raw_zeros, technical, technical_zeros, filtered_ids = [
        defaultdict(set) for _ in range(4)
    ]
    bad = {
        cid
        for cid, r in contexts.items()
        if r["training_status"] == "unresolved_source_time_conflict"
    }
    for r in read(
        "raw_quantities",
        [
            "source_record_id",
            "condition_id",
            "peptide_sequence",
            "biological_replicate_raw",
            "quantity_time_hours",
            "metadata_time_hours",
            "time_status",
            "intensity",
        ],
    ):
        key = (
            r["condition_id"],
            canon(r["peptide_sequence"]),
            r["biological_replicate_raw"],
        )
        if r["time_status"] != "source_agreement":
            bad.add(key[0])
        if (
            r["quantity_time_hours"] == 0
            and r["metadata_time_hours"] == 0
            and r["intensity"] is not None
        ):
            assert math.isfinite(r["intensity"])
            raw_zeros[key].add(r["source_record_id"])
    for r in read(
        "filtered_vectors",
        [
            "source_record_id",
            "condition_id",
            "peptide_sequence",
            "biological_replicate_raw",
            "technical_replicate_raw",
            "times_raw",
        ],
    ):
        key = (
            r["condition_id"],
            canon(r["peptide_sequence"]),
            r["biological_replicate_raw"],
        )
        technical[key].add(r["technical_replicate_raw"])
        if 0 in list(map(float, r["times_raw"].split(";"))):
            technical_zeros[key].add(r["technical_replicate_raw"])
        filtered_ids[key].add(r["source_record_id"])
    grouped = defaultdict(list)
    for r in final:
        grouped[
            r["condition_id"],
            canon(r["peptide_sequence"]),
            r["biological_replicate_raw"],
        ].append(r)
    del final
    ledger = {
        r["curve_id"]: r for r in json.loads((out / "dispositions.json").read_text())
    }
    examples = {
        r["curve_id"]: r for r in json.loads((out / "examples.json").read_text())
    }
    assert len(ledger) == len(grouped)
    candidates, reasons, target_count, slot_count = {}, {}, 0, 0
    for key, rows in grouped.items():
        curve_id, errors = ident(key), set()
        condition, peptide, bio = key
        cid = contexts[condition]["construct_id"]
        parent = parents[cid]["sequence"]
        spans = (
            [
                p
                for p in range(len(parent) - len(peptide) + 1)
                if canon(parent[p : p + len(peptide)]) == peptide
            ]
            if peptide
            else []
        )
        if not peptide or set(peptide + parent) - set("ACDEFGHIKLMNPQRSTVWY"):
            errors.add("unsupported_sequence")
        if any(r["row_status"] != "reported_vector" for r in rows):
            errors.add("annotation_only")
        if any(
            r["product_type_raw"] != "PCP"
            or r["splice_type_raw"] not in (None, "", "NA", "PCP")
            for r in rows
        ):
            errors.add("not_exclusively_source_contiguous")
        if len(spans) != 1:
            errors.add("not_one_IL_equivalent_contiguous_span")
        if condition in bad:
            errors.add("unresolved_condition_time_conflict")
        if (
            not raw_zeros[key]
            or not technical[key]
            or technical[key] - technical_zeros[key]
        ):
            errors.add("no_actual_zero_baseline_for_every_technical_label")
        by_time = defaultdict(list)
        for r in rows:
            if r["row_status"] != "annotation_only":
                for slot, (t, v) in enumerate(
                    zip(
                        r["times_raw"].split(";"),
                        r["intensities_raw"].split(";"),
                        strict=True,
                    )
                ):
                    by_time[float(t)].append(
                        dict(
                            value=number(v),
                            value_raw=v,
                            vector_id=r["source_record_id"],
                            slot_0based=slot,
                        )
                    )
        observations, slots = [], []
        for t, contributors in sorted(by_time.items()):
            v = contributors[0]["value"]
            consistent = all(
                (v is None and a["value"] is None)
                or (
                    v is not None
                    and a["value"] is not None
                    and abs(v - a["value"]) <= 1e-10 * max(1, abs(v), abs(a["value"]))
                )
                for a in contributors
            )
            if not consistent:
                errors.add("conflicting_IL_alias_or_repeated_slot_values")
            slots.append(
                dict(time_hours=t, contributors=contributors, consistent=consistent)
            )
            slot_count += len(contributors)
            if consistent and t > 0 and v is not None and v > 0:
                observations.append(
                    dict(time_hours=t, target_value=v, source_slots=contributors)
                )
        if len(observations) < 3:
            errors.add("fewer_than_three_positive_nonzero_times")
        record = dict(
            curve_id=curve_id,
            condition_id=condition,
            construct_id=cid,
            peptide_IL_sequence=peptide,
            peptide_aliases=sorted({r["peptide_sequence"] for r in rows}),
            biological_replicate_raw=bio,
            source_vector_ids=sorted(r["source_record_id"] for r in rows),
            candidate_start_positions_0based=spans,
            source_slots=slots,
            **{k: v for k, v in assignments[cid].items() if k != "construct_id"},
        )
        assert {
            k: v for k, v in ledger[curve_id].items() if k not in ("reasons", "status")
        } == record
        reasons[curve_id] = errors
        if not errors:
            candidates[curve_id] = (key, observations)
            if curve_id in examples:
                e = examples[curve_id]
                assert e["observations"] == observations
                assert e["actual_zero_raw_quantity_ids"] == sorted(raw_zeros[key])
                assert e["filtered_source_vector_ids"] == sorted(filtered_ids[key])
                assert e["parent_sequence"] == parent
                assert e["start_0based"] == spans[0] and e["end_0based"] == spans[
                    0
                ] + len(peptide)
                assert (
                    e["peptide_sequence"] == parent[e["start_0based"] : e["end_0based"]]
                )
                assert e["target_unit"] == "arbitrary_MS_signal"
                assert e["context"] == {k: contexts[condition][k] for k in e["context"]}
                for k, v in record.items():
                    if k != "source_slots":
                        assert e[k] == v
    pairs = defaultdict(list)
    for curve_id, (key, obs) in candidates.items():
        pairs[key[:2]].append((curve_id, {o["time_hours"] for o in obs}))
    admitted = set()
    for group in pairs.values():
        common = sorted(t for t in group[0][1] if all(t in times for _, times in group))
        for curve_id, _ in group:
            if len(group) >= 2 and len(common) >= 3:
                admitted.add(curve_id)
                assert examples[curve_id]["common_positive_times"] == common
                target_count += len(examples[curve_id]["observations"])
            else:
                reasons[curve_id].add("insufficient_repeated_common_positive_times")
    assert set(examples) == admitted
    for curve_id, errors in reasons.items():
        assert sorted(errors) == ledger[curve_id]["reasons"]
        assert ledger[curve_id]["status"] == ("excluded" if errors else "admitted")
    support = {}
    for split in ("train", "val", "test"):
        subset = [e for e in examples.values() if e["split"] == split]
        support[split] = dict(
            curves=len(subset),
            targets=sum(len(e["observations"]) for e in subset),
            parents=len({e["construct_id"] for e in subset}),
            conditions=len({e["condition_id"] for e in subset}),
            peptide_condition_pairs=len(
                {(e["condition_id"], e["peptide_IL_sequence"]) for e in subset}
            ),
            peptide_lengths=sorted({len(e["peptide_sequence"]) for e in subset}),
        )
    report = dict(
        passed=True,
        imported_producer=False,
        n_source_slots=slot_count,
        n_curves_checked=len(ledger),
        n_admitted=len(admitted),
        n_positive_targets=target_count,
        n_components=len(components),
        source_parquets_verified=5,
        partition_support=support,
        reason_counts=dict(
            sorted(Counter(e for errors in reasons.values() for e in errors).items())
        ),
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="Source integrity and eligibility only; no held-out response quality evaluation",
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
