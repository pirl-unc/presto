"""Apply a fixed generic Hitlist recipe; preserve every standard-slot decision."""

from collections import Counter
import json
import math
import sys

from common import ROOT, digest, read, write


def derive(row, member, standard, acquisition, level):
    value = row["target_value"]
    dose = level["nominal_concentration"]
    if row["decision"] != "accepted" or value is None or not math.isfinite(value) or value <= 0:
        raise ValueError("Only accepted finite positive values can be transformed")
    if dose is None or not math.isfinite(dose) or dose <= 0:
        raise ValueError("Invalid nominal dose")
    if row["target_unit"] != "source_signal_arbitrary_units":
        raise ValueError("Unsupported signal unit")
    group = json.dumps(
        [
            member["standard_id"],
            row["aligned.modified_sequence_raw"],
            row["aligned.charge_raw"],
            acquisition["dia_index_from_name"],
        ],
        separators=(",", ":"),
    )
    return {
        **row,
        "group": group,
        "standard_id": member["standard_id"],
        "sequence": standard["peptide_sequence"],
        "method": str(acquisition["dia_index_from_name"]),
        "nominal_level_id": level["nominal_level_id"],
        "nominal_concentration": dose,
        "nominal_concentration_raw": level["nominal_concentration_raw"],
        "nominal_unit": level["measurement_unit"],
        "nominal_status": level["nominal_status"],
        "method_link_precision": acquisition["method_link_precision"],
        "y": math.log10(value),
    }


def main():
    cfg = read(ROOT / "reproduce/launch.json")
    sys.path.insert(0, cfg["hitlist_checkout"])
    import pyarrow.parquet as pq
    from hitlist.dilution_observer_schema import SCHEMAS
    from hitlist.evidence_store import EvidenceStore

    store = EvidenceStore.open(cfg["package"])
    requests, metadata = {}, {}
    for name in (
        "nominal_standards",
        "standard_memberships",
        "nominal_levels",
        "acquisitions",
        "material_contexts",
        "material_links",
        "assertions",
    ):
        requests[name] = {
            "evidence_type": "dilution_" + name,
            "select": SCHEMAS[name].names,
            "limit": None,
        }
        metadata[name] = store.query(**requests[name]).table.to_pylist()
    standards = {r["standard_id"]: r for r in metadata["nominal_standards"]}
    members = {
        r["library_precursor_id"]: r
        for r in metadata["standard_memberships"]
        if r["membership_status"] == "target"
    }
    assert len(standards) == len(metadata["nominal_standards"]) == 10
    assert len(members) == 9 and len(metadata["standard_memberships"]) == 10
    assert len({m["standard_id"] for m in members.values()}) == 9
    assert (
        sum(
            r["membership_status"] == "absent_from_library"
            for r in metadata["standard_memberships"]
        )
        == 1
    )
    acquisitions = {r["acquisition_id"]: r for r in metadata["acquisitions"]}
    levels = {r["nominal_level_id"]: r for r in metadata["nominal_levels"]}
    assert len(acquisitions) == 10 and len(levels) == 5
    assert {a["dia_index_from_name"] for a in acquisitions.values()} == {1, 2}
    assert (
        len({(a["nominal_level_id"], a["dia_index_from_name"]) for a in acquisitions.values()})
        == 10
    )
    assert {a["nominal_level_id"] for a in acquisitions.values()} == set(levels)
    assert len({r["measurement_unit"] for r in levels.values()}) == 1
    assert {r["nominal_status"] for r in levels.values()} == {
        "reported_nominal_solution_not_injected_amount"
    }
    assert all(
        math.isfinite(r["nominal_concentration"]) and r["nominal_concentration"] > 0
        for r in levels.values()
    )
    assert len({r["nominal_concentration"] for r in levels.values()}) == 5
    requests["library_precursors"] = {
        "evidence_type": "dilution_library_precursors",
        "select": SCHEMAS["library_precursors"].names,
        "where": {"library_precursor_id": {"op": "in", "value": sorted(members)}},
        "limit": None,
    }
    metadata["library_precursors"] = store.query(**requests["library_precursors"]).table.to_pylist()
    assert len(metadata["library_precursors"]) == len(members)
    for r in metadata["library_precursors"]:
        m = members[r["library_precursor_id"]]
        assert r["decoy_raw"] == m["decoy_raw"] == "0"
        assert (
            r["modified_sequence_raw"] == standards[m["standard_id"]]["expected_modified_sequence"]
        )
    recipe = read(ROOT / "reproduce/recipe-template.json")
    source = recipe["sources"][0]
    source["where"] = {"aligned.library_precursor_id": {"op": "in", "value": sorted(members)}}
    write(ROOT / "results/recipe-resolved.json", recipe)
    manifest_path = store.materialize(recipe, cfg["recipe_destination"])
    bundle = manifest_path.parent
    decisions = [
        row
        for decision in ("accepted", "excluded", "ambiguous")
        for row in pq.read_table(
            bundle / "sources/dilution_feature_links" / (decision + ".parquet")
        ).to_pylist()
    ]
    decisions.sort(key=lambda r: r["candidate_id"])
    assert len(decisions) == len({r["candidate_id"] for r in decisions}) == 90
    assert (
        len({(r["aligned.library_precursor_id"], r["aligned.acquisition_id"]) for r in decisions})
        == 90
    )
    admitted = []
    maximum = max(r["nominal_concentration"] for r in levels.values())
    for row in decisions:
        member = members[row["aligned.library_precursor_id"]]
        acq = acquisitions[row["aligned.acquisition_id"]]
        if row["decision"] != "accepted":
            continue
        assert row["feature_link_status"] == "matched_identity" and row["n_candidates"] == 1
        assert all(
            row[k] for k in ("form_agrees", "charge_agrees", "precursor_agrees", "target_agrees")
        )
        assert row["aligned.quantitative_value"] == row["scored.quantitative_value"]
        assert row["aligned.acquisition_id"] == row["scored.acquisition_id"]
        assert row["aligned.library_precursor_id"] == row["scored.library_precursor_id"]
        assert row["aligned.decoy_raw"] == row["scored.decoy_raw"] == "0"
        item = derive(
            row, member, standards[member["standard_id"]], acq, levels[acq["nominal_level_id"]]
        )
        item["x"] = math.log10(item["nominal_concentration"] / maximum)
        admitted.append(item)
    assert len({r["group"] for r in admitted}) == 18
    for level in levels:
        train = [r for r in admitted if r["nominal_level_id"] != level]
        assert {r["group"] for r in train} == {r["group"] for r in admitted}
        assert all(
            len({r["x"] for r in train if r["group"] == g}) >= 2
            for g in {r["group"] for r in admitted}
        )
    source_ids = {
        r["source_record_id"]
        for rows in metadata.values()
        for r in rows
        if r.get("source_record_id")
    }
    for row in decisions:
        source_ids.update(
            row[k]
            for k in ("source_record_id", "aligned.source_record_id", "scored.source_record_id")
            if row.get(k)
        )
    requests["source_records"] = {
        "evidence_type": "dilution_source_records",
        "select": SCHEMAS["source_records"].names,
        "where": {"source_record_id": {"op": "in", "value": sorted(source_ids)}},
        "limit": None,
    }
    metadata["source_records"] = store.query(**requests["source_records"]).table.to_pylist()
    assert len(metadata["source_records"]) == len(source_ids)
    write(ROOT / "results/metadata.json", metadata)
    write(ROOT / "results/queries.json", requests)
    write(ROOT / "results/decisions.json", decisions)
    write(ROOT / "results/admitted.json", admitted)
    result = {
        "passed": True,
        "n_intended_standards": 10,
        "n_library_standards": 9,
        "n_candidates": len(decisions),
        "n_admitted": len(admitted),
        "n_groups": 18,
        "decisions": dict(Counter(r["decision"] for r in decisions)),
        "feature_links": dict(Counter(r["feature_link_status"] for r in decisions)),
        "value_status": dict(Counter(r["aligned.value_status"] for r in decisions)),
        "recipe_manifest": str(manifest_path),
        "recipe_manifest_sha256": digest(manifest_path),
        "files": {
            name: digest(ROOT / "results" / name)
            for name in (
                "metadata.json",
                "queries.json",
                "decisions.json",
                "admitted.json",
                "recipe-resolved.json",
            )
        },
    }
    write(ROOT / "results/admission.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
