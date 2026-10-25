"""Independent canonical-table replay; no production query/curation helpers."""

from collections import Counter
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from common import ROOT, digest, read, write


def main():
    cfg = read(ROOT / "reproduce/launch.json")
    package = Path(cfg["package"])
    assert digest(package / "manifest.json") == cfg["package_manifest_sha256"]
    manifest = read(package / "manifest.json")
    checked = set()

    def table(name, field=None, values=None):
        relative = "dilution_observer/typed/" + name + ".parquet"
        if relative not in checked:
            assert digest(package / relative) == manifest["files"][relative]["sha256"]
            checked.add(relative)
        return pq.read_table(
            package / relative, filters=[(field, "in", values)] if field else None
        ).to_pylist()

    metadata = read(ROOT / "results/metadata.json")
    requests = read(ROOT / "results/queries.json")
    source_cells = 0
    for name, rows in metadata.items():
        where = requests[name].get("where", {})
        field = next(iter(where), None)
        actual = table(name, field, where[field]["value"] if field else None)

        def encode(row):
            return json.dumps(row, sort_keys=True)

        assert sorted(map(encode, rows)) == sorted(map(encode, actual)), name
        source_cells += sum(len(r) for r in rows)
    members = {
        r["library_precursor_id"]: r["standard_id"]
        for r in metadata["standard_memberships"]
        if r["library_precursor_id"] is not None and r["decoy_raw"] == "0"
    }
    assert len(members) == 9
    standards = {r["standard_id"]: r for r in metadata["nominal_standards"]}
    acqs = {r["acquisition_id"]: r for r in metadata["acquisitions"]}
    levels = {r["nominal_level_id"]: r for r in metadata["nominal_levels"]}
    aligned = {
        r["measurement_id"]: r
        for r in table("aligned_quantities", "library_precursor_id", list(members))
    }
    links = {
        r["measurement_id"]: r for r in table("feature_links", "measurement_id", list(aligned))
    }
    ids = [r["identification_id"] for r in links.values() if r["identification_id"]]
    scored = {r["identification_id"]: r for r in table("scored_features", "identification_id", ids)}
    decisions = read(ROOT / "results/decisions.json")
    admitted = {r["candidate_id"]: r for r in read(ROOT / "results/admitted.json")}
    assert len(aligned) == len(links) == len(decisions) == 90
    assert set(aligned) == {r["measurement_id"] for r in decisions}
    expected_accepted = set()
    numeric_checks = 0
    for row in decisions:
        link = links[row["measurement_id"]]
        quantity = aligned[row["measurement_id"]]
        score = scored.get(link["identification_id"])
        assert row["candidate_id"] == "dilution_feature_links:" + link["claim_id"]
        for key, value in link.items():
            assert row[key] == value, key
        for prefix, original in (("aligned", quantity), ("scored", score)):
            for key, value in row.items():
                if key.startswith(prefix + "."):
                    assert value == (original[key.split(".", 1)[1]] if original else None), key
        positive = quantity["quantitative_value"] is not None and quantity["quantitative_value"] > 0
        exact = (
            score is not None
            and link["n_candidates"] == 1
            and all(
                link[k]
                for k in ("form_agrees", "charge_agrees", "precursor_agrees", "target_agrees")
            )
        )
        expected = exact and positive
        assert (row["decision"] == "accepted") == expected
        if not expected:
            assert row["target_value"] is None and row["reason_codes"]
            continue
        expected_accepted.add(row["candidate_id"])
        assert row["target_value"] == quantity["quantitative_value"] == score["quantitative_value"]
        assert quantity["native_feature_id"] == score["native_feature_id"]
        acq = acqs[quantity["acquisition_id"]]
        level = levels[acq["nominal_level_id"]]
        standard_id = members[quantity["library_precursor_id"]]
        assert (
            standards[standard_id]["expected_modified_sequence"]
            == quantity["modified_sequence_raw"]
        )
        item = admitted[row["candidate_id"]]
        assert item["standard_id"] == standard_id and item["method"] == str(
            acq["dia_index_from_name"]
        )
        assert item["nominal_level_id"] == acq["nominal_level_id"]
        assert item["nominal_unit"] == level["measurement_unit"]
        assert json.loads(item["group"]) == [
            standard_id,
            quantity["modified_sequence_raw"],
            quantity["charge_raw"],
            acq["dia_index_from_name"],
        ]
        assert abs(item["y"] - math.log10(float(quantity["quantitative_raw"]))) < 1e-12
        maximum = max(float(r["nominal_concentration_raw"]) for r in levels.values())
        assert (
            abs(item["x"] - math.log10(float(level["nominal_concentration_raw"]) / maximum)) < 1e-12
        )
        assert item["nominal_concentration"] == float(level["nominal_concentration_raw"])
        numeric_checks += 3
    assert set(admitted) == expected_accepted
    admission = read(ROOT / "results/admission.json")
    assert admission["n_admitted"] == len(admitted)
    assert admission["decisions"] == dict(Counter(r["decision"] for r in decisions))
    assert admission["feature_links"] == dict(Counter(r["feature_link_status"] for r in decisions))
    assert admission["value_status"] == dict(Counter(r["aligned.value_status"] for r in decisions))
    for filename, expected in admission["files"].items():
        assert digest(ROOT / "results" / filename) == expected
    result = {
        "passed": True,
        "canonical_tables_verified": len(checked),
        "metadata_cells_compared": source_cells,
        "standard_slots_replayed": 90,
        "accepted_ids_independently_reconstructed": len(admitted),
        "numeric_transform_checks": numeric_checks,
        "admission_sha256": digest(ROOT / "results/admission.json"),
    }
    write(ROOT / "results/admission-verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
