"""Materialize shared evidence, then independently replay original source decisions."""

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import openpyxl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    request = json.loads(parser.parse_args().requests.read_text())
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, request["hitlist_checkout"])
    from hitlist.evidence_store import EvidenceStore
    from hitlist.nontryptic_calibration import write_nontryptic_calibration_package

    destination = Path(request["destination"])
    write_nontryptic_calibration_package(
        destination.parent,
        destination,
        assets=request["assets"],
    )
    store = EvidenceStore.open(destination)
    tables = {
        name.removeprefix("nontryptic_"): store.query(
            name, limit=None
        ).table.to_pylist()
        for name in store.bindings
    }
    by_key = lambda name, key: {row[key]: row for row in tables[name]}
    assets = by_key("assets", "source_asset_id")
    sheets = by_key("source_sheets", "sheet_id")
    standards = by_key("standards", "standard_id")
    amounts = by_key("nominal_amounts", "nominal_amount_id")
    slots = by_key("source_slots", "slot_id")
    forms = by_key("export_forms", "export_form_id")
    groups = by_key("transition_groups", "transition_group_id")
    source_books = {
        name: openpyxl.load_workbook(path, data_only=True)
        for name, path in request["assets"].items()
        if name.endswith(".xlsx")
    }
    n_source_cells = 0
    for cell in tables["source_cells"]:
        sheet = sheets[cell["sheet_id"]]
        filename = assets[sheet["source_asset_id"]]["source_file"]
        actual = source_books[filename][sheet["sheet_name"]][cell["cell_ref"]].value
        raw = cell["value_literal"]
        expected = float(raw) if cell["excel_type"] == "n" and raw is not None else raw
        assert actual == expected, (filename, sheet["sheet_name"], cell["cell_ref"])
        n_source_cells += 1

    # Compare all dose/sequence/slot decisions to the earlier independently
    # qualified source-cell ledger, not a newly selected training subset.
    old = json.loads(Path(request["qualified_S3_cells"]).read_text())
    old_by_cell = {(c["sheet"], c["source_cell"]): c for c in old}
    assert len(old_by_cell) == len(old) == len(tables["measurements"])
    seen = set()
    measurements_by_source = {}
    for measurement in tables["measurements"]:
        filename, address = measurement["source_locator"].split("#")
        sheet_name, coordinate = address.split("!")
        key = sheet_name, coordinate
        reference = old_by_cell[key]
        assert key not in seen
        seen.add(key)
        amount = amounts[measurement["nominal_amount_id"]]
        standard = standards[measurement["standard_id"]]
        slot = slots[measurement["slot_id"]]
        original = source_books[filename][sheet_name]
        assert (
            measurement["quantitative_value"]
            == original[coordinate].value
            == reference["raw_signal"]
        )
        assert measurement["value_status"] == reference["signal_status"]
        assert standard["peptide_sequence"] == reference["sequence"]
        assert amount["reported_form_label"] == reference["source_peptide_block"]
        assert (
            amount["nominal_amount_fmol"]
            == original[reference["dose_cell"]].value
            == reference["added_fmol"]
        )
        assert amount["isotopologue_label"] == reference["isotopologue"]
        assert slot["source_slot_ordinal"] == reference["replicate_column_ordinal"]
        assert slot["source_column"] == coordinate[0]
        assert (
            slot["acquisition_id"] is slot["specimen_id"] is slot["aliquot_id"] is None
        )
        assert slot["material_link_precision"] == "unresolved"
        measurements_by_source[key] = measurement["measurement_id"]
    assert seen == set(old_by_cell)

    # Reconcile each original transition row and both frozen aggregate definitions.
    for transition in tables["transitions"]:
        filename, address = transition["source_locator"].split("#")
        sheet_name, coordinates = address.split("!")
        row = int(coordinates.split(":")[0][1:])
        original = source_books[filename][sheet_name]
        group = groups[transition["transition_group_id"]]
        form = forms[group["export_form_id"]]
        assert original[f"A{row}"].value == form["peptide_sequence"]
        assert original[f"B{row}"].value == form["modified_sequence_raw"]
        assert original[f"C{row}"].value == group["source_run_label"]
        assert original[f"H{row}"].value == transition["fragment_ion_raw"]
        assert original[f"J{row}"].value == transition["quantitative_value"]
        assert original[f"L{row}"].value == transition["concat_raw"]
        assert transition["n_duplicate_identity_rows"] == 1
        assert group["acquisition_id"] is None

    def group_key(group):
        form = forms[group["export_form_id"]]
        return (
            assets[group["source_asset_id"]]["source_file"],
            form["modified_sequence_raw"],
            group["source_run_label"],
        )

    prior = json.loads(Path(request["qualified_transition_groups"]).read_text())
    prior_groups = {
        (g["source_file"], g["reported_form"], g["replicate_label"]): g
        for g in prior["groups"]
    }
    assert {group_key(g) for g in groups.values()} == set(prior_groups)
    for aggregate in tables["aggregate_signals"]:
        group = groups[aggregate["transition_group_id"]]
        reference = prior_groups[group_key(group)][aggregate["aggregation_mode"]]
        expected = Decimal(reference["sum"]) if reference["sum"] is not None else None
        actual = (
            Decimal(aggregate["quantitative_raw"])
            if aggregate["quantitative_raw"] is not None
            else None
        )
        assert expected == actual
        assert aggregate["n_transition_rows"] == reference["n_rows"]
        assert aggregate["n_missing_areas"] == reference["area_states"].get(
            "missing", 0
        )
        assert aggregate["n_invalid_areas"] == reference["area_states"].get(
            "invalid", 0
        )
        assert aggregate["n_zero_areas"] == reference["area_states"].get("zero", 0)

    mapping = json.loads(Path(request["qualified_form_mapping"]).read_text())
    source_forms = by_key("reported_forms", "form_id")
    prior_forms = {r["export_form"]: r for r in mapping["reported_form_mappings"]}
    for claim in tables["form_correspondences"]:
        assert claim["n_form_candidates"] == 1
        assert claim["correspondence_status"] == "unique_reported_correspondence"
        export = forms[claim["export_form_id"]]
        source = source_forms[claim["form_id"]]
        reference = prior_forms[export["modified_sequence_raw"]]["source_assertions"][0]
        assert source["peptide_sequence"] == reference["sequence"]
        assert source["isotopologue_label"] == reference["isotopologue"]
        assert (
            json.loads(source["labeled_positions_json"])
            == reference["reported_positions"]
        )
        assert Decimal(source["nominal_mass_shift_raw"]) == Decimal(
            reference["reported_nominal_shift"]
        )
        assert source["canonical_isotope_status"] == "unresolved_reported_claim_only"
    candidates = {}
    for claim in tables["measurement_candidates"]:
        assert claim["acquisition_join_established"] is False
        candidates.setdefault(claim["measurement_id"], set()).add(
            group_key(groups[claim["transition_group_id"]])
        )
    for cell in mapping["S3_comparisons"]:
        expected = {
            (c["source_file"], c["reported_form"], c["replicate_label"])
            for c in cell["candidates"]
        }
        assert (
            candidates[measurements_by_source[(cell["sheet"], cell["source_cell"])]]
            == expected
        )
    conflicts = dict(
        Counter(r["s6a_consistency_status"] for r in tables["reported_forms"])
    )
    assert conflicts == {
        "conflicting_source_assertions": 54,
        "within_0.15_Da": 17,
        "missing_S6A_residue_assertion": 1,
    }
    for book in source_books.values():
        book.close()
    summary = {
        "status": "shared_source_and_independent_replay_passed",
        "package_directory": str(destination),
        "package_schema_version": store.manifest["schema_version"],
        "catalog_schema_version": store.catalog()["schema_version"],
        "source_cells_independently_verified": n_source_cells,
        "reconciliation": store.manifest["reconciliation"],
        "source_signal_states": dict(
            Counter(r["value_status"] for r in tables["measurements"])
        ),
        "isotope_consistency": conflicts,
        "protocol_assertions": len(tables["protocol_assertions"]),
        "source_registry": store.catalog()["source_registry"],
        "inferred_acquisition_links": 0,
        "model_fits": 0,
        "old_biological_heldouts_accessed": False,
        "training_recipe_admitted": False,
    }
    with (root / "results/source-package-verification.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
