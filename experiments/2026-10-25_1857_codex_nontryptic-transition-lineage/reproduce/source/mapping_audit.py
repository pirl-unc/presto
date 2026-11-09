"""Replay reported standard identities and fixed sums; never fit a source join."""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import openpyxl


def source_pattern(text):
    tokens = list(re.finditer(r"\(([A-Z])\)|([A-Z])", text))
    assert "".join(m[0] for m in tokens) == text, text
    return (
        "".join(m[1] or m[2] for m in tokens),
        [i for i, m in enumerate(tokens, 1) if m[1]],
    )


def export_pattern(text):
    tokens = list(re.finditer(r"([A-Z])(?:\[\+([0-9]+)\])?", text))
    assert "".join(m[0] for m in tokens) == text, text
    return (
        "".join(m[1] for m in tokens),
        [i for i, m in enumerate(tokens, 1) if m[2]],
        sum(int(m[2]) for m in tokens if m[2]),
    )


def literal(sheet, coordinate):
    cell = sheet.cells.get(coordinate)
    return cell.value_literal if cell else None


def checked_book(reader, path):
    sheets = reader.read_source_workbook(path)
    independent = openpyxl.load_workbook(path, data_only=True)
    assert [s.name for s in sheets] == independent.sheetnames
    for sheet in sheets:
        for coordinate, cell in sheet.cells.items():
            value = cell.value_literal
            expected = (
                float(value) if cell.excel_type == "n" and value is not None else value
            )
            assert independent[sheet.name][coordinate].value == expected, (
                path,
                coordinate,
            )
    independent.close()
    return {s.name: s for s in sheets}


def summarize_candidates(comparisons):
    result = []
    for sheet in ("DatasetS3A", "DatasetS3B"):
        cells = [c for c in comparisons if c["sheet"] == sheet]
        for peptide in [None, *sorted({c["sequence"] for c in cells})]:
            subset = [c for c in cells if peptide is None or c["sequence"] == peptide]
            for mode in ("product_only", "all_transitions"):
                ordinal = [
                    candidate
                    for cell in subset
                    for candidate in cell["candidates"]
                    if candidate["ordinal_correspondence_candidate_only"]
                    and candidate["mode"] == mode
                ]
                finite = [c for c in ordinal if c["difference"] is not None]
                result.append(
                    {
                        "sheet": sheet,
                        "peptide": peptide,
                        "mode": mode,
                        "source_slots": len(subset),
                        "missing_source_signals": sum(
                            c["raw_signal_literal"] is None for c in subset
                        ),
                        "slots_without_form_dose_candidates": sum(
                            not c["candidates"] for c in subset
                        ),
                        "ordinal_candidates": len(ordinal),
                        "ordinal_numeric_comparisons": len(finite),
                        "exact_equal": sum(c["exact_equal"] is True for c in ordinal),
                        "relative_1e_6_agreement": sum(
                            c["relative_1e_6_agreement"] is True for c in ordinal
                        ),
                        "within_conditional_integer_rounding_bound": sum(
                            c["within_conditional_integer_rounding_bound"] is True
                            for c in ordinal
                        ),
                        "max_absolute_difference": str(
                            max(
                                (abs(Decimal(c["difference"])) for c in finite),
                                default=Decimal(0),
                            )
                        )
                        if finite
                        else None,
                        "max_relative_difference": str(
                            max(
                                (Decimal(c["relative_difference"]) for c in finite),
                                default=Decimal(0),
                            )
                        )
                        if finite
                        else None,
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    request = json.loads(parser.parse_args().requests.read_text())
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "qualified_reader", request["shared_reader"]
    )
    reader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reader
    spec.loader.exec_module(reader)
    books = {Path(p).name: checked_book(reader, Path(p)) for p in request["workbooks"]}
    s6 = books["pnas.2111173118.sd06.xlsx"]["DatasetS6B"]
    assertions = []
    for coordinate, cell in s6.cells.items():
        if re.fullmatch(r"A[0-9]+", coordinate) and cell.value_literal in {
            "1H",
            "2H",
            "3H",
            "4H",
        }:
            row = int(coordinate[1:])
            sequence, positions = source_pattern(literal(s6, f"B{row}"))
            assertions.append(
                {
                    "sequence": sequence,
                    "isotopologue": cell.value_literal,
                    "reported_positions": positions,
                    "reported_nominal_shift": literal(s6, f"D{row}"),
                    "source_sheet": s6.name,
                    "source_cells": {
                        f"{col}{row}": literal(s6, f"{col}{row}") for col in "ABDE"
                    },
                }
            )
    prior = json.loads(Path(request["prior_lineage"]).read_text())
    groups = prior["groups"]
    form_mappings = []
    mapped = {}
    for form in sorted({g["reported_form"] for g in groups}):
        sequence, positions, nominal_shift = export_pattern(form)
        matches = [
            a
            for a in assertions
            if a["sequence"] == sequence
            and a["reported_positions"] == positions
            and Decimal(a["reported_nominal_shift"]) == nominal_shift
        ]
        entry = {
            "export_form": form,
            "sequence": sequence,
            "labeled_positions": positions,
            "nominal_shift": nominal_shift,
            "source_assertions": matches,
            "status": "unique_reported_correspondence"
            if len(matches) == 1
            else "unresolved",
            "canonical_isotope_atoms_resolved": False,
        }
        mapped[form] = entry
        form_mappings.append(entry)

    # Replay every area in both frozen aggregation modes from original source cells.
    for group in groups:
        (sheet,) = books[group["source_file"]].values()
        for mode in ("product_only", "all_transitions"):
            aggregate = group[mode]
            values = []
            states = Counter()
            for row in aggregate["source_rows"]:
                assert [literal(sheet, f"{col}{row}") for col in "ABC"] == [
                    group["peptide"],
                    group["reported_form"],
                    group["replicate_label"],
                ]
                if mode == "product_only":
                    assert literal(sheet, f"H{row}") != "precursor"
                raw = literal(sheet, f"J{row}")
                value = Decimal(raw) if raw not in (None, "") else None
                state = (
                    "missing"
                    if value is None
                    else "invalid"
                    if not value.is_finite() or value < 0
                    else "zero"
                    if value == 0
                    else "positive"
                )
                states[state] += 1
                values.append(value)
            complete = bool(values) and not (states["missing"] or states["invalid"])
            total = sum(values, Decimal(0)) if complete else None
            assert len(values) == aggregate["n_rows"]
            assert dict(states) == aggregate["area_states"]
            assert total == (
                Decimal(aggregate["sum"]) if aggregate["sum"] is not None else None
            )
            assert aggregate["all_areas_integer"] == (
                complete and all(v == v.to_integral_value() for v in values)
            )

    export_summaries = []
    for summary in prior["source_summaries"]:
        issues = summary["issues"]
        subset = [g for g in groups if g["source_file"] == summary["file"]]
        export_summaries.append(
            {
                "file": summary["file"],
                "n_transition_rows": summary["n_transition_rows"],
                "n_groups": len(subset),
                "n_peptides": len(summary["peptides"]),
                "n_reported_forms": len(summary["reported_forms"]),
                "replicate_labels": summary["replicate_labels"],
                "missing_concat": sum(i["reported"] in (None, "") for i in issues),
                "nonempty_concat_disagreements": sum(
                    i["reported"] not in (None, "") for i in issues
                ),
                "duplicate_identity_keys": summary["duplicate_identity_keys"],
                "product_row_count_distribution": dict(
                    Counter(g["product_only"]["n_rows"] for g in subset)
                ),
                "all_transition_count_distribution": dict(
                    Counter(g["all_transitions"]["n_rows"] for g in subset)
                ),
            }
        )

    comparisons = []
    s3_cells = json.loads(Path(request["S3_cells"]).read_text())
    for cell in s3_cells:
        sheet = books["pnas.2111173118.sd03.xlsx"][cell["sheet"]]
        raw = literal(sheet, cell["source_cell"])
        target = Decimal(raw) if raw is not None else None
        assert (float(raw) if raw is not None else None) == cell["raw_signal"]
        assert Decimal(literal(sheet, cell["dose_cell"])) == Decimal(
            str(cell["added_fmol"])
        )
        source_groups = []
        for group in groups:
            form = mapped[group["reported_form"]]
            if ("DirectSpike" in group["source_file"]) != (
                cell["sheet"] == "DatasetS3A"
            ):
                continue
            if form["status"] != "unique_reported_correspondence":
                continue
            assertion = form["source_assertions"][0]
            if (
                assertion["sequence"] != cell["sequence"]
                or assertion["isotopologue"] != cell["isotopologue"]
            ):
                continue
            if Decimal(group["nominal_fmol"]) == Decimal(str(cell["added_fmol"])):
                source_groups.append(group)
        candidates = []
        for group in source_groups:
            for mode in ("product_only", "all_transitions"):
                aggregate = group[mode]
                value = (
                    Decimal(aggregate["sum"]) if aggregate["sum"] is not None else None
                )
                difference = (
                    value - target if value is not None and target is not None else None
                )
                candidates.append(
                    {
                        "source_file": group["source_file"],
                        "reported_form": group["reported_form"],
                        "replicate_label": group["replicate_label"],
                        "mode": mode,
                        "ordinal_correspondence_candidate_only": group[
                            "replicate_suffix_ordinal"
                        ]
                        == cell["replicate_column_ordinal"],
                        "source_rows": aggregate["source_rows"],
                        "sum": aggregate["sum"],
                        "difference": str(difference)
                        if difference is not None
                        else None,
                        "relative_difference": str(abs(difference / target))
                        if difference is not None and target != 0
                        else None,
                        "exact_equal": difference == 0
                        if difference is not None
                        else None,
                        "relative_1e_6_agreement": abs(difference)
                        <= abs(target) * Decimal("0.000001")
                        if difference is not None
                        else None,
                        "within_conditional_integer_rounding_bound": abs(difference)
                        <= Decimal("0.5") * aggregate["n_rows"]
                        if difference is not None and aggregate["all_areas_integer"]
                        else None,
                    }
                )
        comparisons.append(
            {
                **cell,
                "raw_signal_literal": raw,
                "candidates": candidates,
                "acquisition_join_established": False,
            }
        )
    counts = summarize_candidates(comparisons)
    summary = {
        "status": "source_identity_and_sum_diagnostics_no_training_admission",
        "n_s6b_assertions": len(assertions),
        "n_unique_export_forms": len(form_mappings),
        "n_forms_with_unique_reported_correspondence": sum(
            m["status"] == "unique_reported_correspondence" for m in form_mappings
        ),
        "n_groups_replayed": len(groups),
        "source_summaries": export_summaries,
        "sheet_comparison_counts": [c for c in counts if c["peptide"] is None],
        "model_fits": 0,
        "old_biological_heldouts_accessed": False,
        "canonical_isotope_resolution": False,
        "acquisition_join_established": False,
    }
    for name, data in (
        ("mapping-audit-summary.json", summary),
        (
            "mapping-audit-details.json",
            {
                "reported_form_mappings": form_mappings,
                "S3_comparisons": comparisons,
                "comparison_counts": counts,
            },
        ),
    ):
        with (root / "results" / name).open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
