"""Source-grain and aggregation diagnostics; no fitted model or inferred run join."""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import openpyxl


def number(value):
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def amount(label, direct):
    pattern = r"(0\.1|1|10|100)_([123])" if direct else r"([0-9.]+)(amol|fmol)_([123])"
    match = re.fullmatch(pattern, label)
    if match is None:
        return None, None
    if direct:
        return Decimal(match[1]), int(match[2])
    value = number(match[1])
    if value is None:
        return None, None
    return value * (Decimal("0.001") if match[2] == "amol" else 1), int(match[3])


def summarize(rows):
    counts = Counter(row["area_state"] for row in rows)
    complete = bool(rows) and not (counts["missing"] or counts["invalid"])
    values = [number(row["area"]) for row in rows]
    total = sum(values, Decimal(0)) if complete else None
    return {
        "n_rows": len(rows),
        "area_states": dict(counts),
        "sum": str(total) if total is not None else None,
        "all_areas_integer": complete
        and all(v == v.to_integral_value() for v in values),
        "source_rows": [row["source_row"] for row in rows],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    request = json.loads(parser.parse_args().requests.read_text())
    root = Path(__file__).resolve().parents[1]
    scope = json.loads((root / "reproduce/requests/lineage-scope-1.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "verified_source_reader", request["shared_reader"]
    )
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)
    summaries, all_groups = [], []
    for path_text in request["workbooks"]:
        path = Path(path_text)
        direct = "DirectSpike" in path.name
        (sheet,) = source.read_source_workbook(path)
        book = openpyxl.load_workbook(path, data_only=True)
        for coordinate, cell in sheet.cells.items():
            literal = cell.value_literal
            expected = (
                float(literal)
                if cell.excel_type == "n" and literal is not None
                else literal
            )
            assert book[sheet.name][coordinate].value == expected, (
                path.name,
                coordinate,
            )
        book.close()
        cells = sheet.cells

        def value(column, row, cells=cells):
            cell = cells.get(f"{column}{row}")
            return cell.value_literal if cell is not None else None

        assert [value(col, 1) for col in "ABCDEFGHIJKL"] == scope["headers_A_to_L"]
        legend = [
            cell.value_literal
            for pos, cell in cells.items()
            if pos.startswith("N") and pos != "N1" and cell.value_literal
        ]
        if direct:
            assert value("N", 1) == "Replicate "
            assert legend == scope["direct_dose_legend"], legend
        row_numbers = sorted(
            {
                int(re.search(r"[0-9]+", pos)[0])
                for pos in cells
                if re.match(r"^[A-L][0-9]+$", pos)
            }
            - {1}
        )
        grouped, identities, issues = defaultdict(list), Counter(), []
        for row in row_numbers:
            values = {col: value(col, row) for col in "ABCDEFGHIJKL"}
            if all(v is None for v in values.values()):
                continue
            assert all(values[col] for col in "ABC"), (
                path.name,
                row,
                "missing identity",
            )
            assert re.sub(r"\[\+[0-9]+\]", "", values["B"]) == values["A"], (
                path.name,
                row,
                "form mismatch",
            )
            identity = tuple(values[col] for col in "ABCDEFGH")
            identities[identity] += 1
            area = number(values["J"])
            state = (
                "missing"
                if values["J"] in (None, "")
                else "invalid"
                if area is None or area < 0
                else "zero"
                if area == 0
                else "positive"
            )
            if values["L"] != values["H"] + "_" + values["G"]:
                issues.append(
                    {"row": row, "kind": "concat_disagreement", "reported": values["L"]}
                )
            grouped[(values["A"], values["B"], values["C"])].append(
                {
                    "source_row": row,
                    "area": values["J"],
                    "area_state": state,
                    "fragment": values["H"],
                    "precursor_charge": values["E"],
                }
            )
        groups = []
        for (peptide, form, replicate), rows in sorted(grouped.items()):
            dose, ordinal = amount(replicate, direct)
            products = [row for row in rows if row["fragment"] != "precursor"]
            groups.append(
                {
                    "source_file": path.name,
                    "source_sheet": sheet.name,
                    "peptide": peptide,
                    "reported_form": form,
                    "replicate_label": replicate,
                    "nominal_fmol": str(dose) if dose is not None else None,
                    "replicate_suffix_ordinal": ordinal,
                    "peptide_ions": dict(Counter(row["fragment"] for row in rows)),
                    "precursor_charges": sorted(
                        {row["precursor_charge"] for row in rows}
                    ),
                    "product_only": summarize(products),
                    "all_transitions": summarize(rows),
                }
            )
        summaries.append(
            {
                "file": path.name,
                "shared_reader_cells_verified": len(cells),
                "n_transition_rows": sum(len(rows) for rows in grouped.values()),
                "n_groups": len(groups),
                "peptides": sorted({key[0] for key in grouped}),
                "reported_forms": sorted({key[1] for key in grouped}),
                "replicate_labels": sorted({key[2] for key in grouped}),
                "duplicate_identity_keys": [
                    {"key": list(key), "n_rows": n}
                    for key, n in identities.items()
                    if n > 1
                ],
                "issues": issues,
                "dose_legend": legend,
            }
        )
        all_groups.extend(groups)
    source_cells = json.loads(Path(request["S3_cells"]).read_text())
    comparisons = []
    for cell in source_cells:
        direct = cell["sheet"] == "DatasetS3A"
        groups = [
            g
            for g in all_groups
            if ("DirectSpike" in g["source_file"]) == direct
            and g["reported_form"] == cell["source_peptide_block"]
            and number(g["nominal_fmol"]) == Decimal(str(cell["added_fmol"]))
        ]
        target = (
            number(str(cell["raw_signal"])) if cell["raw_signal"] is not None else None
        )
        candidates = []
        for group in groups:
            for mode in ("product_only", "all_transitions"):
                aggregate = group[mode]
                predicted = number(aggregate["sum"])
                difference = (
                    predicted - target
                    if predicted is not None and target is not None
                    else None
                )
                candidates.append(
                    {
                        "replicate_label": group["replicate_label"],
                        "mode": mode,
                        "ordinal_correspondence_candidate_only": group[
                            "replicate_suffix_ordinal"
                        ]
                        == cell["replicate_column_ordinal"],
                        "sum": aggregate["sum"],
                        "difference": str(difference)
                        if difference is not None
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
                        "n_source_rows": aggregate["n_rows"],
                    }
                )
        comparisons.append(
            {
                "sheet": cell["sheet"],
                "cell": cell["source_cell"],
                "reported_form": cell["source_peptide_block"],
                "raw_signal": cell["raw_signal"],
                "candidates": candidates,
                "acquisition_join_established": False,
            }
        )
    result = {
        "status": "source_diagnostics_only_semantic_admission_pending",
        "source_summaries": summaries,
        "groups": all_groups,
        "S3_comparisons": comparisons,
        "model_fits": 0,
    }
    with (root / "results/transition-lineage.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_groups": len(all_groups),
                "n_source_cells_compared": len(comparisons),
                "source_summaries": summaries,
            }
        )
    )


if __name__ == "__main__":
    main()
