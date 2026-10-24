"""Independently decode original workbook cells and inventory candidate quantity blocks."""

import hashlib
import json
import math
import posixpath
import re
import resource
import shutil
import time
import zipfile
from collections import Counter
from datetime import timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/lymph_source_qualification/2026-10-24_1829_v1"
MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
BLOCKS = {
    "Suppl Table S6 (a)": [("human_native", "B", ["K"])],
    "Suppl Table S4(e)": [
        ("dc_dia_database", "B", list("MNOPQR")),
        ("dc_dia_library", "AL", ["AW", "AX", "AY", "AZ", "BA", "BB"]),
    ],
}


def col_index(column):
    value = 0
    for letter in column.upper():
        value = value * 26 + ord(letter) - 64
    return value


def column_name(value):
    assert value > 0
    output = ""
    while value:
        value, rem = divmod(value - 1, 26)
        output = chr(65 + rem) + output
    return output


def translated_formula(formula, anchor, coordinate):
    def position(value):
        match = re.fullmatch(r"([A-Z]+)([0-9]+)", value)
        assert match
        return col_index(match.group(1)), int(match.group(2))

    ac, ar = position(anchor)
    cc, cr = position(coordinate)
    pattern = re.compile(r"(?<![A-Za-z0-9_])([$]?)([A-Za-z]{1,3})([$]?)([1-9][0-9]*)")

    def shift(match):
        abs_c, col, abs_r, row = match.groups()
        col = column_name(col_index(col) + (0 if abs_c else cc - ac))
        row = int(row) + (0 if abs_r else cr - ar)
        assert row > 0
        return abs_c + col + abs_r + str(row)

    # Preserve quoted Excel literals; only translate unquoted A1 references.
    pieces = re.split("(\"(?:[^\"]|\"\")*\"|'(?:[^']|'')*')", formula)
    return "".join(
        piece if piece.startswith(('"', "'")) else pattern.sub(shift, piece)
        for piece in pieces
    )


def decode_rows(archive, path, strings, formats, formatted_cells):
    shared = {}
    with archive.open(path) as handle:
        for _, node in ET.iterparse(handle, events=["end"]):
            if node.tag != MAIN + "row":
                continue
            cells = {}
            for cell in node.findall(MAIN + "c"):
                coordinate, kind = cell.get("r"), cell.get("t")
                formula_node = cell.find(MAIN + "f")
                formula = None if formula_node is None else formula_node.text or ""
                if formula_node is not None and formula_node.get("t") == "shared":
                    key = formula_node.get("si")
                    if formula:
                        shared[key] = (coordinate, formula)
                    else:
                        anchor, formula = shared[key]
                        formula = translated_formula(formula, anchor, coordinate)
                value = cell.findtext(MAIN + "v")
                if formula is not None:
                    value = "=" + formula
                elif kind == "s":
                    value = strings[int(value)] if value else None
                elif kind == "inlineStr":
                    value = "".join(p.text or "" for p in cell.iter(MAIN + "t"))
                elif kind == "b" and value is not None:
                    value = bool(int(value))
                elif kind not in ("str", "e") and value not in (None, ""):
                    value = float(value)
                fmt = formats.get(int(cell.get("s", "0")), "")
                if isinstance(value, float) and re.search(
                    r"\[(?:h+|m+|s+)\]", fmt, re.IGNORECASE
                ):
                    formatted_cells.append(
                        {
                            "sheet_path": path,
                            "coordinate": coordinate,
                            "raw_numeric_value": value,
                            "number_format": fmt,
                        }
                    )
                    value = str(timedelta(milliseconds=round(value * 86400000)))
                if value is not None:
                    cells[coordinate] = value
            yield int(node.get("r")), cells
            node.clear()


def main():
    started = time.perf_counter()
    destination = ROOT / "results/independent-verification.json"
    assert not destination.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/verify_workbooks.py")
    saved = json.loads((ROOT / "results/workbook-inventory.json").read_text())
    for item in saved["verified_assets"]:
        data = (RAW / item["name"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert hashlib.md5(data, usedforsecurity=False).hexdigest() == item["md5"]
    cell_count = sheet_count = 0
    formatted_cells = []
    quantity_blocks = []
    for book in saved["workbooks"]:
        with zipfile.ZipFile(RAW / book["filename"]) as archive:
            assert archive.testzip() is None
            styles = ET.fromstring(archive.read("xl/styles.xml"))
            num_formats = {46: "[h]:mm:ss"}
            num_formats.update(
                {
                    int(n.get("numFmtId")): n.get("formatCode")
                    for n in styles.findall(MAIN + "numFmts/" + MAIN + "numFmt")
                }
            )
            formats = {
                i: num_formats.get(int(n.get("numFmtId")), "")
                for i, n in enumerate(styles.find(MAIN + "cellXfs"))
            }
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml")):
                    strings.append("".join(p.text or "" for p in item.iter(MAIN + "t")))
            relationships = {
                n.get("Id"): n.get("Target")
                for n in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            }
            sheet_nodes = ET.fromstring(archive.read("xl/workbook.xml")).findall(
                MAIN + "sheets/" + MAIN + "sheet"
            )
            assert [n.get("name") for n in sheet_nodes] == [
                s["name"] for s in book["sheets"]
            ]
            for node, expected in zip(sheet_nodes, book["sheets"], strict=True):
                target = relationships[node.get(REL + "id")]
                path = (
                    target.lstrip("/")
                    if target.startswith("/")
                    else posixpath.normpath("xl/" + target)
                )
                first = {}
                counters = {
                    name: {
                        "n_peptide_rows": 0,
                        "quantity_states": Counter(),
                        "forms": set(),
                    }
                    for name, _, _ in BLOCKS.get(expected["name"], [])
                }
                for row_number, cells in decode_rows(
                    archive, path, strings, formats, formatted_cells
                ):
                    if row_number <= 12:
                        first[row_number] = cells
                    elif not counters:
                        break
                    if row_number <= 2:
                        continue
                    for name, sequence_col, quantity_cols in BLOCKS.get(
                        expected["name"], []
                    ):
                        sequence = cells.get(sequence_col + str(row_number))
                        if not isinstance(sequence, str) or not sequence.strip():
                            continue
                        counter = counters[name]
                        counter["n_peptide_rows"] += 1
                        counter["forms"].add(sequence)
                        for col in quantity_cols:
                            value = cells.get(col + str(row_number))
                            state = (
                                "missing"
                                if value is None
                                else "non_numeric"
                                if not isinstance(value, (int, float))
                                else "nonfinite"
                                if not math.isfinite(value)
                                else "positive"
                                if value > 0
                                else "zero"
                                if value == 0
                                else "negative"
                            )
                            counter["quantity_states"][state] += 1
                for number, expected_cells in enumerate(expected["first_12_rows"], 1):
                    actual = first.get(number, {})
                    assert actual == expected_cells, (
                        book["filename"],
                        expected["name"],
                        number,
                        actual,
                        expected_cells,
                    )
                    cell_count += len(actual)
                sheet_count += 1
                for name, counter in counters.items():
                    quantity_blocks.append(
                        {
                            "block": name,
                            "filename": book["filename"],
                            "sheet": expected["name"],
                            "n_peptide_rows": counter["n_peptide_rows"],
                            "n_distinct_literal_peptide_forms": len(counter["forms"]),
                            "quantity_states": dict(counter["quantity_states"]),
                            "training_admission_or_normalization_applied": False,
                        }
                    )
    result = {
        "status": "passed",
        "verified_asset_hashes": len(saved["verified_assets"]),
        "workbooks": len(saved["workbooks"]),
        "sheets": sheet_count,
        "original_cells_independently_verified": cell_count,
        "candidate_quantity_blocks": quantity_blocks,
        "duration_style_raw_values_preserved": formatted_cells,
        "quantity_count_scope": "Original peptide-bearing rows and raw cells, not independent specimens or admitted targets.",
        "model_training_or_predictions": False,
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
