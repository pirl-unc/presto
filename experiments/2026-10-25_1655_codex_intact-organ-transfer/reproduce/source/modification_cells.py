"""Check original OOXML column K for every F3 assignment, independently of Hitlist."""

import argparse
import json
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("--requests", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.requests.read_text())
root = Path(__file__).resolve().parents[1]
records = {
    r["source_record_id"]: r
    for r in pq.read_table(
        request["source_records"],
        columns=["source_record_id", "source_file", "sheet_name", "source_row_index"],
    ).to_pylist()
}
assignments = pq.read_table(
    request["assignments"],
    columns=["assignment_id", "source_record_id", "modifications_json"],
).to_pylist()
targets = {
    (
        records[a["source_record_id"]]["sheet_name"],
        records[a["source_record_id"]]["source_row_index"],
    ): a
    for a in assignments
    if records[a["source_record_id"]]["source_file"] == "F3_SF7_SearchResults.xlsx"
}
ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
results = []
with zipfile.ZipFile(request["workbook"]) as archive:
    shared = (
        [
            "".join(n.itertext())
            for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]
        if "xl/sharedStrings.xml" in archive.namelist()
        else []
    )
    links = {
        r.attrib["Id"]: r.attrib["Target"]
        for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    }
    book = ET.fromstring(archive.read("xl/workbook.xml"))
    for sheet in book.find(ns + "sheets"):
        path = links[sheet.attrib[rel]]
        path = (
            path.lstrip("/")
            if path.startswith("/")
            else posixpath.normpath("xl/" + path)
        )
        for row in ET.fromstring(archive.read(path)).iter(ns + "row"):
            key = (sheet.attrib["name"], int(row.attrib["r"]))
            if key not in targets:
                continue
            address = "K" + str(key[1])
            cells = [c for c in row.findall(ns + "c") if c.attrib.get("r") == address]
            assert len(cells) <= 1
            cell = cells[0] if cells else None
            raw = None if cell is None else cell.findtext(ns + "v")
            kind = None if cell is None else cell.attrib.get("t", "n")
            value = shared[int(raw)] if kind == "s" and raw is not None else raw
            if kind == "inlineStr":
                value = "".join(n.text or "" for n in cell.iter(ns + "t"))
            canonical = json.loads(targets[key]["modifications_json"])[
                "Unlocalized Mod"
            ]
            assert value == canonical, key
            results.append(
                {
                    "assignment_id": targets[key]["assignment_id"],
                    "sheet": key[0],
                    "cell": address,
                    "cell_present": bool(cells),
                    "type": kind,
                    "raw_value": raw,
                    "resolved_value": value,
                }
            )
assert len(results) == len(targets) == 4198
counts = Counter(
    json.dumps(
        {k: r[k] for k in ("cell_present", "type", "raw_value", "resolved_value")},
        sort_keys=True,
    )
    for r in results
)
result = {
    "status": "passed",
    "n_assignment_cells": len(results),
    "cell_patterns": dict(counts),
    "numeric_assay_outcomes_queried": False,
    "coordinates": results,
}
with (root / "results/original-modification-cells.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(
    json.dumps(
        {
            "status": result["status"],
            "n_assignment_cells": len(results),
            "cell_patterns": dict(counts),
        }
    )
)
