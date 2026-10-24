"""Check every metadata value against the original OOXML cell payload."""

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import openpyxl

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    start = time.perf_counter()
    results = []
    for name in ("project-download.xlsx", "sample-download.xlsx", "datasets-download.xlsx"):
        path = args.data_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = json.loads(path.with_suffix(".xlsx.receipt.json").read_text())
        assert digest == receipt["sha256"]
        book = openpyxl.load_workbook(path, read_only=True, data_only=False)
        with ZipFile(path) as archive:
            assert archive.testzip() is None
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                strings = [
                    "".join(x.itertext())
                    for x in ET.fromstring(archive.read("xl/sharedStrings.xml"))
                ]
            rels = {
                x.get("Id"): x.get("Target")
                for x in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            }
            root = ET.fromstring(archive.read("xl/workbook.xml"))
            for sheet in root.find("s:sheets", NS):
                sheet_name = sheet.get("name")
                target = rels[sheet.get(f"{{{REL}}}id")]
                target = target.lstrip("/") if target.startswith("/") else "xl/" + target
                source = ET.fromstring(archive.read(target))
                expected, types = {}, Counter()
                for cell in source.findall("s:sheetData/s:row/s:c", NS):
                    assert cell.find("s:f", NS) is None
                    kind = cell.get("t", "n")
                    node = cell.find("s:v", NS)
                    value = node.text if node is not None else None
                    inline = cell.find("s:is", NS)
                    if kind == "s" and value is not None:
                        value = strings[int(value)]
                    elif inline is not None:
                        value = "".join(inline.itertext())
                    elif kind == "n" and value is not None:
                        value = float(value)
                    if value in (None, ""):
                        types["empty_payload"] += 1
                        continue
                    expected[cell.get("r")] = value
                    types[kind] += 1
                n_values = len(expected)
                for row in book[sheet_name]:
                    for cell in row:
                        if cell.value in (None, ""):
                            continue
                        assert expected.pop(cell.coordinate) == cell.value, (
                            name,
                            sheet_name,
                            cell.coordinate,
                        )
                assert not expected
                results.append(
                    {
                        "workbook": name,
                        "sheet": sheet_name,
                        "sha256": digest,
                        "n_nonempty_values": n_values,
                        "raw_cell_types": dict(types),
                    }
                )
        book.close()
    result = {
        "status": "pass",
        "method": "original OOXML versus openpyxl, every nonempty value",
        "sheets": results,
        "n_nonempty_values": sum(x["n_nonempty_values"] for x in results),
        "n_formula_cells": 0,
        "training_observations": 0,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
