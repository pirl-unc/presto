"""Independent original OOXML decoding, preserving raw cached numeric values."""

import hashlib
import json
import posixpath
import resource
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / "data/mapp2025_source/2026-10-24_2130"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def main():
    start = time.monotonic()
    expected = json.loads((EXP / "results/workbook-audit.json").read_text())["sheets"]
    checks = []
    for file in sorted({r["file"] for r in expected}):
        with zipfile.ZipFile(RAW / file) as archive:
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                strings = ["".join(n.itertext()) for n in root.findall("s:si", NS)]
            relations = {
                n.attrib["Id"]: n.attrib["Target"]
                for n in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            }
            book = ET.fromstring(archive.read("xl/workbook.xml"))
            for sheet in book.findall("s:sheets/s:sheet", NS):
                name = sheet.attrib["name"]
                matched = [r for r in expected if r["file"] == file and r["sheet"] == name]
                if not matched:
                    continue
                target = relations[sheet.attrib[REL]]
                target = (
                    target.lstrip("/")
                    if target.startswith("/")
                    else posixpath.normpath("xl/" + target)
                )
                sha = hashlib.sha256()
                count = 0
                formula_count = 0
                with archive.open(target) as stream:
                    for _, element in ET.iterparse(stream, events=("end",)):
                        if element.tag != "{" + NS["s"] + "}c":
                            continue
                        raw_value = element.find("s:v", NS)
                        dtype = element.attrib.get("t", "n")
                        value = None
                        if dtype == "inlineStr":
                            child = element.find("s:is", NS)
                            value = "".join(child.itertext()) if child is not None else None
                        elif raw_value is not None and raw_value.text is not None:
                            raw = raw_value.text
                            if dtype == "s":
                                value = strings[int(raw)]
                            elif dtype == "b":
                                value = raw == "1"
                            elif dtype in {"str", "e"}:
                                value = raw
                            else:
                                value = float(raw)
                        formula_count += element.find("s:f", NS) is not None
                        if value is not None and value != "":
                            sha.update(
                                (
                                    json.dumps(
                                        [element.attrib["r"], value],
                                        ensure_ascii=False,
                                        allow_nan=False,
                                    )
                                    + "\n"
                                ).encode()
                            )
                            count += 1
                        element.clear()
                check = {
                    "file": file,
                    "sheet": name,
                    "value_cells": count,
                    "cell_sha256": sha.hexdigest(),
                    "formula_cells": formula_count,
                }
                assert check["value_cells"] == matched[0]["value_cells"], check
                assert check["cell_sha256"] == matched[0]["cell_sha256"], check
                checks.append(check)
    receipt = {
        "checks": checks,
        "total_cells": sum(x["value_cells"] for x in checks),
        "seconds": time.monotonic() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    (EXP / "results/xml-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
