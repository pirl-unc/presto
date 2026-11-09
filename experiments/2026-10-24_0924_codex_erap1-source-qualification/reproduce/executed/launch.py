"""Read-only OOXML source inventory; does not curate training observations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
EXPECTED = "a9d7bcbae52b25e55aec5f725578238fd0e6719720153dc829b999dd4bde5d2c"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inventory(data, name, cell_stream):
    with ZipFile(BytesIO(data)) as archive:
        assert archive.testzip() is None
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")):
                strings.append("".join(item.itertext()))
        relationships = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        result = {"name": name, "bytes": len(data), "sha256": sha(data), "sheets": []}
        for sheet in workbook.find("s:sheets", NS):
            target = relationships[sheet.attrib[f"{{{REL}}}id"]]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            root = ET.fromstring(archive.read(target))
            labels, formulas, counts, occupied = [], [], Counter(), []
            for cell in root.findall("s:sheetData/s:row/s:c", NS):
                kind = cell.get("t", "n")
                node = cell.find("s:v", NS)
                raw = node.text if node is not None else None
                inline = cell.find("s:is", NS)
                value = strings[int(raw)] if kind == "s" and raw is not None else raw
                if inline is not None:
                    value = "".join(inline.itertext())
                formula = cell.find("s:f", NS)
                if value is None and formula is None:
                    continue
                record = {
                    "workbook": name, "sheet": sheet.get("name"),
                    "address": cell.get("r"), "type": kind, "raw_value": raw,
                    "value": value, "formula": formula.text if formula is not None else None,
                    "formula_attributes": formula.attrib if formula is not None else None,
                    "style": cell.get("s"),
                }
                cell_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                occupied.append(cell.get("r"))
                counts[kind] += 1
                if kind in ("s", "inlineStr", "str"):
                    labels.append({"address": cell.get("r"), "value": value})
                if formula is not None:
                    formulas.append(record)
            merges = root.find("s:mergeCells", NS)
            dim = root.find("s:dimension", NS)
            result["sheets"].append({
                "name": sheet.get("name"), "state": sheet.get("state", "visible"),
                "dimension": dim.get("ref") if dim is not None else None,
                "occupied_cells": len(occupied), "cell_types": dict(counts),
                "formulas": formulas, "labels": labels,
                "merged_ranges": [item.get("ref") for item in merges] if merges is not None else [],
            })
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.archive.read_bytes()
    assert sha(source) == EXPECTED
    assert len(source) == 479402
    shutil.copyfile(__file__, args.out / "executed-launch.py")
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
    record = {"argv": sys.argv, "utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.executable, "python_version": platform.python_version(),
              "git_commit": git, "tracked_dirty": dirty, "archive_sha256": sha(source),
              "training": None, "evaluation": None, "workbooks": []}
    with ZipFile(BytesIO(source)) as archive, gzip.open(args.out / "cells.jsonl.gz", "wt") as cells:
        assert archive.testzip() is None
        for member in archive.infolist():
            assert member.filename in {
                "Source Data_Main Figures.xlsx", "Source Data_Supplementary Figures.xlsx"}
            data = archive.read(member.filename)
            (args.out / member.filename).write_bytes(data)
            record["workbooks"].append(inventory(data, member.filename, cells))
    record["elapsed_seconds"] = time.perf_counter() - started
    record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.out / "inventory.json", record)
    print(json.dumps({"out": str(args.out), "elapsed_seconds": record["elapsed_seconds"],
                      "sheets": [[book["name"], len(book["sheets"])] for book in record["workbooks"]]}))


if __name__ == "__main__":
    main()
