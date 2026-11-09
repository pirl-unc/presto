"""Independent OOXML evidence audit; never produces training targets."""
from __future__ import annotations

import collections
import gzip
import hashlib
import io
import json
import posixpath
import resource
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import openpyxl

FAMILY = Path(__file__).resolve().parents[1]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PRIMARY = {
    "dasgupta2014_s001": dict(sheet="Table S1", start=8, sequence="K", ratios=["O", "P", "Q", "S", "T", "U"], annotation="V"),
    "gelman2013_s004": dict(sheet="Sheet1", start=10, sequence="J", ratios=["N", "O", "P", "R", "S", "T"], annotation="U"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def raw_workbook(path):
    """Decode source OOXML without using openpyxl parsing or cell helpers."""
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for node in ET.fromstring(archive.read("xl/sharedStrings.xml")):
                strings.append("".join(t.text or "" for t in node.findall(".//m:t", NS)))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            target = targets[sheet.get("{" + REL + "}id")]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            root = ET.fromstring(archive.read(member))
            cells = {}
            for row in root.findall("m:sheetData/m:row", NS):
                for cell in row.findall("m:c", NS):
                    raw = cell.findtext("m:v", default=None, namespaces=NS)
                    kind = cell.get("t", "n")
                    if kind == "s":
                        value = strings[int(raw)] if raw is not None else None
                    elif kind == "inlineStr":
                        value = "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                    elif raw is None:
                        value = None
                    elif kind == "b":
                        value = bool(int(raw))
                    elif kind == "n":
                        value = float(raw)
                    elif kind in {"str", "e"}:
                        value = raw
                    else:
                        raise ValueError((path, cell.get("r"), kind))
                    formula = cell.find("m:f", NS)
                    cells[cell.get("r")] = dict(
                        attributes=dict(cell.attrib), raw_v=raw, value=value,
                        formula=None if formula is None else ET.tostring(formula, encoding="unicode"),
                        source_xml=ET.tostring(cell, encoding="unicode"),
                    )
            layout = {
                "sheet_attributes": dict(sheet.attrib), "member": member,
                "dimension": root.find("m:dimension", NS).get("ref"),
                "rows": [dict(r.attrib) for r in root.findall("m:sheetData/m:row", NS)],
                "columns": [dict(c.attrib) for c in root.findall("m:cols/m:col", NS)],
                "merged_cells": [dict(c.attrib) for c in root.findall("m:mergeCells/m:mergeCell", NS)],
                "related_members": [n for n in archive.namelist() if "comment" in n.lower() or n == "xl/styles.xml"],
            }
            yield sheet.get("name"), cells, layout


def source_rows(cells):
    rows = collections.defaultdict(dict)
    for address, cell in cells.items():
        split = next(i for i, c in enumerate(address) if c.isdigit())
        rows[int(address[split:])][address[:split]] = cell["value"]
    return dict(rows)


def main():
    started = time.perf_counter()
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    assets = json.loads((FAMILY / "results/retrieval.json").read_text())["assets"]
    output = FAMILY / "results/workbook-audit.json"
    assert not output.exists()
    inventory, primary_rows = [], []
    for asset in assets:
        path = Path(asset["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        if not asset["http_success"] or not zipfile.is_zipfile(path):
            continue
        tick = time.perf_counter()
        library = openpyxl.load_workbook(io.BytesIO(path.read_bytes()), data_only=True)
        sheets, layouts = [], {}
        with gzip.open(FAMILY / "results" / (asset["id"] + "-cells.jsonl.gz"), "wt") as dump:
            for name, cells, layout in raw_workbook(path):
                layouts[name] = layout
                assert not any(c["formula"] is not None for c in cells.values()), "formula requires explicit cache/translation audit"
                errors = []
                worksheet = library[name]
                for address, cell in cells.items():
                    actual = worksheet[address].value
                    if actual != cell["value"]:
                        errors.append(dict(cell=address, xml=cell["value"], openpyxl=actual))
                    dump.write(json.dumps(dict(sheet=name, cell=address, **cell), ensure_ascii=False) + "\n")
                # Catch library cells omitted by independent XML parsing too.
                assert all(c.value is None or c.coordinate in cells for row in worksheet for c in row)
                rows = source_rows(cells)
                sheets.append(dict(
                    sheet=name, dimension=layout["dimension"], n_explicit_cells=len(cells),
                    n_nonempty_cells=sum(c["value"] is not None for c in cells.values()),
                    n_nonempty_rows=sum(any(v is not None for v in row.values()) for row in rows.values()),
                    n_formula_cells=0, n_parser_disagreements=len(errors),
                    parser_disagreements=errors, type_counts=dict(collections.Counter(c["attributes"].get("t", "n") for c in cells.values())),
                    legend_and_headers={str(k): v for k, v in rows.items() if k <= 20},
                ))
                pc = PRIMARY.get(asset["id"])
                if pc and name == pc["sheet"]:
                    for index, row in sorted(rows.items()):
                        if index < pc["start"] or not any(v is not None for v in row.values()):
                            continue
                        primary_rows.append(dict(
                            asset=asset["id"], sheet=name, row=index,
                            cell_line=row.get("A"), experiment=row.get("B"), charge=row.get("C"),
                            observed_mass=row.get("E"), sequence_raw=row.get(pc["sequence"]),
                            ratios=[row.get(c) for c in pc["ratios"]],
                            ratio_cells=[c + str(index) for c in pc["ratios"]],
                            average_annotation=row.get(pc["annotation"]),
                        ))
        library.close()
        seconds = time.perf_counter() - tick
        assert seconds < config["audit"]["parse_seconds_per_asset"]
        write_json(FAMILY / "results" / (asset["id"] + "-layout.json"), layouts)
        inventory.append(dict(asset=asset["id"], source_sha256=asset["sha256"], sheets=sheets, seconds=seconds))
        print(asset["id"], "cells", sum(s["n_explicit_cells"] for s in sheets), "disagreements", sum(s["n_parser_disagreements"] for s in sheets), flush=True)
    with gzip.open(FAMILY / "results/primary-row-audit.jsonl.gz", "wt") as stream:
        for row in primary_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = dict(
        workbooks=inventory, n_primary_source_rows=len(primary_rows),
        primary_by_asset=dict(collections.Counter(r["asset"] for r in primary_rows)),
        n_parser_disagreements=sum(s["n_parser_disagreements"] for a in inventory for s in a["sheets"]),
        n_explicit_cells=sum(s["n_explicit_cells"] for a in inventory for s in a["sheets"]),
        seconds=time.perf_counter()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_training_admission=True,
    )
    write_json(output, summary)
    assert summary["peak_rss_bytes"] < config["audit"]["peak_rss_bytes"]
    assert summary["n_parser_disagreements"] == 0
    print({k: v for k, v in summary.items() if k != "workbooks"}, flush=True)


if __name__ == "__main__":
    main()
