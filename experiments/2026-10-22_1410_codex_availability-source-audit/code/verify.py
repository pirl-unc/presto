"""Independently reconstruct inventory checks directly from original OOXML."""

import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from datacache import install_bundle


FAMILY = Path(__file__).resolve().parents[2 if Path(__file__).parent.name == "source" else 1]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xml_cells(path):
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(x.itertext()) for node in root for x in [node]
                       if node.tag.endswith("}si")]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cells = {}
        for cell in root.findall(".//s:sheetData/s:row/s:c", NS):
            value = cell.find("s:v", NS)
            if cell.get("t") == "inlineStr":
                text = "".join(x.text or "" for x in cell.findall(".//s:t", NS))
            elif value is None:
                continue
            elif cell.get("t") == "s":
                text = strings[int(value.text)]
            elif cell.get("t") in ("str", "e"):
                text = value.text
            else:
                text = float(value.text)
            cells[cell.get("r")] = text
    return cells


def main():
    assets = json.loads((FAMILY / "reproduce/input_assets.json").read_text())["assets"]
    bundle_id = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    source_root = FAMILY.parents[1] / "data/availability_sources"
    snapshots = source_root / "snapshots/2026-10-22"
    paths = install_bundle(source_root / bundle_id, assets, source_paths={
        name: str(snapshots / name) for name in assets if (snapshots / name).is_file()
    })
    for name, path in paths.items():
        raw = Path(path).read_bytes()
        assert len(raw) == assets[name]["size"]
        assert hashlib.sha256(raw).hexdigest() == assets[name]["sha256"]
    tables = json.loads((FAMILY / "results/workbook_inventory.json").read_text())
    verified_cells = 0
    for table in tables:
        cells = read_xml_cells(paths[f"PXD057588/table{table['table']}.xlsx"])
        sequences = {key: value for key, value in cells.items()
                     if key.startswith("A") and key[1:].isdigit() and int(key[1:]) >= 6}
        assert len(sequences) == table["primary_rows"]
        assert len(set(sequences.values())) == table["distinct_sequence_labels"]
        for header in table["header_cells"]:
            assert cells[header["cell"]] == header["value"]
        for channel in table["channels"]:
            assert cells[channel["channel_cell"]] == channel["source_channel_label"]
            assert cells[channel["sample_cell"]] == channel["source_sample_label"]
            values = [cells[channel["column"] + key[1:]] for key in sequences]
            assert len(values) == channel["numeric_cells"]
            assert math.isclose(sum(values), channel["sum"], rel_tol=1e-12)
            assert min(values) == channel["minimum"]
            assert max(values) == channel["maximum"]
            assert sum(x == 0 for x in values) == channel["zero_cells"]
            verified_cells += len(values)
    with Path(paths["PXD057588/community.sdrf.tsv"]).open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    result = json.loads((FAMILY / "results/sdrf_inventory.json").read_text())
    assert len(rows) - 1 == result["rows"]
    assert len(rows[0]) == result["columns"]
    for column in result["column_values"]:
        index = column["column_index_1based"] - 1
        assert rows[0][index] == column["name"]
        assert dict(collections.Counter(row[index] for row in rows[1:])) == column["values"]
    output = {
        "status": "passed", "verified_asset_hashes": len(assets),
        "independent_OOXML_tables": len(tables), "verified_quantity_cells": verified_cells,
        "verified_sdrf_cells": (len(rows) - 1) * len(rows[0]),
        "scope": "Byte identities, source headers, primary counts, channel values/aggregates; not biological eligibility or SDRF ontology validation.",
    }
    (FAMILY / "results/independent_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
