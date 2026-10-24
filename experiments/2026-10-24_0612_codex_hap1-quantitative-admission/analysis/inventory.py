"""Inventory original source grains and raw fields; do not create training labels."""

import csv
import hashlib
import io
import json
import math
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import openpyxl

experiment = Path(__file__).resolve().parents[1]
config = json.loads((experiment / "reproduce/launch.json").read_text())
root = Path(config["source_root"])
started = time.monotonic()
inventory = {"members": [], "csv_tables": {}, "sample_sheets": {}, "receipts": []}
for path in sorted((root / "retrieval").glob("*/receipt.json")):
    receipt = json.loads(path.read_text())
    body = path.parent / "body"
    assert hashlib.sha256(body.read_bytes()).hexdigest() == receipt["sha256"]
    assert body.stat().st_size == receipt["n_bytes"]
    inventory["receipts"].append(receipt)

with zipfile.ZipFile(root / "retrieval/europepmc-supplements/body") as archive:
    assert archive.testzip() is None
    for member in archive.infolist():
        data = archive.read(member)
        inventory["members"].append(
            {
                "name": member.filename,
                "size_bytes": member.file_size,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        if member.filename.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
            headers = reader.fieldnames
            counts = {name: Counter() for name in headers}
            n_rows = 0
            for row in reader:
                assert None not in row and all(value is not None for value in row.values())
                n_rows += 1
                for name, value in row.items():
                    if value == "":
                        counts[name]["empty"] += 1
                    elif value == "NA":
                        counts[name]["literal_NA"] += 1
                    else:
                        try:
                            number = float(value)
                        except ValueError:
                            counts[name]["text"] += 1
                        else:
                            counts[name]["finite" if math.isfinite(number) else "nonfinite"] += 1
                            if number == 0:
                                counts[name]["zero"] += 1
                            if 1 <= number <= 2:
                                counts[name]["between_1_and_2_not_assumed_imputed"] += 1
            inventory["csv_tables"][member.filename] = {
                "n_rows": n_rows,
                "n_columns": len(headers),
                "headers": headers,
                "field_status_counts": counts,
                "semantic_status": "raw header preserved; numeric magnitudes do not establish units",
            }
        if member.filename.endswith(".xlsx"):
            workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=False)
            for sheet in workbook:
                rows = list(sheet.iter_rows(values_only=True))
                assert len(rows[0]) == len(set(rows[0]))
                inventory["sample_sheets"][sheet.title] = {
                    "source_member": member.filename,
                    "n_rows": len(rows) - 1,
                    "headers": rows[0],
                    "rows": [dict(zip(rows[0], row)) for row in rows[1:]],
                }

article = ET.parse(root / "retrieval/europepmc-xml/body")
inventory["license_text"] = ["".join(p.itertext()) for p in article.findall(".//permissions")]
inventory["supplement_links"] = [
    {"attributes": node.attrib, "text": "".join(node.itertext())}
    for node in article.findall(".//supplementary-material")
]
files = []
for path in sorted((root / "retrieval").glob("pride-files-page*/body")):
    rows = json.loads(path.read_text())
    files.extend(rows)
    assert len(rows) <= 100
assert len(rows) < 100, "Need another deposit page"
assert len({row["accession"] for row in files}) == len(files)
inventory["n_deposited_files"] = len(files)
inventory["n_raw_files"] = sum(row["fileName"].lower().endswith(".raw") for row in files)
inventory["processed_files"] = [r for r in files if not r["fileName"].lower().endswith(".raw")]
deposited_raw = {row["fileName"].removesuffix(".raw") for row in files}
inventory["sample_run_reconciliation"] = {}
for name, sheet in inventory["sample_sheets"].items():
    names = [row["raw_file"] for row in sheet["rows"]]
    inventory["sample_run_reconciliation"][name] = {
        "n_rows": len(names),
        "n_unique_raw_names": len(set(names)),
        "absent_from_deposit": sorted(set(names) - deposited_raw),
    }
sdrf_path = root / "deposits/PXD056426_community_annotated.sdrf.tsv"
if sdrf_path.exists():
    with sdrf_path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        sdrf = list(reader)
    inventory["sdrf"] = {"headers": reader.fieldnames, "n_rows": len(sdrf)}
    for key in reader.fieldnames:
        if "replicate" in key or "data file" in key or "source name" == key:
            inventory["sdrf"][key] = dict(Counter(row[key] for row in sdrf))
inventory["elapsed_seconds"] = time.monotonic() - started
(experiment / "results/source-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
print(
    json.dumps(
        {
            "n_members": len(inventory["members"]),
            "csv_tables": {
                k: {x: v[x] for x in ["n_rows", "n_columns"]}
                for k, v in inventory["csv_tables"].items()
            },
            "run_reconciliation": inventory["sample_run_reconciliation"],
            "n_deposited_files": len(files),
            "n_raw_files": inventory["n_raw_files"],
            "elapsed_seconds": inventory["elapsed_seconds"],
        },
        indent=2,
    )
)
