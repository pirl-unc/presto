"""Independently reconstruct the read-only XML inventory through openpyxl."""

import argparse
from collections import Counter
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import resource
import time
from zipfile import ZipFile

import openpyxl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    start = time.perf_counter()
    inventory = json.loads((args.inventory_dir / "inventory.json").read_text())
    with gzip.open(args.inventory_dir / "cells.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    keyed = {(x["workbook"], x["sheet"], x["address"]): x for x in rows}
    assert len(keyed) == len(rows)
    checked, numbers, text = 0, 0, 0
    with ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        for meta in inventory["workbooks"]:
            data = archive.read(meta["name"])
            assert data == (args.inventory_dir / meta["name"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == meta["sha256"]
            workbook = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=False)
            assert workbook.sheetnames == [s["name"] for s in meta["sheets"]]
            for sheet, smeta in zip(workbook, meta["sheets"], strict=True):
                assert sheet.sheet_state == smeta["state"]
                observed = Counter()
                for row in sheet:
                    for cell in row:
                        if cell.value is None:
                            continue
                        item = keyed.pop((meta["name"], sheet.title, cell.coordinate))
                        observed[item["type"]] += 1
                        if item["type"] == "n":
                            assert isinstance(cell.value, (int, float))
                            assert float(item["value"]) == cell.value
                            numbers += 1
                        else:
                            assert cell.value == item["value"]
                            text += 1
                        assert item["formula"] is None and cell.data_type != "f"
                        checked += 1
                assert dict(observed) == smeta["cell_types"]
                assert sum(observed.values()) == smeta["occupied_cells"]
            workbook.close()
    assert not keyed
    result = {
        "status": "pass", "reader": "openpyxl", "reader_version": openpyxl.__version__,
        "n_workbooks": len(inventory["workbooks"]),
        "n_sheets": sum(len(b["sheets"]) for b in inventory["workbooks"]),
        "n_nonempty_cells": checked, "n_numeric_cells": numbers, "n_text_cells": text,
        "n_formula_cells": 0, "original_bytes_preserved": True,
        "numeric_exact_float_matches": True, "training_measurements_admitted": 0,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
