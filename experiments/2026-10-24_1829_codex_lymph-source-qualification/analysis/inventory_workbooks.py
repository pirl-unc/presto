"""Verify original assets and inventory workbook headers without fitting labels."""

import hashlib
import json
import resource
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/lymph_source_qualification/2026-10-24_1829_v1"


def main():
    started = time.perf_counter()
    destination = ROOT / "results/workbook-inventory.json"
    assert not destination.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/inventory_workbooks.py")
    receipt = json.loads((ROOT / "results/processed.json").read_text())
    meta_path = RAW / "PMC11233987.319.json"
    meta_asset = next(a for a in receipt["assets"] if a["name"] == meta_path.name)
    assert hashlib.md5(
        meta_path.read_bytes(), usedforsecurity=False
    ).hexdigest() == meta_asset["expected_etag"].strip('"')
    metadata = json.loads(meta_path.read_text())
    urls = metadata["media_urls"] + [
        metadata["xml_url"],
        metadata["pdf_url"],
        metadata["text_url"],
    ]
    declared_md5 = {
        Path(urlparse(u).path).name: parse_qs(urlparse(u).query)["md5"][0] for u in urls
    }
    declared_md5[meta_path.name] = meta_asset["expected_etag"].strip('"')
    hashes = []
    for asset in receipt["assets"]:
        assert asset["status"] == "retrieved"
        path = RAW / asset["name"]
        content = path.read_bytes()
        assert len(content) == asset["expected_bytes"] == asset["bytes"]
        assert hashlib.sha256(content).hexdigest() == asset["sha256"]
        assert asset["headers"]["ETag"] == asset["expected_etag"]
        expected = declared_md5[path.name]
        assert len(expected) == 32 and "-" not in expected
        assert hashlib.md5(content, usedforsecurity=False).hexdigest() == expected
        hashes.append(
            {
                "name": asset["name"],
                "bytes": len(content),
                "sha256": asset["sha256"],
                "md5": expected,
            }
        )
    books = []
    for path in sorted(RAW.glob("*.xlsx")):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            uncompressed = sum(item.file_size for item in archive.infolist())
        book = openpyxl.load_workbook(path, read_only=True, data_only=False)
        sheets = []
        for sheet in book:
            rows = []
            for row in sheet.iter_rows(min_row=1, max_row=min(sheet.max_row, 12)):
                cells = {
                    cell.coordinate: cell.value
                    for cell in row
                    if cell.value is not None
                }
                rows.append(cells)
            sheets.append(
                {
                    "name": sheet.title,
                    "declared_max_row": sheet.max_row,
                    "declared_max_column": sheet.max_column,
                    "first_12_rows": rows,
                }
            )
        book.close()
        books.append(
            {
                "filename": path.name,
                "uncompressed_bytes": uncompressed,
                "sheets": sheets,
            }
        )
        print(
            path.name,
            [
                (s["name"], s["declared_max_row"], s["declared_max_column"])
                for s in sheets
            ],
            flush=True,
        )
    result = {
        "status": "retrieved_and_header_inventory_complete",
        "scope": "Source dimensions/first12rows only; dimensions are not admitted example counts.",
        "verified_assets": hashes,
        "workbooks": books,
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "model_training_or_predictions": False,
    }
    destination.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["verified_assets", "workbooks"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
