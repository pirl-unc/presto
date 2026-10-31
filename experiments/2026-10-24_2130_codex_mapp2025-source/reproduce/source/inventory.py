"""Read-only workbook and deposited material inventory; no training adapter."""

import hashlib
import json
import resource
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import openpyxl

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / "data/mapp2025_source/2026-10-24_2130"


def main():
    start = time.monotonic()
    inventory = []
    with zipfile.ZipFile(RAW / "supplement.zip") as archive:
        for info in archive.infolist():
            entry = {"member": info.filename, "bytes": info.file_size, "crc32": info.CRC}
            chosen = None
            if "Table 3-" in info.filename:
                chosen = "Table3.xlsx"
            elif "Table 5 -" in info.filename:
                chosen = "Table5.xlsx"
            elif info.filename.endswith(".docx"):
                chosen = "captions.docx"
            if chosen:
                payload = archive.read(info)
                entry.update(sha256=hashlib.sha256(payload).hexdigest(), local_name=chosen)
                dst = RAW / chosen
                if dst.exists():
                    assert dst.read_bytes() == payload
                else:
                    dst.write_bytes(payload)
            inventory.append(entry)
    (EXP / "results/member-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    sheets = []
    for name in ["Table3.xlsx", "Table5.xlsx", "Data_Deposition_Table.xlsx"]:
        book = openpyxl.load_workbook(RAW / name, read_only=True, data_only=True)
        for sheet in book:
            top = []
            for row in sheet.iter_rows(min_row=1, max_row=min(4, sheet.max_row)):
                top.append({cell.coordinate: cell.value for cell in row if cell.value is not None})
            sheets.append({"file": name, "sheet": sheet.title, "rows": sheet.max_row,
                           "cols": sheet.max_column, "top": top})
        book.close()
    (EXP / "results/sheet-inventory.json").write_text(json.dumps(sheets, indent=2) + "\n")
    with zipfile.ZipFile(RAW / "captions.docx") as doc:
        root = ET.fromstring(doc.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        text = "\n".join("".join(p.itertext()) for p in root.findall(".//w:p", ns))
        (EXP / "results/captions.txt").write_text(text + "\n")
    stats = {"seconds": time.monotonic() - start,
             "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3}
    (EXP / "results/inventory-runtime.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({"sheets": [{k: v for k, v in s.items() if k != "top"} for s in sheets],
                      "runtime": stats}, indent=2))


if __name__ == "__main__":
    main()
