"""Read-only PDF page/attachment inventory and rendering; no figure digitization."""

import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/endolysosomal_source_qualification/2026-10-24_1521_oztemiz-v1"
output = ROOT / "results/pdf-inventory"
output.mkdir(exist_ok=False)
started = time.perf_counter()
records = []
for path in sorted(RAW.glob("supplement-*.pdf")):
    assert path.read_bytes().startswith(b"%PDF-")
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_path = output / f"{path.stem}-page-{i + 1}.txt"
        text_path.write_text(text)
        pages.append(
            {
                "page": i + 1,
                "characters": len(text),
                "text_path": str(text_path.relative_to(ROOT)),
                "size_points": [float(page.mediabox.width), float(page.mediabox.height)],
            }
        )
    attachments = []
    for name, payloads in reader.attachments.items():
        attachments.append(
            {
                "name": name,
                "n_payloads": len(payloads),
                "sizes": [len(p) for p in payloads],
                "sha256": [hashlib.sha256(p).hexdigest() for p in payloads],
            }
        )
    subprocess.run(
        ["pdftoppm", "-scale-to", "1600", "-png", str(path), str(output / path.stem)], check=True
    )
    subprocess.run(
        ["pdfinfo", str(path)], stdout=(output / f"{path.stem}-pdfinfo.txt").open("w"), check=True
    )
    subprocess.run(
        ["pdftotext", "-layout", str(path), str(output / f"{path.stem}-poppler.txt")], check=True
    )
    record = {
        "name": path.name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "n_pages": len(pages),
        "pages": pages,
        "attachments": attachments,
        "metadata": dict(reader.metadata or {}),
    }
    records.append(record)
    print(
        json.dumps(
            {
                "file": path.name,
                "pages": len(pages),
                "attachments": attachments,
                "text": " ".join((reader.pages[0].extract_text() or "").split())[:650],
            }
        ),
        flush=True,
    )
assert len(records) == 5
receipt = {
    "files": records,
    "seconds": time.perf_counter() - started,
    "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    "numeric_labels_generated": False,
    "rendering": "Poppler, longest page dimension1600pixels",
}
(output / "inventory.json").write_text(json.dumps(receipt, indent=2) + "\n")
