"""Bounded retrieval of public primary-source metadata; no quantitative curator."""

import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/do_source_qualification/2026-10-24_1818_v1"
ASSETS = [
    (
        "article.xml",
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6398211/fullTextXML",
    ),
    (
        "proteomexchange.xml",
        "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD010301&outputMode=XML",
    ),
    (
        "massive.html",
        "https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000082570",
    ),
]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    destination = ROOT / "results/acquisition.json"
    assert not destination.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/acquire.py")
    records = []
    total = 0
    for name, url in ASSETS:
        path = RAW / name
        assert not path.exists()
        assert shutil.disk_usage(RAW).free >= 2 * 1024**3
        record = {
            "name": name,
            "url": url,
            "started_utc": datetime.now(UTC).isoformat(),
        }
        started = time.perf_counter()
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                record.update(
                    status_code=response.status_code,
                    final_url=response.url,
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower()
                        in ("content-type", "content-length", "last-modified", "etag")
                    },
                )
                response.raise_for_status()
                size = 0
                digest = hashlib.sha256()
                with path.open("xb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        size += len(block)
                        total += len(block)
                        if size > 100 * 1024**2 or total > 512 * 1024**2:
                            raise RuntimeError("Prospective byte cap exceeded")
                        stream.write(block)
                        digest.update(block)
                record.update(
                    status="retrieved",
                    bytes=size,
                    sha256=digest.hexdigest(),
                    path=str(path),
                )
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        destination.write_text(
            json.dumps({"assets": records, "total_download_bytes": total}, indent=2)
            + "\n"
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
