"""Retrieve source MAPP results, preserving the complete archive provenance."""

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / "data/mapp2025_source/2026-10-24_2130"
NAME = "EC_infection_Degradome_MAPP_Peptides.zip"


def main():
    source = next(x for x in json.loads((RAW / "files.json").read_text()) if x["fileName"] == NAME)
    assert source["fileSizeBytes"] < 512 * 1024**2
    url = next(x["value"] for x in source["publicFileLocations"] if x["value"].startswith("ftp:"))
    path = RAW / NAME
    assert not path.exists() and not path.with_suffix(".partial").exists()
    start = time.monotonic()
    digest = hashlib.sha256()
    size = 0
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        path.with_suffix(".partial").open("wb") as f,
    ):
        while chunk := response.read(1024**2):
            size += len(chunk)
            assert size <= source["fileSizeBytes"]
            digest.update(chunk)
            f.write(chunk)
    assert size == source["fileSizeBytes"]
    path.with_suffix(".partial").rename(path)
    receipt = {
        "url": url,
        "path": str(path),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "source_checksum": source["checksum"],
        "seconds": time.monotonic() - start,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    (EXP / "results/archive-retrieval.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
