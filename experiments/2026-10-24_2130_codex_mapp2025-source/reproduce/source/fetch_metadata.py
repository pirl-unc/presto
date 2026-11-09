"""Retrieve the deposited original material table and separate community SDRF."""

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / "data/mapp2025_source/2026-10-24_2130"
NAMES = ["Data_Deposition_Table.xlsx", "PXD052314_community_annotated.sdrf.tsv"]


def main():
    metadata = json.loads((RAW / "files.json").read_text())
    receipts = []
    for name in NAMES:
        source = next(x for x in metadata if x["fileName"] == name)
        url = next(x["value"] for x in source["publicFileLocations"] if x["value"].startswith("ftp:"))
        path = RAW / name
        if path.exists():
            raise FileExistsError(path)
        start = time.monotonic()
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read(20 * 1024**2 + 1)
        assert len(content) == source["fileSizeBytes"] < 20 * 1024**2
        sha1 = hashlib.sha1(content).hexdigest()
        supplied = source.get("fileChecksum", "")
        path.write_bytes(content)
        receipts.append({
            "url": url, "path": str(path), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(), "sha1": sha1,
            "supplied_checksum": supplied, "seconds": time.monotonic() - start,
            "utc": datetime.now(timezone.utc).isoformat(),
        })
        (EXP / "results/metadata-retrieval.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps(receipts[-1]), flush=True)


if __name__ == "__main__":
    main()
