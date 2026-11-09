"""Retrieve a frozen list of public HTTP assets with explicit byte/disk bounds."""

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/do_source_qualification/2026-10-24_1818_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    assert not args.receipt.exists()
    source = ROOT / "reproduce/source/retrieve.py"
    if source.exists():
        assert source.read_bytes() == Path(__file__).read_bytes()
    else:
        shutil.copy2(__file__, source)
    config = json.loads(args.manifest.read_text())
    total = sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file())
    receipts = []
    for asset in config["assets"]:
        path = RAW / asset["name"]
        assert not path.exists()
        assert shutil.disk_usage(RAW).free >= 2 * 1024**3
        record = {**asset, "started_utc": datetime.now(UTC).isoformat()}
        started = time.perf_counter()
        try:
            with requests.get(asset["url"], stream=True, timeout=(15, 60)) as response:
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
        except (requests.RequestException, OSError, RuntimeError) as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record["seconds"] = time.perf_counter() - started
        receipts.append(record)
        args.receipt.write_text(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "assets": receipts,
                    "total_stored_bytes": total,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
