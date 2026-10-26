"""Bounded public-asset retrieval, with one cross-tool operation ledger."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

import psutil

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
RAW = ROOT / "data/endolysosomal_source_audit" / FAMILY.name


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("name")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error("name must be one filename")
    path = RAW / args.name
    if path.exists():
        raise FileExistsError("preserve existing asset")
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    ledger_path = FAMILY / "results/discovery-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if len(ledger["operations"]) >= ledger["limit"]:
        raise RuntimeError("combined network-operation budget exhausted")
    start = time.monotonic()
    beginning = datetime.fromisoformat(config["utc"])
    if (datetime.now(timezone.utc) - beginning).total_seconds() > 600:
        raise RuntimeError("original-asset retrieval window closed")
    entry = {
        "number": len(ledger["operations"]) + 1,
        "tool": "urllib",
        "url": args.url,
        "path": str(path.relative_to(ROOT)),
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
    }
    ledger["operations"].append(entry)
    save(ledger_path, ledger)
    RAW.mkdir(parents=True, exist_ok=True)
    downloaded = sum(x.get("bytes", 0) for x in ledger["operations"])
    size = 0
    peak = 0
    try:
        with urllib.request.urlopen(args.url, timeout=60) as response, path.open("xb") as stream:
            entry.update(
                status_code=response.status, headers=dict(response.headers), final_url=response.url
            )
            while True:
                peak = max(peak, psutil.Process().memory_info().rss)
                if peak > config["local_rss_limit_bytes"]:
                    raise RuntimeError("RSS guard")
                if psutil.virtual_memory().available < config["available_ram_floor"]:
                    raise RuntimeError("available RAM guard")
                if shutil.disk_usage(ROOT).free < config["disk_floor"]:
                    raise RuntimeError("disk reserve guard")
                if time.monotonic() - start > 60:
                    raise TimeoutError("per-request wall-time guard")
                if (datetime.now(timezone.utc) - beginning).total_seconds() > 600:
                    raise TimeoutError("original-asset retrieval window")
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                size += len(chunk)
                if (
                    size > config["local_file_limit_bytes"]
                    or downloaded + size > config["local_total_bytes_limit"]
                ):
                    raise RuntimeError("byte guard")
                stream.write(chunk)
        entry.update(status="completed", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except BaseException as exc:
        entry.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        entry.update(bytes=size, elapsed_seconds=time.monotonic() - start, peak_rss_bytes=peak)
        save(ledger_path, ledger)
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
