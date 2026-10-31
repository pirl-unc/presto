"""Bounded public-source acquisition after immutable metadata-only pool selection."""

import hashlib
import json
import resource
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from hitlist.evidence_snapshot import snapshot_file
from hitlist.proteometools_ms1 import BASE_URL, CORE
from hitlist.provenance import file_digest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = REPO / "data/measured_ms1_grouped_observer/2026-10-24_1702_v1"
PILOT = REPO / "data/measured_ms1_source_qualification/2026-10-24_1546_v1"
GIB = 1024**3


def main():
    started = time.perf_counter()
    selection_file = ROOT / "results/selection.json"
    selection = json.loads(selection_file.read_text())
    assert selection["status"] == "selected" and len(selection["selected"]) == 24
    assert not RAW.exists(), (
        "Existing raw generation is immutable; inspect its receipt before any retry"
    )
    free = shutil.disk_usage(REPO).free
    assert free >= 4 * GIB, f"Predeclared free-disk gate: {free / GIB:.2f} GiB available"
    RAW.mkdir(parents=True)
    shutil.copy2(__file__, ROOT / "reproduce/source/acquire.py")
    context = {
        "started_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "python": sys.executable,
        "selection": file_digest(selection_file),
        "initial_disk_free_bytes": free,
        "source_concurrency": 1,
        "max_transfer_bytes": GIB,
        "max_asset_bytes": 128 * 1024**2,
        "raw_directory": str(RAW),
    }
    (ROOT / "reproduce/acquisition.json").write_text(json.dumps(context, indent=2) + "\n")
    for spec in json.loads(CORE.read_text())["assets"]:
        assert file_digest(PILOT / spec["filename"]) == {
            k: spec[k] for k in ("sha256", "size_bytes")
        }
        snapshot_file(PILOT / spec["filename"], RAW / spec["filename"])
    total = 0
    archives = []
    session = requests.Session()
    session.headers["User-Agent"] = "Presto-public-source-qualification/1.0"
    for number, pool in enumerate(selection["selected"], 1):
        name = pool["source_name"]
        path = RAW / name
        receipt_path = RAW / (name + ".receipt.json")
        partial = RAW / (name + ".partial")
        receipt = {
            "pool": pool["pool"],
            "source_url": BASE_URL + name,
            "started_utc": datetime.now(UTC).isoformat(),
            "status": "started",
            "size_bytes": 0,
        }
        transfer_started = time.perf_counter()
        sha = hashlib.sha256()
        try:
            with session.get(receipt["source_url"], stream=True, timeout=(20, 60)) as response:
                receipt.update(
                    http_status=response.status_code,
                    final_url=response.url,
                    headers=dict(response.headers),
                )
                response.raise_for_status()
                if "Content-Length" in response.headers:
                    assert int(response.headers["Content-Length"]) <= context["max_asset_bytes"], (
                        "Asset exceeds prospective cap"
                    )
                with partial.open("xb") as output:
                    for chunk in response.iter_content(1024**2):
                        if not chunk:
                            continue
                        receipt["size_bytes"] += len(chunk)
                        total += len(chunk)
                        assert (
                            receipt["size_bytes"] <= context["max_asset_bytes"] and total <= GIB
                        ), "Transfer cap exceeded"
                        assert shutil.disk_usage(REPO).free >= 2 * GIB + len(chunk), (
                            "Disk floor reached"
                        )
                        output.write(chunk)
                        sha.update(chunk)
            assert receipt["size_bytes"] > 0
            receipt["sha256"] = sha.hexdigest()
            partial.rename(path)
            receipt["status"] = "downloaded"
            archives.append(
                {
                    "filename": name,
                    "source_name": name,
                    **{key: receipt[key] for key in ("sha256", "size_bytes")},
                }
            )
        except Exception as exc:
            receipt["status"] = "failed"
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt["elapsed_seconds"] = time.perf_counter() - transfer_started
            receipt["ended_utc"] = datetime.now(UTC).isoformat()
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {k: receipt[k] for k in ("pool", "status", "size_bytes", "elapsed_seconds")}
                ),
                flush=True,
            )
        print(f"Completed {number}/24; cumulative transfer {total} bytes", flush=True)
    (ROOT / "reproduce/archive-receipts.json").write_text(json.dumps(archives, indent=2) + "\n")
    summary = {
        "n_archives": len(archives),
        "transfer_bytes": total,
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB,
        "final_disk_free_bytes": shutil.disk_usage(REPO).free,
        "original_receipts": [str(RAW / (a["filename"] + ".receipt.json")) for a in archives],
    }
    assert summary["peak_rss_gib"] <= 3
    (ROOT / "results/acquisition-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
