"""Resume identical public assets using PRIDE's documented alternate download service."""

import hashlib
import json
import resource
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from hitlist.proteometools_ms1 import BASE_URL, CORE
from hitlist.provenance import file_digest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = REPO / "data/measured_ms1_grouped_observer/2026-10-24_1702_v1"
GIB = 1024**3
ALTERNATIVE = "https://www.ebi.ac.uk/pride/ws/archive-file-downloader/files/s3/PXD021013/"


def main():
    started = time.perf_counter()
    assert not (ROOT / "reproduce/archive-receipts.json").exists()
    selection = json.loads((ROOT / "results/selection.json").read_text())
    assert len(selection["selected"]) == 24
    shutil.copy2(__file__, ROOT / "reproduce/source/resume_remaining.py")
    for spec in json.loads(CORE.read_text())["assets"]:
        assert file_digest(RAW / spec["filename"]) == {k: spec[k] for k in ("sha256", "size_bytes")}
    initial_receipts = [json.loads(p.read_text()) for p in RAW.glob("*.receipt.json")]
    total = sum(row["size_bytes"] for row in initial_receipts)
    session = requests.Session()
    session.headers["User-Agent"] = "Presto-public-source-qualification/1.0"
    archives = []
    for number, pool in enumerate(selection["selected"], 1):
        name = pool["source_name"]
        path = RAW / name
        original_receipt = RAW / (name + ".receipt.json")
        previous = json.loads(original_receipt.read_text()) if original_receipt.exists() else None
        if path.exists():
            successes = [
                json.loads(p.read_text())
                for p in RAW.glob(name + "*.receipt.json")
                if json.loads(p.read_text()).get("status") == "downloaded"
            ]
            assert successes
            assert all(
                file_digest(path) == {k: record[k] for k in ("sha256", "size_bytes")}
                for record in successes
            )
            print(f"Verified existing {number}/24: {name}", flush=True)
        else:
            urls = [ALTERNATIVE + name] if previous else [BASE_URL + name, ALTERNATIVE + name]
            succeeded = False
            for attempt, url in enumerate(urls, 2 if previous else 1):
                receipt_path = RAW / f"{name}.remaining-{attempt}.receipt.json"
                partial = RAW / f"{name}.remaining-{attempt}.partial"
                assert not receipt_path.exists() and not partial.exists()
                receipt = {
                    "pool": pool["pool"],
                    "source_url": url,
                    "deposited_url": BASE_URL + name,
                    "started_utc": datetime.now(UTC).isoformat(),
                    "status": "started",
                    "size_bytes": 0,
                }
                tick = time.perf_counter()
                sha = hashlib.sha256()
                try:
                    with session.get(url, stream=True, timeout=(20, 60)) as response:
                        receipt.update(
                            http_status=response.status_code,
                            final_url=response.url,
                            headers=dict(response.headers),
                        )
                        response.raise_for_status()
                        if "Content-Length" in response.headers:
                            assert int(response.headers["Content-Length"]) <= 128 * 1024**2, (
                                "Asset cap exceeded"
                            )
                        with partial.open("xb") as output:
                            for chunk in response.iter_content(1024**2):
                                if not chunk:
                                    continue
                                receipt["size_bytes"] += len(chunk)
                                total += len(chunk)
                                assert receipt["size_bytes"] <= 128 * 1024**2 and total <= GIB, (
                                    "Transfer cap exceeded"
                                )
                                assert shutil.disk_usage(REPO).free >= 2 * GIB + len(chunk), (
                                    "Disk floor reached"
                                )
                                output.write(chunk)
                                sha.update(chunk)
                    assert receipt["size_bytes"] > 0
                    receipt["sha256"] = sha.hexdigest()
                    partial.rename(path)
                    receipt["status"] = "downloaded"
                    succeeded = True
                except requests.RequestException as exc:
                    receipt["status"] = "failed"
                    receipt["error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    receipt["status"] = "failed"
                    receipt["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    receipt["elapsed_seconds"] = time.perf_counter() - tick
                    receipt["ended_utc"] = datetime.now(UTC).isoformat()
                    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
                    print(
                        json.dumps(
                            {
                                k: receipt[k]
                                for k in (
                                    "pool",
                                    "source_url",
                                    "status",
                                    "size_bytes",
                                    "elapsed_seconds",
                                )
                            }
                        ),
                        flush=True,
                    )
                if succeeded:
                    break
            assert succeeded, f"Official source routes unavailable for frozen pool {pool['pool']}"
        archives.append({"filename": name, "source_name": name, **file_digest(path)})
    (ROOT / "reproduce/archive-receipts.json").write_text(json.dumps(archives, indent=2) + "\n")
    report = {
        "n_archives": len(archives),
        "transfer_bytes_including_prior_attempts": total,
        "seconds_this_resume": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB,
        "argv": sys.argv,
        "python": sys.executable,
        "selection": file_digest(ROOT / "results/selection.json"),
        "original_receipts": [str(p) for p in sorted(RAW.glob("*.receipt.json"))],
    }
    assert report["peak_rss_gib"] <= 3
    (ROOT / "results/acquisition-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
