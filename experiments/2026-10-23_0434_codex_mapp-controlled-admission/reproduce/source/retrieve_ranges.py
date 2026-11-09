"""Recover prematurely terminated public downloads using verified bounded ranges."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/range-recovery.json").read_text())
ASSETS = Path(CONFIG["asset_root"])
START = time.perf_counter()


def fetch(job):
    item, index, first, last = job
    dest = ASSETS / (item["path"] + ".ranges") / f"{index:04d}.part"
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    result = {"project": item["path"].split("/")[0], "index": index,
              "first": first, "last": last, "attempts": []}
    for attempt in range(CONFIG["max_attempts"]):
        started = time.perf_counter()
        receipt = {"attempt": attempt + 1}
        try:
            request = urllib.request.Request(item["url"], headers={
                "Range": f"bytes={first}-{last}",
                "User-Agent": "Presto-source-admission/1.0",
            })
            with urllib.request.urlopen(request, timeout=60) as response:
                receipt.update(status=response.status, content_range=response.headers.get("Content-Range"))
                assert response.status == 206
                assert receipt["content_range"] == f"bytes {first}-{last}/{item['expected_size_bytes']}"
                data = response.read(last - first + 2)
            assert len(data) == last - first + 1
            with dest.open("xb") as stream:
                stream.write(data)
            receipt.update(passed=True, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        except Exception as error:
            receipt.update(passed=False, error=type(error).__name__ + ": " + str(error))
        receipt["wall_seconds"] = time.perf_counter() - started
        result["attempts"].append(receipt)
        if receipt["passed"]:
            result["passed"] = True
            return result
    result["passed"] = False
    return result


jobs = []
for item in CONFIG["assets"]:
    for index, first in enumerate(range(0, item["expected_size_bytes"], CONFIG["chunk_bytes"])):
        jobs.append((item, index, first, min(first + CONFIG["chunk_bytes"], item["expected_size_bytes"]) - 1))
receipts = []
with ThreadPoolExecutor(max_workers=CONFIG["workers"]) as executor:
    futures = [executor.submit(fetch, job) for job in jobs]
    for future in as_completed(futures):
        result = future.result()
        receipts.append(result)
        print(json.dumps(result), flush=True)
        (ROOT / "results/range-retrieval-chunks.json").write_text(json.dumps(receipts, indent=2) + "\n")
assert all(r["passed"] for r in receipts), "Incomplete ranges; originals and chunks retained"
assembled = []
for item in CONFIG["assets"]:
    dest = ASSETS / item["path"]
    temporary = dest.with_name(dest.name + ".assembled")
    assert not dest.exists() and not temporary.exists()
    sha1, sha256, size = hashlib.sha1(), hashlib.sha256(), 0
    with temporary.open("xb") as out:
        for path in sorted((ASSETS / (item["path"] + ".ranges")).glob("*.part")):
            with path.open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    out.write(data)
                    sha1.update(data)
                    sha256.update(data)
                    size += len(data)
    assert size == item["expected_size_bytes"]
    assert sha1.hexdigest() == item["expected_sha1"], "Assembled checksum differs from PRIDE"
    temporary.rename(dest)
    assembled.append(dict(item, passed=True, sha1=sha1.hexdigest(), sha256=sha256.hexdigest(), size_bytes=size))
    print(json.dumps(assembled[-1]), flush=True)
(ROOT / "results/range-retrieval.json").write_text(json.dumps({
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": time.perf_counter() - START, "assets": assembled,
}, indent=2) + "\n")
