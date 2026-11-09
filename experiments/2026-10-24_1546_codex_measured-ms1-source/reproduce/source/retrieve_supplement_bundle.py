"""Stream one public original supplement bundle under the prospective resource amendment."""

import datetime
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/measured_ms1_source_qualification/2026-10-24_1546_v1"
URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8184761/supplementaryFiles"
target = RAW / "article-2021-supplements-expanded.zip"
partial = RAW / "article-2021-supplements-expanded.zip.partial"
receipt_path = RAW / "article-2021-supplements-expanded.zip.receipt.json"
assert not any(p.exists() for p in [target, partial, receipt_path])
receipt = {
    "url": URL,
    "name": target.name,
    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "cap_bytes": 400 * 1024**2,
    "total_deadline_seconds": 180,
    "resource_amendment": "prospective plan October24 15:59UTC; no scientific eligibility changes",
}
started = time.perf_counter()
count = 0
digest = hashlib.sha256()
try:
    request = urllib.request.Request(URL, headers={"User-Agent": "Presto-public-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
        receipt.update(
            status=response.status,
            final_url=response.url,
            content_type=response.headers.get("Content-Type"),
        )
        while block := response.read(1024**2):
            count += len(block)
            digest.update(block)
            stream.write(block)
            if count > receipt["cap_bytes"]:
                raise ValueError("Expanded source-asset cap exceeded")
            if time.perf_counter() - started > 180:
                raise TimeoutError("Overall source request deadline exceeded")
    partial.rename(target)
    receipt.update(
        bytes=count,
        sha256=digest.hexdigest(),
        path=str(target),
        access_status="retrieved_not_yet_qualified",
    )
except (urllib.error.URLError, TimeoutError, ValueError) as error:
    receipt.update(
        error=str(error),
        partial_bytes=count,
        partial_sha256=digest.hexdigest(),
        partial_path=str(partial),
        access_status="failed_or_over_cap_not_absent",
    )
receipt["seconds"] = time.perf_counter() - started
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
if not target.exists():
    raise SystemExit(2)
