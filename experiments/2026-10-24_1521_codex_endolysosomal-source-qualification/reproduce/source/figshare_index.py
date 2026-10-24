"""Read-only deposit-index query used by the publisher's public article UI."""

import datetime
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/endolysosomal_source_qualification/2026-10-24_1521_oztemiz-v1"
url = "https://www.frontiersin.org/api/v4/articles/figshare"
body = {"doi": "10.3389/falgy.2026.1936335"}
request = urllib.request.Request(
    url,
    data=json.dumps(body).encode(),
    headers={
        "Content-Type": "application/json",
        "User-Agent": "Presto-public-source-qualification/1.0",
    },
)
started = time.perf_counter()
with urllib.request.urlopen(request, timeout=60) as response:
    payload = response.read()
    receipt = {
        "url": url,
        "method": "POST",
        "purpose": "read-only deposit metadata query",
        "body": body,
        "status": response.status,
        "final_url": response.url,
        "headers": dict(response.headers),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
    }
assert not (RAW / "figshare-index.json").exists()
(RAW / "figshare-index.json").write_bytes(payload)
(RAW / "figshare-index.json.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(payload.decode(), flush=True)
