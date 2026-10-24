"""Retrieve the public primary XML through the official Europe PMC API."""

import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/apc_composition_admission/2026-10-24_1755_v1"
URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12616400/fullTextXML"
started = time.perf_counter()
RAW.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, ROOT / "reproduce/source/fetch_primary.py")
receipt = {"source_url": URL, "started_utc": datetime.now(UTC).isoformat()}
try:
    response = requests.get(URL, timeout=(15, 45))
    receipt.update(
        status_code=response.status_code,
        headers=dict(response.headers),
        final_url=response.url,
    )
    response.raise_for_status()
    assert len(response.content) < 10 * 1024**2
    (RAW / "article.xml").write_bytes(response.content)
    receipt.update(
        status="downloaded",
        size_bytes=len(response.content),
        sha256=hashlib.sha256(response.content).hexdigest(),
    )
except Exception as exc:
    receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    receipt["seconds"] = time.perf_counter() - started
    (ROOT / "results/primary-retrieval.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
