"""Bounded original metadata retrieval; no quantitative model or data selection."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parents[1]
C = json.loads((P / "reproduce/processed-retrieval.json").read_text())
D = Path(C["source_directory"])
RESULT = P / "results/processed-retrieval.json"
assert not RESULT.exists(), "Never overwrite a completed retrieval receipt"
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == C["source_sha256"]
assert shutil.disk_usage(P).free >= 1073741824 + 33554432
D.mkdir(parents=True, exist_ok=True)


def timed_out(signum, frame):
    raise TimeoutError("Metadata retrieval exceeded the fixed 240-second budget")


signal.signal(signal.SIGALRM, timed_out)
signal.alarm(240)
start = time.monotonic()
records = []
failed = False
try:
    for asset in C["metadata_assets"]:
        target = D / asset["name"]
        assert not target.exists(), target
        row = {**asset, "started_utc": datetime.now(timezone.utc).isoformat()}
        t = time.monotonic()
        try:
            request = urllib.request.Request(
                asset["url"], headers={"User-Agent": "Presto-public-source-audit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                row["status"] = response.status
                row["final_url"] = response.url
                row["headers"] = dict(response.headers)
                assert response.status == 200
                payload = response.read(C["max_bytes_per_asset"] + 1)
                assert len(payload) <= C["max_bytes_per_asset"], "Asset exceeds fixed cap"
            # Preserve returned bytes even if semantic/format validation fails.
            target.write_bytes(payload)
            row["bytes"] = len(payload)
            row["sha256"] = hashlib.sha256(payload).hexdigest()
            if "expected_bytes" in asset:
                assert len(payload) == asset["expected_bytes"], "Source size mismatch"
            if asset["kind"] == "csv":
                header = payload.decode("utf-8-sig").splitlines()[0]
                assert "," in header, "Expected a comma-separated header"
                row["header"] = header
            elif asset["kind"] == "json":
                parsed = json.loads(payload)
                row["json_root_type"] = type(parsed).__name__
            else:
                parsed = ET.fromstring(payload)
                assert parsed.tag == "article", parsed.tag
                row["xml_root"] = parsed.tag
            row["passed"] = True
        except Exception as exc:
            row["passed"] = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            failed = True
        row["elapsed_seconds"] = time.monotonic() - t
        records.append(row)
        print(json.dumps({k: row[k] for k in ("name", "passed", "elapsed_seconds")}), flush=True)
        (P / "results/processed-retrieval-progress.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        if time.monotonic() - start >= 240:
            break
finally:
    signal.alarm(0)
    receipt = {
        "passed": not failed and len(records) == len(C["metadata_assets"]),
        "assets": records,
        "elapsed_seconds": time.monotonic() - start,
        "os_self_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "Bounded public metadata retrieval only; no model, source effects or predictions",
    }
    RESULT.write_text(json.dumps(receipt, indent=2) + "\n")
sys.exit(0 if receipt["passed"] else 1)
