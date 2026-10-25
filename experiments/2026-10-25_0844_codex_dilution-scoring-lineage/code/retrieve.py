"""Stream the preregistered processed scoring/library assets, preserving failures."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

P = Path(__file__).resolve().parents[1]
C = json.loads((P / "reproduce/retrieve-launch.json").read_text())
D = Path(C["source_directory"])
D.mkdir(parents=True, exist_ok=True)
receipt_path = P / "results/retrieval.json"
assert not receipt_path.exists(), "Never overwrite a completed/failed retrieval"
assert sum(x["expected_bytes"] for x in C["selected_assets"]) <= 512 * 1024**2
records = []
started = time.monotonic()
try:
    for asset in C["selected_assets"]:
        target = D / asset["name"]
        partial = target.with_name(target.name + ".partial")
        assert not target.exists() and not partial.exists(), "Explicit recovery required"
        row = {**asset, "started_utc": datetime.now(timezone.utc).isoformat()}
        begin, size, digest = time.monotonic(), 0, hashlib.sha256()
        records.append(row)
        try:
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "Presto-public-source-audit/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
                row["http_status"] = response.status
                row["final_url"] = response.url
                row["headers"] = dict(response.headers)
                assert response.status == 200
                while block := response.read(1024 * 1024):
                    size += len(block)
                    assert size <= asset["expected_bytes"], "Declared size exceeded"
                    digest.update(block)
                    stream.write(block)
            assert size == asset["expected_bytes"], "Incomplete source asset"
            partial.rename(target)
            row["passed"] = True
        except Exception as exc:
            row["passed"] = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            raise
        finally:
            row["received_bytes"] = size
            row["sha256"] = digest.hexdigest()
            row["elapsed_seconds"] = time.monotonic() - begin
            (P / "results/retrieval-progress.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in ["name", "received_bytes", "passed", "elapsed_seconds"]}), flush=True)
finally:
    receipt_path.write_text(json.dumps({"passed": len(records) == len(C["selected_assets"]) and all(r.get("passed") for r in records), "records": records, "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
