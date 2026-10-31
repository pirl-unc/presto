"""Stream two registered processed archives and the public author manuscript."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/processed-assets.json").read_text())
ASSETS = Path(CONFIG["asset_root"])


def retrieve(item):
    start = time.perf_counter()
    result = dict(item, started_utc=datetime.now(timezone.utc).isoformat())
    dest = ASSETS / item["path"]
    temporary = dest.with_name(dest.name + ".partial")
    try:
        if dest.exists() or temporary.exists():
            raise FileExistsError(str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        sha256, sha1 = hashlib.sha256(), hashlib.sha1()
        size = 0
        request = urllib.request.Request(
            item["url"], headers={"User-Agent": "Presto-source-admission/1.0"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            result.update(
                http_status=response.status,
                content_type=response.headers.get("Content-Type"),
                final_url=response.url,
            )
            with temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    sha256.update(chunk)
                    sha1.update(chunk)
                    size += len(chunk)
        result.update(size_bytes=size, sha256=sha256.hexdigest(), sha1=sha1.hexdigest())
        if "expected_size_bytes" in item:
            assert size == item["expected_size_bytes"], "PRIDE byte count differs"
        if "expected_sha1" in item:
            assert sha1.hexdigest() == item["expected_sha1"], "PRIDE checksum differs"
        with temporary.open("rb") as stream:
            magic = stream.read(8)
        assert magic.startswith(bytes.fromhex(item["magic_hex"])), "Unexpected file format"
        temporary.rename(dest)
        result["passed"] = True
    except Exception as error:
        result.update(passed=False, error=type(error).__name__ + ": " + str(error))
    result["wall_seconds"] = time.perf_counter() - start
    print(json.dumps(result), flush=True)
    return result


results = []
with ThreadPoolExecutor(max_workers=2) as executor:
    for result in executor.map(retrieve, CONFIG["assets"]):
        results.append(result)
        (ROOT / "results/processed-retrieval.json").write_text(json.dumps(results, indent=2) + "\n")
assert all(result["passed"] for result in results), "One or more registered assets failed"
