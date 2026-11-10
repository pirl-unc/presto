"""CPU-only recovery of public original confidence tables; no training."""

import json
from pathlib import Path

import modal

app = modal.App("presto-dp-dm-native-source-20261110")
volume = modal.Volume.from_name("presto-checkpoints", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install("requests==2.32.5")
PREFIX = "2026-11-10_1454_codex_dp-dm-source-recovery/native-v1"


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=900,
    max_containers=2,
    region="eu-west-1",
    volumes={"/vol": volume},
)
def recover(item, queries=None):
    import hashlib
    import sqlite3
    import time
    import requests

    started = time.time()
    root = Path("/vol") / PREFIX
    root.mkdir(parents=True, exist_ok=True)
    path = root / item["name"]
    receipt_path = root / (item["name"] + ".receipt.json")
    if not path.exists():
        partial = path.with_name(path.name + ".partial")
        if partial.exists():
            raise FileExistsError(partial)
        h, h1, n = hashlib.sha256(), hashlib.sha1(), 0
        with requests.get(item["url"], stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with partial.open("xb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    n += len(chunk)
                    if n > item["max_bytes"]:
                        raise ValueError("Source byte limit exceeded")
                    out.write(chunk)
                    h.update(chunk)
                    h1.update(chunk)
        if h1.hexdigest() != item["expected_sha1"]:
            raise ValueError("PRIDE source SHA1 mismatch")
        partial.rename(path)
        receipt = {
            "source": item,
            "n_bytes": n,
            "sha256": h.hexdigest(),
            "sha1": h1.hexdigest(),
            "download_seconds": time.time() - started,
            "gpu": None,
            "cpu": 1,
            "region": "eu-west-1",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    else:
        receipt = json.loads(receipt_path.read_text())
        if receipt["source"] != item or path.stat().st_size != receipt["n_bytes"]:
            raise ValueError("Native source cache mismatch")
        h1 = hashlib.sha1()
        with path.open("rb") as src:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                h1.update(chunk)
        if h1.hexdigest() != item["expected_sha1"]:
            raise ValueError("Cached source checksum mismatch")
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    tables = [
        r[0]
        for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    result = {
        "receipt": receipt,
        "tables": tables,
        "schema": {
            name: [
                list(r)
                for r in connection.execute(
                    'PRAGMA table_info("' + name.replace('"', '""') + '")'
                )
            ]
            for name in tables
        },
    }
    if queries:
        result["queries"] = {}
        for name, sql in queries.items():
            cursor = connection.execute(sql)
            result["queries"][name] = {
                "columns": [r[0] for r in cursor.description],
                "rows": [list(r) for r in cursor],
            }
    connection.close()
    result["seconds"] = time.time() - started
    return result


@app.local_entrypoint()
def main(manifest: str, receipt: str):
    items = json.loads(Path(manifest).read_text())
    if len(items) != 1:
        raise ValueError("Pilot must contain exactly one native file")
    call = recover.spawn(items[0])
    Path(receipt).write_text(
        json.dumps(
            {
                "call_id": call.object_id,
                "app_id": app.app_id,
                "source": items[0],
                "gpu": None,
                "cpu": 1,
                "memory_mib": 2048,
                "region": "eu-west-1",
            },
            indent=2,
        )
        + "\n"
    )
    print(call.object_id, flush=True)
