"""Stream public processed result files without retaining their bytes in RAM."""

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests


def fetch(item, root):
    path = root / item["name"]
    receipt_path = root / (item["name"] + ".receipt.json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(path)
    started = time.time()
    receipt = {"url": item["url"], "name": item["name"], "started_unix": started}
    digest, legacy_digest, n_bytes = hashlib.sha256(), hashlib.sha1(), 0
    partial = path.with_name(path.name + ".partial")
    try:
        with requests.get(item["url"], timeout=(15, 45), stream=True) as response:
            receipt.update(
                status=response.status_code,
                final_url=response.url,
                content_type=response.headers.get("Content-Type"),
            )
            response.raise_for_status()
            with partial.open("xb") as out:
                for part in response.iter_content(1024 * 1024):
                    n_bytes += len(part)
                    if n_bytes > item["max_bytes"]:
                        raise ValueError("Source exceeds declared byte limit")
                    out.write(part)
                    digest.update(part)
                    legacy_digest.update(part)
            receipt.update(
                n_bytes=n_bytes,
                sha256=digest.hexdigest(),
                sha1=legacy_digest.hexdigest(),
            )
            if item.get("expected_sha1") and receipt["sha1"] != item["expected_sha1"]:
                raise ValueError("PRIDE SHA1 mismatch")
            partial.rename(path)
            receipt["verified"] = True
    except (requests.RequestException, ValueError) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["verified"] = False
    receipt["seconds"] = time.time() - started
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = json.loads(args.manifest.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda item: fetch(item, args.output), items):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
