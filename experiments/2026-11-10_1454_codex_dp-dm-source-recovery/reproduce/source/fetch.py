"""Fetch a frozen list of public source documents, retaining successes/failures."""

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
    try:
        with requests.get(item["url"], timeout=(15, 45), stream=True) as response:
            receipt.update(
                status=response.status_code,
                final_url=response.url,
                content_type=response.headers.get("Content-Type"),
            )
            data = bytearray()
            for part in response.iter_content(128 * 1024):
                data.extend(part)
                if len(data) > item.get("max_bytes", 25 * 1024 * 1024):
                    raise ValueError("Source request exceeded declared size limit")
            path.write_bytes(data)
            receipt.update(n_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    except (requests.RequestException, ValueError) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    receipt["seconds"] = time.time() - started
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = json.loads(args.manifest.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda item: fetch(item, args.output), items))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
