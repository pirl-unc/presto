"""Preserve a public primary-source HTTP response and its retrieval receipt."""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    start = time.perf_counter()
    request = Request(args.url, headers={"User-Agent": "Presto source qualification/1.0"})
    try:
        response = urlopen(request, timeout=30)
    except HTTPError as error:
        response = error
    with response:
        body = response.read()
        receipt = {
            "url": args.url,
            "resolved_url": response.url,
            "status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "utc": datetime.now(UTC).isoformat(),
            "argv": sys.argv,
            "elapsed_seconds": time.perf_counter() - start,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body)
    args.out.with_suffix(args.out.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
