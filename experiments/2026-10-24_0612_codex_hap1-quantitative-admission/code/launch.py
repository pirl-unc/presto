"""Record public-source retrieval, without curating a model-facing dataset."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("url")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("name must be one path component")
    experiment = Path(__file__).resolve().parents[1]
    config = json.loads((experiment / "reproduce/launch.json").read_text())
    root = Path(config["source_root"]) / "retrieval" / args.name
    root.mkdir(parents=True, exist_ok=False)
    frozen = experiment / "reproduce/source/launch.py"
    source = Path(__file__).read_bytes()
    if frozen.exists() and frozen.read_bytes() != source:
        raise RuntimeError("Launcher changed after its first invocation")
    frozen.write_bytes(source)
    started = time.monotonic()
    receipt = {
        "requested_url": args.url,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.executable,
        "platform": platform.platform(),
        "environment": {key: os.environ.get(key) for key in config["environment"]},
        "presto_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=experiment, text=True
        ).strip(),
        "launcher_sha256": hashlib.sha256(source).hexdigest(),
        "body_path": str(root / "body"),
        "purpose": "source-discovery-only; not a curated training observation",
    }
    try:
        with requests.get(args.url, timeout=(10, 30), stream=True) as response:
            receipt.update(
                status_code=response.status_code,
                final_url=response.url,
                content_type=response.headers.get("Content-Type"),
                response_headers=dict(response.headers),
                redirects=[{"url": r.url, "status": r.status_code} for r in response.history],
            )
            n_bytes = 0
            with (root / "body").open("wb") as output:
                for chunk in response.iter_content(65536):
                    n_bytes += len(chunk)
                    if n_bytes > 50_000_000:
                        raise ValueError("Source-discovery response exceeds 50 MB")
                    output.write(chunk)
            body = (root / "body").read_bytes()
            receipt.update(n_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    receipt["elapsed_seconds"] = time.monotonic() - started
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (experiment / f"results/retrieval-{args.name}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "response_headers"}, indent=2))
    if "error" in receipt or receipt.get("status_code", 999) >= 400:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
