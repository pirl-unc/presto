"""Bounded public-source retrieval; retain failed access and original bytes."""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
RAW = Path(CONFIG["raw_directory"])
LIMITS = CONFIG["limits"]


def fetch(url, name):
    assert Path(name).name == name
    target = RAW / name
    receipt_path = RAW / (name + ".receipt.json")
    assert not target.exists() and not receipt_path.exists(), "Preserve previous attempts"
    started = time.monotonic()
    previous_seconds = sum(json.loads(p.read_text())["seconds"] for p in RAW.glob("*.receipt.json"))
    receipt = {"url": url, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "argv": sys.argv, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Presto-source-audit/1.0"})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status = getattr(response, "status", None)
            receipt.update(
                status=status,
                final_url=response.url,
                headers={k: v for k, v in response.headers.items() if k.lower() != "set-cookie"},
            )
            received = 0
            with target.open("wb") as stream:
                while True:
                    if shutil.disk_usage(RAW).free < LIMITS["disk_floor_bytes"] + 32 * 1024**2:
                        raise RuntimeError("Free disk floor plus margin")
                    if time.monotonic() - started > 30:
                        raise RuntimeError("Per-request wall cap; partial bytes retained")
                    chunk = response.read(1024**2)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > LIMITS["asset_bytes"]:
                        raise RuntimeError("Per-asset cap; partial bytes retained")
                    if (
                        sum(p.stat().st_size for p in RAW.iterdir() if p.is_file()) + len(chunk)
                        > LIMITS["total_raw_bytes"]
                    ):
                        raise RuntimeError("Family raw-byte cap; partial bytes retained")
                    if (
                        previous_seconds + time.monotonic() - started
                        > LIMITS["acquisition_seconds"]
                    ):
                        raise RuntimeError("Acquisition time cap; partial bytes retained")
                    stream.write(chunk)
            receipt["body_complete"] = True
            receipt["access_status"] = (
                "retrieved" if status is None or status < 400 else "failed_not_absent"
            )
    except (OSError, RuntimeError, ValueError) as error:
        receipt.update(error=repr(error), access_status="failed_not_absent")
    if target.exists():
        receipt.update(
            path=str(target),
            size_bytes=target.stat().st_size,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )
    receipt["seconds"] = time.monotonic() - started
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if receipt['access_status'] != 'retrieved':
        raise SystemExit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--name", required=True)
args = parser.parse_args()
fetch(args.url, args.name)
