"""Bounded public downloads with strict byte-range validation and receipts."""

import argparse
import datetime
import hashlib
import json
import resource
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
RAW = Path(CONFIG["raw_directory"])
LIMITS = CONFIG["limits"]

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--size", type=int, required=True)
parser.add_argument("--sha1")
parser.add_argument("--range", dest="byte_range")
parser.add_argument("--object-size", type=int)
args = parser.parse_args()
assert Path(args.name).name == args.name
assert 0 <= args.size <= LIMITS["asset_bytes"]
target = RAW / args.name
receipt_path = RAW / (args.name + ".receipt.json")
assert not target.exists() and not receipt_path.exists()
prior_seconds = sum(json.loads(p.read_text())["seconds"] for p in RAW.glob("*.receipt.json"))
started = time.monotonic()
receipt = {
    "argv": sys.argv,
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "url": args.url,
    "expected_size_bytes": args.size,
    "expected_sha1": args.sha1,
    "requested_range": args.byte_range,
    "access_status": "failed_not_absent",
}


def guard(next_bytes=0):
    if time.monotonic() - started > LIMITS["request_seconds"]:
        raise RuntimeError("Request wall cap; retain partial bytes")
    if prior_seconds + time.monotonic() - started > LIMITS["acquisition_seconds"]:
        raise RuntimeError("Family acquisition time cap")
    if shutil.disk_usage(RAW).free < LIMITS["disk_floor_bytes"] + LIMITS["start_margin_bytes"] + next_bytes:
        raise RuntimeError("Unchanged free-space reserve")
    total = sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file())
    if total + next_bytes > LIMITS["total_raw_bytes"]:
        raise RuntimeError("Family raw-byte cap")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > LIMITS["rss_bytes"]:
        raise RuntimeError("Local macOS RSS cap")


try:
    guard(args.size)
    headers = {"User-Agent": "Presto-source-qualification/1.0", "Accept-Encoding": "identity"}
    if args.byte_range:
        start, end = map(int, args.byte_range.split(":"))
        assert args.object_size and 0 <= start <= end < args.object_size
        assert end - start + 1 == args.size
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(args.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=LIMITS["request_seconds"])
    except urllib.error.HTTPError as error:
        response = error
    with response:
        receipt.update(status=response.status, final_url=response.url,
                       headers={k: v for k, v in response.headers.items() if k.lower() != "set-cookie"})
        if args.byte_range:
            expected = f"bytes {start}-{end}/{args.object_size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Server did not honor exact requested byte range; body not downloaded")
        elif response.status != 200:
            raise RuntimeError(f"Unexpected source HTTP status {response.status}; body not admitted")
        if response.headers.get("Content-Length") is not None:
            if int(response.headers["Content-Length"]) != args.size:
                raise RuntimeError("Response Content-Length differs from expected source length")
        received = 0
        with target.open("xb") as output:
            while True:
                guard()
                block = response.read(min(1024**2, args.size - received + 1))
                if not block:
                    break
                guard(len(block))
                if received + len(block) > args.size:
                    raise RuntimeError("Body exceeds expected size; extra bytes not written")
                output.write(block)
                received += len(block)
        if received != args.size:
            raise RuntimeError("Incomplete source body")
        receipt["body_complete"] = True
        if args.sha1 and hashlib.sha1(target.read_bytes()).hexdigest() != args.sha1:
            raise RuntimeError("Deposited SHA1 mismatch")
        receipt["access_status"] = "retrieved_and_length_verified"
except (OSError, RuntimeError, ValueError, AssertionError) as error:
    receipt["error"] = repr(error)
if target.exists():
    receipt.update(path=str(target), size_bytes=target.stat().st_size,
                   sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   sha1=hashlib.sha1(target.read_bytes()).hexdigest())
receipt["seconds"] = time.monotonic() - started
receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k not in ("headers", "argv")}), flush=True)
raise SystemExit(0 if receipt["access_status"] == "retrieved_and_length_verified" else 1)
