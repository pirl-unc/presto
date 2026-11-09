"""Collect only lightweight receipts for the BA projection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal


FAMILY = Path(os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[2])).resolve()
VOLUME = modal.Volume.from_name("presto-checkpoints")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(call_name, output_name, *, timeout):
    call = json.loads((FAMILY / "reproduce" / call_name).read_text())
    receipt = modal.FunctionCall.from_id(call["call_id"]).get(timeout=timeout)
    results = FAMILY / "results"
    results.mkdir(parents=True, exist_ok=True)
    if "summary_bytes" in receipt:
        target = results / "summary.json"
        with target.open("wb") as stream:
            VOLUME.read_file_into_fileobj(receipt["summary_path"], stream)
        assert target.stat().st_size == receipt["summary_bytes"]
        assert sha256(target) == receipt["summary_sha256"]
    if "verification" in receipt:
        target = results / "verification.json"
        with target.open("wb") as stream:
            VOLUME.read_file_into_fileobj(receipt["verification"]["path"], stream)
        assert target.stat().st_size == receipt["verification"]["bytes"]
        assert sha256(target) == receipt["verification"]["sha256"]
    (results / output_name).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    collect(
        "verify-call.json" if args.verify else "call.json",
        "verification-receipt.json" if args.verify else "receipt.json",
        timeout=1800 if args.verify else 3600,
    )
