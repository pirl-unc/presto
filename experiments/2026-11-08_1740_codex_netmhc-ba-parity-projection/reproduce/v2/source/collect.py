"""Collect lightweight v2 continuation receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import modal


FAMILY = Path(os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[3]))
VOLUME = modal.Volume.from_name("presto-checkpoints")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    name = "verify" if args.verify else "projection"
    call = json.loads(
        (FAMILY / "reproduce/v2" / ("verify-call.json" if args.verify else "call.json")).read_text()
    )
    receipt = modal.FunctionCall.from_id(call["call_id"]).get(timeout=3600)
    results = FAMILY / "results/v2"
    results.mkdir(parents=True, exist_ok=True)
    artifact = receipt["verification"] if args.verify else receipt["summary"]
    artifact_name = "verification.json" if args.verify else "summary.json"
    artifact_path = results / artifact_name
    with artifact_path.open("wb") as stream:
        VOLUME.read_file_into_fileobj(artifact["path"], stream)
    if artifact_path.stat().st_size != artifact["bytes"] or sha256(artifact_path) != artifact[
        "sha256"
    ]:
        raise ValueError(f"collected {artifact_name} differs from remote receipt")
    (results / f"{name}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
