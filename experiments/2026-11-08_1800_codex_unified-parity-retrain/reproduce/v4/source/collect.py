"""Collect and verify the atomic independent preflight receipt."""

from __future__ import annotations

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
    call = json.loads((FAMILY / "reproduce/v4/preflight-v2-verification-call.json").read_text())
    compact = modal.FunctionCall.from_id(call["call_id"]).get(timeout=7200)
    artifact = compact["receipt"]
    results = FAMILY / "results/v4"
    results.mkdir(parents=True, exist_ok=True)
    receipt_path = results / "preflight-v2-verification-r2-receipt.json"
    with receipt_path.open("wb") as stream:
        VOLUME.read_file_into_fileobj(artifact["path"], stream)
    if (
        receipt_path.stat().st_size != artifact["bytes"]
        or sha256(receipt_path) != artifact["sha256"]
    ):
        raise ValueError("collected verification receipt differs from remote artifact")
    (results / "transport-receipt.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
