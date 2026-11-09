"""Collect the full CPU runtime-tree probe receipt."""

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
    call = json.loads((FAMILY / "reproduce/runtime-tree-probe/call.json").read_text())
    compact = modal.FunctionCall.from_id(call["call_id"]).get(timeout=600)
    artifact = compact["receipt"]
    results = FAMILY / "results/runtime-tree-probe"
    results.mkdir(parents=True, exist_ok=True)
    receipt_path = results / "receipt.json"
    with receipt_path.open("wb") as stream:
        VOLUME.read_file_into_fileobj(artifact["path"], stream)
    if (
        receipt_path.stat().st_size != artifact["bytes"]
        or sha256(receipt_path) != artifact["sha256"]
    ):
        raise ValueError("collected runtime-tree receipt differs from remote artifact")
    (results / "transport-receipt.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
