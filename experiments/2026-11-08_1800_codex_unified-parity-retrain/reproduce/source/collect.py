"""Collect lightweight preflight receipts without copying source data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import modal


FAMILY = Path(os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    stem = "preflight-verification" if args.verify else "preflight"
    call = json.loads((FAMILY / "reproduce" / f"{stem}-call.json").read_text())
    receipt = modal.FunctionCall.from_id(call["call_id"]).get(timeout=7200)
    results = FAMILY / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{stem}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
