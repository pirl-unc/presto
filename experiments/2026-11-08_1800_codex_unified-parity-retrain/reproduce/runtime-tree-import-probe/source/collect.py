"""Collect the exact inline runtime-tree import-probe receipt."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

FAMILY = Path(
    os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[3])
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    call = json.loads(
        (FAMILY / "reproduce/runtime-tree-import-probe/call.json").read_text()
    )
    receipt = modal.FunctionCall.from_id(call["call_id"]).get(timeout=600)
    results = FAMILY / "results/runtime-tree-import-probe"
    results.mkdir(parents=True, exist_ok=False)
    write_json(results / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "before": receipt["before"]["summary"],
                "after": receipt["after"]["summary"],
                "diff": receipt["diff"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
