"""Collect and verify the compact runtime-tree import-probe v2 receipt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal

FAMILY = Path(
    os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[3])
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(rows: list[dict]) -> dict:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def apply_diff(rows: list[dict], diff: dict) -> list[dict]:
    by_path = {row["path"]: row for row in rows}
    for row in diff["removed"]:
        if by_path.pop(row["path"]) != row:
            raise ValueError(f"removed row differs from baseline: {row['path']}")
    for change in diff["changed"]:
        if by_path[change["before"]["path"]] != change["before"]:
            raise ValueError(
                f"changed row differs from baseline: {change['before']['path']}"
            )
        by_path[change["after"]["path"]] = change["after"]
    for row in diff["added"]:
        if row["path"] in by_path:
            raise ValueError(f"added row already exists: {row['path']}")
        by_path[row["path"]] = row
    return [by_path[path] for path in sorted(by_path)]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    call = json.loads(
        (FAMILY / "reproduce/runtime-tree-import-probe-v2/call.json").read_text()
    )
    receipt = modal.FunctionCall.from_id(call["call_id"]).get(timeout=600)
    baseline_path = FAMILY / "results/runtime-tree-probe/receipt.json"
    if sha256(baseline_path) != receipt["baseline_receipt_sha256"]:
        raise ValueError("full baseline receipt differs from v2 contract")
    baseline = json.loads(baseline_path.read_text())
    if summary(baseline["rows"]) != receipt["before"]:
        raise ValueError("full baseline does not match v2 pre-import summary")
    reconstructed = apply_diff(baseline["rows"], receipt["diff"])
    if summary(reconstructed) != receipt["after"]:
        raise ValueError("canonical diff does not reconstruct post-import tree")

    results = FAMILY / "results/runtime-tree-import-probe-v2"
    results.mkdir(parents=True, exist_ok=False)
    write_json(results / "receipt.json", receipt)
    write_json(results / "reconstructed-after-rows.json", reconstructed)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
