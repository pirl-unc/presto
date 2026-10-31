"""Offline receipt and publication-integrity replay; no source access or fitting."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["published_files"].items():
        data = (root / relative).read_bytes()
        assert len(data) == expected["bytes"], relative
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], relative
    ledger = json.loads((root / "results/discovery-ledger.json").read_text())
    operations = [op for call in ledger["calls"] for op in call["operations"]]
    assert operations == list(range(1, 13)), operations
    count = sum(
        len(call["request"].get("search_query", []))
        + len(call["request"].get("open", []))
        for call in ledger["calls"]
    )
    assert count == len(operations) <= ledger["operation_limit"]
    receipt_names = {call["receipt_name"] for call in ledger["calls"]}
    evidence = json.loads((root / "results/evidence.json").read_text())
    assert len(evidence["candidates"]) == 2
    for candidate in evidence["candidates"]:
        assert set(candidate["evidence_receipts"]) <= receipt_names
        assert candidate["decision"] in {
            "excluded_for_this_endpoint",
            "not_qualified_within_screen",
        }
    raw_root = root.parents[1] / ledger["local_receipts_directory"]
    present = 0
    for call in ledger["calls"]:
        path = raw_root / call["receipt_name"]
        if path.exists():
            data = path.read_bytes()
            assert len(data) == call["receipt_bytes"]
            assert hashlib.sha256(data).hexdigest() == call["receipt_sha256"]
            original = json.loads(data)
            assert original["request"] == call["request"]
            assert original["operations"] == call["operations"]
            present += 1
    print(
        json.dumps(
            {
                "published_files_verified": len(manifest["published_files"]),
                "scientific_network_operations": count,
                "candidate_decisions": len(evidence["candidates"]),
                "local_raw_receipts_verified": present,
                "local_raw_receipts_unavailable": len(ledger["calls"]) - present,
                "biological_validation": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
