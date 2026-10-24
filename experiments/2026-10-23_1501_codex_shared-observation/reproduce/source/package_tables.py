"""Preserve original CSV bytes and normalize text copies for Git review."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]


def main():
    receipt = []
    for name in (
        "comparison.csv",
        "within-condition.csv",
        "within-condition-time-ranks.csv",
    ):
        path = FAMILY / "results" / name
        original = path.read_bytes()
        normalized = original.replace(b"\r\n", b"\n")
        assert list(csv.reader(io.StringIO(original.decode()))) == list(
            csv.reader(io.StringIO(normalized.decode()))
        )
        archive = path.with_suffix(".original.csv.gz")
        assert not archive.exists()
        archive.write_bytes(gzip.compress(original, mtime=0))
        assert gzip.decompress(archive.read_bytes()) == original
        path.write_bytes(normalized)
        receipt.append(
            {
                "path": str(path.relative_to(FAMILY)),
                "original_archive": str(archive.relative_to(FAMILY)),
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
                "parsed_rows_identical": True,
            }
        )
    (FAMILY / "results/table-packaging.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
