#!/usr/bin/env python3
"""Verify and summarize the registered lineage/funnel preflight outputs."""

import csv
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    experiment_dir = Path(sys.argv[1])
    results_dir = experiment_dir / "results"
    lineage_path = results_dir / "class2_lineage.json"
    funnel_json_path = results_dir / "hitlist_preflight" / "data_funnel.json"
    funnel_csv_path = results_dir / "hitlist_preflight" / "data_funnel.csv"

    lineage = json.loads(lineage_path.read_text())
    funnel = json.loads(funnel_json_path.read_text())
    expected = {"binding": 4, "ms": 235}
    actual = funnel.get("drop_reasons", {}).get("unresolved_flank")
    if actual != expected:
        raise RuntimeError(f"Expected unresolved-flank drops {expected}, got {actual}")

    with funnel_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    csv_counts = {
        row["name"]: int(row["count"])
        for row in rows
        if row["kind"] == "drop_reasons" and row["stage"] == "unresolved_flank"
    }
    if csv_counts != expected:
        raise RuntimeError(f"Expected unresolved-flank CSV rows {expected}, got {csv_counts}")

    paths = (lineage_path, funnel_json_path, funnel_csv_path)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "class_ii_lineage_status": lineage["status"],
                "unresolved_flank_drops": actual,
                "artifacts": {
                    str(path.relative_to(experiment_dir)): {
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in paths
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
