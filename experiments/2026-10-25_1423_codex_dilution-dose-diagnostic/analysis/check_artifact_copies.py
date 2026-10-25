"""Post-fit bookkeeping: copied source fields, CSV cells and frozen-source integrity.

This adds no model, selection rule, fit, metric or numerical experiment.
"""

import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    config = json.loads((root / "reproduce/launch.json").read_text())
    for relative, expected in config["frozen_code_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    assert (
        hashlib.sha256((root / "reproduce/plan.md").read_bytes()).hexdigest()
        == config["plan_sha256"]
    )
    originals = {
        row["candidate_id"]: row for row in json.loads((results / "admitted.json").read_text())
    }
    n_predictions, n_copied_fields = 0, 0
    for name in ("development-predictions.json", "descriptive-full-predictions.json"):
        for row in json.loads((results / name).read_text()):
            for key, value in originals[row["candidate_id"]].items():
                assert row[key] == value, (name, row["candidate_id"], key)
                n_copied_fields += 1
            n_predictions += 1
    summary = json.loads((results / "summary.json").read_text())
    n_csv_cells = 0
    with (results / "comparison.csv").open() as stream:
        for row in csv.DictReader(stream):
            for key, value in summary["by_arm"][row["arm"]]["overall"].items():
                if isinstance(value, (int, float)):
                    assert float(row[key]) == value
                elif value is None:
                    assert row[key] == ""
                else:
                    assert row[key] == value
                n_csv_cells += 1
    receipt = {
        "passed": True,
        "frozen_file_hashes": len(config["frozen_code_sha256"]),
        "prediction_payloads": n_predictions,
        "copied_source_fields": n_copied_fields,
        "summary_csv_cells": n_csv_cells,
        "scope": "post-fit artifact bookkeeping only; no added scientific analysis",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (results / "artifact-copy-verification.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
