"""Reconcile descriptor recovery while requiring identical selector populations."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(before, after):
    rows = []
    for selector in ("panel", "bootstrap"):
        old, new = before[selector], after[selector]
        for name in (
            "selector",
            "observed_records",
            "retained_records",
            "stats",
            "non_descriptor_payload_sha256",
            "retained_non_descriptor_sha256",
            "source_descriptors_sha256",
            "expected_columns",
        ):
            assert old[name] == new[name], f"Changed {selector} {name}"
        assert new["actual_columns"] == new["expected_columns"], f"Wrong columns: {selector}"
        assert len(old["field_counts"]) == len(new["field_counts"]) == 4
        for a, b in zip(old["field_counts"], new["field_counts"], strict=True):
            for name in ("selector", "field", "records", "source_present"):
                assert a[name] == b[name], f"Changed field population: {name}"
            assert b["matched"] == b["records"]
            assert b["source_present"] == b["record_present"]
            assert b["dropped"] == b["invented"] == b["changed"] == 0
            rows.append(
                dict(
                    b,
                    before_record_present=a["record_present"],
                    recovered_values=b["record_present"] - a["record_present"],
                )
            )
    return rows


def main():
    from presto.scripts.experiment_registry import write_reproducibility_bundle

    output = EXPERIMENT / "results" / "compare"
    if output.exists():
        raise FileExistsError(output)
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    assert not git_status, "Comparison must start from a clean committed tree"
    write_reproducibility_bundle(
        out_dir=output,
        source_script=str(Path(__file__).relative_to(ROOT)),
        argv=sys.argv,
        environment={
            name: os.environ.get(name, "")
            for name in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    )
    results, receipts, inputs = {}, {}, {}
    for condition in ("before", "after"):
        directory = EXPERIMENT / "results" / condition
        receipt = json.loads((directory / "invocation.json").read_text())
        receipts[condition] = receipt
        assert not receipt["git"]["dirty"]
        assert json.loads((directory / "status.json").read_text())["status"] == "completed"
        assert digest(directory / "reproduce/source/launch.py") == receipt["launcher_sha256"]
        # Production snapshots are pinned by preserved git commits, not mutable files.
        for name, expected in receipt["production_sha256"].items():
            source = subprocess.check_output(
                ["git", "show", f"{receipt['git']['commit']}:{name}"], cwd=ROOT
            )
            assert hashlib.sha256(source).hexdigest() == expected, f"Wrong pinned source: {name}"
        results[condition] = json.loads((directory / "result.json").read_text())
        with (directory / "field_counts.csv").open() as handle:
            csv_rows = list(csv.DictReader(handle))
        expected_rows = [
            {k: str(v) for k, v in row.items()}
            for selector in ("panel", "bootstrap")
            for row in results[condition][selector]["field_counts"]
        ]
        assert csv_rows == expected_rows
        inputs[condition] = {
            name: digest(directory / name)
            for name in ("result.json", "field_counts.csv", "invocation.json", "status.json")
        }
    for name in (
        "packages",
        "python",
        "platform",
        "environment",
        "source",
        "source_sha256",
        "contract",
    ):
        assert receipts["before"][name] == receipts["after"][name], f"Changed contract: {name}"
    rows = compare(results["before"], results["after"])
    payload = dict(
        status="completed",
        phase_sha256=inputs,
        field_counts=rows,
        populations_and_non_descriptor_payloads_identical=True,
        every_descriptor_and_selected_column_matches_source=True,
    )
    (output / "result.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    with (output / "field_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
