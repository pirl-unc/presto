#!/usr/bin/env python
"""Reconcile frozen routes and require identical payloads for supported records."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
FIELDS = (
    "source",
    "record_type",
    "value_type",
    "assay_type",
    "assay_method",
    "response",
    "has_numeric_value",
)
ALLOWED = {
    ("elution_ms", "binding_affinity"),
    ("elution_ms", "binding_qualitative"),
    ("elution_ms", "binding_structure"),
    ("elution_ms", "presentation_non_ms"),
    ("binding_affinity", "binding_qualitative"),
    ("binding_affinity", "binding_structure"),
    ("binding_kon", "binding_association_constant"),
}


def compare(before, after):
    assert before["source_sha256"] == after["source_sha256"]
    for name in (
        "rows_scanned",
        "rows_by_source",
        "rows_dropped_invalid_peptide",
        "rows_sanitized_optional_sequences",
    ):
        assert before["stats"][name] == after["stats"][name], name
    keyed = [
        {tuple(row[name] for name in FIELDS): row for row in run["groups"]}
        for run in (before, after)
    ]
    assert all(len(index) == len(run["groups"]) for index, run in zip(keyed, (before, after)))
    assert keyed[0].keys() == keyed[1].keys(), "Descriptor populations changed"
    transitions, totals, changed = Counter(), Counter(), []
    for key, old in keyed[0].items():
        new = keyed[1][key]
        assert old["rows"] == new["rows"] and old["numeric_examples"] == new["numeric_examples"], (
            key
        )
        if old["bucket"] == new["bucket"]:
            assert old["emitted"] == new["emitted"], f"Supported payload changed: {key}"
            totals["unchanged_groups"] += 1
            totals["unchanged_classified_rows"] += new["rows"]
            totals["unchanged_emitted_records"] += sum(
                item["records"] for item in new["emitted"].values()
            )
        else:
            transition = (old["bucket"], new["bucket"])
            assert transition in ALLOWED, f"Unplanned transition: {transition}"
            assert new["emitted"] == {}, f"Changed group invented a target: {key}"
            removed = sum(item["records"] for item in old["emitted"].values())
            totals["removed_incorrect_targets"] += removed
            transitions[transition] += old["rows"]
            changed.append(
                {
                    **dict(zip(FIELDS, key)),
                    "before": old["bucket"],
                    "after": new["bucket"],
                    "rows": old["rows"],
                    "removed_records": removed,
                }
            )
    totals["before_pre_cap_records"] = sum(before["stats"]["counts_before_cap"].values())
    totals["after_pre_cap_records"] = sum(after["stats"]["counts_before_cap"].values())
    assert (
        totals["before_pre_cap_records"] - totals["after_pre_cap_records"]
        == totals["removed_incorrect_targets"]
    )
    assert totals["unchanged_emitted_records"] == totals["after_pre_cap_records"]
    assert (
        sum(after["stats"]["skipped_by_reason"].values())
        == after["stats"]["skipped_unroutable_or_missing_label"]
    )
    totals.update(groups=len(keyed[0]), changed_classified_rows=sum(transitions.values()))
    return {
        "summary": dict(totals),
        "transitions": [
            {"before": old, "after": new, "rows": count}
            for (old, new), count in sorted(transitions.items())
        ],
        "changed_groups": changed,
        "before_stats": before["stats"],
        "after_stats": after["stats"],
        "source_sha256": before["source_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT / "results/comparison")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    paths = {name: EXPERIMENT / f"results/{name}/result.json" for name in ("before", "after")}
    receipt = {
        "agent_model": "Codex / GPT-6",
        "argv": [sys.executable, *sys.argv],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            name: subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()
            for name, command in {
                "commit": ["rev-parse", "HEAD"],
                "status": ["status", "--porcelain"],
            }.items()
        },
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
    }
    receipt["git"]["dirty"] = bool(receipt["git"]["status"])
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    snapshot = EXPERIMENT / "reproduce/source" / args.output_dir.name
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, snapshot / "compare.py")
    try:
        result = compare(*(json.loads(paths[name].read_text()) for name in ("before", "after")))
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        with (args.output_dir / "transitions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("before", "after", "rows"))
            writer.writeheader()
            writer.writerows(result["transitions"])
    except BaseException as exc:
        (args.output_dir / "status.json").write_text(
            json.dumps({"status": "failed", "error": repr(exc)}) + "\n"
        )
        raise
    (args.output_dir / "status.json").write_text('{"status": "completed"}\n')
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
