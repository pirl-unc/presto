"""Inventory curated arm metadata only; never load peptide or outcome tables."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import yaml


FIELDS = (
    "condition_genetic_variants",
    "condition_infection",
    "condition_stimulation",
    "condition_antigen_exposure",
)
PIN = "43e2266"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hitlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    commit = subprocess.check_output(
        ["git", "-C", str(args.hitlist), "rev-parse", "HEAD"], text=True
    ).strip()
    if not commit.startswith(PIN):
        raise ValueError("Use the frozen Hitlist commit, not changed curation")
    paths = (
        "hitlist/data/pmid_overrides.yaml",
        "hitlist/conditions.py",
        "hitlist/data/condition_vocabulary.yaml",
    )
    inputs = {}
    contents = {}
    for name in paths:
        frozen = subprocess.check_output(["git", "-C", str(args.hitlist), "show", f"{commit}:{name}"])
        current = (args.hitlist / name).read_bytes()
        if current != frozen:
            raise ValueError(f"Dirty input: {name}")
        inputs[name] = {"bytes": len(current), "sha256": hashlib.sha256(current).hexdigest()}
        contents[name] = current
    studies = yaml.safe_load(contents[paths[0]])
    records = []
    counts = Counter()
    study_sets = defaultdict(set)
    total_arms = 0
    for study in studies:
        study_id = str(study.get("pmid") or study.get("submission_id") or "unknown")
        for index, arm in enumerate(study.get("ms_samples") or []):
            total_arms += 1
            for field in FIELDS:
                value = arm.get(field)
                if not value:
                    continue
                row = {
                    "study_id": study_id,
                    "arm_index": index,
                    "condition_id": arm.get("condition_id", ""),
                    "mhc_class": arm.get("mhc_class", "unknown"),
                    "field": field,
                    "value": value,
                    "status": arm.get("condition_status", ""),
                    "evidence": arm.get("condition_evidence", ""),
                    "reference": arm.get("condition_reference", ""),
                }
                records.append(row)
                counts[(field, row["mhc_class"])] += 1
                study_sets[field].add(study_id)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "source_arms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "hitlist_commit": commit,
        "inputs": inputs,
        "n_curated_studies": len(studies),
        "n_curated_arms": total_arms,
        "n_arms_with_any_added_field": len({(r["study_id"], r["arm_index"]) for r in records}),
        "fields": {
            field: {
                "n_arms": sum(n for (f, _), n in counts.items() if f == field),
                "n_studies": len(study_sets[field]),
                "n_arms_by_declared_mhc_class": {
                    c: n for (f, c), n in sorted(counts.items()) if f == field
                },
                "n_arms_by_annotation_status": dict(Counter(r["status"] for r in records if r["field"] == field)),
            }
            for field in FIELDS
        },
        "contract": "curation metadata only; no export join, peptide count, outcome, or split access",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
