#!/usr/bin/env python
"""Compare observed and emitted descriptors in actual merged binding selectors."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Users/iskander/code/presto/data/merged_deduped.tsv")
SOURCE_HASH = "46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c"
FIELDS = ("assay_type", "assay_method", "effector_culture_condition", "apc_culture_condition")
ALLELES = ["HLA-A*02:01", "HLA-A*03:01"]


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"


def inspect_selector(path, selector):
    from presto.data.collate import PrestoCollator
    from presto.scripts import train_iedb as runner

    constructor, classify = runner.BindingRecord, runner.classify_assay_type
    collator = PrestoCollator()
    current = None
    count = 0
    field_counts = {name: Counter() for name in FIELDS}
    payload_hash, descriptor_hash = hashlib.sha256(), hashlib.sha256()
    actual_columns, expected_columns = Counter(), Counter()

    def columns(descriptors, measurement):
        prep, geometry, readout = collator._factorize_binding_assay_method(
            descriptors["assay_method"]
        )
        return (
            collator._categorize_binding_assay_type(descriptors["assay_type"] or measurement),
            collator._categorize_binding_assay_method(descriptors["assay_method"]),
            prep,
            geometry,
            readout,
        )

    def classify_record(record):
        nonlocal current
        bucket = classify(record)
        current = record if bucket == "binding_affinity" else None
        return bucket

    def binding_record(**kwargs):
        nonlocal count
        record = constructor(**kwargs)
        assert current is not None and (record.peptide, record.mhc_allele) == (
            current.peptide,
            current.mhc_allele,
        )
        count += 1
        expected = {name: str(getattr(current, name) or "").strip() for name in FIELDS}
        actual = {name: str(getattr(record, name) or "").strip() for name in FIELDS}
        for name in FIELDS:
            counters = field_counts[name]
            counters["records"] += 1
            counters["source_present"] += bool(expected[name])
            counters["record_present"] += bool(actual[name])
            counters["matched"] += actual[name] == expected[name]
            counters["dropped"] += bool(expected[name]) and not actual[name]
            counters["invented"] += bool(actual[name]) and not expected[name]
            counters["changed"] += (
                bool(actual[name]) and bool(expected[name]) and actual[name] != expected[name]
            )
        payload_hash.update(
            encode({key: value for key, value in vars(record).items() if key not in FIELDS})
        )
        descriptor_hash.update(encode(expected))
        actual_columns[columns(actual, record.measurement_type)] += 1
        expected_columns[columns(expected, record.measurement_type)] += 1
        return record

    runner.BindingRecord, runner.classify_assay_type = binding_record, classify_record
    try:
        if selector == "panel":
            records, stats = runner.load_binding_records_for_alleles_from_merged_tsv(
                path, alleles=ALLELES, max_records=1, cap_sampling="head", sampling_seed=17
            )
        elif selector == "bootstrap":
            records, stats = runner.load_probe_allele_binding_bootstrap_from_merged_tsv(
                path,
                probe_alleles=ALLELES,
                max_records=2000,
                max_peptides=500,
                max_rows_per_peptide=4,
                sampling_seed=17,
            )
        else:
            raise ValueError(selector)
    finally:
        runner.BindingRecord, runner.classify_assay_type = constructor, classify
    retained = hashlib.sha256()
    for record in records:
        retained.update(
            encode({key: value for key, value in vars(record).items() if key not in FIELDS})
        )
    if selector == "bootstrap":
        assert count == len(records) == stats["records_added"]
    else:
        assert len(records) == min(1, count) == stats["rows_selected"]
    rows = []
    for name, counts in field_counts.items():
        values = {
            key: counts[key]
            for key in (
                "records",
                "source_present",
                "record_present",
                "matched",
                "dropped",
                "invented",
                "changed",
            )
        }
        assert values["records"] == sum(
            values[key] for key in ("matched", "dropped", "invented", "changed")
        )
        rows.append(dict(selector=selector, field=name, **values))
    column_names = ("assay_type", "assay_method", "prep", "geometry", "readout")
    return dict(
        selector=selector,
        observed_records=count,
        retained_records=len(records),
        stats=stats,
        non_descriptor_payload_sha256=payload_hash.hexdigest(),
        retained_non_descriptor_sha256=retained.hexdigest(),
        source_descriptors_sha256=descriptor_hash.hexdigest(),
        field_counts=rows,
        actual_columns=[
            dict(zip(column_names, key, strict=True), records=n)
            for key, n in sorted(actual_columns.items())
        ],
        expected_columns=[
            dict(zip(column_names, key, strict=True), records=n)
            for key, n in sorted(expected_columns.items())
        ],
    )


def main():
    import presto
    from presto.scripts.experiment_registry import write_reproducibility_bundle

    assert Path(presto.__file__).resolve().parent == ROOT, "Wrong Presto checkout imported"
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("before", "after"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or EXPERIMENT / "results" / args.condition
    if output.exists():
        raise FileExistsError(output)
    git = {
        name: subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()
        for name, command in {
            "commit": ["rev-parse", "HEAD"],
            "status": ["status", "--porcelain"],
        }.items()
    }
    git["dirty"] = bool(git["status"])
    environment = {
        name: os.environ.get(name, "")
        for name in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    }
    bundle = write_reproducibility_bundle(
        out_dir=output,
        source_script=str(Path(__file__).relative_to(ROOT)),
        argv=sys.argv,
        environment=environment,
    )
    production = {
        name: hash_file(ROOT / name)
        for name in (
            "scripts/train_iedb.py",
            "data/loaders.py",
            "data/collate.py",
            "data/cross_source_dedup.py",
            "data/assay_types.py",
            "data/source_lineage.py",
        )
    }
    receipt = dict(
        agent_model="Codex / GPT-6",
        condition=args.condition,
        git=git,
        argv=[sys.executable, *sys.argv],
        cwd=str(ROOT),
        python=sys.version,
        platform=platform.platform(),
        environment=environment,
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        source=str(SOURCE),
        source_sha256=SOURCE_HASH,
        production_sha256=production,
        launcher_sha256=hash_file(Path(__file__)),
        reproduce_bundle=str(bundle),
        contract=json.loads((EXPERIMENT / "reproduce/launch.json").read_text()),
    )
    json_write(output / "invocation.json", receipt)
    started = time.perf_counter()
    try:
        assert hash_file(SOURCE) == SOURCE_HASH
        results = {}
        for selector in ("panel", "bootstrap"):
            phase_start = time.perf_counter()
            results[selector] = inspect_selector(SOURCE, selector)
            results[selector]["elapsed_seconds"] = time.perf_counter() - phase_start
            print(selector, results[selector]["observed_records"], "records", flush=True)
        assert hash_file(SOURCE) == SOURCE_HASH
        for name, expected in production.items():
            assert hash_file(ROOT / name) == expected, f"Production changed during scan: {name}"
        json_write(output / "result.json", results)
        rows = [row for result in results.values() for row in result["field_counts"]]
        with (output / "field_counts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    except BaseException as exc:
        json_write(
            output / "status.json",
            dict(status="failed", error=repr(exc), elapsed_seconds=time.perf_counter() - started),
        )
        raise
    json_write(
        output / "status.json",
        dict(status="completed", git=git, elapsed_seconds=time.perf_counter() - started),
    )


if __name__ == "__main__":
    main()
