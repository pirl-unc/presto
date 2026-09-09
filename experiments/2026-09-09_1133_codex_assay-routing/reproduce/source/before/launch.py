#!/usr/bin/env python
"""Observe the canonical loader's complete pre-cap routing and record payloads."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_HASH = "46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c"
FIELDS = ("source", "record_type", "value_type", "assay_type", "assay_method", "response")
RECORD_MODALITIES = {
    "BindingRecord": "binding",
    "KineticsRecord": "kinetics",
    "StabilityRecord": "stability",
    "ProcessingRecord": "processing",
    "ElutionRecord": "elution",
    "TCellRecord": "tcell",
    "TcrEvidenceRecord": "tcr_evidence",
}


def json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def freeze(args, output):
    source_receipt = (
        ROOT
        / "experiments/2026-09-09_1106_codex_output-coverage"
        / "reproduce/inventory_v2_invocation.json"
    )
    receipt = {
        "agent_model": "Codex / GPT-6",
        "condition": args.condition,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "git": {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")},
        "environment": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "packages": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "environment_source_receipt": {
            "path": str(source_receipt),
            "sha256": hash_file(source_receipt),
        },
        "source": str(ROOT / "data/merged_deduped.tsv"),
        "expected_source_sha256": SOURCE_HASH,
        "retained_cap_per_modality": 1,
        "sampling": "head",
        "sampling_seed": 17,
        "production_files": {},
    }
    receipt["git"]["dirty"] = bool(receipt["git"]["status"])
    snapshot = EXPERIMENT / "reproduce/source" / output.name
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ("data/cross_source_dedup.py", "scripts/train_iedb.py", "data/loaders.py"):
        source = ROOT / name
        receipt["production_files"][name] = hash_file(source)
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(__file__, snapshot / "launch.py")
    receipt["launcher_sha256"] = hash_file(Path(__file__))
    json_write(output / "invocation.json", receipt)
    json_write(EXPERIMENT / "reproduce" / f"{output.name}_invocation.json", receipt)
    return receipt


def inspect_loader(path):
    from presto.scripts import train_iedb as runner

    original_classify = runner.classify_assay_type
    original_append = runner._append_with_cap_sampling
    groups = Counter()
    buckets = Counter()
    emitted = Counter()
    payload_hashes = defaultdict(hashlib.sha256)
    payload_counts = Counter()
    numeric_examples = defaultdict(list)
    current = None

    def classify(record):
        nonlocal current
        bucket = original_classify(record)
        key = tuple(getattr(record, field) or "" for field in FIELDS) + (record.value is not None,)
        current = (key, bucket)
        groups[current] += 1
        buckets[bucket] += 1
        if record.value is not None and len(numeric_examples[current]) < 3:
            numeric_examples[current].append(record.value)
        if sum(buckets.values()) % 250000 == 0:
            print(f"Classified {sum(buckets.values()):,} rows", flush=True)
        return bucket

    def append(records, record, limit, **kwargs):
        modality = RECORD_MODALITIES[type(record).__name__]
        emitted[modality] += 1
        payload_key = (*current, modality)
        payload_counts[payload_key] += 1
        payload_hashes[payload_key].update(
            json.dumps(
                record.__dict__, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            + b"\n"
        )
        return original_append(records, record, limit, **kwargs)

    runner.classify_assay_type = classify
    runner._append_with_cap_sampling = append
    try:
        *_, stats = runner.load_records_from_merged_tsv(
            path,
            max_binding=1,
            max_kinetics=1,
            max_stability=1,
            max_processing=1,
            max_elution=1,
            max_tcell=1,
            max_vdjdb=1,
            cap_sampling="head",
            sampling_seed=17,
        )
    finally:
        runner.classify_assay_type = original_classify
        runner._append_with_cap_sampling = original_append
    if buckets != stats["rows_by_assay"]:
        raise AssertionError("Observed classifications differ from loader statistics")
    if any(emitted[name] != count for name, count in stats["counts_before_cap"].items()):
        raise AssertionError("Observed records differ from loader pre-cap counts")
    rows = []
    for (key, bucket), count in sorted(groups.items()):
        row = dict(zip((*FIELDS, "has_numeric_value"), key))
        row.update(bucket=bucket, rows=count, numeric_examples=numeric_examples[(key, bucket)])
        row["emitted"] = {
            modality: {
                "records": payload_counts[(key, bucket, modality)],
                "ordered_sha256": payload_hashes[(key, bucket, modality)].hexdigest(),
            }
            for modality in RECORD_MODALITIES.values()
            if payload_counts[(key, bucket, modality)]
        }
        rows.append(row)
    return {"stats": stats, "groups": rows, "observed_pre_cap_records": dict(emitted)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=["before", "after"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or EXPERIMENT / "results" / args.condition
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    receipt = freeze(args, output)
    started = time.perf_counter()
    path = ROOT / "data/merged_deduped.tsv"
    try:
        if hash_file(path) != SOURCE_HASH:
            raise RuntimeError("Source differs from the declared merged input")
        result = inspect_loader(path)
        if hash_file(path) != SOURCE_HASH:
            raise RuntimeError("Source changed during the full-loader audit")
        for name, digest in receipt["production_files"].items():
            if hash_file(ROOT / name) != digest:
                raise RuntimeError(f"Production code changed during audit: {name}")
        result.update(source_sha256=SOURCE_HASH, condition=args.condition)
        json_write(output / "result.json", result)
        with (output / "groups.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=result["groups"][0])
            writer.writeheader()
            for row in result["groups"]:
                writer.writerow(
                    {
                        **row,
                        "emitted": json.dumps(row["emitted"], sort_keys=True),
                        "numeric_examples": json.dumps(row["numeric_examples"]),
                    }
                )
    except BaseException as exc:
        json_write(
            output / "status.json",
            {
                "status": "failed",
                "error": repr(exc),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
    json_write(
        output / "status.json",
        {
            "status": "completed",
            "git": receipt["git"],
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        json.dumps({"stats": result["stats"], "groups": len(result["groups"])}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
