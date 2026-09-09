#!/usr/bin/env python
"""Audit publication metadata at the actual merged reader/record boundary."""

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
FIELDS = ("pmid", "doi", "reference_text", "evidence_row_id", "assay_iri", "reference_iri")
MODALITIES = {
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


def freeze(args, output):
    from presto.scripts.experiment_registry import write_reproducibility_bundle

    git_state = {
        name: subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()
        for name, command in {
            "commit": ["rev-parse", "HEAD"],
            "branch": ["branch", "--show-current"],
            "status": ["status", "--porcelain"],
        }.items()
    }
    git_state["dirty"] = bool(git_state["status"])
    # Explicit nonempty allowlist prevents the shared helper's prefix-based
    # environment fallback from including unrelated credentials.
    environment = {name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    bundle = write_reproducibility_bundle(
        out_dir=output,
        source_script=str(Path(__file__).relative_to(ROOT)),
        argv=sys.argv,
        environment=environment,
    )
    production_files = {}
    for name in (
        "scripts/train_iedb.py",
        "data/loaders.py",
        "data/source_lineage.py",
        "data/collate.py",
        "data/cross_source_dedup.py",
        "training/holdout_eval.py",
        "scripts/focused_binding_probe.py",
    ):
        path = ROOT / name
        if not path.exists():
            continue
        production_files[name] = hash_file(path)
        target = bundle / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    source_receipt = (
        ROOT
        / "experiments/2026-09-09_1106_codex_output-coverage/reproduce/inventory_v2_invocation.json"
    )
    receipt = {
        "agent_model": "Codex / GPT-6",
        "condition": args.condition,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state,
        "argv": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": environment,
        "packages": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "package_source_receipt": {
            "path": str(source_receipt),
            "sha256": hash_file(source_receipt),
        },
        "production_files": production_files,
        "launcher_sha256": hash_file(Path(__file__)),
        "source_sha256": SOURCE_HASH,
        "source": str(ROOT / "data/merged_deduped.tsv"),
        "reproduce_bundle": str(bundle),
        "retained_cap_per_modality": 1,
        "sampling": "head",
        "sampling_seed": 17,
    }
    json_write(output / "invocation.json", receipt)
    return receipt


def inspect_loader(path):
    from presto.scripts import train_iedb as runner

    original_csv, original_append = runner.csv, runner._append_with_cap_sampling
    current = None
    input_rows = 0
    columns = []
    source_available, emitted = Counter(), Counter()
    field_counts = defaultdict(Counter)
    payload_hashes = defaultdict(hashlib.sha256)
    source_hashes = defaultdict(hashlib.sha256)
    examples = defaultdict(list)

    def reader(*args, **kwargs):
        nonlocal current, input_rows, columns
        source = original_csv.DictReader(*args, **kwargs)
        columns = list(source.fieldnames or [])
        for row in source:
            input_rows += 1
            current = row
            for name in FIELDS:
                source_available[name] += bool((row.get(name) or "").strip())
            if input_rows % 250000 == 0:
                print(f"Read {input_rows:,} source rows", flush=True)
            yield row

    class CSVProxy:
        def __getattr__(self, name):
            return reader if name == "DictReader" else getattr(original_csv, name)

    def append(records, item, limit, **kwargs):
        modality = MODALITIES[type(item).__name__]
        assert current is not None
        assert item.peptide == str(current["peptide"]).strip().upper()
        emitted[modality] += 1
        observed = {}
        for name in FIELDS:
            expected = (current.get(name) or "").strip()
            actual = str(getattr(item, name, "") or "").strip()
            observed[name] = expected
            counts = field_counts[(modality, name)]
            counts["records"] += 1
            counts["source_present"] += bool(expected)
            counts["record_present"] += bool(actual)
            counts["matched"] += actual == expected
            counts["dropped"] += bool(expected) and not actual
            counts["invented"] += bool(actual) and not expected
            counts["changed"] += bool(actual) and bool(expected) and actual != expected
            if actual != expected and len(examples[(modality, name)]) < 3:
                examples[(modality, name)].append(
                    {
                        "input_record_ordinal": input_rows,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        source_hashes[modality].update(json.dumps(observed, sort_keys=True).encode() + b"\n")
        payload = {key: value for key, value in vars(item).items() if key not in FIELDS}
        payload_hashes[modality].update(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode() + b"\n"
        )
        return original_append(records, item, limit, **kwargs)

    runner.csv = CSVProxy()
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
        runner.csv, runner._append_with_cap_sampling = original_csv, original_append
    assert input_rows == stats["rows_scanned"] + stats["rows_dropped_invalid_peptide"]
    assert all(emitted[name] == count for name, count in stats["counts_before_cap"].items())
    rows = []
    for modality in MODALITIES.values():
        for name in FIELDS:
            counts = {
                key: field_counts[(modality, name)][key]
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
            assert counts["records"] == sum(
                counts[key] for key in ("matched", "dropped", "invented", "changed")
            )
            rows.append({"modality": modality, "field": name, **counts})
    return {
        "stats": stats,
        "input_rows": input_rows,
        "source_columns": columns,
        "source_fields_present": {name: name in columns for name in FIELDS},
        "source_field_availability": dict(source_available),
        "field_counts": rows,
        "non_lineage_payload_sha256": {
            name: payload_hashes[name].hexdigest() for name in MODALITIES.values()
        },
        "source_metadata_sha256": {
            name: source_hashes[name].hexdigest() for name in MODALITIES.values()
        },
        "mismatch_examples": {
            f"{modality}:{name}": values for (modality, name), values in examples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=["before", "after"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    output = args.output_dir or EXPERIMENT / "results" / args.condition
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    receipt = freeze(args, output)
    started = time.perf_counter()
    source = ROOT / "data/merged_deduped.tsv"
    try:
        assert hash_file(source) == SOURCE_HASH, "Source differs from declared input"
        result = inspect_loader(source)
        assert hash_file(source) == SOURCE_HASH, "Source changed during audit"
        for name, expected in receipt["production_files"].items():
            assert hash_file(ROOT / name) == expected, (
                f"Production code changed during audit: {name}"
            )
        result.update(source_sha256=SOURCE_HASH, condition=args.condition)
        json_write(output / "result.json", result)
        with (output / "field_counts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=result["field_counts"][0])
            writer.writeheader()
            writer.writerows(result["field_counts"])
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
        },
    )
    print(
        json.dumps(
            {"records": result["stats"]["counts_before_cap"], "fields": result["field_counts"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
