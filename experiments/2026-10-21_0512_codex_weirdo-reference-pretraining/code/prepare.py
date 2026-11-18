#!/usr/bin/env python
"""Freeze and audit the prospectively selected peptide-source cohort."""

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from presto.data.modular_auxiliary import weirdo_observations
from presto.data.modular_reference import audit_weirdo_reference
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    file_sha256,
    write_observations,
)
from presto.training.modular_run import preflight

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "experiments/2026-10-21_0512_codex_weirdo-reference-pretraining"
COHORT = ROOT / "data/modular_cohorts/2026-10-21-weirdo-reference"
REFERENCE = Path("/Users/iskander/code/weirdo/data/swissprot-sequences.csv")
SPLITS = REFERENCE.parent / "category-splits"
PINS = {
    "reference": "e9c44a34fcd3f8fb1f4151d489985dd5578a947eda3e62213d5109ace36e115d",
    "archive": "99b21eb93196cf2b0a0b1ae4bbb63066224a6864e48dd3a8d47fe67bd34fcf79",
    "metadata": "571356e92fd1da71cf173be9f5e59c6aa128467f347de37aa30815d78923fc1f",
}
LIMITS = {"train": 5000, "validation": 1000, "test": 1000}
TRAINING = {
    "seed": 42,
    "batch_size": 64,
    "max_epochs": 20,
    "max_steps": None,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "identity_weight": 0.1,
    "patience": 5,
    "min_epochs": 5,
    "checkpoint_every_steps": 100,
    "test_policy": "evaluate_selected_once",
    "coverage_policy": "required",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FAMILY)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--reconstruct-missing-groups", action="store_true")
    args = parser.parse_args()
    output, cohort = args.output.resolve(), args.cohort.resolve()
    if (
        (output / "results/preparation.json").exists()
        or (output / "reproduce/preparation-launch.json").exists()
        or (cohort / "manifest.json").exists()
    ):
        raise ValueError("refusing to overwrite an executed preparation")
    for name in ("results", "reproduce/source", "launch_logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    cohort.mkdir(parents=True, exist_ok=True)
    try:
        artifact_name = output.relative_to(ROOT / "experiments")
    except ValueError:
        artifact_name = Path(output.name)
    raw = ROOT / "artifacts" / artifact_name
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "reference-audit.json.gz").exists():
        raise ValueError("raw artifact name already used; choose a distinct fresh family name")
    invocation = {
        "agent_model": "Codex / GPT-6",
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "argv": sys.argv,
        "executable": sys.executable,
        "cwd": str(ROOT),
        "python": sys.version,
        "environment": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "versions": {name: version(name) for name in ("numpy", "torch", "scikit-learn")},
        "selection": {"limits": LIMITS, "seeds": {"train": 42, "validation": 43, "test": 44}},
        "pins": PINS,
        "device": "cpu",
        "gpu": None,
    }
    write_json(output / "reproduce/preparation-launch.json", invocation)
    for relative in (
        "data/modular_auxiliary.py",
        "data/modular_reference.py",
        "data/modular_sources.py",
        "data/modular.py",
        "data/modular_lineage.py",
        "data/modular_quantitative.py",
        "data/modular_context.py",
    ):
        target = output / "reproduce/source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(__file__, output / "reproduce/source/prepare.py")
    started = time.monotonic()
    try:
        rows = list(
            weirdo_observations(
                SPLITS,
                archive_sha256=PINS["archive"],
                metadata_sha256=PINS["metadata"],
                reference_sha256=PINS["reference"],
                limits=LIMITS,
                seed=42,
                allow_missing_groups=args.reconstruct_missing_groups,
            )
        )
        metadata = json.loads((SPLITS / "splits.json").read_text())
        (output / "results/selected.jsonl.gz").write_bytes(
            gzip.compress(
                "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows).encode(),
                mtime=0,
            )
        )
        audit = audit_weirdo_reference(
            rows,
            REFERENCE,
            reference_sha256=PINS["reference"],
            categories=metadata["categories"],
            salt=metadata["salt"],
            max_seconds=900,
            reconstruct_missing_groups=args.reconstruct_missing_groups,
            progress=lambda n: (
                print(f"reference proteins audited={n}", flush=True) if n % 32768 == 0 else None
            ),
        )
        audit_path = raw / "reference-audit.json.gz"
        audit_path.write_bytes(gzip.compress(json.dumps(audit, sort_keys=True).encode(), mtime=0))
        summaries = [
            {key: value for key, value in record.items() if key != "proteins"}
            for record in audit["records"]
        ]
        (output / "results/membership-audit.jsonl.gz").write_bytes(
            gzip.compress(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in summaries).encode(),
                mtime=0,
            )
        )
        report = {key: value for key, value in audit.items() if key != "records"}
        report.update(
            selection=invocation["selection"],
            pins=PINS,
            full_audit={"path": str(audit_path), "sha256": file_sha256(audit_path)},
            original_table_verification=metadata["table_verification"],
            original_table_verification_scope=(
                "comparison against another k-mer table; not this selected-row reference audit"
            ),
        )
        if audit["status"] != "passed":
            write_json(output / "results/preparation.json", report)
            raise ValueError("selected reference admission failed; inspect complete discrepancies")
        prepared = []
        for row, record in zip(rows, audit["records"], strict=True):
            prepared.append(
                replace(
                    row,
                    group_id=f"weirdo-reference:{PINS['reference']}:{record['component']}",
                    provenance={
                        **row.provenance,
                        "reference_membership_audited": True,
                        "source_family_keys": sorted(
                            {protein["family_key"] for protein in record["proteins"]}
                        ),
                        "source_group_component": record["component"],
                        "reference_family_bucket": record["reference_bucket"],
                        "grouping_evidence": "audited_reference_component",
                        "reference_audit_record_sha256": hashlib.sha256(
                            json.dumps(record, sort_keys=True).encode()
                        ).hexdigest(),
                    },
                )
            )
        receipt = write_observations(cohort / "observations.jsonl", prepared)
        manifest = {
            "schema": MANIFEST_VERSION,
            "model": {
                "endpoints": [],
                "identity_tasks": ["peptide_source"],
                "source_categories": metadata["categories"],
                "dropout": 0.1,
            },
            "training": TRAINING,
            "sources": [
                {
                    "path": "observations.jsonl",
                    "sha256": receipt["sha256"],
                    "source": "weirdo_swissprot_reference",
                    "population": "label-blind 5000/1000/1000 eight-residue reference subset",
                    "label_policy": (
                        "ten independent pinned reference-division memberships; "
                        "all selected labels reconstructed against every reference protein"
                    ),
                    "split_policy": (
                        "source mnemonic-prefix CRC32 folds, no selected cross-fold occurrence; "
                        "group IDs connect every shared source-family key; "
                        "homology independence unverified"
                    ),
                }
            ],
        }
        manifest_path = cohort / "manifest.json"
        write_json(manifest_path, manifest)
        _, _, _, admitted = preflight(ObservationIndex(manifest_path))
        write_json(output / "results/preflight.json", admitted)
        for split in LIMITS:
            subset = [row for row in prepared if row.split == split]
            report.setdefault("label_counts", {})[split] = {
                category: dict(
                    Counter(row.identity_labels[f"peptide_source:{category}"] for row in subset)
                )
                for category in metadata["categories"]
            }
            report.setdefault("components_by_split", {})[split] = len(
                {row.group_id for row in subset}
            )
        report.update(
            manifest={"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            normalized_observations=receipt,
            elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        )
        write_json(output / "results/preparation.json", report)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "selected",
                        "reference_proteins",
                        "components_by_split",
                        "elapsed_seconds",
                        "peak_rss_bytes",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            output / "results/preparation-failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
