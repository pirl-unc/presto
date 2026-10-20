"""Append canonical assay applicability to immutable native source inspection rows."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import resource
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datacache import install_bundle, materialize
import presto
from presto.data.assay_context import ASSAY_CONTEXT_POLICY, assay_context_fields
from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.species_context import native_species_context
from presto.training.output_contract import build_output_contract

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
CACHE = Path.home() / ".hitlist/presto-assay-applicability"
BASE = "https://github.com/pirl-unc/presto/releases/download/native-presenter-context-audit-v1/"
INPUTS = {
    "prior-observations.tsv": {
        "url": BASE + "presenter-context-observations.tsv",
        "sha256": "feb29f9a60b8e8ae5dc6a3af0477a25ade615e370f0d28044e729dfcde11c11e",
        "size": 10513967,
    },
    "prior-ic50.tsv": {
        "url": BASE + "ic50-processing-contexts.tsv",
        "sha256": "cb74c379457e07eafdc91adc3a0256c0d8bfad3ec3ba2c70933bafdf1f6d6f69",
        "size": 3046490,
    },
    "training.parquet": {
        "url": "https://github.com/pirl-unc/hitlist/releases/download/human-cohort-corrected-v1/training.parquet",
        "sha256": "f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd",
        "size": 188175209,
    },
}


def pin(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size_bytes": p.stat().st_size}


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_rows(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    assert (
        presto.__version__ == "0.1.22"
        and Path(presto.__file__).resolve() == ROOT / "__init__.py"
    )
    paths = install_bundle(
        CACHE / "inputs", INPUTS, download_options={"show_progress": False}
    )
    code_names = [
        "data/assay_context.py",
        "data/source_lineage.py",
        "data/species_context.py",
        "data/collate.py",
        "data/loaders.py",
        "data/hitlist_source.py",
        "data/frozen_partitions.py",
        "training/output_coverage.py",
        "training/output_contract.py",
        "training/coverage_preflight.py",
        "training/supported_outputs.py",
    ]
    code = {name: pin(ROOT / name) for name in code_names}
    specs = {
        name: {
            "path": str(path),
            "sha256": pin(path)["sha256"],
            "size": pin(path)["size_bytes"],
        }
        for name, path in paths.items()
    }
    specs["audit.py"] = {
        "path": str(Path(__file__)),
        "sha256": pin(__file__)["sha256"],
        "size": pin(__file__)["size_bytes"],
    }
    specs["assay_context.py"] = {
        "path": str(ROOT / "data/assay_context.py"),
        "sha256": code["data/assay_context.py"]["sha256"],
        "size": code["data/assay_context.py"]["size_bytes"],
    }
    outputs = [
        "prior-observations.tsv",
        "prior-ic50.tsv",
        "audit.py",
        "assay_context.py",
        "observations.tsv",
        "ic50.tsv",
        "native-evidence.tsv",
        "coverage.tsv",
        "policy.json",
        "output-contract.json",
        "audit.json",
    ]

    def build(inputs, target):
        started = time.monotonic()
        old = read(inputs["prior-observations.tsv"])
        ic50 = read(inputs["prior-ic50.tsv"])
        ids = {row["evidence_row_id"] for row in old}
        assert len(ids) == len(old) == 9053 and len(ic50) == 2614
        native = {}
        for batch in pq.ParquetFile(inputs["training.parquet"]).iter_batches(
            batch_size=5000,
            columns=["evidence_row_id", "assay_method", "evidence_kind"],
        ):
            for row in batch.to_pylist():
                if row["evidence_row_id"] in ids:
                    cleaned = {k: "" if v is None else str(v) for k, v in row.items()}
                    if row["evidence_row_id"] in native:
                        assert native[row["evidence_row_id"]] == cleaned
                    native[row["evidence_row_id"]] = cleaned
        assert set(native) == ids
        appended = {}
        counts = Counter()
        ic50_counts = Counter()
        new = []
        for row in old:
            source = native[row["evidence_row_id"]]
            assert (
                row["assay_method"] == source["assay_method"]
                and not row["generated_id"]
            )
            sample = PrestoSample(
                peptide=row["peptide"],
                species_context=native_species_context(row),
                source_assay_method=source["assay_method"],
                source_evidence_kind=source["evidence_kind"],
            )
            fields = {
                "source_assay_method": source["assay_method"],
                "source_evidence_kind": source["evidence_kind"],
                "parent_assay_contexts": "",
                **assay_context_fields(sample),
            }
            assert not set(fields) & set(row)
            batch = PrestoCollator()([sample])
            assert all(
                batch.source_lineage[key] == [value] for key, value in fields.items()
            )
            appended[row["evidence_row_id"]] = fields
            new.append({**row, **fields})
            counts[
                (
                    row["source_partition"],
                    source["evidence_kind"],
                    source["assay_method"],
                    fields["processing_context_applicability"],
                    fields["processing_context_availability"],
                )
            ] += 1
        new_ic50 = [{**row, **appended[row["evidence_row_id"]]} for row in ic50]
        for row in new_ic50:
            ic50_counts[row["processing_context_applicability"]] += 1
        assert dict(ic50_counts) == {"not_applicable": 2533, "applicable": 81}
        write_rows(target["observations.tsv"], new)
        write_rows(target["ic50.tsv"], new_ic50)
        write_rows(
            target["native-evidence.tsv"],
            [native[row["evidence_row_id"]] for row in old],
        )
        fields = [
            "source_partition",
            "source_evidence_kind",
            "source_assay_method",
            "processing_context_applicability",
            "processing_context_availability",
        ]
        write_rows(
            target["coverage.tsv"],
            [
                {**dict(zip(fields, key)), "n_observations": value}
                for key, value in sorted(counts.items())
            ],
        )
        write(
            target["policy.json"],
            {
                "policy": ASSAY_CONTEXT_POLICY,
                "inspection_only": True,
                "new_predictive_inputs": False,
                "historical_fields_changed": False,
                "source_inputs": INPUTS,
                "classification_basis": "Explicit purified/cellular MHC preparation or native MS/elution evidence; conflicting or unspecified evidence stays unknown.",
                "generated_controls": "No generated rows in this audit; unit/integration tests preserve parent preparation provenance and require unanimous applicability.",
                "limitations": [
                    "No new full-input qualification, model fitting or predictions.",
                    "Unknown specimen/donor identities and failed historical IC50 quality remain.",
                    "Not-applicable processing context does not invalidate an affinity measurement.",
                ],
            },
        )
        write(target["output-contract.json"], build_output_contract().to_dict())
        for name in [
            "prior-observations.tsv",
            "prior-ic50.tsv",
            "audit.py",
            "assay_context.py",
        ]:
            shutil.copyfile(inputs[name], target[name])
        write(
            target["audit.json"],
            {
                "commit": commit,
                "dirty": False,
                "agent": "codex",
                "model": "GPT-6",
                "presto_version": presto.__version__,
                "code": code,
                "inputs": INPUTS,
                "n_observations": 9053,
                "n_unchanged_ic50_observations_and_targets": 2614,
                "n_ic50_by_applicability": dict(ic50_counts),
                "n_canonical_collator_metadata_rows": 9053,
                "runtime_seconds": time.monotonic() - started,
                "process_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
                "runtime_scope": "builder; imports/acquisition excluded",
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ["datacache", "numpy", "pandas", "pyarrow", "torch"]
                },
                "execution": "CPU additive source metadata audit; no fitting, predictions or historical changes",
            },
        )

    result = materialize(
        CACHE / "audit" / commit,
        specs,
        transform={
            "version": "assay-context-applicability-audit-v1",
            "options": {"commit": commit},
        },
        outputs={n: {} for n in outputs},
        builder=build,
    )
    write(
        FAMILY / "results/audit-receipt.json",
        {
            "commit": commit,
            "paths": {n: str(p) for n, p in result.items()},
            "files": {n: pin(p) for n, p in result.items()},
        },
    )
    shutil.copyfile(result["audit.json"], FAMILY / "results/audit.json")
    print(Path(result["audit.json"]).read_text())


if __name__ == "__main__":
    main()
