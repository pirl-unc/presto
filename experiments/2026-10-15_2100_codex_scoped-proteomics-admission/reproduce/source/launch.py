"""Bounded admission audit; no training or held-out evaluation."""

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from datacache import materialize
import hitlist
from hitlist.downloads import data_dir
import presto
from presto.data.collate import PrestoCollator
from presto.data.loaders import PrestoDataset
from presto.data.scoped_bulk import load_scoped_bulk_records
from presto.data.source_lineage import source_lineage_fields
from presto.scripts.experiment_registry import (
    initialize_experiment_dir,
    write_reproducibility_bundle,
)
from presto.training.data_support import audit_split_support, write_split_support_artifacts
from presto.training.output_contract import build_output_contract
from presto.training.output_coverage import OutputCoverageCensus, write_output_coverage_artifacts

DIRECTORY = Path(__file__).resolve().parents[1]
ROOT = DIRECTORY.parents[1]
PIN = "441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b"
LIMIT = 10_000
SEED = 42


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def timeout(*_):
    raise TimeoutError("Admission audit exceeded its 15-minute wall budget")


def run():
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    torch.set_num_threads(1)
    source = str(Path(__file__).resolve().relative_to(ROOT))
    metadata = {
        "dataset_contract": {"manifest_sha256": PIN, "max_records": LIMIT, "seed": SEED},
        "training": {"status": "not_run", "gpu": "none", "model_fitting": False},
        "tested": ["published-scope admission, canonical collation and support audit"],
    }
    initialize_experiment_dir(
        out_dir=str(DIRECTORY),
        slug="scoped-proteomics-admission",
        title="Published scoped proteomics admission audit",
        source_script=source,
        agent_label="codex",
        metadata=metadata,
    )
    write_reproducibility_bundle(
        out_dir=DIRECTORY,
        source_script=source,
        environment={
            name: os.environ[name]
            for name in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            if name in os.environ
        },
    )
    started = time.monotonic()
    print("Admitting the pinned public scope and collapsing all occurrences...", flush=True)
    records, receipt = load_scoped_bulk_records(
        bundle_sha256=PIN,
        max_records=LIMIT,
        seed=SEED,
    )
    assert len(records) == LIMIT
    assert all(row.n_replicates_detected is None and row.excision_label is None for row in records)
    print("Admitted counts:", receipt["n_records_by_evidence"], flush=True)
    dataset = PrestoDataset(bulk_ms_records=records)
    assert len(dataset) == len({row.sample_id for row in dataset}) == LIMIT
    assert all(row.source_bundle_id == f"sha256:{PIN}" for row in dataset)
    assert all(row.candidate_lookup_key == row.peptide for row in dataset)
    with OutputCoverageCensus(build_output_contract()) as census:
        audit = audit_split_support(
            {"admitted": dataset}, collator=PrestoCollator(), output_census=census
        )
        assert audit["lineage"]["issue_count"] == 0
        coverage = census.report()
        detected = census.counts("admitted", "ms_detectability_logit", roles=["proxy"])
        assert detected["observations"] == LIMIT
        assert (
            census.counts("admitted", "ms_detectability_logit", roles=["direct"])["observations"]
            == 0
        )
        assert census.counts("admitted", "excision_logit")["observations"] == 0
    membership = [
        {
            "sample_id": row.sample_id,
            "peptide": row.peptide,
            "detectability_label": row.ms_detectability_label,
            "excision_label": row.excision_label,
            "flank_n": row.flank_n,
            "flank_c": row.flank_c,
            **source_lineage_fields(row, include_scoped=True),
        }
        for row in dataset
    ]
    assert all(row["candidate_lookup_key"] == row["peptide"] for row in membership)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    summary = {
        "status": "admission_audit_passed",
        "experiment": DIRECTORY.name,
        "agent": "codex / GPT-6",
        "git_commit": commit,
        "runtime_seconds": time.monotonic() - started,
        "runtime_scope": "admission and support audit before artifact publication",
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "gpu": "none",
            "torch_threads": torch.get_num_threads(),
        },
        "imports": {"presto": presto.__file__, "hitlist": hitlist.__file__},
        "versions": {
            "presto": presto.__version__,
            "hitlist": hitlist.__version__,
            **{
                name: importlib.metadata.version(name) for name in ("torch", "pyarrow", "datacache")
            },
        },
        "n_records": len(records),
        "n_candidate_linked_records": len(records),
        "n_source_observation_linked_records": sum(bool(row.evidence_row_id) for row in records),
        "n_records_by_evidence": receipt["n_records_by_evidence"],
        "full_coverage": receipt["full_coverage"],
        "canonical_lineage_audit": audit["lineage"],
        "detectability_support": detected,
        "partitions": "not_assigned; admission audit only",
        "validation_metrics": "not_applicable; no fitting or evaluation",
        "test_metrics": "not_applicable; no fitting or evaluation",
        "prediction_exports": "not_produced; no model inference",
        "limitations": receipt["source_registry"]["limitations"],
        "traceability_note": (
            "Canonical census traceable_observations counts source observations; "
            "scoped absence has candidate linkage, not a measured observation ID."
        ),
    }
    output_names = (
        "admitted.tsv",
        "admitted.parquet",
        "summary.json",
        "source_receipt.json",
        "output_coverage.json",
        "output_coverage.csv",
        "split_support.json",
        "split_support.csv",
        "mil_split_support.csv",
        "row_split_support.csv",
        "README.md",
    )

    def build(inputs, outputs):
        destination = Path(outputs["summary.json"]).parent
        with Path(outputs["admitted.tsv"]).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(membership[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(membership)
        pq.write_table(
            pa.Table.from_pylist(membership), outputs["admitted.parquet"], compression="zstd"
        )
        write_json(outputs["summary.json"], summary)
        write_json(outputs["source_receipt.json"], receipt)
        write_output_coverage_artifacts(destination, coverage)
        support_paths = write_split_support_artifacts(destination, audit)
        assert {path.name for path in support_paths.values()} == {
            "split_support.json",
            "split_support.csv",
            "mil_split_support.csv",
            "row_split_support.csv",
        }
        Path(outputs["README.md"]).write_text(
            "# Published-scope admission audit\n\n"
            "10,000 post-collapse reservoir peptides; seed 42. All rows retain frozen "
            "candidate lookup and scope identity. Positive observations retain source IDs; "
            "absence has no invented measured ID. See source_receipt.json for input and "
            "projection hashes, citations, cell/assay/search contracts and limitations.\n\n"
            "No partitions, training or held-out evaluation. This is admission evidence, "
            "not qualification of intrinsic detectability or biological independence.\n"
        )

    sources = {
        name: {
            "path": str(Path(receipt["projection"]["directory"]) / name),
            "sha256": item["sha256"],
            "size": item["size_bytes"],
        }
        for name, item in receipt["projection"]["artifacts"].items()
    }
    artifacts = materialize(
        data_dir() / "presto-scoped-proteomics" / "qualification" / DIRECTORY.name,
        sources,
        transform={
            "version": "scoped-admission-audit-v1",
            "options": {
                "manifest_sha256": PIN,
                "max_records": LIMIT,
                "seed": SEED,
                "code_commit": commit,
                "launcher_sha256": digest(__file__),
            },
        },
        outputs={name: {} for name in output_names},
        builder=build,
        download_options={"show_progress": False},
    )
    artifacts = {
        name: {"path": str(path), "sha256": digest(path), "size_bytes": Path(path).stat().st_size}
        for name, path in artifacts.items()
    }
    summary = json.loads(Path(artifacts["summary.json"]["path"]).read_text())
    write_json(DIRECTORY / "results" / "summary.json", summary)
    write_json(DIRECTORY / "results" / "artifacts.json", artifacts)
    print(json.dumps({"summary": summary, "artifacts": artifacts}, indent=2))
    signal.alarm(0)


if __name__ == "__main__":
    run()
