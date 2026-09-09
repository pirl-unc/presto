"""Observational instrumentation around the canonical, uncapped data preflight."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_files(entries, *, root=None):
    checked = {}
    for entry in entries:
        path = Path(entry["remote_path"]) if root is None else root / entry["relative_path"]
        digest = file_hash(path)
        if digest != entry["sha256"]:
            raise RuntimeError(
                f"Frozen input changed: {path}; expected {entry['sha256']}, got {digest}"
            )
        checked[str(path)] = digest
    return checked


class CandidateSelection:
    """Two smallest sample-ID hashes per observed train stratum, independent of order.

    These candidates only enable later prospective diagnostic selection. They
    are not a fitting subset or evidence that all their columns were optimized.
    """

    def __init__(self):
        self.cells = {}
        self.missing_identity_hits = 0
        self.synthetic_hits = 0

    def add(self, sample, stratum):
        source = str(sample.sample_source or "")
        if sample.synthetic_kind or source == "synthetic" or source.startswith("synthetic_"):
            self.synthetic_hits += 1
            return
        identity = str(sample.sample_id)
        if not identity:
            self.missing_identity_hits += 1
            return
        prefix = (hashlib.sha256(identity.encode()).hexdigest(), identity)
        cell = self.cells.setdefault(stratum, {})
        if len(cell) >= 2 and prefix > max(cell)[:2]:
            return
        payload = dataclasses.asdict(sample)
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        rank = (*prefix, payload_hash)
        if len(cell) < 2 or rank < max(cell):
            cell[rank] = payload
            if len(cell) > 2:
                del cell[max(cell)]

    def write(self, output):
        samples = {}
        strata = []
        for cell, candidates in sorted(self.cells.items()):
            ids = []
            for (_, identity, payload_hash), sample in sorted(candidates.items()):
                candidate_id = f"{identity}:snapshot:{payload_hash}"
                samples[candidate_id] = {"candidate_id": candidate_id, "sample": sample}
                ids.append(candidate_id)
            strata.append({"stratum": cell, "candidate_ids": ids})
        write_json(
            output / "training_candidates.json",
            {
                "selection": (
                    "two minimum SHA256(sample_id) per endpoint/column/role/family/"
                    "source/response-or-qualifier stratum"
                ),
                "split": "train",
                "strata": strata,
                "samples": [samples[key] for key in sorted(samples)],
                "missing_identity_observation_hits": self.missing_identity_hits,
                "excluded_synthetic_observation_hits": self.synthetic_hits,
                "identity_note": (
                    "Snapshot hashes break duplicate sample-ID ties; "
                    "they are not original observation IDs."
                ),
                "note": (
                    "No optimization performed; later update budget/configuration "
                    "must be frozen separately."
                ),
            },
        )


@contextmanager
def retained_census(output):
    from presto.training import coverage_preflight

    original = coverage_preflight.OutputCoverageCensus
    selection = CandidateSelection()

    class RetainedCensus(original):
        def __init__(self, contract):
            super().__init__(contract, path=output / "output_coverage.sqlite")
            self.previous_row = 0
            self.previous_observation = 0
            self.last_progress = 0.0
            self.count_queries = 0

        def add_batch(self, split, samples, batch):
            super().add_batch(split, samples, batch)
            if split == "train":
                # Observation primary-key range keeps this bounded to the new
                # chunk; a row_id-only predicate would scan prior chunks.
                rows = self.db.execute(
                    "SELECT id,row_id,endpoint,column_name,role,family,loss_type,target,qualifier "
                    "FROM observations WHERE id>? ORDER BY id",
                    (self.previous_observation,),
                ).fetchall()
                for (
                    observation,
                    row,
                    endpoint,
                    column,
                    role,
                    family,
                    loss,
                    target,
                    qualifier,
                ) in rows:
                    sample = samples[row - self.previous_row - 1]
                    response = (
                        ("positive" if target > 0.5 else "negative")
                        if loss == "bce"
                        else (f"class={int(target)}" if loss == "ce" else f"qualifier={qualifier}")
                    )
                    cell = (endpoint, column, role, family, str(sample.sample_source), response)
                    if role == "synthetic":
                        selection.synthetic_hits += 1
                    else:
                        selection.add(sample, cell)
                    self.previous_observation = observation
            else:
                self.previous_observation = self.db.execute(
                    "SELECT COALESCE(MAX(id),0) FROM observations"
                ).fetchone()[0]
            self.previous_row += len(samples)
            if time.monotonic() - self.last_progress >= 30:
                self.progress("collation")

        def progress(self, phase):
            self.last_progress = time.monotonic()
            payload = {
                "phase": phase,
                "split_rows": dict(self.splits),
                "count_queries": self.count_queries,
                "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "time_unix": time.time(),
            }
            write_json(output / "progress.json", payload)
            print("Census progress:", json.dumps(payload), flush=True)

        def counts(self, *args, **kwargs):
            if time.monotonic() - self.last_progress >= 30:
                self.progress("summary_queries")
            result = super().counts(*args, **kwargs)
            self.count_queries += 1
            return result

        def report(self):
            selection.write(output)
            write_json(
                output / "census_state.json",
                {
                    "evidence_sha256": self.sha256,
                    "splits": self.splits,
                    "objective_hits": self.objective_hits,
                    "contract": self.contract.to_dict(),
                },
            )
            self.progress("summary_queries")
            result = super().report()
            self.progress("report_complete")
            return result

    coverage_preflight.OutputCoverageCensus = RetainedCensus
    try:
        yield selection
    finally:
        coverage_preflight.OutputCoverageCensus = original


def run(condition):
    import hitlist
    import mhcseqs
    from presto.scripts import train_iedb

    args = json.loads((EXPERIMENT / "conditions.json").read_text())[condition]
    output = Path(args["run_dir"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = json.loads((ROOT / "source_receipt.json").read_text())
    sources = json.loads((EXPERIMENT / "input_manifest.json").read_text())
    state = {
        "condition": condition,
        "status": "running",
        "source": receipt,
        "args": args,
        "started_unix": time.time(),
        "agent_model": "Codex / GPT-6",
        "python": sys.version,
        "platform": platform.platform(),
        "requested_cpu": [4, 8],
        "requested_memory_mib": [65536, 196608],
        "gpu": None,
        "timeout_seconds": 14400,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "HITLIST_DATA_DIR",
                "PRESTO_MHCSEQS_SEARCH_DIR",
                "TQDM_DISABLE",
            )
        },
    }
    write_json(output / "invocation.json", state)
    write_json(
        output / "environment.json",
        {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
    )
    hardware = {}
    for path in (
        "/proc/meminfo",
        "/proc/cpuinfo",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/cpu.max",
    ):
        if Path(path).is_file():
            hardware[path] = Path(path).read_text()
    write_json(output / "hardware.json", hardware)
    try:
        package_files = {}
        for package in (hitlist, mhcseqs):
            directory = Path(package.__file__).parent
            package_files[package.__name__] = {
                str(path.relative_to(directory)): file_hash(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.suffix in {".py", ".yaml", ".json"}
            }
        write_json(output / "package_sources.json", package_files)
        expected_curation = "e0270f2b417619a318ef03549e7cb7d46231bb3dbb6a088d8367a239cd6b3308"
        if package_files["hitlist"]["data/pmid_overrides.yaml"] != expected_curation:
            raise RuntimeError("Hitlist curation differs from the frozen inventory")
        verify_files(receipt["files"], root=ROOT)
        state["sources_before"] = verify_files(sources)
        if not Path(train_iedb.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError(f"Unexpected Presto import: {train_iedb.__file__}")
        with retained_census(output):
            train_iedb.run(argparse.Namespace(**args))
        state["sources_after"] = verify_files(sources)
        verify_files(receipt["files"], root=ROOT)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - started
        state["peak_rss_native"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(output / "status.json", state)
