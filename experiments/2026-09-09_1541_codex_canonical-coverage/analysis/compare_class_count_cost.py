"""Synthetic SQL cost of class-endpoint counts: base, PR #66 head and the working tree.

Software check only, on synthetic rows and the local machine. It is not a real-data
or full-report runtime claim; the reuse spec requires a registered family for that.
Two workloads run on the 11-class ``species_of_origin_logits`` endpoint:

- ``cold_single_column_claim``: one filtered class-column count with nothing cached,
  the supported-output gate's least favourable case for reuse.
- ``report_selection_work``: what ``report()`` asks of one class endpoint across the
  overall population and three source groups, the population and then every column.

Statement totals are deterministic. Wall times are the best of interleaved
repetitions and move with machine load. Every implementation must return identical
counts.

Run from the repository root:
``python experiments/2026-09-09_1541_codex_canonical-coverage/analysis/compare_class_count_cost.py [samples]``
"""

import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from presto.training.output_contract import build_output_contract

ROOT = Path(__file__).resolve().parents[3]
FAMILY = Path(__file__).resolve().parents[1]
REVISIONS = {
    "base_8b5f60f": "8b5f60fd40684b8af5edfae3c3c52a42fc8b2bfa",
    "pr66_3532ae1": "3532ae1e193b9c7881f4e1d4e0ac10936d5a97d8",
    "working_tree": None,
}
ENDPOINT = "species_of_origin_logits"
SOURCES = ("iedb", "mcpas", "vdjdb")
REPETITIONS = 3


def load(label, source, directory):
    path = Path(directory) / f"{label}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(f"presto.training._coverage_{label}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_rows(samples, classes):
    rng = random.Random(0)
    sample_rows, facet_rows, observation_rows = [], [], []
    for row_id in range(1, samples + 1):
        source = SOURCES[row_id % len(SOURCES)]
        sample_rows.append((row_id, "train", f"s{row_id}", source, f"o{row_id}", 1, f"P{row_id % 5000}"))
        facet_rows.extend(
            [
                (row_id, "mhc_class", rng.choice(("I", "II"))),
                (row_id, "species", rng.choice(("human", "mouse"))),
                (row_id, "peptide_source", rng.choice(("self", "viral", "tumor"))),
                (row_id, "resolved_allele", f"HLA-{rng.randrange(50)}"),
            ]
        )
        for occurrence in range(1 if rng.random() < 0.8 else 2):
            key = f"k{row_id}" if rng.random() < 0.9 else f"k{rng.randrange(samples)}"
            observation_rows.append(
                (row_id, ENDPOINT, "", "direct", "measured", "x", key,
                 float(rng.randrange(len(classes))), None, "ce", 1, "row", occurrence)
            )
    return sample_rows, facet_rows, observation_rows


def cold_single_column_claim(census, classes):
    return census.counts(
        "train", ENDPOINT, classes[3], roles=["direct"], families=["measured"], sources=["iedb"]
    )


def report_selection_work(census, classes):
    results = []
    selections = [{}] + [
        dict(roles=["direct"], families=["measured"], sources=[source]) for source in SOURCES
    ]
    for filters in selections:
        results.append(census.counts("train", ENDPOINT, **filters))
        results.extend(census.counts("train", ENDPOINT, column, **filters) for column in classes)
    return results


def measure(census, work, classes, within_report):
    # PR #66's first cache was global; clear it so every repetition starts cold.
    if getattr(census, "_class_counts", None) is not None:
        census._class_counts.clear()
    # The working tree reuses counts only inside report(), so report work runs in a
    # reuse block and a supported-output claim does not.
    reuse = getattr(census, "reusing_counts", None)
    scope = reuse() if within_report and reuse is not None else contextlib.nullcontext()
    statements = []
    census.db.set_trace_callback(statements.append)
    start = time.perf_counter()
    with scope:
        value = work(census, classes)
    seconds = time.perf_counter() - start
    census.db.set_trace_callback(None)
    return value, len(statements), seconds


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    contract = build_output_contract()
    classes = contract.outputs[ENDPOINT].columns
    sample_rows, facet_rows, observation_rows = synthetic_rows(samples, classes)
    workloads = {
        "cold_single_column_claim": (cold_single_column_claim, False),
        "report_selection_work": (report_selection_work, True),
    }
    sources, censuses = {}, {}
    with tempfile.TemporaryDirectory() as directory:
        for label, revision in REVISIONS.items():
            source = (
                (ROOT / "training/output_coverage.py").read_bytes()
                if revision is None
                else subprocess.check_output(
                    ["git", "show", f"{revision}:training/output_coverage.py"], cwd=ROOT
                )
            )
            sources[label] = hashlib.sha256(source).hexdigest()
            census = load(label, source, directory).OutputCoverageCensus(contract)
            census.db.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?)", sample_rows)
            census.db.executemany("INSERT INTO facets VALUES (?,?,?)", facet_rows)
            census.db.executemany(
                "INSERT INTO observations(row_id,endpoint,column_name,role,family,origin,"
                "observation_key,target,qualifier,loss_type,instances,scope,occurrence) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                observation_rows,
            )
            census.db.commit()
            censuses[label] = census
        timings = {label: {name: [] for name in workloads} for label in REVISIONS}
        statements = {label: {} for label in REVISIONS}
        reference = {}
        for repetition in range(REPETITIONS):
            for label, census in censuses.items():
                for name, (work, within_report) in workloads.items():
                    value, count, seconds = measure(census, work, classes, within_report)
                    reference.setdefault(name, value)
                    assert value == reference[name], (label, name)
                    assert statements[label].setdefault(name, count) == count
                    timings[label][name].append(round(seconds, 4))
                    print(f"rep {repetition} {label:14} {name:26} {count:3} statements {seconds:7.2f}s", flush=True)
        for census in censuses.values():
            census.close()
    summary = {
        "verification": "synthetic software cost check; not a real-data or full-report runtime claim",
        "endpoint": ENDPOINT,
        "classes": len(classes),
        "samples": samples,
        "observations": len(observation_rows),
        "facet_rows": len(facet_rows),
        "repetitions": REPETITIONS,
        "identical_counts_across_implementations": True,
        "implementation_source_sha256": sources,
        "machine": {
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
        "sql_statements": statements,
        "best_seconds": {
            label: {name: min(values) for name, values in per.items()}
            for label, per in timings.items()
        },
        "all_seconds": timings,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = FAMILY / "results/verification/class_count_cost.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sql_statements": statements, "best_seconds": summary["best_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
