"""Software differential parity for output-coverage count reuse against a selected base.

No training, predictive evaluation or runtime claim. The unmodified base census is
loaded from git history beside the working-tree census. Both consume the
canonical-coverage fixtures in identical chunks, with class counts requested
between chunks so cached selections must survive appends correctly. Reports,
repeated reports and every filtered class count must match exactly. SQL statement
totals and public ``counts()`` calls per report are recorded so the change in the
report count-call metric is explicit.

Run from the repository root with explicit ``--base-commit`` and ``--output``.
The family README records the current integration command. Historical source and
results remain frozen separately.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from presto.data.collate import TCR_EVIDENCE_METHOD_BINS, PrestoCollator
from presto.training.output_contract import (
    RESIDUAL_MODES,
    OutputConfiguration,
    build_output_contract,
)
from presto.training.output_coverage import OutputCoverageCensus

ROOT = Path(__file__).resolve().parents[3]
FAMILY = Path(__file__).resolve().parents[1]
FILTERS = (
    {},
    {"sources": ["iedb"]},
    {"sources": ["other"]},
    {"sources": ["iedb", "other"]},
    {"roles": ["auxiliary"]},
    {"roles": []},
    {"families": ["absent"]},
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counting(census_type):
    class Counting(census_type):
        calls = 0

        def counts(self, *args, **kwargs):
            self.calls += 1
            return super().counts(*args, **kwargs)

    return Counting


def traced(census, action):
    statements, calls = [], census.calls
    census.db.set_trace_callback(statements.append)
    try:
        value = action()
    finally:
        census.db.set_trace_callback(None)
    return value, len(statements), census.calls - calls


def fixture_rows():
    unit = load("coverage_fixtures", ROOT / "tests/test_output_coverage.py")
    canonical = load("canonical_fixtures", FAMILY / "code/test_census.py")
    rows = canonical.samples()
    for split, examples in rows.items():
        extra = unit.sample(
            sample_id=f"{split}-extra",
            evidence_row_id=f"{split}-extra",
            bind_value=25,
            bind_measurement_type="KD",
            binding_assay_type="KD",
            elution_label=1,
            tcr_evidence_label=1,
            tcr_evidence_method_bins=(TCR_EVIDENCE_METHOD_BINS[0],),
        )
        examples.extend(
            [
                extra,
                replace(extra, sample_id=f"{split}-repeat"),
                replace(extra, sample_id=f"{split}-conflicting", mhc_class="II"),
                replace(
                    extra,
                    sample_id=f"{split}-other",
                    evidence_row_id=f"{split}-other",
                    sample_source="other",
                    bind_qual=1,
                    bind_value=100,
                    mhc_class="II",
                ),
            ]
        )
    return rows


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main(base_commit, output):
    torch.set_num_threads(1)
    base_commit = git("rev-parse", base_commit)
    rows = fixture_rows()
    base_source = subprocess.check_output(
        ["git", "show", f"{base_commit}:training/output_coverage.py"], cwd=ROOT
    )
    results, comparisons = [], 0
    with tempfile.TemporaryDirectory() as directory:
        base_path = Path(directory) / "base.py"
        base_path.write_bytes(base_source)
        base = load("presto.training.output_coverage_base", base_path)
        for topology, residual, grouping in itertools.product(
            ("collapsed", "expanded"), RESIDUAL_MODES, ("merged_kd", "split_kd_proxy")
        ):
            config = OutputConfiguration(
                latent_topology=topology,
                affinity_assay_residual_mode=residual,
                kd_grouping_mode=grouping,
            )
            contract = build_output_contract(config)
            classes = {
                name: spec.columns
                for name, spec in contract.outputs.items()
                if spec.shape == "classes"
            }
            with (
                counting(base.OutputCoverageCensus)(contract) as before,
                counting(OutputCoverageCensus)(contract) as after,
            ):
                for split, examples in rows.items():
                    for offset in range(0, len(examples), 2):
                        chunk = examples[offset : offset + 2]
                        batch = PrestoCollator()(chunk)
                        for census in (before, after):
                            census.add_batch(split, chunk, batch)
                            census.counts(split, "mhc_class_logits", "I")
                for census in (before, after):
                    census.declare_split("empty")
                reference, base_statements, base_calls = traced(before, before.report)
                candidate, statements, calls = traced(after, after.report)
                assert reference == candidate, asdict(config)
                assert after.report() == reference, "Repeated report changed cached results"
                for split, (endpoint, columns), filters, facets in itertools.product(
                    ("train", "val", "test", "empty"), classes.items(), FILTERS, (False, True)
                ):
                    for column in ("", *columns):
                        options = dict(include_facets=facets, **filters)
                        assert before.counts(split, endpoint, column, **options) == after.counts(
                            split, endpoint, column, **options
                        ), (asdict(config), split, endpoint, column, options)
                        comparisons += 1
            results.append(
                {
                    "config": asdict(config),
                    "complete_report_equal": True,
                    "repeated_report_equal": True,
                    "report_sha256": hashlib.sha256(
                        json.dumps(reference, sort_keys=True).encode()
                    ).hexdigest(),
                    "report_sql_statements": {"base": base_statements, "candidate": statements},
                    "report_counts_calls": {"base": base_calls, "candidate": calls},
                }
            )
    candidate_source = (ROOT / "training/output_coverage.py").read_bytes()
    summary = {
        "verification": "software differential parity; no timing or predictive claims",
        "base_commit": base_commit,
        "base_source_sha256": hashlib.sha256(base_source).hexdigest(),
        "candidate_commit": git("rev-parse", "HEAD"),
        "candidate_worktree_dirty": bool(git("status", "--porcelain", "training", "tests")),
        "candidate_source_sha256": hashlib.sha256(candidate_source).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples_per_nonempty_split": len(rows["train"]),
        "configurations": results,
        "filtered_class_count_comparisons": comparisons,
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"Exact report and repeated-report parity for {len(results)} configurations; "
        f"{comparisons} filtered class counts equal. {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.base_commit, args.output)
