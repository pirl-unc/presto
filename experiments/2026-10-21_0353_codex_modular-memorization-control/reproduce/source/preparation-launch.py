"""Frozen real-data capacity check; no held-out predictive qualification."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from presto.data.groove import groove_record
from presto.data.modular import AA, MHCInput, Observation, observation_target
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    file_sha256,
    write_observations,
)
from presto.models.modular import ModularConfig, ModularPresto
from presto.models.modular_support import feature_report
from presto.training.modular import load_checkpoint, save_checkpoint, train_step
from presto.training.modular_evaluation import evaluate, regression_metrics
from presto.training.modular_run import _accumulate_coverage, preflight


ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "experiments/2026-10-21_0353_codex_modular-memorization-control"
COHORT = ROOT / "data/modular_cohorts/2026-10-21-memorization-control"
SOURCES = {
    "binding": {
        "path": "/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/"
        "bundles/2026-10-16T07-20-43Z/binding.parquet",
        "sha256": "2b3f6ea2dd647e7fc54d214eef420feaebc4e81fab97ffc0565a2de1f8695ba0",
    },
    "catalog": {
        "path": "/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv",
        "sha256": "ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1",
    },
    "historical_membership": {
        "path": "/Users/iskander/.hitlist/presto-corrected-human-qualification/"
        "public-downloads/v1/bundles/2026-10-16T08-32-18Z/membership.tsv",
        "sha256": "8959e60c0d9b8049aec25095b9c354d456b48f97ba8fb31d7b786e2861113e99",
    },
}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_lines(path, records):
    with Path(path).open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def receipt():
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "python": sys.version,
        "executable": sys.executable,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pyarrow", "mhcseqs", "mhcgnomes")
        },
        "environment": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "argv": sys.argv,
        "agent_model": "Codex / GPT-6",
        "device": "cpu",
        "gpu": None,
    }


def catalog_rows():
    with Path(SOURCES["catalog"]["path"]).open() as stream:
        rows = list(csv.DictReader(stream))
    # This catalog's canonical human prefix is explicit and pinned, not guessed
    # from a cell/species annotation in the assay table.
    return {
        row["two_field_allele"].replace("HomoSapiens-", "HLA-", 1): row
        for row in rows
        if row["source"] == "imgt" and row["species"] == "Homo sapiens"
    }


def resolved_pair(restriction, mhc_class, catalog):
    tokens = restriction.split("/")
    tokens = [token if token.startswith("HLA-") else "HLA-" + token for token in tokens]
    if len(tokens) != (1 if mhc_class == "I" else 2):
        raise ValueError("missing explicit chain restriction")
    records = [catalog.get(token) for token in tokens]
    if any(record is None for record in records):
        raise ValueError("unresolved exact catalog allele")
    for record in records:
        if (
            record["mhc_class"] != mhc_class
            or record["groove_status"] != "ok"
            or record["is_functional"] != "True"
            or any(
                record[name] != "False" for name in ("is_null", "is_questionable", "is_pseudogene")
            )
        ):
            raise ValueError("unsupported catalog groove or allele flag")
    if mhc_class == "I":
        if records[0]["chain"] != "alpha":
            raise ValueError("class-I restriction is not an alpha chain")
        first, second = records[0]["groove1"], records[0]["groove2"]
    else:
        by_chain = {row["chain"]: row for row in records}
        if set(by_chain) != {"alpha", "beta"}:
            raise ValueError("class-II restriction lacks a distinct alpha/beta pair")
        first, second = by_chain["alpha"]["groove1"], by_chain["beta"]["groove2"]
    pair = MHCInput(
        first,
        second,
        mhc_class,
        species1="human",
        species2="human",
        provenance={
            "reported_restriction": restriction,
            "catalog_alleles": tokens,
            "partner_imputation": False,
        },
    )
    pair.validate()
    return pair, tokens


def prepare(cohort, output):
    if cohort.exists():
        raise ValueError("refusing to overwrite a frozen cohort")
    for source in SOURCES.values():
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError("source checksum mismatch")
    cohort.mkdir(parents=True)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "reproduce/preparation-launch.json", receipt())
    dump(cohort / "source-pins.json", SOURCES)
    with Path(SOURCES["historical_membership"]["path"]).open() as stream:
        heldout = {
            row["peptide"]
            for row in csv.DictReader(stream, delimiter="\t")
            if row["source_partition"] in ("val", "validation", "test")
        }
    if not heldout:
        raise ValueError("historical held-out exclusion set is empty")
    catalog = catalog_rows()
    candidates, rejected = [], Counter()
    columns = [
        "peptide",
        "mhc_restriction",
        "mhc_class",
        "response_measured",
        "measurement_units",
        "measurement_inequality",
        "quantitative_value",
        "quantitative_measurement",
        "assay_method",
        "source",
        "assay_iri",
        "reference_iri",
        "pmid",
        "provenance_id",
        "has_ptm",
        "peptide_modifications",
    ]
    ordinal = 0
    with gzip.open(output / "results/source-dispositions.jsonl.gz", "wt") as dispositions:
        for batch in pq.ParquetFile(SOURCES["binding"]["path"]).iter_batches(
            columns=columns, batch_size=65536
        ):
            for raw in batch.to_pylist():
                ordinal += 1
                reason = None
                try:
                    if (
                        raw["source"] != "iedb"
                        or raw["response_measured"]
                        != "half maximal inhibitory concentration (IC50)"
                    ):
                        raise ValueError("outside direct IEDB IC50 scope")
                    if raw["measurement_units"] != "nM" or raw["measurement_inequality"] not in (
                        None,
                        "",
                        "=",
                    ):
                        raise ValueError("not an exact nM observation")
                    if not (raw["assay_method"] or "").startswith("purified MHC/"):
                        raise ValueError("not a purified-MHC assay")
                    if raw["has_ptm"] or raw["peptide_modifications"]:
                        raise ValueError("modified peptide")
                    peptide, cls = raw["peptide"], raw["mhc_class"]
                    if (
                        not isinstance(peptide, str)
                        or set(peptide) - set(AA)
                        or cls not in ("I", "II")
                        or not (8 if cls == "I" else 10) <= len(peptide) <= 25
                    ):
                        raise ValueError("outside canonical peptide/class/length scope")
                    if peptide in heldout:
                        raise ValueError("historical held-out peptide")
                    value = raw["quantitative_value"]
                    if (
                        value is None
                        or not math.isfinite(value)
                        or value <= 0
                        or not raw["assay_iri"]
                    ):
                        raise ValueError("missing finite positive outcome or original assay ID")
                    if float(raw["quantitative_measurement"]) != value:
                        raise ValueError("raw text/numeric target disagreement")
                    restriction = raw["mhc_restriction"] or ""
                    if not restriction.startswith("HLA-"):
                        raise ValueError("not an explicit human HLA restriction")
                    pair, tokens = resolved_pair(restriction, cls, catalog)
                    key = json.dumps((peptide, cls, pair.groove1, pair.groove2))
                    candidates.append(
                        {"ordinal": ordinal, "raw": raw, "pair": pair, "tokens": tokens, "key": key}
                    )
                except (ValueError, TypeError) as error:
                    reason = str(error)
                    rejected[reason] += 1
                dispositions.write(
                    json.dumps(
                        {
                            "ordinal": ordinal,
                            "assay_iri": raw["assay_iri"],
                            "disposition": "excluded" if reason else "candidate_pending",
                            "reason": reason,
                        }
                    )
                    + "\n"
                )
    by_assay, by_input = defaultdict(list), defaultdict(list)
    for candidate in candidates:
        by_assay[candidate["raw"]["assay_iri"]].append(candidate)
        by_input[candidate["key"]].append(candidate)
    conflicts = {
        key
        for key, group in by_input.items()
        if len({item["raw"]["quantitative_value"] for item in group}) > 1
    }
    bad_assays = {
        key
        for key, group in by_assay.items()
        if len({(item["key"], item["raw"]["quantitative_value"]) for item in group}) > 1
    }
    consistent = [
        min(group, key=lambda item: (item["raw"]["assay_iri"], item["ordinal"]))
        for key, group in by_input.items()
        if key not in conflicts
        and not any(item["raw"]["assay_iri"] in bad_assays for item in group)
    ]
    selected = []
    for cls in ("I", "II"):
        choices = sorted(
            (item for item in consistent if item["raw"]["mhc_class"] == cls),
            key=lambda item: hashlib.sha256(("42:" + item["key"]).encode()).hexdigest(),
        )[:16]
        if (
            len(choices) != 16
            or np.ptp([math.log10(item["raw"]["quantitative_value"]) for item in choices]) < 1
        ):
            raise ValueError("fixed class selection lacks 16 inputs or one log10 unit of range")
        selected.extend(choices)
    selected_ordinals = {item["ordinal"] for item in selected}
    selected_tokens = sorted({token for item in selected for token in item["tokens"]})
    for token in selected_tokens:
        record = catalog[token]
        parsed = groove_record(
            record["sequence"], mhc_class=record["mhc_class"], chain=record["chain"]
        )
        if (parsed.groove_half_1, parsed.groove_half_2, parsed.status) != (
            record["groove1"],
            record["groove2"],
            record["groove_status"],
        ):
            raise ValueError(
                f"current parser disagrees with pinned selected catalog record: {token}"
            )
    write_lines(
        output / "results/candidate-dispositions.jsonl",
        [
            {
                "ordinal": item["ordinal"],
                "assay_iri": item["raw"]["assay_iri"],
                "input_sha256": hashlib.sha256(item["key"].encode()).hexdigest(),
                "disposition": "selected"
                if item["ordinal"] in selected_ordinals
                else "excluded_inconsistent_exact_input"
                if item["key"] in conflicts or item["raw"]["assay_iri"] in bad_assays
                else "not_selected_or_duplicate_input",
                "value_nM": item["raw"]["quantitative_value"],
            }
            for item in candidates
        ],
    )
    dump(
        output / "results/selected-catalog.json",
        {token: catalog[token] for token in selected_tokens},
    )
    rows = [
        Observation(
            "ic50:" + item["raw"]["assay_iri"],
            "hitlist-v4-iedb",
            "ic50",
            item["raw"]["peptide"],
            (item["pair"],),
            value=item["raw"]["quantitative_value"],
            unit="nM",
            split="train",
            group_id="peptide:" + item["raw"]["peptide"],
            provenance={
                "measurement_namespace": "iedb",
                "original_measurement_id": item["raw"]["assay_iri"],
                "raw_record": item["raw"],
                "source_sha256": SOURCES["binding"]["sha256"],
                "source_ordinal": item["ordinal"],
                "target_policy": "physical_unclipped_v1",
            },
        )
        for item in selected
    ]
    written = write_observations(cohort / "observations.jsonl", rows)
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": asdict(
            ModularConfig(
                endpoints=("ic50",), mhc_classes=("I", "II"), species=("human",), dropout=0
            )
        ),
        "training": {
            "seed": 42,
            "batch_size": 32,
            "learning_rate": 0.0003,
            "weight_decay": 0,
            "max_epochs": 2000,
            "max_steps": 2000,
            "identity_weight": 0,
            "test_policy": "no_test_training_diagnostic",
            "no_test_reason": "Registered 32-example capacity diagnostic; "
            "all rows are fit targets, historical held-out peptides excluded.",
        },
        "sources": [
            {
                "path": written["path"],
                "sha256": written["sha256"],
                "source": "hitlist-v4-iedb",
                "population": "32 consistent exact purified IC50 inputs, "
                "16 I and 16 explicitly paired II",
                "label_policy": "raw exact log10(nM); no generated labels or clipping",
                "split_policy": "training-only, no held-out inference",
            }
        ],
    }
    dump(cohort / "manifest.json", manifest)
    _, _, _, admission = preflight(ObservationIndex(cohort / "manifest.json"))
    dump(output / "results/admission.json", admission)
    dump(
        output / "results/curation.json",
        {
            "raw_rows": ordinal,
            "excluded": dict(rejected),
            "candidates": len(candidates),
            "consistent_unique_inputs": len(consistent),
            "conflicting_input_groups": len(conflicts),
            "conflicting_original_assays": len(bad_assays),
            "historical_heldout_peptides_excluded": len(heldout),
            "selected": 32,
            "selected_ids": [row.observation_id for row in rows],
            "manifest_path": str(cohort / "manifest.json"),
            "manifest_sha256": file_sha256(cohort / "manifest.json"),
            "source_pins": SOURCES,
        },
    )
    print(
        json.dumps(
            {
                "prepared": 32,
                "candidates": len(candidates),
                "conflicts": len(conflicts),
                "manifest": str(cohort / "manifest.json"),
            }
        ),
        flush=True,
    )


def gate(metrics):
    exact = metrics["ic50"]["exact"]
    checks = {"overall": exact, **metrics["ic50"]["by_class"]}
    return set(checks) == {"overall", "I", "II"} and all(
        value["rmse"] <= 0.1
        and value["spearman"] is not None
        and value["spearman"] >= 0.95
        and 0.8 <= value["sd_ratio"] <= 1.2
        for value in checks.values()
    )


def fit(cohort, output):
    results = output / "results"
    if (results / "summary.json").exists() or (results / "history.jsonl").exists():
        raise ValueError("refusing to rerun or overwrite a diagnostic")
    prepared = json.loads((results / "curation.json").read_text())
    if file_sha256(cohort / "manifest.json") != prepared["manifest_sha256"]:
        raise ValueError("prepared manifest changed before fitting")
    index = ObservationIndex(cohort / "manifest.json")
    config, training, scales, admission = preflight(index)
    rows = list(index)
    if len(rows) != 32 or set(index.partitions) != {"train:assay"}:
        raise ValueError("diagnostic membership differs from the 32-row training-only contract")
    repro = output / "reproduce"
    dump(repro / "launch.json", receipt())
    shutil.copyfile(__file__, repro / "source/launch.py")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = ModularPresto(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0)
    coverage, history = {}, []
    raw_dir = ROOT / "artifacts" / output.name
    raw_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "manifest_sha256": index.manifest_sha256,
        "contract": training,
        "qualification": "training_only_not_qualified",
    }
    save_checkpoint(raw_dir / "initial.pt", model, scales, optimizer=optimizer, metadata=metadata)
    start = time.monotonic()
    metrics, predictions = evaluate(model, rows, range(32), scales)
    write_lines(results / "initial-predictions.jsonl", predictions)
    history.append(
        {"step": 0, "elapsed_seconds": time.monotonic() - start, "metrics": metrics, "gate": False}
    )
    reached, reason, step = False, "max_steps", 0
    for step in range(1, 2001):
        evidence = train_step(model, optimizer, rows, scales)
        _accumulate_coverage(coverage, evidence)
        if step % 25 == 0 or time.monotonic() - start >= 1200 or step == 2000:
            metrics, predictions = evaluate(model, rows, range(32), scales)
            reached = gate(metrics)
            history.append(
                {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - start,
                    "loss": evidence["loss"],
                    "metrics": metrics,
                    "gate": reached,
                }
            )
            write_lines(results / "history.jsonl", history)
            write_lines(results / f"predictions-step-{step:04d}.jsonl", predictions)
            save_checkpoint(
                raw_dir / "latest.pt",
                model,
                scales,
                optimizer=optimizer,
                metadata={**metadata, "step": step, "coverage": coverage},
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "rmse": metrics["ic50"]["exact"]["rmse"],
                        "gate": reached,
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
            if reached:
                reason = "fixed_training_gate_passed"
                break
            if time.monotonic() - start >= 1200:
                reason = "optimization_wall_time_limit"
                break
    save_checkpoint(
        raw_dir / "final.pt",
        model,
        scales,
        optimizer=optimizer,
        metadata={**metadata, "step": step, "coverage": coverage},
    )
    restored, restored_scales, _ = load_checkpoint(raw_dir / "final.pt")
    replay_metrics, replay_predictions = evaluate(restored, rows, range(32), restored_scales)
    if replay_predictions != predictions:
        raise ValueError("fresh checkpoint predictions differ")
    restored.predict(rows)  # Exercise public trained-feature/parameter guards too.
    write_lines(results / "final-predictions.jsonl", replay_predictions)
    features = feature_report(restored)
    missing = [name for name, value in coverage.items() if not value["gradient_and_update_steps"]]
    missing_features = {
        name: value["exposed_without_update"]
        for name, value in features.items()
        if value["exposed_without_update"]
    }
    targets = [observation_target(row) for row in rows]
    summary = {
        "decision": "passed_training_only_gate"
        if reached and not missing and not missing_features
        else "failed_training_only_gate",
        "stopping_reason": reason,
        "steps": step,
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "metrics": replay_metrics,
        "mean_target_training_comparator": regression_metrics(
            targets, [float(np.mean(targets))] * 32, binding=True
        ),
        "coverage": {
            "untrained_parameters": missing,
            "exposed_features_without_updates": missing_features,
        },
        "fresh_checkpoint_replay": "identical",
        "validation": None,
        "test": None,
        "no_test_reason": training["no_test_reason"],
        "raw_artifacts": str(raw_dir),
        "checkpoint_sha256": file_sha256(raw_dir / "final.pt"),
        "qualification": "not_qualified_for_generalization",
        "device": "cpu",
        "gpu": None,
        "source_manifest_sha256": index.manifest_sha256,
    }
    dump(results / "coverage.json", {"parameters": coverage, "features": features})
    dump(results / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "steps": step,
                "artifact": str(results / "summary.json"),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit"))
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--output", type=Path, default=FAMILY)
    args = parser.parse_args()
    for subdir in ("results", "reproduce/source", "launch_logs"):
        (args.output / subdir).mkdir(parents=True, exist_ok=True)
    (prepare if args.phase == "prepare" else fit)(args.cohort.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
