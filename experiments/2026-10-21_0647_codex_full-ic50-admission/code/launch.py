"""Frozen data admission; no model predictions, fitting or reselection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
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

from presto.data.groove import groove_record
from presto.data.modular import Observation, observation_target
from presto.data.modular_hitlist import AdmissionError, HumanMHCCatalog, purified_ic50_observation
from presto.data.modular_lineage import measurement_key
from presto.data.modular_preparation import prepare_observations
from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, file_sha256
from presto.training.modular_run import preflight


ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "experiments/2026-10-21_0647_codex_full-ic50-admission"
COHORT = ROOT / "data/modular_cohorts/2026-10-21-full-ic50"
SALT = "presto-direct-ic50-v1"
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
    "memorization": {
        "path": str(
            ROOT / "data/modular_cohorts/2026-10-21-memorization-control/observations.jsonl"
        ),
        "sha256": "145957e201c596ba82db4c900eb3715c34c81605ffadf887d0811f6f1d282ad5",
    },
}
POPULATION = (
    "all source-admitted direct IEDB/CEDAR purified human HLA IC50 within canonical sequence scope"
)
LABEL_POLICY = (
    "physical unclipped IC50; exact/censored raw bounds; "
    "absent export qualifier means exact with missingness retained"
)
SPLIT_POLICY = (
    "exact-peptide/original-assay components; preserve historical and memorization "
    "assignments; novel fixed hash80/10/10; not study/donor qualified"
)


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def historical_assignments():
    assignments = defaultdict(set)
    with Path(SOURCES["historical_membership"]["path"]).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            split = {"val": "validation"}.get(row["source_partition"], row["source_partition"])
            require(split in ("train", "validation", "test"), "unknown historical split")
            if row["peptide"]:
                assignments[row["peptide"]].add(split)
    with Path(SOURCES["memorization"]["path"]).open() as stream:
        memorized = [json.loads(line) for line in stream]
    require(len(memorized) == 32, "memorization membership changed")
    for row in memorized:
        require(row["split"] == "train", "unexpected memorization split")
        assignments[row["peptide"]].add("train")
    return assignments, {row["peptide"] for row in memorized}


def assign_components(rows, historical):
    """Label-blind union; old assignments override only wholly consistent groups."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in (("peptide", row.peptide), ("measurement", measurement_key(row))):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    assigned, deferred, audit = [], {}, []
    for members in groups.values():
        peptides = sorted({rows[i].peptide for i in members})
        digest = hashlib.sha256(json.dumps(peptides).encode()).hexdigest()
        group = "ic50-component:" + digest
        old = {split for peptide in peptides for split in historical.get(peptide, ())}
        bucket = int(hashlib.sha256(f"{SALT}:{digest}".encode()).hexdigest(), 16) % 100
        split = (
            next(iter(old))
            if len(old) == 1
            else "train"
            if bucket < 80
            else "validation"
            if bucket < 90
            else "test"
        )
        record = {
            "group_id": group,
            "peptides": peptides,
            "historical_splits": sorted(old),
            "split": None if len(old) > 1 else split,
        }
        audit.append(record)
        for i in members:
            row = rows[i]
            if len(old) > 1:
                deferred[row.provenance["source_ordinal"]] = {
                    "disposition": "deferred",
                    "reason": "incompatible_historical_component_assignments",
                    "group_id": group,
                }
            else:
                assigned.append(
                    replace(
                        row,
                        split=split,
                        group_id=group,
                        provenance={
                            **row.provenance,
                            "partition_basis": "historical" if old else "novel_component_hash",
                        },
                    )
                )
    assigned.sort(key=lambda row: row.provenance["source_ordinal"])
    return assigned, deferred, audit


def binding_state(row):
    value = observation_target(row)
    threshold = math.log10(500)
    if row.qualifier == "=":
        return "binder" if value <= threshold else "nonbinder"
    if row.qualifier in ("<", "<=") and value <= threshold:
        return "binder"
    if (row.qualifier == ">" and value >= threshold) or (
        row.qualifier == ">=" and value > threshold
    ):
        return "nonbinder"
    return "ambiguous_bound"


def summary(rows):
    counts, alleles, studies, classes, groups = (
        Counter(),
        Counter(),
        defaultdict(set),
        defaultdict(list),
        defaultdict(list),
    )
    for row in rows:
        cls = row.mhc[0].mhc_class
        counts[(row.split, cls, row.qualifier, row.source, binding_state(row))] += 1
        alleles[(row.split, cls, row.mhc[0].provenance["reported_restriction"])] += 1
        raw = row.provenance["raw_record"]
        study = "pmid:" + str(raw["pmid"]) if raw.get("pmid") else raw.get("reference_iri")
        if study:
            studies[row.split].add(study)
        classes[(row.split, cls)].append(row)
        groups[row.group_id].append(row)
    class_rows, failures = [], []
    for split in ("train", "validation", "test"):
        for cls in ("I", "II"):
            members = classes[(split, cls)]
            exact = [row for row in members if row.qualifier == "="]
            count = Counter(row.group_id for row in exact)
            outcomes = Counter(binding_state(row) for row in exact)
            values = [observation_target(row) for row in members]
            class_rows.append(
                {
                    "split": split,
                    "class": cls,
                    "n": len(members),
                    "exact": len(exact),
                    "exact_groups": len(count),
                    "exact_binding": dict(outcomes),
                    "log10_nM_range": [min(values), max(values)] if values else None,
                    "largest_exact_group": max(count.values(), default=0),
                }
            )
            if len(exact) < 50 or len(count) < 20:
                failures.append(f"{split}/{cls}: fewer than50 exact measurements or20 groups")
            if split != "train" and min(outcomes["binder"], outcomes["nonbinder"]) < 10:
                failures.append(f"{split}/{cls}: fewer than10 exact binders/nonbinders")
    if not any(
        row.split == "train" and row.mhc[0].mhc_class == "II" and len(row.peptide) > 9
        for row in rows
    ):
        failures.append("no class-II multi-register training input")
    return {
        "class_support": class_rows,
        "census": [
            dict(zip(("split", "class", "qualifier", "source", "binding_state"), key), count=n)
            for key, n in sorted(counts.items())
        ],
        "allele_census": [
            dict(zip(("split", "class", "restriction"), key), count=n)
            for key, n in sorted(alleles.items())
        ],
        "component_size_counts": dict(Counter(len(values) for values in groups.values())),
        "study_overlap": {
            f"{first}/{second}": len(studies[first] & studies[second])
            for first, second in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        },
        "unknown_independence": (
            "binding export has no donor/specimen/acquisition identity fields; "
            "shared studies are reported, not resolved"
        ),
        "prior_partition_rows": dict(
            Counter((row.split + ":" + row.provenance["partition_basis"]) for row in rows)
        ),
        "readiness": "failed" if failures else "passed",
        "readiness_failures": failures,
    }


def main(output, cohort):
    import pyarrow.parquet as pq

    started = time.monotonic()
    require(not cohort.exists(), "refusing to overwrite a frozen cohort")
    require(not (output / "results/admission.json").exists(), "admission already exists")
    for source in SOURCES.values():
        require(file_sha256(source["path"]) == source["sha256"], "pinned source mismatch")
    raw_dir = ROOT / "artifacts" / output.name
    require(not raw_dir.exists(), "raw output directory already exists")
    raw_dir.mkdir(parents=True)
    cohort.mkdir(parents=True)
    (output / "results").mkdir(parents=True, exist_ok=True)
    receipt = {
        "agent_model": "Codex / GPT-6",
        "argv": sys.argv,
        "executable": sys.executable,
        "cwd": str(ROOT),
        "python": sys.version,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "environment": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pyarrow", "mhcseqs", "mhcgnomes")
        },
        "sources": SOURCES,
        "catalog_producer_version": "not recorded in source CSV",
        "salt": SALT,
        "training": None,
        "evaluation": None,
        "device": "cpu",
        "gpu": None,
    }
    dump(output / "reproduce/executed-launch.json", receipt)
    dump(output / "reproduce/launch.json", receipt)
    for relative in (
        "data/modular_hitlist.py",
        "data/modular.py",
        "data/modular_lineage.py",
        "data/modular_preparation.py",
        "data/modular_sources.py",
        "training/modular_run.py",
    ):
        destination = output / "reproduce/source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(__file__, output / "reproduce/source/launch.py")
    with Path(SOURCES["catalog"]["path"]).open() as stream:
        catalog = HumanMHCCatalog(csv.DictReader(stream))
    candidates, rejected = [], Counter()
    ledger = raw_dir / "raw-admission.jsonl.gz"
    with gzip.open(ledger, "wt") as stream:
        ordinal = 0
        for batch in pq.ParquetFile(SOURCES["binding"]["path"]).iter_batches(batch_size=16384):
            for raw in batch.to_pylist():
                ordinal += 1
                record = {
                    "ordinal": ordinal,
                    "assay_iri": raw.get("assay_iri"),
                    "source": raw.get("source"),
                    "class": raw.get("mhc_class"),
                    "peptide_length": len(raw["peptide"])
                    if isinstance(raw.get("peptide"), str)
                    else None,
                }
                try:
                    candidates.append(
                        purified_ic50_observation(
                            raw,
                            catalog,
                            source_sha256=SOURCES["binding"]["sha256"],
                            ordinal=ordinal,
                        )
                    )
                    record.update(disposition="candidate_pending", reason=None)
                except AdmissionError as error:
                    record.update(disposition=error.disposition, reason=error.reason)
                    rejected[error.reason] += 1
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    # Check every used alias, not only the arbitrarily first representative.
    aliases = catalog.aliases()
    dump(output / "results/used-catalog-aliases.json", aliases)
    for name, records in aliases.items():
        for record in records:
            parsed = groove_record(
                record["sequence"], mhc_class=record["mhc_class"], chain=record["chain"]
            )
            require(
                (parsed.groove_half_1 or "", parsed.groove_half_2 or "", parsed.status)
                == (record["groove1"], record["groove2"], record["groove_status"]),
                "strict parser disagrees with used catalog assignment: "
                f"{name}/{record['source_id']}",
            )
    historical, memorized = historical_assignments()
    assigned, final, components = assign_components(candidates, historical)
    dump(output / "results/components.json", components)
    prepared = prepare_observations(
        assigned,
        raw_dir / "preparation",
        population=POPULATION,
        label_policy=LABEL_POLICY,
        split_policy=SPLIT_POLICY,
    )
    with (raw_dir / "preparation/dispositions.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            original = record["raw"]["provenance"]["source_ordinal"]
            require(original not in final, "duplicate final source disposition")
            final[original] = {
                "disposition": record["disposition"],
                "reason": record["reason"],
                "group_id": record["raw"]["group_id"],
                "measurement_lineage": record["lineage"],
            }
    disposition_counts = Counter()
    with (
        gzip.open(ledger, "rt") as source,
        gzip.open(output / "results/source-dispositions.jsonl.gz", "wt") as destination,
    ):
        for line in source:
            record = json.loads(line)
            if record["disposition"] == "candidate_pending":
                record.update(final.pop(record["ordinal"]))
            disposition_counts[record["disposition"]] += 1
            destination.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    require(
        not final and sum(disposition_counts.values()) == ordinal, "source row conservation failed"
    )
    rows = []
    for source in prepared["sources"]:
        with Path(source["path"]).open() as stream:
            rows.extend(Observation.from_dict(json.loads(line)) for line in stream)
    require(len(rows) == disposition_counts["admitted"], "canonical observation count differs")
    require(
        all(row.split == "train" for row in rows if row.peptide in memorized),
        "memorized peptide entered held-out split",
    )
    buckets = defaultdict(list)
    for row in rows:
        bucket = (
            row.split
            if row.split != "train"
            else "train-exact"
            if row.qualifier == "="
            else "train-censored"
        )
        buckets[(row.source, bucket)].append(row)
    descriptors = []
    for (source, bucket), values in sorted(buckets.items()):
        path = cohort / f"{source}-{bucket}.jsonl"
        with path.open("x") as stream:
            for row in values:
                stream.write(json.dumps(row.to_dict(), sort_keys=True, allow_nan=False) + "\n")
        descriptors.append(
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "source": source,
                "population": POPULATION,
                "label_policy": LABEL_POLICY,
                "split_policy": SPLIT_POLICY,
                "bucket": bucket,
                "count": len(values),
            }
        )
    manifests = {}
    for condition in ("exact", "with-censor"):
        sources = [
            {key: value for key, value in item.items() if key not in ("bucket", "count")}
            for item in descriptors
            if condition == "with-censor" or item["bucket"] != "train-censored"
        ]
        manifest = {
            "schema": MANIFEST_VERSION,
            "model": {
                "endpoints": ["ic50"],
                "identity_tasks": [],
                "mhc_classes": ["I", "II"],
                "species": ["human"],
                "dropout": 0.1,
            },
            "training": {"identity_weight": 0},
            "sources": sources,
        }
        path = cohort / f"manifest-{condition}.json"
        dump(path, manifest)
        _, _, _, check = preflight(ObservationIndex(path))
        dump(output / f"results/preflight-{condition}.json", check)
        manifests[condition] = {"path": str(path), "sha256": file_sha256(path)}
    for source in SOURCES.values():
        require(
            file_sha256(source["path"]) == source["sha256"], "source changed during preparation"
        )
    report = {
        "status": "admitted",
        "raw_rows": ordinal,
        "candidate_rows": len(candidates),
        "admitted_rows": len(rows),
        "dispositions": dict(disposition_counts),
        "initial_reasons": dict(rejected),
        "catalog_dispositions": dict(catalog.dispositions),
        "catalog_used_names": len(aliases),
        "catalog_ambiguous_names": sorted(
            name for name, values in catalog.assignments.items() if len(values) > 1
        ),
        "independent_input_contradictions": prepared["independent_input_contradictions"],
        "manifests": manifests,
        "shared_sources": descriptors,
        "raw_artifacts": str(raw_dir),
        "raw_preparation": prepared,
        **summary(rows),
        "training": None,
        "evaluation": None,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }
    dump(output / "results/admission.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "raw_rows",
                    "candidate_rows",
                    "admitted_rows",
                    "dispositions",
                    "readiness",
                    "readiness_failures",
                    "elapsed_seconds",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FAMILY)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    args = parser.parse_args()
    main(args.output, args.cohort)
