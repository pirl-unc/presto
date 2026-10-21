#!/usr/bin/env python
"""Source-pinned MHC domain annotation admission; no fitting or predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from presto.data.modular import AA, MHCInput, Observation
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    file_sha256,
    write_observations,
)
from presto.training.modular_run import preflight

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "experiments/2026-10-21_0618_codex_mhc-reference-admission"
SOURCE = Path("/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv")
SOURCE_SHA256 = "ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1"
TAXA = {
    "Homo sapiens": "human",
    "Mus musculus (Mouse)": "mouse",
    "Rattus norvegicus (Norway rat)": "rat",
    "Macaca mulatta": "rhesus_macaque",
}
SALT = "presto-mhc-reference-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_gzip(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    path.write_bytes(gzip.compress(data.encode(), mtime=0))


def domains(row):
    """Return a fixed exclusion reason or the genuine source groove slots."""
    if row["species"] not in TAXA:
        return "taxon_outside_frozen_vocabulary", []
    if not row["source"] or not row["source_id"]:
        return "missing_source_reference", []
    if row["is_functional"] != "True" or any(
        row[name] != "False" for name in ("is_null", "is_questionable", "is_pseudogene")
    ):
        return "unsupported_functional_flags", []
    if row["groove_status"] != "ok" or row["groove_flags"]:
        return "unsupported_groove_status_or_flags", []
    expected = {
        ("I", "alpha"): ("groove1", "groove2"),
        ("II", "alpha"): ("groove1",),
        ("II", "beta"): ("groove2",),
    }.get((row["mhc_class"], row["chain"]))
    if expected is None:
        return "unsupported_class_chain", []
    if any(bool(row[name]) != (name in expected) for name in ("groove1", "groove2")):
        return "missing_or_unexpected_domain", []
    found = []
    for name in expected:
        value = row[name]
        if not 60 <= len(value) <= 130 or set(value) - set(AA):
            return "noncanonical_or_out_of_scope_domain", []
        if row[f"{name}_len"] != str(len(value)):
            return "declared_domain_length_mismatch", []
        position = row["sequence"].find(value)
        if position < 0:
            return "domain_absent_from_source_protein", []
        found.append((name, value, position))
    if len(found) == 2 and found[0][2] + len(found[0][1]) > found[1][2]:
        return "domain_order_or_overlap", []
    return None, found


def split_for(component):
    bucket = int(hashlib.sha256(f"{SALT}:{component}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def prepare(raw_rows):
    dispositions, eligible, memberships = [], {}, defaultdict(set)
    for index, row in enumerate(raw_rows):
        reason, values = domains(row)
        disposition = {
            "source_row": index,
            **{
                name: row[name]
                for name in (
                    "source",
                    "source_id",
                    "two_field_allele",
                    "gene",
                    "species",
                    "mhc_class",
                    "chain",
                    "groove_status",
                    "groove_flags",
                    "is_functional",
                    "is_null",
                    "is_questionable",
                    "is_pseudogene",
                )
            },
            "status": "excluded" if reason else "pending",
            "reason": reason,
            "annotations": [],
            "primary_annotations": [],
            "alias_annotations": [],
        }
        dispositions.append(disposition)
        if reason is None:
            eligible[index] = values
            for _, value, _ in values:
                memberships[value].add(TAXA[row["species"]])
    conflicts = {
        value: sorted(species) for value, species in memberships.items() if len(species) > 1
    }
    aliases = defaultdict(list)
    for index, values in eligible.items():
        if any(value in conflicts for _, value, _ in values):
            dispositions[index].update(
                status="deferred", reason="identical_sequence_species_conflict"
            )
            continue
        row = raw_rows[index]
        for name, value, position in values:
            key = name, value, TAXA[row["species"]]
            aliases[key].append({"source_row": index, "protein_position": position})
        dispositions[index].update(status="admitted_source", reason=None)
    keys = sorted(aliases)
    parents = list(range(len(keys)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen_sequence, seen_parent = {}, {}
    for i, key in enumerate(keys):
        links = [(seen_sequence, key[1])] + [
            (seen_parent, alias["source_row"]) for alias in aliases[key]
        ]
        for seen, identity in links:
            if identity in seen:
                parents[root(i)] = root(seen[identity])
            else:
                seen[identity] = i
    components = defaultdict(list)
    for i, key in enumerate(keys):
        components[root(i)].append(key)
    component_ids = {
        key: digest(sorted((slot, sequence) for slot, sequence, _ in value))
        for key, value in components.items()
    }
    observations, annotation_records = [], []
    for i, key in enumerate(keys):
        slot, sequence, species = key
        source_aliases = aliases[key]
        annotation_id = digest(key)
        component = component_ids[root(i)]
        split = split_for(component)
        label = "mhc1" if slot == "groove1" else "mhc2"
        row = Observation(
            f"mhc-domain:{annotation_id}",
            "mhc_reference_domains",
            "identity",
            mhc=(
                MHCInput(
                    sequence if slot == "groove1" else "", sequence if slot == "groove2" else ""
                ),
            ),
            split=split,
            group_id=f"mhc-reference:{SOURCE_SHA256}:{component}",
            identity_labels={label: species},
            provenance={
                "measurement_lineage": {
                    "namespace": f"mhc-reference:{SOURCE_SHA256}",
                    "measurement_id": annotation_id,
                },
                "source_reference_sha256": SOURCE_SHA256,
                "source_aliases": [
                    {
                        **alias,
                        **{
                            name: raw_rows[alias['source_row']][name]
                            for name in (
                                "source",
                                "source_id",
                                "two_field_allele",
                                "gene",
                                "species",
                                "mhc_class",
                                "chain",
                            )
                        },
                    }
                    for alias in source_aliases
                ],
                "label_definition": "source-annotated molecular species of this reference domain",
                "domain_slot": slot,
                "split_policy": {"salt": SALT, "boundaries": [70, 15, 15], "component": component},
                "independence": (
                    "exact-domain and original-reference-chain components; "
                    "homology independence unverified"
                ),
                "pairing_target_generated": False,
            },
        )
        row.validate()
        observations.append(row)
        annotation_records.append(
            {
                "observation_id": row.observation_id,
                "slot": slot,
                "sequence": sequence,
                "species": species,
                "split": split,
                "component": component,
                "aliases": source_aliases,
            }
        )
        for alias in source_aliases:
            dispositions[alias["source_row"]]["annotations"].append(row.observation_id)
            kind = "primary_annotations" if alias == source_aliases[0] else "alias_annotations"
            dispositions[alias["source_row"]][kind].append(row.observation_id)
    for disposition in dispositions:
        if disposition["status"] == "admitted_source" and not disposition["primary_annotations"]:
            disposition.update(status="duplicate_alias", reason="all_domains_already_represented")
    # Replay every admitted fact from original source rows, independently of alias construction.
    for row in observations:
        label, species = next(iter(row.identity_labels.items()))
        slot = "groove1" if label == "mhc1" else "groove2"
        sequence = getattr(row.mhc[0], slot)
        for alias in row.provenance["source_aliases"]:
            original = raw_rows[alias["source_row"]]
            require(original[slot] == sequence, "replay sequence mismatch")
            require(TAXA[original["species"]] == species, "replay species mismatch")
            require(
                original["sequence"][
                    alias["protein_position"] : alias["protein_position"] + len(sequence)
                ]
                == sequence,
                "replay protein position mismatch",
            )
            require(
                dispositions[alias["source_row"]]["status"]
                in ("admitted_source", "duplicate_alias"),
                "alias does not trace to an admitted source row",
            )
    for disposition in dispositions:
        index = disposition["source_row"]
        expected = (
            len(eligible[index])
            if disposition["status"] in ("admitted_source", "duplicate_alias")
            else 0
        )
        require(len(disposition["annotations"]) == expected, "source domain conservation failed")
    require(len(dispositions) == len(raw_rows), "raw row conservation failed")
    return observations, dispositions, annotation_records, conflicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FAMILY)
    parser.add_argument(
        "--cohort", type=Path, default=ROOT / "data/modular_cohorts/2026-10-21-mhc-reference"
    )
    args = parser.parse_args()
    output, cohort = args.output.resolve(), args.cohort.resolve()
    if (output / "reproduce/executed-launch.json").exists() or cohort.exists():
        raise ValueError("refusing to overwrite an executed admission")
    before = SOURCE.stat()
    if file_sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("pinned MHC reference hash mismatch")
    for name in ("results", "reproduce/source", "launch_logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    cohort.mkdir(parents=True)
    receipt = {
        "agent_model": "Codex / GPT-6",
        "kind": "data_admission_only",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "argv": sys.argv,
        "cwd": str(ROOT),
        "executable": sys.executable,
        "python": sys.version,
        "source": {
            "path": str(SOURCE),
            "sha256": SOURCE_SHA256,
            "producer_version": "not recorded in the source CSV",
        },
        "runtime_versions": {
            name: version(name) for name in ("numpy", "torch", "mhcseqs", "mhcgnomes")
        },
        "environment": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "taxon_mapping": TAXA,
        "salt": SALT,
        "device": "cpu",
        "gpu": None,
        "training": None,
        "predictions": None,
    }
    write_json(output / "reproduce/executed-launch.json", receipt)
    write_json(output / "reproduce/launch.json", receipt)
    shutil.copyfile(__file__, output / "reproduce/source/launch.py")
    for relative in (
        "data/modular.py",
        "data/modular_sources.py",
        "data/modular_lineage.py",
        "training/modular_run.py",
    ):
        destination = output / "reproduce/source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    started = time.monotonic()
    try:
        with SOURCE.open(newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
        observations, dispositions, annotations, conflicts = prepare(raw_rows)
        after = SOURCE.stat()
        require(
            (before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns)
            and file_sha256(SOURCE) == SOURCE_SHA256,
            "pinned source changed during admission",
        )
        write_gzip(output / "results/source-dispositions.jsonl.gz", dispositions)
        write_gzip(output / "results/annotations.jsonl.gz", annotations)
        write_gzip(
            output / "results/species-conflicts.jsonl.gz",
            ({"sequence": key, "species": value} for key, value in sorted(conflicts.items())),
        )
        observation_receipt = write_observations(cohort / "observations.jsonl", observations)
        manifest = {
            "schema": MANIFEST_VERSION,
            "model": {
                "endpoints": [],
                "identity_tasks": ["mhc_species"],
                "species": list(TAXA.values()),
            },
            "sources": [
                {
                    "path": "observations.jsonl",
                    "sha256": observation_receipt["sha256"],
                    "source": "mhc_reference_domains",
                    "population": (
                        "all domains passing the prospectively frozen four-taxon reference contract"
                    ),
                    "label_policy": (
                        "one reference species annotation per unique slot/sequence "
                        "with source aliases; conflicts deferred"
                    ),
                    "split_policy": (
                        "exact-domain and original-reference-chain connected components; "
                        "fixed SHA25670/15/15; homology independence unverified"
                    ),
                }
            ],
        }
        write_json(cohort / "manifest.json", manifest)
        _, _, _, admission = preflight(ObservationIndex(cohort / "manifest.json"))
        write_json(output / "results/preflight.json", admission)
        counts, component_counts = {}, {}
        for split in ("train", "validation", "test"):
            subset = [row for row in observations if row.split == split]
            counts[split] = dict(
                Counter(next(iter(row.identity_labels.values())) for row in subset)
            )
            component_counts[split] = {
                species: len(
                    {row.group_id for row in subset if species in row.identity_labels.values()}
                )
                for species in TAXA.values()
            }
        readiness_failures = [
            {
                "split": split,
                "species": species,
                "annotations": counts[split].get(species, 0),
                "components": component_counts[split][species],
            }
            for split in counts
            for species in TAXA.values()
            if counts[split].get(species, 0) < 20 or component_counts[split][species] < 3
        ]
        component_sizes = Counter(row.group_id for row in observations)
        counts_by_origin = Counter(
            (
                row.split,
                next(iter(row.identity_labels.values())),
                row.provenance["domain_slot"],
                alias["mhc_class"],
                alias["source"],
            )
            for row in observations
            for alias in row.provenance["source_aliases"]
        )
        report = {
            "status": "admitted",
            "readiness": "failed" if readiness_failures else "passed",
            "readiness_failures": readiness_failures,
            "raw_rows": len(raw_rows),
            "dispositions": dict(Counter(row["status"] for row in dispositions)),
            "reasons": dict(Counter(row["reason"] for row in dispositions if row["reason"])),
            "unique_annotations": len(observations),
            "source_domain_aliases": sum(
                len(row.provenance["source_aliases"]) for row in observations
            ),
            "conflicting_sequences": len(conflicts),
            "counts_by_split_species": counts,
            "components_by_split_species": component_counts,
            "source_domain_alias_counts": [
                {
                    "split": key[0],
                    "species": key[1],
                    "slot": key[2],
                    "mhc_class": key[3],
                    "source": key[4],
                    "aliases": count,
                }
                for key, count in sorted(counts_by_origin.items())
            ],
            "components": len(component_sizes),
            "component_size_counts": dict(sorted(Counter(component_sizes.values()).items())),
            "replay": {
                "status": "passed",
                "annotation_labels": len(observations),
                "raw_row_dispositions": len(raw_rows),
            },
            "manifest": {
                "path": str(cohort / "manifest.json"),
                "sha256": file_sha256(cohort / "manifest.json"),
            },
            "normalized_observations": observation_receipt,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "training": None,
            "evaluation": None,
            "qualification": "reference admission only; no model trained or biologically qualified",
        }
        write_json(output / "results/admission.json", report)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "readiness",
                        "raw_rows",
                        "dispositions",
                        "unique_annotations",
                        "conflicting_sequences",
                        "counts_by_split_species",
                        "components_by_split_species",
                        "elapsed_seconds",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            output / "results/failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
