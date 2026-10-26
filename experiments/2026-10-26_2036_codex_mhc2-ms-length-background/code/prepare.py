"""Prepare the prospectively specified DP/DQ8 cohort without model scoring."""

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from presto.data.modular import Context, Observation
from presto.data.modular_sources import MANIFEST_VERSION, file_sha256

from cohort_rules import backgrounds, digest, kmers, partition, training_frequencies

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
PRIOR = ROOT / "experiments/2026-10-26_1954_codex_mhc2-ms-transfer-cohort"
AFFINITY = (
    ROOT
    / "experiments/2026-10-26_1929_codex_mhc2-censor-refinement"
    / "reproduce/exact_continuation-manifest.json"
)
INPUT = Path("/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01")
NATIVE = Path("/private/tmp/presto-mhc2-20261026/ms-source/training.parquet")
RAW = Path("/private/tmp/presto-mhc2-20261026/ms-length-background/cohort")
PARENT = Path(
    "/Users/iskander/code/presto/artifacts/2026-10-26_1610_codex_ic50-convergence/seed42/selected.pt"
)
PARENT_SHA = "db2585e63489cb5aac8c7a32fd7e522bcbc51035846c05f8cd2794c4449e3dc5"
NATIVE_SHA = "f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd"
STARTED = time.monotonic()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024**3, "RSS cap"
    assert psutil.virtual_memory().available >= 10 * 1024**3, "RAM floor"
    assert shutil.disk_usage(FAMILY).free >= 10 * 1024**3, "disk floor"
    assert time.monotonic() - STARTED < 300, "runtime cap"


def timeout(_signal, _frame):
    raise TimeoutError("preparation300second cap")


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    repro = FAMILY / "reproduce"
    repro.mkdir(exist_ok=True)
    (repro / "source").mkdir(exist_ok=True)
    receipt_path = repro / "launch.json"
    if receipt_path.exists():
        raise FileExistsError("existing preparation receipt; no automatic retry")
    state = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "agent_model": "Codex / GPT-6",
        "status": "starting",
        "argv": sys.argv,
        "python": sys.executable,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "code_hashes": {},
        "environment": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "requested_modal_gpu": None,
        "device": "cpu",
        "startup_ram": psutil.virtual_memory().available,
        "limits": {"seconds": 300, "rss": 2 * 1024**3, "ram_floor": 10 * 1024**3},
    }
    for path in sorted((FAMILY / "code").glob("*.py")):
        shutil.copyfile(path, repro / "source" / path.name)
        state["code_hashes"][path.name] = file_sha256(path)
    write(receipt_path, state)
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        assert state["startup_ram"] >= 13 * 1024**3, "startup RAM reserve"
        assert shutil.disk_usage(FAMILY).free >= 15 * 1024**3, "startup disk reserve"
        guard()
        RAW.mkdir(parents=True, exist_ok=False)
        (FAMILY / "results").mkdir(exist_ok=True)
        prior = read(PRIOR / "reproduce/launch.json")
        assert read(PRIOR / "results/independent-verification.json")["status"] == "passed"
        for name, expected in prior["input_hashes"].items():
            assert file_sha256(INPUT / name) == expected
        assert file_sha256(NATIVE) == NATIVE_SHA
        assert file_sha256(PARENT) == PARENT_SHA
        assert file_sha256(AFFINITY) == prior["affinity_manifest"]["sha256"]
        state["inputs"] = {
            "admitted": prior["input_hashes"],
            "native": {"path": str(NATIVE), "sha256": NATIVE_SHA},
            "affinity_manifest": prior["affinity_manifest"],
            "parent": {"path": str(PARENT), "sha256": PARENT_SHA},
        }
        initializer_segments = set()
        for source in read(AFFINITY)["sources"]:
            path = Path(source["path"])
            assert file_sha256(path) == source["sha256"]
            with path.open() as stream:
                for line in stream:
                    initializer_segments.update(kmers(json.loads(line)["peptide"]))
        contributors = {
            row["provenance_id"]: row["original_pmids"]
            for row in pq.read_table(INPUT / "contributor-summary.parquet").to_pylist()
        }
        # Retain compact representatives, not corpus-sized duplicate raw records.
        units, source_lineage, rejected = {}, [], []
        with (INPUT / "observations.jsonl").open("rb") as stream:
            ordinal = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                item = json.loads(line)
                raw = item["provenance"]["raw_record"]
                study = str(raw.get("pmid") or "")
                if study in {"35251023", "34433824"}:
                    assert sorted(set(contributors[raw["provenance_id"]])) == [study]
                    assert item["value"] == 1 and raw["has_ptm"] is False
                    if len(item["mhc"]) != 1:
                        rejected.append(
                            dict(id=item["observation_id"], reason="multiple_candidates")
                        )
                    else:
                        pair = item["mhc"][0]
                        key = (study, item["peptide"], pair["groove1"], pair["groove2"])
                        unit_id = "ms-unit:" + digest(json.dumps(key))
                        source_lineage.append(
                            dict(
                                unit_id=unit_id,
                                study=study,
                                observation_id=item["observation_id"],
                                admission_ordinal=ordinal,
                                admission_byte_offset=offset,
                                native_ordinal=item["provenance"]["source_ordinal"],
                                provenance_id=raw["provenance_id"],
                            )
                        )
                        if (
                            key not in units
                            or item["observation_id"] < units[key]["observation_id"]
                        ):
                            # Context is constructed through the typed Observation below.
                            item["context"] = {}
                            item["provenance"] = {
                                "measurement_namespace": "hitlist_evidence_row_id",
                                "original_measurement_id": item["observation_id"],
                                "source_observation_id": item["observation_id"],
                                "admitted_source_sha256": prior["input_hashes"][
                                    "observations.jsonl"
                                ],
                                "admitted_byte_offset": offset,
                                "unit_id": unit_id,
                                "study": study,
                                "label_kind": "reported_positive_ms_detection",
                                "context_policy": (
                                    "unasserted_input_original_context_retained_in_source"
                                ),
                                "interpretation": (
                                    "detection_enrichment_against_generated_background"
                                ),
                            }
                            units[key] = item
                ordinal += 1
                if ordinal % 1000 == 0:
                    guard()
        by_study = defaultdict(set)
        for study, peptide, _, _ in units:
            by_study[study].add(peptide)
        assignments, dispositions = partition(
            by_study["35251023"], by_study["34433824"], initializer_segments
        )
        counts = {}
        for split in ("train", "validation", "test", "external"):
            selected = [(key, value) for key, value in assignments.items() if value[0] == split]
            counts[split] = {
                "peptides": len({key[1] for key, _ in selected}),
                "components": len({value[1] for _, value in selected}),
            }
            if split != "train":
                assert counts[split]["peptides"] >= 200, (split, "insufficient peptides")
                assert counts[split]["components"] >= 20, (split, "insufficient components")
        write(FAMILY / "results/preparation-counts.json", counts)
        original_counts = read(
            ROOT
            / "experiments/2026-10-26_2030_codex_mhc2-ms-prediction"
            / "results/preparation-counts.json"
        )
        assert counts == original_counts, "positive membership changed from original contract"
        known_positive = set()
        for batch in pq.ParquetFile(NATIVE).iter_batches(
            batch_size=50000, columns=["peptide", "is_ms_observation", "ms_outcome_status"]
        ):
            for row in batch.to_pylist():
                if row["is_ms_observation"] is True and row["ms_outcome_status"] in {
                    "positive_result",
                    "curated_ligand_table",
                }:
                    known_positive.add(row["peptide"])
            guard()
        frequencies = training_frequencies(assignments)
        write(FAMILY / "results/background-frequencies.json", frequencies)
        controls = backgrounds(assignments, known_positive, initializer_segments, frequencies)
        guard()
        pq.write_table(
            pa.Table.from_pylist(source_lineage), FAMILY / "results/source-lineage.parquet"
        )
        pq.write_table(pa.Table.from_pylist(dispositions), FAMILY / "results/dispositions.parquet")
        write(FAMILY / "results/rejected-candidate-rows.json", rejected)
        with gzip.open(FAMILY / "results/generated-backgrounds.jsonl.gz", "wt") as stream:
            for peptide, control in sorted(controls.items()):
                stream.write(json.dumps({"positive": peptide, **control}) + "\n")
        handles, row_counts = {}, Counter()
        for split in ("train", "validation", "test", "external"):
            for kind in ("positive", "control"):
                handles[(split, kind)] = (RAW / f"{split}-{kind}.jsonl").open("x")
        try:
            for key, item in sorted(units.items()):
                if (key[0], key[1]) not in assignments:
                    continue
                split, component = assignments[(key[0], key[1])]
                row = Observation.from_dict(item)
                row = replace(
                    row,
                    split="test" if split == "external" else split,
                    group_id=component,
                    context=Context(),
                    provenance={**row.provenance, "evaluation_partition": split},
                )
                row.validate()
                handles[(split, "positive")].write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
                control = controls[row.peptide]
                synthetic_id = "generated-length:" + row.provenance["unit_id"]
                negative = replace(
                    row,
                    observation_id=synthetic_id,
                    source="generated_mhc2_ms_background",
                    peptide=control["peptide"],
                    value=0.0,
                    provenance={
                        "measurement_namespace": "generated_mhc2_ms_length_background_v1",
                        "original_measurement_id": synthetic_id,
                        "parent_unit_id": row.provenance["unit_id"],
                        "parent_observation_id": row.observation_id,
                        "parent_peptide": row.peptide,
                        "study": key[0],
                        "evaluation_partition": split,
                        "label_kind": "generated_control_not_measured_negative",
                        "generation_kind": "length_matched_training_frequency_draw",
                        "generation_policy": "presto-ms-length-background-v1",
                        "generation_seed": control["seed"],
                        "generation_attempt": control["attempt"],
                    },
                )
                negative.validate()
                handles[(split, "control")].write(
                    json.dumps(negative.to_dict(), sort_keys=True) + "\n"
                )
                row_counts[split] += 2
                if sum(row_counts.values()) % 1000 == 0:
                    guard()
        finally:
            for stream in handles.values():
                stream.close()
        sources = []
        for (split, kind), stream in sorted(handles.items()):
            path = Path(stream.name)
            sources.append(
                dict(
                    path=str(path),
                    sha256=file_sha256(path),
                    partition=split,
                    source="hitlist_native_ms"
                    if kind == "positive"
                    else "generated_mhc2_ms_background",
                    population="qualified DP/K562 detection enrichment"
                    if split != "external"
                    else "DQ8 external detection enrichment",
                    label_policy="reported_positive_ms_detection"
                    if kind == "positive"
                    else "explicit_generated_sequence_background_not_measured_nondetection",
                    split_policy="frozen_nine_residue_components_and_initializer_exclusion_v1",
                )
            )
        total_bytes = sum(path.stat().st_size for path in RAW.iterdir())
        assert total_bytes <= 250 * 1024**2, "prepared output cap"
        summary = {
            "status": "prepared_pending_independent_verification",
            "schema": MANIFEST_VERSION,
            "sources": sources,
            "counts": counts,
            "row_counts": dict(row_counts),
            "known_positive_sequences": len(known_positive),
            "initializer_segments": len(initializer_segments),
            "output_bytes": total_bytes,
            "training_steps_per_epoch": (row_counts["train"] + 63) // 64,
            "native_final_steps": 3 * ((row_counts["train"] + 63) // 64),
            "initialization": {"path": str(PARENT), "sha256": PARENT_SHA},
        }
        write(FAMILY / "results/cohort.json", summary)
        state["status"] = "prepared_pending_independent_verification"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        state.update(
            elapsed_seconds=time.monotonic() - STARTED,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        write(receipt_path, state)
    print(json.dumps({"status": state["status"], "row_counts": dict(row_counts)}), flush=True)


if __name__ == "__main__":
    main()
