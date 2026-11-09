"""Bounded literal MHC-II metadata inventory; never a training-row selector."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import subprocess
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
BUNDLE = Path(
    "/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z"
)
HITLIST = Path("/private/tmp/presto-mhc2-20261026/hitlist-current")
FILES = ("binding.parquet", "observations.parquet", "other_assays.parquet")
GROUPS = ("mhc_class", "mhc_species", "response_measured", "measurement_units")
COVERAGE = (
    "mhc_restriction",
    "mhc_allele_set",
    "peptide_extended",
    "cell_type",
    "cell_line_name",
    "attributed_sample_label",
    "antigen_processing_comments",
    "culture_condition",
    "assay_comments",
    "provenance_id",
)
FIELDS = tuple(
    dict.fromkeys(
        (
            *GROUPS,
            *COVERAGE,
            "peptide",
            "pmid",
            "assay_method",
            "measurement_inequality",
            "quantitative_value",
            "mhc_allele_provenance",
            "allele_resolution",
            "source",
            "mhc_class_corrected",
            "has_ptm",
            "is_ms_observation",
            "is_monoallelic",
            "ms_outcome_status",
        )
    )
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def records(counter, names):
    return [
        dict(zip(names, key), n_rows=count)
        for key, count in sorted(counter.items(), key=lambda item: repr(item[0]))
    ]


def guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # macOS bytes
    if rss > 3 * 1024**3:
        raise RuntimeError("Audit exceeded prospectively declared 3 GiB RSS")


def main():
    started = time.monotonic()
    signal.alarm(300)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if (HERE / "results/inventory.json").exists():
        raise FileExistsError("Preserve completed audit; use a new experiment for reruns")
    inputs = [BUNDLE / name for name in FILES]
    if sum(p.stat().st_size for p in inputs) > 1024**3:
        raise RuntimeError("Input byte budget exceeded")
    source_files = [
        HITLIST / "hitlist/data/pmid_overrides.yaml",
        HITLIST / "hitlist/data/supplementary.yaml",
        HITLIST / "hitlist/version.py",
    ]
    receipt = {
        "presto_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "inputs": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in [*inputs, *source_files]
        ],
        "script_sha256": digest(Path(__file__)),
        "hitlist_commit": "13b91424295e01286db46e8e9818fcc2fe816834",
        "scope": "Literal metadata only; no biological admission or inferred negatives",
    }
    write("reproduce/execution-inputs.json", receipt)
    snapshot = HERE / "reproduce/source/audit.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    output = {}
    independent = {}
    for name in FILES:
        parquet = pq.ParquetFile(BUNDLE / name)
        missing = set(FIELDS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Unsupported schema: {name}: {missing}")
        all_counts, assay_counts = Counter(), Counter()
        category_counts = {
            key: Counter()
            for key in ("mhc_allele_provenance", "allele_resolution", "source", "ms_outcome_status")
        }
        coverage = Counter()
        studies = defaultdict(Counter)
        study_peptides = defaultdict(set)
        peptides = set()
        sampled = []
        for batch in parquet.iter_batches(batch_size=32768, columns=FIELDS, use_threads=False):
            # Python row arithmetic is independent of the later Arrow group-by replay.
            for key in zip(
                *(batch.column(batch.schema.get_field_index(c)).to_pylist() for c in GROUPS)
            ):
                all_counts[key] += 1
            selected = batch.filter(pc.equal(pc.cast(batch.column(0), pa.string()), "II"))
            for row in selected.to_pylist():
                pmid = str(row["pmid"]) if row["pmid"] is not None else "unknown"
                study = studies[pmid]
                study["n_rows"] += 1
                peptides.add(row["peptide"])
                study_peptides[pmid].add(row["peptide"])
                for field in COVERAGE:
                    if row[field] is not None and row[field] != "":
                        coverage[field] += 1
                        study[f"n_nonempty_{field}"] += 1
                for field in ("has_ptm", "is_ms_observation", "is_monoallelic"):
                    if row[field] is True:
                        coverage[field] += 1
                        study[f"n_true_{field}"] += 1
                for field, counter in category_counts.items():
                    counter[row[field]] += 1
                value = row["quantitative_value"]
                state = (
                    "missing"
                    if value is None
                    else (
                        "nonfinite"
                        if not math.isfinite(value)
                        else "positive"
                        if value > 0
                        else "zero"
                        if value == 0
                        else "negative"
                    )
                )
                assay_counts[
                    (
                        row["mhc_species"],
                        row["assay_method"],
                        row["response_measured"],
                        row["measurement_units"],
                        row["measurement_inequality"],
                        state,
                    )
                ] += 1
                if pmid in ("31495665", "37301199") and study["n_rows"] <= 3:
                    sampled.append(
                        {
                            key: row[key]
                            for key in (
                                "pmid",
                                "provenance_id",
                                "mhc_restriction",
                                "mhc_allele_set",
                                "mhc_allele_provenance",
                                "attributed_sample_label",
                                "cell_line_name",
                            )
                        }
                    )
            guard()
        for pmid, study in studies.items():
            study["n_distinct_peptides"] = len(study_peptides[pmid])
        output[name] = {
            "n_all_rows": parquet.metadata.num_rows,
            "all_class_assay_groups": records(all_counts, GROUPS),
            "n_class_ii_rows": sum(s["n_rows"] for s in studies.values()),
            "n_class_ii_distinct_peptides": len(peptides),
            "n_class_ii_known_pmids": len(set(studies) - {"unknown"}),
            "class_ii_nonempty_or_true_field_counts": dict(coverage),
            "class_ii_categories": {k: records(v, (k,)) for k, v in category_counts.items()},
            "class_ii_assays": records(
                assay_counts,
                (
                    "mhc_species",
                    "assay_method",
                    "response_measured",
                    "measurement_units",
                    "measurement_inequality",
                    "numeric_state",
                ),
            ),
            "class_ii_studies": dict(studies),
            "deterministic_identifier_examples": sampled,
            "schema": str(parquet.schema_arrow),
        }
        # Second pass: different batch size, Arrow aggregation, no producer row loop.
        replay = Counter()
        for batch in parquet.iter_batches(batch_size=131071, columns=GROUPS, use_threads=False):
            table = pa.table(
                {key: pc.cast(batch.column(i), pa.string()) for i, key in enumerate(GROUPS)}
            )
            table = table.append_column("row_counter", pa.array([1] * len(table), pa.int64()))
            for row in (
                table.group_by(list(GROUPS), use_threads=False)
                .aggregate([("row_counter", "sum")])
                .to_pylist()
            ):
                replay[tuple(row[k] for k in GROUPS)] += row["row_counter_sum"]
            guard()
        assert replay == all_counts, f"Independent group replay failed: {name}"
        assert sum(replay.values()) == parquet.metadata.num_rows
        independent[name] = {
            "n_rows_verified": sum(replay.values()),
            "n_joint_groups_verified": len(replay),
            "passed": True,
        }
        print(
            json.dumps(
                {
                    "file": name,
                    "n_class_ii_rows": output[name]["n_class_ii_rows"],
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    overrides = yaml.safe_load(source_files[0].read_text())
    if isinstance(overrides, dict):
        overrides = overrides["studies"]
    curated = [row for row in overrides if row["pmid"] in (31495665, 37301199)]
    for item in receipt["inputs"]:
        assert digest(Path(item["path"])) == item["sha256"]
    write("results/current-curation.json", curated)
    write("results/inventory.json", output)
    write("results/independent-verification.json", independent)
    write(
        "results/runtime.json",
        {
            "seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    signal.alarm(0)


if __name__ == "__main__":
    main()
