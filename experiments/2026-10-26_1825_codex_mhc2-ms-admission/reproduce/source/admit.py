"""Bounded positive-MS admission with complete source dispositions and links."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from presto.data.modular_hitlist import AdmissionError, HumanMHCCatalog
from presto.data.modular_hitlist_ms import class_ii_ms_observation, study_factor_vocabulary

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[1]
SOURCE = Path("/private/tmp/presto-mhc2-20261026/ms-source/training.parquet")
SOURCE_SHA = "f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd"
CATALOG = Path("/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv")
CATALOG_SHA = "ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1"
CONTRIBUTORS = Path(
    "/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/"
    "bundles/2026-10-16T07-20-43Z/observation_contributors.parquet"
)
CONTRIBUTOR_SHA = "41ff93e08bc6ea4af692fee16044726af26c4e20ec650ec6c7cd851974b71db1"
OUTPUT = Path("/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01")
SUMMARY = FAMILY / "results" / "admission-attempt01"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def guard():
    if peak_rss() > 3 * 1024**3:
        raise RuntimeError("admission exceeded 3-GiB RSS limit")
    if sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file()) > 2 * 1024**3:
        raise RuntimeError("admission exceeded 2-GiB output limit")


def main():
    started = time.monotonic()
    signal.alarm(300)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    SUMMARY.mkdir(parents=True, exist_ok=False)
    for path, expected in (
        (SOURCE, SOURCE_SHA),
        (CATALOG, CATALOG_SHA),
        (CONTRIBUTORS, CONTRIBUTOR_SHA),
    ):
        assert sha(path) == expected, path
    with CATALOG.open(newline="") as stream:
        catalog = HumanMHCCatalog(csv.DictReader(stream))
    context_path = FAMILY / "results/input-profile/context-groups.json"
    factors = study_factor_vocabulary(json.loads(context_path.read_text()))
    write_json(SUMMARY / "study-factor-vocabulary.json", factors)
    code = [Path(__file__), REPO / "data/modular_hitlist.py", REPO / "data/modular_hitlist_ms.py"]
    launch = {
        "agent_model": "Codex / GPT-6",
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "argv": sys.argv,
        "python": sys.executable,
        "source": {"path": str(SOURCE), "sha256": SOURCE_SHA},
        "catalog": {"path": str(CATALOG), "sha256": CATALOG_SHA},
        "contributors": {"path": str(CONTRIBUTORS), "sha256": CONTRIBUTOR_SHA},
        "context_profile_sha256": sha(context_path),
        "code": {str(path.relative_to(REPO)): sha(path) for path in code},
        "limits": {
            "threads": 1,
            "seconds": 300,
            "rss_bytes": 3 * 1024**3,
            "output_bytes": 2 * 1024**3,
        },
        "environment_overrides": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": (
                "/private/tmp/presto-mhc2-20261026:"
                "/private/tmp/presto-pr-completion-20261026/dependencies"
            ),
        },
    }
    write_json(FAMILY / "reproduce/admission-launch-attempt01.json", launch)
    for path in code:
        shutil.copyfile(path, FAMILY / "reproduce/source" / path.name)
    dispositions = Counter()
    studies, molecules, contexts = Counter(), Counter(), Counter()
    observed_ids, provenance_ids = set(), set()
    source_rows = candidates = 0
    schema = pa.schema(
        [
            ("source_ordinal", pa.int64()),
            ("evidence_row_id", pa.string()),
            ("disposition", pa.string()),
            ("reason", pa.string()),
        ]
    )
    with (
        (OUTPUT / "observations.jsonl").open("x") as target,
        pq.ParquetWriter(
            OUTPUT / "dispositions.parquet", schema, compression="zstd"
        ) as disposition_writer,
    ):
        for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=4096, use_threads=False):
            indices = pc.indices_nonzero(
                pc.and_(
                    pc.equal(batch.column("mhc_class"), "II"), batch.column("is_ms_observation")
                )
            )
            records = batch.take(indices).to_pylist()
            results = []
            for position, raw in zip(indices.to_pylist(), records):
                ordinal = source_rows + position
                candidates += 1
                try:
                    observation = class_ii_ms_observation(
                        raw,
                        catalog,
                        source_sha256=SOURCE_SHA,
                        ordinal=ordinal,
                        study_factors=factors,
                    )
                except AdmissionError as error:
                    disposition, reason = error.disposition, error.reason
                else:
                    disposition, reason = "admitted", "positive_ms_complete_candidates"
                    if observation.observation_id in observed_ids:
                        raise ValueError("duplicate admitted observation identity")
                    observed_ids.add(observation.observation_id)
                    provenance_ids.add(raw["provenance_id"])
                    studies[str(raw.get("pmid") or "")] += 1
                    names = tuple(
                        sorted(
                            name
                            for pair in observation.mhc
                            for name in pair.provenance["reported_candidates"]
                        )
                    )
                    molecules[names] += 1
                    contexts[
                        (
                            str(raw.get("pmid") or ""),
                            raw.get("condition_id") or "",
                            observation.context.provenance["condition_admitted"],
                        )
                    ] += 1
                    target.write(
                        json.dumps(observation.to_dict(), separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
                dispositions[(disposition, reason)] += 1
                results.append(
                    dict(
                        source_ordinal=ordinal,
                        evidence_row_id=raw.get("evidence_row_id"),
                        disposition=disposition,
                        reason=reason,
                    )
                )
            if results:
                disposition_writer.write_table(pa.Table.from_pylist(results, schema=schema))
            source_rows += len(batch)
            guard()
    print(
        json.dumps(
            {
                "stage": "source_admitted",
                "source_rows": source_rows,
                "class_ii_ms_rows": candidates,
                "admitted": len(observed_ids),
            }
        ),
        flush=True,
    )
    link_columns = [
        "provenance_id",
        "source_record_id",
        "source_dataset",
        "source_row",
        "original_pmid",
        "relationship_status",
    ]
    reader = pq.ParquetFile(CONTRIBUTORS)
    schema = pa.schema([reader.schema_arrow.field(name) for name in link_columns])
    link_counts, contributor_studies = Counter(), defaultdict(set)
    unresolved, unknown_study = set(), set()
    all_links = retained_links = 0
    with pq.ParquetWriter(
        OUTPUT / "contributor-links.parquet", schema, compression="zstd"
    ) as writer:
        for batch in reader.iter_batches(batch_size=8192, columns=link_columns, use_threads=False):
            all_links += len(batch)
            positions = [
                i
                for i, value in enumerate(batch.column("provenance_id").to_pylist())
                if value in provenance_ids
            ]
            if positions:
                selected = pa.Table.from_batches([batch.take(positions)])
                writer.write_table(selected)
                retained_links += len(selected)
                for row in selected.to_pylist():
                    key = row["provenance_id"]
                    link_counts[key] += 1
                    if row["relationship_status"] == "overlap_unresolved":
                        unresolved.add(key)
                    value = str(row["original_pmid"] or "")
                    if value.isdigit() and int(value) > 0:
                        contributor_studies[key].add(value)
                    else:
                        unknown_study.add(key)
            guard()
    assert set(link_counts) == provenance_ids, "missing contributor membership"
    lineage = [
        {
            "provenance_id": key,
            "links": link_counts[key],
            "original_pmids": sorted(contributor_studies[key]),
            "study_status": "unknown" if key in unknown_study else "resolved",
            "overlap_unresolved": key in unresolved,
        }
        for key in sorted(provenance_ids)
    ]
    if lineage:
        pq.write_table(
            pa.Table.from_pylist(lineage),
            OUTPUT / "contributor-summary.parquet",
            compression="zstd",
        )
    write_json(SUMMARY / "by-study.json", dict(studies.most_common()))
    write_json(
        SUMMARY / "candidate-sets.json",
        [{"candidates": list(key), "rows": value} for key, value in molecules.most_common()],
    )
    write_json(
        SUMMARY / "contexts.json",
        [
            {"pmid": key[0], "condition_id": key[1], "condition_admitted": key[2], "rows": value}
            for key, value in contexts.most_common()
        ],
    )
    summary = {
        "status": "admitted evidence; independent verification pending; no training split",
        "source_rows": source_rows,
        "class_ii_ms_rows": candidates,
        "admitted_rows": len(observed_ids),
        "admitted_studies": len(studies),
        "candidate_sets": len(molecules),
        "contexts": len(contexts),
        "dispositions": [
            {"disposition": key[0], "reason": key[1], "rows": count}
            for key, count in dispositions.most_common()
        ],
        "all_source_contributor_links": all_links,
        "retained_contributor_links": retained_links,
        "provenance_ids": len(provenance_ids),
        "unknown_contributor_study_ids": len(unknown_study),
        "unresolved_overlap_ids": len(unresolved),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss(),
        "model_training_rows": 0,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(OUTPUT.iterdir())
            if path.is_file()
        ],
    }
    guard()
    write_json(SUMMARY / "summary.json", summary)
    signal.alarm(0)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"artifacts", "dispositions"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
