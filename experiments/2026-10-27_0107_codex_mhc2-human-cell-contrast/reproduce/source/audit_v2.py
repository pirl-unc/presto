"""Projected source-row audit plus an independent class/count replay."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import time

import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

FAMILY = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/"
    "2026-10-16T07-20-43Z/observations.parquet"
)
FIELDS = (
    "peptide",
    "peptide_extended",
    "pmid",
    "mhc_class",
    "mhc_restriction",
    "mhc_allele_set",
    "mhc_allele_provenance",
    "allele_resolution",
    "assay_comments",
    "antigen_processing_comments",
    "provenance_id",
    "quantitative_value",
    "response_measured",
    "measurement_units",
    "measurement_inequality",
    "is_ms_observation",
    "source",
    "has_ptm",
    "cell_type",
    "cell_line_name",
    "culture_condition",
    "attributed_sample_label",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 2 * 1024**3
    assert psutil.virtual_memory().available >= 10 * 1024**3
    assert shutil.disk_usage(FAMILY).free >= 10 * 1024**3


def main():
    started = time.monotonic()
    receipt = FAMILY / "reproduce/audit-v2-execution.json"
    assert not receipt.exists()
    launch = json.loads((FAMILY / "reproduce/audit-v2-launch.json").read_text())
    assert sha(__file__) == launch["code_sha256"]
    state = {"status": "starting", "launch": launch, "source_path": str(SOURCE)}
    receipt.write_text(json.dumps(state, indent=2) + "\n")

    def timeout(_signal, _frame):
        raise TimeoutError("300second audit limit")

    prior = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        assert psutil.virtual_memory().available >= 13 * 1024**3
        assert shutil.disk_usage(FAMILY).free >= 15 * 1024**3
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        source_sha = sha(SOURCE)
        assert source_sha == "a665f37c0b5917fefb6d12eade530d7e2001de1f0ee73479b21d2c393d173ab4"
        parquet = pq.ParquetFile(SOURCE)
        columns = list(FIELDS) + [
            name
            for name in parquet.schema_arrow.names
            if any(token in name for token in ("condition", "replicate", "intensity", "run_id"))
            and name not in FIELDS
        ]
        assert set(columns) <= set(parquet.schema_arrow.names)
        rows, offset = [], 0
        for batch in parquet.iter_batches(batch_size=16384, columns=columns, use_threads=False):
            target = pc.equal(pc.cast(batch.column("pmid"), pa.string()), "36215666")
            indices = pc.indices_nonzero(target).to_pylist()
            selected = batch.filter(target).to_pylist()
            for position, row in zip(indices, selected, strict=True):
                rows.append({"source_row": offset + position, **row})
            offset += batch.num_rows
            guard()
        assert rows
        counts = Counter(str(row["mhc_class"]) for row in rows)
        positions = {row["source_row"] for row in rows}
        replay_counts, replay_positions, offset = Counter(), set(), 0
        for batch in parquet.iter_batches(
            batch_size=8192, columns=["pmid", "mhc_class"], use_threads=False
        ):
            for i, row in enumerate(batch.to_pylist()):
                if str(row["pmid"]) == "36215666":
                    replay_counts[str(row["mhc_class"])] += 1
                    replay_positions.add(offset + i)
            offset += batch.num_rows
            guard()
        assert counts == replay_counts and positions == replay_positions
        results = FAMILY / "results"
        results.mkdir(exist_ok=False)
        with gzip.open(results / "corrected-export-rows.jsonl.gz", "wt") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        curation = yaml.safe_load((FAMILY / "reproduce/source/pmid_overrides.yaml").read_text())
        studies = curation if isinstance(curation, list) else curation["overrides"]
        study = next(row for row in studies if row["pmid"] == 36215666)
        mapping = study["elution_condition_ids"]
        class_ii = [row for row in rows if str(row["mhc_class"]) == "II"]
        distributions = {}
        for field in columns:
            if field not in {"peptide", "peptide_extended", "provenance_id"}:
                counter = Counter(json.dumps(row[field], sort_keys=True) for row in class_ii)
                distributions[field] = [
                    {"value": json.loads(key), "n": count} for key, count in sorted(counter.items())
                ]
        membership = Counter()
        for row in class_ii:
            evidence = set()
            for field in ("assay_comments", "antigen_processing_comments"):
                if row[field] in mapping:
                    evidence.add(tuple(sorted(mapping[row[field]])))
            membership[json.dumps(sorted(evidence))] += 1
        inventory = json.loads(
            Path(
                "/private/tmp/presto-mhc2-20261026/human-cell-contrast/pride/PXD024417-files.json"
            ).read_text()
        )
        state.update(
            status="completed",
            source_sha256=source_sha,
            rows=len(rows),
            class_counts=dict(counts),
            class_ii_rows=len(class_ii),
            distinct_class_ii_peptides=len({row["peptide"] for row in class_ii}),
            projected_columns=columns,
            independent_count_and_position_replay=True,
            class_ii_distributions=distributions,
            literal_curated_membership=dict(membership),
            pride_file_count=len(inventory),
            pride_file_extensions=dict(Counter(Path(x["fileName"]).suffix for x in inventory)),
            native_rows_sha256=sha(results / "corrected-export-rows.jsonl.gz"),
        )
        (results / "audit.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)
        state.update(
            elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        receipt.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: state.get(k)
                    for k in (
                        "status",
                        "error",
                        "rows",
                        "class_counts",
                        "elapsed_seconds",
                        "peak_rss_bytes",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
