"""Replay coverage with Arrow kernels and repair scalar category serialization."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import signal
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]


def main():
    signal.alarm(300)
    pa.set_cpu_count(1)
    started = time.monotonic()
    source = HERE / "results/inventory.json"
    inventory = json.loads(source.read_text())
    inputs = json.loads((HERE / "reproduce/execution-inputs.json").read_text())["inputs"]
    checks = {}
    for name, original in inventory.items():
        item = next(item for item in inputs if Path(item["path"]).name == name)
        with Path(item["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
        fields = original["class_ii_nonempty_or_true_field_counts"]
        categories = {key: Counter() for key in original["class_ii_categories"]}
        columns = ["mhc_class", "peptide", "pmid", *fields, *categories]
        counts, peptides, pmids, n_rows = Counter(), set(), set(), 0
        for batch in pq.ParquetFile(item["path"]).iter_batches(
            columns=columns, batch_size=16381, use_threads=False
        ):
            table = pa.Table.from_batches([batch])
            table = table.filter(pc.equal(pc.cast(table["mhc_class"], pa.string()), "II"))
            n_rows += len(table)
            peptides.update(pc.unique(table["peptide"]).to_pylist())
            pmids.update(pc.unique(table["pmid"]).to_pylist())
            for field in fields:
                col = table[field]
                valid = (
                    pc.equal(col, True)
                    if pa.types.is_boolean(col.type)
                    else pc.not_equal(pc.cast(col, pa.string()), "")
                )
                counts[field] += (
                    pc.sum(pc.cast(pc.fill_null(valid, False), pa.int64())).as_py() or 0
                )
            for field, counter in categories.items():
                for row in pc.value_counts(pc.cast(table[field], pa.string())).to_pylist():
                    counter[row["values"]] += row["counts"]
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 3 * 1024**3:
                raise RuntimeError("RSS budget exceeded")
        assert dict(counts) == fields
        assert n_rows == original["n_class_ii_rows"]
        assert len(peptides) == original["n_class_ii_distinct_peptides"]
        assert len(pmids - {None}) == original["n_class_ii_known_pmids"]
        original["class_ii_categories"] = {
            field: [
                {field: key, "n_rows": value}
                for key, value in sorted(counter.items(), key=lambda pair: repr(pair[0]))
            ]
            for field, counter in categories.items()
        }
        checks[name] = {
            "coverage_fields_verified": len(fields),
            "distinct_counts_verified": True,
            "n_category_rows_verified": sum(sum(c.values()) for c in categories.values()),
        }
    report = {
        "passed": True,
        "correction": "Replaced class_ii_categories: original serializer truncated scalar keys.",
        "original_inventory_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checks": checks,
        "seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for filename, value in (
        ("inventory-corrected.json", inventory),
        ("coverage-verification.json", report),
    ):
        with (HERE / "results" / filename).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    with (HERE / "reproduce/source/verify_coverage.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    print(json.dumps(report))
    signal.alarm(0)


if __name__ == "__main__":
    main()
