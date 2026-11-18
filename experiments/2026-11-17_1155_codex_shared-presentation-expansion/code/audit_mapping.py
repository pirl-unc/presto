"""Local reference eligibility audit; no new split or prediction metrics."""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from prepare import rows, sha
from presto.data.presentation import (
    map_peptides,
    protein_families,
    read_gencode_proteins,
    source_junction,
)

root = Path(__file__).resolve().parents[3]
e = Path(__file__).resolve().parents[1]
start = time.monotonic()
peptides = defaultdict(set)
for row in rows(e / "results/local-admission/admitted.jsonl.gz"):
    peptides[row["mhc"][0]["mhc_class"]].add(row["peptide"])
print("peptides", {k: len(v) for k, v in peptides.items()}, flush=True)
proteins = read_gencode_proteins(
    root / "data/shared-presentation-20261117/proteins.fa.gz"
)
hits = map_peptides(set.union(*peptides.values()), proteins)
families = protein_families(hits)
counts = {}
for cls, seqs in peptides.items():
    c = Counter()
    for p in seqs:
        if p not in hits:
            c["unmapped"] += 1
            continue
        _, policy = source_junction(p, hits[p], proteins)
        c[policy] += 1
    counts[cls] = dict(c)
path = e / "results/local-reference-mappings.parquet"
schema = pa.schema(
    [
        ("peptide", pa.string()),
        ("protein_id", pa.string()),
        ("start", pa.int32()),
        ("family", pa.string()),
    ]
)
with pq.ParquetWriter(path, schema, compression="zstd") as out:
    batch = []
    for peptide, occurrences in hits.items():
        for protein, start_at in occurrences:
            batch.append(
                {
                    "peptide": peptide,
                    "protein_id": protein,
                    "start": start_at,
                    "family": families[protein],
                }
            )
        if len(batch) >= 50000:
            out.write_table(pa.Table.from_pylist(batch, schema=schema))
            batch = []
    if batch:
        out.write_table(pa.Table.from_pylist(batch, schema=schema))
report = {
    "status": "mapped",
    "purpose": "source eligibility only; final family partitions also incorporate historical training and validation identities",
    "unique_peptides_by_class": {k: len(v) for k, v in peptides.items()},
    "junction_counts": counts,
    "proteins": len(proteins),
    "families": len(set(families.values())),
    "mapped_peptides": len(hits),
    "mapping_sha256": sha(path),
    "seconds": time.monotonic() - start,
    "test_accessed": False,
}
(e / "results/local-reference-summary.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report), flush=True)
