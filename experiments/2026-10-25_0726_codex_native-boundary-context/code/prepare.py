"""Reconcile boundary origins and fixed fold support before reading targets."""

from collections import Counter, defaultdict
import hashlib

import numpy as np

from context import endpoint, names, vectors
from core import CFG, ROOT, dump, inner_fold
from data import outer_fold, read

rows = sorted(
    read("typed/covariates.parquet", [("split", "=", "train")]),
    key=lambda r: r["example_id"],
)
assert len(rows) == CFG["expected_n_contrasts"] and all(
    r["split"] == "train" for r in rows
)
assert all("target_log10_ratio_change" not in r for r in rows)
groups = sorted({r["dependency_group_id"] for r in rows})
parents = read("typed/parents.parquet", [("split", "=", "train")])
assert (
    len(groups) == CFG["expected_n_components"]
    and len(parents) == CFG["expected_n_parents"]
)
parent_map = {p["parent_sequence_sha256"]: p for p in parents}
assert len(parent_map) == len(parents)
for digest, p in parent_map.items():
    assert hashlib.sha256(p["parent_sequence"].encode()).hexdigest() == digest
peptides = sorted({r[k] for r in rows for k in ("peptide_id", "reference_peptide_id")})
origins = read(
    "source_tables/origins.parquet",
    [
        ("peptide_id", "in", peptides),
        ("reference_sequence_sha256", "in", sorted(parent_map)),
    ],
)
by_endpoint = defaultdict(list)
for origin in origins:
    by_endpoint[origin["peptide_id"], origin["reference_sequence_sha256"]].append(
        origin
    )
features = defaultdict(list)
records = {}
for row in rows:
    digest = row["parent_sequence_sha256"]
    p = parent_map[digest]
    assert row["dependency_group_id"] == p["dependency_group_id"]
    parent = p["parent_sequence"]
    for prefix in ("peptide", "reference_peptide"):
        pid, peptide = row[prefix + "_id"], row[prefix + "_sequence"]
        vector, matches = endpoint(parent, peptide)
        key = pid + "|" + digest
        source = by_endpoint[pid, digest]
        expected = {
            (x["candidate_start_1based"], x["candidate_end_1based"]) for x in source
        }
        actual = {(x["start_1based"], x["end_1based"]) for x in matches}
        assert expected == actual and actual, key
        records[key] = dict(
            peptide_id=pid,
            peptide_sequence=peptide,
            parent_sequence_sha256=digest,
            occurrences=matches,
            source_origin_ids=sorted(x["origin_id"] for x in source),
            endpoint_features=vector.tolist(),
        )
    for kind, vector in vectors(
        parent, row["peptide_sequence"], row["reference_peptide_sequence"]
    ).items():
        features[kind].append(vector)
folds = {}
for fold in range(5):
    omitted = [g for g in groups if outer_fold(g) == fold]
    fitted = [g for g in groups if outer_fold(g) != fold]
    counts = {str(j): sum(inner_fold(g) == j for g in fitted) for j in range(3)}
    assert len(omitted) >= 5 and min(counts.values()) >= 5, (
        "Insufficient fixed folds; do not change hashes"
    )
    folds[str(fold)] = dict(
        development_components=omitted, fitting_components=fitted, inner_counts=counts
    )
np.savez_compressed(
    ROOT / "results/features.npz", **{k: np.asarray(v) for k, v in features.items()}
)
dump("results/endpoint-contexts.json", records)
dump("results/feature-names.json", names())
dump(
    "results/preparation.json",
    dict(
        passed=True,
        n_contrasts=len(rows),
        n_components=len(groups),
        n_parents=len(parents),
        n_endpoints=len(records),
        occurrence_counts=dict(
            Counter(len(r["occurrences"]) for r in records.values())
        ),
        example_ids=[r["example_id"] for r in rows],
        folds=folds,
        targets_read=False,
        no_outer_rows_materialized=True,
    ),
)
print("Training-only context origin mapping and fixed fold support passed")
