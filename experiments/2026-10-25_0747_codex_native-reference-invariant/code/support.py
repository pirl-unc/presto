"""Inspect original training endpoint support without materializing effects."""

from collections import Counter, defaultdict

from core import dump, inner_fold
from data import outer, read

rows = sorted(
    read("typed/covariates.parquet", [("split", "=", "train")]),
    key=lambda r: r["example_id"],
)
assert len(rows) == 158 and all(r["split"] == "train" for r in rows)
by_parent = defaultdict(list)
for row in rows:
    assert "target_log10_ratio_change" not in row
    by_parent[row["parent_sequence_sha256"]].append(row)
parent_groups = {p: rs[0]["dependency_group_id"] for p, rs in by_parent.items()}
counts = Counter(parent_groups.values())
assert len(by_parent) == 56 and len(counts) == 55
parents, endpoint_total, pair_total, degrees = [], 0, 0, 0
strata = {
    key: dict(
        n_parents=0,
        n_endpoints=0,
        n_pairs=0,
        component_parent_weight=0.0,
        hierarchical_endpoint_weight=0.0,
    )
    for key in ("n2", "n3", "n4plus")
}
for parent, rs in sorted(by_parent.items()):
    references = {r["reference_peptide_id"] for r in rs}
    assert len(references) == 1
    reference = references.pop()
    nonreference = {r["peptide_id"] for r in rs}
    assert len(nonreference) == len(rs) and reference not in nonreference
    endpoints = sorted({reference, *nonreference})
    n = len(endpoints)
    assert n >= 2
    group = parent_groups[parent]
    assert all(r["dependency_group_id"] == group for r in rs)
    mass = 1 / (55 * counts[group])
    assert all(abs(r["weight"] - mass / (n - 1)) < 1e-14 for r in rs)
    pair_count = n * (n - 1) // 2
    parents.append(
        dict(
            parent_sequence_sha256=parent,
            dependency_group_id=group,
            reference_peptide_id=reference,
            endpoint_ids=endpoints,
            n_endpoints=n,
            n_pairs=pair_count,
            parent_weight=mass,
        )
    )
    bucket = strata["n2" if n == 2 else "n3" if n == 3 else "n4plus"]
    bucket["n_parents"] += 1
    bucket["n_endpoints"] += n
    bucket["n_pairs"] += pair_count
    bucket["component_parent_weight"] += mass
    bucket["hierarchical_endpoint_weight"] += n * (mass / n)
    endpoint_total += n
    pair_total += pair_count
    degrees += n - 1
assert endpoint_total == 214 and degrees == 158
for record in strata.values():
    record["unweighted_parent_fraction"] = record["n_parents"] / 56
    record["unweighted_endpoint_fraction"] = record["n_endpoints"] / 214
groups = sorted(counts)
folds = {}
for f in range(5):
    fit = [g for g in groups if outer(g) != f]
    valid = [g for g in groups if outer(g) == f]
    inner_counts = {str(j): sum(inner_fold(g) == j for g in fit) for j in range(3)}
    assert len(valid) >= 5 and min(inner_counts.values()) >= 5
    folds[str(f)] = dict(
        development_components=valid, fitting_components=fit, inner_counts=inner_counts
    )
dump(
    "results/support.json",
    dict(
        passed=True,
        n_parents=56,
        n_components=55,
        n_endpoints=endpoint_total,
        n_pairs=pair_total,
        n_reference_degrees=degrees,
        strata=strata,
        parents=parents,
        folds=folds,
        targets_read=False,
        no_outer_rows_materialized=True,
        interpretation="Dependent pairs do not increase independent biological sample count.",
    ),
)
print(
    "Target-free fixed support:",
    pair_total,
    "pairs from214endpoints;158within-parent degrees",
)
