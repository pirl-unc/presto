"""Bind original source cells, paired targets and unchanged feature maps."""

from collections import defaultdict
import hashlib
from itertools import combinations
import json

import numpy as np

from algebra import verify
from context import endpoint
from core import ROOT, dump
from data import PRIOR, read

support = json.loads((ROOT / "results/support.json").read_text())
assert support["passed"] and support["targets_read"] is False
original = sorted(
    read("typed/examples.parquet", [("split", "=", "train")]),
    key=lambda r: r["example_id"],
)
assert len(original) == 158 and all(r["split"] == "train" for r in original)
contributors = read(
    "typed/contributors.parquet",
    [("example_id", "in", [r["example_id"] for r in original])],
)
quantities = read(
    "source_tables/quantities.parquet",
    [("measurement_id", "in", sorted({c["measurement_id"] for c in contributors}))],
)
qmap = {q["measurement_id"]: q for q in quantities}
assert len(qmap) == len(quantities) == 1284 and len(contributors) == 1896
labels = ["Cont1", "Cont2", "Cont3", "KD1", "KD2", "KD3"]
cells, provenance = defaultdict(dict), defaultdict(dict)
for c in contributors:
    q = qmap[c["measurement_id"]]
    for field in (
        "source_record_id",
        "peptide_id",
        "acquisition_id",
        "quantitative_raw",
        "quantitative_value",
    ):
        assert c[field] == q[field]
    value = float(q["quantitative_raw"])
    assert value == q["quantitative_value"] > 0
    pid, label = c["peptide_id"], c["sample_label_raw"]
    if label in cells[pid]:
        assert (
            cells[pid][label] == value
            and provenance[pid][label]["measurement_id"] == c["measurement_id"]
        )
    cells[pid][label] = value
    provenance[pid][label] = {
        k: c[k]
        for k in (
            "measurement_id",
            "acquisition_id",
            "sample_label_raw",
            "source_record_id",
            "quantitative_raw",
            "quantitative_value",
        )
    }
assert len(cells) == 214 and all(set(v) == set(labels) for v in cells.values())
parents = {
    r["parent_sequence_sha256"]: r
    for r in read("typed/parents.parquet", [("split", "=", "train")])
}
old_contexts = json.loads((PRIOR / "results/endpoint-contexts.json").read_text())
vectors, effects, records = {}, {}, []
for key, old in sorted(old_contexts.items()):
    pid, digest = old["peptide_id"], old["parent_sequence_sha256"]
    parent = parents[digest]["parent_sequence"]
    assert hashlib.sha256(parent.encode()).hexdigest() == digest
    vector, occurrences = endpoint(parent, old["peptide_sequence"])
    np.testing.assert_array_equal(vector, old["endpoint_features"])
    assert occurrences == old["occurrences"]
    intensities = np.array([cells[pid][label] for label in labels])
    log_values = np.log10(intensities)
    effects[key] = float(log_values[3:].mean() - log_values[:3].mean())
    vectors[key] = vector
    records.append(
        dict(
            endpoint_key=key,
            peptide_id=pid,
            parent_sequence_sha256=digest,
            peptide_sequence=old["peptide_sequence"],
            intensity_values=intensities.tolist(),
            acquisition_labels=labels,
            measurements=[provenance[pid][label] for label in labels],
            unadjusted_log10_fold_change=effects[key],
        )
    )


def features(parent, a, b):
    delta = vectors[a + "|" + parent] - vectors[b + "|" + parent]
    return dict(
        length=delta[:1].tolist(), sequence=delta[:61].tolist(), boundary=delta.tolist()
    )


def target(a, b):
    ratios = np.log10(
        np.array([cells[a][label] for label in labels])
        / np.array([cells[b][label] for label in labels])
    )
    return float(ratios[3:].mean() - ratios[:3].mean())


reference, anchor_effects = [], {}
for row in original:
    parent, a, b = (
        row["parent_sequence_sha256"],
        row["peptide_id"],
        row["reference_peptide_id"],
    )
    y = target(a, b)
    np.testing.assert_allclose(
        y, row["target_log10_ratio_change"], atol=2e-12, rtol=2e-13
    )
    anchor_effects[a + "|" + parent] = row["target_log10_ratio_change"]
    anchor_effects[b + "|" + parent] = 0.0
    # Preserve the exact original target rounding/order for the frozen comparator.
    reference.append({**row, "features": features(parent, a, b)})
with np.load(PRIOR / "results/features.npz", allow_pickle=False) as old:
    for kind in ("length", "sequence", "boundary"):
        np.testing.assert_array_equal(
            [r["features"][kind] for r in reference], old[kind]
        )
paired, checks = [], []
for parent_record in support["parents"]:
    parent = parent_record["parent_sequence_sha256"]
    endpoints = parent_record["endpoint_ids"]
    x = np.array([vectors[pid + "|" + parent] for pid in endpoints])
    y = np.array([effects[pid + "|" + parent] for pid in endpoints])
    checks.append(dict(parent_sequence_sha256=parent, **verify(x, y)))
    for a, b in combinations(endpoints, 2):
        value = target(a, b)
        np.testing.assert_allclose(
            value,
            effects[a + "|" + parent] - effects[b + "|" + parent],
            atol=2e-12,
            rtol=2e-13,
        )
        np.testing.assert_allclose(
            value,
            anchor_effects[a + "|" + parent] - anchor_effects[b + "|" + parent],
            atol=2e-12,
            rtol=2e-13,
        )
        pid = hashlib.sha256(
            ("native-pair-v1|" + parent + "|" + a + "|" + b).encode()
        ).hexdigest()
        paired.append(
            dict(
                example_id=pid,
                peptide_id=a,
                reference_peptide_id=b,
                parent_sequence_sha256=parent,
                dependency_group_id=parent_record["dependency_group_id"],
                split="train",
                weight=parent_record["parent_weight"] / parent_record["n_pairs"],
                target_log10_ratio_change=value,
                features=features(parent, a, b),
            )
        )
paired.sort(key=lambda r: r["example_id"])
assert len(paired) == support["n_pairs"] and len(
    {r["example_id"] for r in paired}
) == len(paired)
assert abs(sum(r["weight"] for r in paired) - 1) < 1e-12
dump("results/reference-rows.json", reference)
dump("results/all_pairs-rows.json", paired)
dump("results/endpoints.json", records)
dump("results/algebra.json", checks)
dump(
    "results/preparation.json",
    dict(
        passed=True,
        n_endpoints=214,
        n_original_quantities=1284,
        n_contributors=1896,
        n_reference_rows=158,
        n_pair_rows=len(paired),
        n_parent_identities=len(checks),
        original_feature_arrays_identical=True,
        maximum_gram_identity_error=max(
            c["maximum_gram_identity_error"] for c in checks
        ),
        maximum_gradient_identity_error=max(
            c["maximum_gradient_identity_error"] for c in checks
        ),
        no_outer_rows_materialized=True,
    ),
)
print(
    "Raw-cell target transforms, feature parity and all parent objective identities verified"
)
