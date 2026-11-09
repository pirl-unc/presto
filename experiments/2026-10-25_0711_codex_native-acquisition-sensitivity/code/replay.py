"""Independent scalar replay from original admitted quantity cells; no core imports."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
PKG = Path(CFG["package"])
MAN = json.loads((PKG / "manifest.json").read_text())
LABELS = ["Cont1", "Cont2", "Cont3", "KD1", "KD2", "KD3"]


def read(name, filters):
    p = PKG / name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == MAN["files"][name]["sha256"]
    return pq.read_table(p, filters=filters).to_pylist()


rows = sorted(
    read("typed/examples.parquet", [("split", "=", "train")]),
    key=lambda r: r["example_id"],
)
assert len(rows) == 158 and all(r["split"] == "train" for r in rows)
ids = [r["example_id"] for r in rows]
cs = read("typed/contributors.parquet", [("example_id", "in", ids)])
qs = read(
    "source_tables/quantities.parquet",
    [("measurement_id", "in", sorted({c["measurement_id"] for c in cs}))],
)
quantities = {q["measurement_id"]: q for q in qs}
assert len(quantities) == len(qs)
lookup = defaultdict(dict)
for c in cs:
    q = quantities[c["measurement_id"]]
    for key in (
        "source_record_id",
        "peptide_id",
        "acquisition_id",
        "quantitative_raw",
        "quantitative_value",
    ):
        assert c[key] == q[key], key
    assert float(q["quantitative_raw"]) == q["quantitative_value"] > 0
    key = c["sample_label_raw"], c["quantity_role"]
    assert key not in lookup[c["example_id"]]
    lookup[c["example_id"]][key] = float(q["quantitative_raw"])


def mean(xs):
    return math.fsum(xs) / len(xs)


w = [r["weight"] for r in rows]
ws = math.fsum(w)
w = [v / ws for v in w]


def dot(xs):
    return math.fsum(a * b for a, b in zip(xs, w))


def spread(xs):
    m = dot(xs)
    return dict(
        mean=m,
        rms=math.sqrt(dot([x * x for x in xs])),
        sd=math.sqrt(dot([(x - m) ** 2 for x in xs])),
    )


def sign(x):
    return (x > 0) - (x < 0)


def agreement(a, b):
    ix = [i for i in range(len(a)) if a[i] != 0 and b[i] != 0]
    mass = math.fsum(w[i] for i in ix)
    same = math.fsum(w[i] for i in ix if sign(a[i]) == sign(b[i]))
    opposite = math.fsum(w[i] for i in ix if sign(a[i]) != sign(b[i]))
    return dict(
        eligible_weight=mass,
        excluded_zero_count=len(a) - len(ix),
        same_sign_fraction=same / mass if mass else None,
        opposite_sign_fraction=opposite / mass if mass else None,
    )


def corr(a, b):
    if min(a) == max(a) or min(b) == max(b):
        return None
    ac = [x - dot(a) for x in a]
    bc = [x - dot(b) for x in b]
    return dot([x * y for x, y in zip(ac, bc)]) / math.sqrt(
        dot([x * x for x in ac]) * dot([x * x for x in bc])
    )


def ranks(xs):
    indices = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    start = 0
    while start < len(xs):
        end = start + 1
        while end < len(xs) and xs[indices[end]] == xs[indices[start]]:
            end += 1
        for k in range(start, end):
            out[indices[k]] = (start + 1 + end) / 2
        start = end
    return out


ratio, ys, residuals, omissions, partitions, stable = [], [], [], [], [], []
for row in rows:
    cells = lookup[row["example_id"]]
    assert len(cells) == 12
    r = [
        math.log10(cells[label, "peptide"] / cells[label, "reference"])
        for label in LABELS
    ]
    y = mean(r[3:]) - mean(r[:3])
    resid = [v - mean(r[:3] if j < 3 else r[3:]) for j, v in enumerate(r)]
    om = [
        mean(r[3:]) - mean([r[k] for k in range(3) if k != j])
        if j < 3
        else mean([r[k] for k in range(3, 6) if k != j]) - mean(r[:3])
        for j in range(6)
    ]
    ps = [
        (
            r[k] - r[c],
            mean([r[i] for i in range(3, 6) if i != k])
            - mean([r[i] for i in range(3) if i != c]),
        )
        for c in range(3)
        for k in range(3, 6)
    ]
    ratio.append(r)
    ys.append(y)
    residuals.append(resid)
    omissions.append(om)
    partitions.append(ps)
    stable.append(y != 0 and all(sign(x) == sign(y) for x in om))


count, largest = 0, 0.0


def check(expected, actual, path=""):
    global count, largest
    if isinstance(expected, dict):
        assert set(expected) == set(actual), path
        for key in expected:
            check(expected[key], actual[key], path + "/" + key)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for i, (a, b) in enumerate(zip(expected, actual)):
            check(a, b, path + f"/{i}")
    elif isinstance(expected, float):
        assert math.isfinite(expected) and math.isfinite(actual), path
        difference = abs(expected - actual)
        largest = max(largest, difference)
        assert math.isclose(
            expected, actual, abs_tol=CFG["atol"], rel_tol=CFG["rtol"]
        ), (path, expected, actual)
        count += 1
    else:
        assert expected == actual, (path, expected, actual)


saved = pq.read_table(ROOT / "results/training-diagnostics.parquet").to_pylist()
for i, (row, output) in enumerate(zip(rows, saved)):
    check(json.loads(row["acquisition_log_ratios_json"]), ratio[i])
    check(float(row["target_log10_ratio_change"]), ys[i])
    e = {
        key: row[key]
        for key in (
            "example_id",
            "dependency_group_id",
            "parent_sequence_sha256",
            "split",
            "weight",
        )
    }
    e.update(target=ys[i], stable_nonzero_sign=stable[i])
    for j, label in enumerate(LABELS):
        e.update(
            {
                f"ratio_{label}": ratio[i][j],
                f"residual_{label}": residuals[i][j],
                f"omitting_{label}": omissions[i][j],
                f"change_{label}": omissions[i][j] - ys[i],
            }
        )
    for j, (c, k) in enumerate((c, k) for c in LABELS[:3] for k in LABELS[3:]):
        e["selected_" + c + "_" + k], e["remaining_" + c + "_" + k] = partitions[i][j]
    check(e, output)

response = spread(ys)
within = math.sqrt(dot([mean([x * x for x in rr]) for rr in residuals]))
eligible = math.fsum(w[i] for i, y in enumerate(ys) if y != 0)
summary = dict(
    n_contrasts=len(rows),
    response=response,
    within_arm_rms=within,
    within_control_rms=math.sqrt(
        dot([mean([x * x for x in rr[:3]]) for rr in residuals])
    ),
    within_kd_rms=math.sqrt(dot([mean([x * x for x in rr[3:]]) for rr in residuals])),
    within_to_response_rms_ratio=within / response["rms"] if response["rms"] else None,
    acquisitions=[
        dict(label=label, **spread([rr[j] for rr in residuals]))
        for j, label in enumerate(LABELS)
    ],
    omissions=[],
    partitions=[],
    stable_nonzero_sign_fraction=math.fsum(w[i] for i, s in enumerate(stable) if s)
    / eligible
    if eligible
    else None,
    sign_eligible_weight=eligible,
    zero_original_response_count=sum(y == 0 for y in ys),
)
for j, label in enumerate(LABELS):
    om = [rr[j] for rr in omissions]
    delta = [a - b for a, b in zip(om, ys)]
    summary["omissions"].append(
        dict(
            omitted_acquisition=label,
            response=spread(om),
            change=spread(delta),
            mean_absolute_change=dot([abs(x) for x in delta]),
            sign=agreement(ys, om),
        )
    )
for j, (c, k) in enumerate((c, k) for c in LABELS[:3] for k in LABELS[3:]):
    a, b = [p[j][0] for p in partitions], [p[j][1] for p in partitions]
    summary["partitions"].append(
        dict(
            control=c,
            kd=k,
            selected=spread(a),
            remaining=spread(b),
            pearson=corr(a, b),
            spearman=corr(ranks(a), ranks(b)),
            sign=agreement(a, b),
        )
    )
values = [p["pearson"] for p in summary["partitions"] if p["pearson"] is not None]
median = statistics.median(values) if values else None
summary.update(
    median_partition_pearson=median,
    defined_partition_correlations=len(values),
    next_step="training_only_boundary_context_comparison"
    if median is not None and median > 0
    else "observation_stability_and_replication_metadata",
    n_components=len({r["dependency_group_id"] for r in rows}),
    n_parents=len({r["parent_sequence_sha256"] for r in rows}),
    no_model_fitted=True,
    no_outer_rows_materialized=True,
)
check(summary, json.loads((ROOT / "results/summary.json").read_text()))
components = []
for group in sorted({r["dependency_group_id"] for r in rows}):
    ix = [i for i, r in enumerate(rows) if r["dependency_group_id"] == group]
    components.append(
        dict(
            dependency_group_id=group,
            n_contrasts=len(ix),
            weight=math.fsum(w[i] for i in ix),
            response_squared_contribution=math.fsum(w[i] * ys[i] ** 2 for i in ix),
            within_squared_contribution=math.fsum(
                w[i] * mean([x * x for x in residuals[i]]) for i in ix
            ),
            omission_change_squared_contribution=math.fsum(
                w[i] * mean([(x - ys[i]) ** 2 for x in omissions[i]]) for i in ix
            ),
            stable_sign_weight=math.fsum(w[i] for i in ix if stable[i]),
        )
    )
check(components, json.loads((ROOT / "results/components.json").read_text()))
receipt = dict(
    passed=True,
    original_quantity_cells=len(qs),
    contributors=len(cs),
    contrasts=len(rows),
    components=len(components),
    scalar_values_checked=count,
    max_absolute_difference=largest,
    rtol=CFG["rtol"],
    atol=CFG["atol"],
    no_production_core_import=True,
    no_outer_rows_materialized=True,
)
(ROOT / "results/replay.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
