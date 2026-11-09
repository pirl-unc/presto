"""Materialize only fixed training records and compute acquisition sensitivity."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core import LABELS, diagnostics

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
PKG = Path(CFG["package"])


def dump(name, value):
    (ROOT / "results" / name).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )


manifest = json.loads((PKG / "manifest.json").read_text())
read_log = []


def read(relative, filters):
    p = PKG / relative
    assert (
        hashlib.sha256(p.read_bytes()).hexdigest()
        == manifest["files"][relative]["sha256"]
    )
    t = pq.read_table(p, filters=filters)
    read_log.append(dict(path=relative, filters=filters, n_materialized_rows=len(t)))
    return t.to_pylist()


rows = read("typed/examples.parquet", [("split", "=", "train")])
rows.sort(key=lambda row: row["example_id"])
assert len(rows) == CFG["expected_training_contrasts"]
assert all(row["split"] == "train" for row in rows)
assert (
    len({r["dependency_group_id"] for r in rows}) == CFG["expected_training_components"]
)
assert (
    len({r["parent_sequence_sha256"] for r in rows}) == CFG["expected_training_parents"]
)
ids = [row["example_id"] for row in rows]
contributors = read("typed/contributors.parquet", [("example_id", "in", ids)])
by_example = defaultdict(list)
for row in contributors:
    by_example[row["example_id"]].append(row)
assert set(by_example) == set(ids)
w = np.array([r["weight"] for r in rows])
assert abs(w.sum() - 1) < 1e-12
ratios = np.array([json.loads(r["acquisition_log_ratios_json"]) for r in rows])
for i, row in enumerate(rows):
    cs = by_example[row["example_id"]]
    assert len(cs) == 12
    assert set(json.loads(row["measurement_ids_json"])) == {
        c["measurement_id"] for c in cs
    }
    for j, label in enumerate(LABELS):
        pair = {c["quantity_role"]: c for c in cs if c["sample_label_raw"] == label}
        assert set(pair) == {"peptide", "reference"}
        v = [float(pair[role]["quantitative_raw"]) for role in ("peptide", "reference")]
        assert min(v) > 0
        assert all(
            v[k] == pair[role]["quantitative_value"]
            for k, role in enumerate(("peptide", "reference"))
        )
        np.testing.assert_allclose(
            np.log10(v[0] / v[1]), ratios[i, j], rtol=CFG["rtol"], atol=CFG["atol"]
        )
summary, y, residual, omitted, pairs, stable = diagnostics(ratios, w)
np.testing.assert_allclose(
    y,
    [r["target_log10_ratio_change"] for r in rows],
    rtol=CFG["rtol"],
    atol=CFG["atol"],
)
summary.update(
    n_components=CFG["expected_training_components"],
    n_parents=CFG["expected_training_parents"],
    no_model_fitted=True,
    no_outer_rows_materialized=True,
)
records = []
for i, row in enumerate(rows):
    result = {
        key: row[key]
        for key in (
            "example_id",
            "dependency_group_id",
            "parent_sequence_sha256",
            "split",
            "weight",
        )
    }
    result.update(target=float(y[i]), stable_nonzero_sign=bool(stable[i]))
    for j, label in enumerate(LABELS):
        result.update(
            {
                f"ratio_{label}": float(ratios[i, j]),
                f"residual_{label}": float(residual[i, j]),
                f"omitting_{label}": float(omitted[i, j]),
                f"change_{label}": float(omitted[i, j] - y[i]),
            }
        )
    for j, part in enumerate(summary["partitions"]):
        key = part["control"] + "_" + part["kd"]
        result["selected_" + key] = float(pairs[i, j, 0])
        result["remaining_" + key] = float(pairs[i, j, 1])
    records.append(result)
pq.write_table(
    pa.Table.from_pylist(records), ROOT / "results/training-diagnostics.parquet"
)
components = []
for group in sorted({r["dependency_group_id"] for r in rows}):
    ix = np.array([r["dependency_group_id"] == group for r in rows])
    mass = w[ix].sum()
    components.append(
        dict(
            dependency_group_id=group,
            n_contrasts=int(ix.sum()),
            weight=float(mass),
            response_squared_contribution=float(w[ix] @ y[ix] ** 2),
            within_squared_contribution=float(w[ix] @ (residual[ix] ** 2).mean(1)),
            omission_change_squared_contribution=float(
                w[ix] @ ((omitted[ix] - y[ix, None]) ** 2).mean(1)
            ),
            stable_sign_weight=float(w[ix & stable].sum()),
        )
    )
dump("summary.json", summary)
dump("components.json", components)
dump(
    "read-scope.json",
    dict(
        reads=read_log,
        note="Parquet filters precede row materialization. Whole-file SHA256 integrity reads include storage bytes of other splits, never diagnostic access to their rows.",
    ),
)
print(json.dumps(summary, allow_nan=False))
