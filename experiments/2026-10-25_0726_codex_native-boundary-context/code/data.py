"""Read the frozen package with training predicates before row materialization."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from core import CFG, ROOT

PKG = Path(CFG["package"])
MANIFEST = json.loads((PKG / "manifest.json").read_text())


def read(relative, filters):
    p = PKG / relative
    assert (
        hashlib.sha256(p.read_bytes()).hexdigest()
        == MANIFEST["files"][relative]["sha256"]
    )
    return pq.read_table(p, filters=filters).to_pylist()


def outer_fold(group):
    return (
        int(
            hashlib.sha256((CFG["outer_namespace"] + "|" + group).encode()).hexdigest(),
            16,
        )
        % 5
    )


def training():
    rows = sorted(
        read("typed/examples.parquet", [("split", "=", "train")]),
        key=lambda r: r["example_id"],
    )
    assert len(rows) == CFG["expected_n_contrasts"] and all(
        r["split"] == "train" for r in rows
    )
    prepare = json.loads((ROOT / "results/preparation.json").read_text())
    assert prepare["passed"]
    assert [r["example_id"] for r in rows] == prepare["example_ids"]
    with np.load(ROOT / "results/features.npz", allow_pickle=False) as features:
        for i, row in enumerate(rows):
            row["features"] = {
                kind: features[kind][i].tolist()
                for kind in ("length", "sequence", "boundary")
            }
    return rows
