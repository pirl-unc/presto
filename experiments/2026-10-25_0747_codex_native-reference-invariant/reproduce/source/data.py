"""Shared integrity checks and training-only reads for the fixed cohort."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from core import CFG, ROOT

PKG = Path(CFG["package"])
PRIOR = Path(CFG["reference_family"])
MAN = json.loads((PKG / "manifest.json").read_text())


def read(relative, filters, columns=None):
    path = PKG / relative
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == MAN["files"][relative]["sha256"]
    )
    return pq.read_table(path, filters=filters, columns=columns).to_pylist()


def outer(group):
    return (
        int(
            hashlib.sha256((CFG["outer_namespace"] + "|" + group).encode()).hexdigest(),
            16,
        )
        % 5
    )


def spaces():
    return {
        name: json.loads((ROOT / "results" / (name + "-rows.json")).read_text())
        for name in ("reference", "all_pairs")
    }
