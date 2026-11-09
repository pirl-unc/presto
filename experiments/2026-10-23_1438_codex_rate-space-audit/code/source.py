"""Independent source reader and NumPy temporal forwards; prior test excluded."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read_groups(cfg):
    path = Path(cfg["recipe_directory"])
    assert (
        hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
        == cfg["recipe_manifest_sha256"]
    )
    manifest = json.loads((path / "manifest.json").read_text())
    for name in ("examples.json", "components.json", "recipe.json"):
        assert (
            hashlib.sha256((path / name).read_bytes()).hexdigest()
            == manifest["files"][name]["sha256"]
        )
    source = json.loads((path / "examples.json").read_text())
    grouped = defaultdict(list)
    for row in source:
        if row["split"] != "train":
            continue
        grouped[row["condition_id"]].append(row)
    groups = []
    for condition, curves in sorted(grouped.items()):
        first = curves[0]
        sequence = first["parent_sequence"]
        ctx = first["context"]
        columns = [np.ones(len(sequence) - 1)]
        for side in (sequence[:-1], sequence[1:]):
            for aa in AA[1:]:
                columns.append(np.array([float(residue == aa) for residue in side]))
        context = math.log(
            ctx["substrate_concentration_um"]
            / ctx["enzyme_substrate_denominator"]
            / 0.2
        )
        columns.append(np.full(len(sequence) - 1, context))
        x = np.stack(columns, axis=1)
        rows = []
        for curve in curves:
            values = {v["time_hours"]: v["target_value"] for v in curve["observations"]}
            times = curve["common_positive_times"]
            ref = times[0]
            for t in times[1:]:
                rows.append((curve, t, ref, values[t], values[ref]))
        groups.append(
            dict(
                condition=condition,
                parent=first["construct_id"],
                split=first["split"],
                sequence=sequence,
                x=x,
                rows=rows,
                ncurves=len(curves),
                a=np.array([r[0]["start_0based"] for r in rows]),
                b=np.array([r[0]["end_0based"] for r in rows]),
                t=np.array([r[1] for r in rows]),
                ref=np.array([r[2] for r in rows]),
                y=np.array([math.log10(r[3]) - math.log10(r[4]) for r in rows]),
            )
        )
    for split in ("train",):
        subset = [g for g in groups if g["split"] == split]
        parents = Counter(g["parent"] for g in subset)
        for g in subset:
            g["w"] = np.array(
                [
                    1
                    / (
                        len(parents)
                        * parents[g["parent"]]
                        * g["ncurves"]
                        * (len(r[0]["common_positive_times"]) - 1)
                    )
                    for r in g["rows"]
                ]
            )
    return groups
