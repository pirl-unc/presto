"""Two training objectives scored on both fixed development target spaces."""

import numpy as np
import pyarrow.parquet as pq

from core import CFG, ROOT, dump, metrics
from data import outer, spaces
from stats import bootstrap, decision

data = spaces()
saved = pq.read_table(ROOT / "results/development-oof-predictions.parquet").to_pylist()
lookup = {
    (r["evaluation_space"], r["training_objective"], r["kind"], r["example_id"]): r
    for r in saved
}
assert len(lookup) == len(saved)
scores, component, uncertainty, intervals_by_space = {}, {}, {}, {}
for space, rows in data.items():
    predictions = {}
    for objective in CFG["training_objectives"]:
        for kind in CFG["representations"]:
            selected = [
                lookup.pop((space, objective, kind, r["example_id"])) for r in rows
            ]
            assert all(
                v["outer_fold"] == outer(r["dependency_group_id"])
                and v["target"] == r["target_log10_ratio_change"]
                and v["weight"] == r["weight"]
                for r, v in zip(rows, selected)
            )
            predictions[objective + ":" + kind] = np.array(
                [r["prediction"] for r in selected]
            )
    pooled = {key: metrics(rows, values) for key, values in predictions.items()}
    folds = {}
    for fold in range(5):
        mask = np.array([outer(r["dependency_group_id"]) == fold for r in rows])
        subset = [r for r, yes in zip(rows, mask) if yes]
        folds[str(fold)] = {
            key: metrics(subset, values[mask]) for key, values in predictions.items()
        }
    scores[space] = dict(pooled=pooled, folds=folds)
    groups, errors, indices, rmses, gains, intervals = bootstrap(rows, predictions)
    component[space] = dict(groups=groups, mse=errors)
    uncertainty[space + "__indices"] = indices
    for prefix, collection in (("rmse", rmses), ("gain", gains)):
        uncertainty.update(
            {
                space + "__" + prefix + "__" + key: value
                for key, value in collection.items()
            }
        )
    intervals_by_space[space] = intervals
assert not lookup
dump("results/metrics.json", scores)
dump("results/component-errors.json", component)
np.savez_compressed(ROOT / "results/bootstrap.npz", **uncertainty)
dump("results/intervals.json", intervals_by_space)
dump(
    "results/decision.json",
    decision(
        scores["all_pairs"]["pooled"],
        scores["all_pairs"]["folds"],
        intervals_by_space["all_pairs"],
    ),
)
print("Matched pair-space decision and separate reference-space diagnostics recorded")
