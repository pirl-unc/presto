"""Fixed development metrics, paired component uncertainty and decision."""

import numpy as np
import pyarrow.parquet as pq

from core import CFG, ROOT, bootstrap, decision, dump, metrics
from data import outer_fold, training

rows = training()
saved = pq.read_table(ROOT / "results/development-oof-predictions.parquet").to_pylist()
lookup = {(r["example_id"], r["kind"]): r for r in saved}
assert len(lookup) == len(saved) == len(rows) * 4
predictions = {}
for kind in CFG["representations"]:
    predictions[kind] = np.array(
        [lookup[r["example_id"], kind]["prediction"] for r in rows]
    )
    assert all(
        lookup[r["example_id"], kind]["outer_fold"]
        == outer_fold(r["dependency_group_id"])
        for r in rows
    )
pooled = {kind: metrics(rows, value) for kind, value in predictions.items()}
folds = {}
for fold in range(5):
    ix = np.array([outer_fold(r["dependency_group_id"]) == fold for r in rows])
    subset = [r for r, include in zip(rows, ix) if include]
    folds[str(fold)] = {
        kind: metrics(subset, value[ix]) for kind, value in predictions.items()
    }
groups, errors, indices, rmses, gains, intervals = bootstrap(rows, predictions)
dump("results/metrics.json", dict(pooled=pooled, folds=folds))
dump("results/component-errors.json", dict(groups=groups, mse=errors))
np.savez_compressed(
    ROOT / "results/bootstrap.npz",
    indices=indices,
    **{"rmse_" + k: v for k, v in rmses.items()},
    **{"gain_" + k: v for k, v in gains.items()},
)
dump("results/decision.json", decision(pooled, folds, intervals))
print("Development metrics and fixed decision recorded")
