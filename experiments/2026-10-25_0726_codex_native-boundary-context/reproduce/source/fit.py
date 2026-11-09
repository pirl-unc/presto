"""Nested development fitting restricted to original training components."""

import time

import pyarrow as pa
import pyarrow.parquet as pq

from core import CFG, ROOT, dump, predict, select
from data import outer_fold, training

rows = training()
models, selections, inner_predictions, predictions, timings = [], [], [], [], []
for fold in range(5):
    fitted = [r for r in rows if outer_fold(r["dependency_group_id"]) != fold]
    omitted = [r for r in rows if outer_fold(r["dependency_group_id"]) == fold]
    assert not {r["dependency_group_id"] for r in fitted} & {
        r["dependency_group_id"] for r in omitted
    }
    for kind in CFG["representations"]:
        started = time.perf_counter()
        if kind == "zero":
            model, means = dict(representation="zero"), {}
        else:
            model, choices, inner, means = select(fitted, kind)
            selections.extend(dict(outer_fold=fold, **record) for record in choices)
            inner_predictions.extend(
                dict(outer_fold=fold, **record) for record in inner
            )
        values = predict(omitted, model)
        timings.append(
            dict(outer_fold=fold, kind=kind, core_seconds=time.perf_counter() - started)
        )
        models.append(
            dict(outer_fold=fold, kind=kind, fitted=model, selection_mean_mse=means)
        )
        for row, value in zip(omitted, values, strict=True):
            predictions.append(
                dict(
                    example_id=row["example_id"],
                    dependency_group_id=row["dependency_group_id"],
                    parent_sequence_sha256=row["parent_sequence_sha256"],
                    original_split="train",
                    outer_fold=fold,
                    kind=kind,
                    target=row["target_log10_ratio_change"],
                    prediction=float(value),
                    weight=row["weight"],
                )
            )
dump("results/models.json", models)
dump("results/inner-fits.json", selections)
dump("results/fit-timing.json", timings)
pq.write_table(
    pa.Table.from_pylist(inner_predictions), ROOT / "results/inner-predictions.parquet"
)
pq.write_table(
    pa.Table.from_pylist(predictions),
    ROOT / "results/development-oof-predictions.parquet",
)
dump(
    "results/fit-summary.json",
    dict(
        nonzero_fits=len(selections) + 15,
        inner_predictions=len(inner_predictions),
        development_predictions=len(predictions),
        no_outer_rows_materialized=True,
    ),
)
print("Nested development fits sealed; no original validation/test evaluation")
