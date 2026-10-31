"""Fit only all-pair models; reuse exact frozen reference-objective controls."""

import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core import CFG, ROOT, dump, predict, select
from data import PRIOR, outer, spaces

data = spaces()
assert json.loads((ROOT / "results/preparation.json").read_text())["passed"]
reference_models = {
    (m["outer_fold"], m["kind"]): m
    for m in json.loads((PRIOR / "results/models.json").read_text())
}
assert len(reference_models) == 20
old_predictions = {
    (r["example_id"], r["kind"]): r
    for r in pq.read_table(
        PRIOR / "results/development-oof-predictions.parquet"
    ).to_pylist()
}
models, inner_fits, inner_predictions, all_predictions, runtimes = [], [], [], [], []
maximum_original_difference = 0.0
for fold in range(5):
    fitted = [r for r in data["all_pairs"] if outer(r["dependency_group_id"]) != fold]
    for kind in CFG["representations"]:
        start = time.perf_counter()
        if kind == "zero":
            model, means = dict(representation="zero"), {}
        else:
            model, choices, inner, means = select(fitted, kind)
            inner_fits.extend(dict(outer_fold=fold, **r) for r in choices)
            inner_predictions.extend(dict(outer_fold=fold, **r) for r in inner)
        runtimes.append(
            dict(outer_fold=fold, kind=kind, core_seconds=time.perf_counter() - start)
        )
        models.append(
            dict(outer_fold=fold, kind=kind, fitted=model, selection_mean_mse=means)
        )
        for objective, current in (
            ("reference", reference_models[fold, kind]["fitted"]),
            ("all_pairs", model),
        ):
            for space, rows in data.items():
                omitted = [r for r in rows if outer(r["dependency_group_id"]) == fold]
                values = predict(omitted, current)
                for row, value in zip(omitted, values, strict=True):
                    if objective == space == "reference":
                        old = old_predictions.pop((row["example_id"], kind))
                        assert (
                            old["outer_fold"] == fold
                            and old["target"] == row["target_log10_ratio_change"]
                            and old["weight"] == row["weight"]
                        )
                        np.testing.assert_allclose(
                            value, old["prediction"], rtol=2e-9, atol=2e-10
                        )
                        maximum_original_difference = max(
                            maximum_original_difference,
                            abs(float(value) - old["prediction"]),
                        )
                    all_predictions.append(
                        dict(
                            training_objective=objective,
                            evaluation_space=space,
                            outer_fold=fold,
                            kind=kind,
                            example_id=row["example_id"],
                            dependency_group_id=row["dependency_group_id"],
                            parent_sequence_sha256=row["parent_sequence_sha256"],
                            original_split="train",
                            weight=row["weight"],
                            target=row["target_log10_ratio_change"],
                            prediction=float(value),
                        )
                    )
assert not old_predictions
dump("results/models.json", models)
dump("results/inner-fits.json", inner_fits)
dump("results/fit-timing.json", runtimes)
pq.write_table(
    pa.Table.from_pylist(inner_predictions), ROOT / "results/inner-predictions.parquet"
)
pq.write_table(
    pa.Table.from_pylist(all_predictions),
    ROOT / "results/development-oof-predictions.parquet",
)
dump(
    "results/fit-summary.json",
    dict(
        new_nonzero_fits=len(inner_fits) + 15,
        reused_final_controls=20,
        original_control_predictions_reproduced=632,
        max_original_prediction_difference=maximum_original_difference,
        n_inner_predictions=len(inner_predictions),
        n_development_predictions=len(all_predictions),
        original_models_refit=False,
        no_outer_rows_materialized=True,
    ),
)
print("New all-pair fits sealed; original controls reused and reproduced")
