"""Freeze three conditions using only the registered training partition."""

import hashlib
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from core import CFG, ROOT, dump, metrics, predict, select

assert json.loads((ROOT / "results/preparation.json").read_text())["passed"]
rows = pq.read_table(
    Path(CFG["package"]) / "typed/examples.parquet", filters=[("split", "=", "train")]
).to_pylist()
assert len(rows) == CFG["expected_n_contrasts"]["train"]
models = {"zero": dict(representation="zero", n_solved_coordinates=0)}
selections, inner_predictions, durations, training_metrics = {}, [], {}, {}
for kind in ("length", "sequence"):
    started = time.perf_counter()
    models[kind], records, predictions, means = select(rows, kind)
    durations[kind] = time.perf_counter() - started
    selections[kind] = dict(
        folds=records, mean_losses={str(k): v for k, v in means.items()}
    )
    inner_predictions.extend(predictions)
dump("results/models.json", models)
dump("results/inner-selection.json", selections)
pq.write_table(
    pa.Table.from_pylist(inner_predictions), ROOT / "results/inner-predictions.parquet"
)
all_predictions = []
for kind, model in models.items():
    values = predict(rows, model)
    training_metrics[kind] = metrics(rows, values)
    all_predictions.extend(
        {**r, "condition": kind, "prediction": float(p)}
        for r, p in zip(rows, values, strict=True)
    )
pq.write_table(
    pa.Table.from_pylist(all_predictions), ROOT / "results/training-predictions.parquet"
)
dump(
    "results/fit-summary.json",
    dict(
        passed=True,
        fit_seconds=durations,
        training_metrics=training_metrics,
        models_sha256=hashlib.sha256(
            (ROOT / "results/models.json").read_bytes()
        ).hexdigest(),
        n_training_contrasts=len(rows),
        heldout_rows_read=0,
    ),
)
print("Three conditions frozen using training data only", flush=True)
