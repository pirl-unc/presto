"""Single frozen validation/test evaluation with component-bootstrap uncertainty."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core import CFG, ROOT, bootstrap, decision, dump, metrics, predict

fitted = json.loads((ROOT / "results/fit-summary.json").read_text())
assert fitted["passed"]
assert (
    hashlib.sha256((ROOT / "results/models.json").read_bytes()).hexdigest()
    == fitted["models_sha256"]
)
models = json.loads((ROOT / "results/models.json").read_text())
scores, intervals = {}, {}
for part in ("val", "test"):
    rows = pq.read_table(
        Path(CFG["package"]) / "typed/examples.parquet", filters=[("split", "=", part)]
    ).to_pylist()
    assert len(rows) == CFG["expected_n_contrasts"][part]
    predictions = {kind: predict(rows, model) for kind, model in models.items()}
    scores[part] = {kind: metrics(rows, values) for kind, values in predictions.items()}
    output = [
        {**r, "condition": kind, "prediction": float(p)}
        for kind, values in predictions.items()
        for r, p in zip(rows, values, strict=True)
    ]
    pq.write_table(
        pa.Table.from_pylist(output), ROOT / "results" / (part + "-predictions.parquet")
    )
    groups, errors, sampled, rmses, gains, intervals[part] = bootstrap(
        rows, predictions
    )
    np.savez_compressed(
        ROOT / "results" / (part + "-bootstrap.npz"),
        sampled_component_indices=sampled,
        **{"rmse_" + k: v for k, v in rmses.items()},
        **{"gain_vs_" + k: v for k, v in gains.items()},
    )
    dump(
        "results/" + part + "-component-errors.json",
        dict(components=groups, conditional_mse=errors),
    )
dump("results/heldout-metrics.json", scores)
dump("results/decision.json", decision(scores, intervals))
print("All three fixed conditions evaluated on validation and test", flush=True)
