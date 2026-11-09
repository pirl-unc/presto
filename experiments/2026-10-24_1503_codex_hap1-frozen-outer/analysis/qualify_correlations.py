"""Post-evaluation numerical qualification; preserve original scores and predictions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "results/evaluation"
strata = pq.read_table(output / "strata.parquet").to_pylist()
records = []
for split in ["validation", "test"]:
    for scope in ["direct", "erap1"]:
        table = pq.read_table(output / split / f"{scope}-predictions.parquet").to_pydict()
        groups = {
            "condition": np.array(table["condition_id"]),
            "condition_component": np.array(
                [
                    a + ":" + b
                    for a, b in zip(table["condition_id"], table["component_id"], strict=True)
                ]
            ),
            "length": np.array([str(len(p)) for p in table["peptide_sequence"]]),
        }
        predictions = {
            k.removeprefix("prediction_"): np.array(v)
            for k, v in table.items()
            if k.startswith("prediction_")
        }
        target = np.array(table["target_log10_ratio"])
        for row in strata:
            if row["split"] != split or row["scope"] != scope or row["pearson"] is None:
                continue
            ix = groups[row["stratum_type"]] == row["stratum"]
            for name, values in [
                ("prediction", predictions[row["model"]][ix]),
                ("target", target[ix]),
            ]:
                # This is SciPy's floating-point warning criterion, not a tuned biological threshold.
                if np.linalg.norm(values - values.mean()) < np.finfo(float).eps ** 0.75 * abs(
                    values.mean()
                ):
                    records.append({**row, "near_constant": name, "range": float(np.ptp(values))})
                    row["pearson"] = row["spearman"] = None
                    break

pq.write_table(
    pa.Table.from_pylist(strata), output / "strata-qualified.parquet", compression="zstd"
)
receipt = {
    "status": "qualified",
    "criterion": "SciPy float64 near-constant warning criterion",
    "reason": "Five component-stratum correlations are dominated by approximately 1e-14 prediction variation.",
    "predictor_or_primary_metric_changes": False,
    "records": records,
    "original_sha256": hashlib.sha256((output / "strata.parquet").read_bytes()).hexdigest(),
    "qualified_sha256": hashlib.sha256(
        (output / "strata-qualified.parquet").read_bytes()
    ).hexdigest(),
    "interpretation": "Use qualified strata; raw scores and all prediction values are preserved.",
}
(output / "correlation-qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
assert len(records) == 5 and {r["model"] for r in records} == {
    "0810_uniform_exact",
    "1007_length_exact",
}
# Verify the derived table differs only in these explicitly listed correlation cells.
original = pq.read_table(output / "strata.parquet").to_pylist()
qualified = pq.read_table(output / "strata-qualified.parquet").to_pylist()
changes = []
for i, (a, b) in enumerate(zip(original, qualified, strict=True)):
    for key in a:
        if a[key] != b[key]:
            assert key in {"pearson", "spearman"} and b[key] is None
            changes.append((i, key))
assert len(changes) == 10
print(
    {
        "qualified_strata": len(records),
        "changed_cells": len(changes),
        "primary_metrics_unchanged": True,
    }
)
