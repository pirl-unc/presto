"""Apply the frozen baseline and dependence contract to original training rows."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from core import ROOT, dump, sequences, source_changes

rows, baselines = source_changes()
support = []
passed = True
for outer in range(5):
    training = [r for r in rows if r["outer_fold"] != outer]
    validation = [r for r in rows if r["outer_fold"] == outer]
    train_seq, val_seq = set(sequences(training)), set(sequences(validation))
    assert not train_seq & val_seq
    cell = {"outer_fold": outer, "n_training_sequences": len(train_seq),
            "n_validation_sequences": len(val_seq), "n_training_observations": len(training),
            "n_validation_observations": len(validation), "inner": []}
    passed &= len(val_seq) >= 10
    for inner in range(3):
        inner_train = [r for r in training if r["inner_fold"] != inner]
        inner_val = [r for r in training if r["inner_fold"] == inner]
        assert not set(sequences(inner_train)) & set(sequences(inner_val))
        count = len(sequences(inner_val))
        passed &= count >= 5
        cell["inner"].append({"inner_fold": inner, "n_training_sequences": len(sequences(inner_train)),
                              "n_validation_sequences": count})
    support.append(cell)
pq.write_table(pa.Table.from_pylist(rows), ROOT / "results/changes.parquet")
dump("results/baselines.json", baselines)
result = {"passed": bool(passed), "scope": "original_training_only", "n_sequences": len(baselines),
          "n_observations": len(rows), "support": support,
          "heldout_original_outcomes_read": False, "baseline_is_supplied_conditioning_measurement": True}
dump("results/preparation.json", result)
print(json.dumps(result), flush=True)
if not passed:
    raise SystemExit("Frozen support gate failed; no change to folds or thresholds")
