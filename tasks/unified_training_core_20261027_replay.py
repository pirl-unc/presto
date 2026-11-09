"""Read-only migration regression against existing frozen predictions; no fit/eval study."""

from pathlib import Path
import gzip
import hashlib
import json

import torch

from presto.data.modular import Observation
from presto.inference.modular import ModularPredictor


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    family = Path(
        "/private/tmp/presto-mhc2-evaluation-20261027/presto/experiments/"
        "2026-10-27_1250_codex_do-dm-native-memory-fit"
    )
    closure = json.loads((family / "results/closure.json").read_text())
    for name, digest in closure["files"].items():
        assert hashlib.sha256((family / name).read_bytes()).hexdigest() == digest, name
    with gzip.open(family / "results/train-predictions.jsonl.gz", "rt") as stream:
        expected = [json.loads(next(stream)) for _ in range(8)]
    by_id = {row["response_id"]: row for row in expected}
    variants = {}
    for variant in ("native_levels", "native_collapsed", "native_shuffled"):
        with (family / f"results/inputs/{variant}.jsonl").open() as stream:
            rows = [
                Observation.from_dict(row)
                for line in stream
                if (row := json.loads(line))["observation_id"] in by_id
            ]
        checkpoint = family / f"results/runs/{variant}/selected.pt"
        predictor = ModularPredictor(checkpoint)
        assert predictor.model.__class__.__module__ == "presto.legacy.modular_v2"
        # Existing reference predictions used single-row batches. Keep the numerical contract.
        predictions = [predictor.predict([row])[0] for row in rows]
        differences = [
            abs(row["transformed_value"] - by_id[row["observation_id"]]["predictions"][variant])
            for row in predictions
        ]
        assert len(differences) == 8
        assert max(differences) == 0.0, (variant, differences)
        variants[variant] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "rows": len(rows),
            "max_absolute_difference": max(differences),
        }
    result = {
        "status": "passed",
        "kind": "historical_checkpoint_compatibility_regression",
        "biological_fit_or_new_accuracy_evaluation": False,
        "reference_scope": "first eight already-recorded training predictions per arm",
        "frozen_files_verified": len(closure["files"]),
        "closure_sha256": hashlib.sha256(
            (family / "results/closure.json").read_bytes()
        ).hexdigest(),
        "variants": variants,
    }
    target = Path(__file__).with_suffix(".json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
