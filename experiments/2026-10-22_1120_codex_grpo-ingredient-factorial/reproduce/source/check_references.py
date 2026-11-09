"""Fresh-process initial-policy and prior-run parity checks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


FAMILY = Path(__file__).resolve().parents[1]
RUN = FAMILY / "results/run"
SPEC = importlib.util.spec_from_file_location(
    "frozen_reference", FAMILY / "reproduce/source/base.py"
)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def main():
    torch.set_num_threads(1)
    rows, _ = base.make_dataset()
    reference_checks, prior_comparisons = [], []
    previous = FAMILY.parent / "2026-10-21_1700_codex_discrete-distribution-comparison/results/run"
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        model = base.Policy()
        saved = np.load(RUN / f"reference-seed{seed}.npz")
        for split in ("train", "validation", "test"):
            subset = [row for row in rows if row["split"] == split]
            actual = np.asarray(
                [row["prediction"] for row in base.predict(model, subset)], dtype=np.float32
            )
            np.testing.assert_array_equal(actual, saved[split])
            reference_checks.append({"seed": seed, "split": split, "status": "exact_match"})
            for old_name, new_name in (
                ("exact", "exact-b0"),
                ("reinforce8", "original-reinforce8"),
            ):
                old_path = previous / f"{old_name}-seed{seed}/{split}-predictions.jsonl"
                if not old_path.exists():
                    prior_comparisons.append({"path": str(old_path), "status": "not_available"})
                    continue
                old = np.asarray(
                    [json.loads(line)["prediction"] for line in old_path.read_text().splitlines()]
                )
                new = np.load(RUN / f"{new_name}-seed{seed}/predictions.npz")[split]
                delta = float(np.abs(new - old).max())
                prior_comparisons.append(
                    {
                        "condition": new_name,
                        "seed": seed,
                        "split": split,
                        "maximum_difference": delta,
                    }
                )
    receipt = {
        "status": "passed",
        "reference_checks": reference_checks,
        "prior_saved_prediction_comparisons": prior_comparisons,
        "note": (
            "Reconstructs initial reference probabilities on each split. Selected-model "
            "comparisons use saved arrays; no checkpoint reselection or rescoring."
        ),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (FAMILY / "results/reference-replay.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "references": len(reference_checks),
                "prior_comparisons": len(prior_comparisons),
            }
        )
    )


if __name__ == "__main__":
    main()
