"""Posthoc initial/final training-only replay, without model selection or fitting."""

import gzip
import json
import shutil
import time

import numpy as np
import torch
from verify_results import (
    ARTIFACTS,
    COHORT,
    FAMILY,
    forward,
    independent_metrics,
    read,
    rows,
)


def main():
    tick = time.monotonic()
    torch.set_num_threads(1)
    examples = [r for r in rows(COHORT / "examples.jsonl") if r["split"] == "train"]
    parents = {
        r["parent_sequence_sha256"]: r["sequence"]
        for r in rows(COHORT / "parents.jsonl")
    }
    original = read(FAMILY / "results/training-summary.json")
    output, all_predictions = [], []
    for run in original["runs"]:
        name = f"{run['variant']}-seed{run['seed']}"
        values = {"run": name, "selected_train": run["metrics"]["train"]}
        for state in ("initial", "last"):
            checkpoint = torch.load(ARTIFACTS / name / f"{state}.pt", weights_only=True)
            weights = {
                k: v.numpy().astype(np.float64)
                for k, v in checkpoint["state_dict"].items()
            }
            records = []
            for example in examples:
                logq, mass = forward(
                    parents[example["parent_sequence_sha256"]],
                    example,
                    weights,
                    run["variant"] != "exact-no-context",
                )
                record = {
                    "run": name,
                    "checkpoint": state,
                    **{
                        k: example[k]
                        for k in (
                            "example_id",
                            "split",
                            "condition",
                            "parent_sequence_sha256",
                            "component_id",
                        )
                    },
                    "peptide_ids": [c["peptide_row_id"] for c in example["candidates"]],
                    "logt": [
                        c["target_log_probability"] for c in example["candidates"]
                    ],
                    "logq": logq.tolist(),
                    "unidentified_selected_log_mass": mass,
                }
                records.append(record)
            values[state + "_train"] = independent_metrics(records)
            all_predictions.extend(records)
        output.append(values)
    result = {
        "scope": "Posthoc initial and final train-only inference; no new fits or test evaluations",
        "seconds": time.monotonic() - tick,
        "runs": output,
    }
    (FAMILY / "results/posthoc-train-diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with gzip.open(
        FAMILY / "results/posthoc-train-predictions.jsonl.gz", "wt"
    ) as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    shutil.copy2(__file__, FAMILY / "reproduce/source/train_fit_diagnostic.py")
    for r in output:
        print(
            r["run"],
            {
                s: round(r[s + "_train"]["reverse_kl"], 4)
                for s in ("initial", "selected", "last")
            },
        )


if __name__ == "__main__":
    main()
