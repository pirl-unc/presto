"""Replay every fold/variance estimate using scalar statistics and run values."""

import gzip
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

FAMILY = Path(__file__).resolve().parents[1]
COHORT = (
    FAMILY.parents[1]
    / "data/availability_recipes/2026-10-23-mapp-complete-run-composition-v1"
)


def read(path):
    return json.loads(Path(path).read_text())


def correlation(a, b, rank=False):
    a, b = np.asarray(a), np.asarray(b)
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    if rank:
        a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / math.sqrt(float(a @ a) * float(b @ b)))


def close(a, b):
    if a is None:
        assert b is None
    else:
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=2e-12)


def main():
    tick = time.monotonic()
    with (COHORT / "run_observations.jsonl").open() as handle:
        observations = [json.loads(line) for line in handle]
    with (COHORT / "examples.jsonl").open() as handle:
        examples = {
            r["parent_sequence_sha256"]: r
            for r in map(json.loads, handle)
            if r["condition"] == "UT"
        }
    values = {
        (r["peptide_row_id"], r["channel"]["channel_id"]): math.log10(
            r["measurement"]["value"]
        )
        for r in observations
        if r["admitted_complete_parent_set"]
    }
    channel_condition = {
        r["channel"]["channel_id"]: r["condition"] for r in observations
    }
    centered = {}
    for parent, example in examples.items():
        for channel in channel_condition:
            v = [values[c["peptide_row_id"], channel] for c in example["candidates"]]
            centered[parent, channel] = [x - statistics.mean(v) for x in v]
    with gzip.open(FAMILY / "results/run-predictions.jsonl.gz", "rt") as handle:
        predictions = [json.loads(line) for line in handle]
    assert len(predictions) == len(examples) * 15
    keys = set()
    maximum = 0.0
    for row in predictions:
        parent = row["parent_sequence_sha256"]
        assert row["peptide_ids"] == [
            c["peptide_row_id"] for c in examples[parent]["candidates"]
        ]
        assert row["split"] == examples[parent]["split"]
        train, heldout = row["training_channels"], row["heldout_channels"]
        assert not set(train) & set(heldout)
        key = parent, row["task"], tuple(heldout)
        assert key not in keys
        keys.add(key)
        if row["task"] == "condition_change":
            assert len(train) == 4 and len(heldout) == 2
            assert [channel_condition[c] for c in heldout] == ["UT", "TNF+IFN"]
            assert set(train) | set(heldout) == set(channel_condition)
            pred = [
                statistics.mean(
                    centered[parent, c][i]
                    for c in train
                    if channel_condition[c] == "TNF+IFN"
                )
                - statistics.mean(
                    centered[parent, c][i]
                    for c in train
                    if channel_condition[c] == "UT"
                )
                for i in range(len(row["peptide_ids"]))
            ]
            target = [
                b - a
                for a, b in zip(
                    centered[parent, heldout[0]], centered[parent, heldout[1]]
                )
            ]
        else:
            assert len(train) == 2 and len(heldout) == 1
            assert {channel_condition[c] for c in train + heldout} == {row["condition"]}
            pred = [
                statistics.mean(centered[parent, c][i] for c in train)
                for i in range(len(row["peptide_ids"]))
            ]
            target = centered[parent, heldout[0]]
        maximum = max(maximum, max(abs(a - b) for a, b in zip(pred, row["prediction"])))
        close(pred, row["prediction"])
        close(target, row["target"])
    effects = read(FAMILY / "results/parent-effects.json")
    assert len(effects) == len(examples)
    for row in effects:
        parent = row["parent_sequence_sha256"]
        delta, noise = [], []
        for i in range(len(row["peptide_ids"])):
            a = [
                centered[parent, c][i]
                for c in channel_condition
                if channel_condition[c] == "UT"
            ]
            b = [
                centered[parent, c][i]
                for c in channel_condition
                if channel_condition[c] == "TNF+IFN"
            ]
            delta.append(statistics.mean(b) - statistics.mean(a))
            noise.append((statistics.variance(a) + statistics.variance(b)) / 3)
        close(delta, row["mean_log10_change"])
        close(noise, row["expected_contrast_sampling_variance"])
        close(
            statistics.mean(d * d for d in delta), row["observed_mean_squared_change"]
        )
        close(statistics.mean(noise), row["mean_contrast_sampling_variance"])
        close(
            statistics.mean(d * d - n for d, n in zip(delta, noise)),
            row["variance_adjusted_mean_squared_change"],
        )
    summary = read(FAMILY / "results/repeatability-summary.json")
    for scope, record in summary["by_scope"].items():
        selected = [r for r in effects if scope == "all" or r["split"] == scope]
        observed = statistics.mean(r["observed_mean_squared_change"] for r in selected)
        noise = statistics.mean(r["mean_contrast_sampling_variance"] for r in selected)
        close(observed, record["observed_mean_squared_change"])
        close(noise, record["mean_contrast_sampling_variance"])
        close(observed - noise, record["variance_adjusted_mean_squared_change"])
        close(math.sqrt(observed), record["observed_change_rms"])
        close(math.sqrt(noise), record["expected_sampling_noise_rms"])
        close(
            sum(r["variance_adjusted_mean_squared_change"] > 0 for r in selected)
            / len(selected),
            record["fraction_parents_positive_variance_adjusted_change"],
        )
    for key, record in summary["by_task"].items():
        scope, task = key.split(":")
        selected = [
            r
            for r in predictions
            if r["task"] == task and (scope == "all" or r["split"] == scope)
        ]
        for name, metrics in record.items():
            target = [v for r in selected for v in r["target"]]
            pred = [
                v if name == "other_run_mean" else 0
                for r in selected
                for v in r["prediction"]
            ]
            errors, offset = [], 0
            for row in selected:
                n = len(row["peptide_ids"])
                errors.append(
                    statistics.mean(
                        (p - t) ** 2
                        for p, t in zip(
                            pred[offset : offset + n], target[offset : offset + n]
                        )
                    )
                )
                offset += n
            assert metrics["n_records"] == len(selected) and metrics[
                "n_peptide_predictions"
            ] == len(target)
            assert metrics["n_parents"] == len(
                {r["parent_sequence_sha256"] for r in selected}
            )
            for k, v in {
                "mse": statistics.mean(errors),
                "rmse": math.sqrt(statistics.mean(errors)),
                "pearson": correlation(pred, target),
                "spearman": correlation(pred, target, True),
                "prediction_sd": statistics.pstdev(pred),
                "target_sd": statistics.pstdev(target),
            }.items():
                close(v, metrics[k])
    output = {
        "passed": True,
        "n_parent_variance_records": len(effects),
        "n_fold_prediction_records": len(predictions),
        "n_peptide_predictions": sum(len(r["prediction"]) for r in predictions),
        "max_scalar_prediction_difference": maximum,
        "seconds": time.monotonic() - tick,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (FAMILY / "results/independent-analysis.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output))


if __name__ == "__main__":
    main()
