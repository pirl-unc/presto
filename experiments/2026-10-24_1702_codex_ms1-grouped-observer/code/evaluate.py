"""Once-only evaluation of sealed observer predictions on reserved pool outcomes."""

import csv
import gzip
import hashlib
import json
import resource
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from model import group_data, metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data/measured_ms1_grouped_observer/2026-10-24_1702_canonical"
FIT = ROOT / "results/fit"
OUTPUT = ROOT / "results/evaluation"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    started = time.perf_counter()
    assert not OUTPUT.exists()
    seal_path = ROOT / "results/pre-test-seal.json"
    seal = json.loads(seal_path.read_text())
    assert not seal["test_outcomes_loaded"]
    for name, expected in seal["files"].items():
        assert digest(FIT / name) == expected
    for name, expected in seal["code"].items():
        assert digest(ROOT / "code" / name) == expected
    for name, expected in seal["input_digests"].items():
        assert digest(DATA / name) == expected["sha256"]
    assert (
        digest(DATA / "test-targets.jsonl")
        == seal["expected_test_target_digest"]["sha256"]
    )
    OUTPUT.mkdir()
    shutil.copy2(__file__, ROOT / "reproduce/source/evaluate.py")
    write(
        OUTPUT / "outcome-access.json",
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "pre_test_seal_sha256": digest(seal_path),
            "sealed_at_utc": seal["sealed_at_utc"],
            "all_sealed_files_verified": True,
        },
    )
    validation = [
        json.loads(line)
        for line in (DATA / "validation.jsonl").read_text().splitlines()
    ]
    test = [
        json.loads(line)
        for line in (DATA / "test-covariates.jsonl").read_text().splitlines()
    ]
    outcomes = [
        json.loads(line)
        for line in (DATA / "test-targets.jsonl").read_text().splitlines()
    ]
    assert [r["measurement_id"] for r in test] == [
        r["measurement_id"] for r in outcomes
    ]
    for row, outcome in zip(test, outcomes, strict=True):
        assert "target" not in row
        row["target"] = outcome["target"]
    chosen = json.loads((FIT / "validation-selection.json").read_text())
    overall, strata, summary = {}, [], []
    for split, rows in (("validation", validation), ("test", test)):
        archive = np.load(FIT / f"{split}-predictions.npz", allow_pickle=False)
        assert list(archive["measurement_ids"]) == [r["measurement_id"] for r in rows]
        _, _, _, weights = group_data(rows)
        target = np.array([r["target"] for r in rows])
        overall[split] = {}
        with gzip.open(OUTPUT / f"{split}-predictions.jsonl.gz", "wt") as stream:
            for name in sorted(set(archive.files) - {"measurement_ids"}):
                predicted = archive[name]
                assert predicted.shape == target.shape
                values = metrics(target, predicted, weights)
                overall[split][name] = values
                summary.append({"split": split, "model": name, **values})
                for field in ("pool", "library"):
                    for key in sorted({r[field] for r in rows}):
                        mask = np.array([r[field] == key for r in rows])
                        strata.append(
                            {
                                "split": split,
                                "model": name,
                                "stratum_type": field,
                                "stratum": key,
                                **metrics(target[mask], predicted[mask], weights[mask]),
                            }
                        )
                for row, prediction, weight in zip(
                    rows, predicted, weights, strict=True
                ):
                    stream.write(
                        json.dumps(
                            {
                                **row,
                                "split": split,
                                "model": name,
                                "prediction": float(prediction),
                                "weight": float(weight),
                            },
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
    length, sequence = chosen["primary_length"], chosen["primary_sequence"]

    def screen(model):
        gains = {
            s: 1 - overall[s][model]["rmse"] / overall[s][length]["rmse"]
            for s in overall
        }
        positive = [
            r
            for r in strata
            if r["split"] == "test"
            and r["model"] == model
            and r["stratum_type"] == "pool"
            and r["pearson"] is not None
            and r["spearman"] is not None
            and r["pearson"] > 0
            and r["spearman"] > 0
        ]
        ratios = {
            s: overall[s][model]["prediction_sd"] / overall[s][model]["target_sd"]
            for s in overall
        }
        gates = {
            "validation_gain_at_least_5_percent": gains["validation"] >= 0.05,
            "test_gain_at_least_5_percent": gains["test"] >= 0.05,
            "positive_correlations_in_three_test_pools": len(positive) >= 3,
            "prediction_sd_at_least_tenth_target_both_splits": min(ratios.values())
            >= 0.1,
        }
        return {
            "model": model,
            "length_comparator": length,
            "rmse_reductions": gains,
            "positive_test_pools": len(positive),
            "prediction_target_sd_ratios": ratios,
            "gates": gates,
            "passed": all(gates.values()),
        }

    result = {
        "primary": screen(sequence),
        "secondary": [screen(f"nonlinear_{s}") for s in (42, 43, 44, "ensemble")],
        "overall": overall,
        "selection": {
            k: chosen[k] for k in ("primary_length", "primary_sequence", "nonlinear")
        },
        "metric_contract": "Equal-pool weighted errors/R2/SD; unweighted Pearson/Spearman; centered log10 measured signal",
        "binding_metrics": "Not applicable: no affinity labels",
        "test_selected_models": False,
    }
    write(OUTPUT / "summary.json", result)
    write(OUTPUT / "strata.json", strata)
    with (OUTPUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    usage = {
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "argv": sys.argv,
        "python": sys.executable,
    }
    assert usage["peak_rss_gib"] <= 3
    write(OUTPUT / "runtime.json", usage)
    print(
        json.dumps(
            {
                "primary": result["primary"],
                "secondary": result["secondary"],
                "runtime": usage,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
