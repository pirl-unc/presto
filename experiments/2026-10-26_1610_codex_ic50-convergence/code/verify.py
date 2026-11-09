"""Independent source joins, metric replay, peptide bootstrap and train-only replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

FAMILY = Path(__file__).resolve().parents[1]
REPRO = FAMILY / "reproduce"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def binding(labels, predictions):
    labels = np.asarray(labels, bool)
    scores = np.asarray(predictions, float)
    calls = scores <= math.log10(500)
    both = len(set(labels)) == 2
    return dict(
        n=len(labels),
        accuracy=float(accuracy_score(labels, calls)),
        balanced_accuracy=float(balanced_accuracy_score(labels, calls)) if both else None,
        precision=float(precision_score(labels, calls, zero_division=0)),
        recall=float(recall_score(labels, calls, zero_division=0)) if labels.any() else None,
        f1=float(f1_score(labels, calls, zero_division=0)),
        auroc=float(roc_auc_score(labels, -scores)) if both else None,
        auprc=float(average_precision_score(labels, -scores)) if labels.any() else None,
    )


def exact_metrics(rows):
    y = np.asarray([r["target"] for r in rows])
    p = np.asarray([r["prediction"] for r in rows])
    varied = len(rows) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0
    return dict(
        n=len(rows),
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        pearson=float(pearsonr(y, p).statistic) if varied else None,
        spearman=float(spearmanr(y, p).statistic) if varied else None,
        prediction_sd=float(p.std()),
        target_sd=float(y.std()),
        sd_ratio=float(p.std() / y.std()) if y.std() else None,
        binding_at_500_nM=binding(y <= math.log10(500), p),
    )


def bound_class(row):
    """Strict and inclusive source inequalities remain distinct at the threshold."""
    q, y, cutoff = row["qualifier"], row["target"], math.log10(500)
    if q == "=":
        return y <= cutoff
    if q in ("<", "<=") and y <= cutoff:
        return True
    if q == ">" and y >= cutoff:
        return False
    if q == ">=" and y > cutoff:
        return False
    return None


def metrics(rows):
    exact = [r for r in rows if r["qualifier"] == "="]
    bounds = [r for r in rows if r["qualifier"] != "="]
    established = [(bound_class(r), r["prediction"]) for r in rows if bound_class(r) is not None]
    result = dict(
        n=len(rows),
        loss=float(np.mean([r["loss"] for r in rows])),
        exact=exact_metrics(exact) if exact else None,
        by_class={},
        binding_with_established_bounds=binding(*zip(*established)) if established else None,
        ambiguous_binding_bounds=len(rows) - len(established),
    )
    for c in ("I", "II"):
        subset = [r for r in exact if r["mhc_classes"] == [c]]
        if subset:
            result["by_class"][c] = exact_metrics(subset)
    if bounds:
        result["censored"] = dict(
            n=len(bounds),
            mean_loss=float(np.mean([r["loss"] for r in bounds])),
            satisfaction=sum(
                r["prediction"] <= r["target"]
                if r["qualifier"] in ("<", "<=")
                else r["prediction"] >= r["target"]
                for r in bounds
            )
            / len(bounds),
        )
    return result


def compare(actual, expected):
    """Compare independently computed fields that are present in the canonical summary."""
    for key, value in actual.items():
        if key not in expected:
            continue
        if isinstance(value, dict):
            compare(value, expected[key])
        elif value is None:
            assert expected[key] is None, (key, value, expected[key])
        else:
            assert math.isclose(value, expected[key], rel_tol=1e-8, abs_tol=1e-9), (
                key,
                value,
                expected[key],
            )


def source_rows(name):
    launch = read(REPRO / "launch.json")
    pin = launch["manifests"][name]
    assert sha(pin["path"]) == pin["sha256"]
    manifest = read(pin["path"])
    source = {}
    for item in manifest["sources"]:
        assert sha(item["path"]) == item["sha256"]
        for row in records(item["path"]):
            assert row["observation_id"] not in source
            source[row["observation_id"]] = row
    counts = Counter(r["split"] for r in source.values())
    assert counts == {"train": 21636, "validation": 2925, "test": 2803}, counts
    if name == "permuted42":
        assignments = read(REPRO / "permutation-assignments.json")
        for c in ("I", "II"):
            a = [r for r in assignments if r["class"] == c]
            assert Counter(r["value"] for r in a) == Counter(r["original_value"] for r in a)
            assert {r["observation_id"] for r in a} == {r["donor_observation_id"] for r in a}
    return source


def verify_rows(saved, source, split):
    expected = {key: row for key, row in source.items() if row["split"] == split}
    assert len(saved) == len(expected)
    assert {r["observation_id"] for r in saved} == set(expected)
    train = np.asarray([math.log10(r["value"]) for r in source.values() if r["split"] == "train"])
    scale = train.std()
    for row in saved:
        original = expected[row["observation_id"]]
        assert original["unit"] == "nM" and original["value_encoding"] == "linear"
        observed = row["observation"]
        # ObservationIndex adds its pinned evidence policy without modifying source fields.
        assert {k: v for k, v in observed.items() if k != "provenance"} == {
            k: v for k, v in original.items() if k != "provenance"
        }
        assert all(observed["provenance"][k] == v for k, v in original["provenance"].items())
        assert row["qualifier"] == original["qualifier"]
        assert row["peptide"] == original["peptide"]
        assert row["group_id"] == original["group_id"]
        assert row["raw_value"] == original["value"]
        assert row["mhc_classes"] == [m["mhc_class"] for m in original["mhc"]]
        assert math.isclose(row["target"], math.log10(original["value"]), abs_tol=1e-12)
        residual = row["prediction"] - row["target"]
        if row["qualifier"] in ("<", "<="):
            residual = max(0, residual)
        elif row["qualifier"] in (">", ">="):
            residual = min(0, residual)
        assert math.isclose(row["loss"], (residual / scale) ** 2, rel_tol=1e-4, abs_tol=2e-6)


def ridge_refit(source):
    folder = FAMILY / "results/baselines"
    coef = read(folder / "coefficients.json")
    rows = [r for r in source.values() if r["split"] == "train"]
    aa = "ACDEFGHIKLMNPQRSTVWY"
    x = np.asarray(
        [
            [
                len(r["peptide"]),
                *[r["peptide"].count(a) / len(r["peptide"]) for a in aa],
                int(r["mhc"][0]["mhc_class"] == "I"),
                int(r["mhc"][0]["mhc_class"] == "II"),
            ]
            for r in rows
        ]
    )
    y = np.log10([r["value"] for r in rows])
    sd = x.std(0)
    sd[sd == 0] = 1
    assert np.allclose(x.mean(0), coef["feature_mean"])
    assert np.allclose(sd, coef["feature_sd"])
    z = np.column_stack([np.ones(len(x)), (x - x.mean(0)) / sd])
    penalty = np.eye(z.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.lstsq(np.vstack([z, penalty]), np.r_[y, np.zeros(z.shape[1])], rcond=None)[0]
    assert np.allclose(beta, coef["ridge"], rtol=1e-7, atol=1e-8)
    assert math.isclose(float(y.mean()), coef["mean"], abs_tol=1e-12)
    return {
        "status": "passed",
        "maximum_coefficient_difference": float(np.max(np.abs(beta - coef["ridge"]))),
    }


def bootstrap(rows, baseline):
    """Resample exact peptides, retaining all observations within each sampled peptide."""
    base = {r["observation_id"]: r for r in baseline}
    groups = defaultdict(lambda: np.zeros(3))
    for r in rows:
        if r["qualifier"] != "=":
            continue
        b = base[r["observation_id"]]
        assert b["target"] == r["target"]
        groups[r["peptide"]] += np.array(
            [(r["prediction"] - r["target"]) ** 2, (b["prediction"] - b["target"]) ** 2, 1]
        )
    blocks = np.asarray([groups[key] for key in sorted(groups)])
    rng = np.random.default_rng(20261021)
    draws = []
    for _ in range(2000):
        s = blocks[rng.integers(0, len(blocks), size=len(blocks))].sum(0)
        draws.append(float(np.sqrt(s[0] / s[2]) - np.sqrt(s[1] / s[2])))
    interval = np.quantile(draws, [0.025, 0.975]).tolist()
    return dict(
        peptide_groups=len(blocks),
        rmse_difference_95_interval=interval,
        upper_below_zero=interval[1] < 0,
        draws=draws,
    )


def replay(name, saved_train):
    from presto.data.modular_sources import ObservationIndex
    from presto.training.modular import load_checkpoint
    from presto.training.modular_evaluation import evaluate

    launch = read(REPRO / "launch.json")
    root = Path(launch["raw_artifacts"]) / name
    for path, pin in read(FAMILY / "results" / name / "artifacts.json").items():
        assert sha(root / path) == pin["sha256"]
    model, scales, _ = load_checkpoint(root / "selected.pt", device="cpu")
    index = ObservationIndex(launch["manifests"][name]["path"])
    _, rows = evaluate(model, index, index.partitions["train:assay"][:64], scales, batch_size=64)
    expected = {r["observation_id"]: r for r in saved_train}
    error = max(abs(r["prediction"] - expected[r["observation_id"]]["prediction"]) for r in rows)
    assert error <= 1e-6, error
    initial, _, _ = load_checkpoint(root / "initial.pt", device="cpu")
    initial_params = dict(initial.named_parameters())
    movement = {
        n: float((p.detach() - initial_params[n].detach()).norm())
        for n, p in model.named_parameters()
    }
    assert all(x > 0 for x in movement.values())
    coverage = read(root / "coverage.json")["selected"]
    assert set(coverage) == set(movement)
    assert all(x["gradient_and_update_steps"] > 0 for x in coverage.values())
    return dict(maximum_prediction_error=error, parameter_change_norms=movement)


def verify(args):
    started = time.monotonic()
    source = source_rows("seed42" if args.condition == "baselines" else args.condition)
    report = {
        "condition": args.condition,
        "status": "passed",
        "metrics": {},
        "prediction_records_verified": 0,
    }
    if args.condition == "baselines":
        report["ridge_refit"] = ridge_refit(source)
        canonical = read(FAMILY / "results/baselines/metrics.json")
        for method in ("mean", "ridge"):
            report["metrics"][method] = {}
            for split in ("train", "validation", "test"):
                rows = records(FAMILY / f"results/baselines/{method}-{split}-predictions.jsonl.gz")
                verify_rows(rows, source, split)
                result = metrics(rows)
                compare(result, canonical[method][split]["ic50"])
                report["metrics"][method][split] = result
                report["prediction_records_verified"] += len(rows)
    else:
        folder = FAMILY / "results" / args.condition
        canonical = read(folder / "summary.json")
        splits = {}
        for split in ("train", "validation", "test"):
            rows = records(folder / f"{split}-predictions.jsonl.gz")
            splits[split] = rows
            verify_rows(rows, source, split)
            result = metrics(rows)
            compare(result, canonical["metrics"][split]["ic50"])
            report["metrics"][split] = result
            report["prediction_records_verified"] += len(rows)
        report["replay"] = replay(args.condition, splits["train"])
        assert canonical["coverage_gate"]["status"] == "passed"
        history = read(folder / "history.json")
        assert all(r["complete_epoch"] for r in history)
        assert canonical["steps"] == 339 * len(history)
        assert canonical["training_exposures"]["ic50"] == 21636 * len(history)
        report["selected_epoch"] = canonical["selected_epoch"]
        report["completed_epochs"] = len(history)
        report["comparisons"] = {}
        for method in ("mean", "ridge"):
            baseline = records(
                FAMILY / f"results/baselines/{method}-validation-predictions.jsonl.gz"
            )
            for c in ("overall", "I", "II"):
                subset = [
                    r for r in splits["validation"] if c == "overall" or r["mhc_classes"] == [c]
                ]
                report["comparisons"][f"{c}_vs_{method}"] = bootstrap(subset, baseline)
        base_train = read(FAMILY / "results/baselines/metrics.json")["mean"]["train"]["ic50"][
            "exact"
        ]["rmse"]
        validation = report["metrics"]["validation"]
        report["scientific_screen_passed"] = (
            report["metrics"]["train"]["exact"]["rmse"] < base_train
            and all(x["upper_below_zero"] for x in report["comparisons"].values())
            and all(
                (x["spearman"] or 0) > 0
                for x in [validation["exact"], *validation["by_class"].values()]
            )
        )
        report["contraction_strata"] = [
            c
            for c, x in {"overall": validation["exact"], **validation["by_class"]}.items()
            if x["sd_ratio"] is not None and x["sd_ratio"] < 0.1
        ]
    report["elapsed_seconds"] = time.monotonic() - started
    path = FAMILY / "results" / f"{args.condition}-independent-verification.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("metrics", "comparisons", "replay")},
            indent=2,
        )
    )


if __name__ == "__main__":
    from launch import guarded

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "condition", choices=("baselines", "seed42", "seed43", "seed44", "permuted42")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with guarded("verify-" + args.condition):
        verify(args)
