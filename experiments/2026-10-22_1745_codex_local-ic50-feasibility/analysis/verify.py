#!/usr/bin/env python
"""Independent saved-prediction metrics and a fresh training-only checkpoint replay."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
import torch

from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training.modular import load_checkpoint
from presto.training.modular_evaluation import evaluate

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "experiments/2026-10-22_1745_codex_local-ic50-feasibility"
RAW = ROOT / "artifacts" / FAMILY.name
CORRECTIONS = []


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def exact_metrics(records):
    exact = [r for r in records if r["qualifier"] == "="]
    y = np.asarray([r["target"] for r in exact])
    p = np.asarray([r["prediction"] for r in exact])
    labels, calls = y <= math.log10(500), p <= math.log10(500)
    varied = np.ptp(p) != 0 and np.ptp(y) != 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return {
            "n": len(exact), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "pearson": float(pearsonr(y, p).statistic) if varied else None,
            "spearman": float(spearmanr(y, p).statistic) if varied else None,
            "prediction_sd": float(p.std()), "target_sd": float(y.std()),
            "binding_at_500_nM": {
                "accuracy": float(accuracy_score(labels, calls)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, calls)),
                "precision": float(precision_score(labels, calls, zero_division=0)),
                "recall": float(recall_score(labels, calls, zero_division=0)),
                "f1": float(f1_score(labels, calls, zero_division=0)),
                "auroc": float(roc_auc_score(labels, -p)),
                "auprc": float(average_precision_score(labels, -p)),
            },
        }


def compare(actual, expected, path):
    for key, value in actual.items():
        if isinstance(value, dict):
            compare(value, expected[key], path + "/" + key)
        elif value is None:
            if expected[key] is not None:
                # Explicitly preserve the discovered original metric bug. Only
                # constant-array Pearson's numerical artifact is corrected.
                assert key == "pearson" and abs(expected[key]) < 1e-12, (key, value, expected[key])
                CORRECTIONS.append({"metric": path + "/" + key, "original": expected[key],
                    "corrected": None, "reason": "undefined correlation: zero input range"})
        else:
            assert np.isclose(value, expected[key], rtol=1e-9, atol=1e-10), (key, value, expected[key])


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    execution = read(FAMILY / "results/execution.json")
    raw = Path(execution["raw_output"])
    for name, receipt in execution["artifacts"].items():
        assert file_sha256(raw / name) == receipt["sha256"]
    index = ObservationIndex(FAMILY / "reproduce/local-manifest.json")
    source = {r.observation_id: r for r in index}
    train_values = np.asarray([math.log10(r.value) for r in source.values() if r.split == "train"])
    mean, scale = train_values.mean(), train_values.std()
    saved_scale = read(FAMILY / "results/resolved-config.json")["scales"]["ic50"]
    assert np.isclose(mean, saved_scale["mean"], rtol=1e-12)
    assert np.isclose(scale, saved_scale["scale"], rtol=1e-12)
    summary = read(FAMILY / "results/summary.json")
    baseline = read(FAMILY / "results/baseline-metrics.json")
    checked = 0
    report = {}
    for method in ("network", "mean", "ridge"):
        report[method] = {}
        for split in ("train", "validation", "test"):
            prefix = "" if method == "network" else f"{method}-"
            predictions = rows(FAMILY / f"results/{prefix}{split}-predictions.jsonl.gz")
            ids = [r["observation_id"] for r in predictions]
            assert len(ids) == len(set(ids))
            assert set(ids) == {k for k, v in source.items() if v.split == split}
            losses = []
            for record in predictions:
                original = source[record["observation_id"]]
                assert record["qualifier"] == original.qualifier
                assert record["peptide"] == original.peptide
                assert record["group_id"] == original.group_id
                assert record["raw_value"] == original.value
                assert record["mhc_classes"] == [m.mhc_class for m in original.mhc]
                assert math.isclose(record["target"], math.log10(original.value), abs_tol=1e-12)
                residual = record["prediction"] - record["target"]
                if original.qualifier in ("<", "<="):
                    residual = max(0, residual)
                elif original.qualifier in (">", ">="):
                    residual = min(0, residual)
                loss = (residual / scale) ** 2
                assert math.isclose(record["loss"], loss, rel_tol=1e-4, abs_tol=2e-6)
                losses.append(loss)
            metrics = exact_metrics(predictions)
            saved = summary["metrics"][split]["ic50"] if method == "network" else baseline[method][split]["ic50"]
            compare(metrics, saved["exact"], f"{method}/{split}/exact")
            assert math.isclose(np.mean(losses), saved["loss"], rel_tol=1e-6, abs_tol=1e-7)
            report[method][split] = {"n": len(predictions), "loss": float(np.mean(losses)),
                                    "exact": metrics, "by_class": {}}
            for mhc_class in ("I", "II"):
                subset = [r for r in predictions if r["mhc_classes"] == [mhc_class]]
                class_metrics = exact_metrics(subset)
                compare(class_metrics, saved["by_class"][mhc_class], f"{method}/{split}/{mhc_class}")
                report[method][split]["by_class"][mhc_class] = class_metrics
            checked += len(predictions)
    # Independently refit the fixed ridge through augmented least squares.
    coefficients = read(FAMILY / "results/baseline-coefficients.json")
    aa = "ACDEFGHIKLMNPQRSTVWY"
    training_rows = [r for r in source.values() if r.split == "train"]
    features = np.asarray([[len(r.peptide), *[r.peptide.count(a)/len(r.peptide) for a in aa],
        int(r.mhc[0].mhc_class == "I"), int(r.mhc[0].mhc_class == "II")] for r in training_rows])
    assert np.allclose(features.mean(0), coefficients["feature_mean"])
    feature_sd = features.std(0)
    feature_sd[feature_sd == 0] = 1
    assert np.allclose(feature_sd, coefficients["feature_sd"])
    z = np.column_stack([np.ones(len(features)), (features-features.mean(0))/feature_sd])
    penalty = np.eye(z.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.lstsq(np.vstack([z, penalty]), np.r_[train_values, np.zeros(z.shape[1])], rcond=None)[0]
    assert np.allclose(beta, coefficients["ridge"], rtol=1e-7, atol=1e-8)
    # Reload once in this fresh process; evaluate only the first saved TRAIN batch.
    selected, scales, _ = load_checkpoint(raw / "selected.pt", device="cpu")
    _, replay = evaluate(selected, index, index.partitions["train:assay"][:64], scales, batch_size=64)
    original = {r["observation_id"]: r for r in rows(FAMILY / "results/train-predictions.jsonl.gz")}
    replay_error = max(abs(r["prediction"] - original[r["observation_id"]]["prediction"]) for r in replay)
    assert replay_error <= 1e-6, replay_error
    initial, _, _ = load_checkpoint(raw / "initial.pt", device="cpu")
    changed = {name: float((value.detach()-dict(initial.named_parameters())[name].detach()).norm())
               for name, value in selected.named_parameters()}
    assert all(x > 0 for x in changed.values())
    coverage = read(FAMILY / "results/coverage.json")["selected"]
    assert set(coverage) == set(changed)
    assert all(v["gradient_and_update_steps"] > 0 for v in coverage.values())
    assert summary["coverage_gate"]["status"] == "passed"
    assert summary["steps"] == 339 and summary["training_exposures"] == {"ic50": 21636}
    history = read(FAMILY / "results/history.json")
    assert len(history) == 1 and history[0]["complete_epoch"]
    result = {"status": "passed", "prediction_records_verified": checked,
        "split_sizes": {k: len(v) for k, v in index.partitions.items()},
        "metrics": report, "baseline_refit": "independent augmented least squares agrees",
        "fresh_process_training_replay_max_abs_error": replay_error,
        "selected_parameter_change_norms": changed, "all_allocated_tensors_updated": True,
        "original_metric_corrections": CORRECTIONS,
        "elapsed_seconds": time.monotonic() - started}
    (FAMILY / "results/independent-verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("metrics", "selected_parameter_change_norms")}, indent=2))


if __name__ == "__main__":
    main()
