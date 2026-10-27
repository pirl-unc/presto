"""Independent saved-prediction metrics; does not run a model or select a checkpoint."""

from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
sys.path.insert(0, str(FAMILY / "code"))
from worker import atomic, read, sha  # noqa: E402 - import the frozen transport helpers

PINS = read(FAMILY / "reproduce/source-pins.json")
CONDITIONS = PINS["conditions"]
RAW = ROOT / "artifacts" / FAMILY.name


def load(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def metrics(rows, scale):
    if not rows:
        return {"n_all": 0, "n": 0}
    y = np.array([r["target"] for r in rows])
    p = np.array([r["prediction"] for r in rows])
    assert np.isfinite(y).all() and np.isfinite(p).all()
    q = np.array([r["qualifier"] for r in rows])
    residual = p - y
    residual[np.isin(q, ["<", "<="])] = np.maximum(0, residual[np.isin(q, ["<", "<="])])
    residual[np.isin(q, [">", ">="])] = np.minimum(0, residual[np.isin(q, [">", ">="])])
    losses = (residual / scale) ** 2
    np.testing.assert_allclose(losses, [r["loss"] for r in rows], rtol=5e-5, atol=3e-6)
    result = dict(n_all=len(rows), loss=float(losses.mean()))
    censored = q != "="
    if censored.any():
        result["censor_satisfaction"] = float((residual[censored] == 0).mean())
    exact = q == "="
    y = y[exact]
    p = p[exact]
    result["n"] = len(y)
    if not len(y):
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = (
            float(spearmanr(y, p).statistic)
            if len(y) > 1 and np.ptp(p) > 0 and np.ptp(y) > 0
            else None
        )
        pearson = (
            float(pearsonr(y, p).statistic)
            if len(y) > 1 and np.ptp(p) > 0 and np.ptp(y) > 0
            else None
        )
    result.update(
        spearman=rho,
        pearson=pearson,
        rmse=float(np.sqrt(np.mean((y - p) ** 2))),
        prediction_sd=float(np.std(p)),
        target_sd=float(np.std(y)),
    )
    labels = y <= math.log10(500)
    predicted = p <= math.log10(500)
    binding = dict(
        accuracy=float(accuracy_score(labels, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted))
        if len(set(labels)) == 2
        else None,
        precision=float(precision_score(labels, predicted, zero_division=0)),
        recall=float(recall_score(labels, predicted, zero_division=0)) if labels.any() else None,
        f1=float(f1_score(labels, predicted, zero_division=0)),
        auroc=float(roc_auc_score(labels, -p)) if len(set(labels)) == 2 else None,
        auprc=float(average_precision_score(labels, -p)) if labels.any() else None,
    )
    result["binding_at_500_nM"] = binding
    return result


def assert_metric(actual, reported):
    for key in ("n", "spearman", "pearson", "rmse"):
        if actual.get(key) is None:
            assert reported.get(key) is None, (key, actual, reported)
        else:
            np.testing.assert_allclose(actual[key], reported[key], rtol=2e-7, atol=1e-9)
    for key, value in actual.get("binding_at_500_nM", {}).items():
        other = reported["binding_at_500_nM"][key]
        if value is None:
            assert other is None
        else:
            np.testing.assert_allclose(value, other, rtol=2e-7, atol=1e-9)


def source_rows(split):
    path = RAW / "upload/inputs" / f"joint-{split}.jsonl"
    assert sha(path) == PINS["joint_input_hashes"][split]
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def core_set(classes):
    result = set()
    for row in source_rows("train"):
        if row["mhc"][0]["mhc_class"] in classes:
            p = row["peptide"]
            result.update(p[i : i + 9] for i in range(len(p) - 8))
    return result


def subgroup_metrics(rows, scale, cores):
    known = []
    novel = []
    alleles = {}
    lengths = {}
    for row in rows:
        p = row["peptide"]
        (known if any(p[i : i + 9] in cores for i in range(len(p) - 8)) else novel).append(row)
        mhc = row["observation"]["mhc"][0]
        allele = hashlib.sha256((mhc["groove1"] + "|" + mhc["groove2"]).encode()).hexdigest()
        alleles.setdefault(allele, []).append(row)
        lengths.setdefault(str(len(p)), []).append(row)
    allele_metrics = {k: metrics(v, scale) for k, v in alleles.items()}
    valid = [m["spearman"] for m in allele_metrics.values() if m.get("spearman") is not None]
    return dict(
        seen_nine_mer=metrics(known, scale),
        no_seen_nine_mer=metrics(novel, scale),
        by_mhc_pair=allele_metrics,
        macro_mhc_spearman=float(np.mean(valid)) if valid else None,
        mhc_pairs_with_defined_spearman=len(valid),
        by_length={k: metrics(v, scale) for k, v in lengths.items()},
    )


def verify(condition):
    spec = CONDITIONS[condition]
    directory = FAMILY / "results" / condition
    report = read(directory / "report.json")
    if report is None:
        raise ValueError("condition has no completed selected-checkpoint report")
    assert sha(report["checkpoint"]) == report["checkpoint_sha256"]
    resolved = read(directory / "resolved-config.json")
    scale = resolved["scales"]["ic50"]["scale"]
    cores = core_set(spec["classes"])
    train = [r for r in source_rows("train") if r["mhc"][0]["mhc_class"] in spec["classes"]]
    train_targets = np.log10([r["value"] for r in train])
    np.testing.assert_allclose(train_targets.mean(), resolved["scales"]["ic50"]["mean"], atol=1e-12)
    output = {}
    table = []
    baseline_results = {}
    originals = {
        split: {r["observation_id"]: r for r in source_rows(split)}
        for split in ("validation", "test")
    }

    paths = [
        (
            "in_class",
            split,
            directory / f"{split}-predictions.jsonl.gz",
            report["summary"]["metrics"][split],
        )
        for split in ("validation", "test")
    ]
    for protocol, transfer in report["transfer"].items():
        for split in ("validation", "test"):
            paths.append(
                (
                    "zero_shot",
                    split,
                    directory / f"{protocol}-{split}-predictions.jsonl.gz",
                    transfer["metrics"][split],
                )
            )
    for scope, split, path, reported in paths:
        records = load(path)
        assert len({r["observation_id"] for r in records}) == len(records)
        assert all(r["split"] == split for r in records)
        classes = (
            spec["classes"]
            if scope == "in_class"
            else [c for c in ("I", "II") if c not in spec["classes"]]
        )
        expected = {
            k: r for k, r in originals[split].items() if r["mhc"][0]["mhc_class"] in classes
        }
        assert set(expected) == {r["observation_id"] for r in records}
        for row in records:
            original = expected[row["observation_id"]]
            assert row["peptide"] == original["peptide"]
            assert row["qualifier"] == original["qualifier"]
            assert row["observation"]["mhc"] == original["mhc"]
            np.testing.assert_allclose(row["target"], math.log10(original["value"]), atol=1e-12)
        overall = metrics(records, scale)
        np.testing.assert_allclose(overall["loss"], reported["ic50"]["loss"], rtol=5e-5, atol=3e-6)
        assert_metric(overall, reported["ic50"]["exact"])
        baseline_path = directory / (
            f"mean-{split}-predictions.jsonl.gz"
            if scope == "in_class"
            else f"mean-transfer-{split}-predictions.jsonl.gz"
        )
        baseline = load(baseline_path)
        assert [r["observation_id"] for r in baseline] == [r["observation_id"] for r in records]
        np.testing.assert_allclose(
            [r["prediction"] for r in baseline], train_targets.mean(), atol=1e-12
        )
        key = split if scope == "in_class" else split + ":transfer"
        actual_baseline = metrics(baseline, scale)
        assert_metric(actual_baseline, report["mean_baselines"][key]["ic50"]["exact"])
        baseline_results[key] = actual_baseline

        for kind in ("I", "II"):
            rows = [r for r in records if r["mhc_classes"] == [kind]]
            if not rows:
                continue
            actual = metrics(rows, scale)
            assert_metric(actual, reported["ic50"]["by_class"][kind])
            output[f"{scope}:{split}:{kind}"] = dict(
                metrics=actual,
                subgroups=subgroup_metrics(rows, scale, cores),
                predictions=str(path),
                prediction_sha256=sha(path),
            )
            table.append(
                dict(
                    condition=condition,
                    seed=spec["seed"],
                    training_classes="+".join(spec["classes"]),
                    evaluation_class=kind,
                    scope=scope,
                    split=split,
                    **{k: actual[k] for k in ("n_all", "n", "loss", "spearman", "pearson", "rmse")},
                    **actual["binding_at_500_nM"],
                )
            )
    result = dict(
        status="passed",
        condition=condition,
        scaler=resolved["scales"]["ic50"],
        checkpoint_sha256=report["checkpoint_sha256"],
        metrics=output,
        table=table,
        mean_baselines=baseline_results,
    )
    atomic(FAMILY / "results" / f"{condition}-independent-verification.json", result)
    return result


def main():
    names = sys.argv[1:] or list(CONDITIONS)
    rows = []
    for name in names:
        rows.extend(verify(name)["table"])
        print(f"{name}: independently verified", flush=True)
    with (FAMILY / "results" / "metrics-independent.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
