"""Independent physical targets, loss, regression, binding and support verification."""

from pathlib import Path
from collections import defaultdict
import csv, gzip, hashlib, json, math, warnings
import numpy as np
import torch
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

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
UPLOAD = ROOT / "artifacts" / F.name / "upload"
PINS = json.loads((F / "reproduce/source-pins.json").read_text())


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def load(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f]


def metrics(rows, scale, binding=True):
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
    if not binding:
        return result
    labels = y <= math.log10(500)
    predicted = p <= math.log10(500)
    binding = dict(
        accuracy=float(accuracy_score(labels, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted))
        if len(set(labels)) == 2
        else None,
        precision=float(precision_score(labels, predicted, zero_division=0)),
        recall=float(recall_score(labels, predicted, zero_division=0))
        if labels.any()
        else None,
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


def physical(row):
    factors = {"pM": 0.001, "nM": 1.0, "uM": 1e3, "mM": 1e6, "M": 1e9, "min": 1.0}
    return math.log10(row["value"] * factors[row["unit"]])


def source_cache():
    cache = {}
    for spec in PINS["input_sources"]:
        path = UPLOAD / "inputs" / spec["path"]
        assert sha(path) == spec["sha256"]
        values = {}
        scales = defaultdict(list)
        for line in path.open():
            row = json.loads(line)
            m = row["mhc"][0]
            y = physical(row)
            if row["split"] == "train":
                if row["qualifier"] == "=":
                    scales[row["endpoint"]].append(y)
            else:
                values[row["observation_id"]] = dict(
                    endpoint=row["endpoint"],
                    target=y,
                    qualifier=row["qualifier"],
                    split=row["split"],
                    peptide=row["peptide"],
                    mhc_class=m["mhc_class"],
                    groove1=m["groove1"],
                    groove2=m["groove2"],
                    assay=row["provenance"]["original_measurement_id"],
                )
        cache[spec["path"]] = {"rows": values, "scales": scales}
    return cache


def verify(condition, cache):
    decision = read(F / "results/validation-decision.json")
    assert decision["test_used_for_decision"] is False
    directory = F / "results" / condition
    report = read(directory / "report.json")
    assert sha(directory / "report.json") == decision["report_sha256"][condition]
    assert sha(report["checkpoint"]) == report["checkpoint_sha256"]
    assert report["summary"]["coverage_gate"]["status"] == "passed"
    resolved = read(directory / "resolved-config.json")
    expected = {}
    train = defaultdict(list)
    for spec in PINS["manifests"][condition]["sources"]:
        data = cache[spec["path"]]
        expected.update(data["rows"])
        for e, v in data["scales"].items():
            train[e].extend(v)
    for e, v in train.items():
        np.testing.assert_allclose(
            [np.mean(v), np.std(v)],
            [resolved["scales"][e]["mean"], resolved["scales"][e]["scale"]],
            rtol=1e-11,
            atol=1e-12,
        )
        assert len(v) == resolved["scales"][e]["exact_count"]
    results = {}
    table = []
    grooves = []
    common = {}
    for split in ("validation", "test"):
        path = directory / f"{split}-predictions.jsonl.gz"
        rows = load(path)
        ids = [r["observation_id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert set(ids) == {k for k, v in expected.items() if v["split"] == split}
        for r in rows:
            source = expected[r["observation_id"]]
            o = r["observation"]
            m = o["mhc"][0]
            for key in ("endpoint", "qualifier", "peptide", "split"):
                assert r[key] == source[key]
            assert (
                r["mhc_classes"] == [source["mhc_class"]]
                and m["groove1"] == source["groove1"]
                and m["groove2"] == source["groove2"]
            )
            assert o["provenance"]["original_measurement_id"] == source["assay"]
            assert (
                abs(r["target"] - source["target"]) < 1e-12
                and abs(physical(o) - source["target"]) < 1e-12
            )
        np.testing.assert_allclose(
            np.mean([r["loss"] for r in rows]),
            report["summary"]["overall_losses"][split]["assay_observation_mean"],
            rtol=1e-12,
        )
        endpoint_results = {}
        all_baselines = load(directory / f"mean-{split}-predictions.jsonl.gz")
        assert [r["observation_id"] for r in all_baselines] == ids
        for baseline_row, prediction_row in zip(all_baselines, rows, strict=True):
            for key in ("target", "qualifier", "endpoint", "mhc_classes", "split"):
                assert baseline_row[key] == prediction_row[key]
        for endpoint in resolved["model"]["endpoints"]:
            subset = [r for r in rows if r["endpoint"] == endpoint]
            scale = resolved["scales"][endpoint]["scale"]
            binding = endpoint != "half_life"
            got = metrics(subset, scale, binding)
            reported = report["summary"]["metrics"][split].get(endpoint, {})
            assert got["n_all"] == reported.get("n", 0)
            if got["n"]:
                assert_metric(got, reported["exact"])
            if got["n_all"]:
                np.testing.assert_allclose(
                    got["loss"], reported["loss"], rtol=5e-5, atol=3e-6
                )
            classes = {}
            class_baselines = {}
            for cls in ("I", "II"):
                rr = [r for r in subset if r["mhc_classes"] == [cls]]
                m = metrics(rr, scale, binding)
                classes[cls] = m
                class_baselines[cls] = metrics(
                    [
                        r
                        for r in all_baselines
                        if r["endpoint"] == endpoint and r["mhc_classes"] == [cls]
                    ],
                    scale,
                    binding,
                )
                if m["n"]:
                    assert_metric(m, reported["by_class"][cls])
                flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
                flat.update(
                    {
                        "binding_" + k: v
                        for k, v in m.get("binding_at_500_nM", {}).items()
                    }
                )
                table.append(
                    dict(
                        condition=condition,
                        variant=PINS["conditions"][condition]["variant"],
                        seed=PINS["conditions"][condition]["seed"],
                        split=split,
                        endpoint=endpoint,
                        mhc_class=cls,
                        **flat,
                    )
                )
                groups = defaultdict(list)
                for r in rr:
                    mhc = r["observation"]["mhc"][0]
                    key = mhc["provenance"]["reported_restriction"]
                    groups[key].append(r)
                for key, values in groups.items():
                    gm = metrics(values, scale, binding)
                    if gm["n"] >= 20:
                        grooves.append(
                            dict(
                                condition=condition,
                                split=split,
                                endpoint=endpoint,
                                mhc_class=cls,
                                restriction=key,
                                **{
                                    k: v
                                    for k, v in gm.items()
                                    if not isinstance(v, dict)
                                },
                            )
                        )
            baseline = [r for r in all_baselines if r["endpoint"] == endpoint]
            assert all(
                r["prediction"] == resolved["scales"][endpoint]["mean"]
                for r in baseline
            )
            baseline_metrics = metrics(baseline, scale, binding)
            if baseline_metrics["n"]:
                assert_metric(
                    baseline_metrics, report["mean_baselines"][split][endpoint]["exact"]
                )
            endpoint_results[endpoint] = dict(
                overall=got,
                by_class=classes,
                mean_baseline_by_class=class_baselines,
                mean_baseline=baseline_metrics,
            )
        results[split] = dict(
            predictions_sha256=sha(path),
            overall_loss=float(np.mean([r["loss"] for r in rows])),
            endpoints=endpoint_results,
        )
        original_ids = set(cache[f"data/ic50-original-{split}.jsonl"]["rows"])
        rr = [r for r in rows if r["observation_id"] in original_ids]
        assert len(rr) == len(original_ids)
        common[split] = {
            cls: metrics(
                [r for r in rr if r["mhc_classes"] == [cls]],
                resolved["scales"]["ic50"]["scale"],
            )
            for cls in ("I", "II")
        }
        for cls in ("I", "II"):
            assert_metric(
                common[split][cls],
                report["common_ic50"][split]["ic50"]["by_class"][cls],
            )
    payload = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)
    coverage = read(directory / "coverage.json")["parameter_elements"]
    masks = payload["element_coverage"]
    assert isinstance(masks["complete_history"], list)
    assert set(masks["complete_history"]) == set(masks["masks"])
    assert set(masks["masks"]) == set(coverage)
    assert all(v["complete_history"] for v in coverage.values())
    for name, mask in masks["masks"].items():
        assert (
            mask.dtype == torch.bool and mask.shape == payload["state_dict"][name].shape
        )
        assert int(mask.sum()) == coverage[name]["elements_with_gradient_and_update"]
    total = sum(v["elements"] for v in coverage.values())
    updated = sum(v["elements_with_gradient_and_update"] for v in coverage.values())
    fixed = sum(v["fixed_padding_elements"] for v in coverage.values())
    redundant = sum(
        v["structurally_redundant_key_bias_elements"] for v in coverage.values()
    )
    result = dict(
        status="passed",
        condition=condition,
        checkpoint_sha256=report["checkpoint_sha256"],
        metrics=results,
        common_ic50=common,
        per_groove_min20=grooves,
        parameter_evidence=dict(
            total=total,
            coordinates_with_gradient_and_update=updated,
            coordinates_without_recorded_update=total - updated,
            fixed_padding=fixed,
            redundant_key_bias=redundant,
            complete_history=True,
        ),
        resources=report["resources"],
        endpoint_class_exposures=report["endpoint_class_examples"],
    )
    write(F / "results" / f"{condition}-independent-verification.json", result)
    return table, grooves, result


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    assert (
        read(F / "results/validation-decision.json")["test_used_for_decision"] is False
    )
    cache = source_cache()
    all_rows = []
    all_grooves = []
    support = []
    for condition in PINS["conditions"]:
        rows, grooves, result = verify(condition, cache)
        all_rows.extend(rows)
        all_grooves.extend(grooves)
        support.append(
            dict(
                condition=condition,
                **result["parameter_evidence"],
                **result["resources"],
            )
        )
        print(condition, result["status"], result["common_ic50"]["test"], flush=True)
    csv_write(F / "results/metrics-independent.csv", all_rows)
    csv_write(F / "results/groove-metrics-min20.csv", all_grooves)
    csv_write(F / "results/coverage-runtime.csv", support)
    write(
        F / "results/verification-summary.json",
        dict(
            status="passed",
            conditions=list(PINS["conditions"]),
            metrics_rows=len(all_rows),
            independent_targets_units_losses_scales_metrics=True,
        ),
    )


if __name__ == "__main__":
    main()
