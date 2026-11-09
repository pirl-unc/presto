"""Evaluate sealed predictions once, preserving all rows and matched comparisons."""

import csv
import datetime
import gzip
import resource
import time

import numpy as np
from core import OBSERVER, ROOT, Data, read, sha, write


def main():
    start = time.perf_counter()
    receipt = read(ROOT / "results/fit-complete.json")
    assert receipt["status"] == "completed" and not receipt["test_labels_read"]
    output = ROOT / "results/evaluation"
    output.mkdir()
    write(
        output / "outcome-access.json",
        {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "fit_receipt_sha256": sha(ROOT / "results/fit-complete.json"),
            "evaluation_code_sha256": sha(__file__),
            "primary_model": "sequence_observer_mse",
            "observer_transfer_model": "observer_fixed",
            "component_bootstrap_seed": 20261024,
            "component_bootstrap_resamples": 2000,
            "test_access": "first_model_evaluation_after_all_checkpoints_and_covariate_predictions_sealed",
        },
    )
    all_metrics = {}
    comparisons = {}
    strata = []
    for block, files in receipt["files"].items():
        directory = ROOT / "results/fit" / block
        for name, digest in files.items():
            assert sha(directory / name) == digest
        data = Data(block)
        predictions = np.load(directory / "sealed-predictions.npz")
        selected = read(directory / "validation-selection.json")
        labels = {s: data.labels(s) for s in ("train", "val", "test")}
        result = {
            name: {
                s: data.metrics(predictions[name], s, labels[s])
                for s in ("train", "val", "test")
            }
            for name in predictions.files
        }
        all_metrics[block] = result
        with gzip.open(output / (block + "-predictions.csv.gz"), "wt") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model",
                    "split",
                    "measurement_id",
                    "source_record_id",
                    "source_cell",
                    "group_id",
                    "component_id",
                    "parent_id",
                    "sequence",
                    "target_centered_log10_area",
                    "target_area_fraction",
                    "prediction_centered_log10_area",
                    "prediction_area_fraction",
                    "weight",
                ],
            )
            writer.writeheader()
            for name in predictions.files:
                prediction = predictions[name]
                fractions = np.zeros(len(data.rows))
                for index in data.indices:
                    logq = prediction[index] * np.log(10)
                    fractions[index] = np.exp(logq - np.logaddexp.reduce(logq))
                for i, row in enumerate(data.rows):
                    split = row["split"]
                    target, p = labels[split]
                    writer.writerow(
                        {
                            "model": name,
                            **{
                                k: row[k]
                                for k in (
                                    "split",
                                    "measurement_id",
                                    "source_record_id",
                                    "source_cell",
                                    "group_id",
                                    "component_id",
                                    "parent_id",
                                    "sequence",
                                )
                            },
                            "target_centered_log10_area": target[i],
                            "target_area_fraction": p[i],
                            "prediction_centered_log10_area": prediction[i],
                            "prediction_area_fraction": fractions[i],
                            "weight": data.weights[split][i],
                        }
                    )
                for split in ("val", "test"):
                    for component in sorted(
                        {r["component_id"] for r in data.rows if r["split"] == split}
                    ):
                        mask = np.array(
                            [
                                r["split"] == split and r["component_id"] == component
                                for r in data.rows
                            ]
                        )
                        strata.append(
                            {
                                "block": block,
                                "model": name,
                                "split": split,
                                "component_id": component,
                                **OBSERVER.metrics(
                                    labels[split][0][mask],
                                    prediction[mask],
                                    data.weights[split][mask],
                                ),
                            }
                        )
        primary = "sequence_observer_mse"
        comparators = [selected["sequence"], "observer_calibrated"]
        gates = {}
        for comparator in comparators:
            gates[comparator] = {
                s: 1 - result[primary][s]["rmse"] / result[comparator][s]["rmse"]
                for s in ("val", "test")
            }
        gates["primary_passes"] = (
            all(gates[c][s] >= 0.05 for c in comparators for s in ("val", "test"))
            and result[primary]["test"]["r2"] > 0
        )
        gates["observer_gains"] = {
            s: 1
            - result["observer_fixed"][s]["rmse"]
            / result[selected["length"]][s]["rmse"]
            for s in ("val", "test")
        }
        gates["observer_passes"] = all(
            x >= 0.05 for x in gates["observer_gains"].values()
        )
        boot = {}
        ids = sorted({r["component_id"] for r in data.rows if r["split"] == "test"})
        rng = np.random.default_rng(20261024)
        draws = rng.integers(0, len(ids), (2000, len(ids)))
        primary_mse = np.array(
            [
                next(
                    r["mse"]
                    for r in strata
                    if r["block"] == block
                    and r["model"] == primary
                    and r["split"] == "test"
                    and r["component_id"] == c
                )
                for c in ids
            ]
        )
        for comparator in comparators:
            baseline_mse = np.array(
                [
                    next(
                        r["mse"]
                        for r in strata
                        if r["block"] == block
                        and r["model"] == comparator
                        and r["split"] == "test"
                        and r["component_id"] == c
                    )
                    for c in ids
                ]
            )
            gain = 1 - np.sqrt(primary_mse[draws].mean(1) / baseline_mse[draws].mean(1))
            boot[comparator] = {
                "relative_RMSE_gain_percentile_95": np.quantile(
                    gain, [0.025, 0.975]
                ).tolist(),
                "n_components": len(ids),
                "scope": "same_study_component_resampling_not_biological_specimen_or_external_transfer_uncertainty",
            }
        comparisons[block] = {
            "selected_controls": selected,
            "gates": gates,
            "paired_component_bootstrap": boot,
        }
    write(output / "metrics.json", all_metrics)
    write(output / "comparisons.json", comparisons)
    write(output / "component-strata.json", strata)
    result = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    write(ROOT / "results/evaluation-complete.json", result)
    print(comparisons)


if __name__ == "__main__":
    main()
