"""Independent post-outcome source, predictor, weighting and metric closure."""

import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import independent
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compare(expected, actual):
    assert set(expected) == set(actual)
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None, (key, actual[key])
        else:
            np.testing.assert_allclose(actual[key], value, rtol=1e-9, atol=1e-10)


def main():
    started = time.perf_counter()
    config = read(ROOT / "reproduce/launch.json")
    for path, expected in config["input_sha256"].items():
        assert digest(path) == expected
    for path, expected in config["frozen_sha256"].items():
        assert digest(ROOT / path) == expected
    checkpoints = read(ROOT / "reproduce/checkpoints.json")
    output = ROOT / "results/evaluation"
    summary = read(output / "summary.json")
    for path, expected in read(output / "predictions-sealed-before-outcomes.json")[
        "sha256"
    ].items():
        assert digest(output / path) == expected
    recipe = Path(config["recipe"])
    reference_ids = {
        r["reference_record_id"]: r["sequence_sha256"]
        for r in pq.read_table(
            recipe / "source/references.parquet", columns=["reference_record_id", "sequence_sha256"]
        ).to_pylist()
    }
    all_origins = defaultdict(set)
    for r in pq.read_table(recipe / "source/occurrences.parquet").to_pylist():
        if r["reference_record_id"] is not None:
            all_origins[r["peptide_sequence"]].add(
                (reference_ids[r["reference_record_id"]], r["start_1based"] - 1, r["end_1based"])
            )
    all_parents = {
        r["parent_id"]: r["reference_sequence"]
        for r in pq.read_table(recipe / "derived/parents.parquet").to_pylist()
    }
    all_inputs = {
        r["parent_id"]: r
        for r in pq.read_table(
            recipe / "derived/precursor_inputs.parquet",
            filters=[("condition_id", "=", "hap1_erap1_ko")],
        ).to_pylist()
    }
    reported_strata = defaultdict(list)
    for row in pq.read_table(output / "strata.parquet").to_pylist():
        reported_strata[row["split"], row["scope"], row["model"]].append(row)
    metrics_count, prediction_count, max_error = 0, 0, 0.0
    components = {
        "train": set(
            pq.read_table(
                Path(config["baseline"]) / "results/fits/cohort.parquet", columns=["component_id"]
            )["component_id"].to_pylist()
        )
    }
    gate_checks = {}
    for split in ["validation", "test"]:
        directory = output / split
        cohort = pq.read_table(directory / "cohort.parquet")
        canonical = pq.read_table(
            recipe / "derived/responses.parquet",
            filters=[("split", "=", split), ("precursor_status", "=", "complete")],
        )
        assert cohort.equals(canonical)
        components[split] = set(cohort["component_id"].to_pylist())
        origins, parents = read(directory / "origins.json"), read(directory / "parents.json")
        erap_peptides = {
            r["peptide_sequence"]
            for r in cohort.to_pylist()
            if r["condition_id"] == "hap1_erap1_ko"
        }
        expected_origins = {
            (p, parent, start, end) for p in erap_peptides for parent, start, end in all_origins[p]
        }
        assert expected_origins == {
            (o["peptide_sequence"], o["parent_id"], o["start0"], o["end0"]) for o in origins
        }
        assert len(origins) == len(expected_origins)
        assert parents == {p: all_parents[p] for p in parents}
        for origin in origins:
            value = all_inputs[origin["parent_id"]]
            assert value["status"] == "complete"
            assert value["log10_response"] == origin["parent_log10_response"]
        for scope in ["direct", "erap1"]:
            table = pq.read_table(directory / f"{scope}-predictions.parquet")
            rows = table.select(cohort.column_names).to_pylist()
            expected_rows = [
                r
                for r in cohort.to_pylist()
                if scope == "direct" or r["condition_id"] == "hap1_erap1_ko"
            ]
            assert rows == expected_rows
            w = independent.weights(rows)
            np.testing.assert_array_equal(w, np.array(table["evaluation_weight"]))
            y = np.array(table["target_log10_ratio"])
            reconstructed = {
                name: independent.direct(rows, cp) for name, cp in checkpoints["direct"].items()
            }
            corrections = {}
            if scope == "erap1":
                for name, record in checkpoints["physical"].items():
                    cp = record["checkpoint"]
                    raw = independent.physical(rows, origins, parents, cp)
                    correction = raw.copy()
                    p = raw + cp["offset"]
                    if cp["arm"].startswith("augmented"):
                        correction -= cp["gamma"] * np.array(
                            [r["precursor_log10_ratio"] for r in rows]
                        )
                        p = reconstructed["sequence_0.1"] + correction + cp["offset"]
                    reconstructed[name], corrections[name] = p, correction
                    np.testing.assert_allclose(
                        correction, np.array(table["correction_" + name]), rtol=1e-10, atol=1e-10
                    )
            groups = defaultdict(list)
            for i, row in enumerate(rows):
                groups["condition", row["condition_id"]].append(i)
                groups[
                    "condition_component", row["condition_id"] + ":" + row["component_id"]
                ].append(i)
                groups["length", str(len(row["peptide_sequence"]))].append(i)
            computed = {}
            for name, prediction in reconstructed.items():
                saved = np.array(table["prediction_" + name])
                np.testing.assert_allclose(prediction, saved, rtol=1e-10, atol=1e-10)
                max_error = max(max_error, float(np.max(abs(prediction - saved))))
                prediction_count += len(rows)
                computed[name] = independent.metrics(y, saved, w)
                compare(computed[name], summary["metrics"][split + ":" + scope][name])
                reports = reported_strata[split, scope, name]
                assert len(reports) == len(groups)
                for row in reports:
                    ix = groups[row["stratum_type"], row["stratum"]]
                    actual = independent.metrics(y[ix], saved[ix], w[ix])
                    compare(actual, {k: row[k] for k in actual})
                    metrics_count += 1
            if scope == "direct":
                spreads = {
                    c: independent.metrics(y[ix], reconstructed["sequence_0.1"][ix], w[ix])[
                        "prediction_sd"
                    ]
                    for (kind, c), ix in groups.items()
                    if kind == "condition"
                }
                gain = (
                    1 - computed["sequence_0.1"]["rmse"] / computed["calibrated_precursor"]["rmse"]
                )
                passed = gain >= 0.02 and len(spreads) == 10 and min(spreads.values()) > 0.001
                for c, sd in spreads.items():
                    np.testing.assert_allclose(
                        sd,
                        summary["gates"][split + ":direct"]["condition_prediction_sd"][c],
                        atol=1e-10,
                    )
            else:
                gain = (
                    1 - computed["1007_augmented_exact"]["rmse"] / computed["sequence_0.1"]["rmse"]
                )
                sd = independent.metrics(y, corrections["1007_augmented_exact"], w)["prediction_sd"]
                passed = gain >= 0.02 and sd > 0.001
                np.testing.assert_allclose(
                    sd, summary["gates"][split + ":erap1"]["correction_sd"], atol=1e-10
                )
            assert bool(passed) == summary["gates"][split + ":" + scope]["passed"]
            np.testing.assert_allclose(
                gain, summary["gates"][split + ":" + scope]["relative_rmse_gain"], atol=1e-10
            )
            gate_checks[split + ":" + scope] = {"passed": bool(passed), "relative_rmse_gain": gain}
    assert not (
        components["train"] & components["validation"]
        or components["train"] & components["test"]
        or components["validation"] & components["test"]
    )
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    assert elapsed < 900 and peak < 4
    result = {
        "status": "passed",
        "seconds": elapsed,
        "peak_gib": peak,
        "n_prediction_values": prediction_count,
        "max_prediction_error": max_error,
        "n_stratum_metric_records": metrics_count,
        "gates": gate_checks,
        "component_disjointness": True,
        "source_scope": "Canonical recipe lineage and all origins/precursors; upstream original-cell audit reused.",
        "no_fitting": True,
    }
    (ROOT / "results/independent-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
