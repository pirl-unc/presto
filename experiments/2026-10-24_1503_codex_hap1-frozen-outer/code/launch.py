"""Training-only qualification followed by one frozen two-split evaluation."""

import argparse
import datetime
import resource
import time
from pathlib import Path

import independent
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from core import (
    INPUT_COLUMNS,
    ROOT,
    Paths,
    Source,
    direct,
    metrics,
    physical_prediction,
    read,
    sha,
    strata,
    weights,
    write,
)


def check_freeze(config):
    for path, expected in config["input_sha256"].items():
        assert sha(path) == expected, path
    for path, expected in config["frozen_sha256"].items():
        assert sha(ROOT / path) == expected, path


def prepare(config, checkpoints):
    started = time.perf_counter()
    baseline = Path(config["baseline"])
    # The archive contains outer-training rows only.
    table = pq.read_table(baseline / "results/fits/cohort.parquet")
    rows = table.to_pylist()
    assert len(rows) == 87260 and all(r["split"] == "train" for r in rows)
    inputs = table.select(INPUT_COLUMNS).to_pylist()
    changed = [{**r, "target_log10_ratio": float("nan"), "weight": -999} for r in inputs]
    records, predictions = [], {}
    for name, cp in checkpoints["direct"].items():
        predicted = direct(inputs, cp)
        assert np.array_equal(predicted, direct(changed, cp))
        reference = pq.read_table(baseline / f"results/fits/{name}-train.parquet").to_pydict()
        assert reference["measurement_id"] == table["measurement_id"].to_pylist()
        np.testing.assert_allclose(
            predicted, reference["prediction_log10_ratio"], atol=1e-10, rtol=1e-10
        )
        verified = independent.direct(inputs, cp)
        np.testing.assert_allclose(predicted, verified, atol=1e-10, rtol=1e-10)
        records.append(
            {
                "model": name,
                "n": len(rows),
                "archive_max_error": float(
                    np.max(abs(predicted - reference["prediction_log10_ratio"]))
                ),
                "independent_max_error": float(np.max(abs(predicted - verified))),
            }
        )
        predictions[name] = predicted
    selected = np.array([r["condition_id"] == "hap1_erap1_ko" for r in rows])
    erap = [r for r, s in zip(inputs, selected, strict=True) if s]
    changed = [{**r, "target_log10_ratio": float("nan")} for r in erap]
    paths = Paths(erap, Source(Path(config["recipe"])))
    assert len(erap) == 8281 and paths.n_paths == 1741916 and len(paths.origins) == 3409
    frozen_baseline = predictions["sequence_0.1"][selected]
    for name, record in checkpoints["physical"].items():
        cp = record["checkpoint"]
        p, _ = physical_prediction(paths, erap, cp, frozen_baseline)
        assert np.array_equal(p, physical_prediction(paths, changed, cp, frozen_baseline)[0])
        reference = pq.read_table(
            Path(record["directory"]) / f"results/fits/{cp['arm']}/train.parquet"
        ).to_pydict()
        assert reference["measurement_id"] == [
            r["measurement_id"] for r, s in zip(rows, selected, strict=True) if s
        ]
        np.testing.assert_allclose(p, reference["prediction_log10_ratio"], atol=1e-10, rtol=1e-10)
        raw = independent.physical(erap, paths.origins, paths.parents, cp)
        verified = raw + cp["offset"]
        if cp["arm"].startswith("augmented"):
            verified += frozen_baseline - cp["gamma"] * np.array(
                [r["precursor_log10_ratio"] for r in erap]
            )
        np.testing.assert_allclose(p, verified, atol=1e-10, rtol=1e-10)
        records.append(
            {
                "model": name,
                "n": len(erap),
                "archive_max_error": float(np.max(abs(p - reference["prediction_log10_ratio"]))),
                "independent_max_error": float(np.max(abs(p - verified))),
            }
        )
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    assert elapsed < 900 and peak < 4
    receipt = {
        "status": "passed",
        "records": records,
        "target_mutation_invariant": True,
        "n_prediction_values": sum(r["n"] for r in records),
        "seconds": elapsed,
        "peak_gib": peak,
        "outer_outcomes_accessed": False,
        "frozen_launch_sha256": sha(ROOT / "reproduce/launch.json"),
    }
    write(ROOT / "results/preparation.json", receipt)
    print(receipt, flush=True)


def evaluate(config, checkpoints):
    started = time.perf_counter()
    assert read(ROOT / "results/preparation.json")["status"] == "passed"
    output = ROOT / "results/evaluation"
    output.mkdir(exist_ok=False)
    source = Source(Path(config["recipe"]))
    population = {}
    # BOTH sets of predictions are sealed before either set of target values is read.
    for split, expected in [("validation", 19744), ("test", 22724)]:
        directory = output / split
        directory.mkdir()
        table = pq.read_table(
            Path(config["recipe"]) / "derived/responses.parquet",
            columns=["measurement_id", *INPUT_COLUMNS],
            filters=[("split", "=", split), ("precursor_status", "=", "complete")],
        )
        assert len(table) == expected
        inputs = table.select(INPUT_COLUMNS).to_pylist()
        pq.write_table(table, directory / "covariates.parquet", compression="zstd")
        predicted = {name: direct(inputs, cp) for name, cp in checkpoints["direct"].items()}
        erap_indices = np.array(
            [i for i, r in enumerate(inputs) if r["condition_id"] == "hap1_erap1_ko"]
        )
        erap = [inputs[i] for i in erap_indices]
        paths = Paths(erap, source)
        write(directory / "origins.json", paths.origins)
        write(directory / "parents.json", paths.parents)
        erap_predicted = {name: p[erap_indices] for name, p in predicted.items()}
        corrections = {}
        for name, record in checkpoints["physical"].items():
            p, correction = physical_prediction(
                paths, erap, record["checkpoint"], erap_predicted["sequence_0.1"]
            )
            erap_predicted[name] = p
            corrections[name] = correction
        np.savez_compressed(directory / "direct-predictions-only.npz", **predicted)
        np.savez_compressed(directory / "erap1-predictions-only.npz", **erap_predicted)
        np.savez_compressed(directory / "corrections-only.npz", **corrections)
        population[split] = {
            "n_observations": expected,
            "n_erap1_observations": len(erap),
            "n_erap1_peptides": len(paths.peptides),
            "n_erap1_origins": len(paths.origins),
            "n_erap1_paths": paths.n_paths,
        }
        del paths
    sealed = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    write(
        output / "predictions-sealed-before-outcomes.json",
        {
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "both_split_predictions_sealed_before_outcome_access",
            "sha256": sealed,
            "population": population,
        },
    )
    write(
        output / "outcomes-access-started.json",
        {
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "policy": "Both validation and test are now inspected; no tuning or new selection.",
        },
    )
    summary, all_strata, gates = {}, [], {}
    for split in ["validation", "test"]:
        directory = output / split
        table = pq.read_table(
            Path(config["recipe"]) / "derived/responses.parquet",
            filters=[("split", "=", split), ("precursor_status", "=", "complete")],
        )
        assert table.select(["measurement_id", *INPUT_COLUMNS]).equals(
            pq.read_table(directory / "covariates.parquet")
        )
        pq.write_table(table, directory / "cohort.parquet", compression="zstd")
        for scope in ["direct", "erap1"]:
            selected = (
                table
                if scope == "direct"
                else table.filter(pc.equal(table["condition_id"], "hap1_erap1_ko"))
            )
            rows = selected.to_pylist()
            w = weights(rows)
            y = np.array(selected["target_log10_ratio"])
            assert abs(w.sum() - 1) < 1e-12
            grouping = strata(rows)
            predictions = dict(np.load(directory / f"{scope}-predictions-only.npz"))
            dump = selected.append_column("evaluation_weight", pa.array(w))
            summaries = {}
            for name, p in predictions.items():
                assert np.isfinite(p).all()
                dump = dump.append_column("prediction_" + name, pa.array(p))
                summaries[name] = metrics(y, p, w)
                for (kind, group), indices in grouping.items():
                    all_strata.append(
                        {
                            "split": split,
                            "scope": scope,
                            "model": name,
                            "stratum_type": kind,
                            "stratum": group,
                            **metrics(y[indices], p[indices], w[indices]),
                        }
                    )
            if scope == "direct":
                spreads = {
                    c: metrics(y[ix], predictions["sequence_0.1"][ix], w[ix])["prediction_sd"]
                    for (kind, c), ix in grouping.items()
                    if kind == "condition"
                }
                gain = (
                    1
                    - summaries["sequence_0.1"]["rmse"] / summaries["calibrated_precursor"]["rmse"]
                )
                gate = {
                    "relative_rmse_gain": gain,
                    "condition_prediction_sd": spreads,
                    "passed": bool(
                        gain >= 0.02 and len(spreads) == 10 and min(spreads.values()) > 0.001
                    ),
                }
            else:
                corrections = dict(np.load(directory / "corrections-only.npz"))
                for name, correction in corrections.items():
                    dump = dump.append_column("correction_" + name, pa.array(correction))
                gain = (
                    1
                    - summaries["1007_augmented_exact"]["rmse"] / summaries["sequence_0.1"]["rmse"]
                )
                sd = metrics(y, corrections["1007_augmented_exact"], w)["prediction_sd"]
                gate = {
                    "relative_rmse_gain": gain,
                    "correction_sd": sd,
                    "passed": bool(gain >= 0.02 and sd > 0.001),
                }
            gates[split + ":" + scope] = gate
            summary[split + ":" + scope] = summaries
            pq.write_table(dump, directory / f"{scope}-predictions.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(all_strata), output / "strata.parquet", compression="zstd")
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    assert elapsed < 1800 and peak < 4
    write(
        output / "summary.json",
        {
            "status": "evaluated_once",
            "metrics": summary,
            "gates": gates,
            "population": population,
            "seconds": elapsed,
            "peak_gib": peak,
            "n_strata": len(all_strata),
            "correlation_policy": "Unweighted Pearson/Spearman; error and SD use hierarchical weights.",
        },
    )
    print({"gates": gates, "seconds": elapsed, "peak_gib": peak}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "evaluate"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read(ROOT / "reproduce/launch.json")
    check_freeze(config)
    checkpoints = read(ROOT / "reproduce/checkpoints.json")
    (prepare if args.phase == "prepare" else evaluate)(config, checkpoints)
