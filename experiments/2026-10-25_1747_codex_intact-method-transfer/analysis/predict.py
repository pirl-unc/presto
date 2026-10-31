"""Frozen prediction and independent algebra checks, without future outcome access."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    request = json.loads(parser.parse_args().requests.read_text())
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "reproduce/launch.json").read_text())
    replay = json.loads(
        (root / "results/admission-independent-replay.json").read_text()
    )
    assert replay["status"] == "passed" and replay["support_passed"]
    admission = json.loads((root / "results/admission.json").read_text())
    package = Path(admission["recipe_directory"])
    rows = pq.read_table(package / "typed/covariates.parquet").to_pylist()
    assert rows and not {
        "target_change",
        "target_log10_ratio",
        "quantitative_value",
        "normalizer_value",
    } & set(rows[0])
    for name, expected in config["model_input_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    physical = module("frozen_model", request["physical_source"])
    direct = module("frozen_0143_core", request["direct_source"])
    torch.set_num_threads(1)
    parent = rows[0]["parent_sequence"]
    assert all(
        r["parent_sequence"] == parent
        and parent[r["start_1based"] - 1 : r["end_1based"]] == r["peptide_sequence"]
        for r in rows
    )
    categories = sorted(set(parent[:-1]))
    aa_index = torch.tensor([categories.index(a) for a in parent[:-1]])
    batch = {
        "start": torch.tensor([r["start_1based"] - 1 for r in rows]),
        "end": torch.tensor([r["end_1based"] for r in rows]),
        "time": torch.tensor([r["time_minutes"] for r in rows], dtype=torch.float64),
    }
    zero_batch = {**batch, "time": torch.zeros(len(rows), dtype=torch.float64)}
    design_rows = [{**r, "elapsed": r["time_minutes"] / 80} for r in rows]
    direct_fits = json.loads(Path(request["direct_fits"]).read_text())
    physical_fits = json.loads(Path(request["physical_fits"]).read_text())
    selected = json.loads(Path(request["selected_models"]).read_text())
    predictions, support, timing, checks = {}, {}, {}, []

    def physical_predict(theta, kind, tag):
        theta = np.asarray(theta, dtype=np.float64)
        with torch.no_grad():
            tensor = torch.tensor(theta)
            values = (
                physical.component(tensor, kind, batch, aa_index)
                - physical.component(tensor, kind, zero_batch, aa_index)
            ).numpy()
            at_zero = (
                physical.component(tensor, kind, zero_batch, aa_index)
                - physical.component(tensor, kind, zero_batch, aa_index)
            ).numpy()
        category_rates = 1e-7 + (1 - 1e-7) / (1 + np.exp(-theta[:-1]))
        rates = (
            np.repeat(category_rates, len(parent) - 1)
            if kind == "uniform"
            else category_rates[aa_index.numpy()]
        )
        tau = 0.001 + (120 - 0.001) / (1 + math.exp(-theta[-1]))
        independent = []
        for r in rows:
            start, end, t = r["start_1based"] - 1, r["end_1based"], r["time_minutes"]
            boundaries = [i for i in (start - 1, end - 1) if 0 <= i < len(rates)]
            outside = math.fsum(
                float(rate) for i, rate in enumerate(rates) if not start <= i < end - 1
            )
            logs = math.fsum(
                math.log(-math.expm1(-(t + tau) * rates[i]))
                - math.log(-math.expm1(-tau * rates[i]))
                for i in boundaries
            )
            independent.append((t * outside + logs) / math.log(10))
        error = float(np.max(np.abs(values - independent)))
        assert error < 1e-10 and np.all(at_zero == 0) and np.all(np.isfinite(values))
        checks.append(
            {
                "model": tag,
                "independent_max_absolute_error": error,
                "zero_time_change_exactly_zero": True,
            }
        )
        return values

    predictions["zero"] = np.zeros(len(rows))
    support["zero"] = [
        {"training_sequence_count": None, "novel_constant_coordinates": None}
        for _ in rows
    ]
    timing["zero"] = 0.0
    for arm in ("shared_slope", "length", "sequence"):
        started = time.perf_counter()
        fits = [
            f
            for f in direct_fits
            if f["arm"] == arm
            and (arm == "shared_slope" or f.get("role") == "selected_outer")
        ]
        assert len(fits) == 5 and sorted(f["outer_fold"] for f in fits) == list(
            range(5)
        )
        values = []
        seen = np.zeros(len(rows), dtype=int)
        novelty = [set() for _ in rows]
        for fit in fits:
            training = (
                fit["training_sequences"]
                if arm == "shared_slope"
                else fit["transform"]["training_sequences"]
            )
            seen += np.array([r["peptide_sequence"] in training for r in rows])
            if arm == "shared_slope":
                predicted = np.array(
                    [r["time_minutes"] / 80 * fit["slope"] for r in rows]
                )
                error = 0.0
            else:
                assert any(
                    s["outer_fold"] == fit["outer_fold"]
                    and s["arm"] == arm
                    and s["alpha"] == fit["alpha"]
                    for s in selected["ridge_selected"]
                )
                predicted = direct.ridge_predict(design_rows, fit)
                transform = fit["transform"]
                independent = []
                for i, r in enumerate(rows):
                    seq = r["peptide_sequence"]
                    raw = [math.log(len(seq))]
                    if arm == "sequence":
                        raw += [seq.count(a) / len(seq) for a in "ACDEFGHIKLMNPQRSTVWY"]
                        raw += [float(seq[0] == a) for a in "ACDEFGHIKLMNPQRSTVWY"]
                        raw += [float(seq[-1] == a) for a in "ACDEFGHIKLMNPQRSTVWY"]
                    standardized = []
                    for j, (x, center, scale, constant) in enumerate(
                        zip(
                            raw,
                            transform["center"],
                            transform["scale"],
                            transform["constant_columns"],
                            strict=True,
                        )
                    ):
                        if constant and abs(x - center) > 1e-12:
                            novelty[i].add(j)
                        standardized.append(0.0 if constant else (x - center) / scale)
                    z = np.asarray(standardized) @ np.array(transform["basis"])
                    independent.append(
                        r["time_minutes"]
                        / 80
                        * (
                            fit["coefficients"][0]
                            + sum(
                                float(a) * b
                                for a, b in zip(z, fit["coefficients"][1:], strict=True)
                            )
                        )
                    )
                error = float(np.max(np.abs(predicted - independent)))
                assert error < 1e-10
                assert np.all(
                    direct.ridge_predict(
                        [{**r, "elapsed": 0.0} for r in design_rows], fit
                    )
                    == 0
                )
            checks.append(
                {
                    "model": f"{arm}-fold{fit['outer_fold']}",
                    "independent_max_absolute_error": error,
                    "zero_time_change_exactly_zero": True,
                }
            )
            values.append(predicted)
        predictions[arm] = np.mean(values, axis=0)
        support[arm] = [
            {
                "training_sequence_count": int(n),
                "novel_constant_coordinates": json.dumps(sorted(v)),
            }
            for n, v in zip(seen, novelty, strict=True)
        ]
        timing[arm] = time.perf_counter() - started
    for arm in sorted({s["arm"] for s in selected["physical_selected"]}):
        started = time.perf_counter()
        values = []
        seen = np.zeros(len(rows), dtype=int)
        choices = [s for s in selected["physical_selected"] if s["arm"] == arm]
        assert len(choices) == 5
        for choice in choices:
            fits = [
                f
                for f in physical_fits
                if all(f[k] == choice[k] for k in ("arm", "outer_fold", "start_number"))
            ]
            assert len(fits) == 1
            f = fits[0]
            seen += np.array(
                [r["peptide_sequence"] in f["training_sequences"] for r in rows]
            )
            values.append(
                physical_predict(
                    f["raw_parameters"], f["kind"], f"{arm}-fold{f['outer_fold']}"
                )
            )
        predictions[arm] = np.mean(values, axis=0)
        support[arm] = [
            {"training_sequence_count": int(n), "novel_constant_coordinates": None}
            for n in seen
        ]
        timing[arm] = time.perf_counter() - started
    summaries = sorted(
        Path(p) for p in config["model_input_sha256"] if p.endswith("/summary.json")
    )
    assert len(summaries) == 14
    for path in summaries:
        started = time.perf_counter()
        fit = json.loads(path.read_text())
        arm = "original_" + path.parent.name.removeprefix("fit-")
        with np.load(
            path.parent / "checkpoints.npz", allow_pickle=False
        ) as checkpoints:
            assert np.array_equal(
                checkpoints["theta"][-1], np.array(fit["raw_parameters"])
            )
            assert list(checkpoints["aa_categories"]) == categories
            seen = [
                r["peptide_sequence"] in set(checkpoints["sequences"]) for r in rows
            ]
        predictions[arm] = physical_predict(fit["raw_parameters"], fit["kind"], arm)
        support[arm] = [
            {"training_sequence_count": int(n), "novel_constant_coordinates": None}
            for n in seen
        ]
        timing[arm] = time.perf_counter() - started
    assert len(predictions) == 21
    output = []
    for arm, values in predictions.items():
        for r, value, flags in zip(rows, values, support[arm], strict=True):
            output.append(
                {
                    **{
                        k: r[k]
                        for k in (
                            "assignment_id",
                            "peptide_sequence",
                            "organ_label",
                            "time_minutes",
                            "split",
                            "weight",
                            "baseline_log10_ratio",
                            "n_baseline_observations",
                        )
                    },
                    "arm": arm,
                    "prediction": float(value),
                    "predicted_log10_ratio": float(value + r["baseline_log10_ratio"]),
                    **flags,
                }
            )
    target = root / "results/frozen-predictions.parquet"
    assert not target.exists()
    pq.write_table(pa.Table.from_pylist(output), target)
    result = {
        "status": "passed",
        "n_arms": 21,
        "n_query_cells": len(rows),
        "n_predictions": len(output),
        "future_outcomes_read": False,
        "fitting_or_selection": False,
        "checks": checks,
        "prediction_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "inference_and_verification_seconds_by_arm": timing,
        "timing_scope": "includes independent prediction checks and checkpoint loading, not prior training",
        "model_input_sha256": config["model_input_sha256"],
    }
    with (root / "results/prediction-verification.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("checks", "model_input_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
