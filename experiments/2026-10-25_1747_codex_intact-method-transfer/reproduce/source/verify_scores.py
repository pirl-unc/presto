"""Independent metric/bootstrap replay from archived, unrounded predictions."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in order[i:j]:
            out[k] = (i + 1 + j) / 2
        i = j
    return out


def correlation(left, right):
    if max(left) == min(left) or max(right) == min(right):
        return None
    n = len(left)
    a = math.fsum(left) / n
    b = math.fsum(right) / n
    x = [v - a for v in left]
    y = [v - b for v in right]
    return math.fsum(i * j for i, j in zip(x, y, strict=True)) / math.sqrt(
        math.fsum(i * i for i in x) * math.fsum(j * j for j in y)
    )


def check(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    else:
        assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), (
            actual,
            expected,
        )


def quantiles(values):
    ordered = sorted(values)
    out = []
    for p in (0.025, 0.975):
        x = p * (len(ordered) - 1)
        lower = math.floor(x)
        upper = math.ceil(x)
        out.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (x - lower))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = pq.read_table(root / "results/predictions.parquet").to_pylist()
    frozen = pq.read_table(root / "results/frozen-predictions.parquet").to_pylist()
    assert len(rows) == len(frozen) == 8925
    package = Path(
        json.loads((root / "results/admission.json").read_text())["recipe_directory"]
    )
    targets = {
        r["assignment_id"]: r
        for r in pq.read_table(package / "typed/examples.parquet").to_pylist()
    }
    for row, pred in zip(rows, frozen, strict=True):
        assert all(row[k] == v for k, v in pred.items())
        assert row["target_change"] == targets[row["assignment_id"]]["target_change"]
        check(
            row["predicted_log10_ratio"],
            row["baseline_log10_ratio"] + row["prediction"],
        )
    with (root / "results/predictions.csv").open(newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    for a, b in zip(csv_rows, rows, strict=True):
        assert a["assignment_id"] == b["assignment_id"] and a["arm"] == b["arm"]
        for key in (
            "prediction",
            "target_change",
            "weight",
            "baseline_log10_ratio",
            "target_log10_ratio",
        ):
            assert float(a[key]) == b[key]
    metrics = json.loads((root / "results/metrics.json").read_text())
    assert len(metrics) == 21 * 2 * 6
    for metric in metrics:
        subset = [
            r
            for r in rows
            if r["arm"] == metric["arm"] and r["split"] == metric["split"]
        ]
        stratum = metric["stratum"]
        if stratum.startswith("organ:"):
            subset = [
                r for r in subset if r["organ_label"] == stratum.removeprefix("organ:")
            ]
        elif stratum.startswith("time:"):
            subset = [
                r
                for r in subset
                if r["time_minutes"] == float(stratum.removeprefix("time:"))
            ]
        assert (
            len(subset) == metric["n_observations"]
            and len({r["peptide_sequence"] for r in subset}) == metric["n_sequences"]
        )
        total = math.fsum(r["weight"] for r in subset)
        w = [r["weight"] / total for r in subset]
        y = [r["target_change"] for r in subset]
        p = [r["prediction"] for r in subset]
        mean_y = math.fsum(a * b for a, b in zip(w, y, strict=True))
        mean_p = math.fsum(a * b for a, b in zip(w, p, strict=True))
        var_y = math.fsum(a * (b - mean_y) ** 2 for a, b in zip(w, y, strict=True))
        var_p = math.fsum(a * (b - mean_p) ** 2 for a, b in zip(w, p, strict=True))
        mse = math.fsum(a * (b - c) ** 2 for a, b, c in zip(w, y, p, strict=True))
        independent = {
            "loss_mse": mse,
            "rmse": math.sqrt(mse),
            "mae": math.fsum(a * abs(b - c) for a, b, c in zip(w, y, p, strict=True)),
            "r2": 1 - mse / var_y,
            "target_sd": math.sqrt(var_y),
            "prediction_sd": math.sqrt(var_p),
            "mean_target": mean_y,
            "mean_prediction": mean_p,
            "pearson": correlation(y, p),
            "spearman": correlation(ranks(y), ranks(p)),
            "target_sd_unweighted": math.sqrt(
                math.fsum((v - math.fsum(y) / len(y)) ** 2 for v in y) / len(y)
            ),
            "prediction_sd_unweighted": math.sqrt(
                math.fsum((v - math.fsum(p) / len(p)) ** 2 for v in p) / len(p)
            ),
        }
        for k, v in independent.items():
            check(metric[k], v)
    summary = json.loads((root / "results/score-summary.json").read_text())
    pooled = {
        m["arm"]: m
        for m in metrics
        if m["split"] == "test" and m["stratum"] == "pooled"
    }
    seq = pooled["sequence"]
    gains = {
        f"versus_{a}": 1 - seq["rmse"] / pooled[a]["rmse"] for a in ("zero", "length")
    }
    for k, v in gains.items():
        check(summary["gains"][k], v)
    ratios = {}
    for organ in ("Brain", "Liver", "Spleen"):
        m = {
            m["arm"]: m
            for m in metrics
            if m["split"] == "test" and m["stratum"] == "organ:" + organ
        }
        ratios[organ] = m["sequence"]["rmse"] / m["length"]["rmse"]
        check(summary["organ_sequence_to_length_rmse_ratios"][organ], ratios[organ])
    gates = [
        gains["versus_zero"] >= 0.05,
        gains["versus_length"] >= 0.05,
        seq["pearson"] > 0,
        seq["spearman"] > 0,
        seq["prediction_sd"] >= 0.1 * seq["target_sd"],
        all(r <= 1.1 for r in ratios.values()),
    ]
    assert list(summary["gates"].values()) == gates and summary[
        "quality_passed"
    ] == all(gates)
    arms = sorted(pooled)
    primary = [r for r in rows if r["split"] == "test" and r["arm"] == "zero"]
    identities = sorted({r["peptide_sequence"] for r in primary})
    lookup = {(r["arm"], r["assignment_id"]): r for r in rows}
    with np.load(root / "results/bootstrap-samples.npz", allow_pickle=False) as saved:
        assert list(saved["sequences"]) == identities
        counts = saved["counts"].copy()
    rng = np.random.default_rng(31415)
    assert counts.shape == (1000, len(identities))
    with (root / "results/bootstrap.csv").open(newline="") as f:
        boot = list(csv.DictReader(f))
    assert len(boot) == 1000
    values = {k: [] for k in (*arms, "gain_vs_zero", "gain_vs_length")}
    invalid = 0
    for draw, (multiplicity, archived) in enumerate(zip(counts, boot, strict=True)):
        expected = np.bincount(
            rng.integers(len(identities), size=len(identities)),
            minlength=len(identities),
        )
        assert np.array_equal(expected, multiplicity) and int(archived["draw"]) == draw
        weight = [
            r["weight"] * int(multiplicity[identities.index(r["peptide_sequence"])])
            for r in primary
        ]
        totals = {
            o: math.fsum(
                w for r, w in zip(primary, weight, strict=True) if r["organ_label"] == o
            )
            for o in ("Brain", "Liver", "Spleen")
        }
        if min(totals.values()) == 0:
            invalid += 1
            assert archived["valid"] == "False"
            continue
        assert archived["valid"] == "True"
        normalized = [
            w / (3 * totals[r["organ_label"]])
            for r, w in zip(primary, weight, strict=True)
        ]
        rmse = {}
        for arm in arms:
            rmse[arm] = math.sqrt(
                math.fsum(
                    w
                    * (
                        lookup[arm, r["assignment_id"]]["prediction"]
                        - r["target_change"]
                    )
                    ** 2
                    for r, w in zip(primary, normalized, strict=True)
                )
            )
            check(float(archived[arm]), rmse[arm])
            values[arm].append(rmse[arm])
        for arm in ("zero", "length"):
            gain = 1 - rmse["sequence"] / rmse[arm]
            key = "gain_vs_" + arm
            check(float(archived[key]), gain)
            values[key].append(gain)
    assert invalid == summary["bootstrap"]["n_invalid_draws"]
    for name, v in values.items():
        for a, b in zip(
            quantiles(v),
            summary["bootstrap"]["percentile_95_intervals"][name],
            strict=True,
        ):
            check(a, b)
    differences = json.loads((root / "results/exact-score-comparison.json").read_text())
    assert len(differences) == 12
    for d in differences:
        a, b = d["score_arm"], d["exact_arm"]
        delta = [
            lookup[a, r["assignment_id"]]["prediction"]
            - lookup[b, r["assignment_id"]]["prediction"]
            for r in primary
        ]
        check(d["score_minus_exact_rmse"], pooled[a]["rmse"] - pooled[b]["rmse"])
        check(
            d["score_vs_exact_prediction_rmse"],
            math.sqrt(
                math.fsum(
                    r["weight"] * v * v for r, v in zip(primary, delta, strict=True)
                )
            ),
        )
        check(d["max_prediction_difference"], max(abs(v) for v in delta))
    result = {
        "status": "passed",
        "n_predictions": 8925,
        "n_stratum_metric_records": len(metrics),
        "n_bootstrap_draws": 1000,
        "n_invalid_draws": invalid,
        "n_exact_score_pairs": 12,
        "quality_passed": summary["quality_passed"],
        "independent_rank_and_weighted_metric_replay": True,
        "bootstrap_rng_weights_and_intervals_replayed": True,
        "csv_parquet_numeric_equality": True,
    }
    with (root / "results/score-independent-replay.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
