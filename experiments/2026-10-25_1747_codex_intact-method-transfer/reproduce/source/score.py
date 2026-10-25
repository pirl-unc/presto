"""Score already frozen predictions with the preregistered balanced endpoint."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    y = np.array([r["target_change"] for r in rows])
    p = np.array([r["prediction"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    w = w / w.sum()
    variance = float(w @ (y - w @ y) ** 2)
    prediction_variance = float(w @ (p - w @ p) ** 2)
    mse = float(w @ (p - y) ** 2)
    return {
        "n_observations": len(rows),
        "n_sequences": len({r["peptide_sequence"] for r in rows}),
        "loss_mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(w @ np.abs(p - y)),
        "r2": 1 - mse / variance if variance > 0 else None,
        "pearson": float(pearsonr(y, p).statistic)
        if np.ptp(y) > 0 and np.ptp(p) > 0
        else None,
        "spearman": float(spearmanr(y, p).statistic)
        if np.ptp(y) > 0 and np.ptp(p) > 0
        else None,
        "target_sd": math.sqrt(variance),
        "prediction_sd": math.sqrt(prediction_variance),
        "target_sd_unweighted": float(y.std()),
        "prediction_sd_unweighted": float(p.std()),
        "mean_target": float(w @ y),
        "mean_prediction": float(w @ p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    request = json.loads(parser.parse_args().requests.read_text())
    assert (
        request["sd_contract"]
        == "population SD under registered observation weights; unweighted SD also reported"
    )
    root = Path(__file__).resolve().parents[1]
    verification = json.loads(
        (root / "results/prediction-verification.json").read_text()
    )
    assert (
        verification["status"] == "passed"
        and verification["future_outcomes_read"] is False
    )
    pred_path = root / "results/frozen-predictions.parquet"
    assert (
        hashlib.sha256(pred_path.read_bytes()).hexdigest()
        == verification["prediction_sha256"]
    )
    package = Path(
        json.loads((root / "results/admission.json").read_text())["recipe_directory"]
    )
    labels = {
        r["assignment_id"]: r
        for r in pq.read_table(package / "typed/examples.parquet").to_pylist()
    }
    predictions = pq.read_table(pred_path).to_pylist()
    joined = []
    for p in predictions:
        row = labels[p["assignment_id"]]
        for k in (
            "peptide_sequence",
            "organ_label",
            "time_minutes",
            "split",
            "weight",
            "baseline_log10_ratio",
            "n_baseline_observations",
        ):
            assert p[k] == row[k]
        joined.append(
            {
                **p,
                "target_change": row["target_change"],
                "target_log10_ratio": row["target_log10_ratio"],
                "measurement_id": row["measurement_id"],
                "normalizer_measurement_id": row["normalizer_measurement_id"],
            }
        )
    arms = sorted({r["arm"] for r in joined})
    assert len(arms) == 21 and len(joined) == 425 * 21
    result = []
    for arm in arms:
        for split in ("test", "extrapolation"):
            chosen = [r for r in joined if r["arm"] == arm and r["split"] == split]
            assert len(chosen) == (236 if split == "test" else 189)
            result.append(
                {"arm": arm, "split": split, "stratum": "pooled", **metrics(chosen)}
            )
            for organ in ("Brain", "Liver", "Spleen"):
                subset = [r for r in chosen if r["organ_label"] == organ]
                result.append(
                    {
                        "arm": arm,
                        "split": split,
                        "stratum": "organ:" + organ,
                        **metrics(subset),
                    }
                )
            for minute in sorted({r["time_minutes"] for r in chosen}):
                subset = [r for r in chosen if r["time_minutes"] == minute]
                result.append(
                    {
                        "arm": arm,
                        "split": split,
                        "stratum": "time:" + str(minute),
                        **metrics(subset),
                    }
                )
    primary = {
        r["arm"]: r for r in result if r["split"] == "test" and r["stratum"] == "pooled"
    }
    seq = primary["sequence"]
    length = primary["length"]
    zero = primary["zero"]
    gains = {
        "versus_zero": 1 - seq["rmse"] / zero["rmse"],
        "versus_length": 1 - seq["rmse"] / length["rmse"],
    }
    organ_ratios = {}
    for organ in ("Brain", "Liver", "Spleen"):
        r = {
            x["arm"]: x
            for x in result
            if x["split"] == "test" and x["stratum"] == "organ:" + organ
        }
        organ_ratios[organ] = r["sequence"]["rmse"] / r["length"]["rmse"]
    gates = {
        "rmse_gain_versus_zero_at_least_5pct": gains["versus_zero"] >= 0.05,
        "rmse_gain_versus_length_at_least_5pct": gains["versus_length"] >= 0.05,
        "positive_pearson": seq["pearson"] is not None and seq["pearson"] > 0,
        "positive_spearman": seq["spearman"] is not None and seq["spearman"] > 0,
        "prediction_sd_at_least_0_1_target_sd": seq["prediction_sd"]
        >= 0.1 * seq["target_sd"],
        "no_organ_more_than_10pct_worse_than_length": all(
            v <= 1.10 for v in organ_ratios.values()
        ),
    }
    matrix_rows = sorted(
        [r for r in joined if r["arm"] == "zero" and r["split"] == "test"],
        key=lambda r: r["assignment_id"],
    )
    ids = [r["assignment_id"] for r in matrix_rows]
    lookup = {(r["arm"], r["assignment_id"]): r for r in joined}
    errors = np.array(
        [
            [
                (lookup[arm, i]["prediction"] - lookup[arm, i]["target_change"]) ** 2
                for arm in arms
            ]
            for i in ids
        ]
    )
    weights = np.array([r["weight"] for r in matrix_rows])
    sequences = sorted({r["peptide_sequence"] for r in matrix_rows})
    indices = np.array([sequences.index(r["peptide_sequence"]) for r in matrix_rows])
    organs = np.array([r["organ_label"] for r in matrix_rows])
    generator = np.random.default_rng(31415)
    boot = []
    multiplicities = []
    invalid = 0
    for draw in range(1000):
        counts = np.bincount(
            generator.integers(0, len(sequences), size=len(sequences)),
            minlength=len(sequences),
        )
        w = weights * counts[indices]
        missing = any(
            w[organs == organ].sum() == 0 for organ in ("Brain", "Liver", "Spleen")
        )
        multiplicities.append(counts)
        if missing:
            invalid += 1
            boot.append(
                {
                    "draw": draw,
                    "valid": False,
                    **{arm: None for arm in arms},
                    "gain_vs_zero": None,
                    "gain_vs_length": None,
                }
            )
            continue
        for organ in ("Brain", "Liver", "Spleen"):
            mask = organs == organ
            w[mask] /= 3 * w[mask].sum()
        rmse = np.sqrt(w @ errors)
        values = dict(zip(arms, rmse.tolist(), strict=True))
        boot.append(
            {
                "draw": draw,
                "valid": True,
                **values,
                "gain_vs_zero": 1 - values["sequence"] / values["zero"],
                "gain_vs_length": 1 - values["sequence"] / values["length"],
            }
        )
    intervals = {
        name: np.quantile(
            [b[name] for b in boot if b["valid"]], [0.025, 0.975]
        ).tolist()
        for name in (*arms, "gain_vs_zero", "gain_vs_length")
    }
    differences = []
    for kind in ("uniform", "residue"):
        exact = "original_" + kind + "-exact-0-42"
        for draws in (16, 64):
            for seed in (42, 43, 44):
                arm = f"original_{kind}-score-{draws}-{seed}"
                delta = np.array(
                    [
                        lookup[arm, i]["prediction"] - lookup[exact, i]["prediction"]
                        for i in ids
                    ]
                )
                differences.append(
                    {
                        "kind": kind,
                        "draws": draws,
                        "seed": seed,
                        "score_arm": arm,
                        "exact_arm": exact,
                        "score_minus_exact_rmse": primary[arm]["rmse"]
                        - primary[exact]["rmse"],
                        "score_vs_exact_prediction_rmse": float(
                            np.sqrt(weights @ delta**2)
                        ),
                        "max_prediction_difference": float(np.abs(delta).max()),
                    }
                )
    write_csv(root / "results/predictions.csv", joined)
    pq.write_table(pa.Table.from_pylist(joined), root / "results/predictions.parquet")
    write_csv(root / "results/metrics.csv", result)
    write_json(root / "results/metrics.json", result)
    write_csv(root / "results/bootstrap.csv", boot)
    np.savez_compressed(
        root / "results/bootstrap-samples.npz",
        counts=np.array(multiplicities),
        sequences=np.array(sequences),
    )
    write_json(root / "results/exact-score-comparison.json", differences)
    summary = {
        "status": "scored",
        "primary_times_minutes": [30, 60],
        "secondary_times_minutes": [90, 120],
        "n_primary_cells": 236,
        "n_secondary_cells": 189,
        "preselected_arm": "sequence",
        "gains": gains,
        "organ_sequence_to_length_rmse_ratios": organ_ratios,
        "gates": gates,
        "quality_passed": all(gates.values()),
        "bootstrap": {
            "draws": 1000,
            "seed": 31415,
            "unit": "peptide identity jointly across organs; equal-organ renormalization",
            "n_invalid_draws": invalid,
            "percentile_95_intervals": intervals,
        },
        "sd_contract": request["sd_contract"],
        "decision": "retain only as relative temporal comparator; concentration/APC validity unproven"
        if all(gates.values())
        else "close failed transfer; no F3 refitting, feature search or secondary-time rescue",
        "new_fits": 0,
    }
    write_json(root / "results/score-summary.json", summary)
    print(json.dumps({"primary": primary, "summary": summary}, allow_nan=False))


if __name__ == "__main__":
    main()
