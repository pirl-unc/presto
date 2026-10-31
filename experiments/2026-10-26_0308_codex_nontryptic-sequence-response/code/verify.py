"""Independent source reconciliation, augmented-LS fits, prediction/metric replay."""

from collections import Counter
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import pearsonr, spearmanr

from common import ROOT, ids_digest, read, request, write

AA = "ACDEFGHIKLMNPQRSTVWY"


def raw_features(s):
    counts = Counter(s)
    return [counts[a] / len(s) for a in AA] + [len(s)]


def independently_fit(model, records):
    raw = [raw_features(s) for s in sorted({r["sequence"] for r in records})]
    means = [statistics.mean(x) for x in zip(*raw, strict=True)]
    scales = [statistics.pstdev(x) for x in zip(*raw, strict=True)]
    active = [i for i, sd in enumerate(scales) if sd > 1e-12] if model["arm"] == "sequence" else []
    assert active == model["active_features"]
    assert np.allclose([means[i] for i in active], model["feature_means"])
    assert np.allclose([scales[i] for i in active], model["feature_scales"])
    seqs = {r["sequence"] for r in records}
    slots = Counter((r["sequence"], r["x"]) for r in records)
    n_doses = Counter(s for s, x in slots)
    design, targets = [], []
    for r in records:
        weight = 1 / (len(seqs) * n_doses[r["sequence"]] * slots[(r["sequence"], r["x"])])
        features = [1.0] + ([] if model["arm"] == "unit" else [r["x"]])
        features += [(raw_features(r["sequence"])[i] - means[i]) / scales[i] for i in active]
        design.append([math.sqrt(weight) * f for f in features])
        targets.append(math.sqrt(weight) * (r["y"] - r["x"] if model["arm"] == "unit" else r["y"]))
    if model["arm"] == "sequence":
        for i in range(2, len(design[0])):
            penalty = [0.0] * len(design[0])
            penalty[i] = math.sqrt(model["alpha"])
            design.append(penalty)
            targets.append(0.0)
    coefficients, _, rank, _ = np.linalg.lstsq(np.asarray(design), np.asarray(targets), rcond=None)
    assert rank == model["n_parameters"]
    assert np.allclose(coefficients, model["coefficients"], rtol=1e-8, atol=1e-9)
    return coefficients


def scalar_prediction(model, row, coefficients):
    features = [1.0] + ([] if model["arm"] == "unit" else [row["x"]])
    raw = raw_features(row["sequence"])
    features += [
        (raw[i] - m) / s
        for i, m, s in zip(
            model["active_features"], model["feature_means"], model["feature_scales"], strict=True
        )
    ]
    return sum(float(c) * x for c, x in zip(coefficients, features, strict=True)) + (
        row["x"] if model["arm"] == "unit" else 0
    )


def scalar_metrics(records):
    y = [r["y"] for r in records]
    p = [r["prediction"] for r in records]
    error = [a - b for a, b in zip(p, y, strict=True)]
    seqs = {r["sequence"] for r in records}
    per_sequence = [
        [e for r, e in zip(records, error, strict=True) if r["sequence"] == s] for s in seqs
    ]
    eq = statistics.mean(statistics.mean(e * e for e in group) for group in per_sequence)

    def corr(a, b, rank=False):
        if statistics.pstdev(a) < 1e-12 or statistics.pstdev(b) < 1e-12:
            return None
        return float((spearmanr if rank else pearsonr)(a, b).statistic)

    adjusted_y = [r["y"] - r["x"] for r in records]
    adjusted_p = [r["prediction"] - r["x"] for r in records]
    yc, pc = [], []
    for s in sorted(seqs):
        group = [r for r in records if r["sequence"] == s]
        ym = statistics.mean(r["y"] for r in group)
        pm = statistics.mean(r["prediction"] for r in group)
        yc.extend(r["y"] - ym for r in group)
        pc.extend(r["prediction"] - pm for r in group)
    return {
        "n_rows": len(records),
        "n_peptides": len(seqs),
        "pooled_mse": statistics.mean(e * e for e in error),
        "pooled_rmse": math.sqrt(statistics.mean(e * e for e in error)),
        "pooled_mae": statistics.mean(abs(e) for e in error),
        "equal_peptide_mse": eq,
        "equal_peptide_rmse": math.sqrt(eq),
        "equal_peptide_mae": statistics.mean(
            statistics.mean(abs(e) for e in group) for group in per_sequence
        ),
        "pearson": corr(y, p),
        "spearman": corr(y, p, True),
        "target_sd": statistics.pstdev(y),
        "prediction_sd": statistics.pstdev(p),
        "dose_adjusted_target_sd": statistics.pstdev(adjusted_y),
        "dose_adjusted_prediction_sd": statistics.pstdev(adjusted_p),
        "dose_adjusted_pearson": corr(adjusted_y, adjusted_p),
        "within_peptide_target_sd": statistics.pstdev(yc),
        "within_peptide_prediction_sd": statistics.pstdev(pc),
        "within_peptide_rmse": math.sqrt(
            statistics.mean((a - b) ** 2 for a, b in zip(yc, pc, strict=True))
        ),
        "within_peptide_pearson": corr(yc, pc),
    }


def assert_metrics(actual, expected):
    for key, value in actual.items():
        if value is None:
            assert expected[key] is None, key
        else:
            assert math.isclose(value, expected[key], rel_tol=1e-8, abs_tol=1e-9), (
                key,
                value,
                expected[key],
            )


def main():
    req = request()
    admission, fitted = Path(req["admission_directory"]), Path(req["fit_directory"])
    rows = read(admission / "admitted.json")
    by_id = {r["measurement_id"]: r for r in rows}
    source = {(r["sheet"], r["source_cell"]): r for r in read(req["qualified_s3_cells"])}
    decisions = read(admission / "decisions.json")
    for r in decisions:
        sheet, cell = r["source_locator"].split("#")[1].split("!")
        original = source[(sheet, cell)]
        assert r["quantitative_value"] == original["raw_signal"]
        assert r["value_status"] == original["signal_status"]
        assert r["standard.peptide_sequence"] == original["sequence"]
        assert r["amount.nominal_amount_fmol"] == original["added_fmol"]
        assert r["slot.source_slot_ordinal"] == original["replicate_column_ordinal"]
        assert r["slot.acquisition_id"] is None
    for r in rows:
        assert r["x"] == math.log10(r["amount.nominal_amount_fmol"])
        assert r["y"] == math.log10(r["quantitative_value"])
        assert "DatasetS3A!" in r["source_locator"]
    models = read(fitted / "models.json")
    replayed = {}
    max_coefficient_error = 0.0
    for m in models:
        train = [
            r
            for r in rows
            if r["gene"] != m["outer_gene"]
            and (m["level"] != "inner" or r["gene"] != m["inner_gene"])
        ]
        assert sorted(r["measurement_id"] for r in train) == m["training_ids"]
        assert ids_digest(m["training_ids"]) == m["training_ids_sha256"]
        assert sorted({r["gene"] for r in train}) == m["training_genes"]
        assert sorted({r["sequence"] for r in train}) == m["training_sequences"]
        coefficients = independently_fit(m, train)
        max_coefficient_error = max(
            max_coefficient_error, float(np.max(np.abs(coefficients - m["coefficients"])))
        )
        replayed[m["model_id"]] = coefficients
    inner, outer = read(fitted / "inner-predictions.json"), read(fitted / "outer-predictions.json")
    max_prediction_error = 0.0
    for pred in inner + outer:
        m = models[pred["model_id"]]
        row = by_id[pred["measurement_id"]]
        expected_gene = m["inner_gene"] if m["level"] == "inner" else m["outer_gene"]
        assert row["gene"] == expected_gene
        assert row["measurement_id"] not in m["training_ids"]
        assert pred.get("target", pred.get("y")) == row["y"]
        value = scalar_prediction(m, row, replayed[m["model_id"]])
        error = abs(value - pred["prediction"])
        assert math.isclose(value, pred["prediction"], rel_tol=1e-8, abs_tol=1e-9)
        max_prediction_error = max(max_prediction_error, error)
    for s in read(fitted / "selections.json"):
        losses = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            preds = [p for p in inner if p["outer_gene"] == s["outer_gene"] and p["alpha"] == alpha]
            expected_ids = {r["measurement_id"] for r in rows if r["gene"] != s["outer_gene"]}
            assert (
                len(preds) == len(expected_ids)
                and {p["measurement_id"] for p in preds} == expected_ids
            )
            records = [{**by_id[p["measurement_id"]], "prediction": p["prediction"]} for p in preds]
            losses[alpha] = scalar_metrics(records)["equal_peptide_mse"]
            assert math.isclose(
                losses[alpha], s["inner_losses"][str(alpha)], rel_tol=1e-8, abs_tol=1e-9
            )
        minimum = min(losses.values())
        chosen = max(
            a for a, v in losses.items() if math.isclose(v, minimum, rel_tol=1e-12, abs_tol=1e-12)
        )
        assert s["chosen_alpha"] == chosen
    summary = read(fitted / "summary.json")
    for arm in ("unit", "slope", "sequence"):
        records = [r for r in outer if r["arm"] == arm]
        assert len(records) == len(rows) and {r["measurement_id"] for r in records} == set(by_id)
        assert_metrics(scalar_metrics(records), summary["arms"][arm])
        for field in ("sequence", "dose_fmol", "gene"):
            for value in {r[field] for r in records}:
                subset = [r for r in records if r[field] == value]
                assert_metrics(
                    scalar_metrics(subset), summary["arms"][arm]["per_" + field][str(value)]
                )
    independent_arms = {
        arm: scalar_metrics([r for r in outer if r["arm"] == arm])
        for arm in ("unit", "slope", "sequence")
    }
    sequence, baseline = independent_arms["sequence"], independent_arms["slope"]
    improvement = 1 - sequence["equal_peptide_rmse"] / baseline["equal_peptide_rmse"]
    gates = {
        "rmse_reduction_at_least_10_percent": improvement >= 0.1,
        "all_outer_learned_dose_slopes_positive": all(
            m["dose_slope"] > 0 for m in models if m["level"] == "outer"
        ),
        "dose_adjusted_sd_at_least_25_percent": sequence["dose_adjusted_prediction_sd"]
        >= 0.25 * sequence["dose_adjusted_target_sd"],
    }
    assert summary["gates"] == gates and summary["passed"] == all(gates.values())
    assert math.isclose(summary["relative_rmse_improvement"], improvement, abs_tol=1e-10)
    assert summary["n_fits"] == len(models)
    assert summary["n_inner_predictions"] == len(inner)
    assert summary["n_outer_predictions"] == len(outer)
    result = {
        "status": "independently_verified",
        "n_source_decisions": len(decisions),
        "n_fits": len(models),
        "n_predictions": len(inner) + len(outer),
        "max_coefficient_error": max_coefficient_error,
        "max_prediction_error": max_prediction_error,
        "solver": "independent_augmented_lstsq",
        "no_independent_biological_test": True,
        "old_biological_heldouts_accessed": False,
    }
    write(ROOT / "results" / req["result_file"], result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
