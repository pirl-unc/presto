"""Independent direct-sum forwards and finite-difference checks of capacity fits."""

import gzip
import json
import math
import resource
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

from source import FAMILY, read_groups


def main():
    tick = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    groups = read_groups(cfg)
    reference = np.array(
        json.loads(
            (
                __import__("pathlib").Path(cfg["reference_family"])
                / "results/fits.json"
            ).read_text()
        )["sequence_context"]["coefficients"]
    )
    allocations = json.loads((FAMILY / "results/rate-allocation.json").read_text())
    fits = json.loads((FAMILY / "results/capacity-fits.json").read_text())
    aa = "ACDEFGHIKLMNPQRSTVWY"
    offset = 0
    for g, allocation in zip(groups, allocations):
        assert g["condition"] == allocation["condition"]
        features = np.array(
            [
                [1.0]
                + [
                    float(pair[side] == residue)
                    for side in (0, 1)
                    for residue in aa[1:]
                ]
                + [g["x"][0, -1]]
                for pair in zip(g["sequence"][:-1], g["sequence"][1:])
            ]
        )
        assert np.array_equal(features, g["x"])
        g["base"] = features @ reference
        spans, inv = np.unique(
            np.column_stack((g["a"], g["b"])), axis=0, return_inverse=True
        )
        g["spans"] = spans
        g["inverse"] = inv
        mapping = np.full(len(g["base"]), -1, dtype=int)
        seen = set()
        for group in allocation["members"]:
            assert not (seen & set(group))
            seen.update(group)
            mapping[group] = offset
            offset += 1
            role = np.array(
                [
                    [
                        (
                            1
                            if a <= bond < b - 1
                            else 2
                            if a > 0 and bond == a - 1
                            else 3
                            if b < len(g["sequence"]) and bond == b - 1
                            else 0
                        )
                        for a, b in spans
                    ]
                    for bond in group
                ]
            )
            assert (
                np.array_equal(role, np.repeat(role[:1], len(group), axis=0))
                and role.any()
            )
        unsupported = set(allocation["unsupported_bonds"])
        assert seen | unsupported == set(range(len(mapping))) and not (
            seen & unsupported
        )
        for bond in unsupported:
            assert all(
                not (
                    a <= bond < b - 1
                    or (a > 0 and bond == a - 1)
                    or (b < len(g["sequence"]) and bond == b - 1)
                )
                for a, b in spans
            )
        g["mapping"] = mapping
    assert offset == fits["free_condition_rates"]["parameter_count"]

    def evaluate(theta, variant):
        predictions = []
        objective = 0.0
        for g in groups:
            logs = (
                g["x"] @ theta
                if variant == "unregularized_sequence"
                else np.where(
                    g["mapping"] >= 0, theta[np.maximum(0, g["mapping"])], g["base"]
                )
            )
            rates = np.exp(logs)
            internal = np.array([math.fsum(rates[a : b - 1]) for a, b in g["spans"]])[
                g["inverse"]
            ]
            cuts = []
            for index, valid in (
                (g["a"] - 1, g["a"] > 0),
                (g["b"] - 1, g["b"] < len(g["sequence"])),
            ):
                rate = rates[np.clip(index, 0, len(rates) - 1)]
                cuts.append(
                    np.where(
                        valid,
                        np.log(-np.expm1(-rate * g["t"]))
                        - np.log(-np.expm1(-rate * g["ref"])),
                        0.0,
                    )
                )
            p = (cuts[0] + cuts[1] - (g["t"] - g["ref"]) * internal) / math.log(10)
            predictions.append(p)
            objective += g["w"] @ ((p - g["y"]) ** 2)
        return objective, predictions

    results = {}
    maximum = 0.0
    count = 0
    rows = []
    for variant, fit in fits.items():
        theta = np.array(fit["coefficients"])
        objective, predictions = evaluate(theta, variant)
        assert abs(objective - fit["final_mse"]) < 1e-10
        indices = (
            np.arange(len(theta))
            if len(theta) <= 40
            else np.unique(
                np.r_[
                    np.linspace(0, len(theta) - 1, 32, dtype=int),
                    np.argsort(np.abs(fit["final_gradient"]))[-16:],
                ]
            )
        )
        errors = []
        for i in indices:
            hi = theta.copy()
            lo = theta.copy()
            hi[i] += 1e-5
            lo[i] -= 1e-5
            derivative = (evaluate(hi, variant)[0] - evaluate(lo, variant)[0]) / 2e-5
            error = abs(derivative - fit["final_gradient"][i])
            assert error < 1e-7
            errors.append(error)
        expected = {}
        for g, p in zip(groups, predictions):
            for source, value, y, w in zip(g["rows"], p, g["y"], g["w"]):
                expected[source[0]["curve_id"], source[1]] = (
                    source,
                    value,
                    y,
                    w,
                    g["parent"],
                )
        seen = set()
        byparent = {}
        allp = []
        ally = []
        allw = []
        with gzip.open(
            FAMILY / f"results/predictions-{variant}.jsonl.gz", "rt"
        ) as stream:
            for line in stream:
                r = json.loads(line)
                key = r["curve_id"], r["time"]
                assert key not in seen
                seen.add(key)
                source, p, y, w, parent = expected[key]
                curve, t, ref, signal, reference_signal = source
                error = abs(r["prediction"] - p)
                maximum = max(maximum, error)
                assert error < 1e-10
                assert (
                    abs(r["target"] - y) < 1e-12
                    and r["weight"] == w
                    and r["reference_time"] == ref
                )
                assert (
                    r["signal"] == signal
                    and r["reference_signal"] == reference_signal
                    and r["split"] == "train"
                )
                assert (
                    r["construct_id"] == curve["construct_id"]
                    and r["condition_id"] == curve["condition_id"]
                )
                byparent.setdefault(parent, []).append((p, y, w))
                allp.append(p)
                ally.append(y)
                allw.append(w)
                count += 1
                rows.append(
                    dict(
                        variant=variant,
                        curve_id=r["curve_id"],
                        time=t,
                        canonical_prediction=float(p),
                        raw_difference=float(p - r["prediction"]),
                    )
                )
        assert seen == set(expected)
        parent_scores = {}
        for parent, values in byparent.items():
            p, y, w = np.array(values).T
            parent_scores[parent] = dict(
                rmse=float(np.sqrt(w @ ((p - y) ** 2) / w.sum())), n=len(p)
            )
        metrics = dict(
            mse=float(objective),
            rmse=math.sqrt(objective),
            pearson=float(pearsonr(allp, ally).statistic),
            spearman=float(spearmanr(allp, ally).statistic),
            prediction_sd=float(np.std(allp)),
            by_parent=parent_scores,
        )
        results[variant] = dict(
            metrics=metrics,
            finite_difference_coordinates=indices.tolist(),
            maximum_gradient_error=max(errors),
            all_parameters_changed=fit["changed_parameter_count"]
            == fit["parameter_count"],
            converged=fit["polish"]["success"],
        )
        print(variant, results[variant]["metrics"]["rmse"], max(errors), flush=True)
    with gzip.open(
        FAMILY / "results/canonical-rank-predictions.jsonl.gz", "wt"
    ) as stream:
        for r in rows:
            stream.write(json.dumps(r, sort_keys=True) + "\n")
    result = dict(
        results=results,
        predictions_verified=count,
        maximum_prediction_error=maximum,
        allocation_parameters=offset,
        qualification="Independent direct-sum predictions, all allocation incidences, finite differences of all40sequence and up to48free-rate coordinates; no optimizer replay or global-optimum claim",
        no_heldout_evaluation=True,
        wall_seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/independent-fit-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
