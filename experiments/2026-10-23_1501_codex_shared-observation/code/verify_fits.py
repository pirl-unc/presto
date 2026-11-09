"""Independent NumPy source/features/fragment-law/metric and optimizer replay."""

import argparse
import gzip
import hashlib
import itertools
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
LN10 = math.log(10)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, tolerance=2e-9):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    assert error <= tolerance, (error, tolerance)
    return error


def centered(x, w):
    return x - np.sum(w * x) / np.sum(w)


def inputs():
    cfg = read(FAMILY / "reproduce/launch.json")
    recipe = Path(cfg["recipe_directory"])
    assert sha(recipe / "manifest.json") == cfg["recipe_manifest_sha256"]
    manifest = read(recipe / "manifest.json")
    assert sha(recipe / "examples.json") == manifest["files"]["examples.json"]["sha256"]
    original = [
        e for e in read(recipe / "examples.json") if e["split"] in ("train", "val")
    ]
    by_condition = defaultdict(list)
    for e in original:
        by_condition[e["condition_id"]].append(e)
    context = read(FAMILY / "results/context-summary.json")
    feature = read(FAMILY / "results/response-feature-basis.json")
    scale, basis = np.asarray(feature["scale"]), np.asarray(feature["basis"])
    groups = []
    for condition, examples in sorted(by_condition.items()):
        e0 = examples[0]
        sequence = e0["parent_sequence"]
        instrument = [
            key.split(":", 1)[1]
            for key in context["per_condition"][condition]
            if key.startswith("instrument:")
        ]
        assert len(instrument) == 1
        instrument = instrument[0]
        protocol = e0["context"]
        enzyme = (
            protocol["substrate_concentration_um"]
            / protocol["enzyme_substrate_denominator"]
        )
        x = np.zeros((len(sequence) - 1, 40))
        x[:, 0], x[:, -1] = 1, math.log(enzyme / 0.2)
        for i, (left, right) in enumerate(zip(sequence[:-1], sequence[1:])):
            if left != "A":
                x[i, AA.index(left)] = 1
            if right != "A":
                x[i, 19 + AA.index(right)] = 1
        rows = []
        for e in examples:
            assert e["parent_sequence"] == sequence
            obs = {o["time_hours"]: o["target_value"] for o in e["observations"]}
            reference = e["common_positive_times"][0]
            for t in e["common_positive_times"]:
                assert obs[t] > 0
                rows.append(
                    dict(
                        curve_id=e["curve_id"],
                        condition_id=condition,
                        construct_id=e["construct_id"],
                        split=e["split"],
                        peptide_sequence=e["peptide_sequence"],
                        start=e["start_0based"],
                        end=e["end_0based"],
                        time=t,
                        reference_time=reference,
                        signal=obs[t],
                        reference_signal=obs[reference],
                        target=math.log10(obs[t]),
                        n_times=len(e["common_positive_times"]),
                        biological_replicate_raw=e["biological_replicate_raw"],
                        instrument=instrument,
                    )
                )
        raw = np.zeros((len(rows), 40))
        offset = feature["instruments"].index(instrument) * 20
        for i, r in enumerate(rows):
            peptide = r["peptide_sequence"]
            raw[i, offset] = len(peptide) / 40
            for j, aa in enumerate(AA[1:], 1):
                raw[i, offset + j] = peptide.count(aa) / len(peptide)
        groups.append(
            dict(
                condition=condition,
                parent=e0["construct_id"],
                split=e0["split"],
                sequence=sequence,
                rows=rows,
                ncurves=len(examples),
                x=x,
                raw=raw,
                z=raw / scale @ basis,
                a=np.array([r["start"] for r in rows]),
                b=np.array([r["end"] for r in rows]),
                t=np.array([r["time"] for r in rows]),
                y=np.array([r["target"] for r in rows]),
            )
        )
    gram = np.zeros((40, 40))
    for split in ("train", "val"):
        subset = [g for g in groups if g["split"] == split]
        parents = Counter(g["parent"] for g in subset)
        for g in subset:
            g["w"] = np.array(
                [
                    1
                    / (
                        len(parents)
                        * parents[g["parent"]]
                        * g["ncurves"]
                        * r["n_times"]
                    )
                    for r in g["rows"]
                ]
            )
            if split == "train":
                zc = g["raw"] - (g["w"][:, None] * g["raw"]).sum(0) / g["w"].sum()
                gram += zc.T @ (g["w"][:, None] * zc)
    rms = np.sqrt(np.maximum(np.diag(gram), 0))
    close(np.where(rms > 1e-12, rms, 1), scale)
    scaled = gram / scale[:, None] / scale[None, :]
    eigenvalues = np.linalg.eigvalsh(scaled)
    close(eigenvalues, feature["eigenvalues"])
    assert (eigenvalues > eigenvalues[-1] * 1e-10).sum() == 38
    close(basis.T @ basis, np.eye(38))
    close(scaled @ basis, basis * eigenvalues[-38:])
    assert [
        len([r for g in groups if g["split"] == s for r in g["rows"]])
        for s in ("train", "val")
    ] == [90630, 14822]
    return groups


def amount(theta, g):
    rate = np.exp(g["x"] @ theta)
    sums = np.r_[0, np.cumsum(rate)]
    weighted = np.vstack((np.zeros((1, 40)), np.cumsum(rate[:, None] * g["x"], axis=0)))
    a, b, t = g["a"], g["b"], g["t"]
    value = -t * (sums[b - 1] - sums[a])
    jac = -t[:, None] * (weighted[b - 1] - weighted[a])
    for position, mask in ((a - 1, a > 0), (b - 1, b <= len(rate))):
        h = t[mask] * rate[position[mask]]
        value[mask] += np.log(-np.expm1(-h))
        derivative = h * np.exp(-h) / (-np.expm1(-h))
        jac[mask] += derivative[:, None] * g["x"][position[mask]]
    return value / LN10, jac / LN10


def objective(theta, gamma, groups):
    grad = np.zeros(len(theta) + len(gamma))
    mse = 0.0
    for g in groups:
        if g["split"] != "train":
            continue
        values, jac = amount(theta, g)
        if len(gamma):
            values += g["z"] @ gamma
            jac = np.hstack((jac, g["z"]))
        residual = centered(values - g["y"], g["w"])
        mse += g["w"] @ residual**2
        grad += 2 * (g["w"] * residual) @ jac
    reg = np.r_[0, theta[1:], gamma]
    return mse + 0.005 * (reg @ reg), mse, grad + 0.01 * reg


def outputs(name, fit, groups, initial):
    result = []
    for g in groups:
        mass = response = None
        if name in (
            "constant_response",
            "sequence_response",
            "conditional_sequence_response",
        ):
            mass, _ = amount(np.array(fit["theta"]), g)
            response = (
                g["z"] @ np.array(fit["gamma"]) if fit["gamma"] else np.zeros_like(mass)
            )
            values = mass + response
        elif name.startswith("fixed_ratio"):
            mass, _ = amount(initial, g)
            response = (
                g["z"] @ np.array(fit["coefficients"])
                if name.endswith("with_response")
                else np.zeros_like(mass)
            )
            values = mass + response
        elif name == "response_only":
            values = response = g["z"] @ np.array(fit["coefficients"])
        elif name == "time_only":
            time_features = np.column_stack((np.log10(g["t"]), np.log10(g["t"]) ** 2))
            values = response = time_features @ np.array(fit["coefficients"])
        elif name == "constant_signal":
            values = np.zeros_like(g["y"])
        else:
            raise ValueError(name)
        result.append(
            dict(
                g=g,
                values=values,
                centered=centered(values, g["w"]),
                target=centered(g["y"], g["w"]),
                offset=g["w"] @ (g["y"] - values) / g["w"].sum(),
                mass=mass,
                response=response,
            )
        )
    return result


def independent_metrics(results, split):
    subset = [r for r in results if r["g"]["split"] == split]
    p = np.concatenate([r["centered"] for r in subset])
    y = np.concatenate([r["target"] for r in subset])
    w = np.concatenate([r["g"]["w"] for r in subset])
    ratios_p, ratios_y, weights = [], [], []
    for r in subset:
        curves = defaultdict(list)
        for i, row in enumerate(r["g"]["rows"]):
            curves[row["curve_id"]].append(i)
        for ids in curves.values():
            ref = ids[0]
            for i in ids[1:]:
                ratios_p.append(r["values"][i] - r["values"][ref])
                ratios_y.append(r["g"]["y"][i] - r["g"]["y"][ref])
                weights.append(r["g"]["w"][i] * len(ids) / (len(ids) - 1))
    rp, ry, rw = map(np.array, (ratios_p, ratios_y, weights))
    close(w.sum(), 1)
    close(rw.sum(), 1)

    def corr(x, y, fn):
        return (
            None
            if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12
            else float(fn(x, y).statistic)
        )

    masses = [r["mass"] for r in subset if r["mass"] is not None]
    return dict(
        n_observations=len(p),
        shape_mse=float(w @ (p - y) ** 2),
        shape_rmse=float(np.sqrt(w @ (p - y) ** 2)),
        shape_pearson=corr(p, y, pearsonr),
        shape_spearman=corr(p, y, spearmanr),
        prediction_sd=float(p.std()),
        target_sd=float(y.std()),
        ratio_rmse=float(np.sqrt(rw @ (rp - ry) ** 2)),
        ratio_pearson=corr(rp, ry, pearsonr),
        ratio_spearman=corr(rp, ry, spearmanr),
        minimum_log10_amount=float(min(x.min() for x in masses)) if masses else None,
        amount_below_1e_20_count=int(sum((x < -20).sum() for x in masses)),
    )


def verify_predictions(name, results, expected_metrics):
    maximum = 0.0
    count = 0
    with gzip.open(FAMILY / f"results/{name}-predictions.jsonl.gz", "rt") as stream:
        for r in results:
            g = r["g"]
            for i, expected in enumerate(g["rows"]):
                observed = json.loads(next(stream))
                for key, value in expected.items():
                    assert observed[key] == value, (name, key, observed[key], value)
                for key, value in dict(
                    weight=g["w"][i],
                    raw_prediction=r["values"][i],
                    centered_prediction=r["centered"][i],
                    centered_target=r["target"][i],
                    profiled_offset=r["offset"],
                    log10_amount=None if r["mass"] is None else r["mass"][i],
                    log10_response=None if r["response"] is None else r["response"][i],
                ).items():
                    if value is None:
                        assert observed[key] is None
                    else:
                        maximum = max(maximum, close(observed[key], value))
                count += 1
        assert next(stream, None) is None
    summaries = {s: independent_metrics(results, s) for s in ("train", "val")}
    rank_differences = {}
    for split, metrics in summaries.items():
        for key, value in metrics.items():
            stored = expected_metrics[split][key]
            if value is None:
                assert stored is None
            else:
                error = close(value, stored, 1e-5 if "spearman" in key else 2e-9)
                if "spearman" in key:
                    rank_differences[f"{split}:{key}"] = error
    return dict(
        count=count,
        maximum_numeric_error=maximum,
        independent_metrics=summaries,
        raw_rank_discrepancies=rank_differences,
    )


def conditional_gradient(theta, gamma, groups, rng_state):
    generator = torch.Generator()
    generator.set_state(torch.tensor(rng_state, dtype=torch.uint8))
    gradient = np.zeros(40)
    work = dict(partitions=0, bond_draws=0, presence_terms=0)
    for g in groups:
        if g["split"] != "train":
            continue
        full, jac = amount(theta, g)
        coefficient = (
            2 * g["w"] * centered(full + g["z"] @ gamma - g["y"], g["w"]) / LN10
        )
        rate = np.exp(g["x"] @ theta)
        for t in np.unique(g["t"]):
            mask = g["t"] == t
            spans, first, inverse = np.unique(
                np.column_stack((g["a"][mask], g["b"][mask])),
                axis=0,
                return_index=True,
                return_inverse=True,
            )
            combined = np.bincount(
                inverse, weights=coefficient[mask], minlength=len(spans)
            )
            mass = np.exp(full[mask][first] * LN10)
            scores = jac[mask][first] * LN10
            proposal = np.abs(combined) * np.linalg.norm(scores, axis=1)
            weights = (
                0.9 * proposal / proposal.sum() + 0.1 / len(proposal)
                if proposal.sum() > 0
                else np.full(len(proposal), 1 / len(proposal))
            )
            chosen = torch.multinomial(
                torch.from_numpy(weights), 512, replacement=True, generator=generator
            ).numpy()
            cuts = torch.rand(
                (512, len(rate)), dtype=torch.float64, generator=generator
            ).numpy() < -np.expm1(-rate * t)
            for i, j in enumerate(chosen):
                start, end = spans[j]
                cuts[i, start : end - 1] = False
                if start:
                    cuts[i, start - 1] = True
                if end <= len(rate):
                    cuts[i, end - 1] = True
            present = np.zeros((512, len(spans)), dtype=bool)
            prefix = np.column_stack(
                (np.zeros(512, dtype=int), np.cumsum(cuts, axis=1))
            )
            for j, (start, end) in enumerate(spans):
                present[:, j] = (prefix[:, end - 1] - prefix[:, start]) == 0
                if start:
                    present[:, j] &= cuts[:, start - 1]
                if end <= len(rate):
                    present[:, j] &= cuts[:, end - 1]
            assert present[np.arange(512), chosen].all()
            density = present @ (weights / mass)
            gradient += (
                (present / density[:, None]).mean(0) * combined / mass
            ) @ scores
            work["partitions"] += 512
            work["bond_draws"] += cuts.size
            work["presence_terms"] += present.size
    gradient[1:] += 0.01 * theta[1:]
    return gradient, work


def verify_updates(name, fit, groups):
    initial = np.r_[fit["initial_theta"], fit["initial_gamma"]]
    m = np.zeros_like(initial)
    v = np.zeros_like(initial)
    current = initial.copy()
    coverage = np.zeros(len(initial), dtype=bool)
    maxerr = 0.0
    selected = []
    with gzip.open(FAMILY / f"results/{name}-updates.jsonl.gz", "rt") as stream:
        for step, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row["step"] == step
            pre = np.array(row["preupdate_parameters"])
            grad = np.array(row["gradient"])
            maxerr = max(maxerr, close(current, pre, 2e-10))
            coverage |= grad != 0
            if step in (1, 100, 500, 1000, 2000):
                loss, mse, exact = objective(pre[:40], pre[40:], groups)
                close(loss, row["preupdate_loss"])
                close(mse, row["preupdate_mse"])
                if name.startswith("conditional"):
                    sampled, work = conditional_gradient(
                        pre[:40], pre[40:], groups, fit["preupdate_rngs"][str(step)]
                    )
                    exact[:40] = sampled
                    assert all(work[k] == row["work"][k] for k in work)
                error = close(grad, exact)
                selected.append(dict(step=step, gradient_error=error))
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            current = pre - 0.03 / (1 - 0.9**step) * m / (
                np.sqrt(v) / np.sqrt(1 - 0.999**step) + 1e-8
            )
            maxerr = max(maxerr, close(current, row["postupdate_parameters"], 2e-10))
    assert step == 2000
    close(current, np.r_[fit["theta"], fit["gamma"]], 2e-10)
    assert coverage.tolist() == fit["gradient_coverage"]
    assert int(np.count_nonzero(current - initial)) == fit["changed_parameter_count"]
    close(np.concatenate([s["exp_avg"] for s in fit["optimizer_states"]]), m)
    close(np.concatenate([s["exp_avg_sq"] for s in fit["optimizer_states"]]), v)
    loss, mse, gradient = objective(current[:40], current[40:], groups)
    close(loss, fit["final_loss"])
    close(mse, fit["final_mse"])
    close(gradient, fit["exact_final_gradient"])
    return dict(
        updates=step,
        maximum_adam_error=maxerr,
        replayed_gradients=selected,
        final_gradient_norm=float(np.linalg.norm(gradient)),
        coverage=int(coverage.sum()),
        parameters=len(current),
    )


def verify_toy():
    rate = np.array([0.07, 0.13, 0.21, 0.1, 0.15])
    t = 2.1
    h = rate * t
    probability = -np.expm1(-h)
    states = np.array(list(itertools.product([False, True], repeat=5)))
    p = np.prod(np.where(states, probability, 1 - probability), axis=1)
    spans = [(0, 3), (1, 4), (2, 6)]
    coefficient = np.array([0.3, -0.7, 0.4])
    present = np.array(
        [
            [
                not row[a : b - 1].any()
                and (not a or row[a - 1])
                and (b == 6 or row[b - 1])
                for a, b in spans
            ]
            for row in states
        ]
    )
    masses = p @ present
    fullscore = np.where(states, h * (1 - probability) / probability, -h)
    conditional = np.array(
        [
            (p[:, None] * present[:, i, None] * fullscore).sum(0) / masses[i]
            for i in range(3)
        ]
    )
    exact = coefficient @ conditional
    errors = []
    for saved in read(FAMILY / "results/conditional-toy.json"):
        weights = np.ones(3) / 3
        if saved["kind"] != "uniform":
            raw = np.abs(coefficient) * np.linalg.norm(conditional, axis=1)
            weights = 0.9 * raw / raw.sum() + 0.1 / 3
        density = present @ (weights / masses)
        q = p * density
        close(q, saved["q"])
        close(q.sum(), 1)
        valid = q > 0
        gradient = (
            (q[valid] / density[valid])[:, None]
            * present[valid]
            * (coefficient / masses)
        ).sum(0) @ conditional
        errors.append(close(gradient, exact, 1e-12))
    return dict(states=32, maximum_expectation_error=max(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exact", "conditional"], required=True)
    args = parser.parse_args()
    tick = time.perf_counter()
    torch.set_num_threads(1)
    cfg = read(
        FAMILY
        / (
            "reproduce/fitting.json"
            if args.mode == "exact"
            else "reproduce/conditional-fitting.json"
        )
    )
    for name, digest in cfg["source_hashes"].items():
        assert sha(FAMILY / name) == digest
    assert sha(cfg["initial_fits_path"]) == cfg["initial_fits_sha256"]
    groups = inputs()
    initial = np.array(
        read(cfg["initial_fits_path"])["sequence_context"]["coefficients"]
    )
    fits = (
        read(FAMILY / "results/fits.json")
        if args.mode == "exact"
        else {
            "conditional_sequence_response": read(
                FAMILY / "results/conditional_sequence_response-fit.json"
            )
        }
    )
    stored = (
        read(FAMILY / "results/metrics.json")
        if args.mode == "exact"
        else {
            "conditional_sequence_response": read(
                FAMILY / "results/conditional-fit-metrics.json"
            )
        }
    )
    records = {}
    for name, fit in fits.items():
        result = outputs(name, fit, groups, initial)
        record = verify_predictions(name, result, stored[name])
        if "exact_final_gradient" in fit:
            record["optimizer"] = verify_updates(name, fit, groups)
        if "solver" in fit:
            d = len(fit["coefficients"])
            gram = np.eye(d) * fit["penalty"] / 2
            rhs = np.zeros(d)
            for g in groups:
                if g["split"] != "train":
                    continue
                z = (
                    g["z"]
                    if fit["kind"] == "response"
                    else np.column_stack((np.log10(g["t"]), np.log10(g["t"]) ** 2))
                )
                z = z - (g["w"][:, None] * z).sum(0) / g["w"].sum()
                base = (
                    amount(initial, g)[0]
                    if name.startswith("fixed")
                    else np.zeros(len(g["y"]))
                )
                y = centered(g["y"] - base, g["w"])
                gram += z.T @ (g["w"][:, None] * z)
                rhs += z.T @ (g["w"] * y)
            record["ridge_error"] = close(
                np.linalg.solve(gram, rhs), fit["coefficients"]
            )
        records[name] = record
        print(
            name, record.get("optimizer"), record["maximum_numeric_error"], flush=True
        )
    sensitivities = read(
        FAMILY
        / (
            "results/scale-sensitivity.json"
            if args.mode == "exact"
            else "results/conditional-scale-sensitivity.json"
        )
    )
    for row in sensitivities:
        fit = fits[row["variant"]]
        theta = np.array(fit["theta"])
        theta[0] += row["log10_rate_scale_shift"] * LN10
        loss, mse, _ = objective(theta, np.array(fit["gamma"]), groups)
        close(loss, row["total_loss"], 2e-8)
        close(mse, row["profile_mse"], 2e-8)
        masses = np.concatenate(
            [amount(theta, g)[0] for g in groups if g["split"] == "train"]
        )
        close(masses.min(), row["minimum_log10_amount"])
        close(
            np.partition(masses, (len(masses) - 1) // 2)[(len(masses) - 1) // 2],
            row["median_log10_amount"],
        )
        assert int((masses < -20).sum()) == row["below_1e_20_count"]
    summary = dict(
        mode=args.mode,
        all_checks_pass=True,
        variants=records,
        conditional_toy=verify_toy(),
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="All final predictions/metrics/gradients, source inputs and weights, ridge controls, every Adam arithmetic update, five selected gradient replays per trained variant; independent candidate form replay and every stochastic gradient not claimed",
    )
    (FAMILY / f"results/independent-{args.mode}-verification.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print("VERIFIED", args.mode, summary["seconds"], flush=True)


if __name__ == "__main__":
    main()
