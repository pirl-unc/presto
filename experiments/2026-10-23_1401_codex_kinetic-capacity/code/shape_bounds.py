"""Exact small convex projections onto necessary temporal shape constraints."""

import gzip
import itertools
import json
import resource
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.stats import pearsonr, spearmanr

from source import FAMILY, read_groups


def constraints(times, boundaries):
    m = len(times) - 1
    rows = []
    rhs = []
    for j in range(m):
        row = np.zeros(m)
        row[j] = 1
        if j:
            row[j - 1] = -1
        rows.append(row)
        rhs.append(boundaries * np.log10(times[j + 1] / times[j]))
    for j in range(m - 1):
        left = times[j + 1] - times[j]
        right = times[j + 2] - times[j + 1]
        row = np.zeros(m)
        row[j + 1] = 1 / right
        row[j] = -1 / right - 1 / left
        if j:
            row[j - 1] = 1 / left
        rows.append(row)
        rhs.append(0.0)
    return np.array(rows), np.array(rhs)


def project(means, weights, a, b):
    """Enumerate independent active sets; select feasible KKT solutions."""
    n, m = means.shape
    inv = 1 / weights
    candidates = []
    multipliers = []
    active_sets = []
    fixed_rows = []
    for size in range(m + 1):
        for active in itertools.combinations(range(len(b)), size):
            if size:
                ai = a[list(active)]
                bi = b[list(active)]
                matrix = (ai * inv) @ ai.T
                if np.linalg.matrix_rank(matrix) < size:
                    continue
                lam = (means @ ai.T - bi) @ np.linalg.inv(matrix)
                p = np.eye(m) - (inv[:, None] * ai.T) @ np.linalg.solve(matrix, ai)
                shift = (inv[:, None] * ai.T) @ np.linalg.solve(matrix, bi)
                fixed = []
                for column in range(m):
                    if (
                        np.linalg.matrix_rank(np.vstack((ai, np.eye(m)[column])))
                        == size
                    ):
                        p[column] = 0.0
                        fixed.append(column)
                z = means @ p.T + shift
                dual = np.zeros((n, len(b)))
                dual[:, list(active)] = lam
            else:
                z = means.copy()
                dual = np.zeros((n, len(b)))
                fixed = []
            feasible = np.max(z @ a.T - b, axis=1) < 1e-10
            feasible &= np.min(dual, axis=1) > -1e-10
            score = np.sum(weights * (z - means) ** 2, axis=1)
            candidates.append((z, np.where(feasible, score, np.inf)))
            multipliers.append(dual)
            active_sets.append(active)
            fixed_rows.append(fixed)
    scores = np.stack([c[1] for c in candidates], axis=1)
    chosen = np.argmin(scores, axis=1)
    assert np.isfinite(scores[np.arange(n), chosen]).all()
    z = np.stack([c[0] for c in candidates], axis=1)[np.arange(n), chosen]
    dual = np.stack(multipliers, axis=1)[np.arange(n), chosen]
    # KKT proof for the diagonal weighted least-squares problem.
    stationarity = weights * (z - means) + dual @ a
    assert np.max(np.abs(stationarity)) < 1e-9
    assert np.max(z @ a.T - b) < 1e-9 and np.min(dual) > -1e-9
    assert np.max(np.abs(dual * (z @ a.T - b))) < 1e-9
    return (
        z,
        dual,
        [active_sets[i] for i in chosen],
        dict(
            stationarity=float(np.max(np.abs(stationarity))),
            feasibility=float(max(0.0, np.max(z @ a.T - b))),
            complementarity=float(np.max(np.abs(dual * (z @ a.T - b)))),
        ),
    )


def main():
    tick = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    groups = read_groups(cfg)
    assert all(g["split"] == "train" for g in groups)
    assert not (FAMILY / "results/shape-metrics.json").exists()
    curves = []
    buckets = defaultdict(list)
    source_rows = []
    for g in groups:
        byspan = defaultdict(list)
        for i, (a, b) in enumerate(zip(g["a"], g["b"])):
            byspan[int(a), int(b)].append(i)
        for (start, end), indices in sorted(byspan.items()):
            assert 0 <= start < end <= len(g["sequence"]) and end - start < len(
                g["sequence"]
            )
            refs = set(g["ref"][indices])
            assert len(refs) == 1
            times = np.r_[next(iter(refs)), sorted(set(g["t"][indices]))]
            assert (np.diff(times) > 0).all()
            ys = g["y"][indices]
            ws = g["w"][indices]
            observed = g["t"][indices]
            weight = np.array([ws[observed == t].sum() for t in times[1:]])
            mean = np.array(
                [
                    ws[observed == t] @ ys[observed == t] / weight[j]
                    for j, t in enumerate(times[1:])
                ]
            )
            normalized = weight / weight.sum()
            boundaries = int(start > 0) + int(end < len(g["sequence"]))
            key = tuple(times), boundaries, tuple(normalized)
            row = dict(
                condition=g["condition"],
                parent=g["parent"],
                start=start,
                end=end,
                times=times.tolist(),
                boundaries=boundaries,
                means=mean.tolist(),
                weights=weight.tolist(),
                indices=indices,
                group=g,
            )
            buckets[key].append(len(curves))
            curves.append(row)
    kkt = []
    crosschecks = []
    for (times, boundaries, weights), indices in buckets.items():
        means = np.array([curves[i]["means"] for i in indices])
        w = np.array(weights)
        for variant, extra in (("immediate_shape", 0), ("entry_shape", 1)):
            a, b = constraints(np.array(times), boundaries + extra)
            z, dual, active, proof = project(means, w, a, b)
            kkt.append(
                dict(
                    variant=variant,
                    times=times,
                    boundaries=boundaries,
                    curves=len(indices),
                    **proof,
                )
            )
            # Independent SLSQP formulation for deterministic first/middle/last rows in every bucket.
            for index in sorted(set((0, len(indices) // 2, len(indices) - 1))):
                mu = means[index]
                solution = minimize(
                    lambda v: float(np.sum(w * (v - mu) ** 2)),
                    mu,
                    jac=lambda v: 2 * w * (v - mu),
                    constraints=[LinearConstraint(a, -np.inf, b)],
                    method="SLSQP",
                    options={"ftol": 1e-12, "maxiter": 1000},
                )
                error = float(np.max(np.abs(solution.x - z[index])))
                assert solution.success and error < 1e-5, (solution.message, error)
                crosschecks.append(
                    dict(
                        variant=variant,
                        curve=indices[index],
                        max_solution_error=error,
                        objective_error=abs(
                            float(solution.fun)
                            - float(np.sum(w * (z[index] - mu) ** 2))
                        ),
                    )
                )
            for local, index in enumerate(indices):
                curves[index][variant] = z[local].tolist()
                curves[index][variant + "_dual"] = dual[local].tolist()
                curves[index][variant + "_active"] = active[local]
    for curve in curves:
        g = curve.pop("group")
        indices = curve.pop("indices")
        times = curve["times"]
        for index in indices:
            original, t, ref, signal, reference = g["rows"][index]
            j = times.index(t) - 1
            source_rows.append(
                dict(
                    curve_id=original["curve_id"],
                    condition_id=g["condition"],
                    construct_id=g["parent"],
                    start=curve["start"],
                    end=curve["end"],
                    time=t,
                    reference_time=ref,
                    signal=signal,
                    reference_signal=reference,
                    target=float(g["y"][index]),
                    weight=float(g["w"][index]),
                    replicate_mean=curve["means"][j],
                    immediate_shape=curve["immediate_shape"][j],
                    entry_shape=curve["entry_shape"][j],
                )
            )
    result = {}
    for variant in ("replicate_mean", "immediate_shape", "entry_shape"):
        y = np.array([r["target"] for r in source_rows])
        p = np.array([r[variant] for r in source_rows])
        w = np.array([r["weight"] for r in source_rows])
        assert abs(w.sum() - 1) < 1e-12
        byparent = {}
        for parent in sorted({r["construct_id"] for r in source_rows}):
            mask = np.array([r["construct_id"] == parent for r in source_rows])
            wp = w[mask] / w[mask].sum()
            byparent[parent] = dict(
                rmse=float(np.sqrt(wp @ ((p[mask] - y[mask]) ** 2))), n=int(mask.sum())
            )
        result[variant] = dict(
            mse=float(w @ ((p - y) ** 2)),
            rmse=float(np.sqrt(w @ ((p - y) ** 2))),
            pearson=float(pearsonr(p, y).statistic),
            spearman=float(spearmanr(p, y).statistic),
            prediction_sd=float(p.std()),
            n=len(y),
            by_parent=byparent,
        )
    assert result["replicate_mean"]["mse"] <= result["entry_shape"]["mse"] + 1e-12
    assert result["entry_shape"]["mse"] <= result["immediate_shape"]["mse"] + 1e-12
    for name, rows in (("shape-curves", curves), ("shape-predictions", source_rows)):
        with gzip.open(FAMILY / f"results/{name}.jsonl.gz", "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    (FAMILY / "results/shape-metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    qualification = dict(
        kkt=kkt,
        independent_slsqp=crosschecks,
        n_curves=len(curves),
        n_predictions=len(source_rows),
        n_buckets=len(buckets),
        wall_seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_heldout_evaluation=True,
        projection_is_necessary_not_sufficient_process_bound=True,
    )
    (FAMILY / "results/shape-qualification.json").write_text(
        json.dumps(qualification, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                v: {k: m[k] for k in ("mse", "rmse", "pearson", "spearman")}
                for v, m in result.items()
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in qualification.items()
                if k not in ("kkt", "independent_slsqp")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
