"""Fixed complete-path importance qualification and rare-product diagnostics."""

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad
from scipy.linalg import expm, expm_frechet
from scipy.special import logsumexp

from boundary import sample

FAMILY = Path(__file__).resolve().parents[1]


def exact(rates, eta, t, start, end):
    n = len(rates)
    size = n + 1
    q = np.zeros((2**n, 2**n))
    dq = np.zeros((n + 1, 2**n, 2**n))
    indicator = np.zeros(2**n)
    for state in range(2**n):
        cut = [k for k in range(n) if state & (1 << k)]
        edges = [0, *[k + 1 for k in cut], size]
        indicator[state] = int((start, end) in list(zip(edges[:-1], edges[1:])))
        for k in range(n):
            if k in cut:
                continue
            left = max([0, *[j + 1 for j in cut if j < k]])
            right = min([size, *[j + 1 for j in cut if j > k]])
            length = math.log((right - left) / 20)
            hazard = rates[k] * math.exp(eta * length)
            target = state | (1 << k)
            for j, sign in [(state, -1), (target, 1)]:
                q[state, j] += sign * hazard
                dq[k, state, j] += sign * hazard
                dq[-1, state, j] += sign * hazard * length
    p = float(expm(q * t)[0] @ indicator)
    gradient = np.array(
        [expm_frechet(q * t, d * t, compute_expm=False)[0] @ indicator for d in dq]
    )
    assert p > 0
    return math.log(p), gradient / p


def rare_reference(rates, t):
    # The [1,5) fixture has no allowed outside bonds. At eta0 its boundary
    # times conditional on presence are independent truncated exponentials.
    left, right = rates[0], rates[-1]
    internal = sum(rates[1:-1])
    norm = (-math.expm1(-left * t)) * (-math.expm1(-right * t))
    logp = math.log(norm) - internal * t
    grad = -np.array(rates) * t
    grad[0] = left * t / math.expm1(left * t)
    grad[-1] = right * t / math.expm1(right * t)
    l6, l5, l4 = [math.log(x / 20) for x in [6, 5, 4]]

    def integrand(first, second, early, late):
        score = (
            l6
            + l5
            - (internal + early + late) * l6 * first
            - (internal + late) * l5 * (second - first)
            - internal * l4 * (t - second)
        )
        return early * late * math.exp(-early * first - late * second) * score / norm

    eta = 0.0
    error = 0.0
    for early, late in [(left, right), (right, left)]:
        inner = lambda first: quad(
            lambda second: integrand(first, second, early, late),
            first,
            t,
            epsabs=1e-9,
            epsrel=1e-11,
        )[0]
        value, err = quad(inner, 0, t, epsabs=1e-8, epsrel=1e-11)
        eta += value
        error += err
    return logp, np.r_[grad, eta], error


def group_summary(result):
    logw = result.log_weight.numpy()
    scores = result.original_score.numpy()
    maximum = float(logw.max())
    scaled = np.exp(logw - maximum)
    total = float(scaled.sum())
    total2 = float((scaled**2).sum())
    numerator = scaled @ scores
    logmean = maximum + math.log(total) - math.log(len(logw))
    return dict(
        log_weight_max=maximum,
        sum_scaled_weights=total,
        sum_scaled_weight_squared=total2,
        sum_scaled_weight_score=numerator.tolist(),
        log_mean=logmean,
        log_gradient=(numerator / total).tolist(),
        ess=total * total / total2,
        proposal_score_mean=result.proposal_score.mean(0).tolist(),
        log_density_ratio_error=float(
            np.max(
                np.abs(
                    logw
                    - (
                        result.original_log_density - result.proposal_log_density
                    ).numpy()
                )
            )
        ),
        underflowed_linear_weights=int((np.exp(logw) == 0).sum()),
    )


def run_state(state, index, cfg, rare=False):
    rates = np.array(state["rates"], dtype=float)
    if rare:
        logp, truth, quad_error = rare_reference(rates, state["time"])
    else:
        logp, truth = exact(rates, state["eta"], state["time"], *state["target"])
        quad_error = None
    rows = []
    traces = []
    for ki, count in enumerate(cfg["counts"]):
        groups = []
        tick = time.perf_counter()
        for gi in range(cfg["groups"]):
            seed = (
                cfg["seed_base"]
                + (50000000 if rare else 0)
                + 1000000 * index
                + 10000 * ki
                + gi
            )
            result = sample(
                torch.tensor(np.log(rates), dtype=torch.float64),
                state["eta"],
                *state["target"],
                state["time"],
                count,
                generator=torch.Generator().manual_seed(seed),
                record_paths=ki == 0 and gi == 0,
            )
            group = group_summary(result)
            group.update(seed=seed, count=count)
            groups.append(group)
            if ki == 0 and gi == 0:
                for j in range(min(8, count)):
                    traces.append(
                        dict(
                            state=state,
                            rare=rare,
                            seed=seed,
                            draw=j,
                            log_weight=float(result.log_weight[j]),
                            original_score=result.original_score[j].tolist(),
                            proposal_score=result.proposal_score[j].tolist(),
                            original_log_density=float(result.original_log_density[j]),
                            proposal_log_density=float(result.proposal_log_density[j]),
                            cuts=result.cuts[j].tolist(),
                            boundary_times=result.boundary_times[j].tolist(),
                            boundary_bonds=result.boundary_bonds[j].tolist(),
                            event_bonds=result.event_bonds[j].tolist(),
                            event_times=result.event_times[j].tolist(),
                            event_forced=result.event_forced[j].tolist(),
                        )
                    )
            if count == cfg["gates"]["primary_count"] and gi == 0:
                name = f"primary-{'rare' if rare else 'ordinary'}-{index:02d}.npz"
                np.savez_compressed(
                    FAMILY / "results" / name,
                    log_weight=result.log_weight.numpy(),
                    original_score=result.original_score.numpy(),
                    proposal_score=result.proposal_score.numpy(),
                    logp=result.original_log_density.numpy(),
                    logq=result.proposal_log_density.numpy(),
                    cuts=result.cuts.numpy(),
                )
                group["raw_archive"] = name
        logmeans = np.array([g["log_mean"] for g in groups])
        gradients = np.array([g["log_gradient"] for g in groups])
        relmean = math.expm1(
            float(logsumexp(logmeans) - math.log(len(logmeans)) - logp)
        )
        bias = float((logmeans - logp).mean())
        norm = float(np.linalg.norm(truth))
        meanerr = float(np.linalg.norm(gradients.mean(0) - truth) / norm)
        rmserr = float(np.sqrt(((gradients - truth) ** 2).sum(1).mean()) / norm)
        gate = (
            abs(relmean) <= cfg["gates"]["maximum_mean_relative_count_error"]
            and abs(bias) <= cfg["gates"]["maximum_abs_mean_log_error"]
            and meanerr <= cfg["gates"]["maximum_mean_log_gradient_relative_error"]
            and rmserr <= cfg["gates"]["maximum_rms_log_gradient_relative_error"]
        )
        row = dict(
            state_index=index,
            rare=rare,
            state=state,
            count=count,
            exact_log_probability=logp,
            exact_log_gradient=truth.tolist(),
            rare_eta_quadrature_error=quad_error,
            mean_relative_probability_error=relmean,
            mean_log_probability_bias=bias,
            mean_log_gradient_relative_error=meanerr,
            rms_log_gradient_relative_error=rmserr,
            minimum_ess=min(g["ess"] for g in groups),
            mean_ess=float(np.mean([g["ess"] for g in groups])),
            seconds=time.perf_counter() - tick,
            primary=count == cfg["gates"]["primary_count"],
            gate_pass=bool(gate),
            groups=groups,
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ["groups", "exact_log_gradient"]
                }
            ),
            flush=True,
        )
    return rows, traces


def main():
    torch.set_num_threads(1)
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    frozen = json.loads((FAMILY / "reproduce/execution.json").read_text())
    for path, digest in frozen["source_hashes"].items():
        assert hashlib.sha256((FAMILY / path).read_bytes()).hexdigest() == digest
    output = FAMILY / "results/qualification.json"
    assert not output.exists(), "Refuse to overwrite qualification"
    tick = time.perf_counter()
    rows = []
    traces = []
    index = 0
    for eta in cfg["eta"]:
        for t in cfg["times_hours"]:
            for target in cfg["targets"]:
                state = dict(
                    rates=cfg["rates_per_hour"], eta=eta, time=t, target=target
                )
                a, b = run_state(state, index, cfg)
                rows.extend(a)
                traces.extend(b)
                index += 1
    for i, case in enumerate(cfg["rare_cases"]):
        a, b = run_state(case, i, cfg, True)
        rows.extend(a)
        traces.extend(b)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    (FAMILY / "results/path-traces.json").write_text(
        json.dumps(traces, indent=2) + "\n"
    )
    summary = dict(
        n_conditions=len(rows),
        n_groups=sum(len(r["groups"]) for r in rows),
        n_sampled_paths=sum(r["count"] * len(r["groups"]) for r in rows),
        n_traces=len(traces),
        ordinary_primary_pass=sum(
            r["gate_pass"] for r in rows if r["primary"] and not r["rare"]
        ),
        ordinary_primary_total=sum(r["primary"] and not r["rare"] for r in rows),
        rare_primary_pass=sum(
            r["gate_pass"] for r in rows if r["primary"] and r["rare"]
        ),
        rare_primary_total=sum(r["primary"] and r["rare"] for r in rows),
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
