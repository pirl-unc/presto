"""Independent dense generators and fixed full-path Monte Carlo qualification."""

import itertools
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

from presto.models.fragment_state import (
    fragment_count_derivative,
    fragment_expectations,
    sample_fragment_paths,
)

FAMILY = Path(__file__).resolve().parents[1]


def dump(name, value):
    (FAMILY / "results" / name).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )


def dense_fragment(logs, eta):
    size = len(logs) + 1
    spans = [(a, b) for a in range(size) for b in range(a + 1, size + 1)]
    index = {span: i for i, span in enumerate(spans)}
    matrix = np.zeros((len(spans), len(spans)))
    for i, (a, b) in enumerate(spans):
        for k in range(a, b - 1):
            rate = np.exp(logs[k]) * ((b - a) / 20) ** eta
            matrix[i, i] -= rate
            matrix[index[a, k + 1], i] += rate
            matrix[index[k + 1, b], i] += rate
    initial = np.zeros(len(spans))
    initial[index[0, size]] = 1
    return matrix, initial, spans


def dense_partitions(logs, eta):
    bonds = len(logs)
    size = bonds + 1
    states = np.array(list(itertools.product([False, True], repeat=bonds)))
    mapping = {tuple(row): i for i, row in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for i, cuts in enumerate(states):
        boundaries = [0, *list(np.flatnonzero(cuts) + 1), size]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            for k in range(a, b - 1):
                rate = np.exp(logs[k]) * ((b - a) / 20) ** eta
                changed = cuts.copy()
                changed[k] = True
                matrix[mapping[tuple(changed)], i] += rate
                matrix[i, i] -= rate
    spans = [(a, b) for a in range(size) for b in range(a + 1, size + 1)]
    presence = np.zeros((len(states), len(spans)))
    for i, cuts in enumerate(states):
        boundaries = [0, *list(np.flatnonzero(cuts) + 1), size]
        products = set(zip(boundaries[:-1], boundaries[1:]))
        for j, span in enumerate(spans):
            presence[i, j] = span in products
    initial = np.zeros(len(states))
    initial[0] = 1
    return matrix, initial, states, presence


def reward_expectation(parameters, timepoint, weights):
    matrix, initial, _ = dense_fragment(parameters[:-1], parameters[-1])
    return float((expm(matrix * timepoint) @ initial) @ weights)


def gradient_reference(parameters, timepoint, weights, step):
    result = []
    for i in range(len(parameters)):
        a = parameters.copy()
        b = parameters.copy()
        a[i] += step
        b[i] -= step
        result.append(
            (
                reward_expectation(a, timepoint, weights)
                - reward_expectation(b, timepoint, weights)
            )
            / (2 * step)
        )
    return np.array(result)


def replay_path(logs, eta, timepoint, event_bonds, event_times):
    size = len(logs) + 1
    cuts = np.zeros(len(logs), dtype=bool)
    gradient = np.zeros(len(logs) + 1)
    density = 0.0
    previous = 0.0
    events = [(int(k), float(t)) for k, t in zip(event_bonds, event_times) if k >= 0]
    for bond, now in events + [(-1, timepoint)]:
        boundaries = [0, *list(np.flatnonzero(cuts) + 1), size]
        hazards = np.zeros(len(logs))
        length = np.zeros(len(logs))
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            hazards[a : b - 1] = np.exp(logs[a : b - 1]) * ((b - a) / 20) ** eta
            length[a : b - 1] = math.log((b - a) / 20)
        dt = now - previous
        assert dt >= 0
        density -= hazards.sum() * dt
        gradient[:-1] -= hazards * dt
        gradient[-1] -= (hazards * length).sum() * dt
        if bond >= 0:
            assert not cuts[bond]
            density += math.log(hazards[bond])
            gradient[bond] += 1
            gradient[-1] += length[bond]
            cuts[bond] = True
        previous = now
    return cuts, gradient, density


def main():
    tick = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    assert not (FAMILY / "results/qualification.json").exists()
    logs0 = np.log(cfg["fixture"]["rates_per_hour"])
    times = cfg["fixture"]["times_hours"]
    mathematical = []
    probes = []
    traces = []
    for state, eta_value in enumerate(cfg["fixture"]["eta"]):
        matrix, initial, spans = dense_fragment(logs0, eta_value)
        partition, p0, states, presence = dense_partitions(logs0, eta_value)
        weights = np.sin(np.arange(len(spans))) + 1.1
        starts = torch.tensor([a for a, b in spans])
        lasts = torch.tensor([b - 1 for a, b in spans])
        logs = torch.tensor(logs0, dtype=torch.float64, requires_grad=True)
        eta = torch.tensor(eta_value, dtype=torch.float64, requires_grad=True)
        result = fragment_expectations(logs, eta, times)
        values = result.counts[:, starts, lasts]
        exact = np.stack([expm(matrix * t) @ initial for t in times])
        partition_values = np.stack(
            [(expm(partition * t) @ p0) @ presence for t in times]
        )
        amount_error = float(np.max(np.abs(values.detach().numpy() - exact)))
        partition_error = float(np.max(np.abs(partition_values - exact)))
        assert (
            max(amount_error, partition_error)
            <= cfg["tolerances"]["expected_amount_absolute"]
        )
        lengths = np.array([b - a for a, b in spans])
        assert np.all(presence @ lengths == 6)
        mass_error = float(np.max(np.abs(values.detach().numpy() @ lengths - 6)))
        assert mass_error <= cfg["tolerances"]["residue_mass_absolute"]
        action_input = torch.zeros((6, 6), dtype=torch.float64)
        vector = np.arange(len(spans)) / 30
        action_input[starts, lasts] = torch.tensor(vector)
        action = (
            fragment_count_derivative(action_input, logs, eta)[starts, lasts]
            .detach()
            .numpy()
        )
        action_error = float(np.max(np.abs(action - matrix @ vector)))
        assert action_error < cfg["tolerances"]["dense_operator"]
        if eta_value == 0:
            independent = []
            for t in times:
                row = []
                p = -np.expm1(-np.exp(logs0) * t)
                for a, b in spans:
                    row.append(
                        (p[a - 1] if a else 1)
                        * (p[b - 1] if b < 6 else 1)
                        * np.exp(-t * np.exp(logs0[a : b - 1]).sum())
                    )
                independent.append(row)
            assert np.max(np.abs(exact - np.array(independent))) < 1e-12
        for ti, t in enumerate(times):
            expected_gradient = gradient_reference(
                np.r_[logs0, eta_value],
                t,
                weights,
                cfg["tolerances"]["finite_difference_step"],
            )
            if t:
                gradient = torch.autograd.grad(
                    values[ti] @ torch.tensor(weights), (logs, eta), retain_graph=True
                )
                gradient = np.r_[gradient[0].numpy(), gradient[1].item()]
            else:
                gradient = np.zeros(6)
            gradient_error = float(np.max(np.abs(gradient - expected_gradient)))
            assert gradient_error <= cfg["tolerances"]["gradient_absolute"]
            mathematical.append(
                dict(
                    eta=eta_value,
                    time=t,
                    amount_error=amount_error,
                    partition_error=partition_error,
                    mass_error=mass_error,
                    action_error=action_error,
                    gradient_error=gradient_error,
                    exact_gradient=gradient.tolist(),
                    finite_difference_gradient=expected_gradient.tolist(),
                    expected_counts=exact[ti].tolist(),
                    spans=spans,
                    poisson_steps=result.poisson_steps,
                    operator_applications=result.operator_applications,
                    error_bound=result.expected_count_error_bound,
                )
            )
            if not t:
                continue
            state_prob = expm(partition * t) @ p0
            for count in cfg["monte_carlo"]["counts"]:
                now = time.perf_counter()
                groups = []
                total_hist = np.zeros(32, dtype=np.int64)
                for rep in range(cfg["monte_carlo"]["groups"]):
                    seed = (
                        cfg["monte_carlo"]["seed"]
                        + 100000 * state
                        + 1000 * ti
                        + count
                        + rep
                    )
                    trace = rep == 0 and count == cfg["monte_carlo"]["counts"][0]
                    samples = sample_fragment_paths(
                        logs.detach(),
                        eta.detach(),
                        t,
                        count,
                        generator=torch.Generator().manual_seed(seed),
                        record_paths=trace,
                    )
                    codes = (
                        samples.cuts.long() * torch.tensor([16, 8, 4, 2, 1])[None, :]
                    ).sum(1)
                    histogram = torch.bincount(codes, minlength=32).numpy()
                    total_hist += histogram
                    score = torch.cat(
                        (samples.log_rate_score, samples.length_score[:, None]), 1
                    )
                    rewards = torch.tensor(presence @ weights)[codes]
                    rs = (rewards[:, None] * score).mean(0)
                    estimate = (
                        count / (count - 1) * (rs - rewards.mean() * score.mean(0))
                    )
                    sums = torch.zeros((32, 6), dtype=torch.float64).index_add_(
                        0, codes, score
                    )
                    groups.append(
                        dict(
                            seed=seed,
                            state_counts=histogram.tolist(),
                            state_score_sums=sums.tolist(),
                            reward_mean=float(rewards.mean()),
                            score_mean=score.mean(0).tolist(),
                            reward_score_mean=rs.tolist(),
                            loo_gradient=estimate.tolist(),
                            log_density_min=float(samples.log_density.min()),
                            log_density_max=float(samples.log_density.max()),
                        )
                    )
                    if trace:
                        for i in range(min(32, count)):
                            bonds = samples.event_bonds[i].tolist()
                            event_times = samples.event_times[i].tolist()
                            cuts, g, d = replay_path(
                                logs0, eta_value, t, bonds, event_times
                            )
                            assert np.array_equal(cuts, samples.cuts[i].numpy())
                            error = max(
                                float(np.max(np.abs(g - score[i].numpy()))),
                                abs(d - float(samples.log_density[i])),
                            )
                            assert error < 1e-12
                            traces.append(
                                dict(
                                    eta=eta_value,
                                    time=t,
                                    seed=seed,
                                    sample_index=i,
                                    event_bonds=bonds,
                                    event_times=event_times,
                                    cuts=cuts.tolist(),
                                    score=g.tolist(),
                                    log_density=d,
                                    replay_error=error,
                                )
                            )
                gradients = np.array([g["loo_gradient"] for g in groups])
                norm = np.linalg.norm(gradient)
                mean_error = np.linalg.norm(gradients.mean(0) - gradient)
                relative = float(mean_error / norm) if norm else None
                rms = (
                    float(
                        np.sqrt(np.mean(np.sum((gradients - gradient) ** 2, axis=1)))
                        / norm
                    )
                    if norm
                    else None
                )
                n = count * len(groups)
                observed = total_hist @ presence
                expected = exact[ti]
                variance = n * expected * (1 - expected)
                positive = variance > 0
                z = float(
                    np.max(
                        np.abs(observed[positive] - n * expected[positive])
                        / np.sqrt(variance[positive])
                    )
                )
                assert np.all(observed[expected == 0] == 0)
                gate = (
                    count == cfg["monte_carlo"]["counts"][-1]
                    and z <= 6
                    and np.isfinite(gradients).all()
                    and (relative <= 0.2 if norm else mean_error <= 2e-7)
                )
                row = dict(
                    eta=eta_value,
                    time=t,
                    count=count,
                    groups=groups,
                    expected_state_probabilities=state_prob.tolist(),
                    mean_gradient_relative_error=relative,
                    rms_gradient_relative_error=rms,
                    maximum_marginal_standard_errors=z,
                    qualification_gate_pass=bool(gate),
                    seconds=time.perf_counter() - now,
                )
                probes.append(row)
                dump("path-probes.json", probes)
                dump("path-traces.json", traces)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in [
                                "eta",
                                "time",
                                "count",
                                "mean_gradient_relative_error",
                                "rms_gradient_relative_error",
                                "maximum_marginal_standard_errors",
                                "qualification_gate_pass",
                                "seconds",
                            ]
                        }
                    ),
                    flush=True,
                )
        dump("mathematical-checks.json", mathematical)
    qualified = [r for r in probes if r["count"] == cfg["monte_carlo"]["counts"][-1]]
    summary = dict(
        mathematics_pass=True,
        discrete_gate_pass=all(r["qualification_gate_pass"] for r in qualified),
        n_primary_cases=len(qualified),
        n_traces_replayed=len(traces),
        n_sampled_paths=sum(r["count"] * len(r["groups"]) for r in probes),
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="Numerical five-bond process qualification; no biological training/targets or APC concentration validation",
    )
    dump("qualification.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
