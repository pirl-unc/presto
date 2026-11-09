"""Independent mask-generator/Frechet/path and actual-parent artifact replay."""

import json
import math
import resource
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm, expm_frechet

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(name):
    return json.loads((FAMILY / name).read_text())


def close(a, b, tolerance=1e-10):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert error <= tolerance, (error, tolerance)
    return error


def mask_generator(logs, eta):
    bonds = len(logs)
    size = bonds + 1
    n = 2**bonds
    q = np.zeros((n, n))
    derivatives = np.zeros((bonds + 1, n, n))
    spans = [(a, b) for a in range(size) for b in range(a + 1, size + 1)]
    presence = np.zeros((n, len(spans)))
    for state in range(n):
        cuts = [k for k in range(bonds) if state & (1 << (bonds - k - 1))]
        boundaries = [0, *[k + 1 for k in cuts], size]
        products = list(zip(boundaries[:-1], boundaries[1:]))
        for j, span in enumerate(spans):
            presence[state, j] = span in products
        for k in range(bonds):
            bit = 1 << (bonds - k - 1)
            if state & bit:
                continue
            left = max([0, *[j + 1 for j in cuts if j < k]])
            right = min([size, *[j + 1 for j in cuts if j > k]])
            length = math.log((right - left) / 20)
            hazard = math.exp(logs[k] + eta * length)
            target = state | bit
            q[state, target] = hazard
            q[state, state] -= hazard
            derivatives[k, state, target] = hazard
            derivatives[k, state, state] -= hazard
            derivatives[-1, state, target] = hazard * length
            derivatives[-1, state, state] -= hazard * length
    return q, derivatives, presence, spans


def verify_toy():
    cfg = read("reproduce/launch.json")
    logs = np.log(cfg["fixture"]["rates_per_hour"])
    cached = {}
    errors = []
    for eta in cfg["fixture"]["eta"]:
        q, dq, presence, spans = mask_generator(logs, eta)
        weights = np.sin(np.arange(len(spans))) + 1.1
        reward = presence @ weights
        close(q.sum(1), 0)
        close(presence @ np.array([b - a for a, b in spans]), 6)
        for t in cfg["fixture"]["times_hours"]:
            p = expm(q * t)[0]
            gradient = np.array(
                [expm_frechet(q * t, d * t, compute_expm=False)[0] @ reward for d in dq]
            )
            cached[eta, t] = dict(
                probability=p,
                counts=p @ presence,
                gradient=gradient,
                q=q,
                dq=dq,
                presence=presence,
                reward=reward,
            )
    for row in read("results/mathematical-checks.json"):
        ref = cached[row["eta"], row["time"]]
        errors.append(
            dict(
                eta=row["eta"],
                time=row["time"],
                amount_error=close(row["expected_counts"], ref["counts"]),
                gradient_error=close(row["exact_gradient"], ref["gradient"]),
            )
        )
    probe_results = []
    for row in read("results/path-probes.json"):
        ref = cached[row["eta"], row["time"]]
        n = row["count"]
        gradients = []
        hist = np.zeros(32, dtype=int)
        close(row["expected_state_probabilities"], ref["probability"])
        for group in row["groups"]:
            counts = np.array(group["state_counts"])
            sums = np.array(group["state_score_sums"])
            assert counts.sum() == n and (counts >= 0).all()
            hist += counts
            mean_reward = counts @ ref["reward"] / n
            mean_score = sums.sum(0) / n
            mean_product = ref["reward"] @ sums / n
            gradient = n / (n - 1) * (mean_product - mean_reward * mean_score)
            close(group["reward_mean"], mean_reward)
            close(group["score_mean"], mean_score)
            close(group["reward_score_mean"], mean_product)
            close(group["loo_gradient"], gradient)
            gradients.append(gradient)
        gradients = np.array(gradients)
        norm = np.linalg.norm(ref["gradient"])
        rel = np.linalg.norm(gradients.mean(0) - ref["gradient"]) / norm
        rms = np.sqrt(((gradients - ref["gradient"]) ** 2).sum(1).mean()) / norm
        close(rel, row["mean_gradient_relative_error"], 2e-8)
        close(rms, row["rms_gradient_relative_error"], 2e-8)
        total = n * len(gradients)
        p = ref["counts"]
        variance = total * p * (1 - p)
        valid = variance > 0
        z = float(
            np.max(
                np.abs((hist @ ref["presence"])[valid] - total * p[valid])
                / np.sqrt(variance[valid])
            )
        )
        close(z, row["maximum_marginal_standard_errors"], 1e-7)
        gate = n == 65536 and z <= 6 and rel <= 0.2
        assert gate == row["qualification_gate_pass"]
        probe_results.append(
            dict(
                eta=row["eta"],
                time=row["time"],
                count=n,
                mean_relative_error=float(rel),
                rms_relative_error=float(rms),
                maximum_marginal_standard_errors=z,
                gate=bool(gate),
            )
        )
    trace_error = 0
    for row in read("results/path-traces.json"):
        ref = cached[row["eta"], row["time"]]
        state = 0
        previous = 0.0
        density = 0.0
        score = np.zeros(6)
        events = [
            (k, t) for k, t in zip(row["event_bonds"], row["event_times"]) if k >= 0
        ]
        for bond, t in events + [(-1, row["time"])]:
            dt = t - previous
            assert dt >= 0
            density += ref["q"][state, state] * dt
            score += ref["dq"][:, state, state] * dt
            if bond >= 0:
                target = state | (1 << (4 - bond))
                assert target != state
                hazard = ref["q"][state, target]
                density += math.log(hazard)
                score += ref["dq"][:, state, target] / hazard
                state = target
            previous = t
        trace_error = max(
            trace_error, close(score, row["score"]), close(density, row["log_density"])
        )
        assert [bool(state & (1 << (4 - k))) for k in range(5)] == row["cuts"]
    return dict(
        mathematical_cases=errors,
        probes=probe_results,
        n_traces=288,
        maximum_trace_error=trace_error,
    )


def rates(case, coef):
    sequence = case["sequence"]
    x = np.zeros((len(sequence) - 1, 40))
    x[:, 0] = 1
    x[:, -1] = math.log(
        case["substrate_concentration_um"] / case["enzyme_substrate_denominator"] / 0.2
    )
    for i, (left, right) in enumerate(zip(sequence[:-1], sequence[1:])):
        if left != "A":
            x[i, AA.index(left)] = 1
        if right != "A":
            x[i, 19 + AA.index(right)] = 1
    return np.exp(x @ coef), x


def verify_runtime():
    summary = read("results/runtime-summary.json")
    assert summary["complete"]
    cfg = read("reproduce/runtime.json")
    inputs = read("results/runtime-inputs.json")
    coef = np.array(inputs["initial_coefficients"])
    bycase = {}
    verified = []
    for row in summary["cases"]:
        case = cfg["cases"][row["case_id"]]
        item = inputs["cases"][case["input_index"]]
        size = item["n_residues"]
        rate, x = rates(item, coef)
        i = np.arange(size)
        weight = (np.sin(i[:, None] + 1) * np.cos(i[None, :] + 0.3) + 1.1) / size
        lengths = i[None, :] - i[:, None] + 1
        valid = lengths > 0
        if row["status"] != "passed":
            verified.append(
                dict(
                    case_id=row["case_id"],
                    backend=row["backend"],
                    status=row["status"],
                    scope="Recorded failure retained; no successful-output claim",
                )
            )
            continue
        archive = np.load(
            FAMILY / f"results/runtime-{row['case_id']:02d}-{row['backend']}.npz"
        )
        checks = {}
        if row["backend"] == "exact":
            counts = archive["counts"]
            assert (counts >= 0).all() and (counts[~valid] == 0).all()
            close((counts * lengths * valid).sum(), size, 1e-8)
            close((counts * weight).sum(), row["metrics"]["readout"])
            if case["eta"] == 0:
                prefix = np.r_[0, np.cumsum(rate)]
                survival = np.exp(-24 * (prefix[None, :] - prefix[:, None]))
                boundary = -np.expm1(-24 * rate)
                expected = (
                    np.r_[1, boundary][:, None]
                    * np.r_[boundary, 1][None, :]
                    * survival
                    * valid
                )
                checks["independent_amount_error"] = close(counts, expected)
                contribution = expected * weight
                cumulative = contribution.cumsum(0)[:, ::-1].cumsum(1)[:, ::-1]
                h = 24 * rate
                cutscore = h * np.exp(-h) / (-np.expm1(-h))
                gradient = (
                    cutscore * (contribution.sum(0)[:-1] + contribution.sum(1)[1:])
                    - h * cumulative[np.arange(size - 1), np.arange(1, size)]
                )
                checks["independent_log_rate_gradient_error"] = close(
                    gradient, archive["log_rate_gradient"]
                )
            gradient = np.r_[archive["log_rate_gradient"], archive["eta_gradient"]]
        else:
            cuts = archive["cuts"]
            counts = np.zeros((size, size))
            rewards = []
            for cut in cuts:
                boundaries = [0, *list(np.flatnonzero(cut) + 1), size]
                rewards.append(
                    sum(
                        weight[a, b - 1]
                        for a, b in zip(boundaries[:-1], boundaries[1:])
                    )
                )
                for a, b in zip(boundaries[:-1], boundaries[1:]):
                    counts[a, b - 1] += 1 / len(cuts)
            close(counts, archive["mean_counts"])
            close(rewards, archive["path_rewards"])
            rewards = np.array(rewards)
            score = archive["path_scores"]
            gradient = (
                len(cuts)
                / (len(cuts) - 1)
                * ((rewards[:, None] * score).mean(0) - rewards.mean() * score.mean(0))
            )
            close(gradient, archive["loo_gradient"])
            close((counts * lengths * valid).sum(), size)
            close(rewards.mean(), row["metrics"]["readout"])
        bycase.setdefault(row["case_id"], {})[row["backend"]] = dict(
            counts=counts, gradient=gradient, x=x, lengths=lengths, valid=valid
        )
        verified.append(
            dict(
                case_id=row["case_id"],
                backend=row["backend"],
                status="verified",
                **checks,
            )
        )
    comparisons = []
    for key, value in bycase.items():
        if set(value) != {"exact", "sampled"}:
            continue
        a, b = value["exact"], value["sampled"]
        size = a["counts"].shape[0]
        exact_mass = a["counts"] * a["lengths"] * a["valid"] / size
        sampled_mass = b["counts"] * b["lengths"] * b["valid"] / size
        theta_gradient = np.r_[a["gradient"][:-1] @ a["x"], a["gradient"][-1]]
        sampled_theta = np.r_[b["gradient"][:-1] @ b["x"], b["gradient"][-1]]
        comparisons.append(
            dict(
                case_id=key,
                tagged_residue_mass_TV=float(
                    np.abs(exact_mass - sampled_mass).sum() / 2
                ),
                copy_count_rmse=float(
                    np.sqrt(
                        np.mean(
                            (a["counts"][a["valid"]] - b["counts"][b["valid"]]) ** 2
                        )
                    )
                ),
                raw_bond_gradient_relative_error=float(
                    np.linalg.norm(b["gradient"] - a["gradient"])
                    / np.linalg.norm(a["gradient"])
                ),
                sequence_parameter_gradient_relative_error=float(
                    np.linalg.norm(sampled_theta - theta_gradient)
                    / np.linalg.norm(theta_gradient)
                ),
                scope="One K512 group and deterministic artificial readout; no biological training quality or variance guarantee",
            )
        )
    return dict(cases=verified, comparisons=comparisons)


def main():
    tick = time.perf_counter()
    toy = verify_toy()
    runtime = verify_runtime()
    result = dict(
        all_checks_pass=True,
        toy=toy,
        runtime=runtime,
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="Independent mask generator/Frechet derivatives, all saved Monte Carlo sufficient statistics,288toy event traces and every successful runtime output. Full-scale path scores not regenerated; failures preserved.",
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["toy", "runtime"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
