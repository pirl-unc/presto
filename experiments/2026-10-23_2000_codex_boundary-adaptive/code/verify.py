"""Independent fragment generator, closed rare scores, path and statistic replay."""

import json
import math
import resource
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.special import logsumexp

from path_reference import path_values

FAMILY = Path(__file__).resolve().parents[1]


def close(a, b, tolerance=1e-9):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert error <= tolerance, (error, tolerance)
    return error


def reference(state):
    rates = np.array(state["rates"])
    eta = state["eta"]
    t = state["time"]
    n = len(rates) + 1
    spans = [(a, b) for a in range(n) for b in range(a + 1, n + 1)]
    index = {span: i for i, span in enumerate(spans)}
    a = np.zeros((len(spans), len(spans)))
    der = np.zeros((n, len(spans), len(spans)))
    for parent, (left, right) in enumerate(spans):
        log_length = math.log((right - left) / 20)
        for k in range(left, right - 1):
            hazard = rates[k] * math.exp(eta * log_length)
            for child, sign in [
                (parent, -1),
                (index[left, k + 1], 1),
                (index[k + 1, right], 1),
            ]:
                a[child, parent] += sign * hazard
                der[k, child, parent] += sign * hazard
                der[-1, child, parent] += sign * hazard * log_length
    row = index[tuple(state["target"])]
    col = index[0, n]
    p = expm(t * a)[row, col]
    dp = np.array(
        [expm_frechet(t * a, t * d, compute_expm=False)[row, col] for d in der]
    )
    return math.log(p), dp / p


def rare_reference(state):
    r = np.array(state["rates"])
    t = state["time"]
    left, right = r[0], r[-1]
    norm1 = -math.expm1(-left * t)
    norm2 = -math.expm1(-right * t)
    surv1 = math.exp(-left * t)
    surv2 = math.exp(-right * t)
    mean1 = 1 / left - t * surv1 / norm1
    mean2 = 1 / right - t * surv2 / norm2
    minimum = (
        (-math.expm1(-(left + right) * t)) / (left + right)
        - surv2 * norm1 / left
        - surv1 * norm2 / right
        + t * surv1 * surv2
    ) / (norm1 * norm2)
    maximum = mean1 + mean2 - minimum
    l6, l5, l4 = [math.log(v / 20) for v in [6, 5, 4]]
    eta = (
        l6
        + l5
        - sum(r[1:-1]) * (l4 * t + (l6 - l5) * minimum + (l5 - l4) * maximum)
        - l5 * (left * mean1 + right * mean2)
        - (left + right) * (l6 - l5) * minimum
    )
    grad = -r * t
    grad[0] = left * t * surv1 / norm1
    grad[-1] = right * t * surv2 / norm2
    return math.log(norm1) + math.log(norm2) - sum(r[1:-1]) * t, np.r_[grad, eta]


def trace_check(row):
    logp, logq, sp, sq = path_values(row["state"], row)
    return max(
        close(logp, row["original_log_density"]),
        close(logq, row["proposal_log_density"]),
        close(logp - logq, row["log_weight"]),
        close(sp, row["original_score"]),
        close(sq, row["proposal_score"]),
    )


def main():
    tick = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    rows = json.loads((FAMILY / "results/qualification.json").read_text())
    results = []
    truth = {}
    raw_count = 0
    maxraw = 0.0
    for row in rows:
        key = row["rare"], row["state_index"]
        if key not in truth:
            truth[key] = (rare_reference if row["rare"] else reference)(row["state"])
        logp, grad = truth[key]
        reference_error = max(
            close(logp, row["exact_log_probability"]),
            close(grad, row["exact_log_gradient"], 1e-8),
        )
        logmeans = []
        gradients = []
        plain_gradients = []
        for group in row["groups"]:
            n = group["count"]
            maximum = group["log_weight_max"]
            total = group["sum_scaled_weights"]
            total2 = group["sum_scaled_weight_squared"]
            numerator = np.array(group["sum_scaled_weight_score"])
            logmean = maximum + math.log(total / n)
            plain = numerator / total
            qnumerator = np.array(group["sum_scaled_weight_proposal_score"])
            qsum = n * np.array(group["proposal_score_mean"])
            estimate = plain - (qsum - qnumerator / total) / (n - 1)
            close(plain, group["plain_log_gradient"])
            plain_gradients.append(plain)
            close(logmean, group["log_mean"])
            close(estimate, group["log_gradient"])
            close(total * total / total2, group["ess"], 1e-8)
            logmeans.append(logmean)
            gradients.append(estimate)
            if "raw_archive" in group:
                raw = np.load(FAMILY / "results" / group["raw_archive"])
                lw = raw["log_weight"]
                score = raw["original_score"]
                scaled = np.exp(lw - lw.max())
                maxraw = max(
                    maxraw,
                    close(lw.max(), maximum),
                    close(scaled.sum(), total, 1e-8),
                    close(scaled @ score, numerator, 1e-6),
                    close(scaled @ raw["proposal_score"], qnumerator, 1e-6),
                    close((scaled**2).sum(), total2, 1e-8),
                    close(raw["proposal_score"].mean(0), group["proposal_score_mean"]),
                )
                close(lw, raw["logp"] - raw["logq"])
                assert (
                    int((np.exp(lw) == 0).sum()) == group["underflowed_linear_weights"]
                )
                raw_count += 1
        means = np.array(logmeans)
        estimates = np.array(gradients)
        norm = np.linalg.norm(grad)
        rel = math.expm1(float(logsumexp(means) - math.log(len(means)) - logp))
        bias = float(np.mean(means - logp))
        meanerr = np.linalg.norm(estimates.mean(0) - grad) / norm
        rms = np.sqrt(((estimates - grad) ** 2).sum(1).mean()) / norm
        close(rel, row["mean_relative_probability_error"])
        close(bias, row["mean_log_probability_bias"])
        close(meanerr, row["mean_log_gradient_relative_error"])
        close(rms, row["rms_log_gradient_relative_error"])
        plain_array = np.array(plain_gradients)
        close(
            np.linalg.norm(plain_array.mean(0) - grad) / norm,
            row["plain_mean_log_gradient_relative_error"],
        )
        close(
            np.sqrt(((plain_array - grad) ** 2).sum(1).mean()) / norm,
            row["plain_rms_log_gradient_relative_error"],
        )
        unbiased = np.exp(means - logp)[:, None] * estimates
        close(
            np.linalg.norm(unbiased.mean(0) - grad) / norm,
            row["controlled_numerator_mean_relative_error"],
        )
        g = cfg["gates"]
        passed = (
            abs(rel) <= g["maximum_mean_relative_count_error"]
            and abs(bias) <= g["maximum_abs_mean_log_error"]
            and meanerr <= g["maximum_mean_log_gradient_relative_error"]
            and rms <= g["maximum_rms_log_gradient_relative_error"]
        )
        assert passed == row["gate_pass"]
        results.append(
            dict(
                rare=row["rare"],
                state_index=row["state_index"],
                count=row["count"],
                reference_error=reference_error,
                gate_verified=True,
            )
        )
    traces = json.loads((FAMILY / "results/path-traces.json").read_text())
    trace_error = max(trace_check(r) for r in traces)
    old = (
        FAMILY.parent
        / "2026-10-23_1816_codex_fragment-state-qualification/results/path-probes.json"
    )
    unconditional = json.loads(old.read_text())
    comparisons = []
    for row in rows:
        if row["rare"] or not row["primary"]:
            continue
        state = row["state"]
        start, end = state["target"]
        indicator = []
        for mask in range(32):
            cuts = [k for k in range(5) if mask & (1 << (4 - k))]
            edges = [0, *[k + 1 for k in cuts], 6]
            indicator.append(int((start, end) in list(zip(edges[:-1], edges[1:]))))
        indicator = np.array(indicator)
        logp, grad = truth[False, row["state_index"]]
        for reference_row in unconditional:
            if (
                reference_row["eta"] != state["eta"]
                or reference_row["time"] != state["time"]
            ):
                continue
            k = reference_row["count"]
            means = []
            scores = []
            for group in reference_row["groups"]:
                hits = int(np.array(group["state_counts"]) @ indicator)
                numerator = indicator @ np.array(group["state_score_sums"]) / k
                means.append(hits / k)
                scores.append(numerator / math.exp(logp))
            comparisons.append(
                dict(
                    state_index=row["state_index"],
                    state=state,
                    count=k,
                    zero_hit_groups=sum(p == 0 for p in means),
                    groups=len(means),
                    mean_relative_probability_error=float(
                        np.mean(means) / math.exp(logp) - 1
                    ),
                    mean_gradient_relative_error=float(
                        np.linalg.norm(np.mean(scores, axis=0) - grad)
                        / np.linalg.norm(grad)
                    ),
                    scope="Original unconditional shared-path samples, different K; no matched runtime claim",
                )
            )
    (FAMILY / "results/unconditional-reference.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    output = dict(
        all_checks_pass=True,
        n_group_summaries=sum(len(r["groups"]) for r in rows),
        n_full_raw_groups=raw_count,
        maximum_raw_statistic_error=maxraw,
        n_traces=len(traces),
        maximum_trace_error=trace_error,
        conditions=results,
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="Independent21-fragment reference, closed rare eta scores, all sufficient-statistic algebra,48 complete raw primary groups and384 full event traces; remaining individual paths not replayed",
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in output.items() if k != "conditions"}), flush=True
    )


if __name__ == "__main__":
    main()
