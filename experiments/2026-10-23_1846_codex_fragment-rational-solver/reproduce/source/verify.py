"""Artifact replay with independent NumPy storage, rates and analytic limits."""

import json
import math
import resource
import time
from pathlib import Path

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return json.loads(path.read_text())


def close(a, b, tolerance=1e-10):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert error <= tolerance, (error, tolerance)
    return error


def numpy_solve(rhs, rates, eta, pole, dt):
    size = len(rates) + 1
    prefix = np.r_[0, np.cumsum(rates)]
    left, right = np.r_[0, rates], np.r_[rates, 0]
    row = np.zeros(size, dtype=complex)
    column = row.copy()
    result = np.zeros((size, size), dtype=complex)
    for length in range(size, 0, -1):
        a = np.arange(size - length + 1)
        b = a + length - 1
        g = (length / 20) ** eta
        numerator = rhs[a, b] + dt * (right[b] * row[a] + left[a] * column[b])
        x = numerator / (pole + dt * g * (prefix[b] - prefix[a]))
        result[a, b] = x
        row[a] += g * x
        column[b] += g * x
    return result


def numpy_counts(logs, eta, order, substeps, coefficients):
    coef = coefficients[str(order)]
    size = len(logs) + 1
    y = np.zeros((size, size))
    y[0, -1] = 1
    for _ in range(substeps):
        for a, p in zip(coef["alpha"], coef["theta"]):
            y -= (
                2
                * (
                    complex(*a)
                    * numpy_solve(y, np.exp(logs), eta, complex(*p), 24 / substeps)
                ).real
            )
        y *= coef["alpha0"]
    return y


def main():
    tick = time.perf_counter()
    cfg = read(FAMILY / "reproduce/launch.json")
    prior = REPO / cfg["prior_family"]
    inputs = read(prior / "results/runtime-inputs.json")
    summary = read(FAMILY / "results/runtime-summary.json")
    assert summary["complete"]
    coefficients = read(FAMILY / "reproduce/coefficients.json")
    rows = []
    outputs = {}
    for row in summary["cases"]:
        if row["status"] != "completed":
            rows.append(dict(name=row["name"], status=row["status"]))
            continue
        item = inputs["cases"][row["input_index"]]
        sequence = item["sequence"]
        size = len(sequence)
        beta = inputs["initial_coefficients"]
        context = math.log(
            item["substrate_concentration_um"]
            / item["enzyme_substrate_denominator"]
            / 0.2
        )
        logs = np.array(
            [
                beta[0]
                + (beta[AA.index(a)] if a != "A" else 0)
                + (beta[19 + AA.index(b)] if b != "A" else 0)
                + context * beta[-1]
                for a, b in zip(sequence[:-1], sequence[1:])
            ]
        )
        archive = np.load(FAMILY / f"results/{row['name']}.npz")
        close(logs, archive["log_rates"])
        counts, gradient = archive["counts"], archive["gradient"]
        indices = np.arange(size)
        length = indices[None, :] - indices[:, None] + 1
        valid = length > 0
        weight = (
            np.sin(indices[:, None] + 1) * np.cos(indices[None, :] + 0.3) + 1.1
        ) / size
        replay = numpy_counts(
            logs, row["eta"], row["order"], row["substeps"], coefficients
        )
        checks = dict(
            name=row["name"],
            forward_replay_error=close(counts, replay),
            mass_error=close((counts * length * valid).sum(), size, 1e-8),
            readout_error=close((counts * weight).sum(), row["readout"]),
            negative_residue_mass=float(
                -np.minimum(counts, 0).__mul__(length * valid).sum() / size
            ),
        )
        assert (counts[~valid] == 0).all()
        assert int((counts[valid] < 0).sum()) == row["negative_count"]
        close(counts[valid].min(), row["minimum_count"])
        oldpath = prior / f"results/runtime-{row['case_id']:02d}-exact.npz"
        if oldpath.exists():
            old = np.load(oldpath)
            checks["uniformization_count_error"] = close(counts, old["counts"])
            checks["uniformization_gradient_error"] = close(
                gradient, np.r_[old["log_rate_gradient"], old["eta_gradient"]], 1e-8
            )
        else:
            directions = [
                np.r_[np.sin(np.arange(size - 1) + 1), 0],
                np.r_[np.cos(np.arange(size - 1) + 1), 0],
                np.r_[np.zeros(size - 1), 1],
            ]
            finite = []
            for d in directions:
                d = d / np.linalg.norm(d)
                h = 1e-5
                up = numpy_counts(
                    logs + h * d[:-1],
                    row["eta"] + h * d[-1],
                    row["order"],
                    row["substeps"],
                    coefficients,
                )
                down = numpy_counts(
                    logs - h * d[:-1],
                    row["eta"] - h * d[-1],
                    row["order"],
                    row["substeps"],
                    coefficients,
                )
                fd = float(((up - down) * weight).sum() / (2 * h))
                finite.append(
                    dict(
                        finite_difference=fd,
                        autograd_projection=float(gradient @ d),
                        error=close(fd, gradient @ d, 1e-7),
                    )
                )
            checks["directional_finite_differences"] = finite
        if row["eta"] == 0:
            rates = np.exp(logs)
            prefix = np.r_[0, np.cumsum(rates)]
            internal = np.where(valid, prefix[None, :] - prefix[:, None], 0)
            boundary = -np.expm1(-24 * rates)
            analytic = (
                np.r_[1, boundary][:, None]
                * np.r_[boundary, 1][None, :]
                * np.exp(-24 * internal)
                * valid
            )
            checks["independent_cut_count_error"] = close(counts, analytic)
            contribution = analytic * weight
            cum = contribution.cumsum(0)[:, ::-1].cumsum(1)[:, ::-1]
            h = 24 * rates
            analytic_gradient = (
                h
                * np.exp(-h)
                / (-np.expm1(-h))
                * (contribution.sum(0)[:-1] + contribution.sum(1)[1:])
                - h * cum[np.arange(size - 1), np.arange(1, size)]
            )
            checks["independent_cut_gradient_error"] = close(
                gradient[:-1], analytic_gradient
            )
        rows.append(checks)
        outputs[row["case_id"], row["order"], row["substeps"]] = (counts, gradient)
    comparisons = []
    for case in [14, 15, 19, 20]:
        primary = outputs[case, 48, 2]
        for order in [16, 48]:
            for steps in [1, 2]:
                value = outputs[case, order, steps]
                comparisons.append(
                    dict(
                        case_id=case,
                        order=order,
                        substeps=steps,
                        count_error_vs_cram48_two_steps=close(value[0], primary[0]),
                        gradient_error_vs_cram48_two_steps=close(
                            value[1], primary[1], 1e-8
                        ),
                    )
                )
    result = dict(
        all_replay_checks_pass=True,
        biological_log_admission=False,
        cases=rows,
        formerly_timed_out_comparisons=comparisons,
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="All saved forward outputs; full gradients against17 existing uniformization cases and eta0 formulas; three independent FD directions on16 order/substep cases without uniformization outputs. Same rational approximation NumPy replay does not certify true error universally. Negativity retained.",
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["cases", "formerly_timed_out_comparisons"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
