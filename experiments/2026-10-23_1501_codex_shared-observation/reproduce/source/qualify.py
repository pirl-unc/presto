"""Profile/gradient mathematics and fixed-budget sampled score qualification."""

import json
import math
import resource
import time

import numpy as np
import torch

from observation import (
    FAMILY,
    LN10,
    center,
    dump,
    load_data,
    log_amount,
    objective,
    sha,
)
from presto.models.independent_cut import (
    log_fragment_expectation,
    partition_contains,
    sample_partitions,
)


def prepare(theta, gamma, groups):
    records = []
    integrated = torch.zeros_like(theta)
    for g in groups:
        if g["split"] != "train":
            continue
        logs = g["x"] @ theta
        value = log_amount(theta, g) + g["z"] @ gamma
        residual = center(value - g["y"], g["w"])
        coefficient = 2 * g["w"] * residual / LN10
        assert abs(float(coefficient.sum())) < 1e-10
        for t in g["t"].unique(sorted=True).tolist():
            mask = g["t"] == t
            spans, inverse = torch.stack((g["a"][mask], g["b"][mask]), 1).unique(
                dim=0, sorted=True, return_inverse=True
            )
            starts, ends = spans.T
            combined = torch.zeros(len(spans), dtype=torch.float64)
            combined.scatter_add_(0, inverse, coefficient[mask])
            times = torch.full((len(spans),), t, dtype=torch.float64)
            mass = log_fragment_expectation(logs, starts, ends, times).exp()
            assert (mass > 0).all() and torch.isfinite(mass).all()
            h = logs.exp() * t
            cut_score = h * torch.exp(-h) / (-torch.expm1(-h))
            scores = torch.zeros((len(spans), len(logs)), dtype=torch.float64)
            for j, (a, b) in enumerate(spans.tolist()):
                scores[j, a : b - 1] = -h[a : b - 1]
                if a:
                    scores[j, a - 1] = cut_score[a - 1]
                if b <= len(logs):
                    scores[j, b - 1] = cut_score[b - 1]
            conditional = scores @ g["x"]
            integrated += combined @ conditional
            records.append(
                dict(
                    condition=g["condition"],
                    time=t,
                    logs=logs,
                    mass=mass,
                    coefficient=combined,
                    starts=starts,
                    ends=ends,
                    conditional=conditional,
                    hazard=h,
                    cut_score=cut_score,
                    x=g["x"],
                )
            )
    return records, integrated


def estimate(records, count, generator):
    rb = torch.zeros(40, dtype=torch.float64)
    full = torch.zeros_like(rb)
    zero = events = partitions = bonds = terms = 0
    for record in records:
        cuts, _ = sample_partitions(
            record["logs"], record["time"], count, generator=generator
        )
        present = partition_contains(cuts, record["starts"], record["ends"])
        occurrence = present.sum(0).double()
        factor = record["coefficient"] / record["mass"]
        rb += (occurrence / count * factor) @ record["conditional"]
        score = torch.where(cuts, record["cut_score"], -record["hazard"]) @ record["x"]
        reward = present.double() @ factor
        baseline = (reward.sum() - reward) / (count - 1)
        full += ((reward - baseline)[:, None] * score).mean(0)
        zero += int((occurrence == 0).sum())
        events += len(occurrence)
        partitions += count
        bonds += cuts.numel()
        terms += present.numel()
    return (
        rb,
        full,
        dict(
            zero_span_times=zero,
            span_times=events,
            partitions=partitions,
            bond_draws=bonds,
            presence_terms=terms,
        ),
    )


def main():
    tick = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((FAMILY / "reproduce/qualification.json").read_text())
    for name, h in config["source_hashes"].items():
        assert sha(FAMILY / name) == h
    cfg, groups, feature, initial = load_data()
    dump(FAMILY / "results/response-feature-basis.json", feature)
    gamma = torch.zeros(
        feature["allocated_dimensions"], dtype=torch.float64, requires_grad=True
    )
    theta = initial.clone().requires_grad_()
    loss, data = objective(theta, gamma, groups)
    grads = torch.autograd.grad(loss, (theta, gamma))
    errors = []
    profile_errors = []
    dense_lstsq_errors = []
    for g in groups:
        rates = np.exp(g["x"].numpy() @ initial.numpy())
        independent = []
        for a, b, t in zip(g["a"].tolist(), g["b"].tolist(), g["t"].tolist()):
            value = -t * math.fsum(rates[a : b - 1])
            if a:
                value += math.log(-math.expm1(-rates[a - 1] * t))
            if b <= len(rates):
                value += math.log(-math.expm1(-rates[b - 1] * t))
            independent.append(value / LN10)
        original = log_amount(initial, g).numpy()
        errors.append(float(np.max(np.abs(original - independent))))
        y = g["y"].numpy()
        w = g["w"].numpy()
        dense_offset = np.linalg.lstsq(
            np.sqrt(w)[:, None], np.sqrt(w) * (y - original), rcond=None
        )[0][0]
        offset = math.fsum(float(a*b) for a,b in zip(w,y-original))/math.fsum(w)
        dense_lstsq_errors.append(abs(float(dense_offset)-offset))
        residual = center(torch.tensor(original) - g["y"], g["w"]).numpy()
        profile_errors.append(float(np.max(np.abs(residual - (original + offset - y)))))
        assert (
            np.max(
                np.abs(
                    center(torch.tensor(original + 17.0) - g["y"], g["w"]).numpy()
                    - residual
                )
            )
            < 1e-12
        )
    assert max(errors) < 1e-10 and max(profile_errors) < 1e-12
    vector = torch.cat((theta.detach(), gamma.detach())).numpy()
    gradient = torch.cat(grads).numpy()
    finite = []
    h = 1e-5
    for i in range(len(vector)):
        a = vector.copy()
        b = vector.copy()
        a[i] += h
        b[i] -= h
        va = objective(torch.from_numpy(a[:40]), torch.from_numpy(a[40:]), groups)[0]
        vb = objective(torch.from_numpy(b[:40]), torch.from_numpy(b[40:]), groups)[0]
        finite.append(float((va - vb) / (2 * h)))
    fd_error = float(np.max(np.abs(np.array(finite) - gradient)))
    assert fd_error < 1e-6
    qualification = dict(
        source_counts={
            s: sum(len(g["rows"]) for g in groups if g["split"] == s)
            for s in ("train", "val")
        },
        parameters=40 + len(gamma),
        response_features=feature,
        original_amount_max_error=max(errors),
        profile_lstsq_error=max(profile_errors),
        dense_lstsq_offset_roundoff=max(dense_lstsq_errors),
        full_coordinate_finite_difference_error=fd_error,
        initial_profile_mse=float(data),
        initial_total_loss=float(loss),
        initial_gradient=gradient.tolist(),
        mathematical_checks_passed=True,
    )
    dump(FAMILY / "results/mathematical-qualification.json", qualification)
    print(
        json.dumps(
            {
                k: qualification[k]
                for k in (
                    "source_counts",
                    "parameters",
                    "initial_profile_mse",
                    "full_coordinate_finite_difference_error",
                )
            }
        ),
        flush=True,
    )
    probe = []
    uniform = initial.clone()
    uniform.zero_()
    uniform[0] = math.log(0.01)
    for state, values in (
        ("uniform_0p01", uniform),
        ("saved_ratio_initialization", initial),
    ):
        param = values.clone().requires_grad_()
        response = gamma.detach().clone().requires_grad_()
        loss, _ = objective(param, response, groups)
        exact = torch.autograd.grad(loss, param)[0]
        records, integrated = prepare(param.detach(), response.detach(), groups)
        reg = param.detach().clone() * 0.01
        reg[0] = 0
        integration_error = float((integrated + reg - exact).abs().max())
        assert integration_error < 1e-10
        for count in config["sample_counts"]:
            begin = time.perf_counter()
            rb_values = []
            full_values = []
            work = []
            for rep in range(config["groups"]):
                gen = torch.Generator().manual_seed(
                    config["seed"]
                    + rep
                    + 1000 * count
                    + 1000000 * int(state.startswith("saved"))
                )
                rb, full, stats = estimate(records, count, gen)
                rb_values.append(rb + reg)
                full_values.append(full + reg)
                work.append(stats)
            values_rb = torch.stack(rb_values)
            values_full = torch.stack(full_values)
            norm = float(exact.norm())
            mean_error = float((values_rb.mean(0) - exact).norm()) / norm
            record = dict(
                state=state,
                count=count,
                exact_kinetic_gradient=exact.tolist(),
                integrated_score_error=integration_error,
                exact_gradient_norm=norm,
                rb_gradients=values_rb.tolist(),
                full_gradients=values_full.tolist(),
                rb_mean_relative_error=mean_error,
                rb_rms_relative_error=float(
                    (values_rb - exact).square().sum(1).mean().sqrt()
                )
                / norm,
                full_mean_relative_error=float((values_full.mean(0) - exact).norm())
                / norm,
                full_rms_relative_error=float(
                    (values_full - exact).square().sum(1).mean().sqrt()
                )
                / norm,
                mean_zero_span_time_fraction=float(
                    np.mean([r["zero_span_times"] / r["span_times"] for r in work])
                ),
                minimum_probability=min(float(r["mass"].min()) for r in records),
                maximum_probability=max(float(r["mass"].max()) for r in records),
                work=work,
                seconds=time.perf_counter() - begin,
                finite_rb_gradient=bool(torch.isfinite(values_rb).all()),
                gate_pass=count == 512
                and mean_error <= config["max_mean_relative_error"]
                and bool(torch.isfinite(values_rb).all()),
            )
            probe.append(record)
            dump(FAMILY / "results/sampled-gradient-probes.json", probe)
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "state",
                            "count",
                            "rb_mean_relative_error",
                            "rb_rms_relative_error",
                            "mean_zero_span_time_fraction",
                            "seconds",
                            "gate_pass",
                        )
                    }
                ),
                flush=True,
            )
    result = dict(
        all_mathematical_checks_pass=True,
        sampled_fit_gate_pass=all(r["gate_pass"] for r in probe if r["count"] == 512),
        gate="Both fixed states: finite hybrid RB gradients and mean kinetic-gradient error <=20% at K512,16 groups; no epsilon amount or count imputation",
        sampled_objective_scope="Exact profile residuals and fragment-mass denominators; complete-partition REINFORCE and Rao-Blackwell conditional-score estimators. Not pure simulation-only fitting.",
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    dump(FAMILY / "results/qualification-summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
