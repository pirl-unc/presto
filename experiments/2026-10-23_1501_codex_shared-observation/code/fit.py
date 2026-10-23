"""Fixed shared-observation fits and matched hybrid, with original controls."""

import gzip
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
    metrics,
    objective,
    predictions,
    sha,
)
from qualify import estimate, prepare
from presto.models.independent_cut import (
    log_fragment_expectation,
    partition_contains,
    sample_partitions,
)


def save_predictions(name, rows):
    with gzip.open(FAMILY / f"results/{name}-predictions.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return {s: metrics([r for r in rows if r["split"] == s]) for s in ("train", "val")}


def evaluate(theta, gamma, groups):
    with torch.no_grad():
        amounts = [log_amount(theta, g) for g in groups]
        responses = [
            g["z"] @ gamma if gamma.numel() else torch.zeros_like(g["y"])
            for g in groups
        ]
    return predictions(
        groups, [a + b for a, b in zip(amounts, responses)], amounts, responses
    )


def ridge(groups, base, kind, penalty):
    designs = [
        g["z"]
        if kind == "response"
        else torch.stack((g["t"].log10(), g["t"].log10().square()), 1)
        for g in groups
    ]
    d = designs[0].shape[1]
    gram = np.eye(d) * penalty / 2
    rhs = np.zeros(d)
    for g, x, b in zip(groups, designs, base):
        if g["split"] != "train":
            continue
        w = g["w"].numpy()
        z = x.numpy()
        z = z - (w[:, None] * z).sum(0) / w.sum()
        y = (g["y"] - b).numpy()
        y -= w @ y / w.sum()
        gram += (z * w[:, None]).T @ z
        rhs += z.T @ (w * y)
    coef = np.linalg.solve(gram, rhs)
    response = [x @ torch.tensor(coef) for x in designs]
    return coef, response


def fast_prepare(theta, gamma, groups):
    records = []
    for g in groups:
        if g["split"] != "train":
            continue
        logs = g["x"] @ theta
        error = center(log_amount(theta, g) + g["z"] @ gamma - g["y"], g["w"])
        coeff = 2 * g["w"] * error / LN10
        for template in g["templates"]:
            starts, ends = template["spans"].T
            t = template["time"]
            combined = torch.zeros(len(starts), dtype=torch.float64)
            combined.scatter_add_(0, template["inverse"], coeff[template["mask"]])
            mass = log_fragment_expectation(
                logs, starts, ends, torch.full((len(starts),), t, dtype=torch.float64)
            ).exp()
            if not bool((mass > 0).all()):
                raise FloatingPointError(
                    "A sampled exact denominator is zero; no imputation permitted"
                )
            h = logs.exp() * t
            cut_score = h * torch.exp(-h) / (-torch.expm1(-h))
            prefix = torch.cat(
                (
                    torch.zeros((1, 40), dtype=torch.float64),
                    (h[:, None] * g["x"]).cumsum(0),
                )
            )
            score = -(prefix[ends - 1] - prefix[starts])
            a = starts > 0
            b = ends <= len(logs)
            score[a] += cut_score[starts[a] - 1, None] * g["x"][starts[a] - 1]
            score[b] += cut_score[ends[b] - 1, None] * g["x"][ends[b] - 1]
            records.append(
                dict(
                    logs=logs,
                    time=t,
                    mass=mass,
                    coefficient=combined,
                    starts=starts,
                    ends=ends,
                    conditional=score,
                )
            )
    return records


def fast_estimate(records, count, generator):
    gradient = torch.zeros(40, dtype=torch.float64)
    zero = events = partitions = bonds = terms = 0
    for r in records:
        cuts, _ = sample_partitions(r["logs"], r["time"], count, generator=generator)
        present = partition_contains(cuts, r["starts"], r["ends"])
        occurrences = present.sum(0).double()
        gradient += (occurrences / count * r["coefficient"] / r["mass"]) @ r[
            "conditional"
        ]
        zero += int((occurrences == 0).sum())
        events += len(occurrences)
        partitions += count
        bonds += cuts.numel()
        terms += present.numel()
    return gradient, dict(
        zero_span_times=zero,
        span_times=events,
        partitions=partitions,
        bond_draws=bonds,
        presence_terms=terms,
    )


def main():
    tick = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((FAMILY / "reproduce/fitting.json").read_text())
    for name, h in config["source_hashes"].items():
        assert sha(FAMILY / name) == h
    qualification = json.loads(
        (FAMILY / "results/qualification-summary.json").read_text()
    )
    assert qualification["all_mathematical_checks_pass"]
    assert (
        sha(FAMILY / "results/qualification-summary.json")
        == config["qualification_sha256"]
    )
    _, groups, feature, initial = load_data()
    for g in groups:
        g["templates"] = []
        for t in g["t"].unique(sorted=True).tolist():
            mask = g["t"] == t
            spans, inverse = torch.stack((g["a"][mask], g["b"][mask]), 1).unique(
                dim=0, sorted=True, return_inverse=True
            )
            g["templates"].append(dict(time=t, mask=mask, spans=spans, inverse=inverse))
    rank = feature["allocated_dimensions"]
    zero = torch.zeros(rank, dtype=torch.float64)
    fast_checks = []
    for perturb in (0.0, 0.01):
        theta = initial + perturb * torch.sin(torch.arange(40, dtype=torch.float64))
        slow, _ = prepare(theta, zero, groups)
        fast = fast_prepare(theta, zero, groups)
        error = max(
            float((a["conditional"] - b["conditional"]).abs().max())
            for a, b in zip(slow, fast)
        )
        gen = torch.Generator().manual_seed(777)
        a, _, _ = estimate(slow, 32, gen)
        gen = torch.Generator().manual_seed(777)
        b, _ = fast_estimate(fast, 32, gen)
        grad_error = float((a - b).abs().max())
        assert error < 1e-10 and grad_error < 1e-9
        fast_checks.append(
            dict(
                perturbation=perturb,
                conditional_score_error=error,
                same_draw_gradient_error=grad_error,
            )
        )
    dump(FAMILY / "results/fast-score-qualification.json", fast_checks)
    assert not (FAMILY / "results/fits.json").exists()
    fits = {}
    all_metrics = {}
    fixed_amount = [log_amount(initial, g) for g in groups]
    zeros = [torch.zeros_like(g["y"]) for g in groups]
    for name, base, kind, penalty in [
        ("response_only", zeros, "response", config["penalty"]),
        ("time_only", zeros, "time", 0.0),
        ("fixed_ratio_with_response", fixed_amount, "response", config["penalty"]),
    ]:
        coef, response = ridge(groups, base, kind, penalty)
        rows = predictions(
            groups,
            [b + r for b, r in zip(base, response)],
            fixed_amount if name.startswith("fixed") else None,
            response,
        )
        all_metrics[name] = save_predictions(name, rows)
        fits[name] = dict(
            coefficients=coef.tolist(),
            kind=kind,
            penalty=penalty,
            solver="profiled weighted ridge normal equations",
        )
    all_metrics["fixed_ratio"] = save_predictions(
        "fixed_ratio", evaluate(initial, torch.zeros(0, dtype=torch.float64), groups)
    )
    all_metrics["constant_signal"] = save_predictions(
        "constant_signal", predictions(groups, zeros)
    )
    fits["fixed_ratio"] = dict(theta=initial.tolist())
    fits["constant_signal"] = dict(parameters=0)
    variants = ["constant_response", "sequence_response"]
    if qualification["sampled_fit_gate_pass"]:
        variants.append("sampled_sequence_response")
    for name in variants:
        theta = torch.nn.Parameter(initial.clone())
        gamma = torch.nn.Parameter(
            torch.zeros(0 if name == "constant_response" else rank, dtype=torch.float64)
        )
        parameters = [theta] + ([gamma] if gamma.numel() else [])
        optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
        coverage = torch.zeros(40 + gamma.numel(), dtype=torch.bool)
        history = []
        states = {}
        now = time.perf_counter()
        generator = torch.Generator().manual_seed(config["sample_seed"])
        totalwork = dict(partitions=0, bond_draws=0, presence_terms=0)
        with gzip.open(FAMILY / f"results/{name}-updates.jsonl.gz", "wt") as stream:
            for step in range(1, config["steps"] + 1):
                optimizer.zero_grad()
                value, data = objective(theta, gamma, groups, config["penalty"])
                assert torch.isfinite(value)
                value.backward()
                work = None
                if name.startswith("sampled"):
                    records = fast_prepare(theta.detach(), gamma.detach(), groups)
                    sampled, work = fast_estimate(
                        records, config["sample_count"], generator
                    )
                    reg = theta.detach().clone() * config["penalty"]
                    reg[0] = 0
                    theta.grad = sampled + reg
                    for k in totalwork:
                        totalwork[k] += work[k]
                gradient = torch.cat([p.grad for p in parameters])
                assert torch.isfinite(gradient).all()
                coverage |= gradient != 0
                before = torch.cat([p.detach() for p in parameters]).tolist()
                optimizer.step()
                stream.write(
                    json.dumps(
                        dict(
                            step=step,
                            preupdate_parameters=before,
                            gradient=gradient.tolist(),
                            postupdate_parameters=torch.cat(
                                [p.detach() for p in parameters]
                            ).tolist(),
                            preupdate_loss=float(value.detach()),
                            preupdate_mse=float(data.detach()),
                            work=work,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                if step in (1, 100, 500, 1000, config["steps"]):
                    states[str(step)] = dict(
                        theta=theta.detach().tolist(), gamma=gamma.detach().tolist()
                    )
                if step % 100 == 0:
                    row = dict(
                        step=step,
                        preupdate_loss=float(value.detach()),
                        preupdate_mse=float(data.detach()),
                        seconds=time.perf_counter() - now,
                    )
                    history.append(row)
                    print(name, row, flush=True)
        final, data = objective(theta, gamma, groups, config["penalty"])
        grad = torch.cat(torch.autograd.grad(final, parameters))
        all_metrics[name] = save_predictions(
            name, evaluate(theta.detach(), gamma.detach(), groups)
        )
        record = dict(
            theta=theta.detach().tolist(),
            gamma=gamma.detach().tolist(),
            initial_theta=initial.tolist(),
            initial_gamma=torch.zeros_like(gamma).tolist(),
            final_loss=float(final.detach()),
            final_mse=float(data.detach()),
            exact_final_gradient=grad.tolist(),
            gradient_coverage=coverage.tolist(),
            changed_parameter_count=int(
                (torch.cat([theta.detach() - initial, gamma.detach()]) != 0).sum()
            ),
            history=history,
            checkpoints=states,
            steps=config["steps"],
            seconds=time.perf_counter() - now,
            work=totalwork,
            optimizer_states=[
                {
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in optimizer.state[p].items()
                }
                for p in parameters
            ],
        )
        fits[name] = record
        dump(FAMILY / f"results/{name}-fit.json", record)
        dump(FAMILY / "results/fits-partial.json", fits)
        dump(FAMILY / "results/metrics-partial.json", all_metrics)
        print(name, all_metrics[name], flush=True)
    sensitivity = []
    for name in variants:
        fit = fits[name]
        theta = torch.tensor(fit["theta"], dtype=torch.float64)
        gamma = torch.tensor(fit["gamma"], dtype=torch.float64)
        for shift in config["log10_rate_scale_shifts"]:
            perturbed = theta.clone()
            perturbed[0] += shift * LN10
            train = [g for g in groups if g["split"] == "train"]
            with torch.no_grad():
                value, data = objective(perturbed, gamma, train, config["penalty"])
                amounts = torch.cat([log_amount(perturbed, g) for g in train])
            sensitivity.append(
                dict(
                    variant=name,
                    log10_rate_scale_shift=shift,
                    profile_mse=float(data),
                    total_loss=float(value),
                    minimum_log10_amount=float(amounts.min()),
                    median_log10_amount=float(amounts.median()),
                    below_1e_20_count=int((amounts < -20).sum()),
                )
            )
    dump(FAMILY / "results/scale-sensitivity.json", sensitivity)
    dump(FAMILY / "results/fits.json", fits)
    dump(FAMILY / "results/metrics.json", all_metrics)
    dump(
        FAMILY / "results/fitting-runtime.json",
        dict(
            seconds=time.perf_counter() - tick,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            sampled_fit_performed=qualification["sampled_fit_gate_pass"],
            variants=variants,
        ),
    )


if __name__ == "__main__":
    main()
