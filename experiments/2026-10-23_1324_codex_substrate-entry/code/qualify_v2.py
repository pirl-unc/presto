"""Qualify the registered fast-entry quadrature amendment on training data."""

import importlib.util
import json
import math
import resource
import time

import numpy as np
import torch

from common import (
    FAMILY,
    designs,
    dump,
    entry_objective,
    load_data,
    prepare_points,
    sha,
)
from qualify import ctmc
from presto.models.substrate_entry import log_entry_fragment_expectation


def main():
    tick = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    registration = json.loads((FAMILY / "reproduce/numerical-v2.json").read_text())
    assert sha(__file__) == registration["source_sha256"]
    cfg, groups = load_data()
    groups = [prepare_points(g) for g in groups]
    assert not (FAMILY / "results/qualification-v2.json").exists()
    math_rows = []
    logs = np.log([0.2, 0.7])
    for entry in (0.01, 1.0, 63.999, 64.001, 100.0, 1000.0, 1.0e6):
        for t in (0.1, 1.0, 6.0):
            spans, expected, jac = ctmc(logs, math.log(entry), t)
            starts = torch.tensor([a for a, b in spans])
            ends = torch.tensor([b for a, b in spans])
            times = torch.full((len(spans),), t, dtype=torch.float64)
            theta = torch.tensor(
                [*logs, math.log(entry)], dtype=torch.float64, requires_grad=True
            )
            actual = log_entry_fragment_expectation(
                theta[:2], theta[2], starts, ends, times, order=64
            ).exp()
            coefficients = torch.linspace(0.2, 1.3, len(spans), dtype=torch.float64)
            gradient = (
                torch.autograd.grad(actual @ coefficients, theta)[0].detach().numpy()
            )
            math_rows.append(
                dict(
                    entry=entry,
                    time=t,
                    max_log_amount_error=float(
                        np.max(np.abs(actual.detach().log().numpy() - np.log(expected)))
                    ),
                    max_gradient_error=float(
                        np.max(np.abs(gradient - jac @ coefficients.numpy()))
                    ),
                    mass_error=abs(
                        float((actual.detach() * (ends - starts)).sum()) - 3
                    ),
                )
            )
    dump(FAMILY / "results/ctmc-v2.json", math_rows)
    assert max(r["max_log_amount_error"] for r in math_rows) < 1e-6
    assert max(r["max_gradient_error"] for r in math_rows) < 1e-7
    assert max(r["mass_error"] for r in math_rows) < 1e-11
    old = json.loads(open(cfg["previous_family"] + "/results/fits.json").read())
    failure = json.loads(
        (
            FAMILY
            / "attempts/fixed-age-quadrature/results/precision-failure-entry_constant.json"
        ).read_text()
    )
    failed = failure["checkpoints"]["500"]
    design, basis, audit = designs(groups, "sequence_context")
    assert torch.equal(basis, torch.eye(40, dtype=torch.float64))
    states = [
        ("initial", old["sequence_context"]["checkpoints"]["0"]),
        ("previous_fit", old["sequence_context"]["coefficients"]),
        ("failed_constant", [failed[0]] + [0.0] * 39),
    ]
    train = [(g, x) for g, x in zip(groups, design) if g["split"] == "train"]
    numerical = []
    timings = []
    for state, weights in states:
        entries = registration["entry_rates"] + (
            [math.exp(failed[1])] if state == "failed_constant" else []
        )
        for entry in entries:
            records = {}
            for order in registration["orders"]:
                theta = torch.tensor(
                    [*weights, math.log(entry)], dtype=torch.float64, requires_grad=True
                )
                now = time.perf_counter()
                loss, _ = entry_objective(
                    theta, groups, design, entry=True, order=order
                )
                gradient = torch.autograd.grad(loss, theta)[0]
                with torch.no_grad():
                    values = torch.cat(
                        [
                            log_entry_fragment_expectation(
                                x @ theta[:-1],
                                theta[-1],
                                g["point_starts"],
                                g["point_ends"],
                                g["point_times"],
                                order=order,
                                validate=False,
                            )
                            for g, x in train
                        ]
                    )
                records[order] = (
                    float(loss.detach()),
                    gradient.numpy(),
                    values.numpy(),
                )
                timings.append(
                    dict(
                        state=state,
                        entry_rate=entry,
                        order=order,
                        seconds=time.perf_counter() - now,
                    )
                )
            reference = records[max(registration["orders"])]
            for order, (loss, gradient, values) in records.items():
                numerical.append(
                    dict(
                        state=state,
                        entry_rate=entry,
                        order=order,
                        objective=loss,
                        max_log_amount_difference=float(
                            np.max(np.abs(values - reference[2]))
                        ),
                        max_gradient_difference=float(
                            np.max(np.abs(gradient - reference[1]))
                        ),
                    )
                )
            print(
                state,
                entry,
                [
                    (r["order"], r["max_log_amount_difference"])
                    for r in numerical
                    if r["state"] == state and r["entry_rate"] == entry
                ],
                flush=True,
            )
    dump(FAMILY / "results/training-numerics-v2.json", numerical)
    dump(FAMILY / "results/numerical-runtime-v2.json", timings)
    eligible = [
        q
        for q in registration["orders"][:-1]
        if all(
            r["max_log_amount_difference"] < cfg["log_amount_tolerance"]
            and r["max_gradient_difference"] < cfg["gradient_tolerance"]
            for r in numerical
            if r["order"] == q
        )
    ]
    assert eligible, "No numerical order qualifies; preserve and replan"
    # Original initial probes used lambda=1. Verify their quadrature path remains bitwise identical.
    spec = importlib.util.spec_from_file_location(
        "presto.models._entry_original",
        FAMILY / "attempts/fixed-age-quadrature/substrate_entry.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = torch.tensor(
        states[0][1] + [0.0], dtype=torch.float64, requires_grad=True
    )
    unchanged = True
    for g, x in train:
        args = (
            x @ initial[:-1],
            initial[-1],
            g["point_starts"],
            g["point_ends"],
            g["point_times"],
        )
        a = module.log_entry_fragment_expectation(*args, order=32, validate=False)
        b = log_entry_fragment_expectation(*args, order=32, validate=False)
        ga = torch.autograd.grad(a.sum(), initial, retain_graph=True)[0]
        gb = torch.autograd.grad(b.sum(), initial)[0]
        unchanged &= torch.equal(a, b) and torch.equal(ga, gb)
    assert unchanged
    result = dict(
        passed=True,
        selected_order=min(eligible),
        initial_probe_values_and_gradients_bitwise_unchanged=unchanged,
        training_span_times=sum(len(g["point_starts"]) for g, x in train),
        test_not_evaluated=True,
        wall_seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    dump(FAMILY / "results/qualification-v2.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
