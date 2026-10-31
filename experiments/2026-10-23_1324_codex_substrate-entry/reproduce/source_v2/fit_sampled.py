"""Fixed-budget continuous-trajectory hybrid fit; no test evaluation."""

import base64
import gzip
import json
import math
import resource
import time

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
from fit import precision
from sampled import training_gradient


def main():
    tick = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((FAMILY / "reproduce/sampled-fitting.json").read_text())
    for name, h in config["source_hashes"].items():
        assert sha(FAMILY / name) == h
    cfg, groups = load_data()
    groups = [prepare_points(g) for g in groups]
    design, basis, audit = designs(groups, "sequence_context")
    assert torch.equal(basis, torch.eye(40, dtype=torch.float64))
    assert not (FAMILY / "results/fit-entry_sampled.json").exists()
    assert not (FAMILY / "results/sampled-trace.jsonl.gz").exists()
    initial = torch.zeros(41, dtype=torch.float64)
    initial[0] = math.log(0.01)
    theta = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([theta], lr=0.03)
    generator = torch.Generator().manual_seed(config["seed"])
    coverage = torch.zeros(41, dtype=torch.bool)
    history = []
    checks = {}
    checkpoints = {"0": initial.tolist()}
    work = {}
    with gzip.open(FAMILY / "results/sampled-trace.jsonl.gz", "wt") as trace:
        for step in range(1, config["updates"] + 1):
            optimizer.zero_grad()
            before = theta.detach().tolist()
            rng = (
                base64.b64encode(generator.get_state().numpy().tobytes()).decode()
                if step in (1, 100, 500, 1000, 2000)
                else None
            )
            with torch.no_grad():
                _, gradient, counts = training_gradient(
                    theta,
                    groups,
                    design,
                    order=config["order"],
                    count=config["samples"],
                    generator=generator,
                )
            assert torch.isfinite(gradient).all()
            theta.grad = gradient
            coverage |= gradient != 0
            optimizer.step()
            for k, v in counts.items():
                work[k] = work.get(k, 0) + v
            trace.write(
                json.dumps(
                    dict(
                        step=step,
                        before=before,
                        after=theta.detach().tolist(),
                        gradient=gradient.tolist(),
                        rng_before=rng,
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
            if step % 100 == 0:
                loss, data = entry_objective(
                    theta, groups, design, entry=True, order=config["order"]
                )
                exact = torch.autograd.grad(loss, theta)[0]
                history.append(
                    dict(
                        step=step,
                        objective=float(loss.detach()),
                        data_mse=float(data.detach()),
                        sampled_gradient_norm=float(gradient.norm()),
                        exact_gradient_norm=float(exact.norm()),
                        wall_seconds=time.perf_counter() - tick,
                    )
                )
                print(history[-1], flush=True)
            if step in (100, 500, 1000, 2000):
                checkpoints[str(step)] = theta.detach().tolist()
                checks[str(step)] = precision(
                    theta, groups, design, True, config["order"]
                )
                if (
                    checks[str(step)]["max_log_amount_difference"]
                    > cfg["log_amount_tolerance"]
                    or checks[str(step)]["max_gradient_difference"]
                    > cfg["gradient_tolerance"]
                ):
                    dump(
                        FAMILY / "results/precision-failure-sampled.json",
                        dict(
                            step=step,
                            checkpoints=checkpoints,
                            checks=checks,
                            history=history,
                            work=work,
                        ),
                    )
                    raise RuntimeError("Frozen precision gate failed; no rescue")
    loss, data = entry_objective(
        theta, groups, design, entry=True, order=config["order"]
    )
    gradient = torch.autograd.grad(loss, theta)[0]
    changed = theta.detach() != initial
    assert coverage.all() and changed.all()
    state = optimizer.state[theta]
    result = dict(
        coefficients=theta.detach().tolist(),
        entry=True,
        feature_audit=audit,
        history=history,
        checkpoints=checkpoints,
        precision_checks=checks,
        objective=float(loss.detach()),
        data_mse=float(data.detach()),
        final_gradient=gradient.tolist(),
        gradient_coverage=coverage.tolist(),
        parameters_changed=changed.tolist(),
        optimizer_state={
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in state.items()
        },
        fit_wall_seconds=time.perf_counter() - tick,
        work=work,
        seed=config["seed"],
        samples=config["samples"],
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        test_not_evaluated=True,
    )
    dump(FAMILY / "results/fit-entry_sampled.json", result)
    print(
        {k: result[k] for k in ("objective", "data_mse", "fit_wall_seconds", "work")},
        flush=True,
    )


if __name__ == "__main__":
    main()
