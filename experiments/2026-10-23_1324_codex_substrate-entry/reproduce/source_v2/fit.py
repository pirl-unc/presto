"""Matched fixed-budget development fits; the previously used test set is closed."""

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
    entry_predict,
    load_data,
    metrics,
    prepare_points,
    sha,
)
from presto.models.substrate_entry import log_entry_fragment_expectation


def precision(theta, groups, design, entry, order):
    if not entry:
        return dict(max_log_amount_difference=0.0, max_gradient_difference=0.0)
    gradients = []
    values = []
    for q in (order, order * 2):
        loss, _ = entry_objective(theta, groups, design, entry=True, order=q)
        gradients.append(torch.autograd.grad(loss, theta)[0])
        with torch.no_grad():
            values.append(
                torch.cat(
                    [
                        log_entry_fragment_expectation(
                            x @ theta[:-1],
                            theta[-1],
                            g["point_starts"],
                            g["point_ends"],
                            g["point_times"],
                            order=q,
                            validate=False,
                        )
                        for g, x in zip(groups, design)
                        if g["split"] == "train"
                    ]
                )
            )
    return dict(
        max_log_amount_difference=float((values[0] - values[1]).abs().max()),
        max_gradient_difference=float((gradients[0] - gradients[1]).abs().max()),
    )


def main():
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((FAMILY / "reproduce/fitting.json").read_text())
    assert sha(__file__) == config["source_sha256"]
    cfg, groups = load_data()
    groups = [prepare_points(g) for g in groups]
    assert all(g["split"] in ("train", "val") for g in groups)
    assert not (FAMILY / "results/fits.json").exists()
    fits = {}
    predictions = []
    for variant in config["variants"]:
        entry = variant.startswith("entry_")
        kind = "constant" if variant.endswith("constant") else "sequence_context"
        design, basis, audit = designs(groups, kind)
        n = design[0].shape[1]
        assert torch.equal(basis, torch.eye(n, dtype=torch.float64))
        initial = torch.zeros(n + int(entry), dtype=torch.float64)
        initial[0] = math.log(0.01)
        theta = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([theta], lr=0.03)
        coverage = torch.zeros_like(theta, dtype=torch.bool)
        history = []
        checkpoints = {"0": initial.tolist()}
        checks = {}
        tick = time.perf_counter()
        for step in range(1, config["updates"] + 1):
            optimizer.zero_grad()
            loss, data = entry_objective(
                theta, groups, design, entry=entry, order=config["order"]
            )
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(theta.grad).all()
            coverage |= theta.grad != 0
            optimizer.step()
            if step % 100 == 0:
                history.append(
                    dict(
                        step=step,
                        objective=float(loss.detach()),
                        data_mse=float(data.detach()),
                        gradient_norm=float(theta.grad.norm()),
                        wall_seconds=time.perf_counter() - tick,
                    )
                )
                print(variant, history[-1], flush=True)
            if step in (100, 500, 1000, 2000):
                checkpoints[str(step)] = theta.detach().tolist()
                checks[str(step)] = precision(
                    theta, groups, design, entry, config["order"]
                )
                if (
                    checks[str(step)]["max_log_amount_difference"]
                    > cfg["log_amount_tolerance"]
                    or checks[str(step)]["max_gradient_difference"]
                    > cfg["gradient_tolerance"]
                ):
                    dump(
                        FAMILY / f"results/precision-failure-v2-{variant}.json",
                        dict(
                            step=step,
                            checkpoints=checkpoints,
                            checks=checks,
                            history=history,
                        ),
                    )
                    raise RuntimeError(
                        "Frozen numerical gate failed; no precision rescue under old contract"
                    )
        loss, data = entry_objective(
            theta, groups, design, entry=entry, order=config["order"]
        )
        gradient = torch.autograd.grad(loss, theta)[0]
        changed = theta.detach() != initial
        assert coverage.all() and changed.all()
        state = optimizer.state[theta]
        fits[variant] = dict(
            coefficients=theta.detach().tolist(),
            entry=entry,
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
        )
        dump(FAMILY / f"results/fit-{variant}.json", fits[variant])
        with torch.no_grad():
            for g, x in zip(groups, design):
                pred = entry_predict(
                    x @ (theta[:-1] if entry else theta),
                    theta[-1] if entry else None,
                    g,
                    config["order"],
                )
                for row, value, weight in zip(
                    g["rows"], pred.tolist(), g["weight"].tolist()
                ):
                    predictions.append(
                        dict(
                            row,
                            variant=variant,
                            prediction=value,
                            weight=weight,
                            predicted_log10_signal=math.log10(row["reference_signal"])
                            + value,
                            predicted_signal=row["reference_signal"] * 10**value,
                        )
                    )
    scores = {
        v: {
            split: metrics(
                [r for r in predictions if r["variant"] == v and r["split"] == split]
            )
            for split in ("train", "val")
        }
        for v in fits
    }
    dump(FAMILY / "results/fits.json", fits)
    dump(FAMILY / "results/metrics.json", scores)
    with gzip.open(FAMILY / "results/predictions.jsonl.gz", "wt") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    dump(
        FAMILY / "results/fitting-runtime.json",
        dict(
            updates=config["updates"] * len(config["variants"]),
            wall_seconds=time.perf_counter() - start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            test_not_evaluated=True,
        ),
    )
    print(
        {
            v: {s: round(m["rmse"], 6) for s, m in split.items()}
            for v, split in scores.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
