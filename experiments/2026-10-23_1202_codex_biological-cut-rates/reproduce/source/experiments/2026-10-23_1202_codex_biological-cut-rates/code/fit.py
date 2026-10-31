"""Fixed final-checkpoint exact fits and declared controls, no test tuning."""

import gzip
import math
import resource
import time

import torch

from common import FAMILY, designs, dump, load_data, metrics, objective, predict


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg, groups = load_data()
    assert not (FAMILY / "results/fits.json").exists()
    audits = {}
    prepared = {}
    for variant in cfg["variants"]:
        design, basis, audit = designs(groups, variant)
        prepared[variant] = design, basis
        audits[variant] = audit
    assert (
        audits["sequence_context"]["allocated_dimensions"]
        > audits["sequence"]["allocated_dimensions"]
    ), "Context unsupported; preserve failure and declare a new launch contract"
    dump(FAMILY / "results/feature-audit.json", audits)
    print({k: v["allocated_dimensions"] for k, v in audits.items()}, flush=True)
    fits = {}
    all_predictions = []
    for variant in cfg["variants"]:
        start = time.monotonic()
        design, basis = prepared[variant]
        initial = torch.zeros(basis.shape[0], dtype=torch.float64)
        initial[0] = math.log(cfg["initial_rate_per_hour"])
        theta = torch.nn.Parameter(basis.T @ initial)
        optimizer = torch.optim.Adam([theta], lr=cfg["learning_rate"])
        coverage = torch.zeros_like(theta, dtype=torch.bool)
        history = []
        checkpoints = {"0": theta.detach().tolist()}
        for step in range(1, cfg["epochs"] + 1):
            optimizer.zero_grad()
            loss, data_loss = objective(
                theta, groups, design, basis, cfg["l2_nonintercept"]
            )
            if not torch.isfinite(loss):
                dump(
                    FAMILY / f"results/failed-{variant}.json",
                    dict(step=step, coefficients=theta.detach().tolist()),
                )
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            assert torch.isfinite(theta.grad).all()
            coverage |= theta.grad != 0
            optimizer.step()
            if step % 100 == 0:
                history.append(
                    dict(
                        step=step,
                        objective=float(loss.detach()),
                        data_mse=float(data_loss.detach()),
                        gradient_norm=float(theta.grad.norm()),
                    )
                )
            if step in (100, 500, 1000, 2000):
                checkpoints[str(step)] = theta.detach().tolist()
                print(variant, step, history[-1], flush=True)
        final_objective, final_data = objective(
            theta, groups, design, basis, cfg["l2_nonintercept"]
        )
        final_gradient = torch.autograd.grad(final_objective, theta)[0]
        changed = (theta.detach() - basis.T @ initial) != 0
        assert coverage.all() and changed.all(), (variant, coverage, changed)
        fits[variant] = dict(
            coefficients=(basis @ theta.detach()).tolist(),
            basis_coefficients=theta.detach().tolist(),
            feature_audit=audits[variant],
            history=history,
            checkpoints=checkpoints,
            final_objective=float(final_objective.detach()),
            final_data_mse=float(final_data.detach()),
            final_gradient=final_gradient.tolist(),
            gradient_coverage=coverage.tolist(),
            parameters_changed=changed.tolist(),
            wall_seconds=time.monotonic() - start,
        )
        for g, x in zip(groups, design):
            with torch.no_grad():
                p = predict(x @ theta, g).tolist()
            for r, v, w in zip(g["rows"], p, g["weight"].tolist()):
                all_predictions.append(
                    dict(
                        r,
                        variant=variant,
                        prediction=v,
                        weight=w,
                        predicted_log10_signal=math.log10(r["reference_signal"]) + v,
                        predicted_signal=r["reference_signal"] * 10**v,
                    )
                )
    # Same parent/condition/curve weighting, no fitted intercept at reference time.
    x = torch.cat(
        [
            torch.stack(
                (
                    torch.log(g["times"] / g["refs"]),
                    torch.log(g["times"] / g["refs"]) ** 2,
                ),
                dim=1,
            )
            for g in groups
            if g["split"] == "train"
        ]
    )
    y = torch.cat([g["target"] for g in groups if g["split"] == "train"])
    w = torch.cat([g["weight"] for g in groups if g["split"] == "train"])
    beta = torch.linalg.lstsq(x * w.sqrt()[:, None], y * w.sqrt()).solution
    fits["shared_time"] = dict(
        coefficients=beta.tolist(),
        training_design_rank=int(torch.linalg.matrix_rank(x)),
    )
    for g in groups:
        z = torch.log(g["times"] / g["refs"])
        shared = beta[0] * z + beta[1] * z**2
        for variant, values in (
            ("shared_time", shared.tolist()),
            ("no_change", [0.0] * len(z)),
        ):
            for r, v, ww in zip(g["rows"], values, g["weight"].tolist()):
                all_predictions.append(
                    dict(
                        r,
                        variant=variant,
                        prediction=v,
                        weight=ww,
                        predicted_log10_signal=math.log10(r["reference_signal"]) + v,
                        predicted_signal=r["reference_signal"] * 10**v,
                    )
                )
    scores = {
        variant: {
            split: metrics(
                [
                    r
                    for r in all_predictions
                    if r["variant"] == variant and r["split"] == split
                ]
            )
            for split in ("train", "val", "test")
        }
        for variant in fits.keys() | {"no_change"}
    }
    with gzip.open(FAMILY / "results/predictions.jsonl.gz", "wt") as f:
        for r in all_predictions:
            f.write(__import__("json").dumps(r, sort_keys=True, allow_nan=False) + "\n")
    dump(FAMILY / "results/fits.json", fits)
    dump(FAMILY / "results/metrics.json", scores)
    dump(
        FAMILY / "results/runtime.json",
        dict(
            wall_seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            optimizer_updates=cfg["epochs"] * len(cfg["variants"]),
            device="CPU float64 one thread",
            parameter_design_rows={
                v: sum(
                    len(x)
                    for x, g in zip(prepared[v][0], groups)
                    if g["split"] == "train"
                )
                for v in cfg["variants"]
            },
        ),
    )
    print(
        {
            v: {s: round(m["rmse"], 5) for s, m in splits.items()}
            for v, splits in scores.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
