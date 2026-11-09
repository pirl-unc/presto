"""Train the declared partial analytic hybrid using complete sampled cut states."""

import gzip
import json
import math
import resource
import time

import torch

from common import FAMILY, designs, dump, load_data, metrics, objective, predict, sha
from presto.models.independent_cut import (
    log_fragment_expectation,
    partition_contains,
    sample_partitions,
)


def templates(groups, design):
    records = []
    for g, x in zip(groups, design):
        if g["split"] != "train":
            continue
        spans, inverse = torch.stack((g["starts"], g["ends"]), dim=1).unique(
            dim=0, sorted=True, return_inverse=True
        )
        records.append(
            dict(
                group=g,
                design=x,
                spans=spans,
                inverse=inverse,
                times=torch.cat((g["times"], g["refs"])).unique(sorted=True).tolist(),
            )
        )
    return records


def sampled_gradient(theta, records, basis, penalty, count, generator):
    gradient = torch.zeros_like(theta)
    partitions = 0
    bond_draws = 0
    presence_terms = 0
    for record in records:
        g, x = record["group"], record["design"]
        logs = x @ theta
        rates = logs.exp()
        coeff = 2 * g["weight"] * (predict(logs, g) - g["target"]) / math.log(10)
        starts, ends = record["spans"].T
        n = len(rates)
        inverse = record["inverse"]
        for t in record["times"]:
            combined = torch.zeros(len(starts), dtype=torch.float64)
            for observed, sign in ((g["times"], 1), (g["refs"], -1)):
                mask = observed == t
                combined.scatter_add_(0, inverse[mask], sign * coeff[mask])
            mass = log_fragment_expectation(
                logs,
                starts,
                ends,
                torch.full((len(starts),), t, dtype=torch.float64),
                validate=False,
            ).exp()
            if not (mass > 0).all():
                raise FloatingPointError(
                    "Exact normalization underflow; no pseudocount substitution"
                )
            h = rates * t
            cut_score = h * torch.exp(-h) / (-torch.expm1(-h))
            prefix = torch.cat(
                (
                    torch.zeros((1, x.shape[1]), dtype=torch.float64),
                    torch.cumsum(h[:, None] * x, dim=0),
                ),
                dim=0,
            )
            conditional = -(prefix[ends - 1] - prefix[starts])
            for index, valid in ((starts - 1, starts > 0), (ends - 1, ends <= n)):
                safe = index.clamp(0, n - 1)
                conditional += (
                    torch.where(valid, cut_score[safe], 0.0)[:, None] * x[safe]
                )
            cuts, _ = sample_partitions(logs, t, count, generator=generator)
            present = partition_contains(cuts, starts, ends)
            gradient += (present.double().mean(0) * combined / mass) @ conditional
            partitions += count
            bond_draws += cuts.numel()
            presence_terms += present.numel()
    original = basis @ theta
    reg = penalty * original
    reg[0] = 0
    gradient += basis.T @ reg
    return gradient, dict(
        partitions=partitions, bond_draws=bond_draws, presence_terms=presence_terms
    )


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((FAMILY / "reproduce/sampled-fit.json").read_text())
    assert sha(__file__) == config["sha256"]
    cfg, groups = load_data()
    design, basis, audit = designs(groups, "sequence_context")
    records = templates(groups, design)
    fits = json.loads((FAMILY / "results/fits.json").read_text())
    probes = json.loads((FAMILY / "results/gradient-probes-rb.json").read_text())
    checks = []
    with torch.no_grad():
        for state, values in [
            ("initial", fits["sequence_context"]["checkpoints"]["0"]),
            ("final", fits["sequence_context"]["basis_coefficients"]),
        ]:
            for count in (32, 512):
                generator = torch.Generator().manual_seed(
                    20261023 + 1000 * count + 1000000 * int(state == "final")
                )
                actual, _ = sampled_gradient(
                    torch.tensor(values, dtype=torch.float64),
                    records,
                    basis,
                    cfg["l2_nonintercept"],
                    count,
                    generator,
                )
                expected = next(
                    p for p in probes if p["state"] == state and p["count"] == count
                )["local_gradients"][0]
                error = float(
                    (actual - torch.tensor(expected, dtype=torch.float64)).abs().max()
                )
                assert error < 1e-10
                checks.append(dict(state=state, count=count, max_gradient_error=error))
    dump(FAMILY / "results/sampled-fast-gradient-check.json", checks)
    assert not (FAMILY / "results/sampled-fit.json").exists()
    theta = torch.nn.Parameter(
        torch.tensor(fits["sequence_context"]["checkpoints"]["0"], dtype=torch.float64)
    )
    optimizer = torch.optim.Adam([theta], lr=cfg["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    coverage = torch.zeros_like(theta, dtype=torch.bool)
    history = []
    checkpoints = {"0": theta.detach().tolist()}
    work = {}
    fit_started = time.monotonic()
    for step in range(1, config["updates"] + 1):
        optimizer.zero_grad()
        try:
            with torch.no_grad():
                gradient, counts = sampled_gradient(
                    theta,
                    records,
                    basis,
                    cfg["l2_nonintercept"],
                    config["count"],
                    generator,
                )
            assert torch.isfinite(gradient).all()
            coverage |= gradient != 0
            theta.grad = gradient
            optimizer.step()
        except (FloatingPointError, AssertionError):
            dump(
                FAMILY / "results/sampled-failure.json",
                dict(step=step, coefficients=theta.detach().tolist(), history=history),
            )
            raise
        for key, value in counts.items():
            work[key] = work.get(key, 0) + value
        if step % 100 == 0:
            with torch.no_grad():
                loss, data = objective(
                    theta, groups, design, basis, cfg["l2_nonintercept"]
                )
            assert torch.isfinite(loss)
            history.append(
                dict(
                    step=step,
                    objective=float(loss),
                    data_mse=float(data),
                    sampled_gradient_norm=float(gradient.norm()),
                    wall_seconds=time.monotonic() - fit_started,
                )
            )
            print(history[-1], flush=True)
        if step in (100, 500, 1000, 2000):
            checkpoints[str(step)] = theta.detach().tolist()
    loss, data = objective(theta, groups, design, basis, cfg["l2_nonintercept"])
    exact_gradient = torch.autograd.grad(loss, theta)[0]
    assert coverage.all()
    predictions = []
    with torch.no_grad():
        for g, x in zip(groups, design):
            values = predict(x @ theta, g).tolist()
            for r, v, w in zip(g["rows"], values, g["weight"].tolist()):
                predictions.append(
                    dict(
                        r,
                        variant="sampled_rb512_seed42",
                        prediction=v,
                        weight=w,
                        predicted_log10_signal=math.log10(r["reference_signal"]) + v,
                        predicted_signal=r["reference_signal"] * 10**v,
                    )
                )
    with gzip.open(FAMILY / "results/sampled-predictions.jsonl.gz", "wt") as f:
        for r in predictions:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
    result = dict(
        coefficients=(basis @ theta.detach()).tolist(),
        basis_coefficients=theta.detach().tolist(),
        checkpoints=checkpoints,
        history=history,
        gradient_coverage=coverage.tolist(),
        exact_final_gradient=exact_gradient.tolist(),
        objective=float(loss.detach()),
        data_mse=float(data.detach()),
        metrics={
            s: metrics([r for r in predictions if r["split"] == s])
            for s in ("train", "val", "test")
        },
        sampling_work=work,
        fit_wall_seconds=time.monotonic() - fit_started,
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        one_seed_only=True,
        estimator="Unconditional complete partitions; span-presence score with irrelevant bonds integrated; exact residual/probability hybrid",
    )
    dump(FAMILY / "results/sampled-fit.json", result)
    print({s: m["rmse"] for s, m in result["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
