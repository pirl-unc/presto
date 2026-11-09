"""Prospectively gated conditional-importance biological comparison."""

import gzip
import json
import resource
import time

import torch

from conditional import estimate
from fit import evaluate, fast_prepare, save_predictions
from observation import FAMILY, LN10, dump, load_data, log_amount, objective, sha


def main():
    tick = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads((FAMILY / "reproduce/conditional-fitting.json").read_text())
    for name, digest in cfg["source_hashes"].items():
        assert sha(FAMILY / name) == digest
    for name, digest in cfg["artifact_hashes"].items():
        assert sha(FAMILY / name) == digest
    assert sha(cfg["initial_fits_path"]) == cfg["initial_fits_sha256"]
    gate = json.loads((FAMILY / "results/conditional-qualification.json").read_text())
    assert gate["gate_pass"] and gate["primary_proposal"] == cfg["proposal"]
    name = "conditional_sequence_response"
    destination = FAMILY / f"results/{name}-updates.jsonl.gz"
    assert not destination.exists()
    _, groups, feature, initial = load_data()
    for g in groups:
        g["templates"] = []
        for t in g["t"].unique(sorted=True).tolist():
            mask = g["t"] == t
            spans, inverse = torch.stack((g["a"][mask], g["b"][mask]), 1).unique(
                dim=0, sorted=True, return_inverse=True
            )
            g["templates"].append(dict(time=t, mask=mask, spans=spans, inverse=inverse))
    theta = torch.nn.Parameter(initial.clone())
    gamma = torch.nn.Parameter(
        torch.zeros(feature["allocated_dimensions"], dtype=torch.float64)
    )
    parameters = [theta, gamma]
    optimizer = torch.optim.Adam(
        parameters,
        lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]),
        eps=cfg["epsilon"],
    )
    generator = torch.Generator().manual_seed(cfg["sample_seed"])
    coverage = torch.zeros(theta.numel() + gamma.numel(), dtype=torch.bool)
    selected = set(cfg["replay_steps"])
    states, history, rngs = {}, [], {}
    totalwork = dict(partitions=0, bond_draws=0, presence_terms=0)
    now = time.perf_counter()
    with gzip.open(destination, "wt") as stream:
        for step in range(1, cfg["steps"] + 1):
            if step in selected:
                rngs[str(step)] = generator.get_state().tolist()
            optimizer.zero_grad()
            value, data = objective(theta, gamma, groups, cfg["penalty"])
            assert torch.isfinite(value)
            value.backward()
            records = fast_prepare(theta.detach(), gamma.detach(), groups)
            sampled, work = estimate(
                records, cfg["sample_count"], generator, cfg["proposal"]
            )
            reg = theta.detach().clone() * cfg["penalty"]
            reg[0] = 0
            theta.grad = sampled + reg
            gradient = torch.cat([p.grad for p in parameters])
            assert torch.isfinite(gradient).all()
            coverage |= gradient != 0
            before = torch.cat([p.detach() for p in parameters]).tolist()
            optimizer.step()
            for k in totalwork:
                totalwork[k] += work[k]
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
            if step in selected:
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
    final, data = objective(theta, gamma, groups, cfg["penalty"])
    gradient = torch.cat(torch.autograd.grad(final, parameters))
    result_metrics = save_predictions(
        name, evaluate(theta.detach(), gamma.detach(), groups)
    )
    record = dict(
        theta=theta.detach().tolist(),
        gamma=gamma.detach().tolist(),
        initial_theta=initial.tolist(),
        initial_gamma=torch.zeros_like(gamma).tolist(),
        final_loss=float(final.detach()),
        final_mse=float(data.detach()),
        exact_final_gradient=gradient.tolist(),
        gradient_coverage=coverage.tolist(),
        changed_parameter_count=int(
            (torch.cat([theta.detach() - initial, gamma.detach()]) != 0).sum()
        ),
        history=history,
        checkpoints=states,
        steps=cfg["steps"],
        seconds=time.perf_counter() - now,
        work=totalwork,
        optimizer_states=[
            {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in optimizer.state[p].items()
            }
            for p in parameters
        ],
        final_rng=generator.get_state().tolist(),
        preupdate_rngs=rngs,
    )
    dump(FAMILY / f"results/{name}-fit.json", record)
    dump(FAMILY / "results/conditional-fit-metrics.json", result_metrics)
    sensitivity = []
    train = [g for g in groups if g["split"] == "train"]
    for shift in cfg["log10_rate_scale_shifts"]:
        perturbed = theta.detach().clone()
        perturbed[0] += shift * LN10
        with torch.no_grad():
            loss, mse = objective(perturbed, gamma.detach(), train, cfg["penalty"])
            amounts = torch.cat([log_amount(perturbed, g) for g in train])
        sensitivity.append(
            dict(
                variant=name,
                log10_rate_scale_shift=shift,
                profile_mse=float(mse),
                total_loss=float(loss),
                minimum_log10_amount=float(amounts.min()),
                median_log10_amount=float(amounts.median()),
                below_1e_20_count=int((amounts < -20).sum()),
            )
        )
    dump(FAMILY / "results/conditional-scale-sensitivity.json", sensitivity)
    runtime = dict(
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        complete=True,
        sampled_fit_performed=True,
        variant=name,
        scope="Conditional-importance hybrid; exact forward values, response gradients and conditional scores retained",
    )
    dump(FAMILY / "results/conditional-fitting-runtime.json", runtime)
    print(json.dumps(dict(metrics=result_metrics, runtime=runtime)), flush=True)


if __name__ == "__main__":
    main()
