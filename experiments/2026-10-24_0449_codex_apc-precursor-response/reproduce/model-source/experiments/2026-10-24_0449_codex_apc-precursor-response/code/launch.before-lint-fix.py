"""Fixed local APC response comparison with a separately invoked final evaluation."""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import time

import numpy as np
import torch
from core import FAMILY, Dataset, Model, metric_module, objective, read, sha, write
from scipy import optimize


def qualify(data):
    start = time.perf_counter()
    output = FAMILY / "results/qualification.json"
    assert not output.exists()
    write(FAMILY / "results/feature-coverage.json", data.feature_coverage)
    write(FAMILY / "results/prior-rate-coefficients.json", data.prior_rate_coefficients)
    np.savez_compressed(
        FAMILY / "results/feature-transforms.npz",
        **{"bond_" + k: v for k, v in data.bond_transform.items()},
        **{"condition_" + k: v for k, v in data.condition_transform.items()},
    )
    pack = data.pack("train")
    probes, passed = [], True
    for kind in ("uniform", "sequence"):
        model = Model(data, kind)
        states = [
            np.zeros(model.n_parameters),
            np.random.default_rng(data.cfg["gradient_state_seed"]).normal(size=model.n_parameters)
            * data.cfg["gradient_state_scale"],
        ]
        for probe, state in enumerate(states):
            theta = torch.tensor(state, dtype=torch.float64, requires_grad=True)
            loss, _, pred, _ = objective(theta, model, pack, penalty=False)
            exact = torch.autograd.grad(loss, theta)[0].detach().numpy()
            directions = []
            rng = np.random.default_rng(111 + probe)
            for _ in range(3):
                direction = rng.normal(size=len(state))
                direction /= np.linalg.norm(direction)
                h = 1e-6
                plus = objective(torch.tensor(state + h * direction), model, pack, penalty=False)[0]
                minus = objective(torch.tensor(state - h * direction), model, pack, penalty=False)[
                    0
                ]
                finite, analytic = float((plus - minus) / (2 * h)), float(exact @ direction)
                assert abs(finite - analytic) < 1e-7
                directions.append({"finite_difference": finite, "analytic": analytic})
            estimates = []
            for rep in range(data.cfg["gradient_repeats"]):
                generator = torch.Generator().manual_seed(data.cfg["seed"] + probe * 100 + rep)
                uniforms = torch.rand(
                    (len(pack["groups"]), data.cfg["sample_count"], 5),
                    dtype=torch.float64,
                    generator=generator,
                )
                theta = torch.tensor(state, dtype=torch.float64, requires_grad=True)
                sampled, _, sampled_pred, _ = objective(theta, model, pack, uniforms, penalty=False)
                assert torch.equal(sampled_pred.detach(), pred.detach())
                estimates.append(torch.autograd.grad(sampled, theta)[0].detach().numpy())
            estimates = np.array(estimates)
            error = estimates - exact
            norm = float(np.linalg.norm(exact))
            mean_error = float(np.linalg.norm(error.mean(0)))
            rms_error = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
            gate = (
                norm > 0
                and mean_error / norm <= data.cfg["gradient_mean_gate"]
                and rms_error / norm <= data.cfg["gradient_rms_gate"]
            )
            blocks = {}
            for name, indices in {
                "rho_and_sequence": slice(0, model.nr),
                "treatment_rate": model.a_slice,
                "condition_offset": model.b_slice,
            }.items():
                e = error[:, indices]
                blocks[name] = {
                    "reference_norm": float(np.linalg.norm(exact[indices])),
                    "mean_error": float(np.linalg.norm(e.mean(0))),
                    "rms_error": float(np.sqrt(np.mean(np.sum(e**2, axis=1)))),
                }
            row = {
                "representation": kind,
                "probe": probe,
                "state": state.tolist(),
                "exact_gradient": exact.tolist(),
                "gradient_estimates": estimates.tolist(),
                "mean_relative_error": mean_error / norm,
                "rms_relative_error": rms_error / norm,
                "blocks": blocks,
                "gate_pass": gate,
                "finite_difference_checks": directions,
            }
            probes.append(row)
            passed &= gate
            write(FAMILY / "results/gradient-probes.json", probes)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "representation",
                            "probe",
                            "mean_relative_error",
                            "rms_relative_error",
                            "gate_pass",
                        )
                    }
                ),
                flush=True,
            )
        # Identical treatment/control input gives exact shared-rate cancellation.
        no_input = Model(data, kind, precursor=False)
        theta = torch.zeros(no_input.n_parameters, dtype=torch.float64, requires_grad=True)
        loss, _, prediction, _ = objective(theta, no_input, pack, penalty=False)
        gradient = torch.autograd.grad(loss, theta)[0]
        assert torch.equal(prediction, torch.zeros_like(prediction))
        assert gradient[: no_input.nr].abs().max() < 1e-12
    result = {
        "status": "completed",
        "sampling_admitted": bool(passed),
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "test_evaluated": False,
    }
    assert result["peak_rss_bytes"] < data.cfg["rss_bytes_cap"]
    write(output, result)
    print(json.dumps(result), flush=True)


def baselines(data):
    output = FAMILY / "results/baselines"
    assert not output.exists()
    output.mkdir()
    rows, indices = data.rows("train")
    y, weight = (np.array([r[k] for r in rows]) for k in ("target_log10_ratio", "weight"))
    x = data.cx[indices]
    mask = np.ones(x.shape[1])
    mask[0] = 0
    records = []
    for variant in data.cfg["baselines"]:
        start = time.perf_counter()
        precursor = "precursor" in variant
        base = data.baseline_precursor[indices] if precursor else np.zeros(len(rows))
        theta = np.zeros(x.shape[1])
        result = {"variant": variant, "precursor": precursor, "n_parameters": 0, "converged": True}
        if variant.startswith("condition"):

            def objective_control(coef):
                residual = base + x @ coef - y
                value = weight @ residual**2 + data.cfg["penalty"] * (mask * coef**2).sum()
                grad = 2 * x.T @ (weight * residual) + 2 * data.cfg["penalty"] * mask * coef
                return value, grad

            fit = optimize.minimize(
                objective_control,
                theta,
                method="L-BFGS-B",
                jac=True,
                options={k: data.cfg[k] for k in ("maxiter", "gtol", "ftol")},
            )
            theta = fit.x
            result.update(
                n_parameters=len(theta),
                n_parameters_changed=int(np.sum(abs(theta) > 1e-12)),
                converged=bool(fit.success),
                convergence_message=str(fit.message),
                iterations=int(fit.nit),
                final_gradient_inf=float(abs(fit.jac).max()),
            )
        np.savez_compressed(output / (variant + ".npz"), theta=theta)
        result.update(
            training_data_loss=float(weight @ (base + x @ theta - y) ** 2),
            seconds=time.perf_counter() - start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        assert result["seconds"] < data.cfg["exact_seconds_cap"]
        assert result["peak_rss_bytes"] < data.cfg["rss_bytes_cap"]
        records.append(result)
        print(json.dumps(result), flush=True)
    write(
        output / "summary.json", {"status": "completed", "fits": records, "test_evaluated": False}
    )


def train(data, arm):
    settings = data.cfg["arms"][arm]
    sampled = settings["method"] == "hybrid"
    output = FAMILY / "results/arms" / arm
    assert not output.exists()
    output.mkdir(parents=True)
    if sampled and not read(FAMILY / "results/qualification.json")["sampling_admitted"]:
        write(output / "status.json", {"arm": arm, "status": "not_admitted"})
        return
    model = Model(data, settings["kind"], settings["precursor"])
    pack = data.pack("train")
    theta = torch.nn.Parameter(torch.zeros(model.n_parameters, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [theta], lr=data.cfg["lr"], betas=tuple(data.cfg["betas"]), eps=data.cfg["eps"]
    )
    generator = torch.Generator().manual_seed(data.cfg["seed"])
    cap = data.cfg["hybrid_seconds_cap"] if sampled else data.cfg["exact_seconds_cap"]
    trajectory, gradients, losses = [theta.detach().numpy().copy()], [], []
    start = time.perf_counter()
    status, projection = "completed", None
    for step in range(data.cfg["steps"]):
        optimizer.zero_grad()
        uniforms = (
            torch.rand(
                (len(pack["groups"]), data.cfg["sample_count"], 5),
                dtype=torch.float64,
                generator=generator,
            )
            if sampled
            else None
        )
        total, loss, _, _ = objective(theta, model, pack, uniforms)
        assert torch.isfinite(total)
        total.backward()
        assert torch.isfinite(theta.grad).all()
        gradients.append(theta.grad.detach().numpy().copy())
        losses.append([float(total.detach()), float(loss.detach())])
        optimizer.step()
        trajectory.append(theta.detach().numpy().copy())
        elapsed = time.perf_counter() - start
        if (step + 1) % 20 == 0:
            torch.save(
                {
                    "theta": theta.detach(),
                    "optimizer": optimizer.state_dict(),
                    "generator": generator.get_state(),
                    "steps": step + 1,
                },
                output / f"step-{step + 1:03d}.pt",
            )
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step + 1,
                        "loss_before_update": losses[-1],
                        "seconds": elapsed,
                    }
                ),
                flush=True,
            )
        if sampled and step == 19:
            projection = elapsed * data.cfg["steps"] / 20
            if projection > cap:
                status = "stopped_projected_runtime"
                break
        if (
            elapsed > cap
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > data.cfg["rss_bytes_cap"]
        ):
            status = "stopped_resource_cap"
            break
    final_total, final_loss, _, _ = objective(theta, model, pack)
    final_gradient = torch.autograd.grad(final_total, theta)[0]
    np.savez_compressed(
        output / "fit.npz",
        theta=theta.detach().numpy(),
        trajectory=np.array(trajectory),
        gradients=np.array(gradients),
        losses=np.array(losses),
        final_exact_gradient=final_gradient.detach().numpy(),
    )
    result = {
        "arm": arm,
        "status": status,
        "steps": len(losses),
        "settings": settings,
        "n_parameters": model.n_parameters,
        "n_parameters_changed": int(np.sum(abs(trajectory[-1]) > 1e-12)),
        "training_data_loss": float(final_loss.detach()),
        "training_objective": float(final_total.detach()),
        "final_exact_gradient_inf": float(final_gradient.abs().max()),
        "seconds": time.perf_counter() - start,
        "runtime_projection_after_20_steps": projection,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "n_score_evaluations": len(losses) * 2 * len(pack["groups"]) * data.cfg["sample_count"]
        if sampled
        else 0,
        "fit_sha256": sha(output / "fit.npz"),
        "test_evaluated": False,
    }
    write(output / "status.json", result)
    print(json.dumps(result), flush=True)


def evaluate(data):
    output = FAMILY / "results/evaluation"
    assert not output.exists(), "Held-out evaluation is opened once"
    attempts = [read(FAMILY / "results/arms" / arm / "status.json") for arm in data.cfg["arms"]]
    baselines = read(FAMILY / "results/baselines/summary.json")
    assert baselines["status"] == "completed" and all(
        r["status"]
        in {"completed", "not_admitted", "stopped_projected_runtime", "stopped_resource_cap"}
        for r in attempts
    )
    paths = list((FAMILY / "results/arms").glob("*/fit.npz")) + list(
        (FAMILY / "results/baselines").glob("*.npz")
    )
    paths += [
        FAMILY / "results/feature-transforms.npz",
        FAMILY / "results/baselines/summary.json",
        FAMILY / "results/prior-rate-coefficients.json",
    ]
    write(
        FAMILY / "reproduce/pre-evaluation-receipt.json",
        {
            "checkpoints_and_constants": {
                str(p.relative_to(FAMILY)): sha(p) for p in sorted(paths)
            },
            "attempts": attempts,
            "model_execution_sha256": sha(FAMILY / "reproduce/model-execution.json"),
            "test_policy": data.cfg["test_policy"],
        },
    )
    output.mkdir()
    predictions, physical = {}, {}
    for record in baselines["fits"]:
        name = record["variant"]
        theta = np.load(FAMILY / "results/baselines" / (name + ".npz"))["theta"]
        predictions[name] = data.cx @ theta + (
            data.baseline_precursor if record["precursor"] else 0
        )
    for record in attempts:
        if record["status"] != "completed":
            continue
        arm, settings = record["arm"], record["settings"]
        path = FAMILY / "results/arms" / arm / "fit.npz"
        assert sha(path) == record["fit_sha256"]
        model = Model(data, settings["kind"], settings["precursor"])
        with torch.no_grad():
            p, extra = model.forward(
                torch.tensor(np.load(path)["theta"]), np.arange(len(data.groups))
            )
        predictions[arm] = p.numpy()
        physical[arm] = {k: v.numpy() for k, v in extra.items()}
    metric = metric_module().metrics
    metrics, strata = [], []
    for variant, pred in predictions.items():
        with gzip.open(output / (variant + "-predictions.jsonl.gz"), "wt") as stream:
            for split in ("train", "val", "test"):
                rows, indices = data.rows(split)
                for row, i in zip(rows, indices):
                    row.update(variant=variant, prediction_log10_ratio=float(pred[i]))
                    if variant in physical:
                        row.update(
                            {
                                k + "_per_unit_input": float(v[i])
                                for k, v in physical[variant].items()
                            }
                        )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                entry = {"variant": variant, "split": split, **metric(rows)}
                metrics.append(entry)
                for field in ("condition_id", "component_id", "multiple_parent_candidates"):
                    groups = {}
                    for row in rows:
                        value = (
                            row["n_candidate_parent_sequences"] > 1
                            if field == "multiple_parent_candidates"
                            else row[field]
                        )
                        groups.setdefault(str(value), []).append(row)
                    for value, members in sorted(groups.items()):
                        strata.append(
                            {
                                "variant": variant,
                                "split": split,
                                "stratum": field,
                                "stratum_id": value,
                                **metric(members),
                            }
                        )
                if split != "train":
                    print(json.dumps(entry), flush=True)
    write(
        output / "summary.json",
        {
            "metrics": metrics,
            "attempts": attempts,
            "baseline_fits": baselines["fits"],
            "test_evaluated": True,
            "scope": "macrophage MHC-bound peptide response; not free/APC concentrations",
        },
    )
    write(output / "stratified-metrics.json", strata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["qualify", "baselines", "train", "evaluate"])
    parser.add_argument("arm", nargs="?")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = read(FAMILY / "reproduce/model-execution.json")
    for path, digest in receipt["source_sha256"].items():
        assert sha(path) == digest, path
    data = Dataset()
    if args.action == "qualify":
        qualify(data)
    else:
        assert read(FAMILY / "results/qualification.json")["status"] == "completed"
        if args.action == "baselines":
            baselines(data)
        elif args.action == "train":
            assert args.arm in data.cfg["arms"]
            train(data, args.arm)
        else:
            evaluate(data)


if __name__ == "__main__":
    main()
