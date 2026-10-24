"""Separate numerical admission, fitting, and sealed final evaluation commands."""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from core import (
    FAMILY,
    Dataset,
    PhysicalModel,
    baseline_modules,
    config,
    objective,
    read,
    sha,
    write,
)
from scipy import optimize


def check_source():
    receipt = read(FAMILY / "reproduce/execution.json")
    for path, digest in receipt["source_sha256"].items():
        assert sha(path) == digest, path


def qualify(data):
    start = time.perf_counter()
    destination = FAMILY / "results/qualification.json"
    assert not destination.exists()
    write(FAMILY / "results/data-coverage.json", data.summary())
    np.savez_compressed(
        FAMILY / "results/feature-transforms.npz",
        **{"bond_" + k: v for k, v in data.bond_arrays.items()},
        **{"condition_" + k: v for k, v in data.condition_arrays.items()},
    )
    pack = data.packing("train")
    probes = []
    admission = {}
    for representation in ("uniform", "sequence"):
        model = PhysicalModel(data, representation)
        states = [
            np.zeros(model.n_parameters),
            np.random.default_rng(data.cfg["gradient_state_seed"]).normal(size=model.n_parameters)
            * data.cfg["gradient_state_scale"],
        ]
        states[1][model.b_slice.start] = 0.0
        passed = True
        for probe, state in enumerate(states):
            theta = torch.tensor(state, dtype=torch.float64, requires_grad=True)
            loss, _, pred, _ = objective(theta, model, pack, penalized=False)
            exact = torch.autograd.grad(loss, theta)[0].detach().numpy()
            if probe == 0:
                assert torch.equal(pred, torch.zeros_like(pred))
                assert np.array_equal(exact[: 1 + model.rank], np.zeros(1 + model.rank))
            rng = np.random.default_rng(111 + probe)
            directions = []
            for _ in range(3):
                direction = rng.normal(size=len(state))
                direction /= np.linalg.norm(direction)
                h = 1e-6
                plus = objective(torch.tensor(state + h * direction), model, pack, penalized=False)[
                    0
                ]
                minus = objective(
                    torch.tensor(state - h * direction), model, pack, penalized=False
                )[0]
                finite = float((plus - minus) / (2 * h))
                analytic = float(exact @ direction)
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
                sampled_loss, _, sampled_pred, _ = objective(
                    theta, model, pack, uniforms=uniforms, penalized=False
                )
                assert torch.equal(sampled_pred.detach(), pred.detach())
                gradient = torch.autograd.grad(sampled_loss, theta)[0].detach().numpy()
                if probe == 0:
                    assert np.max(abs(gradient[: 1 + model.rank])) < 1e-12
                estimates.append(gradient)
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
                "rho_and_sequence": slice(0, 1 + model.rank),
                "treatment_rate": model.a_slice,
                "condition_offset": model.b_slice,
            }.items():
                e = error[:, indices]
                reference_norm = float(np.linalg.norm(exact[indices]))
                blocks[name] = {
                    "reference_norm": reference_norm,
                    "mean_error": float(np.linalg.norm(e.mean(0))),
                    "rms_error": float(np.sqrt(np.mean(np.sum(e**2, axis=1)))),
                }
            row = {
                "representation": representation,
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
        admission[representation + "_hybrid"] = passed
        admission[representation + "_exact"] = True
    result = {
        "status": "completed",
        "admission": admission,
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "n_training_response_groups": len(pack["groups"]),
        "sample_count": data.cfg["sample_count"],
        "test_evaluated": False,
    }
    assert result["peak_rss_bytes"] < data.cfg["rss_bytes_cap"]
    write(destination, result)
    print(json.dumps(result), flush=True)


def train(data, arm):
    qualification = read(FAMILY / "results/qualification.json")
    destination = FAMILY / "results/arms" / arm
    assert not destination.exists()
    destination.mkdir(parents=True)
    if not qualification["admission"][arm]:
        write(
            destination / "status.json",
            {"arm": arm, "status": "not_admitted", "reason": "registered gradient gate"},
        )
        return
    representation, method = arm.split("_")
    model = PhysicalModel(data, representation)
    pack = data.packing("train")
    theta = torch.nn.Parameter(torch.zeros(model.n_parameters, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [theta], lr=data.cfg["lr"], betas=tuple(data.cfg["betas"]), eps=data.cfg["eps"]
    )
    generator = torch.Generator().manual_seed(data.cfg["seed"])
    cap = data.cfg["hybrid_seconds_cap"] if method == "hybrid" else data.cfg["exact_seconds_cap"]
    trajectory, gradients, losses, trace_probes = [theta.detach().numpy().copy()], [], [], []
    start = time.perf_counter()
    status = "completed"
    runtime_projection = None
    for step in range(data.cfg["steps"]):
        optimizer.zero_grad()
        uniforms = None
        rng_before = generator.get_state()
        if method == "hybrid":
            uniforms = torch.rand(
                (len(pack["groups"]), data.cfg["sample_count"], 5),
                dtype=torch.float64,
                generator=generator,
            )
        total, unpenalized, _, extra = objective(
            theta, model, pack, uniforms=uniforms, trace=step in {0, 99, 199}
        )
        assert torch.isfinite(total)
        total.backward()
        assert torch.isfinite(theta.grad).all()
        gradients.append(theta.grad.detach().numpy().copy())
        losses.append([float(total.detach()), float(unpenalized.detach())])
        if step in {0, 99, 199}:
            trace_probes.append(
                {"step": step, "trace": extra["trace"], "rng_state_before": rng_before.tolist()}
            )
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
                destination / f"step-{step + 1:03d}.pt",
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
        if method == "hybrid" and step == 19:
            runtime_projection = elapsed * data.cfg["steps"] / 20
            if runtime_projection > cap:
                status = "stopped_projected_runtime"
                break
        if (
            elapsed > cap
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > data.cfg["rss_bytes_cap"]
        ):
            status = "stopped_resource_cap"
            break
    final_total, final_data, _, _ = objective(theta, model, pack)
    final_gradient = torch.autograd.grad(final_total, theta)[0]
    np.savez_compressed(
        destination / "fit.npz",
        theta=theta.detach().numpy(),
        trajectory=np.array(trajectory),
        gradients=np.array(gradients),
        losses=np.array(losses),
        final_exact_gradient=final_gradient.detach().numpy(),
    )
    write(destination / "trace-probes.json", trace_probes)
    result = {
        "arm": arm,
        "status": status,
        "steps": len(losses),
        "n_parameters": model.n_parameters,
        "n_parameters_changed": int(np.sum(abs(trajectory[-1]) > 1e-12)),
        "training_data_loss": float(final_data.detach()),
        "training_objective": float(final_total.detach()),
        "final_exact_gradient_inf": float(final_gradient.abs().max()),
        "seconds": time.perf_counter() - start,
        "runtime_projection_after_20_steps": runtime_projection,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "n_conditional_age_samples": len(losses)
        * 2
        * len(pack["groups"])
        * data.cfg["sample_count"]
        if method == "hybrid"
        else 0,
        "n_score_evaluations": len(losses) * 2 * len(pack["groups"]) * data.cfg["sample_count"]
        if method == "hybrid"
        else 0,
        "fit_sha256": sha(destination / "fit.npz"),
        "test_evaluated": False,
    }
    write(destination / "status.json", result)
    print(json.dumps(result), flush=True)


def baselines(data):
    destination = FAMILY / "results/baselines"
    assert not destination.exists()
    destination.mkdir()
    training, _ = baseline_modules()
    old_config = read(Path(data.cfg["baseline_family"]) / "reproduce/launch.json")
    rows, indices = data.rows("train")
    y, censor, weight = (
        np.array([r[k] for r in rows]) for k in ("target_log10_ratio", "censor_code", "weight")
    )
    fit_records = []
    start = time.perf_counter()
    for variant in old_config["variants"]:
        arm_start = time.perf_counter()
        if variant in {"no_change", "exact_train_mean", "censored_constant"}:
            exact = censor == 0
            value = 0.0
            if variant == "exact_train_mean":
                value = float(weight[exact] @ y[exact] / weight[exact].sum())
            if variant == "censored_constant":
                value = optimize.brentq(
                    lambda p: float(weight @ training.residuals(p, y, censor)),
                    float(y.min() - 1),
                    float(y.max() + 1),
                    xtol=1e-14,
                )
            result = {
                "variant": variant,
                "constant": value,
                "n_parameters": int(variant != "no_change"),
                "converged": True,
            }
        else:
            x, features, arrays = training.design(
                data.groups,
                data.split == "train",
                data.group_weights,
                variant,
                old_config["relative_rank_threshold"],
            )
            xt = x[indices]
            fit = optimize.minimize(
                training.objective,
                np.zeros(x.shape[1]),
                args=(xt, y, censor, weight, old_config["penalty"]),
                method="L-BFGS-B",
                jac=True,
                options={k: old_config[k] for k in ("maxiter", "gtol", "ftol")},
            )
            np.savez_compressed(destination / (variant + ".npz"), theta=fit.x, **arrays)
            result = {
                "variant": variant,
                "features": features,
                "n_parameters": len(fit.x),
                "n_parameters_changed": int(np.sum(abs(fit.x) > 1e-12)),
                "converged": bool(fit.success),
                "convergence_message": str(fit.message),
                "iterations": int(fit.nit),
                "final_gradient_inf": float(abs(fit.jac).max()),
                "training_objective": float(fit.fun),
            }
        result["seconds"] = time.perf_counter() - arm_start
        assert result["seconds"] < old_config["seconds_per_fit_cap"]
        fit_records.append(result)
        print(
            json.dumps({k: result[k] for k in ("variant", "n_parameters", "converged", "seconds")}),
            flush=True,
        )
    receipt = {
        "status": "completed",
        "fits": fit_records,
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "test_evaluated": False,
        "original_baseline_contract": old_config,
    }
    assert (
        receipt["seconds"] < old_config["seconds_family_cap"]
        and receipt["peak_rss_bytes"] < old_config["rss_bytes_cap"]
    )
    write(destination / "summary.json", receipt)


def baseline_prediction(data, fit):
    _, independent = baseline_modules()
    if "constant" in fit:
        return np.full(len(data.groups), fit["constant"])
    arrays = np.load(FAMILY / "results/baselines" / (fit["variant"] + ".npz"))
    names = fit["features"]["all_feature_names"]
    known, predictions = {}, []
    for group in data.groups:
        values = tuple(independent.feature(group, name) for name in names)
        if values not in known:
            raw = np.array(values)
            z = (raw[arrays["keep"]] - arrays["mean"][arrays["keep"]]) / arrays["scale"]
            coordinates = z @ arrays["projection"]
            known[values] = float(arrays["theta"][0] + coordinates @ arrays["theta"][1:])
        predictions.append(known[values])
    return np.array(predictions)


def evaluate(data):
    output = FAMILY / "results/evaluation"
    assert not output.exists(), "Test is opened once for this fixed comparison"
    attempts = [read(FAMILY / "results/arms" / arm / "status.json") for arm in data.cfg["arms"]]
    baseline = read(FAMILY / "results/baselines/summary.json")
    assert baseline["status"] == "completed" and all(
        r["status"]
        in {"completed", "not_admitted", "stopped_projected_runtime", "stopped_resource_cap"}
        for r in attempts
    )
    # Freeze every input before producing any validation/test comparison.
    checkpoint_files = sorted((FAMILY / "results/arms").glob("*/fit.npz")) + sorted(
        (FAMILY / "results/baselines").glob("*.npz")
    )
    write(
        FAMILY / "results/final-evaluation-inputs.json",
        {
            "checkpoints": {str(p.relative_to(FAMILY)): sha(p) for p in checkpoint_files},
            "attempts": attempts,
            "source_receipt_sha256": sha(FAMILY / "reproduce/execution.json"),
            "test_policy": data.cfg["test_policy"],
        },
    )
    output.mkdir()
    _, metrics_module = baseline_modules()
    predictions, physical = {}, {}
    for fit in baseline["fits"]:
        predictions[fit["variant"]] = baseline_prediction(data, fit)
    for record in attempts:
        if record["status"] != "completed":
            continue
        arm = record["arm"]
        path = FAMILY / "results/arms" / arm / "fit.npz"
        assert sha(path) == record["fit_sha256"]
        model = PhysicalModel(data, arm.split("_")[0])
        theta = torch.tensor(np.load(path)["theta"], dtype=torch.float64)
        with torch.no_grad():
            p, extra = model.forward(theta, np.arange(len(data.groups)))
        predictions[arm] = p.numpy()
        physical[arm] = (extra["log_control_amount"].numpy(), extra["log_treated_amount"].numpy())
    metrics, strata = [], []
    for variant, prediction in predictions.items():
        with gzip.open(output / (variant + "-predictions.jsonl.gz"), "wt") as stream:
            for split in ("train", "val", "test"):
                rows, indices = data.rows(split)
                for row, index in zip(rows, indices):
                    row["variant"] = variant
                    row["prediction_log10_ratio"] = float(prediction[index])
                    if variant in physical:
                        row["log_control_amount_per_unit_input"] = float(
                            physical[variant][0][index]
                        )
                        row["log_treated_amount_per_unit_input"] = float(
                            physical[variant][1][index]
                        )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                entry = {"variant": variant, "split": split, **metrics_module.metrics(rows)}
                metrics.append(entry)
                for field in (
                    "condition_id",
                    "component_id",
                    "modified_form",
                    "multiple_parent_candidates",
                ):
                    groups = {}
                    for row in rows:
                        value = (
                            bool(row["modifications"])
                            if field == "modified_form"
                            else row["n_candidate_parent_sequences"] > 1
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
                                **metrics_module.metrics(members),
                            }
                        )
                if split != "train":
                    print(json.dumps(entry), flush=True)
    summary = {
        "metrics": metrics,
        "attempts": attempts,
        "baseline_fits": baseline["fits"],
        "test_evaluated": True,
        "scope": "native extract treated/control response; not APC concentration",
        "dataset": data.summary(),
    }
    write(output / "summary.json", summary)
    write(output / "stratified-metrics.json", strata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["qualify", "baselines", "train", "evaluate"])
    parser.add_argument("arm", nargs="?")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check_source()
    data = Dataset()
    if args.action == "qualify":
        qualify(data)
    elif args.action == "baselines":
        assert read(FAMILY / "results/qualification.json")["status"] == "completed"
        baselines(data)
    elif args.action == "train":
        assert args.arm in config()["arms"]
        train(data, args.arm)
    else:
        evaluate(data)


if __name__ == "__main__":
    main()
