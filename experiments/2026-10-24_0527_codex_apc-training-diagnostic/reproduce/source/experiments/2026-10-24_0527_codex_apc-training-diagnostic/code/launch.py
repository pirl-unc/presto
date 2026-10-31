"""Original-training-only fixed diagnostic and biologically gated sampled comparison."""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import time

import numpy as np
import torch
from core import (
    FAMILY,
    Dataset,
    Model,
    calibration,
    metric_module,
    objective,
    offsets,
    read,
    sha,
    write,
)


def check_source():
    for path, digest in read(FAMILY / "reproduce/execution.json")["sha256"].items():
        assert sha(path) == digest, path


def folder(fold):
    return FAMILY / "results" / ("full" if fold == -1 else f"fold-{fold}")


def jacobian(model, pack, state):
    theta = torch.tensor(state, dtype=torch.float64)
    category = torch.tensor(model.data.condition_ids[pack["groups"]])
    group_weight = theta.new_zeros(len(pack["groups"])).index_add(0, pack["owner"], pack["weight"])

    def centered(t):
        p, _ = model.forward(t, pack["groups"])
        means = p.new_zeros(model.nc).index_add(0, category, group_weight * p) / p.new_zeros(
            model.nc
        ).index_add(0, category, group_weight)
        return group_weight.sqrt() * (p - means[category])

    # Directional autodiff avoids a large reverse-mode vectorized intermediate.
    columns = []
    for i in range(model.n_parameters):
        direction = torch.zeros_like(theta)
        direction[i] = 1
        columns.append(
            torch.autograd.functional.jvp(centered, theta, direction)[1].detach().numpy()
        )
    matrix = np.stack(columns, axis=1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-8)) if singular[0] else 0
    return matrix, {
        "rank_relative_1e8": rank,
        "n_parameters": model.n_parameters,
        "singular_values": singular.tolist(),
        "column_norms": np.linalg.norm(matrix, axis=0).tolist(),
    }


def qualify():
    start = time.perf_counter()
    output = FAMILY / "results/qualification.json"
    assert not output.exists()
    data = Dataset()
    pack = data.pack("train")
    cal = calibration(data, pack)
    probes = []
    for flexible in (False, True):
        model = Model(data, flexible, cal["gamma"])
        states = [
            np.zeros(model.n_parameters),
            np.random.default_rng(data.cfg["gradient_state_seed"]).normal(size=model.n_parameters)
            * data.cfg["gradient_state_scale"],
        ]
        for index, state in enumerate(states):
            theta = torch.tensor(state, dtype=torch.float64, requires_grad=True)
            loss = objective(theta, model, pack, 0)[0]
            gradient = torch.autograd.grad(loss, theta)[0].detach().numpy()
            rng = np.random.default_rng(111 + index)
            directions = []
            for _ in range(3):
                d = rng.normal(size=len(state))
                d /= np.linalg.norm(d)
                h = 1e-6
                finite = float(
                    (
                        objective(torch.tensor(state + h * d), model, pack, 0)[0]
                        - objective(torch.tensor(state - h * d), model, pack, 0)[0]
                    )
                    / (2 * h)
                )
                analytic = float(d @ gradient)
                assert abs(finite - analytic) < 1e-7, (finite, analytic)
                directions.append({"finite_difference": finite, "analytic": analytic})
            probes.append(
                {
                    "flexible": flexible,
                    "state_index": index,
                    "state": state.tolist(),
                    "gradient": gradient.tolist(),
                    "directions": directions,
                }
            )
        matrix, info = jacobian(model, pack, states[0])
        name = "flexible" if flexible else "shared"
        np.savez_compressed(FAMILY / "results" / f"initial-{name}-jacobian.npz", jacobian=matrix)
        write(FAMILY / "results" / f"initial-{name}-jacobian.json", info)
    write(
        output,
        {
            "status": "passed",
            "seconds": time.perf_counter() - start,
            "calibration": cal,
            "probes": probes,
            "original_held_out_used": False,
        },
    )
    print(
        json.dumps(
            {
                "qualification": "passed",
                "seconds": time.perf_counter() - start,
                "gamma": cal["gamma"],
            }
        ),
        flush=True,
    )


def prediction_dump(data, variant, group_predictions, output):
    split = "train" if data.fold == -1 else "inner_val"
    rows, indices = data.rows(split)
    with gzip.open(output, "wt") as stream:
        for row, index in zip(rows, indices):
            row.update(
                variant=variant,
                fit_fold=data.fold,
                prediction_log10_ratio=float(group_predictions[index]),
            )
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def fit_controls(data, pack, cal):
    output = folder(data.fold) / "controls"
    assert not output.exists()
    output.mkdir(parents=True)
    records = []
    for variant, gamma in (
        ("no_change", 0.0),
        ("condition", 0.0),
        ("precursor_fixed", 1.0),
        ("precursor_calibrated", cal["gamma"]),
    ):
        start = time.perf_counter()
        raw = torch.tensor(gamma * data.baseline_precursor, dtype=torch.float64)
        b = (
            torch.zeros(len(data.conditions), dtype=torch.float64)
            if variant == "no_change"
            else offsets(data, pack, raw[pack["groups"]])
        )
        prediction = (raw + b[torch.tensor(data.condition_ids)]).numpy()
        error = prediction[pack["groups"]][pack["owner"].numpy()] - pack["target"].numpy()
        result = {
            "variant": variant,
            "gamma": gamma,
            "offsets": b.tolist(),
            "train_mse": float(pack["weight"].numpy() @ error**2),
            "seconds": time.perf_counter() - start,
            "n_fitted_parameters": 0
            if variant == "no_change"
            else len(b) + int(variant == "precursor_calibrated"),
        }
        records.append(result)
        np.savez_compressed(output / (variant + ".npz"), prediction=prediction, offsets=b.numpy())
        prediction_dump(data, variant, prediction, output / (variant + "-predictions.jsonl.gz"))
    write(output / "summary.json", records)


def train(data, arm, sampled=False):
    cfg = data.cfg["arms"][arm]
    variant = arm + "_hybrid" if sampled else arm
    destination = folder(data.fold) / variant
    assert not destination.exists()
    destination.mkdir(parents=True)
    pack = data.pack("train")
    cal = calibration(data, pack)
    model = Model(data, cfg["flexible"], cal["gamma"])
    theta = torch.nn.Parameter(torch.zeros(model.n_parameters, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [theta], lr=data.cfg["lr"], betas=tuple(data.cfg["betas"]), eps=data.cfg["eps"]
    )
    generator = torch.Generator().manual_seed(data.cfg["seed"])
    history, gradients, losses = [theta.detach().numpy().copy()], [], []
    start = time.perf_counter()
    status, projection = "completed", None
    cap = data.cfg["hybrid_seconds_cap"] if sampled else data.cfg["exact_seconds_cap"]
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
        total, loss, _, _, _ = objective(theta, model, pack, cfg["penalty"], uniforms)
        assert torch.isfinite(total)
        total.backward()
        assert torch.isfinite(theta.grad).all()
        gradients.append(theta.grad.detach().numpy().copy())
        losses.append([float(total.detach()), float(loss.detach())])
        optimizer.step()
        history.append(theta.detach().numpy().copy())
        elapsed = time.perf_counter() - start
        if (step + 1) % 100 == 0:
            torch.save(
                {
                    "theta": theta.detach(),
                    "optimizer": optimizer.state_dict(),
                    "generator": generator.get_state(),
                    "step": step + 1,
                },
                destination / f"step-{step + 1:03d}.pt",
            )
            print(
                json.dumps(
                    {
                        "fold": data.fold,
                        "variant": variant,
                        "step": step + 1,
                        "training_loss": losses[-1][1],
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
    total, loss, _p, b, _ = objective(theta, model, pack, cfg["penalty"])
    gradient = torch.autograd.grad(total, theta)[0]
    with torch.no_grad():
        raw, extra = model.forward(theta, np.arange(len(data.groups)))
        prediction = raw + b[torch.tensor(data.condition_ids)]
    np.savez_compressed(
        destination / "fit.npz",
        theta=theta.detach().numpy(),
        offsets=b.detach().numpy(),
        prediction=prediction.numpy(),
        trajectory=np.array(history),
        gradients=np.array(gradients),
        losses=np.array(losses),
        final_exact_gradient=gradient.detach().numpy(),
        **{k: v.numpy() for k, v in extra.items()},
    )
    result = {
        "variant": variant,
        "fold": data.fold,
        "status": status,
        "steps": len(losses),
        "gamma": cal["gamma"],
        "n_kinetic_parameters": model.n_parameters,
        "n_changed": int(np.sum(abs(history[-1]) > 1e-12)),
        "train_mse": float(loss.detach()),
        "training_objective": float(total.detach()),
        "final_exact_gradient_inf": float(gradient.abs().max()),
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "projected_seconds_after_20": projection,
        "n_score_evaluations": len(losses) * 2 * len(pack["groups"]) * data.cfg["sample_count"]
        if sampled
        else 0,
        "fit_sha256": sha(destination / "fit.npz"),
    }
    write(destination / "status.json", result)
    if status == "completed":
        prediction_dump(data, variant, prediction.numpy(), destination / "predictions.jsonl.gz")
        if data.fold == -1:
            matrix, info = jacobian(model, pack, history[-1])
            np.savez_compressed(destination / "jacobian.npz", jacobian=matrix)
            write(destination / "jacobian.json", info)
    print(json.dumps(result), flush=True)


def exact():
    assert read(FAMILY / "results/qualification.json")["status"] == "passed"
    for fold in [-1, *range(5)]:
        data = Dataset(fold)
        directory = folder(fold)
        assert not directory.exists()
        directory.mkdir()
        pack = data.pack("train")
        cal = calibration(data, pack)
        write(directory / "calibration.json", cal)
        write(
            directory / "coverage.json",
            {
                **data.feature_coverage,
                "n_fitting_groups": len(pack["groups"]),
                "n_fitting_observations": len(pack["rows"]),
                "fold": fold,
            },
        )
        np.savez_compressed(
            directory / "transforms.npz",
            **{"bond_" + k: v for k, v in data.bond_transform.items()},
            **{"condition_" + k: v for k, v in data.condition_transform.items()},
        )
        fit_controls(data, pack, cal)
        for arm in data.cfg["arms"]:
            train(data, arm)


def collect(final=False):
    data = Dataset()
    variants = [
        "no_change",
        "condition",
        "precursor_fixed",
        "precursor_calibrated",
        *data.cfg["arms"],
    ]
    sample_status = read(FAMILY / "results/sample-status.json") if final else None
    if final and sample_status["status"] == "completed":
        variants.append("flexible_unpenalized_hybrid")
    metric = metric_module().metrics
    all_metrics, strata, spread, paths = [], [], [], []
    for variant in variants:
        for phase in ("full_train", "inner_cv"):
            rows = []
            for fold in [-1] if phase == "full_train" else range(5):
                p = (
                    folder(fold) / variant / "predictions.jsonl.gz"
                    if variant not in variants[:4]
                    else folder(fold) / "controls" / (variant + "-predictions.jsonl.gz")
                )
                assert p.exists(), p
                paths.append(p)
                with gzip.open(p, "rt") as stream:
                    rows.extend(json.loads(line) for line in stream)
            assert len(rows) == 2952 and len({r["observation_id"] for r in rows}) == 2952
            assert all(r["original_split"] == "train" for r in rows)
            out = {"variant": variant, "phase": phase, **metric(rows)}
            all_metrics.append(out)
            for field in ("condition_id", "component_id"):
                for value in sorted({r[field] for r in rows}):
                    chosen = [r for r in rows if r[field] == value]
                    strata.append(
                        {
                            "variant": variant,
                            "phase": phase,
                            "stratum": field,
                            "stratum_id": value,
                            **metric(chosen),
                        }
                    )
            for condition in data.conditions:
                chosen = [r for r in rows if r["condition_id"] == condition]
                w = np.array([r["weight"] for r in chosen])
                w /= w.sum()
                p = np.array([r["prediction_log10_ratio"] for r in chosen])
                spread.append(
                    {
                        "variant": variant,
                        "phase": phase,
                        "condition": condition,
                        "weighted_prediction_sd": float(np.sqrt(w @ (p - w @ p) ** 2)),
                    }
                )
    if not final:
        reference = next(
            m["exact_rmse"]
            for m in all_metrics
            if m["variant"] == "precursor_calibrated" and m["phase"] == "inner_cv"
        )
        candidate = next(
            m["exact_rmse"]
            for m in all_metrics
            if m["variant"] == "flexible_unpenalized" and m["phase"] == "inner_cv"
        )
        gain = 1 - candidate / reference
        variation = [
            r["weighted_prediction_sd"]
            for r in spread
            if r["variant"] == "flexible_unpenalized" and r["phase"] == "inner_cv"
        ]
        passed = gain >= data.cfg["sampled_biological_rmse_gain"] and all(
            v > data.cfg["sampled_variation_gate"] for v in variation
        )
        write(
            FAMILY / "results/biological-gate.json",
            {
                "passed": passed,
                "rmse_gain": gain,
                "required_gain": data.cfg["sampled_biological_rmse_gain"],
                "condition_prediction_sd": variation,
                "scope": "original-training-only inner CV; not external biological validation",
            },
        )
    name = "final" if final else "exact"
    write(
        FAMILY / "results" / (name + "-summary.json"),
        {
            "metrics": all_metrics,
            "strata": strata,
            "spread": spread,
            "prediction_files_sha256": {str(p.relative_to(FAMILY)): sha(p) for p in paths},
            "original_validation_test_used": False,
        },
    )
    print(json.dumps({"phase": name, "metrics": all_metrics}), flush=True)


def sampled():
    output = FAMILY / "results/sample-status.json"
    assert not output.exists()
    if not read(FAMILY / "results/biological-gate.json")["passed"]:
        write(
            output,
            {
                "status": "not_admitted",
                "reason": "predesignated biological inner-CV gate failed; no sampler extension",
            },
        )
        return
    data = Dataset()
    pack = data.pack("train")
    model = Model(data, True, calibration(data, pack)["gamma"])
    probes = []
    passed = True
    for index, state in enumerate(
        [
            np.zeros(model.n_parameters),
            np.random.default_rng(data.cfg["gradient_state_seed"]).normal(size=model.n_parameters)
            * data.cfg["gradient_state_scale"],
        ]
    ):
        theta = torch.tensor(state, requires_grad=True)
        total, _, p, _, _ = objective(theta, model, pack, 0)
        exact = torch.autograd.grad(total, theta)[0].detach().numpy()
        estimates = []
        for repeat in range(data.cfg["gradient_repeats"]):
            generator = torch.Generator().manual_seed(data.cfg["seed"] + index * 100 + repeat)
            uniforms = torch.rand(
                (len(pack["groups"]), data.cfg["sample_count"], 5),
                dtype=torch.float64,
                generator=generator,
            )
            theta = torch.tensor(state, requires_grad=True)
            total, _, ps, _, _ = objective(theta, model, pack, 0, uniforms)
            assert torch.equal(p.detach(), ps.detach())
            estimates.append(torch.autograd.grad(total, theta)[0].detach().numpy())
        error = np.array(estimates) - exact
        norm = np.linalg.norm(exact)
        mean = float(np.linalg.norm(error.mean(0)) / norm)
        rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))) / norm)
        gate = mean <= data.cfg["gradient_mean_gate"] and rms <= data.cfg["gradient_rms_gate"]
        passed &= gate
        probes.append(
            {
                "state": state.tolist(),
                "exact_gradient": exact.tolist(),
                "estimates": np.array(estimates).tolist(),
                "mean_relative_error": mean,
                "rms_relative_error": rms,
                "passed": gate,
            }
        )
    write(FAMILY / "results/sampled-qualification.json", {"passed": bool(passed), "probes": probes})
    if not passed:
        write(output, {"status": "not_admitted", "reason": "fixed numerical gradient gate failed"})
        return
    statuses = []
    for fold in [-1, *range(5)]:
        train(Dataset(fold), "flexible_unpenalized", sampled=True)
        statuses.append(read(folder(fold) / "flexible_unpenalized_hybrid/status.json"))
    write(
        output,
        {
            "status": "completed"
            if all(r["status"] == "completed" for r in statuses)
            else "partial",
            "attempts": statuses,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["qualify", "exact", "collect_exact", "sampled", "collect_final"]
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check_source()
    if args.action == "qualify":
        qualify()
    elif args.action == "exact":
        exact()
    elif args.action == "collect_exact":
        collect()
    elif args.action == "sampled":
        sampled()
    else:
        collect(final=True)


if __name__ == "__main__":
    main()
