"""Nested training-capacity diagnostics; no transferable free-rate model claim."""

import gzip
import hashlib
import json
import math
import resource
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import minimize

from source import FAMILY, read_groups
from presto.models.independent_cut import log_fragment_expectation


def numpy_forward(log_rates, g):
    rates = np.exp(log_rates)
    n = len(rates)
    a, b, t, ref = g["a"], g["b"], g["t"], g["ref"]
    prefix = np.r_[0.0, rates.cumsum()]
    inside = prefix[b - 1] - prefix[a]
    p = -(t - ref) * inside
    parts = []
    for index, valid in ((a - 1, a > 0), (b - 1, b <= n)):
        safe = np.clip(index, 0, n - 1)
        h = rates[safe] * t
        hr = rates[safe] * ref
        p += np.where(valid, np.log(-np.expm1(-h)) - np.log(-np.expm1(-hr)), 0.0)
        d = np.where(
            valid,
            h * np.exp(-h) / (-np.expm1(-h)) - hr * np.exp(-hr) / (-np.expm1(-hr)),
            0.0,
        )
        parts.append((safe, d))
    p /= math.log(10)
    coefficient = 2 * g["w"] * (p - g["y"]) / math.log(10)
    bond = sum(
        (
            np.bincount(index, weights=coefficient * d, minlength=n)
            for index, d in parts
        ),
        start=np.zeros(n),
    )
    diff = np.bincount(
        a, weights=-coefficient * (t - ref), minlength=n + 1
    ) - np.bincount(b - 1, weights=-coefficient * (t - ref), minlength=n + 1)
    bond += diff.cumsum()[:-1] * rates
    return p, bond


def allocate(groups, reference):
    initial = []
    audits = []
    for g in groups:
        n = len(g["sequence"]) - 1
        spans = sorted(set(zip(g["a"], g["b"])))
        role = np.zeros((n, len(spans)), dtype=np.uint8)
        for column, (a, b) in enumerate(spans):
            role[a : b - 1, column] = 1
            if a:
                role[a - 1, column] = 2
            if b <= n:
                role[b - 1, column] = 3
        signatures = {}
        indices = np.full(n, -1, dtype=np.int64)
        base = g["x"] @ reference
        members = []
        for bond, row in enumerate(role):
            if not row.any():
                continue
            signature = row.tobytes()
            if signature not in signatures:
                signatures[signature] = len(members)
                members.append([])
            members[signatures[signature]].append(bond)
        for group in members:
            # Multiple equivalent bonds are internal-only, so their summed hazard is all that is observed.
            if len(group) > 1:
                assert not (role[group[0]] >= 2).any()
            value = math.log(np.exp(base[group]).mean())
            position = len(initial)
            initial.append(value)
            indices[group] = position
        g["base_log_rates"] = base
        g["free_index"] = indices
        audits.append(
            dict(
                condition=g["condition"],
                bonds=n,
                parameters=len(members),
                unsupported_bonds=np.where(indices < 0)[0].tolist(),
                members=members,
                equivalence="Identical internal/left/right incidence over admitted spans; grouped internal hazards preserve every observed expectation. Other nonlinear equivalences are not ruled out.",
            )
        )
    return np.array(initial), audits


def tensors(groups):
    for g in groups:
        g["tx"] = torch.tensor(g["x"], dtype=torch.float64)
        for source, target in (
            ("a", "ta"),
            ("b", "tb"),
            ("t", "tt"),
            ("ref", "tr"),
            ("w", "tw"),
            ("y", "ty"),
            ("base_log_rates", "tbase"),
            ("free_index", "tindex"),
        ):
            g[target] = torch.from_numpy(g[source].copy())


def log_rates(theta, g, variant):
    if variant == "unregularized_sequence":
        return g["tx"] @ theta
    index = g["tindex"]
    return torch.where(index >= 0, theta[index.clamp(min=0)], g["tbase"])


def loss(theta, groups, variant):
    result = theta.new_zeros(())
    for g in groups:
        logs = log_rates(theta, g, variant)
        p = (
            log_fragment_expectation(logs, g["ta"], g["tb"], g["tt"], validate=False)
            - log_fragment_expectation(logs, g["ta"], g["tb"], g["tr"], validate=False)
        ) / math.log(10)
        result += (g["tw"] * (p - g["ty"]) ** 2).sum()
    return result


def independent(theta, groups, variant):
    objective = 0.0
    gradient = np.zeros_like(theta)
    predictions = []
    for g in groups:
        logs = (
            g["x"] @ theta
            if variant == "unregularized_sequence"
            else np.where(
                g["free_index"] >= 0,
                theta[np.maximum(g["free_index"], 0)],
                g["base_log_rates"],
            )
        )
        p, bond = numpy_forward(logs, g)
        objective += g["w"] @ ((p - g["y"]) ** 2)
        predictions.append(p)
        if variant == "unregularized_sequence":
            gradient += g["x"].T @ bond
        else:
            valid = g["free_index"] >= 0
            gradient += np.bincount(
                g["free_index"][valid], weights=bond[valid], minlength=len(theta)
            )
    return objective, gradient, predictions


def main():
    tick = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    stage = json.loads((FAMILY / "reproduce/fitting.json").read_text())
    for name, h in stage["source_hashes"].items():
        assert hashlib.sha256((FAMILY / name).read_bytes()).hexdigest() == h
    path = FAMILY.parents[1] / "models/independent_cut.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == stage["primitive_sha256"]
    old_path = __import__("pathlib").Path(cfg["reference_family"]) / "results/fits.json"
    assert (
        hashlib.sha256(old_path.read_bytes()).hexdigest()
        == cfg["reference_fits_sha256"]
    )
    saved = json.loads(old_path.read_text())["sequence_context"]
    reference = np.array(saved["coefficients"])
    groups = read_groups(cfg)
    assert {g["split"] for g in groups} == {"train"}
    free, audit = allocate(groups, reference)
    tensors(groups)
    assert not (FAMILY / "results/capacity-fits.json").exists()
    qualification = []
    baseline = independent(reference, groups, "unregularized_sequence")[0]
    assert abs(baseline - saved["data_mse"]) < 1e-12
    for variant, initial in (
        ("unregularized_sequence", reference),
        ("free_condition_rates", free),
    ):
        initial_loss, _, _ = independent(initial, groups, variant)
        assert abs(initial_loss - baseline) < 1e-12
        for perturb in (0.0, 0.1):
            value = initial + perturb * np.sin(np.arange(len(initial)) + 1)
            theta = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            objective = loss(theta, groups, variant)
            grad = torch.autograd.grad(objective, theta)[0].detach().numpy()
            expected, gradient, _ = independent(value, groups, variant)
            row = dict(
                variant=variant,
                perturbation=perturb,
                objective_error=abs(float(objective.detach()) - expected),
                gradient_error=float(np.max(np.abs(grad - gradient))),
            )
            assert row["objective_error"] < 1e-10 and row["gradient_error"] < 1e-9, row
            qualification.append(row)
    (FAMILY / "results/rate-allocation.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    (FAMILY / "results/fitting-qualification.json").write_text(
        json.dumps(qualification, indent=2) + "\n"
    )
    fits = {}
    for variant, initial in (
        ("unregularized_sequence", reference),
        ("free_condition_rates", free),
    ):
        theta = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
        optimizer = torch.optim.Adam([theta], lr=0.01)
        history = []
        checkpoints = {"0": initial.tolist()}
        coverage = np.zeros(len(initial), dtype=bool)
        now = time.perf_counter()
        for step in range(1, 2001):
            optimizer.zero_grad()
            objective = loss(theta, groups, variant)
            assert torch.isfinite(objective)
            objective.backward()
            assert torch.isfinite(theta.grad).all()
            coverage |= theta.grad.detach().numpy() != 0
            optimizer.step()
            if step % 100 == 0:
                history.append(
                    dict(
                        step=step,
                        preupdate_mse=float(objective.detach()),
                        preupdate_gradient_norm=float(theta.grad.norm()),
                        seconds=time.perf_counter() - now,
                    )
                )
                print(variant, history[-1], flush=True)
            if step in (100, 500, 1000, 2000):
                checkpoints[str(step)] = theta.detach().tolist()
        adam = theta.detach().numpy().copy()
        mse, gradient, _ = independent(adam, groups, variant)
        adam_state = {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in optimizer.state[theta].items()
        }
        (FAMILY / f"results/post-adam-{variant}.json").write_text(
            json.dumps(
                dict(
                    coefficients=adam.tolist(),
                    mse=mse,
                    gradient=gradient.tolist(),
                    optimizer_state=adam_state,
                ),
                indent=2,
            )
            + "\n"
        )
        evaluations = []
        last = {}
        iterations = [0]

        class EvaluationLimit(Exception):
            pass

        def fun(vector):
            if len(evaluations) >= 500:
                raise EvaluationLimit
            param = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
            value = loss(param, groups, variant)
            grad = torch.autograd.grad(value, param)[0].detach().numpy()
            assert np.isfinite(grad).all() and np.isfinite(float(value.detach()))
            evaluations.append(
                dict(
                    mse=float(value.detach()), gradient_norm=float(np.linalg.norm(grad))
                )
            )
            last.update(x=vector.copy(), fun=float(value.detach()), jac=grad.copy())
            return float(value.detach()), grad

        def callback(_):
            iterations[0] += 1

        try:
            polished = minimize(
                fun,
                adam,
                jac=True,
                method="L-BFGS-B",
                callback=callback,
                options={
                    "maxfun": 500,
                    "maxiter": 500,
                    "gtol": 1e-8,
                    "ftol": 1e-15,
                    "maxls": 20,
                },
            )
        except EvaluationLimit:
            polished = SimpleNamespace(
                **last,
                success=False,
                message="Explicit 500-function-evaluation limit; last evaluated state retained, not a convergence claim",
                nfev=len(evaluations),
                nit=iterations[0],
            )
        value, gradient, preds = independent(polished.x, groups, variant)
        assert (
            abs(value - float(polished.fun)) < 1e-9
            and np.max(np.abs(gradient - polished.jac)) < 1e-8
        )
        record = dict(
            coefficients=polished.x.tolist(),
            initial=initial.tolist(),
            parameter_count=len(initial),
            gradient_coverage=coverage.tolist(),
            history=history,
            checkpoints=checkpoints,
            post_adam=dict(
                coefficients=adam.tolist(),
                mse=mse,
                gradient=independent(adam, groups, variant)[1].tolist(),
                optimizer_state=adam_state,
            ),
            final_mse=value,
            final_gradient=gradient.tolist(),
            polish=dict(
                success=bool(polished.success),
                message=str(polished.message),
                nfev=int(polished.nfev),
                nit=int(polished.nit),
                evaluations=evaluations,
            ),
            wall_seconds=time.perf_counter() - now,
            all_allocated_parameters_trained=bool(coverage.all()),
            changed_parameter_count=int(np.count_nonzero(polished.x != initial)),
        )
        fits[variant] = record
        (FAMILY / f"results/fit-{variant}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(
            variant,
            {
                "parameters": len(initial),
                "post_adam_mse": mse,
                "final_mse": value,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "success": bool(polished.success),
            },
            flush=True,
        )
        with gzip.open(
            FAMILY / f"results/predictions-{variant}.jsonl.gz", "wt"
        ) as stream:
            for g, p in zip(groups, preds):
                for source, pred, w, y in zip(g["rows"], p, g["w"], g["y"]):
                    curve, t, ref, signal, reference_signal = source
                    stream.write(
                        json.dumps(
                            dict(
                                variant=variant,
                                curve_id=curve["curve_id"],
                                construct_id=g["parent"],
                                condition_id=g["condition"],
                                split="train",
                                time=t,
                                reference_time=ref,
                                signal=signal,
                                reference_signal=reference_signal,
                                target=y,
                                weight=w,
                                prediction=pred,
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
    (FAMILY / "results/capacity-fits.json").write_text(
        json.dumps(fits, indent=2) + "\n"
    )
    (FAMILY / "results/fitting-runtime.json").write_text(
        json.dumps(
            dict(
                wall_seconds=time.perf_counter() - tick,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                runtime_context=cfg["runtime_context"],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
