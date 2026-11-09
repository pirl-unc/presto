"""Fixed TAP subgroup family with shared conditional response and matched controls."""

import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from core import (
    CONTEXT,
    ETA0,
    EXP,
    LN10,
    RECIPE,
    ROOT,
    SCORES,
    VARIANTS,
    data,
    digest,
    numpy_prediction,
    objective,
    reference,
    standardize,
    write,
)


def fit_route(d, it, ie, variant, part, out):
    started = time.perf_counter()
    rows = [d["rows"][i] for i in it]
    w = reference.weights(rows)
    _, nuisance = reference.fit(rows, rows, np.empty((len(it), 0)), np.empty((len(it), 0)))
    z, zstate = standardize(d["availability"][it], w)
    scores, sstate = standardize(d["scores"][it], w)
    assert scores.shape[1] == 3
    theta = torch.nn.Parameter(torch.zeros(z.shape[1] + 1, dtype=torch.float64))
    eta = torch.nn.Parameter(torch.full((2,), ETA0, dtype=torch.float64))
    with_scores = variant != "route_only_exact"
    beta = torch.nn.Parameter(torch.zeros((2, 3), dtype=torch.float64), requires_grad=with_scores)
    params = [theta, eta] + ([beta] if with_scores else [])
    optimizer = torch.optim.Adam(params, lr=0.03)
    generator = torch.Generator().manual_seed(42) if variant.endswith("sampled") else None
    tensors = [
        torch.tensor(v) for v in [z, scores, d["condition"][it], d["y"][it], d["precursor"][it], w]
    ]
    gamma = nuisance["gamma"]
    checkpoints, history = {}, []
    touched = np.zeros(sum(p.numel() for p in params), dtype=bool)
    sampled_draws = 0
    for step in range(251):
        if step:
            optimizer.zero_grad()
            loss, *_ = objective(theta, eta, beta, *tensors, gamma, generator)
            loss.backward()
            gradients = torch.cat([p.grad.flatten() for p in params])
            assert torch.isfinite(gradients).all() and torch.isfinite(loss)
            touched |= abs(gradients.detach().numpy()) > 1e-12
            optimizer.step()
            if generator is not None:
                sampled_draws += len(it) * 64
        if step % 50:
            continue
        optimizer.zero_grad()
        loss, pred, offsets, _response, _score_term = objective(theta, eta, beta, *tensors, gamma)
        loss.backward()
        gradients = torch.cat([p.grad.flatten() for p in params]).detach().numpy()
        cp = {
            "variant": variant,
            "part": part,
            "step": step,
            "gamma": gamma,
            "availability_transform": zstate,
            "score_transform": sstate,
            "theta": theta.detach().tolist(),
            "eta": eta.detach().tolist(),
            "beta": beta.detach().tolist(),
            "offsets": offsets.detach().tolist(),
            "objective": loss.item(),
            "analytic_gradient": gradients.tolist(),
            "gradient_norm": float(np.linalg.norm(gradients)),
            "supported_parameter_count": len(gradients),
            "gradient_touched_count": int(touched.sum()),
            "fit_indices": it.tolist(),
            "eval_indices": ie.tolist(),
        }
        p, latent = numpy_prediction(d, it, cp)
        err = float(np.max(abs(p - pred.detach().numpy())))
        assert err < 1e-10
        record = {
            "step": step,
            "objective": loss.item(),
            "gradient_norm": cp["gradient_norm"],
            "train": reference.metrics(d["y"][it], p, w),
            "epsilon": torch.sigmoid(eta).detach().tolist(),
            "rho_range": [float(latent["rho"].min()), float(latent["rho"].max())],
            "delivery_sd": float(np.std(latent["relative_delivery"])),
            "torch_numpy_prediction_error": err,
        }
        history.append(record)
        checkpoints[str(step)] = cp
    full, latent = numpy_prediction(d, it, cp)
    eval_prediction, eval_latent = numpy_prediction(d, ie, cp)
    if part == "full":
        rho, epsilon, r = latent["rho"], latent["epsilon"], latent["relative_delivery"]
        dz = -(1 - epsilon) * rho * (1 - rho) / r / LN10
        de = epsilon * rho * (1 - epsilon) / r / LN10
        cond = d["condition"][it]
        jac = [
            dz[:, None] * np.c_[np.ones(len(it)), z],
            np.stack([de * (cond == c) for c in range(2)], axis=1),
        ]
        if with_scores:
            jac.extend([scores * (cond == c)[:, None] for c in range(2)])
        jac = np.concatenate(jac, axis=1)
        for c in range(2):
            select = cond == c
            jac[select] -= (w[select] / w[select].sum()) @ jac[select]
        singular = np.linalg.svd(np.sqrt(w)[:, None] * jac, compute_uv=False)
        cp["profiled_prediction_jacobian"] = {
            "singular_values": singular.tolist(),
            "rank_relative_1e_10": int(np.sum(singular > singular[0] * 1e-10)),
            "columns": jac.shape[1],
            "rows": len(it),
            "interpretation": "Local response identifiability only; no causal routing or molar scale.",
        }
    delta = (history[-2]["objective"] - history[-1]["objective"]) / max(
        abs(history[-2]["objective"]), 1e-8
    )
    fit = {
        "variant": variant,
        "part": part,
        "seconds": time.perf_counter() - started,
        "sampled_route_draws": sampled_draws,
        "last_50_step_relative_objective_change": delta,
        "still_improving_over_1_percent": delta > 0.01,
        "train": reference.metrics(d["y"][it], full, w),
        "evaluation": reference.metrics(d["y"][ie], eval_prediction, d["weights"][ie]),
        "checkpoint": cp,
        "history": history,
    }
    folder = out / variant / part
    folder.mkdir(parents=True)
    write(folder / "fit.json", fit)
    write(folder / "checkpoints.json", checkpoints)
    print(
        json.dumps({k: v for k, v in fit.items() if k not in {"checkpoint", "history"}}), flush=True
    )
    return eval_prediction, eval_latent, fit


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert json.loads((EXP / "results/qualification.json").read_text())["status"] == "passed"
    out = EXP / "results/fits"
    out.mkdir(exist_ok=False)
    for f in (EXP / "code").glob("*.py"):
        shutil.copyfile(f, EXP / "reproduce/source" / f.name)
    shutil.copyfile(CONTEXT / "code/core.py", EXP / "reproduce/source/reference-core.py")
    sources = [
        RECIPE / "manifest.json",
        RECIPE / "derived/responses.parquet",
        CONTEXT / "results/fits/context-features.npz",
        CONTEXT / "code/core.py",
        SCORES / "results/predictor/scores.parquet",
        SCORES / "results/predictor/exposure.parquet",
        EXP / "reproduce/plan.md",
        *sorted((EXP / "code").glob("*.py")),
    ]
    receipt = {
        "agent_model": "Codex / GPT-6",
        "argv": sys.argv,
        "python": sys.executable,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], text=True
        ),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "environment": {
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "PYTHONDONTWRITEBYTECODE",
            ]
        },
        "versions": {
            k: importlib.metadata.version(k)
            for k in ["numpy", "pandas", "torch", "pyarrow", "scipy"]
        },
        "variants": VARIANTS,
        "steps": 250,
        "lr": 0.03,
        "seed": 42,
        "samples": 64,
        "outer_validation_test_scored": False,
    }
    write(EXP / "reproduce/launch.json", receipt)
    d = data()
    pq.write_table(d["table"], out / "cohort.parquet", compression="zstd")
    np.savez_compressed(
        out / "features.npz",
        availability=d["availability"],
        scores=d["scores"],
        fold=d["fold"],
        condition=d["condition"],
        weights=d["weights"],
    )
    predictions = {
        k: d["table"][k]
        for k in [
            "measurement_id",
            "source_record_id",
            "component_id",
            "condition_id",
            "peptide_sequence",
        ]
    }
    predictions.update(
        fold=d["fold"],
        weight=d["weights"],
        target_log10_ratio=d["y"],
        frozen_predictor_training_exposure=d["exposure"],
    )
    summary, strata, fits = {}, [], {}
    for variant in VARIANTS:
        t = time.perf_counter()
        full = np.full(len(d["rows"]), np.nan)
        cv = np.full_like(full, np.nan)
        latent_full, latent_cv = {}, {}
        fits[variant] = {}
        for part in ["full", "0", "1", "2", "3", "4"]:
            it = np.arange(len(full)) if part == "full" else np.flatnonzero(d["fold"] != int(part))
            ie = np.arange(len(full)) if part == "full" else np.flatnonzero(d["fold"] == int(part))
            if variant in {"sequence_scores", "expanded_ridge"}:
                phi = (
                    d["availability"][:, :61] if variant == "sequence_scores" else d["availability"]
                )
                phi = np.c_[phi, d["scores"]]
                tr, ev = [d["rows"][i] for i in it], [d["rows"][i] for i in ie]
                p, cp = reference.fit(tr, ev, phi[it], phi[ie])
                fit = {
                    "variant": variant,
                    "part": part,
                    "checkpoint": cp,
                    "fit_indices": it.tolist(),
                    "eval_indices": ie.tolist(),
                }
                folder = out / variant / part
                folder.mkdir(parents=True)
                write(folder / "fit.json", fit)
                latent = {}
            else:
                p, latent, fit = fit_route(d, it, ie, variant, part, out)
            fits[variant][part] = {
                k: v for k, v in fit.items() if k not in {"checkpoint", "history"}
            }
            if part == "full":
                full[:] = p
                latent_full = latent
            else:
                cv[ie] = p
                for k, v in latent.items():
                    latent_cv.setdefault(k, np.full_like(full, np.nan))[ie] = v
                strata.append(
                    {
                        "variant": variant,
                        "kind": "fold",
                        "label": part,
                        **reference.metrics(d["y"][ie], p, d["weights"][ie]),
                    }
                )
            assert time.perf_counter() - started < 1800
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3 < 4
        for population, p, latent in [("train", full, latent_full), ("inner_cv", cv, latent_cv)]:
            assert np.isfinite(p).all()
            predictions[f"{variant}_{population}"] = p
            for k, v in latent.items():
                predictions[f"{variant}_{population}_{k}"] = v
            for kind, values in [("condition", d["condition"]), ("exposure", d["exposure"])]:
                for value in sorted(set(values)):
                    m = values == value
                    strata.append(
                        {
                            "variant": variant,
                            "kind": population + "_" + kind,
                            "label": str(value),
                            **reference.metrics(d["y"][m], p[m], d["weights"][m]),
                        }
                    )
        summary[variant] = {
            "train": reference.metrics(d["y"], full, d["weights"]),
            "inner_cv": reference.metrics(d["y"], cv, d["weights"]),
            "seconds": time.perf_counter() - t,
        }
        print(json.dumps({"variant": variant, **summary[variant]}), flush=True)
    pq.write_table(pa.table(predictions), out / "predictions.parquet", compression="zstd")
    write(out / "strata.json", strata)
    write(out / "fit-index.json", fits)

    def rmse(v):
        return summary[v]["inner_cv"]["rmse"]

    gain = 1 - rmse("route_scores_exact") / rmse("sequence_scores")
    rich_gain = 1 - rmse("route_scores_exact") / rmse("expanded_ridge")
    sf = {(s["variant"], s["label"]): s["rmse"] for s in strata if s["kind"] == "fold"}
    improved = sum(
        sf["route_scores_exact", str(k)] < sf["expanded_ridge", str(k)] for k in range(5)
    )
    sample_delta = rmse("route_scores_sampled") / rmse("route_scores_exact") - 1
    result = {
        "metrics": summary,
        "gain_vs_sequence_scores": gain,
        "gain_vs_expanded_ridge": rich_gain,
        "folds_improved": improved,
        "development_gate_pass": gain >= 0.02 and rich_gain >= 0.01 and improved >= 4,
        "sampled_relative_rmse_difference": sample_delta,
        "sampled_agreement_pass": abs(sample_delta) <= 0.02,
        "seconds": time.perf_counter() - started,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "outer_validation_test_scored": False,
    }
    write(out / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}), flush=True)


if __name__ == "__main__":
    main()
