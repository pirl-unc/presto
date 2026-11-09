"""Frozen ridge and matched exact/discrete fits; this process never opens test labels."""

import json
import time

import numpy as np
import torch
from core import ROOT, Data, Distribution, metrics, sha, write


def main():
    assert json.loads((ROOT / "results/qualification.json").read_text())["passed"]
    torch.set_num_threads(1)
    data = Data()
    train = data.indices["train"]
    val = data.indices["val"]
    y = data.labels["train"]["y"]
    weights = data.weights[train]
    models = {}
    predictions = {"uniform": np.zeros(len(data.rows))}
    choices = {}
    runtimes = []
    bases = {}
    start_family = time.monotonic()
    for family, indices in [("length", [0]), ("sequence", list(range(61)))]:
        design, basis = data.basis(indices)
        bases[family] = basis
        x = design[train]
        dimension = x.shape[1]
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            start = time.perf_counter()
            name = f"{family}_ridge_{alpha:g}"
            beta = np.linalg.solve(
                x.T @ (weights[:, None] * x) + (alpha / dimension) * np.eye(dimension),
                x.T @ (weights * y),
            )
            predictions[name] = design @ beta
            validation, _ = metrics(
                [data.rows[i] for i in val],
                data.labels["val"]["y"],
                data.labels["val"]["p"],
                predictions[name][val],
            )
            models[name] = {
                "basis": family,
                "beta": beta.tolist(),
                "alpha": alpha,
                "validation": validation,
                "n_parameters": dimension,
                "n_changed_parameters": int((abs(beta) > 1e-12).sum()),
            }
            runtimes.append(
                {
                    "name": name,
                    "wall_seconds": time.perf_counter() - start,
                    "solver": "weighted_ridge",
                }
            )
        choices[family] = min(
            (n for n in models if n.startswith(family + "_ridge")),
            key=lambda n: models[n]["validation"]["rmse"],
        )
    design, basis = data.basis(range(61))
    assert basis == bases["sequence"]
    distribution = Distribution(
        [data.rows[i] for i in train], design[train], data.labels["train"]["p"]
    )
    (ROOT / "results/checkpoints").mkdir()
    for method, seed in [("exact", None), ("reinforce", 42), ("reinforce", 43), ("reinforce", 44)]:
        name = method if seed is None else f"{method}_{seed}"
        beta = torch.zeros(design.shape[1], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([beta], lr=0.02, betas=(0.9, 0.999), eps=1e-8)
        generator = torch.Generator().manual_seed(seed if seed is not None else 0)
        maximum = np.zeros(design.shape[1])
        history = []
        start = time.perf_counter()
        cpu = time.process_time()
        for step in range(500):
            optimizer.zero_grad(set_to_none=True)
            loss = (
                distribution.exact(beta)
                if method == "exact"
                else distribution.sampled(beta, generator)
            )
            loss.backward()
            assert torch.isfinite(beta.grad).all()
            maximum = np.maximum(maximum, abs(beta.grad.detach().numpy()))
            optimizer.step()
            with torch.no_grad():
                exact_objective = float(distribution.exact(beta))
            history.append(
                {
                    "step": step + 1,
                    "exact_objective": exact_objective,
                    "gradient_maxabs": float(beta.grad.abs().max()),
                }
            )
            if time.perf_counter() - start > 1200:
                raise RuntimeError("Per-fit runtime exceeded")
        elapsed = time.perf_counter() - start
        final = beta.detach().numpy().copy()
        prediction = design @ final
        predictions[name] = prediction
        validation, _ = metrics(
            [data.rows[i] for i in val],
            data.labels["val"]["y"],
            data.labels["val"]["p"],
            prediction[val],
        )
        models[name] = {
            "basis": "sequence",
            "beta": final.tolist(),
            "method": method,
            "seed": seed,
            "validation": validation,
            "n_parameters": len(final),
            "n_changed_parameters": int((abs(final) > 1e-12).sum()),
            "n_gradient_active_parameters": int((maximum > 1e-14).sum()),
            "maximum_coordinate_gradients": maximum.tolist(),
        }
        assert (
            models[name]["n_changed_parameters"]
            == models[name]["n_gradient_active_parameters"]
            == len(final)
        )
        runtimes.append(
            {
                "name": name,
                "wall_seconds": elapsed,
                "cpu_seconds": time.process_time() - cpu,
                "updates": 500,
                "training_candidate_scores": 1000 * len(train),
                "score_forward_passes": 1000,
                "sample_draws": 500 * len(distribution.counts) * 64 if method == "reinforce" else 0,
            }
        )
        write(ROOT / f"results/checkpoints/{name}-trajectory.json", history)
        torch.save(
            {
                "beta": beta.detach(),
                "optimizer": optimizer.state_dict(),
                "generator": generator.get_state(),
            },
            ROOT / f"results/checkpoints/{name}.pt",
        )
        print(
            json.dumps({"model": name, "wall_seconds": elapsed, "validation": validation}),
            flush=True,
        )
    np.savez_compressed(ROOT / "results/covariate-predictions.npz", **predictions)
    write(
        ROOT / "results/models.json",
        {
            "models": models,
            "bases": bases,
            "validation_choices": choices,
            "runtime": runtimes,
            "family_fit_seconds": time.monotonic() - start_family,
        },
    )
    paths = [ROOT / "results/models.json", ROOT / "results/covariate-predictions.npz"] + sorted(
        (ROOT / "results/checkpoints").glob("*")
    )
    write(ROOT / "results/prediction-seal.json", {str(p.relative_to(ROOT)): sha(p) for p in paths})
    print(
        json.dumps({"validation_choices": choices, "sealed_models": len(predictions)}), flush=True
    )


if __name__ == "__main__":
    main()
