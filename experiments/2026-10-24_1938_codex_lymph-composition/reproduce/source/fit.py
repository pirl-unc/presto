"""Fit the preregistered controls and physical models; never load test labels."""

import math
import resource
import time

import numpy as np
import torch
from core import ROOT, Data, read, sha, write


def main():
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = read(ROOT / "reproduce/launch.json")
    frozen = read(ROOT / "reproduce/fitting.json")
    for path, digest in frozen["files"].items():
        assert sha(path) == digest, path
    assert sha(ROOT / "reproduce/launch.json") == frozen["config_sha256"]
    qualification = read(ROOT / "results/qualification.json")
    assert qualification["status"] == "passed"
    output = ROOT / "results/fit"
    output.mkdir()
    summary = {}
    seals = {}
    for block, info in read(ROOT / "results/admission.json")["cohort_support"].items():
        if not info["supported"]:
            continue
        data = Data(block)
        directory = output / block
        directory.mkdir()
        target, fractions = data.labels("train")
        validation = data.labels("val")
        weights = data.weights["train"]
        mask = data.split["train"]
        predictions = {}
        records = {}
        states = {}
        means = weights @ data.raw
        scales = np.sqrt(weights @ (data.raw - means) ** 2)
        scales[scales == 0] = 1
        centered = data.center((data.raw - means) / scales)
        transforms = {"means": means, "scales": scales}
        predictions["zero"] = np.zeros(len(data.rows))
        predictions["observer_fixed"] = data.observer.copy()
        denominator = weights @ (data.observer**2)
        gamma = (
            float((weights * target) @ data.observer / denominator)
            if denominator
            else 0.0
        )
        predictions["observer_calibrated"] = gamma * data.observer
        states["observer_calibrated"] = {"gamma": gamma}
        selected = {}
        for family, width in (("length", 2), ("sequence", 62)):
            x = centered[:, :width]
            _, singular, vt = np.linalg.svd(
                np.sqrt(weights[mask, None]) * x[mask], full_matrices=False
            )
            keep = singular > singular.max() * 1e-10
            projection = vt[keep].T
            design = x @ projection
            dimension = design.shape[1]
            assert dimension > 0
            transforms[family + "_projection"] = projection
            for alpha in config["fit"]["ridge_alphas"]:
                name = f"{family}_{alpha:g}"
                matrix = design.T @ (
                    weights[:, None] * design
                ) + alpha / dimension * np.eye(dimension)
                rhs = design.T @ (weights * target)
                beta = np.linalg.solve(matrix, rhs)
                assert (
                    np.linalg.norm(matrix @ beta - rhs)
                    / max(np.linalg.norm(rhs), 1e-12)
                    < 1e-8
                )
                predictions[name] = design @ beta
                states[name] = {
                    "beta": beta.tolist(),
                    "alpha": alpha,
                    "rank": dimension,
                    "family": family,
                }
            selected[family] = min(
                (name for name in predictions if name.startswith(family + "_")),
                key=lambda name: data.metrics(predictions[name], "val", validation)[
                    "mse"
                ],
            )
        np.savez(directory / "transforms.npz", **transforms)
        for name, prediction in predictions.items():
            records[name] = {
                "train": data.metrics(prediction, "train", (target, fractions)),
                "val": data.metrics(prediction, "val", validation),
                "kind": "direct_control",
            }
        qualified = next(p for p in qualification["probes"] if p["block"] == block)
        arms = [
            (f"{rate}_{obs}_mse", "mse", rate == "sequence", obs == "observer", 42)
            for rate in ("uniform", "sequence")
            for obs in ("alone", "observer")
        ]
        arms += [
            (f"sequence_observer_{method}_{seed}", method, True, True, seed)
            for method in ("exact", "reinforce")
            for seed in config["fit"]["seeds"]
            if method != "reinforce" or qualified["sampled_admitted"]
        ]
        y = torch.tensor(target, dtype=torch.float64)
        p = torch.tensor(fractions, dtype=torch.float64)
        for name, kind, sequence, observer, seed in arms:
            began = time.perf_counter()
            theta = torch.zeros(
                39 if sequence else 1, dtype=torch.float64, requires_grad=True
            )
            with torch.no_grad():
                theta[0] = math.log(0.01)
            initial = theta.detach().numpy().copy()
            optimizer = torch.optim.Adam(
                [theta],
                lr=config["fit"]["lr"],
                betas=tuple(config["fit"]["betas"]),
                eps=config["fit"]["epsilon"],
            )
            generator = torch.Generator().manual_seed(seed)
            maxgrad = np.zeros(len(theta))
            history = []
            trajectory = [initial.tolist()]
            for step in range(config["fit"]["steps"]):
                optimizer.zero_grad()
                score = data.score(theta, sequence, observer)
                loss = data.loss(
                    score, y, p, kind, generator, config["fit"]["reinforce_samples"]
                )
                penalty = (
                    config["fit"]["residue_penalty"] * theta[1:].square().mean()
                    if sequence
                    else theta[0] * 0
                )
                total = loss + penalty
                assert torch.isfinite(total)
                total.backward()
                assert torch.isfinite(theta.grad).all()
                maxgrad = np.maximum(maxgrad, np.abs(theta.grad.numpy()))
                optimizer.step()
                trajectory.append(theta.detach().tolist())
                if step % 25 == 0 or step + 1 == config["fit"]["steps"]:
                    history.append(
                        {
                            "step": step + 1,
                            "training_estimator": loss.item(),
                            "penalty": penalty.item(),
                            "theta_norm": theta.detach().norm().item(),
                            "gradient_norm": theta.grad.norm().item(),
                        }
                    )
                    assert (
                        time.perf_counter() - began < config["limits"]["stage_seconds"]
                    )
                    assert (
                        time.perf_counter() - start
                        < config["limits"]["fit_total_seconds"]
                    )
                    assert (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
                        < config["limits"]["rss_gib"]
                    )
            with torch.no_grad():
                prediction = data.score(theta, sequence, observer).numpy()
            predictions[name] = prediction
            state = {
                "theta": theta.detach().tolist(),
                "sequence": sequence,
                "observer": observer,
                "method": kind,
                "seed": seed,
                "history": history,
                "trajectory": trajectory,
                "max_abs_gradient": maxgrad.tolist(),
                "parameter_change": (theta.detach().numpy() - initial).tolist(),
                "n_allocated_parameters": len(theta),
                "n_changed_parameters": int(
                    np.count_nonzero(theta.detach().numpy() != initial)
                ),
                "n_gradient_supported_parameters": int(
                    np.count_nonzero(maxgrad > 1e-12)
                ),
                "runtime_seconds": time.perf_counter() - began,
                "n_candidate_scores": len(data.rows) * (config["fit"]["steps"] + 1),
                "n_sampled_peptides": config["fit"]["reinforce_samples"]
                * sum(s == "train" for s in data.group_split)
                * config["fit"]["steps"]
                if kind == "reinforce"
                else 0,
            }
            states[name] = state
            torch.save(
                {"optimizer": optimizer.state_dict(), "rng": generator.get_state()},
                directory / (name + "-optimizer.pt"),
            )
            records[name] = {
                "train": data.metrics(prediction, "train", (target, fractions)),
                "val": data.metrics(prediction, "val", validation),
                "kind": "physical",
                "runtime_seconds": state["runtime_seconds"],
            }
            print(
                block, name, "finished", round(state["runtime_seconds"], 3), flush=True
            )
        np.savez(directory / "sealed-predictions.npz", **predictions)
        write(
            directory / "row-identities.json",
            [
                {"measurement_id": r["measurement_id"], "split": r["split"]}
                for r in data.rows
            ],
        )
        write(directory / "states.json", states)
        write(directory / "validation-selection.json", selected)
        write(directory / "train-validation-metrics.json", records)
        seals[block] = {
            path.name: sha(path)
            for path in sorted(directory.iterdir())
            if path.is_file()
        }
        summary[block] = {
            "n_models": len(predictions),
            "selected_controls": selected,
            "observer_calibration": gamma,
            "n_rows": len(data.rows),
        }
    result = {
        "status": "completed",
        "cohorts": summary,
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
        "files": seals,
        "test_labels_read": False,
    }
    write(ROOT / "results/fit-complete.json", result)
    print(result)


if __name__ == "__main__":
    main()
