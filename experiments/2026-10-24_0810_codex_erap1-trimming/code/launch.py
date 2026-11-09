"""Frozen local exact/sampled comparison of ERAP1 N-terminal trimming."""

import argparse
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
from core import ROOT, Dataset, objective, read, sha, write
from scipy.stats import pearsonr, spearmanr


def metrics(y, p, w):
    w = w / w.sum()
    mse = float(w @ (y - p) ** 2)
    var = float(w @ (y - w @ y) ** 2)
    varying = np.ptp(p) > 1e-14 and np.ptp(y) > 0
    return {
        "n_observations": len(y),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(w @ abs(y - p)),
        "r2": 1 - mse / var if var else None,
        "pearson": float(pearsonr(y, p).statistic) if varying else None,
        "spearman": float(spearmanr(y, p).statistic) if varying else None,
        "prediction_sd": float(np.sqrt(w @ (p - w @ p) ** 2)),
    }


def receipt(action):
    destination = ROOT / "reproduce" / action
    destination.mkdir(exist_ok=False)
    repository = ROOT.parents[1]
    files = [
        ROOT / "code/core.py",
        ROOT / "code/launch.py",
        repository / "models/n_terminal_trimming.py",
        repository / "models/steady_processing.py",
    ]
    snapshots = {}
    for file in files:
        target = destination / file.name
        shutil.copyfile(file, target)
        snapshots[file.name] = sha(target)
    write(
        destination / "execution.json",
        {
            "action": action,
            "argv": sys.argv,
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": subprocess.check_output(
                ["git", "status", "--short", "--untracked-files=no"], text=True
            ),
            "source_sha256": snapshots,
            "environment": {
                k: os.environ.get(k)
                for k in [
                    "PYTHONPATH",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                ]
            },
        },
    )


def qualify():
    receipt("qualification")
    start = time.perf_counter()
    d = Dataset()
    np.savez_compressed(ROOT / "results/path-inputs.npz", **d.arrays)
    pq.write_table(
        pa.Table.from_pylist(d.rows), ROOT / "results/cohort.parquet", compression="zstd"
    )
    write(ROOT / "results/origins.json", d.origins)
    write(
        ROOT / "results/parents.json",
        {o["parent_id"]: d.parent_sequences[o["parent_id"]] for o in d.origins},
    )
    write(
        ROOT / "results/input-coverage.json",
        {
            "n_observations": len(d.rows),
            "n_peptides": len(d.peptides),
            "n_paths": len(d.arrays["owner"]),
            "n_origins": len(d.origins),
            "fold_observations": {str(k): int(sum(d.fold == k)) for k in range(5)},
            "n_parameters_sequence": 21,
            "n_parameters_uniform": 2,
            "outer_test_loaded": False,
            "files": {
                name: sha(ROOT / "results" / name)
                for name in ["path-inputs.npz", "cohort.parquet", "origins.json", "parents.json"]
            },
        },
    )
    for fold in ["full", *range(5)]:
        d.baseline_for_fit(fold)
        queries = np.unique(d.owner_rows if fold == "full" else d.owner_rows[d.fold != fold])
        support = d.arrays["counts"][np.isin(d.arrays["owner"], queries)].sum(axis=0)
        assert (support > 0).all()
    baseline, gamma = d.baseline_for_fit("full")
    rng = np.random.default_rng(42)
    direction = rng.normal(size=21)
    probes = [np.zeros(21), 0.2 * direction, -0.2 * direction]
    exact, finite = [], []
    for index, values in enumerate(probes):
        theta = torch.tensor(values, requires_grad=True)
        loss, _, _ = objective(d, theta, "full", "augmented_exact", baseline, gamma, penalty=False)
        gradient = torch.autograd.grad(loss, theta)[0].detach().numpy()
        exact.append(gradient)
        for _ in range(3):
            direction = rng.normal(size=21)
            direction /= np.linalg.norm(direction)
            epsilon = 1e-5
            with torch.no_grad():
                upper = objective(
                    d,
                    torch.tensor(values + epsilon * direction),
                    "full",
                    "augmented_exact",
                    baseline,
                    gamma,
                    penalty=False,
                )[0].item()
                lower = objective(
                    d,
                    torch.tensor(values - epsilon * direction),
                    "full",
                    "augmented_exact",
                    baseline,
                    gamma,
                    penalty=False,
                )[0].item()
            error = abs((upper - lower) / (2 * epsilon) - gradient @ direction)
            finite.append({"probe": index, "absolute_directional_error": error})
            assert error < 1e-7
        print(f"Exact/finite-difference probe {index} passed", flush=True)
    attempts, chosen = [], None
    for k in d.cfg["sample_sizes"]:
        records = []
        for index, values in enumerate(probes):
            gradients = []
            for repeat in range(d.cfg["probe_repetitions"]):
                generator = torch.Generator().manual_seed(5100 + index * 100 + repeat)
                uniforms = torch.rand(
                    (len(d.peptides), k), generator=generator, dtype=torch.float64
                )
                theta = torch.tensor(values, requires_grad=True)
                loss, _, _ = objective(
                    d,
                    theta,
                    "full",
                    "augmented_sampled",
                    baseline,
                    gamma,
                    uniforms=uniforms,
                    penalty=False,
                )
                gradients.append(torch.autograd.grad(loss, theta)[0].detach().numpy())
            gradients = np.array(gradients)
            denominator = max(np.linalg.norm(exact[index]), 1e-12)
            mean_error = np.linalg.norm(gradients.mean(0) - exact[index]) / denominator
            rms = np.sqrt(np.mean(np.sum((gradients - exact[index]) ** 2, axis=1))) / denominator
            records.append(
                {
                    "probe": index,
                    "exact": exact[index].tolist(),
                    "samples": gradients.tolist(),
                    "mean_relative_error": float(mean_error),
                    "rms_relative_error": float(rms),
                    "passed": bool(mean_error <= 0.1 and rms <= 0.25),
                }
            )
        attempts.append({"k": k, "probes": records})
        print(
            json.dumps(
                {
                    "k": k,
                    "probe_errors": [
                        (r["mean_relative_error"], r["rms_relative_error"]) for r in records
                    ],
                }
            ),
            flush=True,
        )
        if all(r["passed"] for r in records):
            chosen = k
            break
    result = {
        "sample_size": chosen,
        "sampler_passed": chosen is not None,
        "finite_difference": finite,
        "attempts": attempts,
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    write(ROOT / "results/qualification.json", result)
    assert result["seconds"] <= 900 and result["peak_gib"] <= 4
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"attempts", "finite_difference"}}),
        flush=True,
    )


def train():
    qualification = read(ROOT / "results/qualification.json")
    receipt("training")
    output = ROOT / "results/fits"
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    d = Dataset()
    expected = read(ROOT / "results/input-coverage.json")
    for name, digest in expected["files"].items():
        assert sha(ROOT / "results" / name) == digest
    arms = ["uniform_exact", "residue_exact", "augmented_exact"]
    if qualification["sampler_passed"]:
        arms.append("augmented_sampled")
    all_results = {}
    for arm in arms:
        arm_dir = output / arm
        arm_dir.mkdir()
        full = cv = None
        cv = np.full(len(d.rows), np.nan)
        correction_cv = np.full(len(d.rows), np.nan)
        fit_records = []
        for fold in ["full", *range(5)]:
            t = time.perf_counter()
            directory = arm_dir / str(fold)
            directory.mkdir()
            baseline, gamma = d.baseline_for_fit(fold)
            parameter = torch.nn.Parameter(
                torch.zeros(2 if arm == "uniform_exact" else 21, dtype=torch.float64)
            )
            optimizer = torch.optim.Adam([parameter], lr=0.03, betas=(0.9, 0.999), eps=1e-8)
            generator = torch.Generator().manual_seed(42)
            torch.save(
                {
                    "parameter": parameter.detach(),
                    "optimizer": optimizer.state_dict(),
                    "rng": generator.get_state(),
                },
                directory / "step-000.pt",
            )
            history = []
            for step in range(1, 151):
                optimizer.zero_grad(set_to_none=True)
                uniforms = (
                    torch.rand(
                        (len(d.peptides), qualification["sample_size"]),
                        generator=generator,
                        dtype=torch.float64,
                    )
                    if arm == "augmented_sampled"
                    else None
                )
                loss, _, detail = objective(
                    d, parameter, fold, arm, baseline, gamma, uniforms=uniforms
                )
                assert torch.isfinite(loss)
                loss.backward()
                gradient = float(parameter.grad.detach().abs().max())
                optimizer.step()
                history.append(
                    {
                        "step": step,
                        "objective": loss.item(),
                        "data_loss": detail["data_loss"].item(),
                        "regularizer": detail["regularizer"].item(),
                        "gradient_inf": gradient,
                        "kappa": float(torch.sigmoid(parameter[-1]).detach()),
                        "parameter_norm": float(parameter.detach().norm()),
                    }
                )
                if step % 50 == 0:
                    torch.save(
                        {
                            "parameter": parameter.detach(),
                            "optimizer": optimizer.state_dict(),
                            "rng": generator.get_state(),
                        },
                        directory / f"step-{step:03d}.pt",
                    )
                    print(
                        json.dumps(
                            {
                                "arm": arm,
                                "fold": fold,
                                **history[-1],
                                "elapsed_seconds": time.perf_counter() - t,
                            }
                        ),
                        flush=True,
                    )
            loss, prediction, detail = objective(d, parameter, fold, arm, baseline, gamma)
            exact_gradient = torch.autograd.grad(loss, parameter)[0].detach().numpy()
            prediction = prediction.detach().numpy()
            correction = detail["correction"].detach().numpy()
            record = {
                "arm": arm,
                "fold": fold,
                "gamma": gamma,
                "parameter": parameter.detach().tolist(),
                "offset": detail["offset"].item(),
                "objective": loss.item(),
                "data_loss": detail["data_loss"].item(),
                "regularizer": detail["regularizer"].item(),
                "exact_gradient_inf": float(abs(exact_gradient).max()),
                "n_changed_parameters": int(np.sum(parameter.detach().numpy() != 0)),
                "n_allocated_parameters": len(parameter),
                "seconds": time.perf_counter() - t,
                "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
                "n_exact_forward_path_scores": 2 * len(d.arrays["owner"]) * 151,
                "n_sampled_path_scores": 2 * len(d.peptides) * qualification["sample_size"] * 150
                if arm == "augmented_sampled"
                else 0,
            }
            write(directory / "fit.json", record)
            write(directory / "history.json", history)
            np.savez_compressed(
                directory / "predictions.npz",
                prediction=prediction,
                correction=correction,
                exact_gradient=exact_gradient,
            )
            assert (
                record["seconds"] <= (1200 if arm == "augmented_sampled" else 600)
                and record["peak_gib"] <= 4
            )
            fit_records.append(record)
            if fold == "full":
                full = prediction.copy()
            else:
                cv[d.fold == fold] = prediction[d.fold == fold]
                correction_cv[d.fold == fold] = correction[d.fold == fold]
            assert time.perf_counter() - start <= 7200
        assert np.isfinite(cv).all()
        for population, values in [("train", full), ("inner_cv", cv)]:
            pq.write_table(
                pa.table(
                    {
                        "measurement_id": [r["measurement_id"] for r in d.rows],
                        "peptide_sequence": [r["peptide_sequence"] for r in d.rows],
                        "component_id": [r["component_id"] for r in d.rows],
                        "fold": d.fold,
                        "target_log10_ratio": d.target,
                        "prediction_log10_ratio": values,
                        "weight": d.weight,
                    }
                ),
                arm_dir / f"{population}.parquet",
                compression="zstd",
            )
        result = {
            "train": metrics(d.target, full, d.weight),
            "inner_cv": metrics(d.target, cv, d.weight),
            "correction_sd_inner_cv": float(
                np.sqrt(d.weight @ (correction_cv - d.weight @ correction_cv) ** 2)
            ),
            "fits": fit_records,
        }
        write(arm_dir / "summary.json", result)
        all_results[arm] = result
    controls = {
        name: {pop: metrics(d.target, pred, d.weight) for pop, pred in values.items()}
        for name, values in d.baselines.items()
    }
    baseline_rmse = controls["sequence_0.1"]["inner_cv"]["rmse"]
    improvement = 1 - all_results["augmented_exact"]["inner_cv"]["rmse"] / baseline_rmse
    result = {
        "arms": all_results,
        "controls": controls,
        "relative_augmented_exact_cv_improvement": improvement,
        "development_gate_pass": bool(
            improvement >= 0.02 and all_results["augmented_exact"]["correction_sd_inner_cv"] > 0.001
        ),
        "outer_validation_test_scored": False,
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    write(output / "summary.json", result)
    print(
        json.dumps(
            {
                "gate": result["development_gate_pass"],
                "improvement": improvement,
                "seconds": result["seconds"],
                "peak_gib": result["peak_gib"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["qualify", "train"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    qualify() if args.action == "qualify" else train()


if __name__ == "__main__":
    main()
