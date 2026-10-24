"""Fit fixed continuous observers and seal test predictions without test outcomes."""

import hashlib
import json
import os
import resource
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from model import (
    FEATURE_NAMES,
    Observer,
    center_numpy,
    center_torch,
    features,
    group_data,
    metrics,
    parameter_penalty,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data/measured_ms1_grouped_observer/2026-10-24_1702_canonical"
OUTPUT = ROOT / "results/fit"
ALPHAS = (0.001, 0.01, 0.1, 1.0)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load(name, expected):
    path = DATA / name
    assert digest(path) == expected["sha256"]
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    for name in ("fit.py", "model.py"):
        shutil.copy2(Path(__file__).parent / name, ROOT / "reproduce/source" / name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    admission = json.loads((ROOT / "results/admission.json").read_text())
    assert admission["status"] == "admitted"
    verification = json.loads((ROOT / "results/source-verification.json").read_text())
    assert verification["status"] == "passed"
    assert (
        verification["n_measured_targets_verified"]
        == admission["coverage"]["n_examples"]
    )
    assert admission["peak_rss_gib"] <= 3
    assert admission["raw_and_package_logical_bytes"] <= 2 * 1024**3
    assert admission["free_disk_bytes"] >= 2 * 1024**3
    rows = {
        split: load(name, admission["split_files"][name])
        for split, name in [
            ("train", "train.jsonl"),
            ("validation", "validation.jsonl"),
            ("test", "test-covariates.jsonl"),
        ]
    }
    assert all("target" not in row for row in rows["test"])
    allowed = {
        "measurement_id",
        "source_record_id",
        "sequence",
        "run_id",
        "pool",
        "library",
        "dependence_component",
    }
    assert all(set(row) == allowed for row in rows["test"])
    raw = {split: features(value) for split, value in rows.items()}
    group = {split: group_data(value) for split, value in rows.items()}
    target = {
        split: np.array([r["target"] for r in rows[split]], dtype=np.float64)
        for split in ("train", "validation")
    }
    weights = group["train"][3]
    means = weights @ raw["train"]
    scales = np.sqrt(weights @ (raw["train"] - means) ** 2)
    mask = np.ptp(raw["train"], axis=0) > 0
    scales[~mask] = 1
    assert np.all(scales > 0)
    x = {split: (value - means) / scales for split, value in raw.items()}
    within_mask = np.zeros(len(FEATURE_NAMES), dtype=bool)
    for index in range(len(group["train"][0])):
        within_mask |= np.ptp(raw["train"][group["train"][1] == index], axis=0) > 0
    masks = {
        "length": mask & within_mask & (np.arange(len(mask)) < 2),
        "sequence": mask & within_mask,
        "nonlinear": mask,
    }
    assert all(np.any(m) for m in masks.values())
    np.savez(
        OUTPUT / "transforms.npz",
        means=means,
        scales=scales,
        **masks,
        training_min=raw["train"].min(axis=0),
        training_max=raw["train"].max(axis=0),
    )
    write(
        OUTPUT / "features.json",
        {
            "raw_feature_names": FEATURE_NAMES,
            "allocated_features": {
                k: [FEATURE_NAMES[i] for i in np.flatnonzero(v)]
                for k, v in masks.items()
            },
            "extrapolation_counts": {
                split: int(
                    np.any(
                        (value < raw["train"].min(axis=0))
                        | (value > raw["train"].max(axis=0)),
                        axis=1,
                    ).sum()
                )
                for split, value in raw.items()
            },
        },
    )
    predictions = {
        split: {"zero": np.zeros(len(value))} for split, value in rows.items()
    }
    records = {
        "zero": {
            split: metrics(target[split], predictions[split]["zero"], group[split][3])
            for split in target
        }
    }
    chosen = {}
    ridge_checks = []
    for family in ("length", "sequence"):
        transformed = {
            split: center_numpy(
                value[:, masks[family]], group[split][1], group[split][2]
            )
            for split, value in x.items()
        }
        design = transformed["train"]
        dimension = design.shape[1]
        for alpha in ALPHAS:
            name = f"{family}_{alpha:g}"
            matrix = design.T @ (
                weights[:, None] * design
            ) + alpha / dimension * np.eye(dimension)
            rhs = design.T @ (weights * target["train"])
            beta = np.linalg.solve(matrix, rhs)
            residual = float(
                np.linalg.norm(matrix @ beta - rhs) / max(np.linalg.norm(rhs), 1e-12)
            )
            assert residual < 1e-9
            np.savez(
                OUTPUT / (name + ".npz"), beta=beta, mask=masks[family], alpha=alpha
            )
            for split, value in transformed.items():
                predictions[split][name] = value @ beta
            records[name] = {
                split: metrics(target[split], predictions[split][name], group[split][3])
                for split in target
            }
            ridge_checks.append(
                {
                    "model": name,
                    "normal_equation_relative_residual": residual,
                    "n_allocated_coefficients": dimension,
                    "n_nonzero_coefficients": int(np.count_nonzero(beta)),
                }
            )
        chosen[family] = min(
            (f"{family}_{a:g}" for a in ALPHAS),
            key=lambda name: (
                records[name]["validation"]["mse"],
                float(name.split("_")[1]),
            ),
        )
    tensor_x = {
        split: torch.tensor(value[:, masks["nonlinear"]], dtype=torch.float64)
        for split, value in x.items()
    }
    tensor_groups = {
        split: (torch.tensor(info[1]), torch.tensor(info[2]))
        for split, info in group.items()
    }
    yt = torch.tensor(target["train"])
    wt = torch.tensor(weights)
    nonlinear = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        network = Observer(int(masks["nonlinear"].sum()))
        optimizer = torch.optim.Adam(
            network.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8
        )
        initial = {name: p.detach().clone() for name, p in network.named_parameters()}
        nonzero = {
            name: torch.zeros_like(p, dtype=torch.bool)
            for name, p in network.named_parameters()
        }
        directory = OUTPUT / f"nonlinear_{seed}"
        directory.mkdir()
        history = []
        best = None
        for step in range(401):
            assert time.process_time() - cpu_started <= 1800, (
                "Prospective fitting CPU budget exceeded"
            )
            if step % 25 == 0:
                with torch.no_grad():
                    values = {
                        split: center_torch(
                            network(tensor_x[split]), *tensor_groups[split]
                        ).numpy()
                        for split in ("train", "validation")
                    }
                observed = {
                    split: metrics(target[split], values[split], group[split][3])
                    for split in target
                }
                state = {
                    name: value.detach().numpy().copy()
                    for name, value in network.state_dict().items()
                }
                np.savez(directory / f"step-{step:03}.npz", **state)
                torch.save(
                    {
                        "model": network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": torch.get_rng_state(),
                        "step": step,
                    },
                    directory / f"step-{step:03}-resume.pt",
                )
                if best is None or (observed["validation"]["mse"], step) < (
                    best["metrics"]["validation"]["mse"],
                    best["step"],
                ):
                    best = {"step": step, "metrics": observed, "state": state}
                history.append({"step": step, "evaluation": observed})
            if step == 400:
                break
            optimizer.zero_grad(set_to_none=True)
            prediction = center_torch(
                network(tensor_x["train"]), *tensor_groups["train"]
            )
            mse = (wt * (prediction - yt).square()).sum()
            penalty = 0.001 * parameter_penalty(network)
            loss = mse + penalty
            assert torch.isfinite(loss)
            loss.backward()
            for name, param in network.named_parameters():
                assert param.grad is not None and torch.isfinite(param.grad).all()
                nonzero[name] |= param.grad != 0
            optimizer.step()
            history.append(
                {
                    "update": step + 1,
                    "online_mse": mse.item(),
                    "penalty": penalty.item(),
                    "loss": loss.item(),
                }
            )
        name = f"nonlinear_{seed}"
        network.load_state_dict(
            {key: torch.tensor(value) for key, value in best["state"].items()}
        )
        for split in rows:
            with torch.no_grad():
                predictions[split][name] = (
                    center_torch(network(tensor_x[split]), *tensor_groups[split])
                    .numpy()
                    .copy()
                )
        np.savez(OUTPUT / (name + ".npz"), **best["state"])
        coverage = {
            key: {
                "n_elements": param.numel(),
                "n_nonzero_gradient_elements": int(nonzero[key].sum()),
                "n_selected_changed_elements": int(
                    (param.detach() != initial[key]).sum()
                ),
            }
            for key, param in network.named_parameters()
        }
        write(directory / "history.json", history)
        write(directory / "coverage.json", coverage)
        records[name] = best["metrics"]
        nonlinear.append(
            {
                "model": name,
                "seed": seed,
                "selected_step": best["step"],
                "coverage": coverage,
            }
        )
        print(
            json.dumps(
                {
                    "model": name,
                    "selected_step": best["step"],
                    "validation_rmse": best["metrics"]["validation"]["rmse"],
                }
            ),
            flush=True,
        )
    for split, split_rows in rows.items():
        predictions[split]["nonlinear_ensemble"] = np.mean(
            [predictions[split][f"nonlinear_{seed}"] for seed in (42, 43, 44)], axis=0
        )
        np.savez(
            OUTPUT / (split + "-predictions.npz"),
            **predictions[split],
            measurement_ids=np.array([r["measurement_id"] for r in split_rows]),
        )
    records["nonlinear_ensemble"] = {
        split: metrics(
            target[split], predictions[split]["nonlinear_ensemble"], group[split][3]
        )
        for split in target
    }
    write(
        OUTPUT / "validation-selection.json",
        {
            "primary_length": chosen["length"],
            "primary_sequence": chosen["sequence"],
            "nonlinear": nonlinear,
            "metrics": records,
            "ridge_checks": ridge_checks,
        },
    )
    usage = {
        "wall_seconds": time.perf_counter() - wall_started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python": sys.version,
        "argv": sys.argv,
        "environment": {
            key: os.environ[key]
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if key in os.environ
        },
    }
    write(OUTPUT / "runtime.json", usage)
    assert usage["peak_rss_gib"] <= 3
    assert usage["cpu_seconds"] <= 1800
    # The only outcome files opened by this program are train and validation.
    seal = {
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "test_outcomes_loaded": False,
        "source_verification_sha256": digest(ROOT / "results/source-verification.json"),
        "expected_test_target_digest": admission["split_files"]["test-targets.jsonl"],
        "input_digests": {
            name: admission["split_files"][name]
            for name in ("train.jsonl", "validation.jsonl", "test-covariates.jsonl")
        },
        "files": {
            str(p.relative_to(OUTPUT)): digest(p)
            for p in sorted(OUTPUT.rglob("*"))
            if p.is_file()
        },
        "code": {
            name: digest(Path(__file__).parent / name)
            for name in ("fit.py", "model.py")
        },
    }
    write(ROOT / "results/pre-test-seal.json", seal)
    print(
        json.dumps(
            {
                "status": "sealed_before_test_outcome_access",
                "chosen": chosen,
                "runtime": usage,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
