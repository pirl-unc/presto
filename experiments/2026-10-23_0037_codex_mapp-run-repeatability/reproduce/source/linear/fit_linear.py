"""Frozen Stage B: regularized observed-set scores, exact versus REINFORCE."""

import copy
import gzip
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from composition_metrics import metrics
from features import design

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
COHORT = ROOT / "data/availability_recipes/2026-10-22-mapp-captured-composition-min2-v1"
OUTPUT = ROOT / "artifacts" / FAMILY.name / "linear-fits"
LAMBDAS = (0.0, 0.001, 0.01, 0.1, 1.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save_rows(path, values):
    payload = "".join(
        json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in values
    ).encode()
    Path(path).write_bytes(gzip.compress(payload, mtime=0))


def padded(matrices, examples, split):
    indices = [
        i for i, e in enumerate(examples) if split is None or e["split"] == split
    ]
    width, dim = max(len(matrices[i]) for i in indices), matrices[0].shape[1]
    x = torch.zeros(len(indices), width, dim)
    logt = torch.zeros(len(indices), width)
    mask = torch.zeros(len(indices), width, dtype=torch.bool)
    for b, i in enumerate(indices):
        n = len(matrices[i])
        x[b, :n] = torch.tensor(matrices[i], dtype=torch.float32)
        logt[b, :n] = torch.tensor(
            [c["target_log_probability"] for c in examples[i]["candidates"]]
        )
        mask[b, :n] = True
    return x, logt, mask, indices


def log_probs(theta, data):
    x, _, mask, _ = data
    return (x @ theta).masked_fill(~mask, -torch.inf).log_softmax(-1)


def exact_kl(logq, data):
    _, logt, mask, _ = data
    diff = torch.where(mask, logq - logt, 0.0)
    return (logq.exp() * diff).sum(-1).mean()


def reinforce(logq, data, generator):
    draw = torch.multinomial(
        logq.detach().exp(), 64, replacement=True, generator=generator
    )
    selected = logq.gather(1, draw)
    cost = (selected - data[1].gather(1, draw)).detach()
    baseline = (cost.sum(-1, keepdim=True) - cost) / 63
    return ((cost - baseline) * selected).mean()


def predict(theta, data, examples):
    with torch.no_grad():
        values = log_probs(theta, data)
    out = []
    for b, i in enumerate(data[3]):
        e = examples[i]
        out.append(
            {
                **{
                    k: e[k]
                    for k in (
                        "example_id",
                        "parent_sequence_sha256",
                        "component_id",
                        "split",
                        "condition",
                    )
                },
                "peptide_ids": [c["peptide_row_id"] for c in e["candidates"]],
                "logq": values[b, : len(e["candidates"])].tolist(),
                "logt": [c["target_log_probability"] for c in e["candidates"]],
                "source_log10_intensities": [
                    c["source_log10_intensity"] for c in e["candidates"]
                ],
            }
        )
    return out


def gradient_check(data):
    x, logt, mask, indices = data
    data = x.double(), logt.double(), mask, indices
    theta = torch.linspace(
        -0.1, 0.1, x.shape[-1], dtype=torch.float64, requires_grad=True
    )
    logq = log_probs(theta, data)
    value = exact_kl(logq, data) + 0.01 * theta.square().sum()
    grad = torch.autograd.grad(value, theta, retain_graph=True)[0]
    q = logq.detach().exp().numpy()
    diff = np.where(mask.numpy(), logq.detach().numpy() - data[1].numpy(), 0.0)
    upstream = q * (diff - (q * diff).sum(-1, keepdims=True))
    independent = (
        np.einsum("bi,bid->d", upstream, data[0].numpy()) / len(indices)
        + 0.02 * theta.detach().numpy()
    )
    analytic_error = float(np.max(np.abs(grad.numpy() - independent)))
    assert analytic_error < 1e-10
    rng = np.random.default_rng(9182)
    differences = []
    for _ in range(5):
        direction = torch.tensor(rng.normal(size=len(theta)), dtype=torch.float64)
        direction /= direction.norm()
        values = []
        for sign in (-1, 1):
            t = theta.detach() + sign * 1e-5 * direction
            values.append(
                float(exact_kl(log_probs(t, data), data) + 0.01 * t.square().sum())
            )
        differences.append(
            abs((values[1] - values[0]) / 2e-5 - float(grad @ direction))
        )
    assert max(differences) < 1e-8
    estimates = []
    generator = torch.Generator().manual_seed(9182)
    for _ in range(128):
        loss = reinforce(logq, data, generator) + 0.01 * theta.square().sum()
        estimates.append(torch.autograd.grad(loss, theta, retain_graph=True)[0].numpy())
    values = np.asarray(estimates)
    np.savez(
        FAMILY / "results/linear-gradient-probe.npz",
        exact=grad.numpy(),
        estimates=values,
    )
    result = {
        "analytic_max_error": analytic_error,
        "finite_difference_max_error": max(differences),
        "sampled_mean_relative_error": float(
            np.linalg.norm(values.mean(0) - grad.numpy()) / np.linalg.norm(grad.numpy())
        ),
        "n_estimates": len(values),
        "draws_per_parent": 64,
        "mean_gradient_se_l2": float(
            np.linalg.norm(values.std(0, ddof=1) / np.sqrt(len(values)))
        ),
    }
    dump(FAMILY / "results/linear-gradient-check.json", result)


def fit(architecture, method, seed, strength, datasets, examples):
    tick = time.monotonic()
    name = f"{architecture}-{method}-seed{seed}-l{strength:g}"
    directory = OUTPUT / name
    directory.mkdir()
    train, validation = datasets["train"], datasets["validation"]
    theta = torch.nn.Parameter(torch.zeros(train[0].shape[-1]))
    optimizer = torch.optim.Adam([theta], lr=0.03)
    generator = torch.Generator().manual_seed(seed)
    coverage = torch.zeros_like(theta, dtype=torch.bool)
    maximum_gradient = torch.zeros_like(theta)
    best, selected_step, selected, selected_coverage = float("inf"), None, None, None
    history = []
    torch.save({"theta": theta.detach().clone()}, directory / "initial.pt")
    for step in range(1001):
        if step % 25 == 0:
            predictions = predict(theta, validation, examples)
            val = metrics(predictions)
            history.append(
                {
                    "step": step,
                    "validation": val,
                    "train_reverse_kl": float(
                        exact_kl(log_probs(theta, train), train).detach()
                    ),
                    "penalty": float((strength * theta.square().sum()).detach()),
                    "seconds": time.monotonic() - tick,
                }
            )
            if val["reverse_kl"] < best:
                best, selected_step = val["reverse_kl"], step
                selected, selected_coverage = theta.detach().clone(), coverage.clone()
        if step == 1000:
            break
        optimizer.zero_grad()
        logq = log_probs(theta, train)
        loss = (
            exact_kl(logq, train)
            if method == "exact"
            else reinforce(logq, train, generator)
        )
        loss = loss + strength * theta.square().sum()
        assert torch.isfinite(loss)
        loss.backward()
        assert theta.grad is not None and torch.isfinite(theta.grad).all()
        coverage |= theta.grad != 0
        maximum_gradient = torch.maximum(maximum_gradient, theta.grad.abs())
        optimizer.step()
        assert torch.isfinite(theta).all()
    torch.save(
        {
            "theta": theta.detach().clone(),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "generator": generator.get_state(),
            "coverage": coverage,
            "max_abs_gradient": maximum_gradient,
        },
        directory / "last.pt",
    )
    torch.save(
        {"theta": selected, "step": selected_step, "coverage": selected_coverage},
        directory / "selected.pt",
    )
    save_rows(
        directory / "validation-predictions.jsonl.gz",
        predict(selected, validation, examples),
    )
    dump(directory / "history.json", history)
    result = {
        "name": name,
        "architecture": architecture,
        "method": method,
        "seed": seed,
        "l2": strength,
        "updates": 1000,
        "selected_step": selected_step,
        "validation_reverse_kl": best,
        "n_parameters": len(theta),
        "n_nonzero_gradient": int(coverage.sum()),
        "n_final_changed": int((theta.detach() != 0).sum()),
        "n_selected_changed": int((selected != 0).sum()),
        "n_selected_nonzero_gradient": int(selected_coverage.sum()),
        "scored_valid_candidates": int(train[2].sum()) * 1000,
        "scored_padded_candidates": train[2].numel() * 1000,
        "target_terms": (
            len(train[3]) * 64 if method == "reinforce64" else int(train[2].sum())
        )
        * 1000,
        "seconds": time.monotonic() - tick,
    }
    assert (
        result["n_nonzero_gradient"]
        == result["n_final_changed"]
        == result["n_parameters"]
    )
    dump(directory / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    assert not OUTPUT.exists()
    assert read(FAMILY / "results/independent-analysis.json")["passed"]
    assert (
        sha(COHORT / "manifest.json")
        == "df78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80"
    )
    for name, value in read(COHORT / "manifest.json")["files"].items():
        assert sha(COHORT / name) == value["sha256"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    examples = rows(COHORT / "examples.jsonl")
    parents = {
        r["parent_sequence_sha256"]: r["sequence"]
        for r in rows(COHORT / "parents.jsonl")
    }
    snapshot = FAMILY / "reproduce/source/linear"
    snapshot.mkdir()
    for name in ("features.py", "fit_linear.py", "composition_metrics.py"):
        shutil.copy2(FAMILY / "code" / name, snapshot / name)
    receipt = {
        "argv": sys.argv,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "source_hashes": {p.name: sha(p) for p in snapshot.iterdir()},
        "stage_b_plan_sha256": sha(FAMILY / "reproduce/stage-b-plan.md"),
        "recipe_manifest_sha256": sha(COHORT / "manifest.json"),
        "platform": platform.platform(),
        "device": "CPU",
        "threads": 1,
        "optimizer": "Adam",
        "learning_rate": 0.03,
        "updates": 1000,
        "lambdas": LAMBDAS,
        "pretraining": None,
        "environment_overrides": {
            k: os.environ[k]
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            )
            if k in os.environ
        },
    }
    dump(FAMILY / "reproduce/linear-training.json", receipt)
    OUTPUT.mkdir()
    datasets, contracts = {}, {}
    for architecture in ("length", "sequence", "interaction"):
        matrices, contract = design(examples, parents, architecture)
        contracts[architecture] = contract
        datasets[architecture] = {
            s: padded(matrices, examples, s)
            for s in ("train", "validation", "test", None)
        }
    dump(FAMILY / "results/feature-contracts.json", contracts)
    gradient_check(datasets["interaction"]["train"])
    tick = time.monotonic()
    results = []
    for architecture in ("length", "sequence", "interaction"):
        conditions = [("exact", 0)] + (
            [("reinforce64", seed) for seed in (42, 43, 44)]
            if architecture == "interaction"
            else []
        )
        for method, seed in conditions:
            for strength in LAMBDAS:
                results.append(
                    fit(
                        architecture,
                        method,
                        seed,
                        strength,
                        datasets[architecture],
                        examples,
                    )
                )
                dump(FAMILY / "results/linear-progress.json", results)
                if time.monotonic() - tick > 900:
                    raise RuntimeError(
                        "Prespecified 15-minute fitting limit exceeded; replan"
                    )
    chosen = []
    for architecture, method, seed in dict.fromkeys(
        (r["architecture"], r["method"], r["seed"]) for r in results
    ):
        candidates = [
            r
            for r in results
            if (r["architecture"], r["method"], r["seed"])
            == (architecture, method, seed)
        ]
        winner = min(
            candidates,
            key=lambda r: (r["validation_reverse_kl"], -r["l2"], r["selected_step"]),
        )
        state = torch.load(OUTPUT / winner["name"] / "selected.pt", weights_only=True)
        predictions = predict(state["theta"], datasets[architecture][None], examples)
        save_rows(
            OUTPUT / winner["name"] / "all-split-predictions.jsonl.gz", predictions
        )
        chosen.append(
            {
                **winner,
                "metrics": {
                    split: metrics([r for r in predictions if r["split"] == split])
                    for split in ("train", "validation", "test")
                },
            }
        )
    dump(
        FAMILY / "results/linear-summary.json",
        {
            "runs": results,
            "selected": chosen,
            "seconds": time.monotonic() - tick,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "best_exact_architecture_by_validation": min(
                (r for r in chosen if r["method"] == "exact"),
                key=lambda r: r["validation_reverse_kl"],
            )["architecture"],
        },
    )


if __name__ == "__main__":
    main()
