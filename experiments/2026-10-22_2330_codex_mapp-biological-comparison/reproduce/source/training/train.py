"""Frozen local exact/REINFORCE comparison on shared biological recipe outputs."""

import argparse
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
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from hitlist.evidence_recipes import verify_training_recipe
from learning import (
    estimator_check,
    metrics,
    observed_log_probs,
    reinforce_loss,
    reverse_kl,
)
from presto.models.peptide_availability import PeptideAvailabilityPolicy

FAMILY = Path(__file__).resolve().parents[1]
PRESTO = FAMILY.parents[1]
COHORT = (
    PRESTO / "data/availability_recipes/2026-10-22-mapp-captured-composition-min2-v1"
)
ARTIFACTS = PRESTO / "artifacts" / FAMILY.name
SEEDS = (42, 43, 44)
VARIANTS = ("exact-context", "reinforce64-context", "exact-no-context")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def evaluate(model, examples, parents):
    predictions = []
    with torch.no_grad():
        for example in examples:
            if model is None:
                logq = torch.full(
                    (len(example["candidates"]),),
                    -np.log(len(example["candidates"])),
                    dtype=torch.float64,
                )
                mass = None
            else:
                logq, logmass, _ = observed_log_probs(
                    model, parents[example["parent_sequence_sha256"]], example
                )
                mass = float(logmass)
            predictions.append(
                {
                    "example_id": example["example_id"],
                    "parent_sequence_sha256": example["parent_sequence_sha256"],
                    "component_id": example["component_id"],
                    "split": example["split"],
                    "condition": example["condition"],
                    "peptide_ids": [c["peptide_row_id"] for c in example["candidates"]],
                    "logq": logq.tolist(),
                    "logt": [
                        c["target_log_probability"] for c in example["candidates"]
                    ],
                    "source_log10_intensities": [
                        c["source_log10_intensity"] for c in example["candidates"]
                    ],
                    "unidentified_selected_log_mass": mass,
                }
            )
    return predictions


def score(predictions):
    result = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in predictions if r["split"] == split]
        result[split] = metrics(subset)
        for label, predicate in [
            ("two", lambda n: n == 2),
            ("at_least_three", lambda n: n >= 3),
        ]:
            result[split + "_" + label] = metrics(
                [r for r in subset if predicate(len(r["peptide_ids"]))]
            )
    return result


def save_predictions(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def gradient_probe(model, example, parents):
    logq, _, _ = observed_log_probs(
        model, parents[example["parent_sequence_sha256"]], example
    )
    logt = torch.tensor(
        [r["target_log_probability"] for r in example["candidates"]], dtype=logq.dtype
    )
    parameters = list(model.parameters())
    exact = torch.cat(
        [
            g.flatten()
            for g in torch.autograd.grad(
                reverse_kl(logq, logt), parameters, retain_graph=True
            )
        ]
    )
    generator = torch.Generator().manual_seed(9182)
    estimates = []
    for _ in range(128):
        loss, _ = reinforce_loss(logq, logt, count=64, generator=generator)
        estimates.append(
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(loss, parameters, retain_graph=True)
                ]
            )
        )
    values = torch.stack(estimates)
    mean = values.mean(0)
    return {
        "example_id": example["example_id"],
        "n_estimates": 128,
        "draws_per_estimate": 64,
        "relative_l2_error": float((mean - exact).norm() / exact.norm()),
        "mean_cosine": float(torch.nn.functional.cosine_similarity(mean, exact, dim=0)),
        "sum_parameter_variances": float(values.var(0, unbiased=False).sum()),
        "exact_gradient_norm": float(exact.norm()),
    }


def fit(variant, seed, examples, parents):
    started = time.monotonic()
    directory = ARTIFACTS / f"{variant}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = PeptideAvailabilityPolicy(
        min_length=1,
        max_length=40,
        context_size=int(variant != "exact-no-context"),
        hidden_size=32,
    )
    initial = copy.deepcopy(model.state_dict())
    torch.save(
        {"state_dict": initial, "variant": variant, "seed": seed},
        directory / "initial.pt",
    )
    if variant == "exact-context" and seed == 42:
        dump(
            FAMILY / "results/gradient-probe.json",
            gradient_probe(
                model, next(r for r in examples if r["split"] == "train"), parents
            ),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    train = [r for r in examples if r["split"] == "train"]
    val = [r for r in examples if r["split"] == "validation"]
    shuffle = torch.Generator().manual_seed(10000 + seed)
    actions = torch.Generator().manual_seed(20000 + seed)
    coverage = {
        name: torch.zeros_like(p, dtype=torch.bool)
        for name, p in model.named_parameters()
    }
    history = []
    best = float("inf")
    selected = None
    selected_coverage = None
    selected_epoch = None
    work = Counter()
    updates = 0
    for epoch in range(1, 41):
        order = torch.randperm(len(train), generator=shuffle).tolist()
        model.train()
        for offset in range(0, len(order), 8):
            optimizer.zero_grad()
            batch = order[offset : offset + 8]
            loss = 0
            for index in batch:
                example = train[index]
                logq, _, scored = observed_log_probs(
                    model, parents[example["parent_sequence_sha256"]], example
                )
                logt = torch.tensor(
                    [r["target_log_probability"] for r in example["candidates"]],
                    dtype=logq.dtype,
                )
                work.update(scored)
                if variant == "reinforce64-context":
                    value, draws = reinforce_loss(
                        logq, logt, count=64, generator=actions
                    )
                    work["policy_draws"] += len(draws)
                    work["target_terms"] += len(draws)
                else:
                    value = reverse_kl(logq, logt)
                    work["target_terms"] += len(example["candidates"])
                loss = loss + value / len(batch)
            assert torch.isfinite(loss)
            loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                coverage[name] |= p.grad != 0
            optimizer.step()
            updates += 1
            assert all(torch.isfinite(p).all() for p in model.parameters())
        model.eval()
        validation = metrics(evaluate(model, val, parents))
        value = validation["reverse_kl"]
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "validation": validation,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if value < best:
            best = value
            selected = copy.deepcopy(model.state_dict())
            selected_epoch = epoch
            selected_coverage = {name: mask.clone() for name, mask in coverage.items()}
        if epoch % 10 == 0:
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "seed": seed,
                        "epoch": epoch,
                        "validation_kl": value,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        if time.monotonic() - started > 1800:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                directory / "runtime-stop.pt",
            )
            dump(directory / "history.json", history)
            raise RuntimeError(
                "Prespecified 30-minute fit limit reached; replan before continuing"
            )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 40,
            "shuffle_rng": shuffle.get_state(),
            "action_rng": actions.get_state(),
        },
        directory / "last.pt",
    )
    torch.save(
        {
            "state_dict": selected,
            "variant": variant,
            "seed": seed,
            "selected_epoch": selected_epoch,
        },
        directory / "selected.pt",
    )
    coverage_report = {
        name: {
            "n_elements": p.numel(),
            "n_nonzero_gradient_elements": int(coverage[name].sum()),
            "n_changed_elements": int((p.detach() != initial[name]).sum()),
            "has_nonzero_gradient": bool(coverage[name].any()),
            "has_changed": bool((p.detach() != initial[name]).any()),
        }
        for name, p in model.named_parameters()
    }
    assert all(
        r["has_nonzero_gradient"] and r["has_changed"] for r in coverage_report.values()
    )
    model.load_state_dict(selected)
    model.eval()
    for name, p in model.named_parameters():
        coverage_report[name]["selected_n_nonzero_gradient_elements"] = int(
            selected_coverage[name].sum()
        )
        coverage_report[name]["selected_n_changed_elements"] = int(
            (p.detach() != initial[name]).sum()
        )
        assert selected_coverage[name].any() and (p.detach() != initial[name]).any()
    torch.save(
        {"final": coverage, "selected": selected_coverage},
        directory / "gradient-coverage.pt",
    )
    predictions = evaluate(model, examples, parents)
    result = {
        "variant": variant,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "optimizer_updates": updates,
        "seconds": time.monotonic() - started,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "coverage": coverage_report,
        "training_work": dict(work),
        "metrics": score(predictions),
    }
    save_predictions(directory / "predictions.jsonl.gz", predictions)
    dump(directory / "history.json", history)
    dump(directory / "summary.json", result)
    print(
        json.dumps(
            {
                "finished": str(directory),
                "seconds": result["seconds"],
                "selected_epoch": selected_epoch,
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check = estimator_check()
    if args.check_only:
        dump(FAMILY / "results/estimator-check.json", check)
        print(json.dumps(check))
        return
    assert not ARTIFACTS.exists()
    import presto.models.peptide_availability as policy_module

    assert (
        Path(policy_module.__file__).resolve()
        == (PRESTO / "models/peptide_availability.py").resolve()
    )
    assert json.loads((FAMILY / "results/admission-min2.json").read_text())[
        "minimum_split_support_gate"
    ]
    assert (
        json.loads((FAMILY / "results/admission-verification-min2.json").read_text())[
            "status"
        ]
        == "passed"
    )
    assert (
        json.loads((FAMILY / "results/nested-admission-verification.json").read_text())[
            "status"
        ]
        == "passed"
    )
    manifest = verify_training_recipe(COHORT)
    receipt = json.loads((FAMILY / "results/admission-min2.json").read_text())
    assert sha(COHORT / "manifest.json") == receipt["recipe_manifest_sha256"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parents = {
        r["parent_sequence_sha256"]: r["sequence"]
        for r in jsonl(COHORT / "parents.jsonl")
    }
    examples = jsonl(COHORT / "examples.jsonl")
    assert all(
        len({r["parent_sequence_sha256"] for r in examples if r["split"] == split})
        >= 10
        for split in ("train", "validation", "test")
    )
    assert all(set(seq) <= set("ACDEFGHIKLMNPQRSTVWY") for seq in parents.values())
    source = FAMILY / "reproduce/source/training"
    source.mkdir()
    for name in ("train.py", "learning.py"):
        shutil.copy2(FAMILY / "code" / name, source / name)
    shutil.copy2(
        PRESTO / "models/peptide_availability.py", source / "peptide_availability.py"
    )
    bundle = {
        "agent_model": "Codex / GPT-6",
        "argv": sys.argv,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "CPU",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PRESTO, text=True
        ).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PRESTO, text=True
        ),
        "source_hashes": {p.name: sha(p) for p in source.iterdir()},
        "recipe_manifest_sha256": sha(COHORT / "manifest.json"),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "seeds": SEEDS,
        "variants": VARIANTS,
        "epochs": 40,
        "learning_rate": 0.003,
        "optimizer": "Adam",
        "batch_size": 8,
        "pretraining": None,
        "loss": "equal protein-condition reverse KL on admitted observed candidates",
        "environment_overrides": {
            k: os.environ[k]
            for k in (
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            )
            if k in os.environ
        },
        "torch_intraop_threads": 4,
        "torch_interop_threads": 1,
        "output": str(ARTIFACTS),
    }
    dump(FAMILY / "reproduce/training.json", bundle)
    ARTIFACTS.mkdir(parents=True)
    uniform = evaluate(None, examples, parents)
    save_predictions(ARTIFACTS / "uniform-predictions.jsonl.gz", uniform)
    dump(FAMILY / "results/uniform-metrics.json", score(uniform))
    tick = time.monotonic()
    results = []
    for seed in SEEDS:
        for variant in VARIANTS:
            results.append(fit(variant, seed, examples, parents))
    dump(
        FAMILY / "results/training-summary.json",
        {
            "runs": results,
            "seconds": time.monotonic() - tick,
            "peak_process_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
    )


if __name__ == "__main__":
    main()
