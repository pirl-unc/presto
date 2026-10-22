"""Controlled GRPO ingredient ablations on the frozen integer-span task."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch


BASE_PATH = Path(__file__).with_name("base.py")
SPEC = importlib.util.spec_from_file_location("span_reference", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
MASK = torch.tensor(base.MASK)
INDEX = torch.full(MASK.shape, -1, dtype=torch.long)
INDEX[MASK] = torch.arange(int(MASK.sum()))


def flat(output):
    return output[0][:, MASK]


def objective(logq, logp, logref, beta):
    return (logq.exp() * (logq - logp + beta * (logq - logref))).sum(-1).mean()


def optimum(logp, logref, beta):
    return ((logp + beta * logref) / (1 + beta)).log_softmax(-1)


def center(cost, standardize=False):
    if cost.shape[-1] < 2:
        raise ValueError("leave-one-out centering needs at least two samples")
    baseline = (cost.sum(-1, keepdim=True) - cost) / (cost.shape[-1] - 1)
    result = cost - baseline
    if standardize:
        result = result / (cost.std(-1, keepdim=True, unbiased=False) + 1e-6)
    return result.detach()


def importance_surrogate(current, old, target, reference, beta, clip, standardize):
    cost = current - target + beta * (current - reference)
    advantage = center(cost, standardize)
    ratio = (current - old).exp()
    raw = ratio * advantage
    clipped = ratio.clamp(0.8, 1.2) * advantage
    loss = torch.maximum(raw, clipped).mean() if clip else raw.mean()
    with torch.no_grad():
        diagnostics = {
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "ratio_ess_fraction": float(
                (ratio.sum(-1).square() / ratio.square().sum(-1) / ratio.shape[-1]).mean()
            ),
            "outside_clip_fraction": float(((ratio < 0.8) | (ratio > 1.2)).float().mean()),
            "active_clip_fraction": float((clipped > raw + 1e-10).float().mean()) if clip else 0.0,
            "cost_sd": float(cost.std(-1, unbiased=False).mean()),
            "near_zero_sd_fraction": float((cost.std(-1, unbiased=False) < 1e-6).float().mean()),
        }
    return loss, diagnostics


def collect(output, target, reference, count, generator):
    starts, offsets, _ = base.sample_spans(*output, count, generator)
    indices = INDEX[starts, offsets]
    return {
        "indices": indices,
        "old": flat(output).gather(1, indices).detach(),
        "target": target.gather(1, indices).detach(),
        "reference": reference.gather(1, indices).detach(),
    }


def reused_loss(output, cache, config):
    current = flat(output).gather(1, cache["indices"])
    return importance_surrogate(
        current,
        cache["old"],
        cache["target"],
        cache["reference"],
        config["beta"],
        config["clip"],
        config["standardize"],
    )


def hybrid_surrogate(joint, conditional, target, sampled_starts, mask):
    batch = torch.arange(len(joint))[:, None].expand_as(sampled_starts)
    valid = mask[sampled_starts]
    selected_joint = joint[batch, sampled_starts].masked_fill(~valid, 0)
    selected_target = target[batch, sampled_starts].masked_fill(~valid, 0)
    conditional_probability = conditional[batch, sampled_starts].exp().detach()
    cost = selected_joint - selected_target
    expected_cost = (conditional_probability * cost).sum(-1)
    baseline = (expected_cost.sum(-1, keepdim=True) - expected_cost) / (
        sampled_starts.shape[-1] - 1
    )
    advantage = (cost - baseline[..., None]).detach()
    loss = (conditional_probability * advantage * selected_joint).sum(-1).mean()
    return loss


def hybrid_loss(output, target, count, generator):
    if count < 2:
        raise ValueError("hybrid baseline needs at least two starts")
    starts = torch.multinomial(output[1].exp(), count, replacement=True, generator=generator)
    loss = hybrid_surrogate(output[0], output[2], target, starts, MASK)
    return loss, int(MASK[starts].sum())


def conditions():
    configs = []
    for clip, std, reuse, beta in itertools.product((False, True), (False, True), (1, 4), (0, 0.1)):
        configs.append(
            {
                "name": f"factorial-c{int(clip)}-s{int(std)}-r{reuse}-b{beta:g}",
                "method": "factorial",
                "clip": clip,
                "standardize": std,
                "reuse": reuse,
                "beta": beta,
                "samples": 8,
                "steps": 1440,
                "budget": "optimizer_matched",
            }
        )
    for beta in (0, 0.1):
        configs.append({"name": f"exact-b{beta:g}", "method": "exact", "beta": beta, "reuse": 1})
    configs.append({"name": "original-reinforce8", "method": "original", "reuse": 1, "beta": 0})
    for samples in (2, 8):
        configs.append(
            {
                "name": f"hybrid-starts{samples}",
                "method": "hybrid",
                "reuse": 1,
                "beta": 0,
                "samples": samples,
            }
        )
    for clip in (False, True):
        configs.append(
            {
                "name": f"query-matched-c{int(clip)}-r4",
                "method": "factorial",
                "clip": clip,
                "standardize": False,
                "reuse": 4,
                "beta": 0,
                "samples": 8,
                "steps": 5760,
                "budget": "target_query_matched",
            }
        )
    for config in configs:
        config.setdefault("steps", 1440)
        config.setdefault("budget", "optimizer_matched")
        config.setdefault("samples", 8)
    assert len(configs) == 23
    return configs


def probabilities(model, rows):
    result = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(rows), 64):
            x, starts, _ = base.tensors(rows[first : first + 64])
            result.append(flat(model(x, starts)).exp().numpy())
    return np.concatenate(result)


def full_metrics(rows, prediction, reference, beta):
    records = [
        {**base.public_row(row), "prediction": q.tolist()} for row, q in zip(rows, prediction)
    ]
    original = base.metrics(records)
    logp = np.log(np.asarray([row["target"] for row in rows]))
    logref = np.log(reference)
    optlog = (logp + beta * logref) / (1 + beta)
    optlog -= np.logaddexp.reduce(optlog, axis=1)[:, None]
    adjusted = [dict(record, target=p.tolist()) for record, p in zip(records, np.exp(optlog))]
    fitted = base.metrics(adjusted)
    q = np.maximum(prediction, 1e-30)
    kl_reference = float((q * (np.log(q) - logref)).sum(-1).mean())
    return {
        "original": original,
        "optimum": fitted,
        "kl_reference": kl_reference,
        "objective": original["reverse_kl"] + beta * kl_reference,
    }


def light_metrics(model, tensors, reference, beta):
    x, starts, target = tensors
    model.eval()
    with torch.no_grad():
        logq = flat(model(x, starts))
        logp = target[:, MASK]
        logref = reference.log()
        opt = optimum(logp, logref, beta)
        return {
            "objective": float(objective(logq, logp, logref, beta)),
            "tv_optimum": float((logq.exp() - opt.exp()).abs().sum(-1).mean() / 2),
        }


def gradient_summary(estimates, exact):
    estimates = torch.stack(estimates)
    mean = estimates.mean(0)
    return {
        "reference_norm": float(exact.norm()),
        "mean_relative_error": float((mean - exact).norm() / exact.norm()),
        "single_relative_error": float(((estimates - exact).norm(dim=1) / exact.norm()).mean()),
        "mean_cosine": float(torch.nn.functional.cosine_similarity(mean, exact, dim=0)),
        "single_cosine": float(
            torch.nn.functional.cosine_similarity(estimates, exact[None]).mean()
        ),
        "sum_parameter_variance": float(estimates.var(0, unbiased=True).sum()),
    }


def probe(rows):
    torch.manual_seed(42)
    sampling = base.Policy()
    x, starts, target = base.tensors(rows[:4])
    old_output = sampling(x, starts)
    logp = target[:, MASK]
    logref = flat(old_output).detach()
    reports = []
    for perturbed, beta, clip, std in itertools.product(
        (False, True), (0, 0.1), (False, True), (False, True)
    ):
        model = copy.deepcopy(sampling)
        if perturbed:
            offset = 0
            with torch.no_grad():
                for parameter in model.parameters():
                    ids = torch.arange(parameter.numel(), dtype=parameter.dtype) + offset
                    parameter.add_(0.05 * ids.sin().reshape_as(parameter))
                    offset += parameter.numel()
        objective(flat(model(x, starts)), logp, logref, beta).backward()
        exact = base.gradient_vector(model).detach().clone()
        generator = torch.Generator().manual_seed(90000 + int(perturbed))
        estimates, diagnostics = [], []
        config = {"beta": beta, "clip": clip, "standardize": std}
        for _ in range(128):
            cache = collect(old_output, logp, logref, 8, generator)
            model.zero_grad(set_to_none=True)
            loss, diagnostic = reused_loss(model(x, starts), cache, config)
            loss.backward()
            estimates.append(base.gradient_vector(model).detach().clone())
            diagnostics.append(diagnostic)
        reports.append(
            {
                "kind": "factorial",
                "perturbed": perturbed,
                **config,
                **gradient_summary(estimates, exact),
                "target_terms": 128 * 4 * 8,
                "diagnostics": {
                    key: float(np.mean([d[key] for d in diagnostics])) for key in diagnostics[0]
                },
            }
        )
    for kind, count in (("original", 8), ("hybrid", 2), ("hybrid", 8)):
        model = copy.deepcopy(sampling)
        objective(flat(model(x, starts)), logp, logref, 0).backward()
        exact = base.gradient_vector(model).detach().clone()
        generator = torch.Generator().manual_seed(90000)
        estimates, queries = [], 0
        for _ in range(128):
            model.zero_grad(set_to_none=True)
            output = model(x, starts)
            if kind == "original":
                loss = base.sampled_loss(output, target, count, generator)
                queries += 4 * count
            else:
                loss, used = hybrid_loss(output, target, count, generator)
                queries += used
            loss.backward()
            estimates.append(base.gradient_vector(model).detach().clone())
        reports.append(
            {
                "kind": kind,
                "samples": count,
                "target_terms": queries,
                **gradient_summary(estimates, exact),
            }
        )
    return {"repeats": 128, "rows": [row["id"] for row in rows[:4]], "conditions": reports}


def train(config, seed, groups, references, output):
    path = output / f"{config['name']}-seed{seed}"
    path.mkdir()
    torch.manual_seed(seed)
    model = base.Policy()
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(10000 + seed)
    generator = torch.Generator().manual_seed(20000 + seed)
    training = base.tensors(groups["train"])
    validation = base.tensors(groups["validation"])
    reference = torch.tensor(references["train"]).log()
    validation_ref = torch.tensor(references["validation"])
    updates = collected = target_terms = dense_scores = 0
    gradient_seen = {name: False for name, _ in model.named_parameters()}
    history, diagnostics = [], []
    best, selected_step = float("inf"), None
    began = time.perf_counter()
    while updates < config["steps"]:
        order = rng.permutation(len(groups["train"]))
        for first in range(0, len(order), 16):
            if updates >= config["steps"]:
                break
            indices = order[first : first + 16]
            x, starts, target = (value[indices] for value in training)
            ref = reference[indices]
            model.train()
            prediction = model(x, starts)
            cache = None
            collected += 1
            if config["method"] == "factorial":
                cache = collect(prediction, target[:, MASK], ref, config["samples"], generator)
                target_terms += len(indices) * config["samples"]
            for reuse_step in range(config["reuse"]):
                optimizer.zero_grad(set_to_none=True)
                if reuse_step:
                    model.train()
                    prediction = model(x, starts)
                dense_scores += len(indices) * int(MASK.sum())
                if config["method"] == "exact":
                    loss = objective(flat(prediction), target[:, MASK], ref, config["beta"])
                    target_terms += len(indices) * int(MASK.sum())
                elif config["method"] == "original":
                    loss = base.sampled_loss(prediction, target, 8, generator)
                    target_terms += len(indices) * 8
                elif config["method"] == "hybrid":
                    loss, used = hybrid_loss(prediction, target, config["samples"], generator)
                    target_terms += used
                else:
                    loss, diagnostic = reused_loss(prediction, cache, config)
                    diagnostics.append({"reuse_step": reuse_step, **diagnostic})
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss: {config['name']}/{seed}/{updates}")
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"missing/nonfinite gradient: {name}")
                    gradient_seen[name] |= bool(torch.any(parameter.grad != 0))
                optimizer.step()
                updates += 1
                if updates % 120 == 0 or updates == config["steps"]:
                    scores = light_metrics(model, validation, validation_ref, config["beta"])
                    history.append({"step": updates, "collected_batches": collected, **scores})
                    if scores["objective"] < best:
                        best, selected_step = scores["objective"], updates
                        best_state = copy.deepcopy(model.state_dict())
    seconds = time.perf_counter() - began
    torch.save(
        {"state": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": updates},
        path / "last-state.pt",
    )
    model.load_state_dict(best_state)
    torch.save(
        {"state": best_state, "config": config, "seed": seed, "step": selected_step},
        path / "selected.pt",
    )
    predictions, metrics = {}, {}
    for split, rows in groups.items():
        predictions[split] = probabilities(model, rows)
        metrics[split] = full_metrics(rows, predictions[split], references[split], config["beta"])
        metrics[split]["contexts"] = {}
        for context in range(4):
            mask = np.asarray([row["context_id"] == context for row in rows])
            subset = [row for row in rows if row["context_id"] == context]
            metrics[split]["contexts"][str(context)] = full_metrics(
                subset, predictions[split][mask], references[split][mask], config["beta"]
            )
    np.savez_compressed(path / "predictions.npz", **predictions)
    np.savez_compressed(path / "replay-training.npz", prediction=predictions["train"][:4])
    base.write_json(path / "history.json", history)
    diagnostic_means = {}
    if diagnostics:
        for reuse_step in range(config["reuse"]):
            subset = [d for d in diagnostics if d["reuse_step"] == reuse_step]
            diagnostic_means[str(reuse_step)] = {
                key: float(np.mean([d[key] for d in subset]))
                for key in subset[0]
                if key != "reuse_step"
            }
    changes = {
        name: float((value.detach() - initial[name]).abs().max())
        for name, value in model.named_parameters()
    }
    report = {
        "config": config,
        "seed": seed,
        "initial_hash": base.state_hash(initial),
        "selected_hash": base.state_hash(best_state),
        "selected_step": selected_step,
        "updates": updates,
        "collected_batches": collected,
        "sampled_rows": collected * 16,
        "fit_target_terms": target_terms,
        "fit_dense_span_scores": dense_scores,
        "seconds_including_validation": seconds,
        "gradient_seen": gradient_seen,
        "parameter_max_change": changes,
        "diagnostics_by_reuse_step": diagnostic_means,
        "metrics": metrics,
        "screen": metrics["validation"]["optimum"]["tv"] <= 0.10
        and metrics["validation"]["optimum"]["noncompatible_mass_mae"] <= 0.05,
    }
    base.write_json(path / "summary.json", report)
    print(
        json.dumps(
            {
                "condition": path.name,
                "selected": selected_step,
                "val_tv": metrics["validation"]["optimum"]["tv"],
                "test_tv": metrics["test"]["original"]["tv"],
                "seconds": seconds,
            }
        ),
        flush=True,
    )
    return report


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    execution = {
        "argv": sys.argv,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True).strip(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "modal_gpu": None,
        "threads": torch.get_num_threads(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "sources": {Path(__file__).name: base.sha(__file__), "base.py": base.sha(BASE_PATH)},
    }
    base.write_json(output / "execution.json", execution)
    (output / "working-tree.diff").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], text=True)
    )
    rows, coefficients = base.make_dataset()
    base.write_rows(output / "dataset.jsonl", [base.public_row(row) for row in rows])
    base.write_json(
        output / "teacher.json", {key: value.tolist() for key, value in coefficients.items()}
    )
    groups = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    base.write_json(output / "gradient-probe.json", probe(groups["train"]))
    references = {}
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        model = base.Policy()
        references[seed] = {split: probabilities(model, values) for split, values in groups.items()}
        np.savez_compressed(output / f"reference-seed{seed}.npz", **references[seed])
    uniform = {
        split: full_metrics(
            values,
            np.full((len(values), int(MASK.sum())), 1 / int(MASK.sum())),
            references[42][split],
            0,
        )["original"]
        for split, values in groups.items()
    }
    results, failures = [], []
    for config in conditions():
        for seed in (42, 43, 44):
            try:
                results.append(train(config, seed, groups, references[seed], output))
            except Exception as error:
                failures.append({"config": config, "seed": seed, "error": repr(error)})
                base.write_json(output / "failures.json", failures)
                print(
                    json.dumps({"failed": config["name"], "seed": seed, "error": repr(error)}),
                    flush=True,
                )
    base.write_json(
        output / "summary.json",
        {
            "conditions": results,
            "uniform": uniform,
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - began,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    base.write_json(
        output / "artifacts.json",
        {
            str(path.relative_to(output)): base.sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    print(json.dumps({"completed": len(results), "failed": len(failures)}), flush=True)
    if failures:
        raise RuntimeError("one or more prespecified conditions failed; artifacts retained")


def replay(output):
    torch.set_num_threads(1)
    rows, _ = base.make_dataset()
    selected = []
    for path in sorted(output.glob("*/selected.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = base.Policy()
        model.load_state_dict(checkpoint["state"])
        actual = probabilities(model, rows[:4])
        expected = np.load(path.parent / "replay-training.npz")["prediction"]
        if not np.array_equal(actual, expected):
            raise AssertionError(path)
        selected.append(path.parent.name)
    base.write_json(
        output.parent / "fresh-process-replay.json",
        {"status": "passed", "checkpoints": selected, "test_rescored": False},
    )
    print(json.dumps({"status": "passed", "checkpoints": len(selected)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "replay"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output) if args.action == "run" else replay(args.output)
