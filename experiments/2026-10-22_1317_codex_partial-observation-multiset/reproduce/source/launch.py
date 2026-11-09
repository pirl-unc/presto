"""Finite biased observations and whole-multiset losses on generic integer spans."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
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
SPEC = importlib.util.spec_from_file_location("observation_base", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
MASK = torch.tensor(base.MASK)


@dataclass(frozen=True)
class Inputs:
    """The complete fitting/selection interface; no teacher probabilities."""

    features: torch.Tensor
    starts: torch.Tensor
    weights: torch.Tensor
    counts: torch.Tensor

    def subset(self, indices):
        return Inputs(*(value[indices] for value in self.values()))

    def values(self):
        return self.features, self.starts, self.weights, self.counts

    def __len__(self):
        return len(self.features)


def make_observations(rows):
    observations = {}
    for mechanism, seed in (("positive", 20261023), ("blind", 20261024)):
        rng = np.random.default_rng(seed)
        records = []
        for row in rows:
            last = np.asarray([row["tokens"][s + o] for s, o in zip(*np.where(base.MASK))])
            weight = 0.05 + 0.95 * (last / 7) ** 3
            if mechanism == "blind":
                weight *= row["compatible"]
            observed = np.asarray(row["target"]) * weight
            assert observed.sum() > 0
            observed /= observed.sum()
            draws = rng.choice(70, size=256, p=observed)
            records.append(
                {
                    "id": row["id"],
                    "weights": weight.tolist(),
                    "draws": draws.tolist(),
                    "counts32": np.bincount(draws[:32], minlength=70).tolist(),
                    "counts256": np.bincount(draws, minlength=70).tolist(),
                }
            )
        observations[mechanism] = records
    return observations


def make_inputs(rows, observations, budget):
    assert [r["id"] for r in rows] == [r["id"] for r in observations]
    return Inputs(
        torch.tensor(np.stack([r["features"] for r in rows])),
        torch.tensor(np.stack([r["starts"] for r in rows])),
        torch.tensor([r["weights"] for r in observations], dtype=torch.float32),
        torch.tensor([r[f"counts{budget}"] for r in observations], dtype=torch.float32),
    )


def observed_logp(log_latent, weights):
    if torch.any(weights.sum(-1) <= 0):
        raise ValueError("each row needs observable support")
    return (log_latent + weights.log()).log_softmax(-1)


def frequencies(counts):
    return counts / counts.sum(-1, keepdim=True)


def exact_cost(logq, frequency, method, count=16):
    if method == "ce":
        safe = logq.masked_fill(frequency == 0, 0)
        return -(frequency * safe).sum(-1)
    cost = (logq.exp() - frequency).square().sum(-1)
    if method == "naive":
        cost = cost + (1 - logq.exp().square().sum(-1)) / count
    return cost


def multiset_cost(counts, frequency, corrected):
    k = counts.sum(-1, keepdim=True)
    if torch.any(k < 2):
        raise ValueError("multiset costs require at least two draws")
    self_term = counts * (counts - 1) / (k * (k - 1)) if corrected else (counts / k).square()
    return (self_term - 2 * counts / k * frequency).sum(-1) + frequency.square().sum(-1)


def multiset_surrogate(logq, frequency, draws, corrected):
    """draws: batch x independent groups x ordered draws (multiplicity retained)."""
    if draws.shape[1] < 2:
        raise ValueError("independent leave-one-out baseline needs at least two groups")
    counts = logq.new_zeros((*draws.shape[:2], logq.shape[-1]))
    counts.scatter_add_(-1, draws, torch.ones_like(draws, dtype=logq.dtype))
    cost = multiset_cost(counts, frequency[:, None], corrected)
    baseline = (cost.sum(-1, keepdim=True) - cost) / (draws.shape[1] - 1)
    selected = logq[:, None].expand_as(counts).gather(-1, draws).sum(-1)
    return ((cost - baseline).detach() * selected).mean()


def sampled_cost(logq, frequency, count, generator, corrected=True):
    draws = torch.multinomial(logq.exp(), 4 * count, replacement=True, generator=generator)
    return multiset_surrogate(logq, frequency, draws.reshape(-1, 4, count), corrected)


def conditions():
    return [
        {
            "name": f"{mechanism}-n{budget}-{method}",
            "mechanism": mechanism,
            "budget": budget,
            "method": method,
            "draws": 64 if method == "multiset64" else 16,
        }
        for mechanism, budget, method in itertools.product(
            ("positive", "blind"), (32, 256), ("ce", "l2", "multiset16", "multiset64", "naive")
        )
    ]


def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        parts = [
            model(batch.features, batch.starts)[0][:, MASK].exp().numpy()
            for first in range(0, len(inputs), 64)
            for batch in [inputs.subset(slice(first, first + 64))]
        ]
    return np.concatenate(parts)


def distribution_metrics(p, q):
    logp, logq = np.log(np.maximum(p, 1e-30)), np.log(np.maximum(q, 1e-30))
    return {
        "tv": float(np.abs(p - q).sum(-1).mean() / 2),
        "forward_kl": float((p * (logp - logq)).sum(-1).mean()),
        "reverse_kl": float((q * (logq - logp)).sum(-1).mean()),
        "rmse_probability": float(np.sqrt(np.mean((p - q) ** 2))),
        "pearson_probability": base.correlation(p.ravel(), q.ravel()),
        "spearman_probability": base.correlation(base.ranks(p.ravel()), base.ranks(q.ravel())),
        "entropy": float(-(q * logq).sum(-1).mean()),
        "target_entropy": float(-(p * logp).sum(-1).mean()),
        "predicted_probability_sd": float(q.std()),
        "target_probability_sd": float(p.std()),
    }


def full_metrics(rows, observations, q, config):
    p = np.asarray([r["target"] for r in rows])
    w = np.asarray([r["weights"] for r in observations])
    counts = np.asarray([r[f"counts{config['budget']}"] for r in observations])
    f = counts / counts.sum(-1, keepdims=True)
    po, qo = p * w, q * w
    po /= po.sum(-1, keepdims=True)
    qo /= qo.sum(-1, keepdims=True)
    ce = float(-(f * np.log(np.maximum(qo, 1e-30))).sum(-1).mean())
    l2 = float(np.square(qo - f).sum(-1).mean())
    expected = ce if config["method"] == "ce" else l2
    if config["method"] == "naive":
        expected += float((1 - np.square(qo).sum(-1)).mean() / config["draws"])
    return {
        "rows": len(rows),
        "observed_count_ce": ce,
        "observed_count_l2": l2,
        "expected_training_cost": expected,
        "latent": distribution_metrics(p, q),
        "observed": distribution_metrics(po, qo),
        "invisible_mass_mae": float(np.abs(((p - q) * (w == 0)).sum(-1)).mean()),
        "invisible_target_mass": float((p * (w == 0)).sum(-1).mean()),
        "invisible_predicted_mass": float((q * (w == 0)).sum(-1).mean()),
    }


def validation_ce(model, inputs):
    model.eval()
    with torch.no_grad():
        logq = observed_logp(model(inputs.features, inputs.starts)[0][:, MASK], inputs.weights)
        return float(exact_cost(logq, frequencies(inputs.counts), "ce").mean())


def fit(config, seed, train_inputs, validation_inputs, path):
    """This function and its selection callback never receive an oracle dataset."""
    path.mkdir()
    torch.manual_seed(seed)
    model = base.Policy()
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(10000 + seed)
    generator = torch.Generator().manual_seed(20000 + seed)
    history, gradient_seen = [], {name: False for name, _ in model.named_parameters()}
    best, steps, draws = float("inf"), 0, 0
    began = time.perf_counter()
    for _ in range(60):
        order = rng.permutation(len(train_inputs))
        for first in range(0, len(order), 16):
            batch = train_inputs.subset(order[first : first + 16])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logq = observed_logp(model(batch.features, batch.starts)[0][:, MASK], batch.weights)
            frequency = frequencies(batch.counts)
            if config["method"] in ("ce", "l2"):
                loss = exact_cost(logq, frequency, config["method"]).mean()
            else:
                loss = sampled_cost(
                    logq,
                    frequency,
                    config["draws"],
                    generator,
                    corrected=config["method"] != "naive",
                )
                draws += len(batch) * 4 * config["draws"]
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite fitting loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"missing/nonfinite gradient: {name}")
                gradient_seen[name] |= bool(torch.any(parameter.grad != 0))
            optimizer.step()
            steps += 1
            if steps % 120 == 0:
                ce = validation_ce(model, validation_inputs)
                history.append({"step": steps, "validation_observed_count_ce": ce})
                if ce < best:
                    best, selected_step = ce, steps
                    best_state = copy.deepcopy(model.state_dict())
    seconds = time.perf_counter() - began
    torch.save(
        {"state": model.state_dict(), "optimizer": optimizer.state_dict(), "step": steps},
        path / "last-state.pt",
    )
    torch.save(
        {"state": best_state, "config": config, "seed": seed, "step": selected_step},
        path / "selected.pt",
    )
    model.load_state_dict(best_state)
    base.write_json(path / "history.json", history)
    return model, {
        "config": config,
        "seed": seed,
        "selected_step": selected_step,
        "updates": steps,
        "seconds_including_validation": seconds,
        "sampled_draws": draws,
        "fit_dense_span_scores": steps * 16 * 70,
        "fit_observed_count_cells_accessed": steps * 16 * 70,
        "initial_hash": base.state_hash(initial),
        "selected_hash": base.state_hash(best_state),
        "gradient_seen": gradient_seen,
        "parameter_max_change": {
            name: float((p.detach() - initial[name]).abs().max())
            for name, p in model.named_parameters()
        },
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base.write_json(
        output / "execution.json",
        {
            "argv": sys.argv,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True).strip(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "modal_gpu": None,
            "threads": 1,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "PYTHONHASHSEED",
                )
            },
            "sources": {"launch.py": base.sha(__file__), "base.py": base.sha(BASE_PATH)},
        },
    )
    (output / "working-tree.diff").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], text=True)
    )
    rows, teacher = base.make_dataset()
    observations = make_observations(rows)
    base.write_rows(output / "oracle-dataset.jsonl", [base.public_row(r) for r in rows])
    base.write_json(
        output / "teacher.json", {key: value.tolist() for key, value in teacher.items()}
    )
    for mechanism, records in observations.items():
        base.write_rows(output / f"observations-{mechanism}.jsonl", records)
    split_masks = {
        split: np.array([r["split"] == split for r in rows])
        for split in ("train", "validation", "test")
    }
    results, failures, controls = [], [], {}
    for config in conditions():
        records = observations[config["mechanism"]]
        inputs = make_inputs(rows, records, config["budget"])
        groups = {split: inputs.subset(mask) for split, mask in split_masks.items()}
        control_key = f"{config['mechanism']}-n{config['budget']}"
        if control_key not in controls:
            controls[control_key] = {}
            for split, mask in split_masks.items():
                subset = [r for r, keep in zip(rows, mask) if keep]
                observed = [r for r, keep in zip(records, mask) if keep]
                controls[control_key][split] = {
                    label: full_metrics(subset, observed, prediction, {**config, "method": "ce"})
                    for label, prediction in (
                        ("uniform", np.full((len(subset), 70), 1 / 70)),
                        ("oracle", np.asarray([r["target"] for r in subset])),
                    )
                }
        for seed in (42, 43, 44):
            path = output / f"{config['name']}-seed{seed}"
            try:
                model, report = fit(config, seed, groups["train"], groups["validation"], path)
                predictions, metrics = {}, {}
                for split, mask in split_masks.items():
                    subset = [r for r, keep in zip(rows, mask) if keep]
                    observed = [r for r, keep in zip(records, mask) if keep]
                    predictions[split] = predict(model, groups[split])
                    metrics[split] = full_metrics(subset, observed, predictions[split], config)
                    metrics[split]["contexts"] = {}
                    for context in range(4):
                        cmask = np.asarray([r["context_id"] == context for r in subset])
                        metrics[split]["contexts"][str(context)] = full_metrics(
                            [r for r, keep in zip(subset, cmask) if keep],
                            [r for r, keep in zip(observed, cmask) if keep],
                            predictions[split][cmask],
                            config,
                        )
                np.savez_compressed(path / "predictions.npz", **predictions)
                np.savez_compressed(
                    path / "replay-training.npz", prediction=predictions["train"][:4]
                )
                report["metrics"] = metrics
                base.write_json(path / "summary.json", report)
                results.append(report)
                print(
                    json.dumps(
                        {
                            "condition": path.name,
                            "step": report["selected_step"],
                            "test_observed_tv": metrics["test"]["observed"]["tv"],
                            "test_latent_tv": metrics["test"]["latent"]["tv"],
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                failures.append({"config": config, "seed": seed, "error": repr(error)})
                base.write_json(output / "failures.json", failures)
                print(json.dumps({"failed": path.name, "error": repr(error)}), flush=True)
    base.write_json(
        output / "summary.json",
        {
            "conditions": results,
            "controls": controls,
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - began,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    base.write_json(
        output / "artifacts.json",
        {str(p.relative_to(output)): base.sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    print(json.dumps({"completed": len(results), "failed": len(failures)}), flush=True)
    if failures:
        raise RuntimeError("prespecified fits failed; failure artifacts preserved")


def replay(output):
    torch.set_num_threads(1)
    rows, _ = base.make_dataset()
    rows = rows[:4]
    checked = []
    for path in sorted(output.glob("*/selected.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = base.Policy()
        model.load_state_dict(checkpoint["state"])
        # Replay only four training rows, not selected held-out evaluation.
        with torch.no_grad():
            actual = (
                model(
                    torch.tensor(np.stack([r["features"] for r in rows])),
                    torch.tensor(np.stack([r["starts"] for r in rows])),
                )[0][:, MASK]
                .exp()
                .numpy()
            )
        expected = np.load(path.parent / "replay-training.npz")["prediction"]
        np.testing.assert_array_equal(actual, expected)
        checked.append(path.parent.name)
    base.write_json(
        output.parent / "fresh-process-replay.json",
        {"status": "passed", "checkpoints": checked, "test_rescored": False},
    )
    print(json.dumps({"status": "passed", "checkpoints": len(checked)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output) if args.action == "run" else replay(args.output)
