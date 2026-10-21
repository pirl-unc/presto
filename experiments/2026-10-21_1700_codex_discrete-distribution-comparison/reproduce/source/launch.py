"""Generic integer-span distribution comparison; no biological inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from torch import nn

LENGTH = 16
WIDTH = 5
ALPHABET = 8
FEATURES = 83
CONTEXTS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MASK = np.arange(LENGTH)[:, None] + np.arange(1, WIDTH + 1) <= LENGTH


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def span_features(tokens, context):
    features = np.zeros((LENGTH, WIDTH, FEATURES), dtype=np.float32)
    for start, offset in zip(*np.where(MASK)):
        piece = np.asarray(tokens[start : start + offset + 1])
        size = len(piece) / WIDTH
        base = np.concatenate(
            [
                np.eye(ALPHABET)[piece[0]],
                np.eye(ALPHABET)[piece[-1]],
                np.bincount(piece, minlength=ALPHABET) / len(piece),
                [size, size**2, start / (LENGTH - 1)],
            ]
        )
        features[start, offset] = np.concatenate(
            [base, base * context[0], base * context[1], context]
        )
    starts = features.sum(axis=1) / MASK.sum(axis=1)[:, None]
    return features, starts.astype(np.float32)


def np_logsoftmax(value, axis):
    maximum = value.max(axis=axis, keepdims=True)
    return value - maximum - np.log(np.exp(value - maximum).sum(axis=axis, keepdims=True))


def teacher_distribution(features, starts, coefficients):
    start = np_logsoftmax(starts @ coefficients["start"], axis=0)
    length = features @ coefficients["length"]
    length = np_logsoftmax(np.where(MASK, length, -np.inf), axis=1)
    return start[:, None] + length


def make_dataset():
    rng = np.random.default_rng(20261021)
    teacher_rng = np.random.default_rng(20261022)
    coefficients = {name: teacher_rng.normal(0, 1.25, FEATURES) for name in ("start", "length")}
    rows, seen = [], set()
    for split, count in (("train", 96), ("validation", 24), ("test", 24)):
        for parent_number in range(count):
            while True:
                tokens = tuple(rng.integers(0, ALPHABET, LENGTH).tolist())
                if tokens not in seen:
                    seen.add(tokens)
                    break
            parent = f"{split}-{parent_number:03d}"
            for context_id, context in enumerate(CONTEXTS):
                features, starts = span_features(tokens, context)
                log_target = teacher_distribution(features, starts, coefficients)
                compatibility = np.zeros_like(MASK)
                response = np.zeros_like(MASK, dtype=float)
                for start, offset in zip(*np.where(MASK)):
                    last = tokens[start + offset]
                    compatibility[start, offset] = (
                        offset + 1 >= 3 and (tokens[start] + last) % 3 != 0
                    )
                    response[start, offset] = 0.2 + 0.8 * (last + 1) / ALPHABET
                rows.append(
                    {
                        "id": f"{parent}-c{context_id}",
                        "parent": parent,
                        "split": split,
                        "tokens": list(tokens),
                        "context_id": context_id,
                        "context": list(context),
                        "features": features,
                        "starts": starts,
                        "log_target": log_target,
                        "target": np.exp(log_target)[MASK].tolist(),
                        "compatible": compatibility[MASK].tolist(),
                        "response": response[MASK].tolist(),
                    }
                )
    return rows, coefficients


def public_row(row):
    return {
        key: value for key, value in row.items() if key not in ("features", "starts", "log_target")
    }


def tensors(rows):
    return (
        torch.tensor(np.stack([row["features"] for row in rows])),
        torch.tensor(np.stack([row["starts"] for row in rows])),
        torch.tensor(np.stack([row["log_target"] for row in rows]), dtype=torch.float32),
    )


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(FEATURES, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        self.length = nn.Sequential(
            nn.Linear(FEATURES, 32), nn.Tanh(), nn.Linear(32, 1, bias=False)
        )
        self.register_buffer("mask", torch.tensor(MASK))

    def forward(self, features, starts):
        start_logp = self.start(starts).squeeze(-1).log_softmax(-1)
        length_logits = self.length(features).squeeze(-1).masked_fill(~self.mask, -torch.inf)
        length_logp = length_logits.log_softmax(-1)
        return start_logp[:, :, None] + length_logp, start_logp, length_logp


def exact_loss(joint, target):
    mask = torch.tensor(MASK, device=joint.device)
    q_log = joint[:, mask]
    return (q_log.exp() * (q_log - target[:, mask])).sum(-1).mean()


def sample_spans(joint, start_logp, length_logp, count, generator):
    starts = torch.multinomial(start_logp.exp(), count, replacement=True, generator=generator)
    batch = torch.arange(len(joint), device=joint.device)[:, None].expand_as(starts)
    length_probs = length_logp.exp()[batch, starts]
    offsets = torch.multinomial(length_probs.reshape(-1, WIDTH), 1, generator=generator).reshape_as(
        starts
    )
    logp = joint[batch, starts, offsets]
    return starts, offsets, logp


def sampled_loss(output, target, count, generator, entropy=True):
    if count < 2:
        raise ValueError("leave-one-out baseline requires at least two draws")
    starts, offsets, logp = sample_spans(*output, count, generator)
    batch = torch.arange(len(logp), device=logp.device)[:, None].expand_as(starts)
    cost = -target[batch, starts, offsets]
    if entropy:
        cost = cost + logp
    baseline = (cost.sum(-1, keepdim=True) - cost) / (count - 1)
    return ((cost - baseline).detach() * logp).mean()


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2)[inverse]


def correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(records):
    target = np.asarray([row["target"] for row in records], dtype=float)
    prediction = np.asarray([row["prediction"] for row in records], dtype=float)
    compatible = np.asarray([row["compatible"] for row in records])
    response = np.asarray([row["response"] for row in records])
    p = np.maximum(target, 1e-30)
    q = np.maximum(prediction, 1e-30)
    middle = (p + q) / 2
    selected_p = p * compatible * response
    selected_q = q * compatible * response
    selected_p /= selected_p.sum(-1, keepdims=True)
    selected_q /= selected_q.sum(-1, keepdims=True)
    result = {
        "rows": len(records),
        "reverse_kl": float((q * np.log(q / p)).sum(-1).mean()),
        "forward_kl": float((p * np.log(p / q)).sum(-1).mean()),
        "cross_entropy": float(-(p * np.log(q)).sum(-1).mean()),
        "tv": float(np.abs(p - q).sum(-1).mean() / 2),
        "js": float(
            ((p * np.log(p / middle)).sum(-1) + (q * np.log(q / middle)).sum(-1)).mean() / 2
        ),
        "target_entropy": float(-(p * np.log(p)).sum(-1).mean()),
        "entropy": float(-(q * np.log(q)).sum(-1).mean()),
        "rmse_probability": float(np.sqrt(np.mean((p - q) ** 2))),
        "pearson_probability": correlation(p.ravel(), q.ravel()),
        "spearman_probability": correlation(ranks(p.ravel()), ranks(q.ravel())),
        "noncompatible_mass_mae": float(
            np.abs((p * ~compatible).sum(-1) - (q * ~compatible).sum(-1)).mean()
        ),
        "noncompatible_target_mass": float((p * ~compatible).sum(-1).mean()),
        "noncompatible_predicted_mass": float((q * ~compatible).sum(-1).mean()),
        "selected_tv": float(np.abs(selected_p - selected_q).sum(-1).mean() / 2),
        "target_probability_sd": float(p.std()),
        "predicted_probability_sd": float(q.std()),
    }
    return result


def predict(model, rows):
    records = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for first in range(0, len(rows), 64):
            batch = rows[first : first + 64]
            if model is None:
                probabilities = np.full((len(batch), MASK.sum()), 1 / MASK.sum())
            else:
                x, starts, _ = tensors(batch)
                probabilities = model(x, starts)[0][:, MASK].exp().numpy()
            for row, probability in zip(batch, probabilities):
                records.append({**public_row(row), "prediction": probability.tolist()})
    return records


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def gradient_vector(model):
    return torch.cat([parameter.grad.flatten() for parameter in model.parameters()])


def gradient_probe(rows):
    torch.manual_seed(42)
    model = Policy()
    x, starts, target = tensors(rows[:4])
    exact_loss(model(x, starts)[0], target).backward()
    reference = gradient_vector(model).detach().clone()
    results = {"reference_norm": float(reference.norm()), "repeats": 128, "conditions": {}}
    for count in (8, 64):
        generator = torch.Generator().manual_seed(8080 + count)
        estimates = []
        for _ in range(128):
            model.zero_grad(set_to_none=True)
            sampled_loss(model(x, starts), target, count, generator).backward()
            estimates.append(gradient_vector(model).detach().clone())
        estimates = torch.stack(estimates)
        mean = estimates.mean(0)
        results["conditions"][str(count)] = {
            "mean_relative_error": float((mean - reference).norm() / reference.norm()),
            "single_relative_error_mean": float(
                ((estimates - reference).norm(dim=1) / reference.norm()).mean()
            ),
            "single_cosine_mean": float(
                torch.nn.functional.cosine_similarity(estimates, reference[None], dim=1).mean()
            ),
            "mean_cosine": float(torch.nn.functional.cosine_similarity(mean, reference, dim=0)),
            "sum_parameter_variances": float(estimates.var(0, unbiased=True).sum()),
            "reward_queries": 128 * len(x) * count,
        }
    return results


def train(method, seed, groups, output):
    out = output / f"{method}-seed{seed}"
    out.mkdir()
    torch.manual_seed(seed)
    model = Policy()
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(10000 + seed)
    sample_rng = torch.Generator().manual_seed(20000 + seed)
    train_x, train_starts, train_target = tensors(groups["train"])
    samples = {"exact": 0, "reinforce8": 8, "reinforce64": 64, "reward_only64": 64}[method]
    history, gradients = [], {name: False for name, _ in model.named_parameters()}
    best, best_state, steps = float("inf"), None, 0
    begin = time.perf_counter()
    for epoch in range(1, 61):
        model.train()
        order = rng.permutation(len(train_x))
        for first in range(0, len(order), 16):
            indices = order[first : first + 16]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[indices], train_starts[indices])
            if method == "exact":
                loss = exact_loss(prediction[0], train_target[indices])
            else:
                loss = sampled_loss(
                    prediction,
                    train_target[indices],
                    samples,
                    sample_rng,
                    entropy=method != "reward_only64",
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss: {method}/{seed}/{epoch}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"missing/nonfinite gradient: {name}")
                gradients[name] |= bool(torch.any(parameter.grad != 0))
            optimizer.step()
            steps += 1
        if epoch % 5 == 0:
            validation = metrics(predict(model, groups["validation"]))
            history.append({"epoch": epoch, "steps": steps, **validation})
            if validation["reverse_kl"] < best:
                best = validation["reverse_kl"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            print(
                f"{method} seed={seed} epoch={epoch} "
                f"val_kl={validation['reverse_kl']:.5f} tv={validation['tv']:.5f}",
                flush=True,
            )
    fit_seconds = time.perf_counter() - begin
    torch.save(model.state_dict(), out / "last-state.pt")
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "method": method, "seed": seed}, out / "selected.pt")
    results = {}
    for split, rows in groups.items():
        records = predict(model, rows)
        write_rows(out / f"{split}-predictions.jsonl", records)
        results[split] = metrics(records)
        results[split]["contexts"] = {
            str(context): metrics([row for row in records if row["context_id"] == context])
            for context in range(4)
        }
    updated = {
        name: float((parameter.detach() - initial[name]).abs().max())
        for name, parameter in model.named_parameters()
    }
    if not all(gradients.values()) or not all(value > 0 for value in updated.values()):
        raise RuntimeError("allocated parameter coverage failed")
    model.eval()
    draws = []
    generator = torch.Generator().manual_seed(30000 + seed)
    with torch.no_grad():
        for row in groups["test"]:
            x, starts, _ = tensors([row])
            chosen, offsets, logp = sample_spans(*model(x, starts), 64, generator)
            samples_for_row = []
            for start, offset, probability in zip(chosen[0], offsets[0], logp[0]):
                start, size = int(start), int(offset) + 1
                samples_for_row.append(
                    {
                        "start": start,
                        "length": size,
                        "tokens": row["tokens"][start : start + size],
                        "log_probability": float(probability),
                    }
                )
            draws.append({"id": row["id"], "samples": samples_for_row})
    write_rows(out / "test-multisets.jsonl", draws)
    write_rows(out / "replay-training.jsonl", predict(model, groups["train"][:4]))
    report = {
        "method": method,
        "seed": seed,
        "initial_state_sha256": state_hash(initial),
        "selected_epoch": best_epoch,
        "updates": steps,
        "fit_seconds_including_validation": fit_seconds,
        "fit_reward_queries": len(groups["train"])
        * 60
        * (MASK.sum().item() if not samples else samples),
        "gradient_nonzero": gradients,
        "max_parameter_updates": updated,
        "metrics": results,
    }
    write_json(out / "history.json", history)
    write_json(out / "summary.json", report)
    return report


def command(args):
    return subprocess.check_output(args, text=True).strip()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    script = Path(__file__).resolve()
    receipt = {
        "argv": sys.argv,
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
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
        "source_sha256": sha(script),
    }
    write_json(output / "execution.json", receipt)
    (output / "working-tree.diff").write_text(command(["git", "diff", "--binary"]))
    rows, coefficients = make_dataset()
    groups = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    write_rows(output / "dataset.jsonl", [public_row(row) for row in rows])
    write_json(
        output / "teacher.json", {key: value.tolist() for key, value in coefficients.items()}
    )
    write_json(output / "gradient-probe.json", gradient_probe(groups["train"]))
    baseline = {}
    for split, subset in groups.items():
        records = predict(None, subset)
        write_rows(output / f"uniform-{split}-predictions.jsonl", records)
        baseline[split] = metrics(records)
    reports = []
    for method in ("exact", "reinforce8", "reinforce64"):
        for seed in (42, 43, 44):
            reports.append(train(method, seed, groups, output))
    reports.append(train("reward_only64", 42, groups, output))
    screens = {}
    for method in ("exact", "reinforce8", "reinforce64"):
        selected = [report for report in reports if report["method"] == method]
        screens[method] = all(
            report["metrics"]["validation"]["reverse_kl"]
            <= 0.25 * baseline["validation"]["reverse_kl"]
            and report["metrics"]["validation"]["tv"] <= 0.10
            and report["metrics"]["validation"]["noncompatible_mass_mae"] <= 0.05
            for report in selected
        )
    write_json(
        output / "summary.json",
        {
            "uniform": baseline,
            "conditions": reports,
            "screens": screens,
            "elapsed_seconds": time.perf_counter() - begin,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    write_json(
        output / "artifacts.json",
        {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    print(json.dumps({"completed": True, "screens": screens}), flush=True)


def replay(output):
    torch.set_num_threads(1)
    rows, _ = make_dataset()
    training = [row for row in rows if row["split"] == "train"][:4]
    checked = []
    for checkpoint in sorted(output.glob("*/selected.pt")):
        model = Policy()
        model.load_state_dict(torch.load(checkpoint, weights_only=True)["state"])
        actual = predict(model, training)
        expected = [
            json.loads(line)
            for line in (checkpoint.parent / "replay-training.jsonl").read_text().splitlines()
        ]
        if actual != expected:
            raise AssertionError(f"fresh-process training replay mismatch: {checkpoint}")
        checked.append(str(checkpoint.relative_to(output)))
    if len(checked) != 10:
        raise AssertionError(f"expected ten selected checkpoints, found {len(checked)}")
    write_json(
        output.parent / "fresh-process-replay.json",
        {"status": "passed", "checkpoints": checked, "test_rescored": False},
    )
    print(f"Exact fresh-process training-row replay passed for {len(checked)} checkpoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    (run if arguments.mode == "run" else replay)(arguments.output)
