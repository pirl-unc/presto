"""Independent arithmetic audit of saved results; never imports the trainer."""

import csv
import hashlib
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import rankdata


FAMILY = Path(__file__).resolve().parents[1]
RUN = FAMILY / "results/run"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def corr(a, b):
    return None if np.ptp(a) == 0 or np.ptp(b) == 0 else float(np.corrcoef(a, b)[0, 1])


def reconstruct(row, teacher):
    tokens, context = row["tokens"], row["context"]
    features = np.zeros((16, 5, 83), dtype=np.float32)
    compatible, response, coordinates = [], [], []
    for start in range(16):
        for size in range(1, min(5, 16 - start) + 1):
            substring = tokens[start : start + size]
            first = [int(symbol == substring[0]) for symbol in range(8)]
            last = [int(symbol == substring[-1]) for symbol in range(8)]
            counts = [substring.count(symbol) / size for symbol in range(8)]
            base = np.asarray(first + last + counts + [size / 5, (size / 5) ** 2, start / 15])
            features[start, size - 1] = np.concatenate(
                [base, context[0] * base, context[1] * base, context]
            )
            compatible.append(size >= 3 and (substring[0] + substring[-1]) % 3 != 0)
            response.append(0.2 + 0.8 * (substring[-1] + 1) / 8)
            coordinates.append((start, size))
    starts = np.stack([features[start, : min(5, 16 - start)].mean(axis=0) for start in range(16)])
    start_scores = starts @ np.asarray(teacher["start"])
    start_logp = start_scores - logsumexp(start_scores)
    target = []
    for start in range(16):
        length_scores = features[start, : min(5, 16 - start)] @ np.asarray(teacher["length"])
        target.extend(np.exp(start_logp[start] + length_scores - logsumexp(length_scores)))
    assert np.allclose(target, row["target"], atol=1e-8, rtol=3e-6)
    assert compatible == row["compatible"]
    assert response == row["response"]
    return coordinates


def recompute(rows):
    p = np.asarray([row["target"] for row in rows])
    q = np.asarray([row["prediction"] for row in rows])
    c = np.asarray([row["compatible"] for row in rows])
    response = np.asarray([row["response"] for row in rows])
    assert np.isfinite(q).all() and np.all(q > 0)
    assert np.allclose(q.sum(axis=1), 1, atol=3e-7)
    q = np.maximum(q, 1e-30)
    p = np.maximum(p, 1e-30)
    middle = (p + q) / 2
    observed_p, observed_q = p * c * response, q * c * response
    observed_p /= observed_p.sum(axis=1, keepdims=True)
    observed_q /= observed_q.sum(axis=1, keepdims=True)
    return {
        "rows": len(rows),
        "reverse_kl": float(np.mean(np.sum(q * (np.log(q) - np.log(p)), axis=1))),
        "forward_kl": float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1))),
        "cross_entropy": float(np.mean(np.sum(-p * np.log(q), axis=1))),
        "tv": float(np.mean(np.sum(abs(p - q), axis=1)) / 2),
        "js": float(np.mean(np.sum(p * np.log(p / middle) + q * np.log(q / middle), axis=1)) / 2),
        "target_entropy": float(np.mean(np.sum(-p * np.log(p), axis=1))),
        "entropy": float(np.mean(np.sum(-q * np.log(q), axis=1))),
        "rmse_probability": float(np.sqrt(np.mean(np.square(p - q)))),
        "pearson_probability": corr(p.ravel(), q.ravel()),
        "spearman_probability": corr(rankdata(p.ravel()), rankdata(q.ravel())),
        "noncompatible_mass_mae": float(np.mean(abs(np.sum((p - q) * ~c, axis=1)))),
        "noncompatible_target_mass": float(np.mean(np.sum(p * ~c, axis=1))),
        "noncompatible_predicted_mass": float(np.mean(np.sum(q * ~c, axis=1))),
        "selected_tv": float(np.mean(np.sum(abs(observed_p - observed_q), axis=1)) / 2),
        "target_probability_sd": float(p.std()),
        "predicted_probability_sd": float(q.std()),
    }


def compare(expected, actual):
    for key, value in actual.items():
        if value is None:
            assert expected[key] is None, key
        else:
            assert np.isclose(expected[key], value, atol=1e-10, rtol=1e-8), (
                key,
                expected[key],
                value,
            )


def main():
    dataset = read_rows(RUN / "dataset.jsonl")
    by_id = {row["id"]: row for row in dataset}
    assert len(by_id) == 576
    teacher = json.loads((RUN / "teacher.json").read_text())
    parents = {}
    coordinate_map = {}
    for row in dataset:
        coordinate_map[row["id"]] = reconstruct(row, teacher)
        parents.setdefault(tuple(row["tokens"]), set()).add(row["split"])
    assert len(parents) == 144 and all(len(splits) == 1 for splits in parents.values())
    hashes = json.loads((RUN / "artifacts.json").read_text())
    for path, digest in hashes.items():
        assert hashlib.sha256((RUN / path).read_bytes()).hexdigest() == digest, path
    execution = json.loads((RUN / "execution.json").read_text())
    frozen = json.loads((FAMILY / "reproduce/source.json").read_text())
    assert execution["source_sha256"] == frozen["launcher_sha256"]
    summary = json.loads((RUN / "summary.json").read_text())
    table, audit, samples = [], [], []
    for split in ("train", "validation", "test"):
        records = read_rows(RUN / f"uniform-{split}-predictions.jsonl")
        compare(summary["uniform"][split], recompute(records))
    initial_by_seed = {}
    for condition in summary["conditions"]:
        name = f"{condition['method']}-seed{condition['seed']}"
        directory = RUN / name
        initial_by_seed.setdefault(condition["seed"], set()).add(condition["initial_state_sha256"])
        history = json.loads((directory / "history.json").read_text())
        best = min(history, key=lambda row: row["reverse_kl"])
        assert best["epoch"] == condition["selected_epoch"]
        assert condition["updates"] == 1440
        assert all(condition["gradient_nonzero"].values())
        assert all(value > 0 for value in condition["max_parameter_updates"].values())
        for split in ("train", "validation", "test"):
            records = read_rows(directory / f"{split}-predictions.jsonl")
            assert {row["id"] for row in records} == {
                row["id"] for row in dataset if row["split"] == split
            }
            for row in records:
                assert {key: value for key, value in row.items() if key != "prediction"} == by_id[
                    row["id"]
                ]
            computed = recompute(records)
            compare(condition["metrics"][split], computed)
            for context in range(4):
                compare(
                    condition["metrics"][split]["contexts"][str(context)],
                    recompute([row for row in records if row["context_id"] == context]),
                )
            audit.append({"condition": name, "split": split, "status": "passed"})
            table.append(
                {
                    "method": condition["method"],
                    "seed": condition["seed"],
                    "split": split,
                    "selected_epoch": condition["selected_epoch"],
                    "fit_seconds": condition["fit_seconds_including_validation"],
                    "logical_fit_reward_queries": condition["fit_reward_queries"],
                    **computed,
                }
            )
        predictions = {row["id"]: row for row in read_rows(directory / "test-predictions.jsonl")}
        drawn = read_rows(directory / "test-multisets.jsonl")
        assert {row["id"] for row in drawn} == set(predictions)
        noncompatible = 0
        for row in drawn:
            source = by_id[row["id"]]
            predicted = predictions[row["id"]]["prediction"]
            coordinates = coordinate_map[row["id"]]
            assert len(row["samples"]) == 64
            for draw in row["samples"]:
                start, size = draw["start"], draw["length"]
                index = coordinates.index((start, size))
                assert draw["tokens"] == source["tokens"][start : start + size]
                assert np.isclose(draw["log_probability"], np.log(predicted[index]), atol=3e-6)
                noncompatible += int(not source["compatible"][index])
        predicted_mass = condition["metrics"]["test"]["noncompatible_predicted_mass"]
        samples.append(
            {
                "condition": name,
                "draws": len(drawn) * 64,
                "empirical_noncompatible_mass": noncompatible / (len(drawn) * 64),
                "predicted_noncompatible_mass": predicted_mass,
            }
        )
    assert all(len(values) == 1 for values in initial_by_seed.values())
    with (FAMILY / "results/metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    dump(
        FAMILY / "results/independent-verification.json",
        {
            "status": "passed",
            "teacher_rows": len(dataset),
            "original_artifact_hashes": len(hashes),
            "condition_splits": audit,
            "discrete_samples": samples,
            "same_initialization_by_seed": True,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    colors = {
        "exact": "#145da0",
        "reinforce8": "#da8b16",
        "reinforce64": "#249a73",
        "reward_only64": "#bb4242",
    }
    for condition in summary["conditions"]:
        method, seed = condition["method"], condition["seed"]
        history = json.loads((RUN / f"{method}-seed{seed}/history.json").read_text())
        axes[0].plot(
            [row["epoch"] for row in history],
            [row["reverse_kl"] for row in history],
            color=colors[method],
            alpha=0.7,
            label=method if seed == 42 else None,
        )
    axes[0].set(
        yscale="log",
        xlabel="Complete training epochs",
        ylabel="Validation reverse KL",
        title="Same distribution objective except reward-only control",
    )
    axes[0].legend(fontsize=8)
    methods = list(colors)
    for index, method in enumerate(methods):
        values = [
            condition["metrics"]["test"]["tv"]
            for condition in summary["conditions"]
            if condition["method"] == method
        ]
        axes[1].scatter(np.full(len(values), index), values, color=colors[method], s=45)
    axes[1].axhline(
        summary["uniform"]["test"]["tv"], color="gray", linestyle="--", label="Uniform baseline"
    )
    axes[1].set(
        xticks=range(len(methods)),
        xticklabels=methods,
        ylabel="Held-out total variation",
        title="Selected checkpoints on new parent strings",
    )
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8)
    fig.savefig(FAMILY / "results/comparison.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "status": "passed",
                "teacher_rows": len(dataset),
                "metric_splits": len(audit),
                "verified_discrete_draws": sum(row["draws"] for row in samples),
            }
        )
    )


if __name__ == "__main__":
    main()
