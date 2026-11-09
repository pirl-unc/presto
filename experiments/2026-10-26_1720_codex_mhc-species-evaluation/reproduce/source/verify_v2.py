"""V2: require the source plus its exact manifest evidence-policy normalization."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch

from common import CONDITIONS, FAMILY, guarded, write

SPECIES = ("human", "mouse", "rat", "rhesus_macaque")
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def close(actual, expected, tolerance=1e-9):
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def audit_metrics(records, report):
    labels = np.array([r["target"] for r in records])
    probabilities = np.array([r["probabilities"] for r in records])
    assert probabilities.shape == (len(labels), 4)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    close(probabilities.sum(axis=1), 1, 2e-7)
    calls = np.argmax(probabilities, axis=1)
    losses = -np.log(
        np.maximum(probabilities[np.arange(len(labels)), labels], np.finfo(float).tiny)
    )
    assert all(r["prediction"] == c for r, c in zip(records, calls))
    close([r["loss"] for r in records], losses, 3e-6)
    confusion = np.zeros((4, 4), int)
    for target, call in zip(labels, calls):
        confusion[target, call] += 1
    close(report["confusion"], confusion)
    group = defaultdict(list)
    for i, row in enumerate(records):
        group[row["group_id"]].append(i)
    assert report["n"] == len(labels) and report["components"] == len(group)
    close(report["ce"], losses.mean())
    close(report["accuracy"], (labels == calls).mean())
    close(report["component_mean_ce"], np.mean([losses[ix].mean() for ix in group.values()]))
    close(
        report["component_mean_accuracy"],
        np.mean([(labels[ix] == calls[ix]).mean() for ix in group.values()]),
    )
    close(report["probability_sd"], probabilities.std(axis=0))
    close(report["brier"], np.sum((probabilities - np.eye(4)[labels]) ** 2, axis=1).mean())
    recalls, f1s = [], []
    for j, name in enumerate(SPECIES):
        actual = report["per_species"][name]
        positive = labels == j
        n, predicted, tp = int(positive.sum()), int((calls == j).sum()), int(confusion[j, j])
        assert actual["n"] == n
        assert actual["components"] == len(
            {r["group_id"] for r, yes in zip(records, positive) if yes}
        )
        precision = tp / predicted if predicted else 0
        recall = tp / n if n else None
        f1 = 2 * tp / (n + predicted) if n else None
        close(actual["precision"], precision)
        close(actual["recall"], recall)
        close(actual["f1"], f1)
        if n:
            recalls.append(recall)
            f1s.append(f1)
        scores = probabilities[:, j]
        auc = (
            (rankdata(scores)[positive].sum() - n * (n + 1) / 2) / (n * (len(labels) - n))
            if 0 < n < len(labels)
            else None
        )
        close(actual["auroc"], auc)
        ap, seen, hits = 0.0, 0, 0
        if n:
            for value in np.unique(scores)[::-1]:
                tied = scores == value
                new_hits = int(positive[tied].sum())
                seen += int(tied.sum())
                hits += new_hits
                ap += new_hits / n * hits / seen
        close(actual["auprc"], ap if n else None)
    close(report["balanced_accuracy"], np.mean(recalls))
    close(report["macro_f1"], np.mean(f1s))
    bins = np.minimum((probabilities.max(1) * 10).astype(int), 9)
    ece = 0.0
    for j, saved in enumerate(report["calibration"]):
        indices = bins == j
        assert saved["n"] == int(indices.sum()) and saved["bin"] == j
        if indices.any():
            confidence = probabilities.max(1)[indices].mean()
            accuracy = (calls[indices] == labels[indices]).mean()
            close(saved["confidence"], confidence)
            close(saved["accuracy"], accuracy)
            ece += indices.mean() * abs(confidence - accuracy)
        else:
            assert saved["confidence"] is None and saved["accuracy"] is None
    close(report["ece_10_bins"], ece)
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("baselines", *CONDITIONS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with guarded("verify-" + args.condition):
        launch = read(FAMILY / "reproduce/launch.json")
        pin = launch["manifests"]["seed42" if args.condition == "baselines" else args.condition]
        assert sha(pin["path"]) == pin["sha256"]
        source = {}
        for item in read(pin["path"])["sources"]:
            assert sha(item["path"]) == item["sha256"]
            for row in rows(item["path"]):
                assert row["observation_id"] not in source
                assert (
                    row["provenance"].get("evidence_policy", item["label_policy"])
                    == item["label_policy"]
                )
                normalized = json.loads(json.dumps(row))
                normalized["provenance"].setdefault("evidence_policy", item["label_policy"])
                source[row["observation_id"]] = normalized
        assert Counter(r["split"] for r in source.values()) == {
            "train": 6092,
            "validation": 694,
            "test": 6432,
        }
        groups = defaultdict(set)
        for r in source.values():
            groups[r["group_id"]].add(r["split"])
        assert all(len(splits) == 1 for splits in groups.values())
        comparison = read(FAMILY / "results/comparison.json")
        conditions = (
            ("prior", "composition") if args.condition == "baselines" else (args.condition,)
        )
        directory = FAMILY / "results" / args.condition
        total = 0
        predicted = {}
        for condition in conditions:
            prefix = condition + "-" if args.condition == "baselines" else ""
            for split in ("train", "validation", "test"):
                records = rows(directory / f"{prefix}{split}-predictions.jsonl.gz")
                assert len({r["observation_id"] for r in records}) == len(records)
                assert {r["observation_id"] for r in records} == {
                    k for k, r in source.items() if r["split"] == split
                }
                for record in records:
                    original = source[record["observation_id"]]
                    assert record["observation"] == original
                    assert record["group_id"] == original["group_id"] and record["split"] == split
                    task, value = next(iter(original["identity_labels"].items()))
                    assert record["identity_task"] == task and record["target"] == SPECIES.index(
                        value
                    )
                measures = comparison["metrics"][condition][split]
                audit_metrics(records, measures)
                for slot, saved in measures["by_slot"].items():
                    audit_metrics([r for r in records if r["identity_task"] == slot], saved)
                predicted[condition, split] = records
                total += len(records)
        evidence = {
            "status": "passed",
            "records_verified": total,
            "scope": (
                "all saved source joins, probabilities and classification metrics; no test rescore"
            ),
        }
        if args.condition == "baselines":
            parameters = read(directory / "parameters.json")
            train = [r for r in source.values() if r["split"] == "train"]

            def design(subset):
                matrix = []
                for row in subset:
                    a, b = row["mhc"][0]["groove1"], row["mhc"][0]["groove2"]
                    assert bool(a) != bool(b)
                    sequence = a + b
                    matrix.append(
                        [
                            len(sequence),
                            *[sequence.count(aa) / len(sequence) for aa in AA],
                            float(bool(b)),
                        ]
                    )
                return np.asarray(matrix, float)

            x = design(train)
            center, scale = x.mean(0), x.std(0)
            scale[scale == 0] = 1
            close(parameters["center"], center)
            close(parameters["scale"], scale)
            labels = np.array(
                [SPECIES.index(next(iter(r["identity_labels"].values()))) for r in train]
            )
            prior = (np.bincount(labels, minlength=4) + 1) / (len(train) + 4)
            close(parameters["prior"], prior)
            w = np.asarray(parameters["coefficients"])
            z = np.column_stack([np.ones(len(x)), (x - center) / scale])
            logits = z @ w
            exponents = np.exp(logits - logits.max(1, keepdims=True))
            p = exponents / exponents.sum(1, keepdims=True)
            gradient = np.zeros_like(w)
            for j in range(4):
                gradient[:, j] = np.sum(z * (p[:, j] - (labels == j))[:, None], axis=0)
                gradient[1:, j] += w[1:, j]
            evidence["independent_gradient_max"] = float(np.abs(gradient).max())
            assert evidence["independent_gradient_max"] <= 1e-3
            for (condition, _split), records in predicted.items():
                inputs = [source[r["observation_id"]] for r in records]
                matrix = np.column_stack([np.ones(len(inputs)), (design(inputs) - center) / scale])
                logits = matrix @ w
                exponents = np.exp(logits - logits.max(1, keepdims=True))
                expected = (
                    exponents / exponents.sum(1, keepdims=True)
                    if condition == "composition"
                    else np.broadcast_to(prior, (len(inputs), 4))
                )
                close([r["probabilities"] for r in records], expected)
        else:
            from presto.data.modular_sources import ObservationIndex
            from presto.training.modular import load_checkpoint
            from presto.training.modular_evaluation import evaluate

            raw = Path(launch["raw_artifacts"]) / args.condition
            for name, meta in read(directory / "artifacts.json").items():
                assert sha(raw / name) == meta["sha256"]
            history, summary = read(directory / "history.json"), read(directory / "summary.json")
            assert len(history) <= 50 and len(history) >= 10
            assert summary["coverage_gate"]["status"] == "passed"
            for epoch, record in enumerate(history, 1):
                assert record["epoch"] == epoch and record["complete_epoch"]
                assert record["steps"] == epoch * math.ceil(6092 / 64)
                assert record["label_exposures"] == {"mhc_species": 6092}
            assert summary["training_exposures"] == {"mhc_species": 6092 * len(history)}
            scores = [
                sum(v["n"] * v["loss"] for v in h["validation"].values()) / 694 for h in history
            ]
            assert summary["selected_epoch"] == int(np.argmin(scores)) + 1
            model, scales, _ = load_checkpoint(raw / "selected.pt", device="cpu")
            initial, _, _ = load_checkpoint(raw / "initial.pt", device="cpu")
            assert all(
                not torch.equal(p, initial.state_dict()[name])
                for name, p in model.named_parameters()
            )
            index = ObservationIndex(pin["path"])
            _, replay = evaluate(
                model, index, index.partitions["train:identity"][:64], scales, batch_size=64
            )
            expected = {r["observation_id"]: r for r in predicted[args.condition, "train"]}
            for record in replay:
                saved = expected[record["observation_id"]]
                assert (
                    record["target"] == saved["target"]
                    and record["observation"] == saved["observation"]
                )
                close(record["probabilities"], saved["probabilities"], 1e-6)
            validation = predicted[args.condition, "validation"]
            for baseline in ("prior", "composition"):
                controls = {
                    r["observation_id"]: r
                    for r in rows(
                        FAMILY / "results/baselines" / f"{baseline}-validation-predictions.jsonl.gz"
                    )
                }
                differences = defaultdict(list)
                for record in validation:
                    control = controls.pop(record["observation_id"])
                    target = record["target"]
                    differences[record["group_id"]].append(
                        float(
                            -np.log(record["probabilities"][target])
                            + np.log(control["probabilities"][target])
                        )
                    )
                assert not controls
                values = np.asarray([np.mean(differences[g]) for g in sorted(differences)])
                generator = np.random.default_rng(20261026)
                samples = [
                    values[generator.integers(0, len(values), len(values))].mean()
                    for _ in range(2000)
                ]
                saved = comparison["screens"][args.condition]["comparisons"][baseline]
                close(saved["component_mean_difference"], values.mean())
                close(saved["interval_95"], np.quantile(samples, [0.025, 0.975]))
            evidence["component_bootstraps_verified"] = 2
            evidence.update(
                complete_epochs=len(history),
                training_replay=len(replay),
                selected_checkpoint_sha256=sha(raw / "selected.pt"),
            )
        write(directory / "independent-verification.json", evidence)
        print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
