#!/usr/bin/env python
"""Fixed reference-membership comparison using the canonical modular trainer."""

from __future__ import annotations

import argparse
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

from presto.data.modular import AA
from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training.modular import load_checkpoint
from presto.training.modular_evaluation import binary_metrics, evaluate
from presto.training.modular_resume import SOURCE_FILES
from presto.training.modular_run import fit_index

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "experiments/2026-10-21_0512_codex_weirdo-reference-pretraining"
MANIFEST = ROOT / "data/modular_cohorts/2026-10-21-weirdo-reference/manifest.json"
MANIFEST_SHA256 = "ca008ea769c6dc5332cb14f4ef1d7bd125c79064223692596608588b8e4cff8f"
CONDITIONS = ("training_prevalence", "composition_logistic", "modular_peptide_source")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_lines(path, values):
    with gzip.open(path, "wt") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def composition(rows):
    return np.asarray(
        [[row.peptide.count(aa) / len(row.peptide) for aa in AA] for row in rows], dtype=float
    )


def binary_report(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if (
        labels.ndim != 1
        or not len(labels)
        or labels.shape != probabilities.shape
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("finite aligned binary labels/probabilities required")
    p = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    losses = -labels * np.log(p) - (1 - labels) * np.log1p(-p)
    result = binary_metrics(labels, probabilities, probabilities >= 0.5)
    bins, ece = np.minimum((probabilities * 10).astype(int), 9), 0.0
    calibration = []
    for i in range(10):
        mask = bins == i
        n = int(mask.sum())
        predicted = float(probabilities[mask].mean()) if n else None
        observed = float(labels[mask].mean()) if n else None
        if n:
            ece += n / len(labels) * abs(predicted - observed)
        calibration.append({"bin": i, "n": n, "predicted": predicted, "observed": observed})
    result.update(
        positives=int(labels.sum()),
        negatives=int(len(labels) - labels.sum()),
        prevalence=float(labels.mean()),
        ce=float(losses.mean()),
        brier=float(np.mean((probabilities - labels) ** 2)),
        ece_10_bins=ece,
        calibration=calibration,
        probability_sd=float(probabilities.std()),
        probability_range=[float(probabilities.min()), float(probabilities.max())],
        probability_quantiles=np.quantile(probabilities, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    )
    return result, losses


def paired_interval(differences, groups, *, resamples=2000, seed=20261021):
    """Sample components uniformly; retain all member peptides and their weights."""
    differences = np.asarray(differences, dtype=float)
    if (
        differences.ndim != 1
        or len(differences) != len(groups)
        or not len(groups)
        or not np.isfinite(differences).all()
    ):
        raise ValueError("aligned finite paired differences and groups required")
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=differences)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        selected = rng.integers(0, len(counts), size=len(counts))
        values.append(float(sums[selected].sum() / counts[selected].sum()))
    return {
        "mean_difference": float(differences.mean()),
        "interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
        "components": len(counts),
        "resamples": resamples,
        "seed": seed,
    }


def network_matrix(path, rows, categories):
    lookup = {}
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            key = record["observation_id"], record["identity_task"]
            if key in lookup:
                raise ValueError("duplicate exported network prediction")
            lookup[key] = record
    matrix = np.empty((len(rows), len(categories)))
    for i, row in enumerate(rows):
        for j, category in enumerate(categories):
            key = row.observation_id, f"peptide_source:{category}"
            record = lookup.pop(key)
            expected = int(row.identity_labels[key[1]] == "present")
            if (
                record["target"] != expected
                or record["peptide"] != row.peptide
                or record["group_id"] != row.group_id
                or record["split"] != row.split
            ):
                raise ValueError("network prediction provenance/label mismatch")
            matrix[i, j] = record["probabilities"][1]
    if lookup:
        raise ValueError("unexpected exported network predictions")
    return matrix


def verify_training_batch(index, raw, output):
    model, scales, _ = load_checkpoint(raw / "selected.pt", device="cpu")
    indices = index.partitions["train:identity"][:64]
    _, records = evaluate(model, index, indices, scales, batch_size=64)
    expected = {}
    with (raw / "train-predictions.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            expected[record["observation_id"], record["identity_task"]] = record
    if any(expected[(r["observation_id"], r["identity_task"])] != r for r in records):
        raise ValueError("fresh-process training-batch predictions changed")
    write_json(
        output / "results/checkpoint-replay.json",
        {
            "status": "passed",
            "scope": "first 64 training peptides; test predictions not rescored",
            "labels": len(records),
            "checkpoint_sha256": file_sha256(raw / "selected.pt"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FAMILY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--preparation", type=Path, default=FAMILY / "attempts/reference-groups")
    parser.add_argument("--mode", choices=("launch", "fit", "replay"), default="launch")
    args = parser.parse_args()
    output, manifest = args.output.resolve(), args.manifest.resolve()
    raw = ROOT / "artifacts" / output.relative_to(ROOT / "experiments") / "fit"
    torch.set_num_threads(1)
    if file_sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("this frozen family requires its audited manifest bytes")
    index = ObservationIndex(manifest)
    if args.mode == "fit":
        fit_index(index, raw, device="cpu")
        return
    if args.mode == "replay":
        verify_training_batch(index, raw, output)
        return
    # Only the real baseline comparison requires these optional dependencies.
    from scipy.special import expit
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    if (output / "reproduce/fit-launch.json").exists() or raw.exists():
        raise ValueError("refusing to overwrite an executed comparison")
    preparation = read_json(args.preparation / "results/preparation.json")
    if preparation["status"] != "passed" or preparation["manifest"]["sha256"] != MANIFEST_SHA256:
        raise ValueError("completed reference admission required")
    for name in ("results", "launch_logs", "reproduce/fit-source"):
        (output / name).mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True)
    receipt = {
        "agent_model": "Codex / GPT-6",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "argv": sys.argv,
        "executable": sys.executable,
        "cwd": str(ROOT),
        "python": sys.version,
        "versions": {name: version(name) for name in ("numpy", "torch", "scipy", "scikit-learn")},
        "environment": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "manifest": {"path": str(manifest), "sha256": MANIFEST_SHA256},
        "preparation": {
            "path": str(args.preparation),
            "sha256": file_sha256(args.preparation / "results/preparation.json"),
        },
        "conditions": CONDITIONS,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "gpu": None,
        "network_subprocess_ceiling_seconds": 900,
    }
    write_json(output / "reproduce/fit-launch.json", receipt)
    for relative in (*SOURCE_FILES, "data/modular_reference.py"):
        target = output / "reproduce/fit-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(__file__, output / "reproduce/fit-source/launch.py")
    started = time.monotonic()
    try:
        rows = {
            split: [index[i] for i in index.partitions[f"{split}:identity"]]
            for split in ("train", "validation", "test")
        }
        categories = index.manifest["model"]["source_categories"]
        labels = {
            split: np.asarray(
                [
                    [
                        int(row.identity_labels[f"peptide_source:{c}"] == "present")
                        for c in categories
                    ]
                    for row in subset
                ]
            )
            for split, subset in rows.items()
        }
        features = {split: composition(subset) for split, subset in rows.items()}
        baseline_started = time.monotonic()
        prior = (labels["train"].sum(axis=0) + 0.5) / (len(rows["train"]) + 1)
        coefficients, intercepts, convergence = [], [], []
        for j, category in enumerate(categories):
            classifier = LogisticRegression(C=1, solver="lbfgs", max_iter=1000)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(features["train"], labels["train"][:, j])
            coefficients.append(classifier.coef_[0].tolist())
            intercepts.append(float(classifier.intercept_[0]))
            convergence.append(
                {
                    "category": category,
                    "iterations": int(classifier.n_iter_[0]),
                    "warnings": [str(w.message) for w in caught],
                }
            )
        baseline_seconds = time.monotonic() - baseline_started
        write_json(
            output / "results/baseline-parameters.json",
            {
                "categories": categories,
                "features": list(AA),
                "prior": prior.tolist(),
                "coefficients": coefficients,
                "intercepts": intercepts,
                "convergence": convergence,
                "elapsed_seconds": baseline_seconds,
                "C": 1,
                "solver": "lbfgs",
                "penalty": "l2",
                "class_weight": None,
            },
        )
        child = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ]
        fit_command = [*child, "--mode", "fit"]
        receipt["fit_argv"] = fit_command
        write_json(output / "reproduce/fit-launch.json", receipt)
        network_started = time.monotonic()
        with (output / "launch_logs/network.log").open("w") as log:
            subprocess.run(
                fit_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900
            )
        network_seconds = time.monotonic() - network_started
        subprocess.run([*child, "--mode", "replay"], cwd=ROOT, check=True, timeout=120)
        summary, validation_losses = {}, {}
        for split, subset in rows.items():
            predictions = {
                CONDITIONS[0]: np.broadcast_to(prior, labels[split].shape),
                CONDITIONS[1]: expit(features[split] @ np.asarray(coefficients).T + intercepts),
                CONDITIONS[2]: network_matrix(
                    raw / f"{split}-predictions.jsonl", subset, categories
                ),
            }
            for condition, matrix in predictions.items():
                per_category, losses = {}, []
                for j, category in enumerate(categories):
                    report, label_losses = binary_report(labels[split][:, j], matrix[:, j])
                    per_category[category] = report
                    losses.append(label_losses)
                observation_losses = np.asarray(losses).mean(axis=0)
                summary.setdefault(condition, {})[split] = {
                    "macro_ce": float(observation_losses.mean()),
                    "categories": per_category,
                    "n": len(subset),
                }
                write_lines(
                    output / "results" / f"{condition}-{split}-predictions.jsonl.gz",
                    (
                        {
                            "observation_id": row.observation_id,
                            "peptide": row.peptide,
                            "split": split,
                            "group_id": row.group_id,
                            "categories": categories,
                            "labels": labels[split][i].tolist(),
                            "probabilities": matrix[i].tolist(),
                        }
                        for i, row in enumerate(subset)
                    ),
                )
                if split == "validation":
                    validation_losses[condition] = observation_losses
        groups = [row.group_id for row in rows["validation"]]
        underpowered = [
            category
            for j, category in enumerate(categories)
            if min(
                int(labels["validation"][:, j].sum()), int((1 - labels["validation"][:, j]).sum())
            )
            < 20
        ]
        comparisons = {
            baseline: paired_interval(
                validation_losses[CONDITIONS[2]] - validation_losses[baseline], groups
            )
            for baseline in CONDITIONS[:2]
        }
        screen = (
            "underpowered"
            if len(set(groups)) < 20 or underpowered
            else "passed"
            if all(value["interval_95"][1] < 0 for value in comparisons.values())
            else "failed"
        )
        report = {
            "status": "completed",
            "validation_screen": {
                "status": screen,
                "underpowered_categories": underpowered,
                "comparisons": comparisons,
            },
            "metrics": summary,
            "elapsed_seconds": time.monotonic() - started,
            "network_subprocess_seconds": network_seconds,
            "baseline_fit_seconds": baseline_seconds,
            "network_training": read_json(raw / "summary.json"),
            "peak_parent_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "peak_child_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "artifacts": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in sorted(raw.rglob("*"))
                if path.is_file()
            },
            "qualification": (
                "not_qualified; reference-membership diagnostic only; "
                "no affinity transfer comparison"
            ),
        }
        write_json(output / "results/comparison.json", report)
        print(
            json.dumps(
                {
                    "validation_screen": report["validation_screen"],
                    "macro_ce": {
                        condition: {split: value["macro_ce"] for split, value in splits.items()}
                        for condition, splits in summary.items()
                    },
                    "elapsed_seconds": report["elapsed_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            output / "results/fit-failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
