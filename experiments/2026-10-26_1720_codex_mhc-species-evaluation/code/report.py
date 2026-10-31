"""Report only saved predictions; no checkpoint loading or test rescoring."""

from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import json
import sys

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch

from common import CONDITIONS, FAMILY, guarded, write

SPECIES = ("human", "mouse", "rat", "rhesus_macaque")


def records(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def arrays(rows):
    y = np.asarray([r["target"] for r in rows], int)
    p = np.asarray([r["probabilities"] for r in rows], float)
    if (
        p.shape != (len(y), 4)
        or not len(y)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1, atol=2e-7)
    ):
        raise ValueError("invalid probability matrix")
    if not np.isin(y, range(4)).all():
        raise ValueError("invalid species index")
    losses = -np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(float).tiny))
    return y, p, losses


def grouped_values(rows, values):
    groups = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        groups[row["group_id"]].append(float(value))
    return {g: float(np.mean(v)) for g, v in sorted(groups.items())}


def metrics(rows):
    y, p, loss = arrays(rows)
    call = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, call, labels=range(4), zero_division=0
    )
    by_species = {}
    for j, name in enumerate(SPECIES):
        mask = y == j
        by_species[name] = dict(
            n=int(mask.sum()),
            components=len({r["group_id"] for r, yes in zip(rows, mask) if yes}),
            precision=float(precision[j]),
            recall=float(recall[j]) if mask.any() else None,
            f1=float(f1[j]) if mask.any() else None,
            auroc=float(roc_auc_score(mask, p[:, j])) if 0 < mask.sum() < len(y) else None,
            auprc=float(average_precision_score(mask, p[:, j])) if mask.any() else None,
        )
    confidence, correct = p.max(1), call == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    calibration = []
    for j in range(10):
        mask = bins == j
        calibration.append(
            dict(
                bin=j,
                n=int(mask.sum()),
                confidence=float(confidence[mask].mean()) if mask.any() else None,
                accuracy=float(correct[mask].mean()) if mask.any() else None,
            )
        )
    group_ce, group_acc = grouped_values(rows, loss), grouped_values(rows, correct)
    return dict(
        n=len(y),
        components=len(group_ce),
        ce=float(loss.mean()),
        accuracy=float(correct.mean()),
        balanced_accuracy=float(recall[support > 0].mean()),
        macro_f1=float(f1[support > 0].mean()),
        per_species=by_species,
        confusion=confusion_matrix(y, call, labels=range(4)).tolist(),
        brier=float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1))),
        ece_10_bins=float(
            sum(
                b["n"] / len(y) * abs(b["accuracy"] - b["confidence"])
                for b in calibration
                if b["n"]
            )
        ),
        calibration=calibration,
        probability_sd=p.std(0).tolist(),
        component_mean_ce=float(np.mean(list(group_ce.values()))),
        component_mean_accuracy=float(np.mean(list(group_acc.values()))),
    )


def bootstrap(network, control):
    lookup = {r["observation_id"]: r for r in control}
    ordered = [lookup[r["observation_id"]] for r in network]
    if len(lookup) != len(network) or any(
        a["group_id"] != b["group_id"] or a["target"] != b["target"]
        for a, b in zip(network, ordered)
    ):
        raise ValueError("paired control mismatch")
    difference = arrays(network)[2] - arrays(ordered)[2]
    values = np.asarray(list(grouped_values(network, difference).values()))
    rng = np.random.default_rng(20261026)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
    return dict(
        component_mean_difference=float(values.mean()),
        interval_95=np.quantile(draws, [0.025, 0.975]).tolist(),
        components=len(values),
        resamples=2000,
        seed=20261026,
    )


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with guarded("report"):
        result = {
            "species": SPECIES,
            "metrics": {},
            "screens": {},
            "status": "incomplete",
            "qualification": (
                "not_qualified; single-domain reference diagnostic; no transfer evaluation"
            ),
        }
        paths = {
            "prior": FAMILY / "results/baselines",
            "composition": FAMILY / "results/baselines",
            **{c: FAMILY / "results" / c for c in CONDITIONS},
        }
        validation = {}
        table = []
        for condition, directory in paths.items():
            prefix = condition + "-" if condition in ("prior", "composition") else ""
            if not (directory / f"{prefix}test-predictions.jsonl.gz").exists():
                continue
            result["metrics"][condition] = {}
            for split in ("train", "validation", "test"):
                rows = records(directory / f"{prefix}{split}-predictions.jsonl.gz")
                measure = metrics(rows)
                measure["by_slot"] = {
                    slot: metrics(subset)
                    for slot in ("mhc1", "mhc2")
                    if (subset := [r for r in rows if r["identity_task"] == slot])
                }
                result["metrics"][condition][split] = measure
                table.append(
                    dict(
                        condition=condition,
                        split=split,
                        **{
                            k: measure[k]
                            for k in (
                                "n",
                                "components",
                                "ce",
                                "accuracy",
                                "balanced_accuracy",
                                "macro_f1",
                                "component_mean_ce",
                                "component_mean_accuracy",
                            )
                        },
                    )
                )
                if split == "validation":
                    validation[condition] = rows
            if condition in CONDITIONS:
                summary = json.loads((directory / "summary.json").read_text())
                check_path = directory / "independent-verification.json"
                verified = (
                    check_path.exists() and json.loads(check_path.read_text())["status"] == "passed"
                )
                m = result["metrics"][condition]["validation"]
                comparisons = {
                    b: bootstrap(validation[condition], validation[b])
                    for b in ("prior", "composition")
                }
                support = all(
                    v["n"] >= 20 and v["components"] >= 10 for v in m["per_species"].values()
                )
                passed = (
                    verified
                    and summary["coverage_gate"]["status"] == "passed"
                    and support
                    and (
                        result["metrics"][condition]["train"]["ce"]
                        < result["metrics"]["prior"]["train"]["ce"]
                        and m["balanced_accuracy"] >= 0.8
                        and all(v["recall"] >= 0.5 for v in m["per_species"].values())
                        and all(v["interval_95"][1] < 0 for v in comparisons.values())
                    )
                )
                result["screens"][condition] = dict(
                    status="passed" if passed else "failed",
                    source_verification=verified,
                    support_minimum_passed=support,
                    comparisons=comparisons,
                    selected_epoch=summary["selected_epoch"],
                    runtime_seconds=summary["elapsed_seconds"],
                )
        if all(c in result["screens"] for c in CONDITIONS):
            result["status"] = "completed"
            result["engineering_screen"] = (
                "passed"
                if all(v["status"] == "passed" for v in result["screens"].values())
                else "failed"
            )
            result["between_seed_validation_balanced_accuracy_sd"] = float(
                np.std(
                    [result["metrics"][c]["validation"]["balanced_accuracy"] for c in CONDITIONS]
                )
            )
        write(FAMILY / "results/comparison.json", result)
        if table:
            with (FAMILY / "results/comparison.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "conditions": list(result["metrics"]),
                    "screens": result["screens"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("report reads saved outputs and takes no arguments")
    main()
