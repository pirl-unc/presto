"""Unclipped target-space metrics and one record per measured outcome."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import math
from typing import Any

import numpy as np
import torch

from ..data.modular import endpoint_spec, observation_target
from .modular import identity_targets, observation_loss


def _plain(value):
    """Detach report-only diagnostics without changing model or loss tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def _correlation(left, right):
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def binary_metrics(labels, scores, decisions):
    labels, scores, decisions = (
        np.asarray(labels, bool),
        np.asarray(scores),
        np.asarray(decisions, bool),
    )
    tp, tn = int(np.sum(labels & decisions)), int(np.sum(~labels & ~decisions))
    fp, fn = int(np.sum(~labels & decisions)), int(np.sum(labels & ~decisions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    result = {
        "n": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auroc": None,
        "auprc": None,
        "balanced_accuracy": None,
    }
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives and negatives:
        result["balanced_accuracy"] = (recall + tn / negatives) / 2
        result["auroc"] = float(
            (_ranks(scores)[labels].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
    if positives:
        order = np.argsort(-scores, kind="stable")
        ordered_scores, ordered_labels = scores[order], labels[order]
        cumulative = np.cumsum(ordered_labels)
        ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]
        recall_points = cumulative[ends] / positives
        precision_points = cumulative[ends] / (ends + 1)
        result["auprc"] = float(np.sum(np.diff(np.r_[0, recall_points]) * precision_points))
    return result


def regression_metrics(targets, predictions, *, binding: bool):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    result = {
        "n": len(targets),
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "pearson": _correlation(targets, predictions),
        "spearman": _correlation(_ranks(targets), _ranks(predictions)),
        "prediction_sd": float(np.std(predictions)),
        "target_sd": float(np.std(targets)),
        "prediction_range": [float(np.min(predictions)), float(np.max(predictions))],
        "prediction_quantiles": np.quantile(predictions, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "sd_ratio": float(np.std(predictions) / np.std(targets)) if np.std(targets) else None,
    }
    if binding:
        result["binding_at_500_nM"] = binary_metrics(
            targets <= math.log10(500), -predictions, predictions <= math.log10(500)
        )
    return result


def evaluate(model, index, indices, scales, *, batch_size=64):
    """Returns serializable predictions, including exact/censored targets and identity labels."""
    was_training = model.training
    model.eval()
    records = []
    try:
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                rows = [index[i] for i in indices[start : start + batch_size]]
                outputs = model(rows)
                for row, output in zip(rows, outputs, strict=True):
                    base = {
                        "observation_id": row.observation_id,
                        "endpoint": row.endpoint,
                        "source": row.source,
                        "group_id": row.group_id,
                        "split": row.split,
                        "peptide": row.peptide,
                        "qualifier": row.qualifier,
                        "mhc_classes": [x.class_override or x.mhc_class for x in row.mhc],
                        "observation": row.to_dict(),
                        "diagnostics": _plain(output.diagnostics),
                        "component_scores": _plain(output.components),
                    }
                    if row.endpoint == "identity":
                        for name, target in identity_targets(row, model).items():
                            probabilities = output.identity[name].softmax(-1).cpu().tolist()
                            loss = -output.identity[name].log_softmax(-1)[target]
                            records.append(
                                {
                                    **base,
                                    "identity_task": name,
                                    "target": target,
                                    "probabilities": probabilities,
                                    "prediction": int(np.argmax(probabilities)),
                                    "loss": float(loss),
                                }
                            )
                    else:
                        loss = observation_loss(output, row, scales)
                        value = float(output.value)
                        spec = endpoint_spec(row.endpoint)
                        if not spec.binary and not spec.quantitative:
                            value = scales[row.endpoint].decode(value)
                        records.append(
                            {
                                **base,
                                "prediction": value,
                                "target": observation_target(row),
                                "raw_value": row.value,
                                "raw_unit": row.unit,
                                "target_space": "logit"
                                if spec.binary
                                else row.quantity_contract.target_space
                                if spec.quantitative
                                else (f"log10({spec.unit})" if spec.log10 else spec.unit),
                                "quantity_contract": asdict(row.quantity_contract)
                                if spec.quantitative
                                else None,
                                "value_encoding": row.value_encoding,
                                "loss": float(loss),
                            }
                        )
    finally:
        model.train(was_training)
    return summarize(records), records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("identity_task", record["endpoint"])].append(record)
    result = {}
    for endpoint, rows in grouped.items():
        metrics = {"n": len(rows), "loss": float(np.mean([row["loss"] for row in rows]))}
        if "identity_task" in rows[0]:
            metrics["accuracy"] = sum(row["target"] == row["prediction"] for row in rows) / len(
                rows
            )
        elif endpoint_spec(endpoint).binary:
            metrics.update(
                binary_metrics(
                    [row["target"] for row in rows],
                    [row["prediction"] for row in rows],
                    [row["prediction"] >= 0 for row in rows],
                )
            )
        else:
            exact = [row for row in rows if row["qualifier"] == "="]
            if exact:
                metrics["exact"] = regression_metrics(
                    [row["target"] for row in exact],
                    [row["prediction"] for row in exact],
                    binding=not endpoint_spec(endpoint).quantitative
                    and endpoint_spec(endpoint).unit == "nM",
                )
                metrics["by_class"] = {}
                for mhc_class in ("I", "II"):
                    subset = [row for row in exact if row["mhc_classes"] == [mhc_class]]
                    if subset:
                        metrics["by_class"][mhc_class] = regression_metrics(
                            [row["target"] for row in subset],
                            [row["prediction"] for row in subset],
                            binding=not endpoint_spec(endpoint).quantitative
                            and endpoint_spec(endpoint).unit == "nM",
                        )
            bounds = [row for row in rows if row["qualifier"] != "="]
            if bounds:
                metrics["censored"] = {
                    "n": len(bounds),
                    "satisfaction": sum(
                        row["prediction"] <= row["target"]
                        if row["qualifier"] in ("<", "<=")
                        else row["prediction"] >= row["target"]
                        for row in bounds
                    )
                    / len(bounds),
                }
        result[endpoint] = metrics
    return result
