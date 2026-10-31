"""Reconstruct the canonical batch reductions from persisted prediction records."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def evaluation_receipt(batch_index, total_loss, losses, metrics):
    """Capture the actual evaluation weights, supports and regularization terms."""
    support = {
        name.removeprefix("batch_support_"): float(value)
        for name, value in metrics.items()
        if name.startswith("batch_support_")
    }
    return {
        "batch_index": batch_index,
        "overall_loss": float(total_loss),
        "supervised_losses": {
            name: float(value) for name, value in losses.items() if name in support
        },
        "support": support,
        "task_weights": {
            name.removeprefix("batch_supervised_weight_"): float(value)
            for name, value in metrics.items()
            if name.startswith("batch_supervised_weight_")
        },
        "regularizers": {
            name: float(value) for name, value in losses.items() if name not in support
        },
    }


def _close(actual, expected, context):
    if (
        not math.isfinite(actual)
        or not math.isfinite(expected)
        or not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-6)
    ):
        raise ValueError(f"{context}: reconstructed {actual} != expected {expected}")


def _same_mapping(actual, expected, context):
    if actual.keys() != expected.keys():
        raise ValueError(f"{context}: different tasks {sorted(actual)} != {sorted(expected)}")
    for name, value in actual.items():
        _close(value, expected[name], f"{context}:{name}")


def _observation_loss(row):
    true, pred = float(row["y_true"]), float(row["y_pred"])
    if not math.isfinite(true) or not math.isfinite(pred):
        raise ValueError("nonfinite prediction or target in prediction artifact")
    kind = row["loss_type"]
    if kind == "bce":
        _close(float(row["y_prob"]), 1 / (1 + math.exp(-max(-700, pred))), "binary probability")
        return max(pred, 0) - pred * true + math.log1p(math.exp(-abs(pred)))
    if kind == "ce":
        logits = [float(value) for value in json.loads(row["class_logits"])]
        names = json.loads(row["class_names"])
        probabilities = json.loads(row["class_probabilities"])
        if not logits or len(logits) != len(names) or len(logits) != len(probabilities):
            raise ValueError("categorical artifact has inconsistent class vectors")
        if true != int(true) or not 0 <= int(true) < len(logits):
            raise ValueError("categorical artifact has an invalid target class")
        maximum = max(logits)
        if not math.isfinite(maximum):
            raise ValueError("categorical artifact has no finite prediction")
        normalizer = sum(math.exp(value - maximum) for value in logits)
        for logit, probability in zip(logits, probabilities):
            _close(float(probability), math.exp(logit - maximum) / normalizer, "class probability")
        predicted = logits.index(maximum)
        if (
            pred != predicted
            or row["target_class_name"] != names[int(true)]
            or row["predicted_class_name"] != names[predicted]
        ):
            raise ValueError("categorical artifact class identity does not match its logits")
        return maximum + math.log(normalizer) - logits[int(true)]
    delta = pred - true
    if kind == "censor":
        qualifier = int(row["qualifier"])
        if qualifier not in (-1, 0, 1):
            raise ValueError("prediction artifact has an invalid censor qualifier")
        delta = max(delta, 0) if qualifier == -1 else min(delta, 0) if qualifier == 1 else delta
    elif kind != "mse":
        raise ValueError(f"unknown observation loss {kind}")
    return delta * delta


def reconcile_prediction_artifact(path, collection, expected_batches=None):
    """Read the CSV back, validate identities/counts, and recompute every objective.

    Panel axes share a loss group. Their record weights include the mean across
    axes; supports retain the original one-axis count. Canonical evaluation
    averages task losses over batches where present and total loss over all
    batches. Regularization is preserved in the receipt, not inferred from rows.
    """
    n_batches = len(collection.batch_losses)
    if expected_batches is not None and len(expected_batches) != n_batches:
        raise ValueError("canonical evaluation and prediction artifact have different batch counts")
    losses = [defaultdict(float) for _ in range(n_batches)]
    counts = [defaultdict(int) for _ in range(n_batches)]
    weights = [defaultdict(float) for _ in range(n_batches)]
    groups = {}
    identities = set()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "task",
            "batch_index",
            "y_true",
            "y_pred",
            "observation_loss",
            "loss_reduction_weight",
            "observation_weight",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("prediction artifact is missing required columns")
        for row in reader:
            batch = int(row["batch_index"])
            if not 0 <= batch < n_batches:
                raise ValueError("prediction artifact has an unexpected batch")
            task, group = row["task"], row["loss_group"]
            position = (
                row["bag_index"] if row["observation_kind"] == "bag" else row["source_row_index"]
            )
            identity = (
                batch,
                task,
                row["observation_kind"],
                position,
                row["component_index"],
                row["selector_index"],
            )
            if identity in identities:
                raise ValueError(f"duplicate observation in prediction artifact: {identity}")
            identities.add(identity)
            loss = _observation_loss(row)
            _close(loss, float(row["observation_loss"]), f"observation loss {identity}")
            weight, support = float(row["loss_reduction_weight"]), float(row["observation_weight"])
            if (
                not math.isfinite(weight)
                or not math.isfinite(support)
                or weight <= 0
                or support <= 0
            ):
                raise ValueError("invalid observation weight in prediction artifact")
            losses[batch][group] += loss * weight
            counts[batch][task] += 1
            weights[batch][task] += support
            if task in groups and groups[task] != group:
                raise ValueError("prediction artifact changes the loss group for a task")
            groups[task] = group
    receipts = []
    task_values = defaultdict(list)
    for index in range(n_batches):
        if dict(counts[index]) != collection.batch_observation_counts[index]:
            raise ValueError(
                f"batch {index}: prediction observation counts do not match effective support"
            )
        support = {}
        for task, value in weights[index].items():
            group = groups[task]
            if group in support:
                _close(value, support[group], f"batch {index}: panel axis support")
            support[group] = value
        _same_mapping(
            losses[index], collection.batch_losses[index], f"batch {index}: collected loss"
        )
        _same_mapping(support, collection.batch_support[index], f"batch {index}: collected support")
        receipt = {
            "batch_index": index,
            "reconstructed_losses": dict(losses[index]),
            "support": support,
            "observation_counts": dict(counts[index]),
        }
        if expected_batches is not None:
            expected = expected_batches[index]
            if expected["batch_index"] != index:
                raise ValueError("canonical evaluation receipt has an unexpected batch index")
            _same_mapping(
                losses[index], expected["supervised_losses"], f"batch {index}: canonical loss"
            )
            _same_mapping(support, expected["support"], f"batch {index}: canonical support")
            total = sum(
                losses[index][name] * weight for name, weight in expected["task_weights"].items()
            ) + sum(expected["regularizers"].values())
            _close(total, expected["overall_loss"], f"batch {index}: overall loss")
            receipt.update(canonical=expected, reconstructed_overall_loss=total)
        receipts.append(receipt)
        for name, value in losses[index].items():
            task_values[name].append(value)
    return {
        "schema_version": 1,
        "verified": True,
        "canonical_evaluation_compared": expected_batches is not None,
        "reduction": (
            "masked means; panel axes averaged; task means over present batches; "
            "overall mean over all batches"
        ),
        "batches": receipts,
        "task_losses": {name: sum(values) / len(values) for name, values in task_values.items()},
        "overall_loss": sum(item["reconstructed_overall_loss"] for item in receipts)
        / max(n_batches, 1)
        if expected_batches is not None
        else None,
    }
