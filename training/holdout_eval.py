"""Generic held-out evaluation for any supervised task.

Until now the only path that produced held-out AUROC/AUPRC/Spearman and
per-example prediction dumps was ``scripts/focused_binding_probe.py``, which is
hardcoded to binding affinity. The unified trainer emitted scalar losses only.
That is why a question about elution AUPRC could not be answered by the
canonical trainer, and why every March 2026 experiment ran the probe script
instead.

This module closes that gap by deriving the metric family from the loss type
already declared in the task registry (``TaskLossSpec.loss_type`` in
``scripts/train_synthetic.py``):

- ``bce``            -> binary metrics (AUROC, AUPRC, F1, balanced accuracy)
- ``mse`` / ``censor`` -> regression metrics (Spearman, Pearson, RMSE)
- ``ce``             -> accuracy

So a task added to the registry gets held-out metrics with no extra wiring, and
the metric choice cannot drift out of step with the loss.

No scikit-learn dependency: the estimators here are small and exact, and Presto
otherwise needs only torch/numpy.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

# A peptide:MHC pair at or under 500 nM is the conventional binder cutoff.
DEFAULT_BINDING_THRESHOLD_NM = 500.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (the Spearman/AUROC convention)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average tied ranks
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    return pearson(_rankdata(y_true), _rankdata(y_pred))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    true_centered = y_true - y_true.mean()
    pred_centered = y_pred - y_pred.mean()
    denominator = math.sqrt(float((true_centered**2).sum() * (pred_centered**2).sum()))
    if denominator <= 0:
        return None
    return float((true_centered * pred_centered).sum() / denominator)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Rank-statistic AUROC (equivalent to the Mann-Whitney U form)."""
    positives = y_true > 0.5
    n_pos = int(positives.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(y_score)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Average precision — the step-wise sum, not trapezoidal interpolation."""
    positives = y_true > 0.5
    n_pos = int(positives.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return None
    order = np.argsort(-y_score, kind="mergesort")
    hits = positives[order].astype(float)
    cumulative_hits = np.cumsum(hits)
    precision = cumulative_hits / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_pos)


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    predicted = y_score >= threshold
    actual = y_true > 0.5
    true_pos = float((predicted & actual).sum())
    false_pos = float((predicted & ~actual).sum())
    false_neg = float((~predicted & actual).sum())
    true_neg = float((~predicted & ~actual).sum())

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    metrics: Dict[str, float] = {
        "accuracy": (true_pos + true_neg) / max(len(y_true), 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_positive": float(actual.sum()),
        "n": float(len(y_true)),
    }
    area_under_roc = auroc(y_true, y_score)
    if area_under_roc is not None:
        metrics["auroc"] = area_under_roc
    area_under_pr = auprc(y_true, y_score)
    if area_under_pr is not None:
        metrics["auprc"] = area_under_pr
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "n": float(len(y_true)),
    }
    rho = spearman(y_true, y_pred)
    if rho is not None:
        metrics["spearman"] = rho
    r = pearson(y_true, y_pred)
    if r is not None:
        metrics["pearson"] = r
    return metrics


def metrics_for_loss_type(
    loss_type: str, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Choose the metric family from the task's declared loss type.

    Deriving this from the registry rather than a second hand-maintained table
    is what keeps a newly added task from silently getting no held-out metrics.
    """
    if len(y_true) == 0:
        return {}
    if loss_type == "bce":
        # Predictions arrive as logits; map to probability for thresholding.
        probabilities = 1.0 / (1.0 + np.exp(-y_pred))
        return binary_metrics(y_true, probabilities)
    if loss_type == "ce":
        return {
            "accuracy": float(np.mean(y_true == y_pred)),
            "n": float(len(y_true)),
        }
    return regression_metrics(y_true, y_pred)


def _logistic(value: float) -> float:
    """Numerically stable sigmoid for a single scalar."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class TaskPredictionAccumulator:
    """Collects masked per-example predictions for one task across batches."""

    def __init__(self, task_name: str, loss_type: str):
        self.task_name = task_name
        self.loss_type = loss_type
        self._true: List[float] = []
        self._pred: List[float] = []
        self._sample_ids: List[str] = []
        self._sources: List[str] = []

    def add(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        mask: Sequence[float],
        sample_ids: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
    ) -> None:
        for position, keep in enumerate(mask):
            if float(keep) <= 0.0:
                continue
            self._true.append(float(y_true[position]))
            self._pred.append(float(y_pred[position]))
            if sample_ids is not None and position < len(sample_ids):
                self._sample_ids.append(str(sample_ids[position]))
            else:
                self._sample_ids.append("")
            if sources is not None and position < len(sources):
                self._sources.append(str(sources[position]))
            else:
                self._sources.append("")

    def __len__(self) -> int:
        return len(self._true)

    def metrics(self) -> Dict[str, float]:
        """Overall metrics, plus a breakdown by what the negatives actually are.

        A binary metric computed over real positives and *synthetic* negatives
        measures whether a peptide looks real, not whether it is presented.
        Mixing them produced AUPRC 1.0000 on 18,324 elution rows with zero
        overlap between the two score distributions -- a number that reads as a
        solved task and is a sanity check.

        So three views are reported when synthetic negatives are present:

        ``<task>``            everything, as before
        ``real_only``         real positives vs real negatives -- the honest
                              number, and empty when the corpus supplies no
                              real negatives, which is itself worth seeing
        ``decoy_<kind>``      real positives vs one decoy family, explicitly
                              labelled as decoy detection rather than biology
        """
        true_all = np.asarray(self._true)
        pred_all = np.asarray(self._pred)
        summary = metrics_for_loss_type(self.loss_type, true_all, pred_all)
        if self.loss_type != "bce" or not self._sources:
            return summary

        sources = np.asarray(self._sources, dtype=object)
        is_synthetic = np.asarray(
            [str(src).startswith("synthetic_negative") for src in sources]
        )
        if not is_synthetic.any():
            return summary

        positives = true_all >= 0.5
        real_mask = ~is_synthetic
        if real_mask.sum() > 0 and len(np.unique(true_all[real_mask])) > 1:
            for name, value in metrics_for_loss_type(
                self.loss_type, true_all[real_mask], pred_all[real_mask]
            ).items():
                summary[f"real_only_{name}"] = value
        else:
            # Stated explicitly: "no real negatives" is a property of the
            # corpus that a reader must see, not an absence to skim past.
            summary["real_only_n_negatives"] = float(
                int((real_mask & ~positives).sum())
            )

        for kind in sorted({str(src) for src in sources[is_synthetic]}):
            selector = positives | (sources == kind)
            if len(np.unique(true_all[selector])) < 2:
                continue
            label = kind.replace("synthetic_negative_", "")
            for name, value in metrics_for_loss_type(
                self.loss_type, true_all[selector], pred_all[selector]
            ).items():
                summary[f"decoy_{label}_{name}"] = value
        return summary

    def rows(self) -> List[Dict[str, Any]]:
        """Per-example rows, for the prediction dumps the experiment contract wants.

        `y_pred` is the model's raw output, which for a `bce` task is a
        **logit**, not a probability -- it ranges over the reals. The metrics
        apply the logistic transform themselves before thresholding, so AUPRC
        and friends are computed correctly, but a reader of this CSV has no way
        to know that from the column alone. `y_prob` carries the transformed
        value for binary tasks so the dump is self-describing, and is empty for
        regression tasks where no such transform applies.

        Without it, a plausible-looking calibration or Brier computation over
        this file would be silently wrong.
        """
        is_binary = self.loss_type == "bce"
        return [
            {
                "task": self.task_name,
                "sample_id": sample_id,
                "y_true": true_value,
                "y_pred": pred_value,
                "y_prob": _logistic(pred_value) if is_binary else "",
            }
            for sample_id, true_value, pred_value in zip(
                self._sample_ids, self._true, self._pred
            )
        ]


def summarize_accumulators(
    accumulators: Iterable[TaskPredictionAccumulator],
) -> Dict[str, Dict[str, float]]:
    """Per-task metric blocks, skipping tasks with no held-out examples."""
    summary: Dict[str, Dict[str, float]] = {}
    for accumulator in accumulators:
        if len(accumulator) == 0:
            continue
        summary[accumulator.task_name] = accumulator.metrics()
    return summary


def flatten_summary(summary: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Flatten to ``<task>_<metric>`` so it drops straight into epoch_metrics.csv."""
    flat: Dict[str, float] = {}
    for task_name, metrics in summary.items():
        for metric_name, value in metrics.items():
            flat[f"{task_name}_{metric_name}"] = float(value)
    return flat


def collect_holdout_predictions(
    model,
    loader,
    device,
    specs: Sequence[Any],
    forward_fn,
    resolve_pred_fn,
    get_target_fn,
    get_mask_fn,
    max_batches: int = 0,
) -> Dict[str, TaskPredictionAccumulator]:
    """Run a held-out pass and collect masked predictions for every task.

    The callables are injected rather than imported so this module stays free of
    a dependency on the training scripts (which import it): ``forward_fn(model,
    batch)`` returns the outputs dict, and the three resolver callables are the
    same ones the loss loop uses, so metrics are computed on exactly the
    tensors that were supervised.
    """
    import torch

    accumulators: Dict[str, TaskPredictionAccumulator] = {
        spec.name: TaskPredictionAccumulator(spec.name, spec.loss_type) for spec in specs
    }

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            moved = batch.to(device) if hasattr(batch, "to") else batch
            outputs = forward_fn(model, moved)
            sample_ids = getattr(moved, "sample_ids", None)
            for spec in specs:
                target = get_target_fn(moved, spec)
                mask = get_mask_fn(moved, spec)
                if target is None or mask is None:
                    continue
                pred = resolve_pred_fn(outputs, spec.pred_paths)
                if pred is None:
                    continue
                # Apply the same target transform the loss uses. Without this
                # the dump compares raw targets (e.g. nM) against predictions in
                # the transformed space, and for the mhcflurry-style affinity
                # encoding -- which inverts, so higher means stronger binder --
                # that flips the sign of every correlation.
                target_tensor = target.reshape(-1).float()
                if getattr(spec, "target_transform", None) is not None:
                    target_tensor = spec.target_transform(target_tensor)
                target_vec = target_tensor.detach().cpu().numpy()
                mask_vec = mask.reshape(-1).float().detach().cpu().numpy()
                pred_vec = pred.reshape(-1).float().detach().cpu().numpy()
                if len(pred_vec) != len(target_vec):
                    # Multi-output heads (e.g. CE logits) do not line up
                    # element-wise; those tasks are scored by loss only.
                    continue
                accumulators[spec.name].add(
                    target_vec,
                    pred_vec,
                    mask_vec,
                    sample_ids,
                    getattr(moved, "sample_sources", None),
                )
    return accumulators


def write_holdout_artifacts(
    out_dir,
    accumulators: Mapping[str, TaskPredictionAccumulator],
    split: str = "val",
    extra_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write ``summary.json`` and ``<split>_predictions.csv`` for a held-out pass.

    This is the artifact contract the experiment registry expects and that the
    unified trainer previously did not produce.
    """
    import csv
    import json
    from pathlib import Path

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = summarize_accumulators(accumulators.values())
    payload: Dict[str, Any] = {"split": split, "tasks": summary}
    if extra_summary:
        payload.update(dict(extra_summary))
    (out_path / "summary.json").write_text(json.dumps(payload, indent=2))

    # Flat metric CSV written here rather than through RunLogger: the logger is
    # closed in the trainer's `finally` block, so anything logged afterwards
    # hits a closed file. Keeping the artifact self-contained avoids coupling
    # this pass to that lifecycle.
    flat = flatten_summary(summary)
    if flat:
        with (out_path / f"{split}_metrics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "metric", "value"])
            for metric_name, value in sorted(flat.items()):
                writer.writerow([split, metric_name, value])

    rows: List[Dict[str, Any]] = []
    for accumulator in accumulators.values():
        rows.extend(accumulator.rows())
    predictions_path = out_path / f"{split}_predictions.csv"
    if rows:
        with predictions_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["task", "sample_id", "y_true", "y_pred", "y_prob"])
            writer.writeheader()
            writer.writerows(rows)
    return payload
