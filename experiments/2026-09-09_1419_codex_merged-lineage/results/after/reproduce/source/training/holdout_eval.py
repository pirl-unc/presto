"""Generic held-out evaluation for any supervised task.

Until now the only path that produced held-out AUROC/AUPRC/Spearman and
per-example prediction dumps was ``scripts/focused_binding_probe.py``, which is
hardcoded to binding affinity. The unified trainer emitted scalar losses only.
That is why a question about elution AUPRC could not be answered by the
canonical trainer, and why every March 2026 experiment ran the probe script
instead.

This module closes that gap by deriving the metric family from the loss type
already declared in the shared task registry (``TaskLossSpec.loss_type`` in
``training/supervision.py``):

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

from presto.data.source_lineage import SOURCE_LINEAGE_FIELDS

# A peptide:MHC pair at or under 500 nM is the conventional binder cutoff.
DEFAULT_BINDING_THRESHOLD_NM = 500.0

# Persist this with held-out artifacts so old row-wise AP can be distinguished
# from the corrected threshold-group estimator without guessing from values.
AUPRC_ESTIMATOR = "average_precision_distinct_thresholds_v2"

PREDICTION_LINEAGE_FIELDS = (
    "peptide",
    "source_mhc_alleles",
    "resolved_mhc_alleles",
    *SOURCE_LINEAGE_FIELDS,
    "source_sample_label",
    "source_sample_attribution",
    "mapping_gene_name",
    "mapping_gene_id",
    "mapping_protein_id",
    "mapping_transcript_id",
    "mapping_position",
    "mapping_proteome",
    "mapping_proteome_source",
    "mapping_is_canonical_transcript",
)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (the Spearman/AUROC convention).

    Tie-averaging is vectorized rather than looped. The held-out summary now
    recomputes this family per task, per synthetic-decoy kind, and per mapping
    category -- tens of passes over the full split -- so an interpreted
    per-element loop here set the cost of the whole evaluation.
    """
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    # Start of each run of equal values, in sorted order.
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    np.not_equal(sorted_values[1:], sorted_values[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    ends = np.append(starts[1:], n)
    # A run covering sorted positions [s, e) holds 1-based ranks s+1..e, whose
    # mean is (s + e + 1) / 2. `cumsum(is_start) - 1` maps each position to its
    # run, so every tied element gets that mean in one scatter.
    group_mean = (starts + ends + 1) / 2.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = group_mean[np.cumsum(is_start) - 1]
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
    """Non-interpolated average precision at distinct score thresholds.

    All observations sharing a score enter together. Counting individual hits
    within a tie makes AP depend on input order, even for a constant predictor.
    As with AUROC, retain the reporting policy of omitting one-class results.
    """
    positives = y_true > 0.5
    n_pos = int(positives.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return None
    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    # Inclusive final position of each score group, including the last group.
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(order) - 1]
    cumulative_hits = np.cumsum(positives[order])[ends]
    precision = cumulative_hits / (ends + 1)
    new_hits = np.diff(np.r_[0, cumulative_hits])
    return float(np.sum(precision * new_hits) / n_pos)


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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binding_threshold_metrics(
    y_true_log10: np.ndarray,
    y_pred_log10: np.ndarray,
    qualifiers: np.ndarray,
    *,
    threshold_nm: float = 500.0,
) -> Dict[str, float]:
    """Qualifier-aware binding classification at an affinity threshold.

    Exact observations are always usable. A ``<=`` observation is definite
    only when its bound is already at or below the threshold; a ``>``
    observation is definite only when its bound is above it. The remaining
    censored rows do not determine a class and are excluded.
    """
    cutoff = math.log10(float(threshold_nm))
    exact = qualifiers == 0
    definite_binder = (qualifiers < 0) & (y_true_log10 <= cutoff)
    definite_nonbinder = (qualifiers > 0) & (y_true_log10 > cutoff)
    usable = exact | definite_binder | definite_nonbinder
    if not usable.any():
        return {}

    labels = y_true_log10[usable] <= cutoff
    predicted = y_pred_log10[usable] <= cutoff
    tp = int(np.sum(predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    metrics: Dict[str, float] = {
        "accuracy": _safe_div(tp + tn, len(labels)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "n": float(len(labels)),
    }
    # Lower log10(nM) means stronger binding, so negate for ranking metrics.
    area_under_roc = auroc(labels.astype(float), -y_pred_log10[usable])
    area_under_pr = auprc(labels.astype(float), -y_pred_log10[usable])
    if area_under_roc is not None:
        metrics["auroc"] = area_under_roc
    if area_under_pr is not None:
        metrics["auprc"] = area_under_pr
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


# Additional identity for effective bag observations. Empty values on legacy
# row records preserve their established schema/meaning.
MIL_OBSERVATION_FIELDS = (
    "observation_kind",
    "batch_index",
    "bag_index",
    "bag_id",
    "source_row_index",
    "instance_count",
    "evaluated_instance_count",
    "bag_instance_indices",
    "output_path",
    "selector_axis",
    "selector_index",
    "selector_name",
    "observation_loss",
    "loss_group",
    "loss_type",
    "loss_reduction_weight",
    "observation_weight",
    "raw_target",
    "raw_unit",
    "target_unit",
    "prediction_unit",
    "component_index",
    "component_name",
    "class_names",
    "class_logits",
    "class_probabilities",
    "target_class_name",
    "predicted_class_name",
    "canonical_output",
)


class PredictionCollection(dict):
    """Accumulators plus the expected per-batch loss/support for artifact closure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_losses = []
        self.batch_support = []
        self.batch_observation_counts = []
        self.output_support = {}


def _declare_output_support(collection, specs):
    for spec in specs:
        collection.output_support.setdefault(
            spec.name,
            {
                "target_observations": 0,
                "exported_observations": 0,
                "missing_output_observations": 0,
                "alias_of": "elution" if spec.name == "ms" else "",
                "columns": {
                    name: {"target": 0, "exported": 0} for name in getattr(spec, "columns", ())
                },
                "components": {
                    name: {"target": 0, "exported": 0}
                    for name in getattr(spec, "component_names", ())
                },
            },
        )


def _record_output_support(collection, targets, predictions):
    for name, target in targets.items():
        entry = collection.output_support[name]
        exported = name in predictions
        entry["target_observations"] += target.support
        entry["exported_observations"] += target.support if exported else 0
        entry["missing_output_observations"] += 0 if exported else target.support
        for field, indices in (
            ("columns", target.selectors),
            ("components", getattr(target, "components", None)),
        ):
            if indices is not None:
                for index, counts in enumerate(entry[field].values()):
                    count = int(((target.mask > 0) & (indices == index)).sum())
                    counts["target"] += count
                    counts["exported"] += count if exported else 0


class TaskPredictionAccumulator:
    """Collects masked per-example predictions for one task across batches."""

    def __init__(self, task_name: str, loss_type: str):
        self.task_name = task_name
        self.loss_type = loss_type
        self._true: List[float] = []
        self._pred: List[float] = []
        self._sample_ids: List[str] = []
        self._sources: List[str] = []
        self._qualifiers: List[int] = []
        self._mapping_categories: List[str] = []
        self._mapping_n_candidates: List[int] = []
        self._mapping_n_genes: List[int] = []
        self._mapping_n_flank_pairs: List[int] = []
        self._flank_context_resolved: List[bool] = []
        self._lineage: List[Dict[str, Any]] = []
        self._observations: List[Dict[str, Any]] = []

    def add(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        mask: Sequence[float],
        sample_ids: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
        mapping_categories: Optional[Sequence[str]] = None,
        mapping_n_candidates: Optional[Sequence[int]] = None,
        mapping_n_genes: Optional[Sequence[int]] = None,
        mapping_n_flank_pairs: Optional[Sequence[int]] = None,
        flank_context_resolved: Optional[Sequence[bool]] = None,
        qualifiers: Optional[Sequence[int]] = None,
        lineage: Optional[Mapping[str, Sequence[Any]]] = None,
        observations: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        for position, keep in enumerate(mask):
            if float(keep) <= 0.0:
                continue
            observation = (
                observations[position] if observations is not None else {"observation_kind": "row"}
            )
            self._observations.append(
                {field: observation.get(field, "") for field in MIL_OBSERVATION_FIELDS}
            )
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
            if qualifiers is not None and position < len(qualifiers):
                self._qualifiers.append(int(qualifiers[position]))
            else:
                self._qualifiers.append(0)
            if mapping_categories is not None and position < len(mapping_categories):
                self._mapping_categories.append(str(mapping_categories[position]))
            else:
                self._mapping_categories.append("")
            for values, destination in (
                (mapping_n_candidates, self._mapping_n_candidates),
                (mapping_n_genes, self._mapping_n_genes),
                (mapping_n_flank_pairs, self._mapping_n_flank_pairs),
            ):
                if values is not None and position < len(values):
                    destination.append(int(values[position]))
                else:
                    destination.append(0)
            if flank_context_resolved is not None and position < len(flank_context_resolved):
                self._flank_context_resolved.append(bool(flank_context_resolved[position]))
            else:
                self._flank_context_resolved.append(False)
            self._lineage.append(
                {
                    field: (
                        values[position]
                        if (values := (lineage or {}).get(field)) is not None
                        and position < len(values)
                        else ""
                    )
                    for field in PREDICTION_LINEAGE_FIELDS
                }
            )

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

        qualifiers = np.asarray(self._qualifiers, dtype=int)
        if self.loss_type == "censor" and len(qualifiers) == len(true_all):
            exact = qualifiers == 0
            if exact.any():
                for name, value in regression_metrics(true_all[exact], pred_all[exact]).items():
                    summary[f"exact_{name}"] = value
            if self.task_name.startswith("binding"):
                for name, value in binding_threshold_metrics(
                    true_all, pred_all, qualifiers, threshold_nm=500.0
                ).items():
                    summary[f"threshold_500nm_{name}"] = value

        if self.loss_type == "bce" and self._sources:
            sources = np.asarray(self._sources, dtype=object)
            is_synthetic = np.asarray(
                [str(src).startswith("synthetic_negative") for src in sources]
            )
            if is_synthetic.any():
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
                    summary["real_only_n_negatives"] = float(int((real_mask & ~positives).sum()))

                for kind in sorted({str(src) for src in sources[is_synthetic]}):
                    selector = positives | (sources == kind)
                    if len(np.unique(true_all[selector])) < 2:
                        continue
                    label = kind.replace("synthetic_negative_", "")
                    for name, value in metrics_for_loss_type(
                        self.loss_type, true_all[selector], pred_all[selector]
                    ).items():
                        summary[f"decoy_{label}_{name}"] = value

        # Every task keeps the same held-out examples while exposing a mapping
        # category breakdown. This is diagnostics, not a different objective.
        if self._mapping_categories:
            categories = np.asarray(self._mapping_categories, dtype=object)
            for category in sorted({str(value) for value in categories if str(value)}):
                selector = categories == category
                for name, value in metrics_for_loss_type(
                    self.loss_type, true_all[selector], pred_all[selector]
                ).items():
                    summary[f"mapping_{category}_{name}"] = value
                if self.loss_type == "censor":
                    category_exact = selector & (qualifiers == 0)
                    if category_exact.any():
                        for name, value in regression_metrics(
                            true_all[category_exact], pred_all[category_exact]
                        ).items():
                            summary[f"mapping_{category}_exact_{name}"] = value
                    if self.task_name.startswith("binding"):
                        for name, value in binding_threshold_metrics(
                            true_all[selector],
                            pred_all[selector],
                            qualifiers[selector],
                            threshold_nm=500.0,
                        ).items():
                            summary[f"mapping_{category}_threshold_500nm_{name}"] = value
        if self.loss_type == "ce":
            losses = [item.get("observation_loss", "") for item in self._observations]
            if losses and all(value != "" for value in losses):
                summary["cross_entropy"] = float(np.mean(losses))
        for field, prefix in (("component_name", "component"), ("selector_name", "column")):
            names = np.asarray([item.get(field, "") for item in self._observations])
            for name in sorted(set(names) - {""}):
                selection = names == name
                for metric, value in metrics_for_loss_type(
                    self.loss_type, true_all[selection], pred_all[selection]
                ).items():
                    summary[f"{prefix}_{name}_{metric}"] = value
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
                "source": source,
                "qualifier": qualifier,
                "y_true": true_value,
                "y_pred": pred_value,
                "y_prob": _logistic(pred_value) if is_binary else "",
                "source_mapping_category": mapping_category,
                "source_mapping_n_candidates": n_candidates,
                "source_mapping_n_genes": n_genes,
                "source_mapping_n_flank_pairs": n_flank_pairs,
                "flank_context_resolved": flank_resolved,
                **lineage,
                **observation,
            }
            for (
                sample_id,
                source,
                qualifier,
                true_value,
                pred_value,
                mapping_category,
                n_candidates,
                n_genes,
                n_flank_pairs,
                flank_resolved,
                lineage,
                observation,
            ) in zip(
                self._sample_ids,
                self._sources,
                self._qualifiers,
                self._true,
                self._pred,
                self._mapping_categories,
                self._mapping_n_candidates,
                self._mapping_n_genes,
                self._mapping_n_flank_pairs,
                self._flank_context_resolved,
                self._lineage,
                self._observations,
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


def _add_mil_predictions(accumulators, predictions, channel, batch, batch_index):
    """Join by collator source positions, never by potentially duplicated IDs."""
    import json

    import torch

    if not predictions:
        return
    n_bags = channel["bag_label"].numel()
    rows = channel["bag_sample_indices"]
    sample_ids = getattr(batch, "sample_ids", [])
    if len(rows) != n_bags or any(i < 0 or i >= len(sample_ids) for i in rows):
        raise ValueError("MIL export requires explicit bag source-row positions")
    if len(channel["bag_sample_ids"]) != n_bags:
        raise ValueError("MIL export requires one bag ID per bag")

    def select(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values = values.reshape(-1).detach().cpu().tolist()
        return [values[i] for i in rows]

    def host(tensor):
        return tensor.detach().cpu().tolist()

    membership = host(channel["instance_to_bag"])
    members = [[] for _ in range(n_bags)]
    for instance_index, bag_index in enumerate(membership):
        members[bag_index].append(instance_index)
    for name, prediction in predictions.items():
        target = prediction.target
        selectors = host(target.selectors) if target.selectors is not None else None
        losses = host(prediction.losses)
        original_counts = host(target.instance_counts)
        evaluated_counts = host(prediction.evaluated_counts)
        observations = [
            {
                "observation_kind": "bag",
                "batch_index": batch_index,
                "bag_index": i,
                "bag_id": channel["bag_sample_ids"][i],
                "source_row_index": rows[i],
                "instance_count": original_counts[i],
                "evaluated_instance_count": evaluated_counts[i],
                "bag_instance_indices": json.dumps(members[i]),
                "output_path": ".".join(target.spec.output_path),
                "selector_axis": target.spec.axis,
                "selector_index": selectors[i] if selectors is not None else "",
                "selector_name": target.spec.columns[selectors[i]] if selectors is not None else "",
                "observation_loss": losses[i],
                "loss_group": name,
                "loss_type": "bce",
                "loss_reduction_weight": 1.0 / target.support,
                "observation_weight": 1.0,
                "raw_target": float(target.labels[i]),
                "raw_unit": "response",
                "target_unit": "response",
                "prediction_unit": "logit",
                "canonical_output": "elution_logit"
                if name == "ms"
                else ".".join(target.spec.output_path),
            }
            for i in range(n_bags)
        ]
        acc = accumulators.setdefault(name, TaskPredictionAccumulator(name, "bce"))
        before = len(acc)
        acc.add(
            host(target.labels),
            host(prediction.logits),
            host(target.mask),
            sample_ids=select(sample_ids),
            sources=select(getattr(batch, "sample_sources", None)),
            mapping_categories=select(getattr(batch, "source_mapping_categories", None)),
            mapping_n_candidates=select(getattr(batch, "source_mapping_n_candidates", None)),
            mapping_n_genes=select(getattr(batch, "source_mapping_n_genes", None)),
            mapping_n_flank_pairs=select(getattr(batch, "source_mapping_n_flank_pairs", None)),
            flank_context_resolved=select(getattr(batch, "flank_context_resolved", None)),
            lineage={
                key: select(values) for key, values in getattr(batch, "source_lineage", {}).items()
            },
            observations=observations,
        )
        if len(acc) - before != target.support:
            raise ValueError(f"MIL export support mismatch for {name}")


def _add_row_predictions(accumulators, predictions, batch, batch_index):
    import json

    import torch

    def host(tensor):
        return tensor.detach().cpu().tolist()

    group_axes = {}
    for prediction in predictions.values():
        name = prediction.target.group
        group_axes[name] = group_axes.get(name, 0) + 1
    for name, prediction in predictions.items():
        target, spec = prediction.target, prediction.target.spec
        rows = host(target.source_rows)

        def select(values, rows=rows, name=name):
            if values is None or len(values) == 0:
                return None
            if isinstance(values, torch.Tensor):
                values = host(values)
            if rows and max(rows) >= len(values):
                raise ValueError(f"{name}: source metadata is shorter than the observation rows")
            return [values[row] for row in rows]

        is_ce = spec.loss_type == "ce"
        values = prediction.values
        predicted = values.argmax(dim=-1) if is_ce else values
        labels = host(target.transformed)
        raw = host(target.raw_target)
        predicted_values = host(predicted)
        components = host(target.components) if target.components is not None else None
        selectors = host(target.selectors) if target.selectors is not None else None
        loss_values = host(prediction.losses)
        weights = host(target.mask / (target.mask.sum() + 1e-8) / group_axes[target.group])
        observation_weights = host(target.mask)
        class_names = getattr(spec, "class_names", ())
        if is_ce and not class_names:
            class_names = tuple(str(i) for i in range(values.shape[1]))
        class_logits = host(values) if is_ce else None
        class_probabilities = host(values.softmax(-1)) if is_ce else None
        columns = getattr(spec, "columns", ())
        component_names = getattr(spec, "component_names", ())
        observations = []
        for i, row in enumerate(rows):
            component = components[i] if components is not None else None
            selector = selectors[i] if selectors is not None else None
            info = {
                "observation_kind": "categorical"
                if is_ce
                else "component"
                if component is not None
                else "panel"
                if selector is not None
                else "row",
                "batch_index": batch_index,
                "source_row_index": row,
                "output_path": ".".join(prediction.output_path),
                "canonical_output": "elution_logit"
                if name == "ms"
                else ".".join(prediction.output_path),
                "raw_target": raw[i],
                "raw_unit": getattr(spec, "raw_unit", ""),
                "target_unit": getattr(spec, "target_unit", ""),
                "prediction_unit": "class_index"
                if is_ce
                else "logit"
                if spec.loss_type == "bce"
                else getattr(spec, "target_unit", ""),
                "component_index": component if component is not None else "",
                "component_name": component_names[component]
                if component_names and component is not None
                else str(component)
                if component is not None
                else "",
                "selector_axis": getattr(spec, "axis", ""),
                "selector_index": selector if selector is not None else "",
                "selector_name": columns[selector]
                if columns and selector is not None
                else str(selector)
                if selector is not None
                else "",
                "observation_loss": loss_values[i],
                "loss_group": target.group,
                "loss_type": spec.loss_type,
                "loss_reduction_weight": weights[i],
                "observation_weight": observation_weights[i],
            }
            if is_ce and target.mask[i] > 0:
                info.update(
                    class_names=json.dumps(class_names),
                    # String -inf preserves structurally masked classes in valid JSON.
                    class_logits=json.dumps(
                        [value if math.isfinite(value) else str(value) for value in class_logits[i]]
                    ),
                    class_probabilities=json.dumps(class_probabilities[i]),
                    target_class_name=class_names[int(labels[i])],
                    predicted_class_name=class_names[int(predicted_values[i])],
                )
            observations.append(info)
        acc = accumulators.setdefault(name, TaskPredictionAccumulator(name, spec.loss_type))
        before = len(acc)
        acc.add(
            labels,
            predicted_values,
            host(target.mask),
            sample_ids=select(getattr(batch, "sample_ids", None)),
            sources=select(getattr(batch, "sample_sources", None)),
            mapping_categories=select(getattr(batch, "source_mapping_categories", None)),
            mapping_n_candidates=select(getattr(batch, "source_mapping_n_candidates", None)),
            mapping_n_genes=select(getattr(batch, "source_mapping_n_genes", None)),
            mapping_n_flank_pairs=select(getattr(batch, "source_mapping_n_flank_pairs", None)),
            flank_context_resolved=select(getattr(batch, "flank_context_resolved", None)),
            qualifiers=host(target.qualifiers) if target.qualifiers is not None else None,
            lineage={
                key: select(value) for key, value in getattr(batch, "source_lineage", {}).items()
            },
            observations=observations,
        )
        if len(acc) - before != target.support:
            raise ValueError(f"{name}: prediction export support mismatch")


def collect_holdout_predictions(
    model,
    loader,
    device,
    specs: Sequence[Any],
    forward_fn,
    resolve_pred_fn,
    get_target_fn,
    get_mask_fn,
    get_qual_fn=None,
    max_batches: int = 0,
    mil_chunk_size: Optional[int] = None,
) -> Dict[str, TaskPredictionAccumulator]:
    """Run a held-out pass and collect masked predictions for every task.

    The callables are injected rather than imported so this module stays free of
    a dependency on the training scripts (which import it): ``forward_fn(model,
    batch)`` returns the outputs dict; ``resolve_pred_fn(outputs, batch, spec)``
    selects the supervised response column using the same resolver as the loss
    loop. Target/mask/qualifier callables share its transformation contract.
    """
    import torch

    from .mil import (
        DEFAULT_MIL_EVAL_CHUNK_SIZE,
        MIL_TASKS,
        get_mil_channel,
        predict_mil_channel,
        resolve_mil_targets,
    )

    if mil_chunk_size is None:
        mil_chunk_size = DEFAULT_MIL_EVAL_CHUNK_SIZE

    from .supervision import (
        PANEL_TASK_SPECS,
        make_row_target,
        pair_row_prediction,
        resolve_row_targets,
        resolve_row_predictions,
        reduce_row_predictions,
        _resolve_output_tensor,
    )

    accumulators = PredictionCollection(
        {spec.name: TaskPredictionAccumulator(spec.name, spec.loss_type) for spec in specs}
    )
    _declare_output_support(accumulators, (*specs, *PANEL_TASK_SPECS))
    for mil_specs in MIL_TASKS.values():
        _declare_output_support(accumulators, mil_specs)

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            moved = batch.to(device) if hasattr(batch, "to") else batch
            outputs = forward_fn(model, moved)
            batch_losses, batch_support = {}, {}
            observation_counts = {}
            bag_tasks = set()
            for prefix, mil_specs in MIL_TASKS.items():
                channel = get_mil_channel(moved, prefix)
                if channel is None or not channel["bag_label"].numel():
                    continue
                targets = resolve_mil_targets(channel, mil_specs)
                predictions = predict_mil_channel(
                    model,
                    channel=channel,
                    targets=targets,
                    device=device,
                    chunk_size=mil_chunk_size,
                )
                _record_output_support(accumulators, targets, predictions)
                _add_mil_predictions(accumulators, predictions, channel, moved, batch_index)
                observation_counts.update(
                    {name: prediction.target.support for name, prediction in predictions.items()}
                )
                batch_losses.update(
                    {name: float(prediction.loss()) for name, prediction in predictions.items()}
                )
                batch_support.update(
                    {
                        name: prediction.target.weight_sum
                        if hasattr(prediction.target, "weight_sum")
                        else float(prediction.target.support)
                        for name, prediction in predictions.items()
                    }
                )
                if prefix == "mil":
                    # The canonical loss replaces elution/presentation/ms row
                    # objectives with all class-split bags from this batch.
                    bag_tasks.update(spec.name for spec in mil_specs)
            row_predictions = {}
            row_targets = {}
            for spec in specs:
                if spec.name in bag_tasks:
                    continue
                target = get_target_fn(moved, spec)
                mask = get_mask_fn(moved, spec)
                if target is None or mask is None or not bool((mask > 0).any()):
                    continue
                selector_key = getattr(spec, "selector_key", None)
                selectors = (
                    getattr(moved, getattr(spec, "selector_context", "tcell_context"), {}).get(
                        selector_key
                    )
                    if selector_key
                    else None
                )
                view = make_row_target(
                    spec,
                    target,
                    mask,
                    raw_target=getattr(moved, "raw_targets", {}).get(
                        getattr(spec, "target_key", spec.name), target
                    ),
                    qualifiers=get_qual_fn(moved, spec) if get_qual_fn is not None else None,
                    selectors=selectors,
                )
                row_targets[spec.name] = view
                pred = resolve_pred_fn(outputs, moved, spec)
                if pred is None:
                    continue
                path = next(
                    (
                        path
                        for path in spec.pred_paths
                        if _resolve_output_tensor(outputs, (path,)) is not None
                    ),
                    (),
                )
                row_predictions[spec.name] = pair_row_prediction(view, pred, path)
            panel_targets = resolve_row_targets(moved, PANEL_TASK_SPECS)
            row_targets.update(panel_targets)
            row_predictions.update(resolve_row_predictions(outputs, panel_targets))
            _record_output_support(accumulators, row_targets, row_predictions)
            _add_row_predictions(accumulators, row_predictions, moved, batch_index)
            losses, support = reduce_row_predictions(row_predictions)
            batch_losses.update({name: float(value) for name, value in losses.items()})
            batch_support.update(support)
            observation_counts.update(
                {name: prediction.target.support for name, prediction in row_predictions.items()}
            )
            accumulators.batch_losses.append(batch_losses)
            accumulators.batch_support.append(batch_support)
            accumulators.batch_observation_counts.append(observation_counts)
    return accumulators


def write_holdout_artifacts(
    out_dir,
    accumulators: Mapping[str, TaskPredictionAccumulator],
    split: str = "val",
    extra_summary: Optional[Mapping[str, Any]] = None,
    expected_batches: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Write split-specific summary, metrics, and predictions for a held-out pass.

    ``summary.json`` remains the validation-summary compatibility alias. A
    later test pass writes ``test_summary.json`` without overwriting it.
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
    payload["metric_estimators"] = {"auprc": AUPRC_ESTIMATOR}

    # Flat metric CSV written here rather than through RunLogger: the logger is
    # closed in the trainer's `finally` block, so anything logged afterwards
    # hits a closed file. Keeping the artifact self-contained avoids coupling
    # this pass to that lifecycle.
    flat = flatten_summary(summary)
    with (out_path / f"{split}_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "metric", "value"])
        for metric_name, value in sorted(flat.items()):
            writer.writerow([split, metric_name, value])

    rows: List[Dict[str, Any]] = []
    for accumulator in accumulators.values():
        rows.extend(accumulator.rows())
    predictions_path = out_path / f"{split}_predictions.csv"
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "sample_id",
                "source",
                "qualifier",
                "y_true",
                "y_pred",
                "y_prob",
                "source_mapping_category",
                "source_mapping_n_candidates",
                "source_mapping_n_genes",
                "source_mapping_n_flank_pairs",
                "flank_context_resolved",
                *PREDICTION_LINEAGE_FIELDS,
                *MIL_OBSERVATION_FIELDS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    if isinstance(accumulators, PredictionCollection):
        from .evaluation_ledger import reconcile_prediction_artifact

        ledger = reconcile_prediction_artifact(predictions_path, accumulators, expected_batches)
        ledger_name = f"{split}_loss_ledger.json"
        (out_path / ledger_name).write_text(json.dumps(ledger, indent=2, allow_nan=False))
        payload["loss_reconciliation"] = {
            "verified": True,
            "canonical_evaluation_compared": ledger["canonical_evaluation_compared"],
            "artifact": ledger_name,
        }
        payload["output_support"] = accumulators.output_support
    elif expected_batches is not None:
        raise ValueError("canonical evaluation reconciliation requires a PredictionCollection")
    rendered_summary = json.dumps(payload, indent=2, allow_nan=False)
    (out_path / f"{split}_summary.json").write_text(rendered_summary)
    if split == "val" or not (out_path / "summary.json").exists():
        (out_path / "summary.json").write_text(rendered_summary)
    return payload
