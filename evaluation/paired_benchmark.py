"""Strict matched-row evaluation helpers for assay-aware comparator reports."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from presto.training.holdout_eval import binary_metrics, regression_metrics


def sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid."""
    value = float(value)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def study_id(record: Mapping) -> str:
    """Return the frozen study identity carried by a modular prediction row."""
    provenance = record.get("observation", {}).get("provenance", {})
    value = provenance.get("study_id") or record.get("group_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"prediction {record.get('observation_id')} lacks study identity")
    return value


def prediction_identity(record: Mapping) -> str:
    """Canonical truth identity used to reject cohort or label drift."""
    observation = record.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("prediction lacks its frozen observation")
    provenance = observation.get("provenance", {})
    assay = record.get("assay_method")
    identity = {
        "observation_id": record.get("observation_id"),
        "endpoint": record.get("endpoint"),
        "source": record.get("source"),
        "group_id": record.get("group_id"),
        "split": record.get("split"),
        "peptide": record.get("peptide"),
        "qualifier": record.get("qualifier"),
        "mhc_classes": record.get("mhc_classes"),
        "target": record.get("target"),
        "raw_value": record.get("raw_value"),
        "raw_unit": record.get("raw_unit"),
        "target_space": record.get("target_space"),
        "assay_method": assay,
        "study_id": provenance.get("study_id") or record.get("group_id"),
        "original_measurement_id": provenance.get("original_measurement_id"),
        "pair_id": provenance.get("pair_id"),
    }
    if identity["split"] != "validation":
        raise ValueError(
            f"prediction {identity['observation_id']} is not on the validation split"
        )
    if not isinstance(identity["observation_id"], str) or not identity["observation_id"]:
        raise ValueError("prediction lacks observation_id")
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


def index_predictions(records: Iterable[Mapping], label: str) -> dict[str, Mapping]:
    """Index one method's predictions, rejecting duplicate observation identities."""
    result = {}
    for record in records:
        observation_id = record.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError(f"{label} prediction lacks observation_id")
        if observation_id in result:
            raise ValueError(f"{label} has duplicate observation_id {observation_id}")
        prediction_identity(record)
        result[observation_id] = record
    return result


def require_aligned_predictions(
    methods: Mapping[str, Sequence[Mapping]], *, expected_rows: int | None = None
) -> dict[str, dict[str, Mapping]]:
    """Require exact row and truth equality across complete prediction dumps."""
    if len(methods) < 2:
        raise ValueError("alignment requires at least two methods")
    indexed = {name: index_predictions(records, name) for name, records in methods.items()}
    first_name = next(iter(indexed))
    first = indexed[first_name]
    if expected_rows is not None and len(first) != expected_rows:
        raise ValueError(f"{first_name} has {len(first)} rows; expected {expected_rows}")
    first_ids = set(first)
    for name, values in indexed.items():
        if set(values) != first_ids:
            missing = sorted(first_ids - set(values))[:3]
            extra = sorted(set(values) - first_ids)[:3]
            raise ValueError(f"{name} row identities differ: missing={missing}, extra={extra}")
        for observation_id in first:
            if prediction_identity(first[observation_id]) != prediction_identity(
                values[observation_id]
            ):
                raise ValueError(f"{name} truth identity differs at {observation_id}")
    return indexed


def _logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logmeanexp requires at least one value")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("logmeanexp received a non-finite value")
    maximum = float(array.max())
    return maximum + math.log(float(np.exp(array - maximum).mean()))


def operational_processing_logit(record: Mapping) -> float:
    """Recover the frozen operational processing-plus-MS score.

    This is a model decomposition, not direct proteolysis supervision. It
    marginalizes each MHC candidate's class-conditioned processing term with
    the same class weights used by the model, candidate-pools those terms with
    normalized log-sum-exp, and then adds the MS-response and observation offset.
    """
    if record.get("endpoint") != "elution_observation":
        raise ValueError("operational processing requires an elution observation")
    components = record.get("component_scores", {})
    try:
        presentation = float(components["elution_presentation"])
        ms_response = float(components["ms_response"])
        prediction = float(record["prediction"])
        candidates = record["diagnostics"]["candidates"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("elution prediction lacks staged decomposition") from error
    candidate_processing = []
    for candidate_index, candidate in enumerate(candidates):
        registers = candidate.get("registers", [])
        if not registers:
            raise ValueError(f"candidate {candidate_index} lacks register diagnostics")
        total_weight = 0.0
        value = 0.0
        for register in registers:
            try:
                weight = float(register["class_probability"])
                processing = float(register["elution_terms"]["processing"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("register lacks its staged processing term") from error
            total_weight += weight
            value += weight * processing
        if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"candidate class weights sum to {total_weight}, not one")
        candidate_processing.append(value)
    offset = prediction - presentation - ms_response
    result = _logmeanexp(candidate_processing) + ms_response + offset
    if not math.isfinite(result):
        raise ValueError("operational processing score is non-finite")
    return result


def qualifier_code(value: str) -> int:
    """Map physical affinity qualifiers to lower/exact/upper numeric codes."""
    normalized = str(value).strip()
    if normalized in {"=", "==", ""}:
        return 0
    if normalized in {"<=", "<"}:
        return -1
    if normalized in {">=", ">"}:
        return 1
    raise ValueError(f"unsupported qualifier {value!r}")


def affinity_threshold_metrics(
    records: Sequence[Mapping], *, threshold_nm: float = 500.0
) -> dict[str, float]:
    """Qualifier-aware affinity classification metrics on long-table rows."""
    cutoff = math.log10(threshold_nm)
    labels, scores = [], []
    for row in records:
        target = float(row["target_log10_nM"])
        prediction = float(row["prediction_log10_nM"])
        qualifier = qualifier_code(row["qualifier"])
        at_or_below = target <= cutoff
        usable = (
            qualifier == 0
            or qualifier < 0 and at_or_below
            or qualifier > 0 and not at_or_below
        )
        if not usable:
            continue
        labels.append(float(at_or_below))
        scores.append(-prediction)
    if not labels:
        return {}
    return binary_metrics(np.asarray(labels), np.asarray(scores), threshold=-cutoff)


def censor_metrics(records: Sequence[Mapping]) -> dict[str, float]:
    """Report satisfaction and violation magnitude for censored affinity rows."""
    satisfied, violations = [], []
    for row in records:
        code = qualifier_code(row["qualifier"])
        if code == 0:
            continue
        target = float(row["target_log10_nM"])
        prediction = float(row["prediction_log10_nM"])
        violation = max(0.0, prediction - target) if code < 0 else max(0.0, target - prediction)
        satisfied.append(violation == 0.0)
        violations.append(violation)
    if not violations:
        return {"n": 0.0}
    array = np.asarray(violations, dtype=float)
    return {
        "n": float(len(array)),
        "satisfaction": float(np.mean(satisfied)),
        "mean_violation_log10_nM": float(array.mean()),
        "rmse_violation_log10_nM": float(np.sqrt(np.mean(array**2))),
    }


def metric_value(records: Sequence[Mapping], metric: str, kind: str) -> float | None:
    """Compute one regression or binary metric from matched long-table rows."""
    if not records:
        return None
    target = np.asarray([float(row["target"]) for row in records], dtype=float)
    score = np.asarray([float(row["score"]) for row in records], dtype=float)
    if not np.isfinite(target).all() or not np.isfinite(score).all():
        raise ValueError("metric input contains non-finite values")
    if kind == "regression":
        metrics = regression_metrics(target, score)
    elif kind == "binary":
        metrics = binary_metrics(target, score)
    else:
        raise ValueError(f"unsupported metric kind {kind}")
    value = metrics.get(metric)
    return None if value is None else float(value)


def _paired_by_method(
    records: Sequence[Mapping], first: str, second: str
) -> tuple[dict[str, Mapping], dict[str, Mapping]]:
    result = {}
    for method in (first, second):
        rows = {}
        for row in records:
            if row["method"] != method:
                continue
            observation_id = row["observation_id"]
            if observation_id in rows:
                raise ValueError(f"{method} duplicates {observation_id}")
            rows[observation_id] = row
        result[method] = rows
    if set(result[first]) != set(result[second]):
        raise ValueError(f"{first} and {second} do not contain identical rows")
    for observation_id, left in result[first].items():
        right = result[second][observation_id]
        if (left["study"], float(left["target"])) != (
            right["study"],
            float(right["target"]),
        ):
            raise ValueError(f"paired truth differs at {observation_id}")
    return result[first], result[second]


def paired_study_metrics(
    records: Sequence[Mapping],
    first: str,
    second: str,
    *,
    metric: str,
    kind: str,
) -> list[dict]:
    """Compute paired per-study method values and differences."""
    left, right = _paired_by_method(records, first, second)
    by_study = defaultdict(list)
    for observation_id, row in left.items():
        by_study[row["study"]].append(observation_id)
    result = []
    for study in sorted(by_study):
        ids = by_study[study]
        left_value = metric_value([left[value] for value in ids], metric, kind)
        right_value = metric_value([right[value] for value in ids], metric, kind)
        if left_value is None or right_value is None:
            continue
        result.append(
            {
                "study": study,
                "metric": metric,
                "first": first,
                "second": second,
                "first_value": left_value,
                "second_value": right_value,
                "delta": left_value - right_value,
                "n": len(ids),
            }
        )
    return result


def study_bootstrap(
    records: Sequence[Mapping],
    method: str,
    *,
    metric: str,
    kind: str,
    replicates: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap the equal-study mean for one method."""
    rows = [row for row in records if row["method"] == method]
    by_study = defaultdict(list)
    for row in rows:
        by_study[row["study"]].append(row)
    values = []
    for study in sorted(by_study):
        value = metric_value(by_study[study], metric, kind)
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError(f"no eligible studies for {method} {metric}")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(replicates, len(array)))].mean(axis=1)
    return {
        "method": method,
        "metric": metric,
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "studies": len(array),
        "replicates": replicates,
        "seed": seed,
    }


def paired_study_bootstrap(
    records: Sequence[Mapping],
    first: str,
    second: str,
    *,
    metric: str,
    kind: str,
    replicates: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap equal-study paired metric differences."""
    per_study = paired_study_metrics(
        records, first, second, metric=metric, kind=kind
    )
    if not per_study:
        raise ValueError(f"no paired eligible studies for {metric}")
    deltas = np.asarray([row["delta"] for row in per_study], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = deltas[
        rng.integers(0, len(deltas), size=(replicates, len(deltas)))
    ].mean(axis=1)
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "estimate": float(deltas.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "studies": len(deltas),
        "replicates": replicates,
        "seed": seed,
    }


def grouped_metrics(
    records: Sequence[Mapping],
    group_keys: Sequence[str],
    functions: Mapping[str, Callable[[list[Mapping]], Mapping[str, float]]],
) -> list[dict]:
    """Apply named metric families to deterministic assay-aware groups."""
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in group_keys)].append(row)
    result = []
    for key in sorted(groups):
        base = dict(zip(group_keys, key, strict=True))
        for family, function in functions.items():
            values = function(groups[key])
            for metric, value in sorted(values.items()):
                result.append({**base, "family": family, "metric": metric, "value": value})
    return result
