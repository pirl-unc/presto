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


def _publication_study_id(provenance: Mapping) -> str | None:
    raw = provenance.get("raw_record")
    raw_map = raw if isinstance(raw, Mapping) else {}
    for candidate in (
        provenance.get("study_id"),
        provenance.get("pmid"),
        provenance.get("pubmed_id"),
        provenance.get("reference_id"),
        provenance.get("reference_iri"),
        raw_map.get("study_id"),
        raw_map.get("pmid"),
        raw_map.get("pubmed_id"),
        raw_map.get("reference_id"),
        raw_map.get("reference_iri"),
    ):
        if candidate is not None and not isinstance(candidate, bool):
            text = str(candidate).strip()
            if text:
                return text
    return None


def study_id(record: Mapping) -> str:
    """Return the frozen publication/study identity carried by a modular prediction row."""
    observation = record.get("observation")
    provenance = observation.get("provenance", {}) if isinstance(observation, Mapping) else {}
    if not isinstance(provenance, Mapping):
        provenance = {}
    value = _publication_study_id(provenance)
    if not value:
        raise ValueError(f"prediction {record.get('observation_id')} lacks study identity")
    return value


def _canonical_mhc(observation: Mapping) -> list[dict]:
    raw_mhc = observation.get("mhc") or ()
    if not isinstance(raw_mhc, Sequence) or isinstance(raw_mhc, (str, bytes)):
        raise ValueError("observation.mhc must be a sequence")
    out = []
    for item in raw_mhc:
        if not isinstance(item, Mapping):
            raise ValueError("MHC candidate must be a mapping")
        prov = item.get("provenance")
        prov_map = prov if isinstance(prov, Mapping) else {}
        out.append(
            {
                "groove1": item.get("groove1"),
                "groove2": item.get("groove2"),
                "mhc_class": item.get("mhc_class"),
                "class_override": item.get("class_override"),
                "species1": item.get("species1"),
                "species2": item.get("species2"),
                "catalog_alleles": prov_map.get("catalog_alleles"),
            }
        )
    return out


def _canonical_context(observation: Mapping) -> dict:
    context = observation.get("context")
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise ValueError("observation.context must be a mapping")
    return {key: value for key, value in context.items() if key != "provenance"}


def prediction_identity(record: Mapping) -> str:
    """Canonical truth and input-context identity used to reject cohort or label drift."""
    observation = record.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("prediction lacks its frozen observation")
    for key in (
        "observation_id",
        "endpoint",
        "source",
        "group_id",
        "split",
        "peptide",
        "qualifier",
    ):
        if key in observation and observation[key] != record.get(key):
            raise ValueError(
                f"prediction {record.get('observation_id')} disagrees with observation.{key}"
            )
    if "value" in observation and observation["value"] != record.get("raw_value"):
        raise ValueError(
            f"prediction {record.get('observation_id')} disagrees with observation.value"
        )
    if "unit" in observation and observation["unit"] != record.get("raw_unit"):
        raise ValueError(
            f"prediction {record.get('observation_id')} disagrees with observation.unit"
        )
    provenance = observation.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("observation.provenance must be a mapping")
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
        "mhc": _canonical_mhc(observation),
        "tcr_a": observation.get("tcr_a") or "",
        "tcr_b": observation.get("tcr_b") or "",
        "context": _canonical_context(observation),
        "target": record.get("target"),
        "raw_value": record.get("raw_value"),
        "raw_unit": record.get("raw_unit"),
        "target_space": record.get("target_space"),
        "quantity_contract": record.get("quantity_contract", observation.get("quantity_contract")),
        "value_encoding": record.get("value_encoding", observation.get("value_encoding", "linear")),
        "tcell_track": record.get("tcell_track", observation.get("tcell_track")),
        "tcell_intensity": record.get("tcell_intensity", observation.get("tcell_intensity")),
        "assay_method": assay,
        "study_id": study_id(record),
        "original_measurement_id": provenance.get("original_measurement_id"),
        "pair_id": provenance.get("pair_id"),
    }
    if identity["split"] != "validation":
        raise ValueError(f"prediction {identity['observation_id']} is not on the validation split")
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
    if threshold_nm <= 0 or not math.isfinite(threshold_nm):
        raise ValueError("threshold_nm must be positive and finite")
    cutoff = math.log10(threshold_nm)
    labels, scores = [], []
    for row in records:
        target = float(row["target_log10_nM"])
        prediction = float(row["prediction_log10_nM"])
        raw_qual = str(row["qualifier"]).strip()
        qualifier = qualifier_code(raw_qual)
        if qualifier == 0:
            is_binder = target <= cutoff
            usable = True
        elif qualifier < 0:
            is_binder = True
            usable = target <= cutoff
        else:
            is_binder = False
            usable = (target >= cutoff) if raw_qual == ">" else (target > cutoff)
        if not usable:
            continue
        labels.append(float(is_binder))
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
    per_study = paired_study_metrics(records, first, second, metric=metric, kind=kind)
    if not per_study:
        raise ValueError(f"no paired eligible studies for {metric}")
    deltas = np.asarray([row["delta"] for row in per_study], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = deltas[rng.integers(0, len(deltas), size=(replicates, len(deltas)))].mean(axis=1)
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
