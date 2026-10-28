"""Deterministic metadata-balanced observation orders for the common trainer."""

from collections import defaultdict
import math
import sys


def validate_balance_by(fields):
    if (
        not isinstance(fields, (list, tuple))
        or any(field not in ("endpoint", "mhc_class") for field in fields)
        or len(fields) != len(set(fields))
    ):
        raise ValueError("balance_by requires distinct endpoint/mhc_class fields")


def stratum(row, fields):
    values = []
    for field in fields:
        if field == "endpoint":
            values.append(row.endpoint)
        else:
            classes = {mhc.class_override or mhc.mhc_class for mhc in row.mhc}
            if len(classes) != 1 or not classes <= {"I", "II"}:
                raise ValueError("balanced mhc_class sampling requires one explicit class per row")
            values.append(next(iter(classes)))
    return tuple(values)


def validate_stratum_weights(fields, weights, known=None):
    """Parse explicit relative sampling frequencies, never loss weights."""
    if not isinstance(weights, (list, tuple)):
        raise ValueError("stratum_weights must be a list")
    if not weights:
        return None
    if not fields:
        raise ValueError("stratum_weights requires balance_by fields")
    parsed = {}
    for record in weights:
        if not isinstance(record, dict) or set(record) != {"values", "weight"}:
            raise ValueError("stratum_weights records require values and weight")
        values, weight = record["values"], record["weight"]
        try:
            numeric = float(weight) if type(weight) in (int, float) else math.nan
        except OverflowError:
            numeric = math.inf
        if (
            not isinstance(values, (list, tuple))
            or len(values) != len(fields)
            or any(not isinstance(value, str) or not value for value in values)
            or not math.isfinite(numeric)
            or numeric <= 0
        ):
            raise ValueError("stratum_weights needs field values and positive finite weights")
        key = tuple(values)
        if key in parsed:
            raise ValueError("duplicate stratum_weights values")
        parsed[key] = numeric
    if known is not None and set(parsed) != set(known):
        raise ValueError("stratum_weights must name every observed training stratum exactly once")
    return parsed


def weighted_quotas(sizes, weights):
    """Cover every unique row, oversampling only to attain requested frequencies."""
    largest = max(weights.values())
    normalized = {key: value / largest for key, value in weights.items()}
    if any(value == 0 for value in normalized.values()):
        raise ValueError("stratum_weights ratio cannot be represented")
    scale = max(sizes[key] / normalized[key] for key in sizes)
    if not math.isfinite(scale) or scale > sys.maxsize:
        raise ValueError("stratum_weights requires an unrepresentable epoch")
    quotas = {key: max(sizes[key], math.ceil(scale * normalized[key])) for key in sizes}
    if sum(quotas.values()) > sys.maxsize:
        raise ValueError("stratum_weights requires an unrepresentable epoch")
    return quotas


def epoch_order(index, indices, fields, rng, weights=()):
    """Cover each stratum, cycling smaller shuffled strata to the largest size.

    Interleave strata so every complete group has one observation from each.
    No held-out rows or target values are read. The caller persists the returned
    order and RNG state, including partially consumed epochs, for exact resume.
    Optional weights set relative stratum frequencies in a shuffled weighted epoch.
    Without weights, retain the original order and RNG consumption exactly.
    """
    parsed = validate_stratum_weights(fields, weights)
    if not fields:
        order = list(indices)
        rng.shuffle(order)
        return order
    groups = defaultdict(list)
    for i in indices:
        groups[stratum(index[i], fields)].append(i)
    if not groups:
        if parsed:
            raise ValueError("stratum_weights supplied without training strata")
        return []
    if parsed is not None:
        validate_stratum_weights(fields, weights, groups)
    if len(groups) == 1:
        order = list(indices)
        rng.shuffle(order)
        return order
    keys = sorted(groups)
    streams = {}
    size = max(map(len, groups.values()))
    quotas = (
        weighted_quotas({key: len(value) for key, value in groups.items()}, parsed)
        if parsed is not None
        else dict.fromkeys(keys, size)
    )
    for key in keys:
        values = []
        while len(values) < quotas[key]:
            cycle = list(groups[key])
            rng.shuffle(cycle)
            values.extend(cycle)
        streams[key] = values[: quotas[key]]
    if parsed is not None:
        order = [i for key in keys for i in streams[key]]
        rng.shuffle(order)
        return order
    rng.shuffle(keys)
    return [streams[key][position] for position in range(size) for key in keys]
