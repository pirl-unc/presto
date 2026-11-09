"""Deterministic metadata-balanced observation orders for the common trainer."""

from collections import defaultdict


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


def epoch_order(index, indices, fields, rng):
    """Cover each stratum, cycling smaller shuffled strata to the largest size.

    Interleave strata so every complete group has one observation from each.
    No held-out rows or target values are read. The caller persists the returned
    order and RNG state, including partially consumed epochs, for exact resume.
    Empty fields preserve the original shuffled-row behavior and RNG consumption.
    """
    if not fields:
        order = list(indices)
        rng.shuffle(order)
        return order
    groups = defaultdict(list)
    for i in indices:
        groups[stratum(index[i], fields)].append(i)
    if not groups:
        return []
    if len(groups) == 1:
        order = list(indices)
        rng.shuffle(order)
        return order
    keys = sorted(groups)
    streams = {}
    size = max(map(len, groups.values()))
    for key in keys:
        values = []
        while len(values) < size:
            cycle = list(groups[key])
            rng.shuffle(cycle)
            values.extend(cycle)
        streams[key] = values[:size]
    rng.shuffle(keys)
    return [streams[key][position] for position in range(size) for key in keys]
