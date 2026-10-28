"""Sampling exposure contracts, independent of assay target values."""

from collections import Counter
from dataclasses import replace
import json
import random

import pytest

from presto.data.modular_sources import ObservationIndex
from presto.tests.test_modular_resume import manifest_fixture
from presto.tests.test_unified_training_core import row
from presto.training.modular_run import preflight
from presto.training.modular_sampling import epoch_order, validate_stratum_weights


def weights(a=3, b=1):
    return [{"values": ["ic50"], "weight": a}, {"values": ["kd"], "weight": b}]


def test_weighted_epochs_cover_unique_rows_and_reach_requested_label_mix():
    rows = [row(endpoint="ic50") for _ in range(6)] + [row(endpoint="kd") for _ in range(8)]
    rows.append(row(endpoint="kd", split="test"))
    order = epoch_order(rows, range(14), ["endpoint"], random.Random(91), weights())
    assert len(order) == 32 and set(order) == set(range(14))
    assert Counter(rows[i].endpoint for i in order) == {"ic50": 24, "kd": 8}
    assert Counter(i for i in order if i < 6) == dict.fromkeys(range(6), 4)
    assert Counter(i for i in order if i >= 6) == dict.fromkeys(range(6, 14), 1)
    assert order == epoch_order(rows, range(14), ["endpoint"], random.Random(91), weights(30, 10))
    changed = [replace(r, value=100000 - i) for i, r in enumerate(rows)]
    assert order == epoch_order(changed, range(14), ["endpoint"], random.Random(91), weights())
    assert order != epoch_order(rows, range(14), ["endpoint"], random.Random(92), weights())


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "3", 10**1000])
def test_invalid_weight_values_fail_before_order_creation(value):
    with pytest.raises(ValueError, match="positive finite"):
        validate_stratum_weights(["endpoint"], weights(value))


def test_weight_schema_and_unknown_or_missing_strata_fail():
    for fields, records in [
        ([], weights()),
        (["endpoint"], None),
        (["endpoint"], [{"values": ["ic50"], "weight": 1, "extra": 2}]),
        (["endpoint", "mhc_class"], weights()),
        (["endpoint"], weights() + weights()),
    ]:
        with pytest.raises(ValueError, match="stratum_weights"):
            validate_stratum_weights(fields, records)
    rows = [row(endpoint="ic50"), row(endpoint="kd")]
    for records in (weights()[:1], weights() + [{"values": ["ec50"], "weight": 1}]):
        with pytest.raises(ValueError, match="every observed training stratum"):
            epoch_order(rows, range(2), ["endpoint"], random.Random(1), records)
    with pytest.raises(ValueError, match="cannot be represented|unrepresentable"):
        epoch_order(rows, range(2), ["endpoint"], random.Random(1), weights(1e-320, 1e308))


def test_manifest_preflight_checks_weighted_training_strata(tmp_path):
    path = manifest_fixture(tmp_path, replay=False)
    document = json.loads(path.read_text())
    document["training"].update(balance_by=["endpoint"], stratum_weights=weights())
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="every observed training stratum"):
        preflight(ObservationIndex(path))
    document["training"]["stratum_weights"] = weights()[:1]
    path.write_text(json.dumps(document))
    preflight(ObservationIndex(path))
