"""Sampling exposure contracts, independent of assay target values."""

from collections import Counter
from dataclasses import replace
import json
import random

import pytest

from presto.data.modular_sources import ObservationIndex, ObservationMetadata
from presto.tests.test_modular_resume import manifest_fixture
from presto.tests.test_unified_training_core import row
from presto.training.modular_run import check_training_strata, preflight
from presto.training.modular_sampling import (
    epoch_order,
    validate_balance_by,
    validate_stratum_weights,
)


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


def test_stratum_weights_are_checked_against_the_rows_actually_trained():
    rows = [row(endpoint="ic50") for _ in range(3)] + [row(endpoint="kd") for _ in range(2)]
    training = {"balance_by": ["endpoint"], "stratum_weights": weights()[:1]}
    check_training_strata(rows, training, range(3))
    with pytest.raises(ValueError, match="every observed training stratum"):
        check_training_strata(rows, training, range(5))


@pytest.mark.parametrize("value", [True, False, 0, 1.5])
def test_preflight_rejects_non_integer_max_steps(tmp_path, value):
    path = manifest_fixture(tmp_path, replay=False)
    document = json.loads(path.read_text())
    document["training"]["max_steps"] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="max_steps"):
        preflight(ObservationIndex(path))


def test_source_stratum_sets_mix_between_sources_within_one_endpoint():
    rows = [row(endpoint="ic50") for _ in range(10)] + [row(endpoint="ic50") for _ in range(2)]
    rows = [
        replace(r, source="canonical" if i < 10 else "perturbation") for i, r in enumerate(rows)
    ]
    fields = ["endpoint", "source"]
    records = [
        {"values": ["ic50", "canonical"], "weight": 2},
        {"values": ["ic50", "perturbation"], "weight": 1},
    ]
    order = epoch_order(rows, range(12), fields, random.Random(5), records)
    assert set(order) == set(range(12))
    counts = Counter(rows[i].source for i in order)
    assert counts["canonical"] == counts["perturbation"] * 2
    metadata = [ObservationMetadata.from_observation(r) for r in rows]
    assert metadata[0].source == "canonical"
    with pytest.raises(ValueError, match="balance_by requires"):
        validate_balance_by(["endpoint", "bogus"])
    validate_balance_by(fields)


def test_bounded_draws_preserve_weights_without_materializing_an_oversampled_epoch():
    from presto.training.modular_sampling import weighted_step_order

    rows = [row(endpoint='ic50') for _ in range(10000)] + [row(endpoint='kd')]
    order = weighted_step_order(
        rows, range(len(rows)), ['endpoint'], random.Random(91), weights(), 1000
    )
    assert len(order) == 1000
    assert order == weighted_step_order(
        rows, range(len(rows)), ['endpoint'], random.Random(91), weights(), 1000
    )
    assert 700 <= sum(rows[i].endpoint == 'ic50' for i in order) <= 800
    changed = [replace(r, value=100000 - i) for i, r in enumerate(rows)]
    assert order == weighted_step_order(
        changed, range(len(rows)), ['endpoint'], random.Random(91), weights(), 1000
    )


def test_study_sampling_uses_source_provenance_and_preserves_protein_split_groups():
    from presto.training.modular_sampling import stratum

    observed = replace(
        row(endpoint='ic50'), group_id='protein:1', provenance={'study_id': 'study:2'}
    )
    metadata = ObservationMetadata.from_observation(observed)
    assert metadata.group_id == 'protein:1'
    assert stratum(metadata, ['study']) == stratum(observed, ['study']) == ('study:2',)
    for missing in [replace(observed, provenance={}), replace(metadata, study=None)]:
        with pytest.raises(ValueError, match='explicit study provenance'):
            stratum(missing, ['study'])


def test_bounded_training_resumes_exactly(tmp_path, monkeypatch):
    from presto.tests.test_modular_affinity_controls import fixture, run_with_interruption

    path = fixture(
        tmp_path,
        sampling_mode='weighted_steps',
        balance_by=['mhc_class'],
        stratum_weights=[{'values': ['I'], 'weight': 1}, {'values': ['II'], 'weight': 1}],
    )
    run_with_interruption(tmp_path, monkeypatch, path, before_step=6)
