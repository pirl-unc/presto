"""Cached admission metadata must preserve actual order, scales and trainability."""

from collections import Counter
from dataclasses import replace
import json
import random

import pytest

from presto.data.modular import MHCInput, TargetScaleFitter, fit_target_scales
from presto.data.modular_sources import ObservationIndex, ObservationMetadata, file_sha256
from presto.models.modular_dependencies import freeze_dependencies, row_has_trainable_loss
from presto.training import modular_run as runner
from presto.training.modular_sampling import epoch_order, stratum
from test_modular_streaming import split_fixture


@pytest.mark.parametrize("fields", [[], ["endpoint"], ["mhc_class"], ["mhc_class", "endpoint"]])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("subset", [[], [0], [5, 0, 4, 1, 0]])
def test_metadata_sampling_matches_rows_and_rng(tmp_path, fields, weighted, subset, monkeypatch):
    index = split_fixture(tmp_path)
    # Exercise multiple nonuniform strata, with duplicates and subset order too.
    manifest = json.loads(index.manifest_path.read_text())
    for source in manifest["sources"]:
        from pathlib import Path

        path = Path(source["path"])
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for i, record in enumerate(records):
            record["endpoint"] = "kd" if i % 3 else "ic50"
            record["mhc"][0]["mhc_class"] = "I" if i % 2 else "II"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        source["sha256"] = file_sha256(path)
    index.manifest_path.write_text(json.dumps(manifest))
    index = ObservationIndex(index.manifest_path)
    rows = list(index)
    groups = Counter(stratum(rows[i], fields) for i in subset)
    weights = (
        [{"values": list(key), "weight": i + 1} for i, key in enumerate(sorted(groups))]
        if weighted and fields
        else []
    )
    left, right = random.Random(913), random.Random(913)
    expected = epoch_order(rows, subset, fields, left, weights)

    def unexpected_read(*args):
        pytest.fail("metadata-only sampling decoded an observation")

    monkeypatch.setattr(type(index), "__getitem__", unexpected_read)
    assert epoch_order(index, subset, fields, right, weights) == expected
    assert left.getstate() == right.getstate()


@pytest.mark.parametrize("classes", [("I",), ("II",), ("I", "II"), (), (None,)])
def test_class_projection_preserves_override_and_invalid_classes(tmp_path, classes):
    row = split_fixture(tmp_path)[0]
    row = replace(row, mhc=tuple(MHCInput("ACDE", "FGHI", value) for value in classes))
    metadata = ObservationMetadata.from_observation(row)
    if len(classes) == 1 and classes[0] is not None:
        assert stratum(row, ["mhc_class"]) == stratum(metadata, ["mhc_class"])
    else:
        for value in (row, metadata):
            with pytest.raises(ValueError, match="one explicit class"):
                stratum(value, ["mhc_class"])
    overridden = replace(row, mhc=(MHCInput("ACDE", "FGHI", "I", class_override="II"),))
    assert stratum(ObservationMetadata.from_observation(overridden), ["mhc_class"]) == ("II",)


def test_preflight_scans_rows_once_and_preserves_exact_train_scales(tmp_path, monkeypatch):
    index = split_fixture(tmp_path)
    expected = fit_target_scales(index)
    original = type(index).__iter__
    visits = []

    def tracked(self):
        for row in original(self):
            visits.append(row.observation_id)
            yield row

    monkeypatch.setattr(type(index), "__iter__", tracked)
    _, _, scales, _ = runner.preflight(index)
    assert scales == expected
    assert len(visits) == len(index)


def test_streaming_scaler_excludes_bounds_and_holdouts_exactly(tmp_path):
    row = split_fixture(tmp_path)[0]
    rows = [replace(row, value=value) for value in (1.0, 10.0, 100.0)]
    rows += [replace(row, value=1e30, qualifier=sign) for sign in ("<", ">", "<=", ">=")]
    rows += [replace(row, split=split, value=1e-20) for split in ("validation", "test")]
    fitter = TargetScaleFitter()
    for observation in rows:
        fitter.add(observation)
    actual = fitter.finish()["ic50"]
    assert actual.mean == 1.0
    assert actual.exact_count == 3
    assert actual.scale == (2 / 3) ** 0.5


@pytest.mark.parametrize("frozen", [False, True])
def test_trainable_filter_matches_loss_dependencies_without_row_reads(
    tmp_path, monkeypatch, frozen
):
    index = split_fixture(tmp_path)
    config, _, _, _ = runner.preflight(index)
    model, _ = runner.initialized_model(config, index)
    if frozen:
        model.trained_parameters = set(dict(model.named_parameters()))
        freeze_dependencies(model, ["ic50"])
    expected = tuple(
        [
            i
            for i in index.partitions.get(f"train:{family}", [])
            if row_has_trainable_loss(model, index[i])
        ]
        for family in ("assay", "identity")
    )

    def unexpected_read(*args):
        pytest.fail("trainability filtering decoded an observation")

    monkeypatch.setattr(type(index), "__getitem__", unexpected_read)
    assert runner.trainable_indices(index, model) == expected
