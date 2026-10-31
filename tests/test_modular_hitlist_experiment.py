"""Split conservation for the registered full-cohort source preparation."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from presto.data.modular import MHCInput, Observation


@pytest.fixture(scope="module")
def recipe():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/2026-10-21_0647_codex_full-ic50-admission/code/launch.py"
    )
    spec = importlib.util.spec_from_file_location("full_ic50_admission_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(ordinal, peptide="ACDEFGHIKL", assay="1", value=10):
    return Observation(
        str(ordinal),
        "test",
        "ic50",
        peptide,
        (MHCInput("A" * 70, "C" * 70, "I", species1="human", species2="human"),),
        value=value,
        unit="nM",
        provenance={
            "measurement_namespace": "assay_iri",
            "original_measurement_id": assay,
            "source_ordinal": ordinal,
        },
    )


def test_original_measurement_and_peptide_links_share_components(recipe):
    rows = [
        observation(1),
        observation(2, assay="2"),
        observation(3, peptide="LMNPQRSTVW", assay="2"),
    ]
    assigned, deferred, components = recipe.assign_components(rows, {"ACDEFGHIKL": {"test"}})
    assert not deferred and len(components) == 1
    assert {row.split for row in assigned} == {"test"}
    assert len({row.group_id for row in assigned}) == 1


def test_conflicting_historical_exposure_defers_whole_component(recipe):
    rows = [observation(1), observation(2, peptide="LMNPQRSTVW")]
    assigned, deferred, _ = recipe.assign_components(
        rows, {"ACDEFGHIKL": {"test"}, "LMNPQRSTVW": {"train"}}
    )
    assert not assigned
    assert set(deferred) == {1, 2}
    assert {item["reason"] for item in deferred.values()} == {
        "incompatible_historical_component_assignments"
    }


def test_partition_does_not_use_target_values_or_input_order(recipe):
    rows = [observation(1), observation(2, peptide="LMNPQRSTVW", assay="2")]
    first, _, _ = recipe.assign_components(rows, {})
    changed, _, _ = recipe.assign_components(
        [replace(row, value=1e8) for row in reversed(rows)], {}
    )
    assert [(row.observation_id, row.group_id, row.split) for row in first] == [
        (row.observation_id, row.group_id, row.split) for row in changed
    ]


@pytest.mark.parametrize(
    "qualifier,value,expected",
    [
        ("=", 500, "binder"),
        ("=", 501, "nonbinder"),
        ("<", 500, "binder"),
        ("<=", 500, "binder"),
        (">", 500, "nonbinder"),
        (">=", 500, "ambiguous_bound"),
        (">=", 501, "nonbinder"),
        ("<", 501, "ambiguous_bound"),
        (">", 499, "ambiguous_bound"),
    ],
)
def test_threshold_support_respects_censor_direction_and_boundary(
    recipe, qualifier, value, expected
):
    assert (
        recipe.binding_state(replace(observation(1, value=value), qualifier=qualifier)) == expected
    )
