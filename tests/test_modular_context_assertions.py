"""Concurrent context claims; all observations here are artificial software fixtures."""

from dataclasses import asdict, replace
import json

import pytest
import torch

from presto.data.modular import Context, ContextAssertion, MHCInput, Observation
from presto.data.modular_context import resolve_cell_context
from presto.data.modular_lineage import semantic_fingerprint
from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, write_observations
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from presto.training.modular_run import preflight


def settings(**changes):
    values = dict(
        endpoints=("elution_observation",),
        mhc_classes=("II",),
        dropout=0,
        processing_pathways=("route_a", "route_b"),
        interventions=("added_factor",),
    )
    return ModularConfig(**(values | changes))


def context(state="present"):
    claim = ContextAssertion(state, {"source": "artificial-fixture", "scope": "material"})
    return Context(
        processing_pathways={"route_a": claim, "route_b": claim},
        interventions={"added_factor": claim},
    )


def observation(**changes):
    values = dict(
        observation_id="fixture",
        source="software-fixture",
        endpoint="elution_observation",
        peptide="ACDEFGHIKL",
        mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "II"),),
        context=context(),
        value=1,
        unit="binary",
        group_id="fixture-group",
    )
    return Observation(**(values | changes))


def test_concurrent_claims_and_unknown_override_preserve_evidence():
    supplied = replace(context(), pathway_overrides={"route_a": ContextAssertion()})
    resolved = resolve_cell_context(
        supplied,
        (),
        (),
        {},
        (),
        processing_pathways=settings().processing_pathways,
        interventions=settings().interventions,
    )
    assert resolved.processing_pathways["route_a"].state == "unknown"
    assert resolved.processing_pathways["route_b"].state == "present"
    assert resolved.interventions["added_factor"].state == "present"
    assert resolved.pathway_sources == {"route_a": "override", "route_b": "supplied"}
    assert resolved.supplied_pathways["route_a"].state == "present"
    assert resolved.supplied_pathways["route_a"].evidence["scope"] == "material"
    row = observation(context=supplied)
    assert Observation.from_dict(row.to_dict()) == row
    assert ModularConfig.from_dict(asdict(settings())) == settings()
    changed_evidence = replace(
        row,
        context=replace(
            supplied,
            interventions={"added_factor": ContextAssertion("present", {"new": "locator"})},
        ),
    )
    assert semantic_fingerprint(changed_evidence) == semantic_fingerprint(row)
    changed_state = replace(
        row, context=replace(supplied, interventions={"added_factor": ContextAssertion("absent")})
    )
    assert semantic_fingerprint(changed_state) != semantic_fingerprint(row)


def test_no_defaults_or_unknown_vocabulary_promotion():
    resolved = resolve_cell_context(
        Context(cell_type="some_cell", species={"apc": "human"}, delivery="protein"),
        (),
        (),
        {},
        (),
        processing_pathways=("route_a",),
        interventions=("added_factor",),
    )
    assert resolved.processing_pathways["route_a"].state == "unknown"
    assert resolved.interventions["added_factor"].state == "unknown"
    with pytest.raises(ValueError, match="pathway outside checkpoint"):
        resolve_cell_context(context(), (), (), {}, (), processing_pathways=("different",))
    ignored = resolve_cell_context(context(), (), (), {}, ())
    assert set(ignored.ignored_fields) == {
        "pathway:route_a",
        "pathway:route_b",
        "intervention:added_factor",
    }
    with pytest.raises(ValueError, match="context assertion state"):
        ContextAssertion("wildtype")
    with pytest.raises(ValueError, match="named context assertions"):
        Context(interventions={"factor": "present"}).validate()


def test_cellular_effect_keeps_binding_and_detector_invariant():
    torch.manual_seed(51)
    network = ModularPresto(
        settings(
            endpoints=(
                "ic50",
                "tcr_binding",
                "elution_observation",
                "bulk_observation",
                "ms_response",
            )
        )
    ).eval()
    original = observation(tcr_a="ACDEFGHIKLM", tcr_b="NPQRSTVWYAC")
    for family in ("pathway_overrides", "intervention_overrides"):
        name = "route_a" if family == "pathway_overrides" else "added_factor"
        changed = replace(context(), **{family: {name: ContextAssertion("absent")}})
        for endpoint in network.config.endpoints:
            row = replace(
                original,
                endpoint=endpoint,
                value=10 if endpoint == "ic50" else 1,
                unit="nM" if endpoint == "ic50" else "binary",
            )
            a, b = network([row, replace(row, context=changed)])
            if endpoint == "elution_observation":
                assert not torch.allclose(a.value, b.value)
            else:
                torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)
            if "ms_response" in a.components:
                torch.testing.assert_close(
                    a.components["ms_response"], b.components["ms_response"], rtol=0, atol=0
                )


def test_trainable_features_checkpoint_and_unseen_state(tmp_path):
    torch.manual_seed(53)
    network = ModularPresto(settings())
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001)
    rows = [
        observation(),
        observation(context=context("absent"), observation_id="control", value=0),
    ]
    train_step(network, optimizer, rows, {})
    labels = network.feature_layout["context_fusion.0.weight"][1]
    trained = {labels[i] for i in network.trained_features["context_fusion.0.weight"]}
    assert {
        f"{prefix}:{name}:{state}"
        for prefix, name in (
            ("pathway", "route_a"),
            ("pathway", "route_b"),
            ("intervention", "added_factor"),
        )
        for state in ("present", "absent")
    } <= trained
    expected = network.eval().predict(rows)[0].value
    path = tmp_path / "assertions.pt"
    save_checkpoint(path, network, {})
    restored, _, _ = load_checkpoint(path)
    torch.testing.assert_close(restored.predict(rows)[0].value, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="untrained input features.*unknown"):
        restored.predict(
            [
                replace(
                    rows[0],
                    context=replace(
                        context(), intervention_overrides={"added_factor": ContextAssertion()}
                    ),
                )
            ]
        )
    changed = ModularPresto(replace(settings(), interventions=("another_factor",)))
    with pytest.raises(ValueError, match="interventions"):
        initialize_from_checkpoint(changed, path)


def test_preflight_requires_training_contrasts_not_unknown_or_heldout_support(tmp_path):
    rows = [
        observation(
            peptide=peptide,
            observation_id=f"r{i}",
            group_id=f"g{i}",
            split=("train", "train", "validation", "test")[i],
            context=context("present" if i % 2 else "absent"),
            provenance={"original_measurement_id": f"m{i}"},
        )
        for i, peptide in enumerate(("ACDEFGHIKL", "MNPQRSTVWY", "ACFGHILMNP", "KLMNPQRSTV"))
    ]

    def index(records, name):
        receipt = write_observations(tmp_path / f"{name}.jsonl", records)
        manifest = {
            "schema": MANIFEST_VERSION,
            "model": asdict(settings()),
            "sources": [
                {
                    "path": receipt["path"],
                    "sha256": receipt["sha256"],
                    "source": "software-fixture",
                    "population": "artificial",
                    "label_policy": "software-only",
                    "split_policy": "distinct fixtures",
                }
            ],
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(manifest))
        return ObservationIndex(path)

    report = preflight(index(rows, "valid"))[3]
    assert report["context_assertion_states"]["pathway"]["route_a"] == ["absent", "present"]
    for field, factor in (
        ("pathway_overrides", "route_a"),
        ("intervention_overrides", "added_factor"),
    ):
        for state in ("unknown", "present"):
            altered = [
                replace(
                    row, context=replace(row.context, **{field: {factor: ContextAssertion(state)}})
                )
                if row.split == "train"
                else row
                for row in rows
            ]
            with pytest.raises(ValueError, match="lacks present/absent training contrast"):
                preflight(index(altered, f"{field}-{state}"))
