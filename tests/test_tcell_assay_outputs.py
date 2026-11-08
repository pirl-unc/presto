"""Software-only fixed-track T-cell contracts; no biological claim or real data."""

from dataclasses import asdict, replace

import pytest
import torch

from presto.data.modular import Context, MHCInput, Observation, TargetScale
from presto.data.tcell_assay import TCellTrack
from presto.models.modular_dependencies import freeze_dependencies
from presto.models.presto import Presto, PrestoConfig
from presto.inference.modular import ModularPredictor
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    observation_loss,
    observation_loss_parts,
    save_checkpoint,
    train_step,
)
from presto.training.modular_evaluation import evaluate


ACTIVATION = TCellTrack("activation", "activation", "biological_activity")
IFNG = TCellTrack("cytokine_release", "ifng_release", "elispot")
PAIR = MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I")
SCALE = {"ic50": TargetScale(2, 1, 2)}


def config(**kwargs):
    values = dict(
        endpoints=("tcell_response",),
        tcell_tracks=(ACTIVATION, IFNG),
        mhc_classes=("I",),
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0,
    )
    values.update(kwargs)
    return PrestoConfig(**values)


def row(track=ACTIVATION, value=1, intensity=None, **kwargs):
    values = dict(
        observation_id=f"{track.key}-{value}-{intensity}",
        source="software-only",
        endpoint="tcell_response",
        peptide="ACDEFGHIKL",
        mhc=(PAIR,),
        value=value,
        unit="binary",
        group_id="artificial-study",
        tcell_track=track,
        tcell_intensity=intensity,
    )
    values.update(kwargs)
    return Observation(**values)


def affinity(value=100, **kwargs):
    values = dict(
        observation_id=f"affinity-{value}",
        source="software-only",
        endpoint="ic50",
        peptide="ACDEFGHIKL",
        mhc=(PAIR,),
        value=value,
        unit="nM",
        group_id="artificial-affinity",
    )
    values.update(kwargs)
    return Observation(**values)


def elution(value=1, **kwargs):
    values = dict(
        observation_id=f"elution-{value}",
        source="software-only",
        endpoint="elution_observation",
        peptide="ACDEFGHIKL",
        mhc=(PAIR,),
        value=value,
        unit="binary",
        group_id="artificial-elution",
        context=Context(delivery="peptide"),
    )
    values.update(kwargs)
    return Observation(**values)


def test_descriptor_and_conditional_target_are_strict_and_roundtrip():
    graded = row(intensity=2)
    assert Observation.from_dict(graded.to_dict()) == graded
    assert graded.to_dict()["tcell_track"] == asdict(ACTIVATION)
    legacy = affinity()
    assert "tcell_track" not in legacy.to_dict()
    assert "tcell_intensity" not in legacy.to_dict()
    assert Observation.from_dict(legacy.to_dict()) == legacy
    with pytest.raises(ValueError, match="normalized lowercase"):
        TCellTrack("Cytokine release", "ifng", "elispot")
    with pytest.raises(ValueError, match="exact positive"):
        row(value=0, intensity=0).validate()
    with pytest.raises(ValueError, match="exact positive"):
        row(intensity=3).validate()
    with pytest.raises(ValueError, match="T-cell output descriptors"):
        replace(legacy, tcell_track=ACTIVATION).validate()
    with pytest.raises(ValueError, match="require the T-cell response"):
        PrestoConfig(tcell_tracks=(ACTIVATION,))
    with pytest.raises(ValueError, match="duplicate T-cell"):
        config(tcell_tracks=(ACTIVATION, ACTIVATION))
    model = Presto(config(tcell_tracks=(ACTIVATION,)))
    with pytest.raises(ValueError, match="outside checkpoint registry"):
        model([row(IFNG)])
    with pytest.raises(ValueError, match="requires a declared output track"):
        model([replace(row(), tcell_track=None)])


def test_fixed_panel_is_counterfactual_and_descriptor_is_output_only():
    torch.manual_seed(17)
    model = Presto(config()).eval()
    activation, ifng = model([row(ACTIVATION), row(IFNG)])
    expected = {ACTIVATION.key, IFNG.key}
    assert set(activation.tcell_values) == expected
    assert set(activation.tcell_intensity_values) == expected
    for key in expected:
        assert torch.equal(activation.tcell_values[key], ifng.tcell_values[key])
        assert torch.equal(activation.tcell_intensity_values[key], ifng.tcell_intensity_values[key])
    assert torch.equal(activation.value, activation.tcell_values[ACTIVATION.key])
    assert torch.equal(ifng.value, ifng.tcell_values[IFNG.key])
    assert activation.reference_value is ifng.reference_value is None
    left = activation.diagnostics["candidates"]
    right = ifng.diagnostics["candidates"]
    assert left[0]["registers"][0]["starts"] == right[0]["registers"][0]["starts"]
    assert torch.equal(
        left[0]["registers"][0]["candidate_scores"],
        right[0]["registers"][0]["candidate_scores"],
    )
    thresholds = model.tcell_assay_layer.ordinal_thresholds
    assert thresholds[0] < thresholds[1]
    for logits in activation.tcell_intensity_values.values():
        assert logits[0] > logits[1]


def test_selected_track_and_graded_rows_receive_only_declared_losses():
    model = Presto(config())
    graded = row(ACTIVATION, intensity=2)
    output = model([graded])[0]
    output.tcell_values[IFNG.key].retain_grad()
    output.tcell_intensity_values[IFNG.key].retain_grad()
    parts = observation_loss_parts(output, graded, {})
    assert set(parts) == {"primary", "conditional_ordinal"}
    observation_loss(output, graded, {}).backward()
    assert model.tcell_assay_layer.factors["family"][ACTIVATION.family].grad.norm() > 0
    assert model.tcell_assay_layer.factors["family"][IFNG.family].grad is None
    assert model.tcell_assay_layer.positivity_decoder[0].weight.grad.norm() > 0
    assert model.tcell_assay_layer.intensity_decoder[0].weight.grad.norm() > 0
    assert model.tcell_assay_layer.ordinal_threshold_base.grad.abs() > 0
    assert output.tcell_values[IFNG.key].grad is None
    assert output.tcell_intensity_values[IFNG.key].grad is None

    binary_only = Presto(config(tcell_ordinal_loss_weight=0))
    optimizer = torch.optim.AdamW(binary_only.parameters(), lr=0.01)
    train_step(binary_only, optimizer, [graded], {})
    assert binary_only.tcell_assay_layer.intensity_decoder[0].weight.grad is None
    assert binary_only.tcell_assay_layer.ordinal_threshold_base.grad is None


def test_parent_extension_updates_only_new_tcell_parameters_and_roundtrips(tmp_path):
    torch.manual_seed(23)
    parent = Presto(
        PrestoConfig(
            mhc_classes=("I",),
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0,
        )
    )
    train_step(
        parent,
        torch.optim.AdamW(parent.parameters(), lr=0.01),
        [affinity(10), affinity(1000)],
        SCALE,
    )
    parent.eval()
    probe = affinity(value=None, unit=None, split="predict")
    with torch.no_grad():
        expected = parent([probe])[0].value.detach().clone()
    parent_state = {name: value.detach().clone() for name, value in parent.state_dict().items()}
    path = tmp_path / "parent.pt"
    save_checkpoint(path, parent, SCALE)

    extended = Presto(config(endpoints=("ic50", "tcell_response")))
    report = initialize_from_checkpoint(extended, path)
    assert set(report["added_tcell_tracks"]) == {ACTIVATION.key, IFNG.key}
    assert report["removed_tcell_tracks"] == []
    freeze_dependencies(extended, ["ic50"])
    assert all(
        name.startswith("tcell_assay_layer.")
        for name, parameter in extended.named_parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in extended.parameters() if parameter.requires_grad], lr=0.01
    )
    train_step(
        extended,
        optimizer,
        [row(ACTIVATION, intensity=0), row(IFNG, value=0)],
        SCALE,
    )
    train_step(
        extended,
        optimizer,
        [row(ACTIVATION, intensity=2), row(IFNG, intensity=1)],
        SCALE,
    )
    assert all(
        torch.equal(value, extended.state_dict()[name]) for name, value in parent_state.items()
    )
    extended.eval()
    with torch.no_grad():
        assert torch.equal(expected, extended([probe])[0].value)
    assert all(
        value["status"] == "directly_supervised"
        and value["conditional_ordinal_status"] == "directly_supervised"
        for value in extended.tcell_capabilities().values()
    )

    final = tmp_path / "extended.pt"
    save_checkpoint(final, extended, SCALE, optimizer=optimizer)
    restored, scales, _ = load_checkpoint(final)
    assert scales == SCALE
    assert restored.tcell_support == extended.tcell_support
    restored.eval()
    for before, after in zip(extended([row(ACTIVATION)]), restored([row(ACTIVATION)]), strict=True):
        assert torch.equal(before.value, after.value)
        for key in before.tcell_values:
            assert torch.equal(before.tcell_values[key], after.tcell_values[key])
            assert torch.equal(
                before.tcell_intensity_values[key], after.tcell_intensity_values[key]
            )
    request = replace(row(ACTIVATION), value=None, unit=None, tcell_intensity=None, split="predict")
    predicted = ModularPredictor(final).predict([request])[0]
    assert predicted["selected_tcell_track"] == ACTIVATION.key
    assert set(predicted["tcell_panel"]) == {ACTIVATION.key, IFNG.key}
    assert all(
        value["conditional_intensity"]["levels"] == ["low", "intermediate", "high"]
        and len(value["conditional_intensity"]["probabilities"]) == 3
        for value in predicted["tcell_panel"].values()
    )
    counterfactual = ModularPredictor(final).predict([replace(request, tcell_track=None)])[0]
    assert counterfactual["selected_tcell_track"] is None
    assert set(counterfactual["tcell_panel"]) == {ACTIVATION.key, IFNG.key}


def test_staged_presentation_parent_shapes_and_predictions_survive_allocation(tmp_path):
    torch.manual_seed(29)
    settings = dict(
        endpoints=("ic50", "elution_observation"),
        mhc_classes=("I",),
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_detach_affinity=True,
    )
    parent = Presto(PrestoConfig(**settings)).eval()
    probes = [
        affinity(value=None, unit=None, split="predict"),
        elution(value=None, unit=None, split="predict"),
    ]
    with torch.no_grad():
        expected = [output.value.detach().clone() for output in parent(probes)]
    path = tmp_path / "staged-parent.pt"
    save_checkpoint(path, parent, SCALE)
    extended = Presto(
        PrestoConfig(
            **{
                **settings,
                "endpoints": (*settings["endpoints"], "tcell_response"),
                "tcell_tracks": (ACTIVATION, IFNG),
            }
        )
    )
    initialize_from_checkpoint(extended, path)
    assert set(parent.state_dict()) < set(extended.state_dict())
    assert all(
        parent.state_dict()[name].shape == extended.state_dict()[name].shape
        and torch.equal(parent.state_dict()[name], extended.state_dict()[name])
        for name in parent.state_dict()
    )
    assert not hasattr(extended, "context_fusion")
    extended.eval()
    with torch.no_grad():
        actual = [output.value for output in extended(probes)]
    assert all(torch.equal(before, after) for before, after in zip(expected, actual, strict=True))


def test_evaluation_preserves_track_panels_and_ordinal_metrics():
    model = Presto(config()).eval()
    rows = [row(ACTIVATION, intensity=0), row(ACTIVATION, intensity=2)]
    metrics, records = evaluate(model, rows, [0, 1], {})
    assert records[0]["tcell_track"] == asdict(ACTIVATION)
    assert set(records[0]["tcell_panel"]) == {ACTIVATION.key, IFNG.key}
    ordinal = metrics["tcell_response"]["conditional_intensity"]
    assert ordinal["n"] == 2
    assert 0 <= ordinal["within_one_level"] <= 1
    assert set(metrics["tcell_response"]["by_tcell_track"]) == {ACTIVATION.key}


def test_empty_configuration_has_no_tcell_state_or_serialized_contract_delta():
    model = Presto(PrestoConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32))
    assert not hasattr(model, "tcell_assay_layer")
    assert not any(name.startswith("tcell_assay_layer.") for name in model.state_dict())
    assert model.tcell_support == {}
    assert "tcell_tracks" not in model.config.to_dict()
    assert "tcell_ordinal_loss_weight" not in model.config.to_dict()
    restored = PrestoConfig.from_dict(model.config.to_dict())
    assert restored == model.config
