"""Staged elution: binding/stability enter only via their own scores (flagged)."""

from dataclasses import asdict, replace

import pytest
import torch

from presto.data.modular import MHCInput, Observation, TargetScale
from presto.models.modular import ModularConfig, ModularPresto
from presto.models.modular_dependencies import freeze_dependencies, parameter_dependencies
from presto.training.modular import (
    batch_loss,
    coverage_summary,
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from presto.training.modular_resume import canonical

STAGED = dict(elution_staging="affinity_downstream", elution_affinity_endpoint="ic50")


def row(endpoint="ic50", *, observation_id="r1", **kwargs):
    defaults = dict(
        observation_id=observation_id,
        source="software-fixture",
        endpoint=endpoint,
        peptide="ACDEFGHIKLMNPQR",
        mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "II"),),
        value=100.0,
        unit="nM",
        group_id="g1",
    )
    defaults.update(kwargs)
    return Observation(**defaults)


def elution(**kwargs):
    kwargs.setdefault("observation_id", "e1")
    return row("elution_observation", value=1, unit="binary", **kwargs)


def half_life():
    return row("half_life", observation_id="h1", value=10.0, unit="min")


def scales():
    return {"ic50": TargetScale(2.0, 1.0, 2), "half_life": TargetScale(1.0, 1.0, 2)}


def build(seed=0, **kwargs):
    kwargs.setdefault("endpoints", ("ic50", "half_life", "elution_observation"))
    torch.manual_seed(seed)
    return ModularPresto(ModularConfig(dropout=0, **kwargs))


def elution_loss(network, rows):
    network.train()
    loss, _ = batch_loss(network, rows, network(rows), scales())
    network.zero_grad(set_to_none=True)
    loss.backward()


def has_grad(network, prefix):
    grads = [p.grad for n, p in network.named_parameters() if n.startswith(prefix)]
    assert grads, prefix
    return any(g is not None and g.abs().sum() > 0 for g in grads)


def test_flag_off_is_identical_to_default_and_contract_is_unchanged():
    default = build(seed=3)
    explicit = build(seed=3, elution_staging="legacy_joint")
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    assert hasattr(default, "cellular_score") and not hasattr(default, "elution_processing")
    rows = [row(), elution()]
    first, second = default.eval()(rows), explicit.eval()(rows)
    for a, b in zip(first, second):
        assert torch.equal(a.value, b.value)
    assert "cellular" in first[1].components
    # Legacy contract carries no staging keys, so existing digests do not move.
    assert default.config.to_dict() == {
        k: v for k, v in asdict(default.config).items() if not k.startswith("elution_")
    }
    with pytest.raises(ValueError, match="require elution_staging"):
        ModularConfig(endpoints=("ic50", "elution_observation"), elution_detach_affinity=True)


def test_config_validation_for_staged_mode():
    with pytest.raises(ValueError, match="affinity endpoint"):
        ModularConfig(
            endpoints=("ic50", "elution_observation"), elution_staging="affinity_downstream"
        )
    with pytest.raises(ValueError, match="elution_observation"):
        ModularConfig(endpoints=("ic50",), **STAGED)
    with pytest.raises(ValueError, match="elution_staging"):
        ModularConfig(elution_staging="bogus")


def test_staged_structure_and_prediction_terms():
    network = build(**STAGED)
    assert not hasattr(network, "cellular_score") and not hasattr(network, "context_fusion")
    assert network.elution_affinity_weight.item() > 0
    output = network.eval()([elution()])[0]
    assert set(output.components) == {"elution_presentation", "ms_response"}
    terms = output.diagnostics["candidates"][0]["registers"][0]["elution_terms"]
    assert set(terms) == {"binding", "processing", "stability"}
    info = output.diagnostics["elution_staging"]
    assert info["processing_term"] == "processing_peptide_only"
    assert info["processing_identified"] is False
    torch.testing.assert_close(output.components["elution_presentation"], sum(terms.values()))
    # Without a stability latent that term is omitted entirely.
    plain = build(endpoints=("ic50", "elution_observation"), **STAGED)
    assert not hasattr(plain, "elution_stability_weight")
    registers = plain.eval()([elution()])[0].diagnostics["candidates"][0]["registers"][0]
    assert "stability" not in registers["elution_terms"]
    sourced = build(source_context=True, **STAGED).eval()([elution()])[0]
    assert sourced.diagnostics["elution_staging"]["processing_term"] == "processing"


def test_elution_logit_is_scaled_negative_affinity_score_for_both_classes():
    network = build(**STAGED).eval()
    with torch.no_grad():
        network.elution_stability_weight.zero_()
        network.offsets["elution_observation"].bias.zero_()
        for module in (network.elution_processing, network.ms_fusion, network.ms_score):
            for p in module.parameters():
                p.zero_()
    for mhc_class, peptide in (("I", "ACDEFGHIK"), ("II", "ACDEFGHIKLMNPQR")):
        mhc = (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", mhc_class),)
        affinity = network([row(mhc=mhc, peptide=peptide)])[0].reference_value
        presented = network([elution(mhc=mhc, peptide=peptide)])[0].value
        torch.testing.assert_close(presented, -network.elution_affinity_weight * affinity)


def test_elution_gradient_reaches_affinity_through_scale_not_old_path():
    network = build(**STAGED)
    elution_loss(network, [elution()])
    assert network.elution_affinity_weight.grad.abs() > 0
    for prefix in (
        "assay_layer.decoder",
        "pmhc_fusion",
        "elution_processing",
        "heads.half_life",
        "elution_stability_weight",
    ):
        assert has_grad(network, prefix), prefix
    assert not any(n.startswith("cellular_score") for n, _ in network.named_parameters())
    legacy = build()
    elution_loss(legacy, [elution()])
    assert has_grad(legacy, "cellular_score")
    assert not has_grad(legacy, "assay_layer.decoder")


def test_detach_stops_elution_gradient_into_affinity_parameters():
    network = build(elution_detach_affinity=True, **STAGED)
    elution_loss(network, [elution()])
    assert not has_grad(network, "assay_layer")
    assert network.elution_affinity_weight.grad.abs() > 0
    assert has_grad(network, "elution_processing")
    only = build(endpoints=("ic50", "elution_observation"), elution_detach_affinity=True, **STAGED)
    elution_loss(only, [elution()])
    assert not has_grad(only, "assay_layer")
    assert not has_grad(only, "pmhc_fusion")
    assert not has_grad(only, "residue_interaction")


def test_coverage_gate_dependencies_and_freezing():
    network = build(**STAGED)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2)
    result = train_step(network, optimizer, [row(), half_life(), elution()], scales())
    assert not coverage_summary([result])["untrained_parameters"]
    assert {"elution_affinity_weight", "elution_stability_weight"} <= network.trained_parameters
    deps = parameter_dependencies(network, "elution_observation")
    assert {
        "elution_affinity_weight",
        "elution_stability_weight",
        "ms_score.weight",
        "assay_layer.factors.endpoint.ic50",
    } <= deps
    assert any(n.startswith("elution_processing.") for n in deps)
    assert any(n.startswith("heads.half_life.") for n in deps)
    assert not any(n.startswith("context_fusion") for n in deps)
    assert "elution_processing.0.weight" in network.feature_layout
    assert "context_fusion.0.weight" not in network.feature_layout
    # Served only once the whole staged path has update evidence.
    with pytest.raises(ValueError, match="no trained"):
        build(**STAGED).predict([elution()])
    assert not network.predict([elution()])[0].components["elution_presentation"].requires_grad
    freeze_dependencies(network, ["component:cellular"])
    assert "elution_affinity_weight" in network.frozen_parameters
    assert not network.elution_affinity_weight.requires_grad


def test_detached_elution_alone_leaves_affinity_parameters_unchanged():
    # No stability latent here: that term would (by design) still train the shared trunk.
    network = build(
        endpoints=("ic50", "elution_observation"), elution_detach_affinity=True, **STAGED
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2)
    before = {n: p.detach().clone() for n, p in network.named_parameters()}
    train_step(network, optimizer, [elution()], scales())
    for name, parameter in network.named_parameters():
        if name.startswith(("assay_layer", "pmhc_fusion", "residue_interaction")):
            assert torch.equal(parameter, before[name]), name
    assert not torch.equal(
        network.elution_affinity_weight.detach(), before["elution_affinity_weight"]
    )


def test_checkpoint_round_trip_and_contract_hash(tmp_path):
    network = build(**STAGED)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2)
    train_step(network, optimizer, [row(), half_life(), elution()], scales())
    expected = network.predict([elution()])[0].value
    path = tmp_path / "staged.pt"
    save_checkpoint(path, network, scales(), optimizer=optimizer)
    restored, _, payload = load_checkpoint(path)
    assert restored.config == network.config and restored.staged_elution
    torch.testing.assert_close(restored.predict([elution()])[0].value, expected)
    assert payload["config"]["elution_staging"] == "affinity_downstream"
    assert restored.trained_parameters == network.trained_parameters
    legacy = canonical(build().config.to_dict())
    assert legacy == canonical(build(elution_staging="legacy_joint").config.to_dict())
    assert canonical(network.config.to_dict()) != legacy
    detached = replace(network.config, elution_detach_affinity=True)
    assert canonical(detached.to_dict()) != canonical(network.config.to_dict())


def test_affinity_checkpoint_can_add_staged_elution_without_changing_affinity(tmp_path):
    from presto.data.assay_measurement import AssayMethod, AssayTrack

    tracks = tuple(
        AssayTrack("ic50", "purified", "competitive", detector)
        for detector in ("radioactivity", "fluorescence")
    )
    observed = [
        row(
            observation_id=f"{cls}-{track.detector}",
            mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", cls),),
            assay_method=AssayMethod(
                track.preparation, track.mechanism, track.detector, "source_tag_unreviewed"
            ),
            value=10 if cls == "I" else 1000,
        )
        for cls in ("I", "II")
        for track in tracks
    ]
    prior = build(endpoints=("ic50",), assay_tracks=tracks)
    scale = {"ic50": scales()["ic50"]}
    train_step(prior, torch.optim.AdamW(prior.parameters(), lr=1e-3), observed, scale)
    path = tmp_path / "affinity.pt"
    save_checkpoint(path, prior, scale)
    expanded = build(
        endpoints=("ic50", "elution_observation"),
        assay_tracks=tracks,
        source_context=True,
        elution_detach_affinity=True,
        **STAGED,
    )
    report = initialize_from_checkpoint(expanded, path)
    assert report["scales"] == scale
    assert set(report["inherited_parameters"]) == set(prior.state_dict())
    assert "elution_affinity_weight" not in expanded.trained_parameters
    assert expanded.assay_support == prior.assay_support
    with torch.no_grad():
        expected = prior.eval()(observed)

    def unchanged(model):
        # Compare the same inference kernel path before/after dependency freezing.
        with torch.no_grad():
            actual = model.eval()(observed)
        for before, after in zip(expected, actual):
            assert torch.equal(before.value, after.value)
            assert torch.equal(before.reference_value, after.reference_value)
            assert before.assay_values.keys() == after.assay_values.keys()
            for key in before.assay_values:
                assert torch.equal(before.assay_values[key], after.assay_values[key])

    unchanged(expanded)
    freeze_dependencies(expanded, ["ic50"])
    inherited = {name: value.clone() for name, value in prior.state_dict().items()}
    initial_processing = expanded.elution_processing[0].weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in expanded.parameters() if p.requires_grad), lr=1e-2)
    train_step(expanded, optimizer, [elution()], scale)
    assert not torch.equal(initial_processing, expanded.elution_processing[0].weight)
    for name, value in inherited.items():
        assert torch.equal(value, expanded.state_dict()[name]), name
    unchanged(expanded)
    save_checkpoint(tmp_path / "expanded.pt", expanded, scale)
    restored, retained_scales, _ = load_checkpoint(tmp_path / "expanded.pt")
    assert retained_scales == scale
    unchanged(restored)


@pytest.mark.parametrize("change", ["mode", "endpoint", "detach"])
def test_existing_elution_contract_cannot_change_during_transfer(tmp_path, change):
    common = dict(endpoints=("ic50", "kd", "elution_observation"))
    prior = build(**common, **STAGED)
    path = tmp_path / "staged.pt"
    save_checkpoint(path, prior, {"ic50": scales()["ic50"], "kd": scales()["ic50"]})
    mode = dict(STAGED)
    if change == "mode":
        mode = {}
    elif change == "endpoint":
        mode["elution_affinity_endpoint"] = "kd"
    else:
        mode["elution_detach_affinity"] = True
    with pytest.raises(ValueError, match="incompatible pretrained feature contract: elution_"):
        initialize_from_checkpoint(build(**common, **mode), path)
