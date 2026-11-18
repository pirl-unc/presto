"""Shared source processing can condition on class without changing affinity inputs."""

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest
import torch

from presto.data.modular import Context
from presto.inference.modular import ModularPredictor
from presto.models.modular import ModularConfig
from presto.models.modular_dependencies import parameter_dependencies
from presto.training.modular import initialize_from_checkpoint, save_checkpoint, train_step
from presto.training.modular_run import initialized_model
from test_apm_state_bottleneck import build, elution, scales


def network(**kwargs):
    return build(d_model=16, n_heads=2, n_layers=1, d_ff=32, source_context=True, **kwargs).eval()


def examples():
    return [
        replace(elution(cls, oid=cls), context=Context(source_n="ACDEF", source_c="GHIKL"))
        for cls in ("I", "II")
    ]


def processing(output):
    return output.diagnostics["candidates"][0]["registers"][0]["elution_terms"]["processing"]


def test_zero_initialization_preserves_existing_weights_rng_and_outputs():
    old = network(seed=17)
    old_rng = torch.get_rng_state().clone()
    new = network(seed=17, elution_pathway_conditioning=True)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert "elution_pathway_conditioning" not in old.config.to_dict()
    assert new.config.to_dict()["elution_pathway_conditioning"] is True
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    for a, b in zip(old(examples()), new(examples()), strict=True):
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="must be boolean"):
        network(elution_pathway_conditioning=1)
    with pytest.raises(ValueError, match="requires staged elution"):
        ModularConfig(elution_pathway_conditioning=True)


def test_pathway_conditions_shared_processing_and_never_intrinsic_affinity():
    net = network(elution_pathway_conditioning=True)
    rows = examples()
    affinity = [replace(r, endpoint="ic50", value=100.0, unit="nM") for r in rows]
    before_affinity = [r.value.detach().clone() for r in net(affinity)]
    with torch.no_grad():
        net.elution_processing[0].weight.zero_()
        net.elution_processing[0].bias.zero_()
        net.elution_processing[2].weight.fill_(1)
        net.elution_pathway.weight[:, 1].fill_(0.5)
    first, second = net(rows)
    assert processing(first).item() == 0
    assert processing(second).item() > 1
    for expected, actual in zip(before_affinity, net(affinity), strict=True):
        torch.testing.assert_close(expected, actual.value, rtol=0, atol=0)
    assert "elution_pathway.weight" in parameter_dependencies(net, "elution_observation")
    assert "elution_pathway.weight" not in parameter_dependencies(net, "ic50")


def test_each_class_trains_shared_source_processing_and_its_pathway_column():
    for index, row in enumerate(examples()):
        net = network(elution_pathway_conditioning=True)
        before = {n: p.detach().clone() for n, p in net.named_parameters()}
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0)
        train_step(net, optimizer, [row], scales())
        for name in (
            "encoder.embedding.weight",
            "elution_processing.0.weight",
            "elution_processing.2.weight",
        ):
            parameter = net.get_parameter(name)
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
            assert not torch.equal(parameter, before[name]), (row.mhc[0].mhc_class, name)
        assert net.elution_pathway.weight[:, index].abs().sum() > 0
        assert net.elution_pathway.weight[:, 1 - index].abs().sum() == 0
        assert net.trained_features["elution_pathway.weight"] == {index}
        with pytest.raises(ValueError, match="untrained input features: processing_class"):
            net.predict([examples()[1 - index]])


def test_expansion_requires_explicit_flag_and_roundtrips_public_inference(tmp_path):
    old = network()
    old_path = tmp_path / "old.pt"
    save_checkpoint(old_path, old, scales())
    new = network(seed=3, elution_pathway_conditioning=True)
    with pytest.raises(ValueError, match="explicit allow_pathway_expansion"):
        initialize_from_checkpoint(new, old_path)
    initialize_from_checkpoint(new, old_path, allow_pathway_expansion=True)
    for a, b in zip(old(examples()), new(examples()), strict=True):
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)
    rows = examples()
    train_step(new, torch.optim.AdamW(new.parameters(), lr=0.01), rows, scales())
    path = tmp_path / "new.pt"
    save_checkpoint(path, new, scales())
    with pytest.raises(ValueError, match="cannot remove pretrained pathway conditioning"):
        initialize_from_checkpoint(network(), path)
    restored = ModularPredictor(path)
    queries = [replace(r, value=None, split="predict") for r in rows]
    result = restored.predict(queries)
    assert len(result) == 2
    for expected, actual in zip(new.predict(queries), result, strict=True):
        assert actual["logit"] == expected.value.item()
        assert actual["diagnostics"]["elution_staging"]["pathway_conditioned"]


def test_manifest_initialization_honors_explicit_expansion(tmp_path):
    path = tmp_path / "parent.pt"
    save_checkpoint(path, network(), scales())
    reference = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "freeze": [],
        "allow_pathway_expansion": True,
    }
    index = SimpleNamespace(
        manifest_path=tmp_path / "manifest.json", manifest={"pretrained": reference}
    )
    model, inherited = initialized_model(network(elution_pathway_conditioning=True).config, index)
    assert inherited and not model.frozen_parameters
    assert model.elution_pathway.weight.count_nonzero() == 0
    reference["allow_pathway_expansion"] = "true"
    with pytest.raises(ValueError, match="must be boolean"):
        initialized_model(model.config, index)


def test_pathway_conditioning_preserves_harmonized_gene_scope():
    from test_harmonized_apc import build as harmonized, randomize

    net = harmonized(elution_pathway_conditioning=True)
    randomize(net)
    with torch.no_grad():
        net.elution_pathway.weight.fill_(0.5)
    for cls, factor in (("I", "transduction:CIITA"), ("II", "knockout:B2M")):
        a, b = net([elution(cls), elution(cls, {factor: "present"})])
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)
