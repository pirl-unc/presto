"""Boundary semantics, true output influence and parent-checkpoint compatibility."""

from dataclasses import replace
import pytest
import torch
from presto.data.modular import Context
from presto.models.junction_processing import JunctionProcessing
from presto.training.modular import initialize_from_checkpoint, save_checkpoint, load_checkpoint
from test_harmonized_apc import build
from test_apm_state_bottleneck import elution, scales


def example(cls='I'):
    return replace(
        elution(cls),
        context=Context(source_n='RST', source_c='VWY', n_terminus=False, c_terminus=False),
    )


def test_optional_module_preserves_existing_weights_rng_and_outputs(tmp_path):
    base = build(seed=9, source_context=True)
    rng = torch.random.get_rng_state()
    expanded = build(seed=9, source_context=True, elution_junction_processing=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert 'elution_junction_processing' not in base.config.to_dict()
    for name, value in base.state_dict().items():
        assert torch.equal(value, expanded.state_dict()[name]), name
    examples = [example('I'), example('II')]
    for a, b in zip(base(examples), expanded(examples)):
        assert torch.equal(a.value, b.value)
    path = tmp_path / 'parent.pt'
    save_checkpoint(path, base, scales())
    with pytest.raises(ValueError, match='explicit allow_junction'):
        initialize_from_checkpoint(expanded, path)
    initialize_from_checkpoint(expanded, path, allow_junction_expansion=True)
    save_checkpoint(tmp_path / 'expanded.pt', expanded, scales())
    reloaded, _, _ = load_checkpoint(tmp_path / 'expanded.pt')
    for a, b in zip(expanded(examples), reloaded.eval()(examples)):
        assert torch.equal(a.value, b.value)


def test_ordered_boundary_features_distinguish_missing_and_true_end():
    module = JunctionProcessing(2)
    r = example()
    states = {
        r.peptide: torch.arange(30.0).reshape(15, 2),
        'RST': torch.arange(6.0).reshape(3, 2) + 50,
        'VWY': torch.arange(6.0).reshape(3, 2) + 80,
    }
    features = module.boundary_features(r, states, 'I', 'N')
    ordered = features[:24].reshape(12, 2)
    assert torch.equal(ordered[:3], torch.zeros(3, 2))
    assert torch.equal(ordered[3:6], states['RST'])
    assert torch.equal(ordered[6:], states[r.peptide][:6])
    flipped = module.boundary_features(r, {**states, 'RST': states['RST'].flip(0)}, 'I', 'N')
    assert not torch.equal(features, flipped)
    missing = replace(r, context=Context())
    assert module.boundary_features(missing, states, 'I', 'N') is None
    end = replace(r, context=Context(source_n='', n_terminus=True))
    assert module.boundary_features(end, states, 'I', 'N') is not None
    with torch.no_grad():
        module.readout.weight.fill_(0.5)
    assert module(missing, states, 'I')[0].item() == 0
    assert module(end, states, 'I')[0].item() != 0


@pytest.mark.parametrize('cls', ['I', 'II'])
def test_junction_drives_presentation_and_encoder_but_not_intrinsic_affinity(cls):
    net = build(source_context=True, elution_junction_processing=True)
    r = example(cls)
    before = net([r])[0].value.detach().clone()
    affinity = replace(r, endpoint='ic50', unit='nM', value=100.0)
    binding_before = net([affinity])[0].value.detach().clone()
    out = net([r])[0]
    out.value.backward()
    assert net.elution_junction.readout.weight.grad.abs().sum() > 0
    with torch.no_grad():
        net.elution_junction.readout.weight.add_(0.1)
    net.zero_grad(set_to_none=True)
    after = net([r])[0]
    assert not torch.equal(before, after.value)
    assert torch.equal(binding_before, net([affinity])[0].value)
    after.value.backward()
    assert net.elution_junction.input.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in net.named_parameters()
        if n.startswith('encoder.')
    )
    assert torch.allclose(after.value, sum(after.components.values()))
