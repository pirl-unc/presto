"""Behavioral checks for scalable dimensions and honest coordinate evidence."""

from dataclasses import replace

import pytest
import torch

from presto.data.modular import Context, MHCInput, Observation, TargetScale
from presto.models.presto import Presto, PrestoConfig
from presto.models.modular_support import parameter_report
from presto.training.modular import (
    train_step,
    save_checkpoint,
    load_checkpoint,
    initialize_from_checkpoint,
)


def fixture():
    return Observation(
        'software',
        'artificial-fixture',
        'ic50',
        'ACDEFGHIKLMN',
        (MHCInput('ACDEFGHIK', 'LMNPQRSTV', 'II'),),
        value=100.0,
        unit='nM',
        group_id='g',
    )


@pytest.mark.parametrize('capacity', [(32, 1, 2, 64), (192, 3, 6, 768)])
def test_capacity_applies_to_every_reachable_route(capacity):
    width, depth, heads, ff = capacity
    config = PrestoConfig(
        endpoints=(
            'ic50',
            'tcr_binding',
            'elution_observation',
            'digest_observation',
            'ms_response',
        ),
        source_context=True,
        d_model=width,
        n_layers=depth,
        n_heads=heads,
        d_ff=ff,
        dropout=0,
    )
    model = Presto(config)
    rows = [
        fixture(),
        replace(
            fixture(),
            endpoint='tcr_binding',
            value=1,
            unit='binary',
            tcr_a='ACDEFGHIK',
            tcr_b='LMNPQRSTV',
        ),
        replace(
            fixture(),
            endpoint='elution_observation',
            value=1,
            unit='binary',
            context=Context(source_n='AAA'),
        ),
        replace(fixture(), endpoint='digest_observation', mhc=(), value=1, unit='binary'),
        replace(fixture(), endpoint='ms_response', mhc=(), value=1, unit='binary'),
    ]
    loss = sum(p.value.square() for p in model(rows))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.embedding.weight.shape[1] == width
    assert len(model.encoder.transformer.layers) == depth
    assert all(p.grad is not None for p in model.parameters())
    for name, (axis, labels) in model.feature_layout.items():
        assert len(labels) == model.get_parameter(name).shape[axis]


@pytest.mark.parametrize(
    'override', [dict(d_model=31), dict(d_model=32, n_heads=3), dict(n_layers=0), dict(d_ff=True)]
)
def test_invalid_capacity_rejected(override):
    with pytest.raises(ValueError, match="positive integer|divisible"):
        PrestoConfig(**override)


def test_tensor_support_requires_same_coordinate_gradient_and_update():
    model = Presto(PrestoConfig(dropout=0))
    weight = model.heads['ic50'].weight

    def first_only(gradient):
        masked = torch.zeros_like(gradient)
        masked[:, 0] = gradient[:, 0]
        return masked

    weight.register_hook(first_only)

    class DisjointUpdate(torch.optim.SGD):
        @torch.no_grad()
        def step(self, closure=None):
            old = weight.clone()
            result = super().step(closure)
            weight.copy_(old)
            weight[:, 1].add_(0.25)
            return result

    result = train_step(
        model,
        DisjointUpdate(model.parameters(), lr=0.01),
        [fixture()],
        {'ic50': TargetScale(2.0, 1.0, 2)},
    )
    evidence = result['coverage']['heads.ic50.weight']
    assert evidence['gradient_norm'] > 0 and evidence['update_norm'] > 0
    assert not evidence['gradient_and_update']
    assert 'heads.ic50.weight' not in model.trained_parameters
    assert not model.trained_elements['heads.ic50.weight'].any()


def test_element_masks_roundtrip_and_historical_unknown(tmp_path):
    model = Presto(PrestoConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0))
    scales = {'ic50': TargetScale(2.0, 1.0, 2)}
    train_step(model, torch.optim.AdamW(model.parameters()), [fixture()], scales)
    path = tmp_path / 'trained.pt'
    save_checkpoint(path, model, scales)
    restored, _, payload = load_checkpoint(path)
    assert parameter_report(restored) == parameter_report(model)
    for name in model.trained_elements:
        torch.testing.assert_close(model.trained_elements[name], restored.trained_elements[name])
    expanded = Presto(replace(model.config, endpoints=('ic50', 'kd')))
    initialize_from_checkpoint(expanded, path)
    torch.testing.assert_close(
        expanded.trained_elements['heads.ic50.weight'], model.trained_elements['heads.ic50.weight']
    )
    assert not expanded.trained_elements['heads.kd.weight'].any()
    payload.pop('element_coverage')
    torch.save(payload, path)
    historical, _, _ = load_checkpoint(path)
    assert not any(r['complete_history'] for r in parameter_report(historical).values())
    model.eval()
    historical.eval()
    torch.testing.assert_close(
        model([fixture()])[0].value, historical([fixture()])[0].value, rtol=0, atol=0
    )
    payload['element_coverage'] = {'masks': {}, 'complete_history': []}
    torch.save(payload, path)
    with pytest.raises(ValueError, match='element coverage'):
        load_checkpoint(path)


def test_attention_key_bias_is_redundant_at_evaluation():
    torch.manual_seed(4)
    attention = torch.nn.MultiheadAttention(16, 2, batch_first=True, dropout=0).double().eval()
    q = torch.randn(2, 3, 16, dtype=torch.float64)
    kv = torch.randn(2, 7, 16, dtype=torch.float64)
    before = attention(q, kv, kv, need_weights=False)[0]
    with torch.no_grad():
        attention.in_proj_bias[16:32].add_(torch.linspace(-10, 10, 16, dtype=torch.float64))
    after = attention(q, kv, kv, need_weights=False)[0]
    torch.testing.assert_close(before, after, rtol=1e-12, atol=1e-12)
