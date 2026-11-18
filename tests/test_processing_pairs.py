"""Pairs cannot use binding/MS shortcuts or leak held-out labels into training."""

from dataclasses import asdict, replace
import copy
import random
from types import SimpleNamespace
import pytest
import torch
from presto.data.affinity_matching import MatchingPolicy, candidate_key
from presto.data.modular import Context
from presto.training.processing_pairs import PairPool, validate_pair, pair_loss, processing_supply
from presto.training.modular import train_step
from test_harmonized_apc import build
from test_apm_state_bottleneck import elution, scales


def pair(cls='I', suffix=''):
    context = Context(source_n='RST', source_c='VWY', n_terminus=False, c_terminus=False)
    p = replace(
        elution(cls, oid='p' + cls + suffix),
        context=context,
        provenance={
            'study_id': 'study' + suffix,
            'source_protein': 'protein',
            'original_measurement_id': 'p' + cls + suffix,
        },
    )
    key = candidate_key(p.mhc[0])
    b = replace(
        p,
        observation_id='b' + cls + suffix,
        peptide='RSTVWYACDEFGHIK',
        value=0,
        provenance={
            **p.provenance,
            'synthetic': True,
            'original_measurement_id': 'b' + cls + suffix,
            'processing_pair_positive_id': p.observation_id,
            'affinity_match': {
                'schema': 'presto.affinity_match.v1',
                'policy': asdict(MatchingPolicy()),
                'predictor': {
                    'name': 'NetMHCpan' if cls == 'I' else 'NetMHCIIpan',
                    'version': 'fixture',
                },
                'source_sha256': 's',
                'score_sha256': 's',
                'positive_scores': {key: {'nm': 100, 'rank': 1}},
                'background_scores': {key: {'nm': 101, 'rank': 1.01}},
            },
        },
    )
    return p, b


def test_source_contract_and_score_tampering_rejected():
    p, b = pair()
    assert validate_pair(p, b) == 'I'
    bad = [
        replace(b, split='validation'),
        replace(b, peptide=p.peptide),
        replace(b, group_id='elsewhere'),
        replace(b, mhc=(replace(b.mhc[0], groove1='YYYYYYYYYYY'),)),
        replace(b, context=Context()),
        replace(b, context=replace(b.context, cell_type='macrophage')),
        replace(b, provenance={**b.provenance, 'synthetic': False}),
    ]
    for altered in bad:
        with pytest.raises(ValueError, match="processing"):
            validate_pair(p, altered)
    provenance = copy.deepcopy(b.provenance)
    next(iter(provenance['affinity_match']['background_scores'].values()))['rank'] = 90
    with pytest.raises(ValueError, match='affinity balance'):
        validate_pair(p, replace(b, provenance=provenance))


def test_processing_readout_excludes_binding_ms_and_selection_gradients():
    def output(supply, binding, selection, ms):
        return SimpleNamespace(
            value=binding + supply + selection + ms,
            components={'ms': ms},
            diagnostics={
                'candidates': [
                    {
                        'registers': [
                            {
                                'class_probability': 1.0,
                                'elution_terms': {
                                    'processing': supply,
                                    'binding': binding,
                                    'selection': selection,
                                },
                            }
                        ]
                    }
                ]
            },
        )

    supply = torch.tensor(0.4, requires_grad=True)
    binding = torch.tensor(10.0, requires_grad=True)
    selection = torch.tensor(20.0, requires_grad=True)
    ms = torch.tensor(30.0, requires_grad=True)
    a = output(supply, binding, selection, ms)
    b = output(supply * 0, binding * 3, selection * 4, ms * 5)
    loss = pair_loss([a, b])
    loss.backward()
    assert supply.grad.abs() > 0
    assert binding.grad is selection.grad is ms.grad is None
    assert torch.equal(processing_supply(a), supply)


def test_pair_sampling_is_complete_class_balanced_and_reproducible():
    rows = [*pair('I'), *pair('II'), *pair('I', '2')]
    pool = PairPool(rows)
    assert pool.auxiliary_indices == {1, 3, 5}
    rng = random.Random(3)
    state = rng.getstate()
    drawn = pool.draw(rows, rng, 4)
    rng.setstate(state)
    assert drawn == pool.draw(rows, rng, 4)
    assert [validate_pair(p, b) for p, b in drawn] == ['I'] * 4 + ['II'] * 4
    assert [r.observation_id for i, r in enumerate(rows) if i not in pool.auxiliary_indices] == [
        'pI',
        'pII',
        'pI2',
    ]


@pytest.mark.parametrize('cls', ['I', 'II'])
def test_training_pair_loss_reaches_shared_processing_and_junction(cls):
    net = build(source_context=True, elution_junction_processing=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    p, b = pair(cls)
    for _ in range(2):
        result = train_step(
            net, optimizer, [p], scales(), processing_pairs=[(p, b)], processing_pair_weight=0.5
        )
    assert result['loss_parts']['processing_pairs'] == 1
    assert result['counts']['elution_observation'] == 1
    assert net.elution_junction.input.weight.grad.abs().sum() > 0
    assert net.elution_processing[0].weight.grad.abs().sum() > 0
    assert all(p.requires_grad for p in net.parameters())
    with pytest.raises(ValueError, match='TRAIN'):
        train_step(net, optimizer, [p], scales(), processing_pairs=[(p, replace(b, split='test'))])


def test_canonical_pair_training_resumes_exactly_without_auxiliary_replay(tmp_path, monkeypatch):
    import json
    from presto.data.modular_sources import write_observations, ObservationIndex
    from presto.tests.test_modular_affinity_controls import fixture, run_with_interruption
    from presto.training import modular_run as runner

    path = fixture(
        tmp_path,
        sampling_mode='weighted_steps',
        balance_by=['mhc_class'],
        stratum_weights=[{'values': ['I'], 'weight': 1}, {'values': ['II'], 'weight': 1}],
        processing_pair_training={'weight': 0.5, 'pairs_per_class': 1},
        max_steps=6,
        coverage_policy='diagnostic',
        coverage_exception_reason='Artificial integration/resume fixture',
    )
    manifest = json.loads(path.read_text())
    manifest['training'].update(
        test_policy='no_test_training_diagnostic', no_test_reason='Artificial resume fixture'
    )
    from presto.data.modular import Observation

    original_path = tmp_path / 'rows.jsonl'
    original = [
        Observation.from_dict(json.loads(line))
        for line in original_path.read_text().splitlines()
        if json.loads(line)['split'] != 'test'
    ]
    source_receipt = write_observations(tmp_path / 'no-test-rows.jsonl', original)
    manifest['sources'][0]['path'] = source_receipt['path']
    manifest['sources'][0]['sha256'] = source_receipt['sha256']
    manifest['model'].update(
        endpoints=['ic50', 'elution_observation'],
        source_context=True,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        elution_staging='affinity_downstream',
        elution_affinity_endpoint='ic50',
        elution_junction_processing=True,
    )
    values = [*pair('I'), *pair('II')]
    written = write_observations(tmp_path / 'pairs.jsonl', values)
    manifest['sources'].append({**manifest['sources'][0], **written, 'source': 'software-fixture'})
    # Source receipts carry statistics in addition to the manifest's strict schema.
    manifest['sources'][-1] = {
        k: v for k, v in manifest['sources'][-1].items() if k in manifest['sources'][0]
    }
    path.write_text(json.dumps(manifest))
    reference = run_with_interruption(tmp_path, monkeypatch, path, before_step=4)
    first = json.loads((reference / 'processing-pair-draws.json').read_text())
    second = json.loads((tmp_path / 'resumed/processing-pair-draws.json').read_text())
    assert first == second and len(first) == 6
    snapshot = torch.load(reference / 'resume.pt', weights_only=True)['metadata']['resume']['state']
    index = ObservationIndex(path)
    assert all('processing_pair_positive_id' not in index[i].provenance for i in snapshot['order'])
    manifest['training'].pop('processing_pair_training')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='explicit TRAIN pair control'):
        runner.preflight(ObservationIndex(path))
