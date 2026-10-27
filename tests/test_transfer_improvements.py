"""Shared interaction, deterministic sampling and resume software checks."""

from collections import Counter
from dataclasses import replace
import json
import random

import pytest
import torch

from presto.data.modular import AA, MHCInput
from presto.data.modular_sources import ObservationIndex, write_observations
from presto.models import Presto, PrestoConfig
from presto.models.modular_dependencies import parameter_dependencies
from presto.training import modular_run as runner
from presto.training.modular_sampling import epoch_order, stratum, validate_balance_by
from presto.tests.test_unified_training_core import row
from presto.tests.test_modular_resume import manifest_fixture, interrupt, assert_same


def test_balancing_covers_majority_and_cycles_only_training_rows():
    rows = [row(mhc_class='I') for _ in range(7)] + [row(mhc_class='II') for _ in range(2)]
    rows += [row(mhc_class='II', split='test')]
    order = epoch_order(rows, range(9), ['mhc_class'], random.Random(13))
    assert len(order) == 14 and 9 not in order
    assert Counter(rows[i].mhc[0].mhc_class for i in order) == {'I': 7, 'II': 7}
    assert Counter(i for i in order if i < 7) == Counter(range(7))
    assert all(
        {rows[i].mhc[0].mhc_class for i in order[j : j + 2]} == {'I', 'II'} for j in range(0, 14, 2)
    )
    assert order == epoch_order(rows, range(9), ['mhc_class'], random.Random(13))
    assert order != epoch_order(rows, range(9), ['mhc_class'], random.Random(14))
    expected = list(range(9))
    random.Random(13).shuffle(expected)
    assert epoch_order(rows, range(9), [], random.Random(13)) == expected


def test_sampling_rejects_ambiguous_metadata_and_not_numeric_labels():
    for fields in ('mhc_class', ['value'], ['mhc_class', 'mhc_class']):
        with pytest.raises(ValueError, match="balance_by"):
            validate_balance_by(fields)
    for mhcs in ((), (MHCInput(AA, AA),), (MHCInput(AA, AA, 'I'), MHCInput(AA, AA, 'II'))):
        with pytest.raises(ValueError, match='explicit class'):
            stratum(row(mhc=mhcs), ['mhc_class'])
    assert stratum(row(value=10), ['endpoint', 'mhc_class']) == stratum(
        row(value=10000), ['endpoint', 'mhc_class']
    )


@pytest.mark.parametrize('kind', ['I', 'II'])
def test_each_class_trains_all_interaction_weights_and_dependency_contract(kind):
    torch.manual_seed(81)
    model = Presto(PrestoConfig(mhc_classes=(kind,), dropout=0))
    output = model([row(mhc_class=kind)])[0]
    output.value.backward()
    names = {
        n
        for n, _ in model.named_parameters()
        if n.startswith(('residue_interaction.', 'interaction_norm.'))
    }
    assert names <= parameter_dependencies(model, 'ic50')
    for n, p in model.named_parameters():
        if n in names:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n


def test_interaction_uses_ordered_residues_before_pooling():
    torch.manual_seed(82)
    model = Presto(PrestoConfig(dropout=0)).eval()
    sample = row(mhc_class='I', peptide=AA[:11])
    _, states = model._encode([sample])
    # Preserve the pooled groove vector while changing which contextual residues
    # are available: a pooled-only implementation is invariant to this change.
    changed = dict(states)
    g = states[sample.mhc[0].groove1]
    changed[sample.mhc[0].groove1] = g.mean(0).expand_as(g)
    candidates = [(0, sample.peptide, [])]
    a = model._interaction_candidates(sample.peptide, sample.mhc[0], candidates, states)
    b = model._interaction_candidates(sample.peptide, sample.mhc[0], candidates, changed)
    assert not torch.allclose(a, b, atol=1e-6)


def test_balanced_resume_replays_identical_order_and_weights(tmp_path, monkeypatch):
    path = manifest_fixture(tmp_path, replay=False)
    index = ObservationIndex(path)
    rows = [
        replace(r, mhc=(replace(r.mhc[0], mhc_class='II' if i in (4, 5, 7, 9) else 'I'),))
        for i, r in enumerate(index)
    ]
    written = write_observations(tmp_path / 'balanced.jsonl', rows)
    manifest = json.loads(path.read_text())
    manifest['model']['mhc_classes'] = ['I', 'II']
    manifest['training']['balance_by'] = ['mhc_class']
    manifest['sources'][0].update(path=written['path'], sha256=written['sha256'])
    path.write_text(json.dumps(manifest))
    original, trace = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), tmp_path / 'reference')
    expected_trace = list(trace)
    monkeypatch.setattr(runner, 'train_step', original)
    original, before = interrupt(monkeypatch, before_step=6)
    with pytest.raises(RuntimeError, match='interruption'):
        runner.fit_index(ObservationIndex(path), tmp_path / 'resumed')
    monkeypatch.setattr(runner, 'train_step', original)
    original, after = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), tmp_path / 'resumed', resume=True)
    assert before[:4] + after == expected_trace
    for name in ('selected.pt', 'resume.pt'):
        a = torch.load(tmp_path / 'reference' / name, weights_only=True)
        b = torch.load(tmp_path / 'resumed' / name, weights_only=True)
        assert_same(a['state_dict'], b['state_dict'])
        assert_same(a['optimizer'], b['optimizer'])
    assert (tmp_path / 'reference/test-predictions.jsonl').read_bytes() == (
        tmp_path / 'resumed/test-predictions.jsonl'
    ).read_bytes()
