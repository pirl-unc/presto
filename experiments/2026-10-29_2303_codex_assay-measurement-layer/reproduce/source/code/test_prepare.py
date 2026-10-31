"""Check whole-study moves, cross-output peptide embargo and immutable reuse."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'assay_experiment_prepare', Path(__file__).with_name('prepare.py')
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def observation(identifier, split, group, peptide, endpoint='ic50'):
    return dict(
        observation_id=identifier,
        split=split,
        group_id=group,
        peptide=peptide,
        endpoint=endpoint,
        mhc=[{'mhc_class': 'I'}],
        provenance={'raw_record': {'assay_method': 'purified MHC/competitive/fluorescence'}},
    )


def test_projection_and_binding(tmp_path):
    old = tmp_path / 'old'
    old.mkdir()
    rows = [
        observation('ov', 'validation', 'old-val', 'A'),
        observation('ot', 'test', 'old-test', 'B'),
        observation('v', 'train', 'extra-val', 'C'),
        observation('t', 'train', 'extra-test', 'D'),
        observation('embargo', 'train', 'peer', 'C', 'elution_observation'),
        observation('overlap', 'train', 'extra-val', 'D'),
        observation('keep', 'train', 'keep', 'E'),
    ]
    source = old / 'rows.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    parent = dict(
        sources=[dict(path=str(source), sha256=module.sha(source), split_policy='fixture')],
        model={},
        training={},
    )
    policy = {'groups': {'extra-val': {'split': 'validation'}, 'extra-test': {'split': 'test'}}}
    output = tmp_path / 'prepared'
    result = module.prepare(parent, old, output, policy)
    assert result['disposition'] == dict(
        original_heldout_retained=2,
        moved_validation=1,
        moved_test=1,
        train_peptide_embargo=1,
        extra_overlap_removed=1,
        retained_train=1,
    )
    prepared = list(module.records(output / 'rows.jsonl'))
    assert [row['observation_id'] for row in prepared] == ['ov', 'ot', 'v', 't', 'keep']
    assert prepared[2]['split'] == 'validation' and prepared[3]['split'] == 'test'
    assert module.prepare(parent, old, output, policy) == result
    altered = copy.deepcopy(policy)
    altered['groups']['extra-val']['split'] = 'test'
    with pytest.raises(AssertionError, match='contract changed'):
        module.prepare(parent, old, output, altered)
    manifests = [module.read(v['path']) for v in result['manifests'].values()]
    assert len(manifests) == 6
    assert all(m['sources'] == manifests[0]['sources'] for m in manifests)
    assert {m['model']['assay_mode'] for m in manifests} == {'pooled', 'calibration', 'factorized'}
