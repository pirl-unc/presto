"""Freeze the prepared and independently verified scientific contract before fitting."""

from pathlib import Path
import hashlib
import json
import time

F = Path(__file__).resolve().parents[1]


def main():
    def read(name):
        return json.loads((F / name).read_text())

    p = read('results/prepared.json')
    audit = read('results/partition-verification.json')
    assert p['status'] == 'prepared' and audit['status'] == 'passed'
    assert p['verification'] == audit
    metadata = p['review_metadata']
    assert metadata['preflight.json']['status'] == 'passed'
    assert set(metadata['endpoint-support.json']['allocated']) == set(p['model']['endpoints'])
    assert set(p['manifests']) == {'wide-multi-s42', 'wide-multi-s43'}
    for name, spec in p['manifests'].items():
        manifest = metadata[name + '.json']
        canonical = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
        assert hashlib.sha256(canonical).hexdigest() == spec['sha256']
        assert manifest['model'] == p['model']
        assert manifest['training']['max_steps'] == p['steps']
        assert manifest['training']['seed'] == spec['seed']
        assert manifest['training']['selection_reduction'] == 'endpoint_mean'
        assert manifest['training']['batch_size'] == 128
    target = F / 'reproduce/training-contract.json'
    if target.exists():
        raise ValueError('Contract already frozen; do not overwrite')
    contract = dict(
        status='frozen',
        family=F.name,
        frozen_unix=time.time(),
        prepared_record_sha256=p['prepared_record_sha256'],
        manifests=p['manifests'],
        model=p['model'],
        counts=p['counts'],
        train_strata=p['train_strata'],
        epoch_quotas=p['epoch_quotas'],
        max_steps=p['steps'],
        weighted_epoch_examples=p['weighted_epoch_examples'],
        background_contract=p['background_contract'],
        verification=audit,
        requested_gpu='H100!',
        max_concurrent_gpus=2,
        seed_selection='minimum equal-endpoint validation loss; seed breaks ties',
        test_used_for_selection=False,
        interpretation=(
            'Whole 2026 study holdouts and older assay holdouts are reported separately; '
            'other 2026 studies can be in TRAIN, so this is not a strict date-cutoff benchmark. '
            'Every TRAIN row is used by the completed fit; the validation-selected checkpoint '
            'may precede full coverage and its actual unique-row exposure is reported.'
        ),
    )
    target.write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
    print('Frozen', p['steps'], 'updates per seed;', len(p['model']['endpoints']), 'outputs')


if __name__ == '__main__':
    main()
