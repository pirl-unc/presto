"""Eight-update artificial trajectory comparison, not biological evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.runtime)
    import torch
    from presto.data.modular import AA, MHCInput, Observation
    from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, write_observations
    from presto.training.modular_run import fit_index

    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True)
    rows = [
        Observation(
            f'a{i}',
            'artificial-runtime-parity',
            'ic50' if i % 2 else 'kd',
            AA[i:] + AA[:i],
            (MHCInput(AA, AA[::-1], 'I' if i % 2 else 'II'),),
            value=10 ** (1 + i % 4),
            unit='nM',
            split='train' if i < 6 else 'validation' if i < 8 else 'test',
            group_id=f'a{i}',
            provenance={'original_measurement_id': f'a{i}'},
        )
        for i in range(10)
    ]
    rows += [
        Observation(
            f'i{i}',
            'artificial-runtime-parity',
            'identity',
            '',
            (MHCInput(AA[i:] + AA[:i], AA[::-1]),),
            identity_labels={'mhc_class': 'I' if i % 2 else 'II'},
            group_id=f'i{i}',
            provenance={'original_measurement_id': f'i{i}'},
        )
        for i in range(4)
    ]
    source = write_observations(output / 'rows.jsonl', rows)
    manifest = {
        'schema': MANIFEST_VERSION,
        'model': {
            'endpoints': ['ic50', 'kd'],
            'mhc_classes': ['I', 'II'],
            'identity_tasks': ['mhc_class'],
            'd_model': 32,
            'n_layers': 1,
            'n_heads': 4,
            'd_ff': 64,
            'dropout': 0.3,
        },
        'training': {
            'batch_size': 2,
            'identity_batch_size': 3,
            'max_epochs': 4,
            'max_steps': 8,
            'checkpoint_every_steps': 2,
            'balance_by': ['endpoint', 'mhc_class'],
            'stratum_weights': [
                {'values': ['ic50', 'I'], 'weight': 2},
                {'values': ['kd', 'II'], 'weight': 1},
            ],
        },
        'sources': [
            {
                **source,
                'source': 'artificial-runtime-parity',
                'population': 'artificial software fixture, no biological interpretation',
                'label_policy': 'artificial numeric labels',
                'split_policy': 'disjoint artificial sequences/groups',
            }
        ],
    }
    path = output / 'manifest.json'
    path.write_text(json.dumps(manifest))
    result = fit_index(ObservationIndex(path), output / 'fit')

    def canonical(value):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            return {
                'shape': list(t.shape),
                'dtype': str(t.dtype),
                'sha256': hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {str(k): canonical(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(v) for v in value]
        return value

    fingerprints = {}
    for name in ('initial.pt', 'last.pt', 'selected.pt'):
        payload = torch.load(output / 'fit' / name, map_location='cpu', weights_only=True)
        fingerprints[name] = {
            k: canonical(payload[k])
            for k in ('state_dict', 'optimizer', 'element_coverage')
            if k in payload
        }
    receipt = {'steps': result['steps'], 'metrics': result['metrics'], 'checkpoints': fingerprints}
    (output / 'parity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(
        json.dumps(
            {
                'steps': result['steps'],
                'sha256': hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest(),
            }
        )
    )


if __name__ == '__main__':
    main()
