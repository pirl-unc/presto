"""Cross-check closed training artifacts independently of the training wrapper."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

E = Path(__file__).resolve().parents[1]
B = E.parent / '2026-11-17_1155_codex_shared-presentation-expansion'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--output', default='fit-verification.json')
    args = parser.parse_args()
    parent_root = B / 'results/fits/expanded-state-sqrt-s42-n6000'
    parent_manifest = read(parent_root / 'manifest.json')
    parent = torch.load(parent_root / 'raw/selected.pt', map_location='cpu', weights_only=True)
    table = [json.loads(line) for line in (E / 'results/inputs/processing-pairs-train.jsonl').read_text().splitlines()]
    pairs = {r['negative_id']: r for r in table}
    results, exposures = [], {}
    for run in args.runs.split(','):
        root = E / 'results/fits' / run
        receipt, manifest = read(root / 'receipt.json'), read(root / 'manifest.json')
        summary, history = read(root / 'raw/summary.json'), read(root / 'raw/history.json')
        exposure = read(root / 'actual-training-exposure.json')
        updates, hardware = read(root / 'update-check.json'), read(root / 'hardware.json')
        path = root / 'raw/selected.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['checkpoint_sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        metadata = checkpoint['metadata']
        steps, seed, arm = receipt['steps'], receipt['seed'], receipt['arm']
        assert summary['steps'] == history[-1]['steps'] == steps
        assert metadata['steps'] == exposure['selected_steps']
        assert metadata['steps'] == [h['steps'] for h in history if h['selected']][-1]
        assert metadata['selection_reduction'] == manifest['training']['selection_reduction'] == 'class_study_mean'
        assert manifest['training']['selection_endpoint'] == 'elution_observation'
        assert manifest['training']['stratum_weights'] == parent_manifest['training']['stratum_weights']
        assert manifest['sources'][:len(parent_manifest['sources'])] == parent_manifest['sources']
        assert checkpoint['scales'] == parent['scales']
        assert checkpoint['frozen_parameters'] == updates['frozen_parameters'] == []
        assert summary['coverage_gate']['status'] == 'passed'
        assert not summary['metrics']['test']
        assert hardware['requested_gpu'] == 'H100!' and 'H100' in hardware['name']
        paired = arm != 'continued'
        assert bool(manifest['training'].get('processing_pair_training')) == paired
        assert bool(checkpoint['config'].get('elution_junction_processing')) == (arm == 'paired-junction')
        auxiliary = None
        if paired:
            draws = read(root / 'raw/processing-pair-draws.json')
            assert len(draws) == steps
            assert summary['processing_pair_training']['draws_by_class'] == {'I': 4 * steps, 'II': 4 * steps}
            assert metadata['processing_pair_training']['draw_steps'] == metadata['steps']
            for batch in draws:
                assert [pairs[i]['class'] for i in batch] == ['I'] * 4 + ['II'] * 4
            auxiliary = {cls: {'draws': 4 * steps,
                'unique_pairs': len({i for batch in draws for i in batch if pairs[i]['class'] == cls}),
                'studies': len({pairs[i]['study'] for batch in draws for i in batch if pairs[i]['class'] == cls})} for cls in ('I', 'II')}
        if arm == 'paired-junction':
            assert torch.count_nonzero(checkpoint['state_dict']['elution_junction.readout.weight']) > 0
            assert all(metadata['coverage'][name]['gradient_and_update_steps'] > 0
                       for name in checkpoint['state_dict'] if name.startswith('elution_junction.'))
        key = (seed, steps)
        executed = exposure['executed']
        if key in exposures:
            assert executed == exposures[key], 'ordinary replay exposure changed between arms'
        exposures[key] = executed
        results.append({'run': run, 'steps': steps, 'selected_steps': metadata['steps'],
            'checkpoint_sha256': receipt['checkpoint_sha256'], 'coverage': 'passed', 'frozen_parameters': [],
            'source_scaler_replay_contract_verified': True, 'auxiliary_exposure': auxiliary,
            'hardware': hardware, 'seconds': receipt['runtime']['total_wall_seconds']})
    result = {'status': 'passed', 'test_accessed': False, 'fits': results,
              'ordinary_exposure_equal_across_arms': True}
    (E / 'results' / args.output).write_text(json.dumps(result, indent=2) + '\n')
    print('Verified', len(results), 'closed fits')


if __name__ == '__main__':
    main()
