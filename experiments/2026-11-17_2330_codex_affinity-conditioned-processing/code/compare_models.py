"""Prespecified parent/continuation/objective/module contrasts and retention guards."""
import argparse
import hashlib
import json

import pandas as pd
from analyze import E, bootstrap, write

ARMS = ('continued', 'paired', 'paired-junction')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    out = E / 'results/analysis'
    frame = pd.read_parquet(out / 'predictions.parquet')
    table = pd.read_csv(out / 'metrics.csv')
    available = set(frame.model)
    required = {f'{arm}-s{seed}-n2000{kind}' for arm in ARMS for seed in (42, 43) for kind in ('', '-terminal')} | {'parent'}
    if not args.partial:
        assert required <= available, sorted(required - available)
    comparisons = []
    for model in sorted(required - {'parent'}):
        if model in available:
            comparisons.append((model, 'parent'))
    for seed in (42, 43):
        for kind in ('', '-terminal'):
            for real, other in (('paired', 'continued'), ('paired-junction', 'paired'), ('paired-junction', 'continued')):
                names = (f'{real}-s{seed}-n2000{kind}', f'{other}-s{seed}-n2000{kind}')
                if set(names) <= available:
                    comparisons.append(names)
    for real, other in comparisons:
        fingerprint_frame = frame[frame.model.isin((real, other))].sort_values(['model', 'population', 'id'])
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(
            fingerprint_frame[['model', 'population', 'id', 'target', 'prediction', 'processing_supply', 'pair', 'group', 'study', 'condition']],
            index=False).values.tobytes()).hexdigest()
        path = out / f'bootstrap-{real}-minus-{other}.json'
        if path.exists() and json.loads(path.read_text()).get('prediction_input_sha256') == fingerprint:
            continue
        result = bootstrap(frame, real, other)
        result['prediction_input_sha256'] = fingerprint
        result['checkpoint_sha256'] = {name: json.loads((E / 'results/evaluation' / name / 'receipt.json').read_text())['checkpoint_sha256'] for name in (real, other)}
        write(path, result)
        print('Completed paired contrast', real, 'minus', other, flush=True)
    guards = []
    for model in sorted(required - {'parent'}):
        if model not in available:
            continue
        for population, classes in (('expanded', ('I', 'II')), ('apc', ('I', 'II')), ('broad', ('I',))):
            for cls in classes:
                cell = table[(table.population == population) & (table['class'] == cls) & (table.readout == 'prediction')]
                delta = cell[cell.model == model].auroc.item() - cell[cell.model == 'parent'].auroc.item()
                guards.append({'model': model, 'population': population, 'class': cls, 'metric': 'auroc', 'delta': delta, 'passed': delta >= -.005, 'rule': 'decline no more than .005'})
        for population in ('affinity', 'affinity_ancestry_screened'):
            for cls in ('I', 'II'):
                cell = table[(table.population == population) & (table['class'] == cls) & (table.endpoint == 'ic50')]
                for metric, limit, direction in (('spearman', -.02, 1), ('rmse_log10_nm', .03, -1)):
                    delta = cell[cell.model == model][metric].item() - cell[cell.model == 'parent'][metric].item()
                    guards.append({'model': model, 'population': population, 'class': cls, 'metric': metric,
                        'delta': delta, 'passed': direction * delta >= direction * limit,
                        'rule': 'decline no more than .02' if direction == 1 else 'rise no more than .03'})
    write(out / 'retention-guards.json', {'checks': guards, 'interpretation': 'operational point-estimate guards, not formal noninferiority tests'})
    pd.DataFrame(guards).to_csv(out / 'retention-guards.csv', index=False)
    print('Contrasts', len(comparisons), 'guard checks', len(guards), flush=True)


if __name__ == '__main__':
    main()
