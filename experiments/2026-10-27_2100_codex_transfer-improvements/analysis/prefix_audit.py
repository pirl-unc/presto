"""Compare the unchanged natural-sampling prefix with the historical H100 runs."""

import json
from pathlib import Path
import torch
from verify import FAMILY, read, atomic


def main():
    prior = FAMILY.parent / '2026-10-27_1930_codex_modal-cross-class'
    results = []
    for seed in (42, 43):
        before = prior / 'results' / f'shared-joint-s{seed}'
        after = FAMILY / 'results' / f'pooled-natural-joint-s{seed}'
        if not (after / 'report.json').exists():
            continue
        a = read(before / 'history.json')
        b = read(after / 'history.json')
        checks = []
        for x in a:
            for y in b:
                if x['steps'] == y['steps']:
                    checks.append(
                        dict(
                            steps=x['steps'],
                            training_loss_delta=y['training_loss'] - x['training_loss'],
                            class_metrics={
                                k: {
                                    m: y['validation']['ic50']['by_class'][k][m]
                                    - x['validation']['ic50']['by_class'][k][m]
                                    for m in ('spearman', 'rmse')
                                }
                                for k in ('I', 'II')
                            },
                            label_exposures_exact=x['label_exposures'] == y['label_exposures'],
                        )
                    )
        left = torch.load(
            Path(read(before / 'report.json')['checkpoint']).parent / 'initial.pt',
            weights_only=True,
            map_location='cpu',
        )
        right = torch.load(
            Path(read(after / 'report.json')['checkpoint']).parent / 'initial.pt',
            weights_only=True,
            map_location='cpu',
        )
        exact = left['state_dict'].keys() == right['state_dict'].keys() and all(
            torch.equal(left['state_dict'][k], right['state_dict'][k]) for k in left['state_dict']
        )
        assert exact and left['scales'] == right['scales']
        results.append(
            dict(seed=seed, initial_weights_exact=exact, scales_exact=True, comparisons=checks)
        )
    result = dict(
        status='numerical_differences_recorded',
        results=results,
        exact_history_equality_attempt=(
            "seed42 failed only on small float32 differences; "
            "not claimed bitwise reproducible training"
        ),
        scope=(
            "common pre-extension points; no test metrics accessed; same H100/library contract; "
            "deterministic algorithms not enabled"
        ),
    )
    atomic(FAMILY / 'results/natural-prefix-parity.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
