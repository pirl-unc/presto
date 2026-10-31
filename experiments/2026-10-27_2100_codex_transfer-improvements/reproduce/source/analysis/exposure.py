"""Reconstruct row exposure from frozen seeded samplers and selected step counts."""

from collections import Counter
import json
import random
from verify import FAMILY, PINS, source_rows, read, atomic


def count(classes, seed, steps, balanced):
    labels = [
        r['mhc'][0]['mhc_class']
        for r in source_rows('train')
        if r['mhc'][0]['mhc_class'] in classes
    ]
    rng = random.Random(seed)
    counts = Counter()
    done = 0
    while done < steps:
        if balanced and len(set(labels)) > 1:
            groups = {k: [i for i, v in enumerate(labels) if v == k] for k in sorted(set(labels))}
            size = max(map(len, groups.values()))
            streams = {}
            for key, group in groups.items():
                values = []
                while len(values) < size:
                    cycle = list(group)
                    rng.shuffle(cycle)
                    values.extend(cycle)
                streams[key] = values[:size]
            keys = list(groups)
            rng.shuffle(keys)
            order = [streams[key][i] for i in range(size) for key in keys]
        else:
            order = list(range(len(labels)))
            rng.shuffle(order)
        for start in range(0, len(order), 64):
            counts.update(labels[i] for i in order[start : start + 64])
            done += 1
            if done == steps:
                return dict(counts)


def main():
    result = {}
    for name, spec in PINS['conditions'].items():
        directory = FAMILY / 'results' / name
        report = read(directory / 'report.json')
        history = read(directory / 'history.json')
        expected = count(spec['classes'], spec['seed'], spec['max_steps'], bool(spec['balance_by']))
        actual = {k: v for k, v in report['class_examples'].items() if v}
        assert actual == expected, (name, actual, expected)
        selected_steps = next(
            h['steps'] for h in history if h['epoch'] == report['summary']['selected_epoch']
        )
        result[name] = dict(
            total_steps=spec['max_steps'],
            actual_class_examples=actual,
            selected_steps=selected_steps,
            selected_class_examples=count(
                spec['classes'], spec['seed'], selected_steps, bool(spec['balance_by'])
            ),
        )
    atomic(FAMILY / 'results/exposure-verification.json', dict(status='passed', conditions=result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
