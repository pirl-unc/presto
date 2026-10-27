"""Incremental independent validation checks, without opening test predictions."""

import math
import json
import numpy as np
from verify import FAMILY, PINS, load, source_rows, metrics, assert_metric, atomic, sha


def main():
    controller = json.loads((FAMILY / 'reproduce/controller.json').read_text())
    original = {r['observation_id']: r for r in source_rows('validation')}
    output = {}
    for name in controller['completed']:
        directory = FAMILY / 'results' / name
        report = json.loads((directory / 'report.json').read_text())
        scale = json.loads((directory / 'resolved-config.json').read_text())['scales']['ic50'][
            'scale'
        ]
        cases = [
            (
                'in_class',
                directory / 'validation-predictions.jsonl.gz',
                PINS['conditions'][name]['classes'],
                report['summary']['metrics']['validation'],
            )
        ]
        for protocol, transfer in report['transfer'].items():
            cases.append(
                (
                    'zero_shot',
                    directory / f'{protocol}-validation-predictions.jsonl.gz',
                    [transfer['target_class']],
                    transfer['metrics']['validation'],
                )
            )
        result = {}
        for scope, path, classes, reported in cases:
            rows = load(path)
            expected = {k: v for k, v in original.items() if v['mhc'][0]['mhc_class'] in classes}
            assert set(expected) == {r['observation_id'] for r in rows} and len(expected) == len(
                rows
            )
            for row in rows:
                raw = expected[row['observation_id']]
                assert row['peptide'] == raw['peptide'] and row['qualifier'] == raw['qualifier']
                assert row['observation']['mhc'] == raw['mhc']
                assert row['split'] == 'validation'
                np.testing.assert_allclose(row['target'], math.log10(raw['value']), atol=1e-12)
            for kind in classes:
                m = metrics([r for r in rows if r['mhc_classes'] == [kind]], scale)
                assert_metric(m, reported['ic50']['by_class'][kind])
                result[kind] = dict(scope=scope, metrics=m, prediction_sha256=sha(path))
        output[name] = result
    atomic(
        FAMILY / 'results/validation-independent-progress.json',
        dict(
            status='passed', completed=len(output), conditions=output, test_predictions_opened=False
        ),
    )
    for name, classes in output.items():
        print(
            name,
            {
                k: {m: round(v['metrics'][m], 4) for m in ('spearman', 'rmse')}
                for k, v in classes.items()
            },
        )


if __name__ == '__main__':
    main()
