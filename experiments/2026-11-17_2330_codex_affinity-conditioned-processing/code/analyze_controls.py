"""Verify and summarize fixed-checkpoint input controls on common examples."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze import E, flatten, metrics, bootstrap, write

MODES = ('original', 'apc-unknown', 'apc-shuffle', 'flank-unknown', 'flank-shuffle')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    args = parser.parse_args()
    standard = pd.read_parquet(E / 'results/analysis/predictions.parquet')
    for run in args.runs.split(','):
        root = E / 'results/controls' / run
        receipt = json.loads((root / 'receipt.json').read_text())
        reference = json.loads((E / 'results/evaluation' / run / 'receipt.json').read_text())
        assert receipt['checkpoint_sha256'] == reference['checkpoint_sha256']
        assert not receipt['test_accessed']
        collected = []
        for population in receipt['populations']:
            for mode in MODES:
                with gzip.open(root / f'{population}-{mode}-predictions.jsonl.gz', 'rt') as stream:
                    collected.extend(flatten(json.loads(line), mode, population) for line in stream)
        frame = pd.DataFrame(collected)
        assert not frame.duplicated(['model', 'population', 'id']).any()
        verification = []
        for population, group in frame.groupby('population'):
            original = group[group.model == 'original'].set_index('id').sort_index()
            expected = standard[(standard.model == run) & (standard.population == population)].set_index('id').sort_index()
            assert list(original.index) == list(expected.index)
            np.testing.assert_array_equal(original.target, expected.target)
            np.testing.assert_allclose(original.prediction, expected.prediction, rtol=0, atol=1e-6)
            for mode, current in group.groupby('model'):
                current = current.set_index('id').sort_index()
                assert list(original.index) == list(current.index)
                for column in ('target', 'peptide', 'pair', 'group', 'condition', 'mhc_signature'):
                    pd.testing.assert_series_equal(original[column], current[column], check_dtype=False)
                verification.append({'population': population, 'mode': mode,
                    'max_absolute_logit_change': float(abs(current.prediction - original.prediction).max()),
                    'changed_score_rows': int((abs(current.prediction - original.prediction) > 1e-6).sum())})
        out = root / 'analysis'
        out.mkdir(exist_ok=True)
        frame.to_parquet(out / 'predictions.parquet', index=False)
        report = metrics(frame)
        write(out / 'metrics.json', report)
        pd.DataFrame(report).to_csv(out / 'metrics.csv', index=False)
        write(out / 'verification.json', verification)
        for mode in MODES[1:]:
            write(out / f'bootstrap-original-minus-{mode}.json', bootstrap(frame, 'original', mode))
        print(run, 'verified rows', len(frame), flush=True)


if __name__ == '__main__':
    main()
