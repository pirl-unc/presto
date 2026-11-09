"""Reconcile preserved source rows and summarize assay agreement without model scores."""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json
import math
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

F = Path(__file__).resolve().parents[1]
ENDPOINTS = ['ic50', 'kd', 'ec50', 'kd_via_ic50', 'kd_via_ec50']
LABELS = ['IC50', 'KD', 'EC50', 'KD (~IC50)', 'KD (~EC50)']


def read_pairs():
    with gzip.open(F / 'results/assay-agreement/paired-values.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    summary = json.loads((F / 'results/assay-agreement/summary.json').read_text())
    pairs = read_pairs()
    exact = defaultdict(list)
    with gzip.open(F / 'results/overlap-lineage/overlap-records.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    for row in records:
        assert row['unit'] == 'nM'
        if row['qualifier'] == '=':
            exact[row['pmhc_sha256'], row['endpoint'], row['group_id']].append(row['value'])
    for row in pairs:
        for side in ('a', 'b'):
            # Recompute aggregates directly from independently reread original values.
            per_study = [
                statistics.median(
                    math.log10(v) for v in exact[row['pmhc_sha256'], row['endpoint_' + side], study]
                )
                for study in row['studies_' + side]
            ]
            expected = statistics.median(per_study)
            assert abs(expected - row[side + '_log10_nM']) < 1e-12
    details = json.loads((F / 'results/overlap-lineage/study-details.json').read_text())
    metadata = defaultdict(list)
    for row in details:
        metadata[row['group_id']].append(row)
    sensitivity = []
    for cls in ('I', 'II'):
        subset = [
            r
            for r in pairs
            if r['scope'] == 'pooled_pmhc'
            and r['mhc_class'] == cls
            and r['endpoint_a'] == 'ic50'
            and r['endpoint_b'] == 'kd_via_ic50'
        ]
        common = Counter(s for r in subset for s in r['studies_a'])
        group, n = common.most_common(1)[0]
        rest = [r for r in subset if group not in r['studies_a']]
        x = [r['a_log10_nM'] for r in rest]
        y = [r['b_log10_nM'] for r in rest]
        sensitivity.append(
            dict(
                mhc_class=cls,
                n=len(subset),
                dominant_ic50_study=group,
                dominant_study_metadata=metadata[group],
                dominant_n=n,
                remaining_n=len(rest),
                remaining_spearman=float(spearmanr(x, y).statistic),
                fraction_within_10_percent=float(
                    np.mean(
                        [abs(r['a_log10_nM'] - r['b_log10_nM']) <= math.log10(1.1) for r in subset]
                    )
                ),
                interpretation=(
                    'Exploratory dominance sensitivity; remaining rows may also reuse data.'
                ),
            )
        )
    assert sum(r['n'] for r in summary['assay_methods']) == 184637
    result = dict(
        status='verified_source_aggregation',
        source_records=len(records),
        verified_pair_aggregates=2 * len(pairs),
        concentration_rows=184637,
        exact_rows=summary['exact_observation_count'],
        bound_rows=184637 - summary['exact_observation_count'],
        sensitivity=sensitivity,
        caution='Study components and assay IDs do not establish independent experiments.',
    )
    (F / 'results/agreement-interpretation.json').write_text(json.dumps(result, indent=2) + '\n')
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#eceff2')
    for cls, ax in zip(('I', 'II'), axes):
        cells = np.full((5, 5), np.nan)
        entries = [
            r
            for r in summary['comparisons']
            if r['mhc_class'] == cls and r['scope'] == 'pooled_pmhc'
        ]
        for r in entries:
            i, j = ENDPOINTS.index(r['endpoint_a']), ENDPOINTS.index(r['endpoint_b'])
            if r['n'] >= 30 and r['spearman'] is not None:
                cells[i, j] = cells[j, i] = r['spearman']
        im = ax.imshow(np.ma.masked_invalid(cells), vmin=0, vmax=1, cmap=cmap)
        for r in entries:
            i, j = ENDPOINTS.index(r['endpoint_a']), ENDPOINTS.index(r['endpoint_b'])
            rho = '—' if r['spearman'] is None else f'{r["spearman"]:.3f}'
            label = f'{rho}\nn={r["n"]:,}'
            color = 'white' if r['n'] >= 30 and r['spearman'] < 0.65 else '#17212b'
            ax.text(j, i, label, ha='center', va='center', fontsize=9, color=color)
            ax.text(i, j, label, ha='center', va='center', fontsize=9, color=color)
        for i in range(5):
            ax.text(i, i, '—', ha='center', va='center', color='#667788')
        ax.set_xticks(range(5), LABELS, rotation=35, ha='right')
        ax.set_yticks(range(5), LABELS)
        ax.set_title(f'MHC-{cls}', fontsize=15, loc='left')
    figure.colorbar(im, ax=axes, shrink=0.65, label='Spearman ρ (colored when n ≥ 30)')
    figure.suptitle(
        'Same-pMHC measurement agreement · exact TRAIN observations\n'
        'Median within study, then across studies; overlap is not independent replication',
        fontsize=13,
    )
    figure.savefig(F / 'results/assay-agreement.png', dpi=180)
    figure.savefig(F / 'results/assay-agreement.pdf')
    plt.close(figure)
    print(json.dumps({k: v for k, v in result.items() if k != 'sensitivity'}, indent=2))


if __name__ == '__main__':
    main()
