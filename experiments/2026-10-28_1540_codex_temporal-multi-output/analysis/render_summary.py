"""Readable side-by-side results from independently verified frozen predictions."""

from pathlib import Path
import json


def read(path):
    return json.loads(Path(path).read_text())


def cell(value):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, rows):
    return '\n'.join(
        ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        + ['| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows]
    )


def paired(rows, conditions, dimensions, counts, scores):
    """Join by cohort identity, never by row order or the presence of a metric."""
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in dimensions)
        cohort = groups.setdefault(key, {})
        assert row['condition'] in conditions and row['condition'] not in cohort
        cohort[row['condition']] = row
    result = []
    for key, cohort in sorted(groups.items()):
        assert set(cohort) == set(conditions), (key, 'missing condition')
        support = tuple(cohort[conditions[0]].get(name) for name in counts)
        assert all(tuple(row.get(name) for name in counts) == support for row in cohort.values())
        result.append(
            [*key, *support] + [cohort[name].get(score) for name in conditions for score in scores]
        )
    return (
        table(
            [*dimensions, *counts]
            + [f'{name} · {score}' for name in conditions for score in scores],
            result,
        )
        if result
        else 'No supported evaluation cohort.'
    )


def render(folder):
    folder = Path(folder)
    verification = read(folder / 'prediction-verification.json')
    decision = read(folder / 'validation-decision.json')
    assert verification['status'] == 'passed'
    assert decision['status'] == 'frozen' and decision['test_used_for_selection'] is False
    conditions = sorted(verification['conditions'])
    assert len(conditions) == 2 and set(conditions) == set(decision['scores'])
    strata = read(folder / 'stratified-metrics.json')
    support = read(folder / 'output-class-support.json')
    runtime = read(folder / 'runtime-and-exposure.json')
    assert set(row['condition'] for row in runtime) == set(conditions)
    assert all(
        item['serving_replay']['status'] == 'passed' for item in verification['conditions'].values()
    )
    physical = [
        row
        for row in strata
        if row['study'] == 'all'
        and row['endpoint'] not in {'elution_observation', 'qualitative_binding'}
    ]
    ligand = [
        row
        for row in strata
        if row['study'] == 'all'
        and row['endpoint'] == 'elution_observation'
        and row['population'] in {'synthetic_same_protein', 'synthetic_reference_proteome'}
    ]
    reported = [
        row
        for row in strata
        if row['endpoint'] == 'elution_observation'
        and row['study'] != 'all'
        and row['population'] == 'source_reported_outcomes'
    ]
    qualitative = [
        row for row in strata if row['endpoint'] == 'qualitative_binding' and row['study'] == 'all'
    ]
    threshold = [
        {**row, **row['binding_at_500_nM']} for row in physical if 'binding_at_500_nM' in row
    ]
    dimensions = ('split', 'era', 'endpoint', 'mhc_class')
    lines = [
        '# Shared multi-output results',
        '',
        'Both fits completed and their saved validation/test targets, losses and metrics '
        'passed independent checks. Public serving replay passed on TRAIN probes. '
        'These checks establish reproducibility; they do not independently qualify '
        'biological mechanisms or future-study accuracy.',
        '',
        f"Validation-selected condition: **{decision['winner']}**. Selection used the "
        'minimum equal-endpoint validation loss; test outcomes were excluded from selection.',
        '',
        '## Exact quantitative measurements',
        '',
        'Bounds contribute one-sided training/evaluation loss, but are excluded from '
        'Spearman, Pearson and RMSE here. Affinity/EC50/proxy RMSE is log10(nM); '
        'half-life is log10(minutes), koff is log10(1/s), and td50 is degrees Celsius. '
        'N/A denotes unavailable or undefined metrics. Recent and older study '
        'cohorts remain separate.',
        '',
        paired(physical, conditions, dimensions, ('n', 'n_exact'), ('spearman', 'pearson', 'rmse')),
        '',
        '## Ligands versus generated unobserved comparators',
        '',
        'Same-protein and reference-proteome comparisons are distinct populations. '
        'Comparators are length-matched, sampled with replacement, and excluded from '
        'the known-positive pool. They are not measured non-binders. AUPRC depends '
        'on the five-comparators-per-parent evaluation contract.',
        '',
        paired(
            ligand,
            conditions,
            ('split', 'era', 'mhc_class', 'population'),
            ('n', 'positives', 'negatives'),
            ('auroc', 'auprc'),
        ),
        '',
        '## Source-reported ligand outcomes',
        '',
        'Each study and assay method is shown separately. Single-class outcome sets '
        'cannot define an AUROC; generated comparators are absent from this table.',
        '',
        paired(
            reported,
            conditions,
            ('split', 'era', 'mhc_class', 'study', 'assay_method'),
            ('n', 'positives', 'negatives'),
            ('auroc', 'auprc'),
        ),
        '',
        '## Qualitative binding',
        '',
        'These are the source assay outcomes; their labels do not imply a universal 500 nM cutoff.',
        '',
        paired(
            qualitative, conditions, dimensions, ('n', 'positives', 'negatives'), ('auroc', 'auprc')
        ),
        '',
        '## Quantitative affinity at 500 nM',
        '',
        'Only exact concentration measurements with an eligible endpoint enter '
        'this thresholded comparison. Proxy assays keep their original output names.',
        '',
        paired(
            threshold,
            conditions,
            dimensions,
            ('n', 'positives', 'negatives'),
            ('auroc', 'auprc', 'balanced_accuracy'),
        ),
        '',
        '## Output support and actual optimizer exposures',
        '',
        'Original measurements and generated training comparators are counted '
        'separately. Repeated exposures are not additional measurements. A trained '
        'output with no held-out observations has no measured generalization result.',
        '',
        paired(
            support,
            conditions,
            ('endpoint', 'mhc_class'),
            ('train_original_n', 'train_generated_n', 'validation_n', 'test_n'),
            ('optimizer_exposures',),
        ),
        '',
        '## Checkpoint coverage and runtime',
        '',
        'Each completed fit traversed the full weighted training epoch. Validation '
        'may select an earlier checkpoint with lower unique-row exposure; both '
        'counts are explicit below. Wall time includes data admission and evaluation.',
        '',
        table(
            [
                'condition',
                'selected steps',
                'completed steps',
                'selected unique rows',
                'completed unique rows',
                'wall hours',
                'optimizer hours',
                'peak GPU GiB',
            ],
            [
                [
                    row['condition'],
                    row['selected_steps'],
                    row['completed_steps'],
                    row['selected_unique_rows'],
                    row['completed_unique_rows'],
                    row['wall_seconds'] / 3600,
                    row['optimizer_seconds'] / 3600,
                    row['peak_gpu_allocated'] / 2**30,
                ]
                for row in runtime
            ],
        ),
        '',
        table(
            ['condition', 'validation overall loss', 'test overall loss', 'serving probes'],
            [
                [
                    name,
                    verification['conditions'][name]['splits']['validation']['overall_loss'],
                    verification['conditions'][name]['splits']['test']['overall_loss'],
                    verification['conditions'][name]['serving_replay']['n'],
                ]
                for name in conditions
            ],
        ),
        '',
        '## Evidence and limitations',
        '',
        'These whole-study holdouts differ from earlier inspected development splits; '
        'scores cannot establish improvement over the earlier IC50 models without '
        'a matched comparison. Some cells contain only one study. Peptide-bootstrap '
        'intervals are conditional on the fitted model and sampled studies; they '
        'do not measure generalization to new studies or training seeds.',
        '',
        '- [All assay/study and threshold classification metrics](stratified-metrics.csv)',
        '- [TRAIN-mean and constant-score comparisons](baseline-comparison.csv)',
        '- [Study concentration](study-concentration.csv)',
        '- [Cluster uncertainty and unavailable-interval reasons](cluster-uncertainty.json)',
        '- [Length, candidate and allele failure strata](failure-strata.csv)',
        '- [Register diagnostics](register-diagnostics.csv)',
        '- [Parameter coordinate coverage](parameter-coordinate-coverage.csv)',
        '- [Input feature coverage](input-feature-coverage.csv)',
        '- [Comparison plot](performance-comparison.png)',
        '- [Independent prediction verification](prediction-verification.json)',
        '- [Experiment contract, exclusions and reproduction](../README.md)',
        '',
        'Register weights have no measured core-localization truth here. Coordinate '
        'updates do not establish identified biological effects. Ligand scores '
        'combine factors of presentation and measurement; they do not separately '
        'qualify antigen processing or MS detectability.',
        '',
        'Per-example prediction dumps and checkpoint hashes are located in each '
        'condition’s collection record:',
        '',
        *[f'- [{name}]({name}/collection.json)' for name in conditions],
        '',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys

    target = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'results'
    )
    (target / 'summary.md').write_text(render(target))
