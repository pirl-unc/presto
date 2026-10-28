"""Report verified saved predictions, sparse support, register behavior and coverage."""

from collections import defaultdict
from pathlib import Path
import csv
import gzip
import json
import sys
import time
import numpy as np
from uncertainty import Metric, bootstrap
from verify_predictions import F, BINARY, groups, metrics, sha


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (F / 'results' / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(name, rows):
    write(name + '.json', rows)
    with (F / 'results' / (name + '.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            )


def length_bin(length):
    for limit, label in ((8, '7-8'), (9, '9'), (10, '10'), (14, '11-14'), (20, '15-20')):
        if length <= limit:
            return label
    return '>20'


def detailed_groups(rows):
    """Exact candidate sets are one stratum; never replicate a bag per candidate."""
    result = defaultdict(list)
    for key, subset in groups(rows).items():
        era, endpoint, cls, population, study, method = key
        if study != 'all' and endpoint != 'elution_observation':
            continue  # Physical study/method strata already exist in verified tables.
        for row in subset:
            candidates = row['observation']['mhc']
            alleles = sorted(
                {
                    allele
                    for candidate in candidates
                    for allele in candidate['provenance'].get('catalog_alleles', [])
                }
            )
            dimensions = {
                'length': length_bin(len(row['peptide'])),
                'candidate_count': str(len(candidates)),
                'catalog_allele_set': ';'.join(alleles) or 'unknown',
            }
            for dimension, value in dimensions.items():
                result[(era, endpoint, cls, population, study, method, dimension, value)].append(
                    row
                )
    return result


def register_summary(rows):
    values = defaultdict(list)
    for row in rows:
        if row['source'] == 'hitlist_ligand_background':
            continue
        registers = []
        for candidate in row['diagnostics'].get('candidates', []):
            for register in candidate.get('registers', []):
                weights = np.asarray(register['weights'], dtype=float)
                if register['class'] == 'II' and len(weights) > 1:
                    registers.append(
                        (
                            float(
                                -np.sum(weights * np.log(weights.clip(1e-30)))
                                / np.log(len(weights))
                            ),
                            float(weights.max()),
                        )
                    )
        if registers:
            # Each observation contributes once, averaging its candidate diagnostics.
            values[row['endpoint']].append(np.mean(registers, axis=0))
    return [
        dict(
            endpoint=endpoint,
            n_observations=len(v),
            mean_normalized_entropy=float(np.mean(v, axis=0)[0]),
            mean_maximum_weight=float(np.mean(v, axis=0)[1]),
        )
        for endpoint, v in sorted(values.items())
    ]


def intervals(rows, token):
    parents = {row['observation_id']: row['peptide'] for row in rows}
    results = []
    for key, subset in groups(rows).items():
        era, endpoint, cls, population, study, method = key
        if endpoint not in {'ic50', 'elution_observation'}:
            continue
        if study != 'all' and endpoint != 'elution_observation':
            continue
        exact = [row for row in subset if row['qualifier'] == '=']
        if not exact:
            continue
        metric = Metric(
            [r['target'] for r in exact],
            [r['prediction'] for r in exact],
            binary=endpoint in BINARY,
        )
        peptide = [
            parents[r['observation']['provenance']['parent_observation_id']]
            if r['source'] == 'hitlist_ligand_background'
            else r['peptide']
            for r in exact
        ]
        components = [r['group_id'] for r in exact]
        for scope, labels, minimum in (
            ('study_component', components, 4),
            ('within_cohort_peptide', peptide, 20),
        ):
            results.append(
                dict(
                    era=era,
                    endpoint=endpoint,
                    mhc_class=cls,
                    population=population,
                    study=study,
                    assay_method=method,
                    scope=scope,
                    **bootstrap(
                        metric, labels, minimum_clusters=minimum, token=token + repr(key) + scope
                    ),
                )
            )
    return results


def main():
    verified = read(F / 'results/prediction-verification.json')
    decision = read(F / 'results/validation-decision.json')
    prepared = read(F / 'results/prepared.json')
    assert verified['status'] == 'passed' and decision['test_used_for_selection'] is False
    strata = read(F / 'results/stratified-metrics.json')
    support, runtime, coverage, features, detailed, registers, uncertainty = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    started = time.time()
    for condition in sorted(verified['conditions']):
        folder = F / 'results' / condition
        report = read(folder / 'report.json')
        collection = read(folder / 'collection.json')
        resolved = read(folder / 'resolved-config.json')
        assert sha(folder / 'report.json') == decision['report_sha256'][condition]
        for endpoint in prepared['model']['endpoints']:
            for cls in ('I', 'II'):
                row = dict(condition=condition, endpoint=endpoint, mhc_class=cls)
                for split in ('train', 'validation', 'test'):
                    prefix = f'{split}:{endpoint}:{cls}:'
                    row[split + '_n'] = sum(
                        n for k, n in prepared['counts'].items() if k.startswith(prefix)
                    )
                    row[split + '_exact'] = prepared['counts'].get(prefix + '=', 0)
                row['optimizer_exposures'] = report['training_exposures'].get(
                    endpoint + ':' + cls, 0
                )
                row['missing_validation'] = row['validation_n'] == 0
                row['missing_test'] = row['test_n'] == 0
                support.append(row)
        runtime.append(
            dict(
                condition=condition,
                selected_steps=report['selected_steps'],
                completed_steps=report['summary']['steps'],
                selected_unique_rows=report['selected_unique_training_rows_seen'],
                completed_unique_rows=report['unique_training_rows_seen'],
                parameters=report['parameters'],
                requested_gpu=report['hardware']['requested_gpu'],
                actual_gpu=report['hardware']['actual_gpu'],
                **report['resources'],
            )
        )
        raw_coverage = read(Path(collection['directory']) / 'raw/coverage.json')
        coverage.extend(
            dict(condition=condition, parameter=k, **v)
            for k, v in raw_coverage['parameter_elements'].items()
        )
        features.extend(
            dict(condition=condition, parameter=k, **v)
            for k, v in raw_coverage['input_features'].items()
        )
        for split in ('validation', 'test'):
            path = Path(collection['predictions'][split]['path'])
            assert (
                sha(path)
                == verified['conditions'][condition]['splits'][split]['predictions_sha256']
            )
            with gzip.open(path, 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            for key, subset in detailed_groups(rows).items():
                if len(subset) < 20:
                    continue
                era, endpoint, cls, population, study, method, dimension, value = key
                score = metrics(subset, resolved['scales'].get(endpoint, {}).get('scale', 1.0))
                if endpoint not in BINARY:
                    exact = [r for r in subset if r['qualifier'] == '=']
                    score['mean_error'] = (
                        float(np.mean([r['prediction'] - r['target'] for r in exact]))
                        if exact
                        else None
                    )
                detailed.append(
                    dict(
                        condition=condition,
                        split=split,
                        era=era,
                        endpoint=endpoint,
                        mhc_class=cls,
                        population=population,
                        study=study,
                        assay_method=method,
                        dimension=dimension,
                        value=value,
                        **score,
                    )
                )
            registers.extend(
                dict(condition=condition, split=split, **r) for r in register_summary(rows)
            )
            if condition == decision['winner']:
                print('Cluster uncertainty:', condition, split, flush=True)
                uncertainty.extend(
                    dict(condition=condition, split=split, **r)
                    for r in intervals(rows, condition + ':' + split)
                )
            del rows
    for name, values in (
        ('output-class-support', support),
        ('runtime-and-exposure', runtime),
        ('parameter-coordinate-coverage', coverage),
        ('input-feature-coverage', features),
        ('failure-strata', detailed),
        ('register-diagnostics', registers),
        ('cluster-uncertainty', uncertainty),
    ):
        table(name, values)
    draw(strata)
    write(
        'reporting-receipt.json',
        dict(
            status='completed',
            argv=sys.argv,
            source_sha256=sha(Path(__file__)),
            uncertainty_source_sha256=sha(F / 'analysis/uncertainty.py'),
            prediction_verification_sha256=sha(F / 'results/prediction-verification.json'),
            selected_condition=decision['winner'],
            started_unix=started,
            finished_unix=time.time(),
            interpretation=(
                'Saved-prediction analysis only. Study/peptide intervals are conditional '
                'on the fitted model. Register weights have no measured truth qualification. '
                'A coordinate update does not establish physical identifiability.'
            ),
        ),
    )


def draw(rows):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conditions = sorted({r['condition'] for r in rows})
    colors = ['#2a6fbb', '#c85836']
    panels = [
        ('ic50', 'reported_measurements', 'spearman', 'Direct IC50 · Spearman'),
        (
            'elution_observation',
            'synthetic_same_protein',
            'auroc',
            'Ligands / same-protein · AUROC',
        ),
        (
            'elution_observation',
            'synthetic_reference_proteome',
            'auroc',
            'Ligands / proteome · AUROC',
        ),
        (
            'elution_observation',
            'synthetic_same_protein',
            'auprc',
            'Ligands / same-protein · AUPRC',
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for ax, (endpoint, population, metric, title) in zip(axes.flat, panels, strict=True):
        subset = [
            r
            for r in rows
            if r['endpoint'] == endpoint
            and r['population'] == population
            and r['study'] == 'all'
            and r.get(metric) is not None
        ]
        keys = sorted({(r['split'], r['era'], r['mhc_class']) for r in subset})
        for index, condition in enumerate(conditions):
            data = {
                (r['split'], r['era'], r['mhc_class']): r
                for r in subset
                if r['condition'] == condition
            }
            ax.scatter(
                [i + (index - 0.5) * 0.15 for i, key in enumerate(keys) if key in data],
                [data[key][metric] for key in keys if key in data],
                color=colors[index],
                label=condition,
                s=40,
            )
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_xticks(range(len(keys)), ['\n'.join(key) for key in keys], fontsize=8)
        ax.set_ylim(-1 if metric == 'spearman' else 0, 1)
        ax.grid(axis='y', alpha=0.2)
        if not keys:
            ax.text(0.5, 0.5, 'No supported evaluation cohort', transform=ax.transAxes, ha='center')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        'Shared multi-output model · whole-study holdouts\n'
        'Each dot is one seed; generated ligand backgrounds are unobserved comparators',
        fontsize=12,
    )
    for suffix in ('png', 'pdf'):
        fig.savefig(F / 'results' / ('performance-comparison.' + suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
