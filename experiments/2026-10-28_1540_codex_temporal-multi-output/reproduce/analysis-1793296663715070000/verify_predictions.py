"""Independent SciPy/sklearn metrics and target/loss checks on saved predictions."""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import warnings
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from verify_cohort import encoded

F = Path(__file__).resolve().parents[1]
BINARY = {'elution_observation', 'qualitative_binding'}
AFFINITY = {'ic50', 'kd', 'ec50', 'kd_via_ic50', 'kd_via_ec50'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def binary(y, score, decision):
    y = np.asarray(y, dtype=bool)
    both = len(set(y)) == 2
    return dict(
        n=len(y),
        positives=int(y.sum()),
        negatives=int((~y).sum()),
        accuracy=float(accuracy_score(y, decision)),
        precision=float(precision_score(y, decision, zero_division=0)),
        recall=float(recall_score(y, decision, zero_division=0)) if y.any() else None,
        f1=float(f1_score(y, decision, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, decision)) if both else None,
        auroc=float(roc_auc_score(y, score)) if both else None,
        auprc=float(average_precision_score(y, score)) if y.any() else None,
    )


def metrics(rows, scale):
    if not rows:
        return dict(n=0)
    endpoint = rows[0]['endpoint']
    assert all(row['endpoint'] == endpoint for row in rows)
    y = np.array([row['target'] for row in rows])
    p = np.array([row['prediction'] for row in rows])
    q = np.array([row['qualifier'] for row in rows])
    assert np.isfinite(y).all() and np.isfinite(p).all()
    if endpoint in BINARY:
        assert np.isin(y, [0, 1]).all() and (q == '=').all()
        losses = np.logaddexp(0, p) - p * y
        result = binary(y, p, p >= 0)
    else:
        residual = p - y
        below, above = np.isin(q, ['<', '<=']), np.isin(q, ['>', '>='])
        residual[below] = np.maximum(0, residual[below])
        residual[above] = np.minimum(0, residual[above])
        losses = (residual / scale) ** 2
        exact = q == '='
        yy, pp = y[exact], p[exact]
        result = dict(n=len(rows), n_exact=int(exact.sum()), n_censored=int((~exact).sum()))
        if (~exact).any():
            result['censor_satisfaction'] = float((residual[~exact] == 0).mean())
        if len(yy):
            varied = len(yy) > 1 and np.ptp(yy) > 0 and np.ptp(pp) > 0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result.update(
                    spearman=float(spearmanr(yy, pp).statistic) if varied else None,
                    pearson=float(pearsonr(yy, pp).statistic) if varied else None,
                    rmse=float(np.sqrt(np.mean((yy - pp) ** 2))),
                    prediction_sd=float(np.std(pp)),
                    target_sd=float(np.std(yy)),
                )
            if endpoint in AFFINITY:
                result['binding_at_500_nM'] = binary(
                    yy <= math.log10(500), -pp, pp <= math.log10(500)
                )
    np.testing.assert_allclose(losses, [r['loss'] for r in rows], rtol=5e-5, atol=3e-6)
    result['loss'] = float(losses.mean())
    return result


def compare(actual, reported, *, binary_output=False):
    assert actual['n'] == reported['n']
    np.testing.assert_allclose(actual['loss'], reported['loss'], rtol=5e-5, atol=3e-6)
    pairs = [(actual, reported)] if binary_output else [(actual, reported.get('exact', {}))]
    if 'binding_at_500_nM' in actual:
        pairs.append((actual['binding_at_500_nM'], reported['exact']['binding_at_500_nM']))
    for left, right in pairs:
        for key in (
            'spearman',
            'pearson',
            'rmse',
            'auroc',
            'auprc',
            'accuracy',
            'balanced_accuracy',
            'precision',
            'recall',
            'f1',
        ):
            if key not in left:
                continue
            if left[key] is None:
                assert right[key] is None
            else:
                np.testing.assert_allclose(left[key], right[key], rtol=2e-7, atol=1e-9)


def groups(rows):
    """Keep source-negative and the two generated comparison populations distinct."""
    result = defaultdict(list)
    for row in rows:
        prov = row['observation']['provenance']
        cls = ';'.join(sorted(set(row['mhc_classes'])))
        era = prov['evaluation_era']
        endpoint = row['endpoint']
        if endpoint != 'elution_observation':
            result[(era, endpoint, cls, 'reported_measurements', 'all', 'all')].append(row)
        elif row['source'] == 'hitlist_ligand_background' or row['target'] == 1:
            population = prov.get('evaluation_population', 'no_background')
            result[(era, endpoint, cls, population, 'all', 'all')].append(row)
        if row['source'] != 'hitlist_ligand_background':
            population = (
                'source_reported_outcomes' if endpoint in BINARY else 'reported_measurements'
            )
            method = prov.get('raw_record', {}).get('assay_method') or 'unknown'
            result[(era, endpoint, cls, population, prov['study_id'], method)].append(row)
    return result


def main():
    decision = json.loads((F / 'results/validation-decision.json').read_text())
    assert decision['status'] == 'frozen' and decision['test_used_for_selection'] is False
    prepared = json.loads((F / 'results/prepared.json').read_text())
    summary, table = {}, []
    for condition in sorted(decision['scores']):
        folder = F / 'results' / condition
        assert sha(folder / 'report.json') == decision['report_sha256'][condition]
        report = json.loads((folder / 'report.json').read_text())
        collection = json.loads((folder / 'collection.json').read_text())
        resolved = json.loads((folder / 'resolved-config.json').read_text())
        assert report['summary']['coverage_gate']['status'] == 'passed'
        assert report['all_training_rows_exposed'] is True
        assert report['serving_replay']['status'] == 'passed'
        splits = {}
        for split in ('validation', 'test'):
            path = Path(collection['predictions'][split]['path'])
            assert sha(path) == collection['predictions'][split]['sha256']
            with gzip.open(path, 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            assert len({r['observation_id'] for r in rows}) == len(rows)
            count = Counter()
            for row in rows:
                obs = row['observation']
                assert row['split'] == split
                for key in ('observation_id', 'peptide', 'source', 'endpoint', 'qualifier'):
                    assert row[key] == obs[key]
                # Linux and macOS libm log10 can differ by a final float64 bit.
                # Keep this far below float32/training precision while rejecting
                # changed physical values, units or endpoint encodings.
                assert math.isclose(row['target'], encoded(obs), rel_tol=1e-14, abs_tol=1e-14)
                cls = ';'.join(sorted(set(row['mhc_classes'])))
                count[f'{split}:{row["endpoint"]}:{cls}:{row["qualifier"]}'] += 1
            assert dict(count) == {
                k: n for k, n in prepared['counts'].items() if k.startswith(split + ':')
            }
            by_endpoint = defaultdict(list)
            for row in rows:
                by_endpoint[row['endpoint']].append(row)
            scores = {}
            for endpoint, subset in by_endpoint.items():
                scale = resolved['scales'].get(endpoint, {}).get('scale', 1.0)
                scores[endpoint] = metrics(subset, scale)
                compare(
                    scores[endpoint],
                    report['summary']['metrics'][split][endpoint],
                    binary_output=endpoint in BINARY,
                )
            for key, subset in groups(rows).items():
                era, endpoint, cls, population, study, method = key
                scale = resolved['scales'].get(endpoint, {}).get('scale', 1.0)
                got = metrics(subset, scale)
                table.append(
                    dict(
                        condition=condition,
                        split=split,
                        era=era,
                        endpoint=endpoint,
                        mhc_class=cls,
                        population=population,
                        study=study,
                        assay_method=method,
                        **got,
                    )
                )
            overall = float(np.mean([r['loss'] for r in rows]))
            np.testing.assert_allclose(overall, report['overall_losses'][split], rtol=1e-12)
            splits[split] = dict(
                n=len(rows), overall_loss=overall, metrics=scores, predictions_sha256=sha(path)
            )
        summary[condition] = dict(
            status='passed', splits=splits, serving_replay=report['serving_replay']
        )
    output = dict(
        status='passed',
        conditions=summary,
        environment={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'scikit-learn')},
    )
    (F / 'results/prediction-verification.json').write_text(
        json.dumps(output, indent=2, allow_nan=False) + '\n'
    )
    (F / 'results/stratified-metrics.json').write_text(
        json.dumps(table, indent=2, allow_nan=False) + '\n'
    )
    columns = sorted({k for row in table for k in row})
    with (F / 'results/stratified-metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in table:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()}
            )
    print('Verified predictions, losses, metrics and strata for', len(summary), 'conditions')


if __name__ == '__main__':
    main()
