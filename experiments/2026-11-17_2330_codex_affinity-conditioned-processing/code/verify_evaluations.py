"""Audit common examples, additive readouts and independently computed metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path(__file__).resolve().parents[1]


def main():
    frame = pd.read_parquet(E / 'results/analysis/predictions.parquet')
    metrics = json.loads((E / 'results/analysis/metrics.json').read_text())
    checks, numeric, largest_error = [], 0, 0.0
    parent_receipt = json.loads((E / 'results/evaluation/parent/receipt.json').read_text())
    for path in sorted((E / 'results/evaluation').glob('*/receipt.json')):
        receipt = json.loads(path.read_text())
        model = path.parent.name
        assert not receipt['test_accessed'] and receipt['affinity_context_max_delta'] == 0
        assert receipt['data_sha256'] == parent_receipt['data_sha256']
        if model != 'parent':
            fit = model.removesuffix('-terminal')
            fit_receipt = json.loads((E / 'results/fits' / fit / 'receipt.json').read_text())
            if model.endswith('-terminal'):
                assert receipt['steps'] == fit_receipt['steps']
            else:
                assert receipt['checkpoint_sha256'] == fit_receipt['checkpoint_sha256']
        for population in receipt['metrics']:
            actual = frame[(frame.model == model) & (frame.population == population)].set_index('id').sort_index()
            baseline = frame[(frame.model == 'parent') & (frame.population == population)].set_index('id').sort_index()
            assert list(actual.index) == list(baseline.index)
            for column in ('target', 'peptide', 'qualifier', 'endpoint', 'class', 'pair', 'group', 'mhc_signature'):
                pd.testing.assert_series_equal(actual[column], baseline[column], check_dtype=False)
            presentation = actual[actual.endpoint == 'elution_observation']
            if not presentation.empty:
                np.testing.assert_allclose(presentation.prediction,
                    presentation.processing_supply + presentation.without_processing_logit, atol=1e-12, rtol=0)
                np.testing.assert_allclose(presentation.prediction,
                    presentation.junction_processing + presentation.without_junction_logit, atol=1e-12, rtol=0)
                if not model.startswith('paired-junction'):
                    np.testing.assert_array_equal(presentation.junction_processing, np.zeros(len(presentation)))
            for record in metrics:
                if record['model'] != model or record['population'] != population:
                    continue
                ep, cls = record['endpoint'], record['class']
                expected = receipt['metrics'][population][ep]['by_class'][cls]
                if ep == 'elution_observation':
                    if record['readout'] != 'prediction':
                        continue
                    assert record['n'] == expected['n']
                    mappings = [('auroc', expected['auroc']), ('auprc', expected['auprc'])]
                else:
                    assert record['exact_n'] == expected['n']
                    mappings = [(k, expected.get(v)) for k, v in [('spearman', 'spearman'), ('pearson', 'pearson'), ('rmse_log10_nm', 'rmse')]]
                    binding = expected.get('binding_at_500_nM', {})
                    mappings += [(k, binding.get(v)) for k, v in [('accuracy', 'accuracy'), ('balanced_accuracy', 'balanced_accuracy'),
                        ('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1'), ('binding_auroc', 'auroc'), ('binding_auprc', 'auprc')]]
                for key, value in mappings:
                    if record.get(key) is None or value is None:
                        continue
                    error = abs(record[key] - value)
                    assert error < 1e-10, (model, population, ep, cls, key, error)
                    largest_error = max(largest_error, error)
                    numeric += 1
            checks.append({'model': model, 'population': population, 'rows': len(actual), 'same_examples_and_targets': True})
    result = {'status': 'passed', 'test_accessed': False, 'cohort_checks': checks,
        'independent_numeric_checks': numeric, 'maximum_numeric_error': largest_error,
        'selection_uncertainty_included': False}
    (E / 'results/evaluation-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Verified', len(checks), 'model/cohort combinations;', numeric, 'numeric checks')


if __name__ == '__main__':
    main()
