"""Critical production-path boundaries before any biological fit."""

import pytest
import copy
import json
import sys
import unittest
from admit import derive
from common import ROOT, read, write
from model import ARMS, fit, fit_fold, predict


def fixture():
    rows = []
    for group, method, n, intercept in (('a', '1', 5, 3.0), ('b', '2', 3, 8.0), ('c', '1', 4, 5.0)):
        for index in range(n):
            x = -index / 3
            rows.append(
                {
                    'candidate_id': group + str(index),
                    'group': group,
                    'method': method,
                    'nominal_level_id': str(index),
                    'x': x,
                    'y': intercept + 1.7 * x,
                }
            )
    return rows


class Boundaries(unittest.TestCase):
    def test_known_slopes_and_equal_group_weight(self):
        rows = fixture()
        for arm in ARMS:
            model = fit(rows, arm)
            for group in ('a', 'b', 'c'):
                assert sum(
                    (w for w, r in zip(model['weights'], rows, strict=True) if r['group'] == group)
                ) == pytest.approx(1 / 3, abs=1e-10)
            if arm in ('shared_slope', 'method_slope'):
                for y, row in zip(predict(model, rows), rows, strict=True):
                    assert y == pytest.approx(row['y'], abs=1e-10)
                for slope in model['coefficients'][len(model['groups']) :]:
                    assert slope == pytest.approx(1.7, abs=1e-10)
            else:
                for group in model['groups']:
                    selected = [r for r in rows if r['group'] == group]
                    expected = sum(
                        (r['y'] - (r['x'] if arm == 'unit_slope' else 0) for r in selected)
                    ) / len(selected)
                    assert model['coefficients'][model['groups'].index(group)] == pytest.approx(
                        expected, abs=1e-10
                    )

    def test_distinct_method_slopes(self):
        rows = fixture()
        for r in rows:
            r['y'] += 2.3 * r['x'] if r['method'] == '2' else 0
        model = fit(rows, 'method_slope')
        assert model['coefficients'][-2] == pytest.approx(1.7, abs=1e-10)
        assert model['coefficients'][-1] == pytest.approx(4.0, abs=1e-10)

    def test_holdout_targets_cannot_change_fit_or_predictions(self):
        rows = fixture()
        altered = copy.deepcopy(rows)
        for row in altered:
            if row['nominal_level_id'] == '0':
                row['y'] += 1000
        for arm in ARMS:
            model, test, values = fit_fold(rows, arm, '0')
            changed, _, other = fit_fold(altered, arm, '0')
            assert model == changed
            assert values == other
            assert set(model['train_ids']).isdisjoint((r['candidate_id'] for r in test))

    def test_unidentified_and_nonfinite_inputs_fail(self):
        rows = fixture()
        for r in rows:
            r['x'] = 0
        for arm in ARMS:
            with pytest.raises(ValueError, match='two distinct levels'):
                fit(rows, arm)
        rows = fixture()
        rows[0]['y'] = float('nan')
        with pytest.raises(ValueError, match='Nonfinite'):
            fit(rows, 'shared_slope')
        with pytest.raises(ValueError, match='duplicate'):
            fit(fixture() + fixture(), 'shared_slope')

    def test_no_positive_only_transform_of_missing_or_zero(self):
        row = {
            'decision': 'accepted',
            'target_value': 3.0,
            'target_unit': 'source_signal_arbitrary_units',
            'aligned.modified_sequence_raw': 'PEPTIDE',
            'aligned.charge_raw': '2',
        }
        level = {
            'nominal_level_id': 'l',
            'nominal_concentration': 1.0,
            'nominal_concentration_raw': '1',
            'measurement_unit': 'test_unit',
            'nominal_status': 'nominal',
        }
        args = [
            {'standard_id': 's'},
            {'peptide_sequence': 'PEPTIDE'},
            {'dia_index_from_name': 1, 'method_link_precision': 'fixture'},
            level,
        ]
        assert derive(row, *args)['y'] == pytest.approx(0.47712125471966244, abs=1e-10)
        for value in (0.0, -1.0, None, float('nan'), float('inf')):
            with pytest.raises(ValueError, match='positive'):
                derive({**row, 'target_value': value}, *args)
        with pytest.raises(ValueError, match='positive'):
            derive({**row, 'decision': 'excluded'}, *args)

    def test_actual_shared_recipe_keeps_missingness_distinct(self):
        cfg = read(ROOT / 'reproduce/launch.json')
        sys.path.insert(0, cfg['hitlist_checkout'])
        from hitlist.evidence_recipe import _decision, _validate

        recipe = read(ROOT / 'reproduce/recipe-template.json')
        _validate(recipe)
        source = recipe['sources'][0]
        row = {key: 'present' for key in source['required_fields']}
        row.update(
            {
                'aligned.quantitative_value': 2.0,
                'aligned.measurement_unit': 'source_signal_arbitrary_units',
                'aligned.value_status': 'positive',
                'feature_link_status': 'matched_identity',
            }
        )
        assert _decision(row, source)[:3] == ('accepted', 2.0, [])
        result = _decision(
            {**row, 'aligned.quantitative_value': 0.0, 'aligned.value_status': 'zero'}, source
        )
        assert result[:3] == ('ambiguous', None, ['value_status_not_allowed'])
        result = _decision({**row, 'feature_link_status': 'missing_feature_id'}, source)
        assert result[:3] == ('excluded', None, ['context_precision_unmet'])
        result = _decision(
            {**row, 'aligned.quantitative_value': None, 'aligned.value_status': 'missing'}, source
        )
        assert result[:3] == ('ambiguous', None, ['missing_target'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Boundaries)
    )
    report = {
        'passed': result.wasSuccessful(),
        'tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'contract': 'synthetic software boundaries only; no biological labels',
    }
    write(ROOT / 'results/boundary-tests.json', report)
    print(json.dumps(report), flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
