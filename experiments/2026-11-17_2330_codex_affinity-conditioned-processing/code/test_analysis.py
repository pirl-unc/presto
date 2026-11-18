"""Independent checks against literal cluster resampling, including score ties."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('processing_analysis_shared', Path(__file__).with_name('analysis_shared.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
complete_pairs, paired_draws, weighted_auc_draws = module.complete_pairs, module.paired_draws, module.weighted_auc_draws


def test_weighted_auc_matches_expanded_rows_with_ties():
    target = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([0.1, 0.4, 0.4, 0.4, 0.8, 0.9])
    weights = np.array([[1, 1, 1, 1, 1, 1], [0, 2, 1, 3, 0, 1], [2, 0, 3, 0, 1, 2]])
    expected = [roc_auc_score(np.repeat(target, w), np.repeat(scores, w)) for w in weights]
    np.testing.assert_allclose(weighted_auc_draws(target, scores, weights), expected, atol=1e-14)


def fixture():
    return pd.DataFrame({
        'pair': ['a', 'a', 'b', 'b', 'c', 'c'],
        'target': [0, 1, 0, 1, 0, 1],
        'bootstrap_group': ['x', 'x', 'x', 'x', 'y', 'y'],
        'prediction_real': [0.2, 0.4, 0.4, 0.4, 0.8, 0.7],
        'prediction_other': [0.3, 0.3, 0.5, 0.1, 0.6, 0.7],
    })


def test_bootstrap_matches_literal_cluster_repeats_and_pair_wins():
    rows = fixture()
    draws = np.array([[0, 0], [1, 1], [0, 1]])
    expected = []
    for draw in draws:
        blocks = []
        wins = {'real': [], 'other': []}
        for index in draw:
            block = rows[rows.bootstrap_group == ['x', 'y'][index]]
            blocks.append(block)
            for _, pair in block.groupby('pair'):
                pair = pair.set_index('target')
                for key in wins:
                    delta = pair.loc[1, 'prediction_' + key] - pair.loc[0, 'prediction_' + key]
                    wins[key].append(float(delta > 0) + 0.5 * float(delta == 0))
        sampled = pd.concat(blocks)
        expected.append([
            roc_auc_score(sampled.target, sampled.prediction_real)
            - roc_auc_score(sampled.target, sampled.prediction_other),
            np.mean(wins['real']) - np.mean(wins['other']),
        ])
    np.testing.assert_allclose(paired_draws(rows, draws, ['x', 'y']), expected, atol=1e-14)
    rows['prediction_other'] = rows.prediction_real
    np.testing.assert_array_equal(paired_draws(rows, draws, ['x', 'y']), np.zeros((3, 2)))


def test_pair_checks_reject_incomplete_duplicate_and_split_cluster_pairs():
    rows = fixture()
    assert complete_pairs(rows)
    assert not complete_pairs(rows.iloc[1:])
    assert not complete_pairs(pd.concat([rows, rows.iloc[:1]]))
    rows.loc[0, 'bootstrap_group'] = 'y'
    with pytest.raises(AssertionError):
        paired_draws(rows, np.array([[0, 1]]), ['x', 'y'])
