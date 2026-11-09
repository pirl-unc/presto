"""Weighted cluster statistics agree with independent expanded-sample references."""

from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uncertainty import Metric, bootstrap


@pytest.mark.parametrize('seed', range(5))
def test_tied_weighted_statistics_match_literal_repeated_samples(seed):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 5, 40)
    prediction = rng.integers(0, 5, 40)
    copies = rng.integers(0, 4, 40)
    y, p = np.repeat(target, copies), np.repeat(prediction, copies)
    result = Metric(target, prediction)(copies)
    assert result['spearman'] == pytest.approx(spearmanr(y, p).statistic)
    assert result['pearson'] == pytest.approx(pearsonr(y, p).statistic)
    assert result['rmse'] == pytest.approx(np.sqrt(np.mean((y - p) ** 2)))
    result = Metric(target >= 2, prediction, binary=True)(copies)
    assert result['auroc'] == pytest.approx(roc_auc_score(y >= 2, p))
    assert result['auprc'] == pytest.approx(average_precision_score(y >= 2, p))


def test_single_label_and_zero_weight_observations_remain_undefined():
    result = Metric([0, 0, 1], [1, 2, 3], binary=True)([1, 2, 0])
    assert result == {'auroc': None, 'auprc': None}
    result = Metric([1, 1, 2], [1, 2, 3])([1, 2, 0])
    assert result['spearman'] is None and result['pearson'] is None


def test_cluster_minimum_and_reproducibility():
    metric = Metric([0, 1, 0, 1], [-1, 1, 0, 2], binary=True)
    assert bootstrap(metric, ['a'] * 4, minimum_clusters=4, token='x')['intervals'] is None
    kwargs = dict(minimum_clusters=2, token='x', draws=20)
    first = bootstrap(metric, ['a', 'a', 'b', 'b'], **kwargs)
    assert first == bootstrap(metric, ['a', 'a', 'b', 'b'], **kwargs)
    assert first['intervals']['auroc']['percentile_95'] == [1.0, 1.0]
