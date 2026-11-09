"""Independent checks cover ties, one-sided bounds and separated ligand populations."""

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_predictions import binary, compare, groups, metrics


def test_tied_binary_scores_and_single_label_cohorts():
    result = binary([0, 1, 0, 1], [1, 1, 1, 1], [True] * 4)
    assert result['auroc'] == 0.5 and result['auprc'] == 0.5
    assert binary([0, 0], [2, -1], [True, False])['auroc'] is None
    assert binary([0, 0], [2, -1], [True, False])['auprc'] is None


def test_inequality_losses_and_physical_threshold_metrics():
    rows = [
        dict(endpoint='ic50', qualifier=q, target=y, prediction=p, loss=loss)
        for q, y, p, loss in [
            ('=', 1.0, 2.0, 0.25),
            ('=', 3.0, 2.0, 0.25),
            ('>', 4.0, 5.0, 0.0),
            ('<', 1.0, 2.0, 0.25),
        ]
    ]
    result = metrics(rows, 2.0)
    assert result['n_exact'] == 2 and result['n_censored'] == 2
    assert result['censor_satisfaction'] == 0.5
    assert result['rmse'] == 1.0 and result['spearman'] is None
    assert result['binding_at_500_nM']['auroc'] == 0.5
    rows[-1]['loss'] = 0.0
    with pytest.raises(AssertionError):
        metrics(rows, 2.0)


def test_binary_loss_and_common_report_agree_independently():
    from presto.training.modular_evaluation import summarize

    rows = [
        dict(
            endpoint='qualitative_binding',
            qualifier='=',
            target=y,
            prediction=p,
            mhc_classes=['I'],
            loss=float(np.logaddexp(0, p) - p * y),
        )
        for y, p in [(0.0, -2.0), (1.0, 1.0), (0.0, 1.0), (1.0, 3.0)]
    ]
    compare(metrics(rows, 1.0), summarize(rows)['qualitative_binding'], binary_output=True)


def test_report_groups_keep_source_negatives_out_of_sampled_auc():
    def row(source, value, population):
        return dict(
            endpoint='elution_observation',
            source=source,
            target=value,
            mhc_classes=['I'],
            observation={
                'provenance': {
                    'evaluation_era': '2026',
                    'evaluation_population': population,
                    'study_id': 'pmid:test',
                    'raw_record': {'assay_method': 'MS'},
                }
            },
        )

    rows = [
        row('reported', 1.0, 'synthetic_same_protein'),
        row('reported', 0.0, 'source_reported_nondetection'),
        row('hitlist_ligand_background', 0.0, 'synthetic_same_protein'),
    ]
    result = groups(rows)
    synthetic = result[('2026', 'elution_observation', 'I', 'synthetic_same_protein', 'all', 'all')]
    reported = result[
        ('2026', 'elution_observation', 'I', 'source_reported_outcomes', 'pmid:test', 'MS')
    ]
    assert synthetic == [rows[0], rows[2]] and reported == rows[:2]
