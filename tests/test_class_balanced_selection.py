import pytest
from presto.training.modular_run import _selection_score


def record(cls, study, loss):
    return {
        'endpoint': 'elution_observation',
        'mhc_classes': [cls, cls],
        'loss': loss,
        'observation': {'provenance': {'study_id': study}},
    }


def score(rows):
    return _selection_score(
        {'elution_observation': {'n': len(rows), 'loss': 999}},
        primary_endpoint='elution_observation',
        reduction='class_study_mean',
        records=rows,
    )


def test_selection_balances_classes_and_studies_not_candidate_counts_or_row_counts():
    rows = [record('I', 'a', 1), record('I', 'b', 3), record('II', 'c', 10)]
    assert score(rows) == 6
    assert score(rows + [record('I', 'a', 1)] * 100) == 6
    assert score(rows + [record('II', 'c', 10)] * 100) == 6


@pytest.mark.parametrize(
    'rows,match',
    [
        ([record('I', 'a', 1)], 'every configured MHC class'),
        ([record('I', 'a', 1), record('II', '', 2)], 'study provenance'),
        ([record('I', 'a', float('nan')), record('II', 'b', 2)], 'finite'),
        (
            [{**record('I', 'a', 1), 'mhc_classes': ['I', 'II']}, record('II', 'b', 2)],
            'one observed',
        ),
    ],
)
def test_invalid_selection_records_fail(rows, match):
    with pytest.raises(ValueError, match=match):
        score(rows)


def test_class_balanced_selection_preflight_rejects_missing_class(tmp_path):
    from presto.tests.test_modular_affinity_controls import fixture
    from presto.data.modular_sources import ObservationIndex
    from presto.training.modular_run import preflight

    path = fixture(tmp_path, selection_endpoint='ic50', selection_reduction='class_study_mean')
    with pytest.raises(ValueError, match='every configured MHC class'):
        preflight(ObservationIndex(path))
