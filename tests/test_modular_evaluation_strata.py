"""Candidate multiplicity and assay identity cannot silently change metric populations."""

import pytest
from presto.training.modular_evaluation import summarize, summarize_strata


def record(label, score, cls, method='mass spectrometry', study='paper'):
    return dict(
        endpoint='elution_observation',
        target=label,
        prediction=score,
        loss=0.1,
        mhc_classes=[cls, cls],
        qualifier='=',
        observation={'provenance': {'study_id': study, 'raw_record': {'assay_method': method}}},
    )


def test_multi_candidate_ligands_count_once_and_separate_classes():
    rows = [record(1, 2, 'I'), record(0, -2, 'I'), record(1, -1, 'II'), record(0, 1, 'II')]
    result = summarize(rows)['elution_observation']
    assert result['n'] == 4
    assert result['by_class']['I']['n'] == 2
    assert result['by_class']['I']['auroc'] == 1
    assert result['by_class']['II']['auroc'] == 0
    strata = summarize_strata(rows)
    assert len(strata) == 2
    assert strata[0]['study'] == 'paper'
    assert strata[0]['metrics']['elution_observation']['n'] == 2


def test_missing_metadata_is_visible_and_bad_group_keys_fail():
    row = record(1, 0, 'I')
    row['observation'] = {}
    assert summarize_strata([row])[0]['assay_method'] == 'unknown'
    assert summarize_strata([row])[0]['study'] == 'unknown'
    with pytest.raises(ValueError, match='stratum fields'):
        summarize_strata([row], fields=('prediction',))


def test_training_diagnostic_limit_never_subsamples_heldouts():
    from types import SimpleNamespace
    from presto.training.modular_run import evaluation_indices

    index = SimpleNamespace(
        partitions={f'{split}:assay': list(range(100)) for split in ('train', 'validation', 'test')}
    )
    training = {'seed': 42, 'train_evaluation_max_rows': 7}
    indices, metadata = evaluation_indices(index, 'train', training)
    assert len(indices) == len(set(indices)) == 7
    assert metadata['available_rows'] == 100 and metadata['evaluated_rows'] == 7
    assert evaluation_indices(index, 'train', training)[0] == indices
    for split in ('validation', 'test'):
        assert evaluation_indices(index, split, training)[0] == list(range(100))
    assert evaluation_indices(index, 'train', {'seed': 42})[0] == list(range(100))


def test_equal_endpoint_selection_is_invariant_to_replication_within_assay():
    from presto.training.modular_run import _selection_score

    metrics = {'ic50': {'loss': 2.0, 'n': 10}, 'elution_observation': {'loss': 0.2, 'n': 100}}
    assert _selection_score(metrics, reduction='endpoint_mean') == pytest.approx(1.1)
    metrics['elution_observation']['n'] *= 10
    assert _selection_score(metrics, reduction='endpoint_mean') == pytest.approx(1.1)
    assert _selection_score(metrics) == pytest.approx(220 / 1010)
