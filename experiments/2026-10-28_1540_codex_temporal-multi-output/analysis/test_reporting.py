"""Reporting retains bag-level outcomes and parent-dependent comparator clusters."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import detailed_groups, register_summary, intervals


def row(identifier, value, source='reported', **extra):
    return dict(
        observation_id=identifier,
        peptide=identifier,
        endpoint='elution_observation',
        source=source,
        target=value,
        prediction=value,
        qualifier='=',
        mhc_classes=['II'],
        group_id='study-component',
        observation=dict(
            provenance=dict(
                study_id='study',
                evaluation_era='2026',
                evaluation_population='synthetic_same_protein',
                raw_record={},
                **extra,
            ),
            mhc=[{'provenance': {'catalog_alleles': [name]}} for name in ('allele-a', 'allele-b')],
        ),
        diagnostics=dict(
            candidates=[
                {'registers': [{'class': 'II', 'weights': [0.5, 0.5]}]},
                {'registers': [{'class': 'II', 'weights': [1.0, 0.0]}]},
            ]
        ),
    )


def test_candidate_sets_do_not_duplicate_measurements():
    positive = row('parent', 1)
    subsets = detailed_groups([positive])
    allele_strata = [(k, v) for k, v in subsets.items() if k[-2] == 'catalog_allele_set']
    assert all(k[-1] == 'allele-a;allele-b' and v == [positive] for k, v in allele_strata)
    register = register_summary([positive])[0]
    assert register['n_observations'] == 1
    assert register['mean_normalized_entropy'] == pytest.approx(0.5)
    assert register['mean_maximum_weight'] == 0.75


def test_generated_comparator_resamples_with_positive_parent(monkeypatch):
    observed = []

    def capture(metric, clusters, **kwargs):
        observed.append((kwargs['token'], clusters))
        return {'status': 'fixture'}

    monkeypatch.setattr('report.bootstrap', capture)
    rows = [
        row('parent', 1),
        row('other-sequence', 0, 'hitlist_ligand_background', parent_observation_id='parent'),
    ]
    intervals(rows, 'fixture')
    assert any(
        clusters == ['parent', 'parent']
        for token, clusters in observed
        if token.endswith('within_cohort_peptide')
    )
    assert all(
        set(clusters) == {'study-component'}
        for token, clusters in observed
        if token.endswith('study_component')
    )
