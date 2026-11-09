"""Screening must detect possible limits without inventing inequalities."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'audit_methods', Path(__file__).with_name('audit_methods.py')
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def row(i, value, qualifier='=', negative=True):
    return dict(
        value_nM=value,
        qualifier=qualifier,
        raw_qualifier='',
        qualitative='Negative' if negative else 'Positive',
        assay_id=str(i),
    )


def test_preparation_is_not_fluorescence():
    a = M.method_axes('purified MHC/competitive/fluorescence')
    b = M.method_axes('cellular MHC/direct/fluorescence')
    assert a['readout'] == b['readout']
    assert a['preparation'] != b['preparation'] and a['mechanism'] != b['mechanism']
    assert M.method_axes(None)['readout'] == 'unknown'


def test_plateau_screen_does_not_relabel():
    rows = [row(i, 20000) for i in range(10)] + [
        row(i + 10, i + 1, negative=False) for i in range(10)
    ]
    heaps = M.heap_candidates(rows)
    assert len(heaps) == 1
    assert heaps[0]['right_limit_review_candidate']
    assert heaps[0]['exact_at_value'] == heaps[0]['exact_missing_raw_qualifier'] == 10
    assert all(r['qualifier'] == '=' for r in rows)


def test_repeated_internal_value_is_not_automatically_upper_limit():
    rows = [row(i, 20000) for i in range(6)] + [row(i + 6, 30000 + i) for i in range(14)]
    heaps = M.heap_candidates(rows)
    assert len(heaps) == 1 and not heaps[0]['right_limit_review_candidate']
    assert heaps[0]['reported_values_above'] == 14


def test_small_or_unrepeated_groups_not_screened():
    assert M.heap_candidates([row(i, 20000) for i in range(19)]) == []
    assert M.heap_candidates([row(i, i + 1) for i in range(20)]) == []


def test_explicit_bounds_remain_separate():
    rows = [row(i, 25000) for i in range(5)] + [row(i + 5, 25000, '>') for i in range(15)]
    heap = M.heap_candidates(rows)[0]
    assert heap['exact_at_value'] == 5 and heap['explicit_right_bounds_at_value'] == 15


def test_method_agreement_excludes_bounds_and_matches_response_strata():
    rows = []
    for i in range(3):
        for method, endpoint, mult in [
            ('purified MHC/competitive/radioactivity', 'ic50', 1),
            ('purified MHC/competitive/fluorescence', 'kd', 10),
        ]:
            rows.append(
                dict(
                    mhc_class='I',
                    qualifier='=',
                    endpoint=endpoint,
                    method=method,
                    pmhc_sha256=str(i),
                    group_id='study',
                    value_nM=10**i * mult,
                )
            )
    rows.append({**rows[0], 'pmhc_sha256': 'bounded', 'qualifier': '>'})
    metrics, pairs = M.correlations(rows)
    full = next(
        r
        for r in metrics
        if r['mhc_class'] == 'I' and r['response_stratum'] == 'all_concentration_labels'
    )
    assert full['n'] == 3 and full['spearman'] == 1
    assert all(r['n'] == 0 for r in metrics if r['response_stratum'] == 'ic50')
    assert len(pairs) == 3
