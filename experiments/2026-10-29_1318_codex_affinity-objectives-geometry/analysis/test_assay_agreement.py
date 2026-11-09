"""Scientific aggregation invariants on synthetic records, without fitting a model."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'assay_agreement', Path(__file__).with_name('assay_agreement.py')
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def row(identity, peptide, endpoint, value, study='a', qualifier='=', unit='nM', groove='g2'):
    return dict(
        observation_id=identity,
        split='train',
        endpoint=endpoint,
        peptide=peptide,
        mhc=[dict(mhc_class='I', groove1='g1', groove2=groove)],
        value=value,
        unit=unit,
        qualifier=qualifier,
        group_id=study,
        provenance=dict(
            measurement_namespace='fixture',
            original_measurement_id=identity,
            raw_record=dict(assay_method='synthetic fixture'),
        ),
    )


def run(tmp_path, rows):
    path = tmp_path / 'rows.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out = tmp_path / 'out'
    result = AUDIT.audit([(path, hashlib.sha256(path.read_bytes()).hexdigest())], out)
    with gzip.open(out / 'paired-values.jsonl.gz', 'rt') as stream:
        pairs = [json.loads(line) for line in stream]
    return result, pairs


def test_one_vote_per_study_and_pmhc_no_cartesian_replication(tmp_path):
    rows = [row(f'a{i}', 'p1', 'ic50', v) for i, v in enumerate([1, 10, 100])]
    rows += [row('b1', 'p1', 'kd', 10), row('b2', 'p1', 'kd', 1000)]
    # This extra study gets one vote despite having only one replicate.
    rows += [row('c', 'p1', 'ic50', 100000, study='b')]
    result, pairs = run(tmp_path, rows)
    selected = [p for p in pairs if p['endpoint_a'] == 'ic50' and p['endpoint_b'] == 'kd']
    assert len(selected) == 2
    within = next(p for p in selected if p['scope'] == 'within_study')
    pooled = next(p for p in selected if p['scope'] == 'pooled_pmhc')
    assert (within['a_log10_nM'], within['b_log10_nM']) == (1, 2)
    assert (pooled['a_log10_nM'], pooled['b_log10_nM']) == (3, 2)
    assert result['exact_observation_count'] == 6


def test_bounds_excluded_units_normalized_and_both_grooves_identify_pmhc(tmp_path):
    rows = [
        row('a', 'p1', 'ic50', 1, unit='uM'),
        row('b', 'p1', 'kd', 1000, study='b'),
        row('c', 'p2', 'ic50', 10, qualifier='>'),
        row('d', 'p2', 'kd', 20),
        row('e', 'p3', 'ic50', 5),
        row('f', 'p3', 'kd', 5, groove='other'),
    ]
    result, pairs = run(tmp_path, rows)
    assert result['exact_observation_count'] == 5
    assert len(pairs) == 2
    assert {p['scope'] for p in pairs} == {'pooled_pmhc', 'disjoint_studies'}
    assert all(p['a_log10_nM'] == p['b_log10_nM'] == 3 for p in pairs)
    comparison = next(
        r
        for r in result['comparisons']
        if r['mhc_class'] == 'I'
        and r['endpoint_a'] == 'ic50'
        and r['endpoint_b'] == 'kd'
        and r['scope'] == 'pooled_pmhc'
    )
    assert comparison['overlap_including_bounds'] == 2
    assert comparison['n'] == 1 and comparison['spearman'] is None


def test_rank_and_linear_agreement_are_distinct(tmp_path):
    rows = []
    for i, (a, b) in enumerate([(1, 1), (10, 100), (100, 10000)]):
        rows += [row(f'a{i}', f'p{i}', 'ic50', a), row(f'b{i}', f'p{i}', 'kd', b)]
    result, _ = run(tmp_path, rows)
    c = next(
        r
        for r in result['comparisons']
        if r['mhc_class'] == 'I'
        and r['endpoint_a'] == 'ic50'
        and r['endpoint_b'] == 'kd'
        and r['scope'] == 'within_study'
    )
    assert c['spearman'] == pytest.approx(1)
    assert c['pearson'] == pytest.approx(1)
    assert c['difference_rmse_log10_nM'] == pytest.approx((5 / 3) ** 0.5)


def test_duplicate_original_assay_cannot_be_cross_output_evidence(tmp_path):
    a, b = row('a', 'p1', 'ic50', 1), row('b', 'p1', 'kd', 1)
    b['provenance']['original_measurement_id'] = 'a'
    with pytest.raises(AssertionError, match='original assay identity'):
        run(tmp_path, [a, b])


def test_source_checksum_is_required(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text(json.dumps(row('a', 'p1', 'ic50', 1)) + '\n')
    with pytest.raises(AssertionError):
        AUDIT.audit([(path, 'incorrect')], tmp_path / 'out')
