"""Rebuild published pair aggregates and heap flags from retained TRAIN records."""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np

F = Path(__file__).resolve().parents[1]


def main():
    verified = json.loads((F / 'results/verification.json').read_text())
    path = Path(verified['artifacts']['concentration-records.jsonl.gz']['path'])
    with path.open('rb') as stream:
        assert (
            hashlib.file_digest(stream, 'sha256').hexdigest()
            == verified['artifacts'][path.name]['sha256']
        )
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    exact = [r for r in rows if r['qualifier'] == '=']
    pair_path = F / 'results/audit/method-pairs.jsonl.gz'
    with gzip.open(pair_path, 'rt') as stream:
        pairs = [json.loads(line) for line in stream]
    for response in {p['response_stratum'] for p in pairs}:
        aggregates = defaultdict(lambda: defaultdict(list))
        for r in exact:
            if response == 'all_concentration_labels' or r['endpoint'] == response:
                aggregates[r['mhc_class'], r['pmhc_sha256'], r['method']][r['group_id']].append(
                    np.log10(r['value_nM'])
                )
        for p in pairs:
            if p['response_stratum'] != response:
                continue
            for side in ('a', 'b'):
                studies = aggregates[p['mhc_class'], p['pmhc_sha256'], p['method_' + side]]
                expected = np.median([np.median(v) for v in studies.values()])
                assert np.isclose(expected, p[side + '_log10_nM'], rtol=0, atol=1e-12)
                assert sorted(studies) == p['studies_' + side]
            assert sorted(set(p['studies_a']) & set(p['studies_b'])) == p['common_studies']
    groups = defaultdict(list)
    fields = ('mhc_class', 'mhc_sha256', 'group_id', 'method', 'endpoint')
    for r in rows:
        groups[tuple(r[k] for k in fields)].append(r)
    reconstructed = {}
    for group, values in groups.items():
        if len(values) < 20:
            continue
        counts = Counter(r['value_nM'] for r in values if r['qualifier'] == '=')
        for v, n in counts.items():
            if n < 5 or n / len(values) < 0.1:
                continue
            same = [r for r in values if r['value_nM'] == v]
            exact_same = [r for r in same if r['qualifier'] == '=']
            negatives = sum(r['qualitative'].lower().startswith('negative') for r in exact_same)
            right = sum(r['qualifier'] in ('>', '>=') for r in same)
            left = sum(r['qualifier'] in ('<', '<=') for r in same)
            above = sum(r['value_nM'] > v for r in values)
            reconstructed[group + (v,)] = dict(
                group_rows=len(values),
                exact_at_value=n,
                fraction_of_group=n / len(values),
                qualitative_negative_exact=negatives,
                explicit_right_bounds_at_value=right,
                explicit_left_bounds_at_value=left,
                reported_values_above=above,
                reported_values_below=sum(r['value_nM'] < v for r in values),
                exact_missing_raw_qualifier=sum(not r['raw_qualifier'] for r in exact_same),
                right_limit_review_candidate=bool(right or (above == 0 and negatives * 2 >= n)),
                left_limit_review_candidate=bool(left),
                assay_ids=sorted(r['assay_id'] for r in exact_same),
            )
    published = json.loads((F / 'results/audit/heap-candidates.json').read_text())
    assert len(published) == len(reconstructed)
    for h in published:
        expected = reconstructed[tuple(h[k] for k in fields) + (h['value_nM'],)]
        for k, v in expected.items():
            assert (sorted(h[k]) if k == 'assay_ids' else h[k]) == v, (k, h[k], v)
    ledger = json.loads((F / 'results/source-ledger.json').read_text())
    sources = Counter()
    for r in ledger['export_metadata']:
        sources[r['source']] += r['n']
    assert sum(sources.values()) == ledger['export']['rows'] == 5440009
    assert sources['supplement'] == 535026
    assert all(
        r['has_ptm'] is None for r in ledger['export_metadata'] if r['source'] == 'supplement'
    )
    assert ledger['train']['source_counts'] == {'iedb': 1641063}
    assert sum(r['n'] for r in ledger['train_studies']) == ledger['train']['rows']
    result = dict(
        status='verified',
        pair_records_rebuilt=len(pairs),
        pair_sides_rebuilt=2 * len(pairs),
        heap_cells_rebuilt=len(published),
        heaped_exact_rows=sum(h['exact_at_value'] for h in published),
        right_limit_review_candidate_rows=sum(
            h['exact_at_value'] for h in published if h['right_limit_review_candidate']
        ),
        export_rows_by_source=dict(sources),
        validation_test_used=False,
        changed_qualifiers=0,
    )
    (F / 'results/aggregate-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
