"""Verify quantitative parsing and projected identities without changing sources."""
import collections
import gzip
import hashlib
import json
import math
import re
import time
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]


def main():
    tick = time.perf_counter()
    audited = json.loads((FAMILY / 'results/canonical-audit.json').read_text())
    root = Path(audited['package_path'])
    assert hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest() == audited['package_manifest_sha256']
    rows = pq.read_table(root / 'tmab/source_rows.parquet').to_pylist()
    quantities = pq.read_table(root / 'tmab/measurements.parquet').to_pylist()
    indexed = {r['source_record_id']: r for r in rows}
    projection_errors = []
    sequence_column = {'dasgupta_s1.xlsx': 10, 'gelman_s1.xlsx': 9,
                       'dasgupta_s2.xlsx': 4, 'gelman_s2.xlsx': 5}
    for row in rows:
        values = json.loads(row['source_row_json'])
        original = values[sequence_column[row['source_file']]]
        expected = str(original).strip() or None if original is not None else None
        if expected != row['source_sequence_raw']:
            projection_errors.append(dict(source_file=row['source_file'], sheet=row['sheet_name'],
                                          row=row['source_row_index'], expected=expected,
                                          actual=row['source_sequence_raw'], disposition=row['source_row_disposition']))
    statuses, role_statuses = collections.Counter(), collections.Counter()
    n_undefined_identity_exact = n_wrapped_exact = n_nonpositive = 0
    for measurement in quantities:
        address = measurement['source_cell']
        col = 0
        for letter in re.match('[A-Z]+', address).group():
            col = col * 26 + ord(letter) - 64
        raw = json.loads(indexed[measurement['source_record_id']]['source_row_json'])[col - 1]
        raw_text = '' if raw is None else str(raw).strip()
        if isinstance(raw, (int, float)):
            expected = ('reported_exact', float(raw), None)
        elif raw_text in {'<0.20', '<0.2', '>5'}:
            expected = ('reported_bound', float(raw_text[1:]), 'lt' if raw_text[0] == '<' else 'gt')
        elif raw_text == '':
            expected = ('missing_source_value', None, None)
        elif raw_text == 'nd':
            expected = ('reported_not_detected', None, None)
        elif raw_text == 'po':
            expected = ('reported_overlap', None, None)
        elif raw_text == '#VALUE!':
            expected = ('reported_nonnumeric', None, None)
        else:
            raise AssertionError(('unregistered source syntax', address, raw))
        assert expected == (measurement['value_status'], measurement['quantitative_value'], measurement['measurement_inequality'])
        if expected[1] is not None:
            assert math.isfinite(expected[1])
            n_nonpositive += expected[1] <= 0
        statuses[expected[0]] += 1
        role_statuses[(measurement['ratio_role'], expected[0])] += 1
        sequence = measurement['peptide_sequence_raw']
        if expected[0] == 'reported_exact':
            n_undefined_identity_exact += sequence is None
            n_wrapped_exact += bool(sequence and sequence.startswith('(') and sequence.endswith(')'))
    pairs = []
    with gzip.open(FAMILY / 'results/cross-paper-candidates.jsonl.gz', 'rt') as stream:
        for line in stream:
            group = json.loads(line)
            labels = {' '.join(r['experiment'].lower().split()) for r in group['rows']}
            pairs.append(len(labels) == 1)
    result = dict(
        n_quantitative_parses_verified=len(quantities), value_status=dict(statuses),
        by_role_and_status=[dict(role=k[0], status=k[1], n_slots=v) for k, v in sorted(role_statuses.items())],
        n_nonpositive_reported_numeric_values=n_nonpositive,
        n_exact_slots_with_no_sequence=n_undefined_identity_exact,
        n_exact_slots_with_outer_parentheses=n_wrapped_exact,
        n_sequence_projection_mismatches=len(projection_errors),
        sequence_projection_mismatches=projection_errors,
        n_candidate_reuse_groups_with_equal_normalized_experiment_label=sum(pairs),
        n_candidate_reuse_groups_with_different_experiment_labels=len(pairs)-sum(pairs),
        seconds=time.perf_counter()-tick,
    )
    out = FAMILY / 'results/semantic-verification.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'sequence_projection_mismatches', 'by_role_and_status'}}, indent=2))


if __name__ == '__main__':
    main()
