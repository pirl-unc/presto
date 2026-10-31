"""Replay pinned Hitlist TMAB records against independently decoded source cells."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
CHECKOUT = Path('/private/tmp/hitlist-native-audit')
COMMIT = 'f4d3a6d0f1933162f9ac6eac42ac1b134c8a27dc'
FILES = {'dasgupta_s1.xlsx': 'dasgupta2014_s001', 'dasgupta_s2.xlsx': 'dasgupta2014_s002',
         'gelman_s1.xlsx': 'gelman2013_s004', 'gelman_s2.xlsx': 'gelman2013_s005'}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def column(index):
    result = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def text(value):
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip() or None


def main():
    tick = time.perf_counter()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip() == COMMIT
    assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=CHECKOUT, text=True) == ''
    sys.path.insert(0, str(CHECKOUT))
    import hitlist
    from hitlist import EvidenceStore, write_tmab_peptidomics_package
    import pyarrow.parquet as pq

    assert Path(hitlist.__file__).resolve() == CHECKOUT / 'hitlist/__init__.py'
    result_file = FAMILY / 'results/canonical-audit.json'
    assert not result_file.exists()
    config = json.loads((FAMILY / 'reproduce/launch.json').read_text())
    assets = {a['id']: a for a in json.loads((FAMILY / 'results/retrieval.json').read_text())['assets']}
    root = Path(config['asset_directory'])
    inputs = root / 'canonical-inputs'
    inputs.mkdir(exist_ok=False)
    for filename, asset_id in FILES.items():
        shutil.copyfile(assets[asset_id]['path'], inputs / filename)
    # Shared public API alone creates the source package; no local source repair.
    package = root / 'hitlist-f4d3a6d-package'
    manifest_path = write_tmab_peptidomics_package(inputs, package)
    store = EvidenceStore.open(package)
    query = store.query('tmab_measurement', select=['measurement_id', 'source_cell', 'quantitative_raw'], limit=None)
    rows = pq.read_table(package / 'tmab/source_rows.parquet').to_pylist()
    measurements = pq.read_table(package / 'tmab/measurements.parquet').to_pylist()
    contexts = pq.read_table(package / 'tmab/protocol_context.parquet').to_pylist()
    lineage = pq.read_table(package / 'tmab/lineage.parquet').to_pylist()
    raw = {}
    for filename, asset_id in FILES.items():
        raw[filename] = collections.defaultdict(dict)
        with gzip.open(FAMILY / 'results' / (asset_id + '-cells.jsonl.gz'), 'rt') as stream:
            for line in stream:
                cell = json.loads(line)
                raw[filename][cell['sheet']][cell['cell']] = cell
    row_map = {}
    n_values = 0
    for row in rows:
        row_map[row['source_record_id']] = row
        values = json.loads(row['source_row_json'])
        source = raw[row['source_file']][row['sheet_name']]
        for index, value in enumerate(values, 1):
            address = column(index) + str(row['source_row_index'])
            expected = source.get(address, {}).get('value')
            assert value == expected, (row['source_row_locator'], address, expected, value)
            n_values += 1
    for record in measurements:
        row = row_map[record['source_record_id']]
        cell = raw[row['source_file']][row['sheet_name']].get(record['source_cell'], {})
        value = cell.get('value')
        # Raw text is based on openpyxl's int/float representation. Source row
        # JSON preserves that representation; compare it and the original value.
        values = json.loads(row['source_row_json'])
        col = ''.join(c for c in record['source_cell'] if c.isalpha())
        index = 0
        for letter in col:
            index = index * 26 + ord(letter) - 64
        source_value = values[index - 1]
        assert value == source_value
        expected_text = None if source_value is None else str(source_value).strip() or None
        assert record['quantitative_raw'] == expected_text
    context_by_id = {c['context_id']: c for c in contexts}
    assert all(m['context_id'] in context_by_id for m in measurements)
    assert len({m['measurement_id'] for m in measurements}) == len(measurements)
    assert {m['measurement_id'] for m in measurements} == {x['measurement_id'] for x in lineage}
    # Use original identity/value assertions, with only a source-independent
    # comparison key. This is a possible-reuse audit, never a deduplication rule.
    with gzip.open(FAMILY / 'results/primary-row-audit.jsonl.gz', 'rt') as stream:
        primary = [json.loads(line) for line in stream]
    by_key = collections.defaultdict(list)
    for row in primary:
        seq = row['sequence_raw']
        if not isinstance(seq, str) or not seq.strip():
            continue
        if sum(isinstance(x, (int, float)) for x in row['ratios'][:5]) < 2:
            continue
        key = json.dumps([row['cell_line'], row['charge'], row['observed_mass'], seq.strip(), row['ratios'][:5]], sort_keys=True)
        by_key[key].append(row)
    candidates = []
    for key, group in by_key.items():
        if len({r['asset'] for r in group}) > 1:
            candidates.append(dict(key_sha256=hashlib.sha256(key.encode()).hexdigest(), rows=group,
                                   interpretation='possible reused source rows; not proven independent or identical specimens'))
    with gzip.open(FAMILY / 'results/cross-paper-candidates.jsonl.gz', 'wt') as stream:
        for group in candidates:
            stream.write(json.dumps(group, ensure_ascii=False) + '\n')
    tentative = [m for m in measurements if '(' in (m['peptide_sequence_raw'] or '') or ')' in (m['peptide_sequence_raw'] or '')]
    oxidized = [m for m in measurements if 'Mox' in (m['peptide_sequence_raw'] or '')]
    disagreements = [r for r in primary if 'wash' in str(r['average_annotation']).lower()]
    semantic = dict(
        n_tentative_sequence_measurement_slots=len(tentative),
        n_tentative_candidate_exact_slots=sum(m['eligibility_status'] == 'candidate_exact' for m in tentative),
        tentative_examples=tentative[:6],
        n_mox_measurement_slots=len(oxidized), mox_examples=oxidized[:6],
        n_possible_cross_paper_fingerprint_groups=len(candidates),
        n_possible_cross_paper_rows=sum(len(c['rows']) for c in candidates),
        candidate_examples=candidates[:3],
        exported_lineage_status=dict(collections.Counter(x['lineage_status'] for x in lineage)),
        n_rows_with_wash_annotation=len(disagreements), wash_examples=disagreements[:5],
        context_assertions=contexts,
        note='These are audit counts, not biological independence or training admission counts.',
    )
    dump(FAMILY / 'results/semantic-findings.json', semantic)
    manifest = json.loads(manifest_path.read_text())
    dump(FAMILY / 'results/canonical-package-manifest.json', manifest)
    summary = dict(
        commit=COMMIT, hitlist_version=hitlist.__version__, package_path=str(package),
        package_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        n_source_rows=len(rows), n_source_row_values_replayed=n_values,
        n_ratio_slots_replayed=len(measurements), n_query_rows=query.table.num_rows,
        coverage=manifest['reconciliation'],
        n_cross_paper_candidate_groups=len(candidates),
        n_tentative_candidate_exact_slots=semantic['n_tentative_candidate_exact_slots'],
        seconds=time.perf_counter()-tick, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_value_replay_pass=True, training_admission=False,
        training_admission_reason='Lineage, tentative identification, modification and context semantics need shared correction/adjudication.',
    )
    dump(result_file, summary)
    assert summary['n_query_rows'] == len(measurements)
    assert summary['seconds'] < 180 and summary['peak_rss_bytes'] < 2 * 1024**3
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
