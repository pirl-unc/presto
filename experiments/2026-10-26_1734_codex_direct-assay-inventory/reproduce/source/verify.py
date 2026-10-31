"""Independent Arrow vector/group-by replay of the literal census."""

from collections import Counter
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import BASE, EXTRA, FAMILY, FIELDS, SOURCES, guarded, sha, write


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if (FAMILY / 'results/independent-verification.json').exists():
        raise FileExistsError('preserve prior verification')
    census = json.loads((FAMILY / 'results/census.json').read_text())
    with guarded('independent-replay'):
        checked = 0
        for name, pin in SOURCES.items():
            assert sha(BASE / name) == pin['sha256']
            source = pq.ParquetFile(BASE / name)
            counts = Counter()
            missing = {key: Counter() for key in FIELDS + EXTRA}
            ids = {key: set() for key in ('assay_iri', 'reference_iri')}
            total = 0
            for batch in source.iter_batches(
                batch_size=2048, columns=FIELDS + EXTRA, use_threads=False
            ):
                table = pa.Table.from_batches([batch])
                v = table['quantitative_value']
                state = pc.if_else(
                    pc.is_null(v),
                    'missing',
                    pc.if_else(
                        pc.invert(pc.is_finite(v)),
                        'nonfinite',
                        pc.if_else(pc.less_equal(v, 0), 'nonpositive', 'positive_finite'),
                    ),
                )
                keys = table.select(FIELDS).append_column('numeric_state', state)
                keys = keys.append_column('ones', pa.array([1] * len(table), type=pa.int32()))
                grouped = keys.group_by(FIELDS + ['numeric_state'], use_threads=False).aggregate(
                    [('ones', 'sum')]
                )
                for row in grouped.to_pylist():
                    counts[tuple(row[k] for k in FIELDS + ['numeric_state'])] += row['ones_sum']
                for key in missing:
                    column = table[key]
                    nulls = column.null_count
                    empty = (
                        int(pc.sum(pc.fill_null(pc.equal(column, ''), False)).as_py())
                        if pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
                        else 0
                    )
                    for state, n in [
                        ('null', nulls),
                        ('empty_string', empty),
                        ('present', len(table) - nulls - empty),
                    ]:
                        if n:
                            missing[key][state] += n
                for key in ids:
                    ids[key].update(
                        x for x in pc.unique(table[key]).to_pylist() if x not in (None, '')
                    )
                total += len(table)
            expected = {
                tuple(r[k] for k in FIELDS + ['numeric_state']): r['rows']
                for r in census['joint_counts']
                if r['file'] == name
            }
            assert counts == expected
            saved = census['sources'][name]
            assert (
                total
                == pin['rows']
                == source.metadata.num_rows
                == saved['rows']
                == sum(counts.values())
            )
            assert {k: dict(v) for k, v in missing.items()} == saved['null_and_empty_counts']
            assert {k: len(v) for k, v in ids.items()} == saved['distinct_nonempty_ids']
            checked += total
        result = {
            'status': 'passed',
            'source_rows_verified': checked,
            'joint_cells_verified': len(census['joint_counts']),
            'method': (
                'independent Arrow vector predicates/group-by; batch2048 vs row-wise4096; '
                'all missingness, ID cardinalities and source hashes replayed'
            ),
            'census_sha256': sha(FAMILY / 'results/census.json'),
            'admitted_rows': 0,
            'model_evaluation': None,
        }
        write(FAMILY / 'results/independent-verification.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
