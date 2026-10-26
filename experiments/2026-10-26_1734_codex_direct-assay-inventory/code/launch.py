"""Count literal metadata with Python row-wise aggregation; do not admit or fit."""

from collections import Counter
import csv
import json
import math
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from common import BASE, EXTRA, FAMILY, FIELDS, SOURCES, guarded, sha, write


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if (FAMILY / 'results/census.json').exists():
        raise FileExistsError('preserve the existing census')
    with guarded('inventory') as receipt:
        for p in (FAMILY / 'code').glob('*.py'):
            shutil.copyfile(p, FAMILY / 'reproduce/source' / p.name)
        write(FAMILY / 'reproduce/launch.json', receipt)
        summaries, joint = {}, []
        for name, pin in SOURCES.items():
            path = BASE / name
            if sha(path) != pin['sha256']:
                raise ValueError('source hash differs: ' + name)
            source = pq.ParquetFile(path)
            assert source.metadata.num_rows == pin['rows']
            counts = Counter()
            missing = {key: Counter() for key in FIELDS + EXTRA}
            ids = {key: set() for key in ('assay_iri', 'reference_iri')}
            total = 0
            for batch in source.iter_batches(
                batch_size=4096, columns=FIELDS + EXTRA, use_threads=False
            ):
                for row in batch.to_pylist():
                    value = row['quantitative_value']
                    numeric = (
                        'missing'
                        if value is None
                        else 'nonfinite'
                        if not math.isfinite(value)
                        else 'nonpositive'
                        if value <= 0
                        else 'positive_finite'
                    )
                    counts[tuple(row[k] for k in FIELDS) + (numeric,)] += 1
                    for key in missing:
                        missing[key][
                            'null'
                            if row[key] is None
                            else 'empty_string'
                            if row[key] == ''
                            else 'present'
                        ] += 1
                    for key in ids:
                        if row[key] not in (None, ''):
                            assert isinstance(row[key], str)
                            ids[key].add(row[key])
                    total += 1
            assert total == pin['rows'] == sum(counts.values())
            summaries[name] = {
                'rows': total,
                'joint_cells': len(counts),
                'null_and_empty_counts': {k: dict(v) for k, v in missing.items()},
                'distinct_nonempty_ids': {k: len(v) for k, v in ids.items()},
                'sha256': pin['sha256'],
                'schema': str(source.schema_arrow),
            }
            joint.extend(
                {
                    'file': name,
                    **dict(zip(FIELDS + ['numeric_state'], key, strict=True)),
                    'rows': count,
                }
                for key, count in sorted(counts.items(), key=lambda kv: json.dumps(kv[0]))
            )
        write(
            FAMILY / 'results/census.json',
            {
                'status': 'completed_literal_inventory_not_admission',
                'sources': summaries,
                'joint_counts': joint,
            },
        )
        with (FAMILY / 'results/joint-counts.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['file', *FIELDS, 'numeric_state', 'rows'])
            writer.writeheader()
            for row in joint:
                writer.writerow({k: json.dumps(v) if k in FIELDS else v for k, v in row.items()})
        print(
            json.dumps(
                {
                    'rows': sum(s['rows'] for s in summaries.values()),
                    'joint_cells': len(joint),
                    'admitted_rows': 0,
                }
            ),
            flush=True,
        )


if __name__ == '__main__':
    main()
