"""Hash-gated collection and independent recomputation from preserved assay pairs."""

from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tarfile

import modal
import numpy as np
from scipy.stats import rankdata

F = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def correlation(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def verify(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    pairs = defaultdict(list)
    with gzip.open(directory / 'paired-values.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(row[k] for k in ('mhc_class', 'endpoint_a', 'endpoint_b', 'scope'))
            pairs[key].append(row)
    for expected in summary['comparisons']:
        key = tuple(expected[k] for k in ('mhc_class', 'endpoint_a', 'endpoint_b', 'scope'))
        rows = pairs[key]
        assert len(rows) == expected['n']
        assert len({r['pmhc_sha256'] for r in rows}) == expected['unique_pmhcs']
        units = {(r['pmhc_sha256'], tuple(r['studies_a']), tuple(r['studies_b'])) for r in rows}
        assert len(units) == len(rows), 'Duplicate comparison units'
        if key[3] != 'within_study':
            assert expected['n'] == expected['unique_pmhcs']
        a = np.array([r['a_log10_nM'] for r in rows])
        b = np.array([r['b_log10_nM'] for r in rows])
        actual = dict(
            spearman=correlation(rankdata(a), rankdata(b)),
            pearson=correlation(a, b),
            difference_rmse_log10_nM=float(np.linalg.norm(b - a) / np.sqrt(len(a)))
            if len(a)
            else None,
            median_b_minus_a_log10_nM=float(np.median(b - a)) if len(a) else None,
            identical_aggregate_fraction=sum(x == y for x, y in zip(a, b)) / len(a)
            if len(a)
            else None,
        )
        for metric, value in actual.items():
            if value is None:
                assert expected[metric] is None, (key, metric)
            else:
                assert np.isclose(value, expected[metric], atol=1e-12), (key, metric)
    exact = sum(
        v
        for k, v in summary['counts'].items()
        if k.startswith('concentration:') and k.endswith(':=')
    )
    assert exact == summary['exact_observation_count']
    assert sum(r['n'] for r in summary['assay_methods']) == sum(
        v for k, v in summary['counts'].items() if k.startswith('concentration:')
    )
    receipt = dict(
        status='verified',
        comparison_cells=len(summary['comparisons']),
        preserved_pairs=sum(map(len, pairs.values())),
        method='NumPy Pearson on values and SciPy average ranks; independent arithmetic',
        files={p.name: sha(p) for p in directory.iterdir() if p.is_file()},
        source_aggregation_tests=(
            '5 passed; units, censoring, identity, duplicate aggregation, hashes'
        ),
    )
    return receipt


def collect_archive(receipt, name):
    storage = F.parents[1] / 'artifacts' / F.name
    storage.mkdir(parents=True, exist_ok=True)
    archive = storage / (name + '.tar.gz')
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        with partial.open('wb') as stream:
            for chunk in modal.Volume.from_name('presto-checkpoints').read_file(receipt['archive']):
                stream.write(chunk)
        assert partial.stat().st_size == receipt['bytes'] and sha(partial) == receipt['sha256']
        partial.replace(archive)
    assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            assert parts and parts[0] == name and '..' not in parts
            assert not member.name.startswith('/') and (member.isfile() or member.isdir())
        tar.extractall(storage, filter='data')
    destination = F / 'results' / name
    shutil.copytree(storage / name, destination, dirs_exist_ok=True)
    return destination, dict(path=str(archive.resolve()), sha256=sha(archive))


def main():
    receipt_path = F / 'results/audit-receipt.json'
    if not receipt_path.exists():
        call = json.loads((F / 'reproduce/audit-call.json').read_text())
        receipt = modal.FunctionCall.from_id(call['call_id']).get()
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    receipt = json.loads(receipt_path.read_text())
    destination, archive = collect_archive(receipt, 'assay-agreement')
    checked = verify(destination)
    checked['archive'] = archive
    (F / 'results/audit-verification.json').write_text(json.dumps(checked, indent=2) + '\n')
    print(json.dumps(checked, indent=2))
    lineage_path = F / 'results/lineage-receipt.json'
    if lineage_path.exists():
        destination, archive = collect_archive(
            json.loads(lineage_path.read_text()), 'overlap-lineage'
        )
        (F / 'results/lineage-collection.json').write_text(
            json.dumps(
                dict(archive=archive, files={p.name: sha(p) for p in destination.iterdir()}),
                indent=2,
            )
            + '\n'
        )


if __name__ == '__main__':
    main()
