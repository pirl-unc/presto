"""Collect an immutable CPU audit; independently verify metrics and threshold counts."""

from collections import Counter, defaultdict
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


def main():
    receipt = json.loads((F / 'results/receipt.json').read_text())
    storage = F.parents[1] / 'artifacts' / F.name
    storage.mkdir(parents=True, exist_ok=True)
    archive = storage / 'audit.tar.gz'
    if not archive.exists():
        temp = archive.with_suffix('.partial')
        with temp.open('wb') as stream:
            for chunk in modal.Volume.from_name('presto-checkpoints').read_file(receipt['archive']):
                stream.write(chunk)
        assert temp.stat().st_size == receipt['bytes'] and sha(temp) == receipt['sha256']
        temp.replace(archive)
    assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            assert parts and parts[0] == 'audit' and '..' not in parts
            assert not member.name.startswith('/') and (member.isfile() or member.isdir())
        tar.extractall(storage, filter='data')
    d = storage / 'audit'
    review = F / 'results/audit'
    review.mkdir(exist_ok=True)
    # Preserve the complete row dump outside git; small review artifacts are canonical.
    for path in d.iterdir():
        if path.name != 'concentration-records.jsonl.gz':
            shutil.copyfile(path, review / path.name)
    summary = json.loads((d / 'summary.json').read_text())
    with gzip.open(d / 'concentration-records.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == summary['counts']['concentration_rows'] == 184637
    assert Counter(r['qualifier'] for r in records) == {
        k.split(':', 1)[1]: v for k, v in summary['counts'].items() if k.startswith('qualifier:')
    }
    for cell in summary['threshold_values']:
        actual = sum(
            r['mhc_class'] == cell['mhc_class']
            and r['endpoint'] == cell['endpoint']
            and r['method'] == cell['method']
            and r['value_nM'] == cell['value_nM']
            and r['qualifier'] == cell['qualifier']
            and (r['raw_qualifier'] or 'missing') == cell['raw_qualifier']
            for r in records
        )
        assert actual == cell['n']
    pairs = defaultdict(list)
    with gzip.open(d / 'method-pairs.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            pairs[
                row['mhc_class'], row['response_stratum'], row['method_a'], row['method_b']
            ].append(row)
    for expected in summary['method_correlations']:
        rows = pairs[
            expected['mhc_class'],
            expected['response_stratum'],
            expected['method_a'],
            expected['method_b'],
        ]
        assert len(rows) == expected['n'] == len({r['pmhc_sha256'] for r in rows})
        x = np.array([r['a_log10_nM'] for r in rows])
        y = np.array([r['b_log10_nM'] for r in rows])
        if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
            assert np.isclose(
                np.corrcoef(rankdata(x), rankdata(y))[0, 1], expected['spearman'], atol=1e-12
            )
            assert np.isclose(np.corrcoef(x, y)[0, 1], expected['pearson'], atol=1e-12)
        else:
            assert expected['spearman'] is None and expected['pearson'] is None
    census = json.loads((d / 'source-census.json').read_text())
    assert sum(r['n'] for r in census) == 821470 + 26694
    result = dict(
        status='verified',
        concentration_records=len(records),
        method_comparison_cells=len(summary['method_correlations']),
        threshold_cells=len(summary['threshold_values']),
        source_census_rows=sum(r['n'] for r in census),
        archive=dict(path=str(archive.resolve()), sha256=sha(archive)),
        artifacts={p.name: dict(path=str(p.resolve()), sha256=sha(p)) for p in d.iterdir()},
        validation_test_used=False,
        changed_qualifiers=0,
    )
    (F / 'results/verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        json.dumps({k: v for k, v in result.items() if k not in ('artifacts', 'archive')}, indent=2)
    )


if __name__ == '__main__':
    main()
