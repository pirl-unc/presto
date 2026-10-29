"""Preserve closed materialization files from the active worker, without altering them."""

from pathlib import Path
import hashlib
import json
import shutil
import time
import modal

FID = '2026-10-28_1540_codex_temporal-multi-output'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    source = Path('/runs') / FID / 'prepared'
    durable = Path('/volume') / FID
    target = durable / 'preparation-attempt4-closed-materialization'
    if (target / 'backup-receipt.json').exists():
        print('Existing closed-materialization backup retained', flush=True)
        return
    deadline = time.monotonic() + 5400
    while not (source / 'wide-multi-s43.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Materialization completion marker did not appear')
        time.sleep(5)
    # Both manifests are written only after every observation/membership stream closes.
    manifest = json.loads((source / 'wide-multi-s43.json').read_text())
    paths = {Path(spec['path']) for spec in manifest['sources']}
    expected = {Path(spec['path']): spec['sha256'] for spec in manifest['sources']}
    for name in (
        'wide-multi-s42.json',
        'wide-multi-s43.json',
        'study-selection.json',
        'endpoint-support.json',
        'human-ensembl112.pep.fa',
        'membership.jsonl.gz',
        'partition-dispositions.jsonl.gz',
    ):
        paths.add(source / name)
    temporary = target.with_name(target.name + '-partial-' + str(time.time_ns()))
    temporary.mkdir()
    files = {}
    for path in sorted(paths):
        assert path.is_relative_to(source)
        destination = temporary / path.relative_to(source)
        shutil.copyfile(path, destination)
        digest = sha(destination)
        assert digest == expected.get(path, sha(path)), path
        files[path.name] = dict(sha256=digest, bytes=destination.stat().st_size)
    receipt = dict(
        status='closed_materialization_backed_up',
        files=files,
        created_unix=time.time(),
        scope='Source rows and frozen membership only; canonical preflight and independent '
        'audit must still pass before any fit. No model predictions.',
    )
    (temporary / 'backup-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert not target.exists()
    temporary.rename(target)
    modal.Volume.from_name('presto-checkpoints').commit()
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
