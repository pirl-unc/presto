"""Acquire public source archives through datacache; do not admit training labels."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
from urllib.parse import urlparse

import datacache

F = Path(__file__).resolve().parents[1]
ROOT = Path('/Users/iskander/code/presto/artifacts') / F.name / 'sources'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def acquire(spec):
    started = time.monotonic()
    directory = ROOT / spec['id']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / spec.get('filename', Path(urlparse(spec['url']).path).name)
    try:
        datacache.fetch_file(
            spec['url'],
            destination=str(path),
            raw=True,
            record_provenance=True,
            timeout=(30, 90),
            max_retries=2,
        )
        with path.open('rb') as stream:
            prefix = stream.read(64).lstrip().lower()
        if prefix.startswith((b'<!doctype html', b'<html')):
            raise ValueError('Source returned HTML instead of a data archive')
        if 'expected_bytes' in spec and path.stat().st_size != spec['expected_bytes']:
            raise ValueError('Downloaded size differs from repository manifest')
        result = dict(
            spec,
            status='downloaded',
            path=str(path),
            bytes=path.stat().st_size,
            sha256=sha(path),
            seconds=time.monotonic() - started,
        )
    except Exception as exc:
        result = dict(
            spec,
            status='unavailable_this_attempt',
            error=repr(exc),
            seconds=time.monotonic() - started,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default='acquisition')
    args = parser.parse_args()
    if not args.batch.replace('-', '').isalnum():
        parser.error('batch must contain letters, numbers and hyphens')
    manifest = F / f'reproduce/{args.batch}-manifest.json'
    result_path = F / f'results/{args.batch}.json'
    assert not result_path.exists(), 'Preserve prior acquisition receipts'
    launch = dict(
        argv=['python', __file__, '--batch', args.batch],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_dirty=subprocess.check_output(['git', 'status', '--short'], text=True),
        source_sha256=sha(__file__),
        manifest_sha256=sha(manifest),
        datacache_version=importlib.metadata.version('datacache'),
        requested_gpu=None,
        started_unix=time.time(),
    )
    (F / f'reproduce/{args.batch}-launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    shutil.copyfile(__file__, F / f'reproduce/source/acquire-{args.batch}.py')
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(acquire, s) for s in json.loads(manifest.read_text())['sources']]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            result_path.write_text(json.dumps(results, indent=2) + '\n')
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k in ('id', 'status', 'bytes', 'error')}
                ),
                flush=True,
            )


if __name__ == '__main__':
    main()
