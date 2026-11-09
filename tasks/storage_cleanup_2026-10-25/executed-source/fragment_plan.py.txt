"""Verify redundant download pieces against retained complete archives."""

import gzip
import json
from pathlib import Path

from audit import OUT, sha
from plan import open_files

ROOT = Path('/private/tmp/presto-mapp-controlled-assets')
opened = open_files()
archives, pieces, skipped = [], [], []
for accession in ('PXD007263', 'PXD009508'):
    archive = ROOT / accession / 'search.tar.gz'
    assert str(archive) not in opened and not archive.is_symlink()
    digest = sha(archive)
    uncompressed = 0
    with gzip.open(archive, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            uncompressed += len(block)
    archives.append(
        {
            'path': str(archive),
            'sha256': digest,
            'size_bytes': archive.stat().st_size,
            'gzip_uncompressed_bytes': uncompressed,
            'gzip_crc_passed': True,
        }
    )
    parts = sorted(archive.with_name(archive.name + '.ranges').glob('*.part'))
    offset = 0
    with archive.open('rb') as source:
        for index, path in enumerate(parts):
            assert path.name == f'{index:04d}.part'
            assert str(path) not in opened and not path.is_symlink()
            payload = path.read_bytes()
            assert payload == source.read(len(payload))
            info = path.stat()
            pieces.append(
                {
                    'path': str(path),
                    'archive': str(archive),
                    'archive_sha256': digest,
                    'offset': offset,
                    'size_bytes': len(payload),
                    'sha256': sha(path),
                    'allocated_bytes': info.st_blocks * 512,
                    'inode': info.st_ino,
                    'mtime_ns': info.st_mtime_ns,
                }
            )
            offset += len(payload)
        assert source.read(1) == b'' and offset == archive.stat().st_size
    partial = archive.with_name(archive.name + '.partial')
    if partial.exists():
        payload = partial.read_bytes()
        with archive.open('rb') as source:
            same = payload == source.read(len(payload))
        if same and str(partial) not in opened and not partial.is_symlink():
            info = partial.stat()
            pieces.append(
                {
                    'path': str(partial),
                    'archive': str(archive),
                    'archive_sha256': digest,
                    'offset': 0,
                    'size_bytes': len(payload),
                    'sha256': sha(partial),
                    'allocated_bytes': info.st_blocks * 512,
                    'inode': info.st_ino,
                    'mtime_ns': info.st_mtime_ns,
                }
            )
        else:
            skipped.append({'path': str(partial), 'reason': 'nonmatching_or_open'})
result = {
    'archives': archives,
    'pieces': pieces,
    'skipped': skipped,
    'redundant_bytes': sum(r['size_bytes'] for r in pieces),
    'restoration': (
        'Read size_bytes from retained archive at offset; verify piece sha256 '
        'before writing to its original path.'
    ),
}
with (OUT / 'fragment-plan.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(
    json.dumps(
        {
            'verified_archives': len(archives),
            'deletable_pieces': len(pieces),
            'redundant_bytes': result['redundant_bytes'],
            'skipped': skipped,
        }
    )
)
