"""Optionally recreate removed download pieces from verified retained archives."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED = Path('/private/tmp/presto-mapp-controlled-assets')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--apply', action='store_true', help='Write missing pieces; default only reports.'
)
args = parser.parse_args()
plan = json.loads((ROOT / 'fragment-plan.json').read_text())
missing = [r for r in plan['pieces'] if not Path(r['path']).exists()]
size = sum(r['size_bytes'] for r in missing)
if not args.apply:
    print(json.dumps({'missing_pieces': len(missing), 'bytes_to_recreate': size, 'writes': False}))
    raise SystemExit(0)
assert shutil.disk_usage(ALLOWED).free >= size + 1024**3
for archive in plan['archives']:
    path = Path(archive['path'])
    assert ALLOWED in path.parents and path.resolve() == path
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    assert digest.hexdigest() == archive['sha256']
for piece in missing:
    path = Path(piece['path'])
    assert ALLOWED in path.parents and path.resolve() == path
    assert path.suffix in {'.part', '.partial'}
    with Path(piece['archive']).open('rb') as source:
        source.seek(piece['offset'])
        payload = source.read(piece['size_bytes'])
    assert hashlib.sha256(payload).hexdigest() == piece['sha256']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as destination:
        destination.write(payload)
print(json.dumps({'restored_pieces': len(missing), 'restored_bytes': size}))
