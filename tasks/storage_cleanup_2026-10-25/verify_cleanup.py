"""Independently check retained bytes, skipped originals and worktree integrity."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
inventory = json.loads((ROOT / 'inventory.json').read_text())
original = [json.loads(line) for line in (ROOT / 'cleanup-journal.jsonl').read_text().splitlines()]
recovered = [
    json.loads(line) for line in (ROOT / 'recovery-journal.jsonl').read_text().splitlines()
]
records = original + recovered


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


verified_sources = set()
clones = [r for r in records if r['action'] == 'share_duplicate_extents']
skipped = [r for r in records if r['action'] == 'skip_platform_metadata_change']
assert len({r['destination'] for r in clones + skipped}) == 553
for row in clones + skipped:
    destination, source = Path(row['destination']), Path(row['source'])
    assert digest(destination) == row['sha256']
    if str(source) not in verified_sources:
        assert digest(source) == row['sha256']
        verified_sources.add(str(source))
    info = destination.stat()
    assert info.st_nlink == 1 and info.st_ino != source.stat().st_ino
    assert info.st_size == row['size_bytes'] and info.st_mtime_ns == row['mtime_ns']
    assert (info.st_mode, info.st_uid, info.st_gid, info.st_flags) == (
        row['mode'],
        row['uid'],
        row['gid'],
        row['flags'],
    )
    attributes = subprocess.check_output(['/usr/bin/xattr', '-lx', str(destination)])
    assert hashlib.sha256(attributes).hexdigest() == row['xattrs_sha256']
    expected_inode = (
        row['new_inode'] if row['action'] == 'share_duplicate_extents' else row['inode']
    )
    assert info.st_ino == expected_inode
    assert not list(destination.parent.glob(destination.name + '.presto-clone-*'))

fragments = json.loads((ROOT / 'fragment-plan.json').read_text())
for archive in fragments['archives']:
    assert digest(archive['path']) == archive['sha256']
for piece in fragments['pieces']:
    assert not Path(piece['path']).exists()
    with Path(piece['archive']).open('rb') as stream:
        stream.seek(piece['offset'])
        value = hashlib.sha256(stream.read(piece['size_bytes'])).hexdigest()
    assert value == piece['sha256']

regenerated_caches = []
cache_rows = [r for r in records if r['action'] == 'remove_regenerable_cache']
for row in cache_rows:
    path = Path(row['path'])
    if path.exists():
        assert path.stat().st_mtime_ns > row['mtime_ns']
        regenerated_caches.append(str(path))

for directory, before in inventory['worktree_state'].items():

    def git(*args, root=directory):
        return subprocess.check_output(['git', *args], cwd=root)

    assert git('rev-parse', 'HEAD').decode().strip() == before['head']
    assert git('status', '--porcelain', '--untracked-files=no').decode() == before['tracked_status']
    assert hashlib.sha256(git('diff', '--binary')).hexdigest() == before['working_diff_sha256']
    assert (
        hashlib.sha256(git('diff', '--cached', '--binary')).hexdigest()
        == before['staged_diff_sha256']
    )

new_family = ROOT.parents[1] / 'experiments/2026-10-25_0143_codex_intact-temporal-representation'
launch = json.loads((new_family / 'reproduce/launch.json').read_text())
for name, expected in launch['inputs'].items():
    assert digest(name) == expected
assert not list(new_family.rglob('*-started.json'))
free = shutil.disk_usage(ROOT).free
result = {
    'passed': True,
    'cloned_files': len(clones),
    'unchanged_metadata_skips': len(skipped),
    'clone_logical_bytes': sum(r['size_bytes'] for r in clones),
    'removed_cache_files': len(cache_rows),
    'cache_allocated_bytes': sum(r['allocated_bytes'] for r in cache_rows),
    'regenerated_caches': regenerated_caches,
    'removed_fragment_files': len(fragments['pieces']),
    'removed_fragment_bytes': sum(r['size_bytes'] for r in fragments['pieces']),
    'retained_complete_archives_verified': len(fragments['archives']),
    'all_removed_fragment_bytes_reconstructible': True,
    'unique_scientific_files_deleted': 0,
    'worktrees_with_unchanged_head_status_working_diff_and_index': len(inventory['worktree_state']),
    'experiment_input_hashes_verified': len(launch['inputs']),
    'new_experiments_started': False,
    'inventory_free_bytes': inventory['free_bytes'],
    'first_action_free_bytes': original[0]['observed_free_bytes'],
    'final_free_bytes': free,
    'observed_change_since_inventory': free - inventory['free_bytes'],
    'capacity_caveat': (
        'Other jobs use the same volume; free-space delta is not solely attributable.'
    ),
}
(ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
