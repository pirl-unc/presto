"""Resume only unfinished clone sharing, skipping platform metadata changes."""

import ctypes
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from apply_cleanup import scoped, unchanged
from audit import OUT, WORKTREES, sha, state
from plan import attrs, open_files

plan = json.loads((OUT / 'cleanup-plan.json').read_text())
old = [json.loads(line) for line in (OUT / 'cleanup-journal.jsonl').read_text().splitlines()]
done = {r['destination']: r for r in old if r['action'] == 'share_duplicate_extents'}
assert len(done) == 3
for name, record in done.items():
    path = Path(name)
    assert sha(path) == record['sha256'] and path.stat().st_ino == record['new_inode']
baseline = {str(root): state(root) for root in WORKTREES}
opened = open_files()
before = shutil.disk_usage(OUT).free
clonefile = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True).clonefile
clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
clonefile.restype = ctypes.c_int
records = []
journal = (OUT / 'recovery-journal.jsonl').open('x')


def record(value):
    value['observed_free_bytes'] = shutil.disk_usage(OUT).free
    records.append(value)
    journal.write(json.dumps(value) + '\n')
    journal.flush()


for index, entry in enumerate(plan['clone_operations']):
    if entry['destination'] in done:
        continue
    source, destination = Path(entry['source']), Path(entry['destination'])
    scoped(source)
    assert str(source) not in opened and str(destination) not in opened
    original = unchanged(destination, entry)
    source_info = source.stat()
    assert source_info.st_dev == original.st_dev and source_info.st_nlink == 1
    assert sha(source) == entry['sha256']
    destination_attrs = attrs(destination)
    assert attrs(source) == destination_attrs
    assert hashlib.sha256(destination_attrs).hexdigest() == entry['xattrs_sha256']
    assert (source_info.st_mode, source_info.st_uid, source_info.st_gid, source_info.st_flags) == (
        original.st_mode,
        original.st_uid,
        original.st_gid,
        original.st_flags,
    )
    for temporary in destination.parent.glob(destination.name + '.presto-clone-*'):
        scoped(temporary)
        assert sha(temporary) == entry['sha256'] and temporary.stat().st_nlink == 1
        record(
            {
                'action': 'discard_preserved_failed_temporary',
                'path': str(temporary),
                'sha256': entry['sha256'],
                'xattrs_sha256': hashlib.sha256(attrs(temporary)).hexdigest(),
                'original_destination_unchanged': True,
            }
        )
        temporary.unlink()
    temporary = destination.with_name(destination.name + f'.presto-clone-{os.getpid()}')
    assert not temporary.exists()
    result = clonefile(os.fsencode(source), os.fsencode(temporary), 0)
    if result:
        raise OSError(ctypes.get_errno(), 'APFS clone failed; no full-copy fallback')
    assert sha(temporary) == entry['sha256']
    temporary_attrs = attrs(temporary)
    if temporary_attrs != destination_attrs:
        record(
            {
                'action': 'skip_platform_metadata_change',
                **entry,
                'temporary_xattrs_sha256': hashlib.sha256(temporary_attrs).hexdigest(),
                'original_destination_unchanged': True,
            }
        )
        temporary.unlink()
        continue
    temporary_info = temporary.stat()
    assert temporary_info.st_ino != source_info.st_ino
    assert (
        temporary_info.st_mode,
        temporary_info.st_uid,
        temporary_info.st_gid,
        temporary_info.st_flags,
    ) == (original.st_mode, original.st_uid, original.st_gid, original.st_flags)
    unchanged(destination, entry)
    os.utime(temporary, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(temporary, destination)
    assert sha(source) == sha(destination) == entry['sha256']
    os.utime(destination, ns=(original.st_atime_ns, original.st_mtime_ns))
    final = destination.stat()
    assert final.st_ino != source.stat().st_ino and final.st_nlink == 1
    assert final.st_mtime_ns == original.st_mtime_ns
    record(
        {
            'action': 'share_duplicate_extents',
            **entry,
            'new_inode': final.st_ino,
            'independent_inode': True,
        }
    )
    if (index + 1) % 100 == 0:
        print(
            json.dumps(
                {'processed_manifest_entries': index + 1, 'free_bytes': shutil.disk_usage(OUT).free}
            ),
            flush=True,
        )

after_state = {str(root): state(root) for root in WORKTREES}
assert baseline == after_state, 'Worktree state changed; inspect concurrent edits'
journal.close()
result = {
    'passed': True,
    'finished_epoch': time.time(),
    'before_free_bytes': before,
    'after_free_bytes': shutil.disk_usage(OUT).free,
    'additional_clones': sum(r['action'] == 'share_duplicate_extents' for r in records),
    'metadata_skips': sum(r['action'] == 'skip_platform_metadata_change' for r in records),
    'worktree_state_unchanged': True,
    'baseline_worktree_state': baseline,
    'recovery_journal_sha256': sha(OUT / 'recovery-journal.jsonl'),
}
with (OUT / 'recovery-result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps({k: v for k, v in result.items() if k != 'baseline_worktree_state'}), flush=True)
