"""Apply the frozen cleanup manifest, retaining all unique scientific bytes."""

import ctypes
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path

from audit import MAIN, OUT, WORKTREES, sha, state
from plan import attrs, open_files


def scoped(path):
    assert not path.is_symlink() and path.resolve() == path
    assert MAIN in path.parents or (
        path.parts[:3] == ('/', 'private', 'tmp') and path.parts[3].startswith('presto')
    )
    assert '.git' not in path.parts and '.venv' not in path.parts


def unchanged(path, record):
    scoped(path)
    info = path.stat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    assert (info.st_ino, info.st_size, info.st_mtime_ns) == (
        record['inode'],
        record['size_bytes'],
        record['mtime_ns'],
    )
    assert sha(path) == record['sha256']
    return info


if __name__ == '__main__':
    plan = json.loads((OUT / 'cleanup-plan.json').read_text())
    fragments = json.loads((OUT / 'fragment-plan.json').read_text())
    assert sha(OUT / 'inventory.json') == plan['inventory_sha256']
    baseline = {str(root): state(root) for root in WORKTREES}
    opened = open_files()
    before = shutil.disk_usage(OUT).free
    clonefile = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True).clonefile
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clonefile.restype = ctypes.c_int
    done = []
    journal = (OUT / 'cleanup-journal.jsonl').open('x')

    def record(value):
        value['observed_free_bytes'] = shutil.disk_usage(OUT).free
        done.append(value)
        journal.write(json.dumps(value) + '\n')
        journal.flush()

    for entry in plan['cache_deletions']:
        path = Path(entry['path'])
        assert any(part in {'__pycache__', '.pytest_cache', '.ruff_cache'} for part in path.parts)
        if str(path) in opened or not path.exists():
            record({'action': 'skip_cache', 'path': str(path), 'reason': 'open_or_missing'})
            continue
        try:
            unchanged(path, entry)
        except AssertionError:
            record({'action': 'skip_cache', 'path': str(path), 'reason': 'changed_since_plan'})
            continue
        path.unlink()
        record({'action': 'remove_regenerable_cache', **entry})

    for archive in fragments['archives']:
        path = Path(archive['path'])
        scoped(path)
        assert str(path) not in opened and sha(path) == archive['sha256']
    for entry in fragments['pieces']:
        path = Path(entry['path'])
        assert str(path) not in opened
        unchanged(path, entry)
        with Path(entry['archive']).open('rb') as source:
            source.seek(entry['offset'])
            retained = source.read(entry['size_bytes'])
        assert hashlib.sha256(retained).hexdigest() == entry['sha256']
        assert retained == path.read_bytes()
        path.unlink()
        record({'action': 'remove_redundant_download_piece', **entry})

    for index, entry in enumerate(plan['clone_operations']):
        source, destination = Path(entry['source']), Path(entry['destination'])
        scoped(source)
        assert str(source) not in opened and str(destination) not in opened
        original = unchanged(destination, entry)
        source_info = source.stat()
        assert source_info.st_dev == original.st_dev and source_info.st_nlink == 1
        assert sha(source) == entry['sha256']
        source_attrs, destination_attrs = attrs(source), attrs(destination)
        assert source_attrs == destination_attrs
        assert hashlib.sha256(destination_attrs).hexdigest() == entry['xattrs_sha256']
        assert (
            source_info.st_mode,
            source_info.st_uid,
            source_info.st_gid,
            source_info.st_flags,
        ) == (original.st_mode, original.st_uid, original.st_gid, original.st_flags)
        temporary = destination.with_name(destination.name + f'.presto-clone-{os.getpid()}')
        assert not temporary.exists()
        result = clonefile(os.fsencode(source), os.fsencode(temporary), 0)
        if result:
            raise OSError(ctypes.get_errno(), 'APFS clone failed; no full-copy fallback')
        assert sha(temporary) == entry['sha256'] and attrs(temporary) == destination_attrs
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
                json.dumps({'cloned_files': index + 1, 'free_bytes': shutil.disk_usage(OUT).free}),
                flush=True,
            )

    after_state = {str(root): state(root) for root in WORKTREES}
    assert baseline == after_state, (
        'Worktree state changed during cleanup; inspect concurrent edits'
    )
    for archive in fragments['archives']:
        assert sha(Path(archive['path'])) == archive['sha256']
    journal.close()
    after = shutil.disk_usage(OUT).free
    result = {
        'finished_epoch': time.time(),
        'passed': True,
        'before_free_bytes': before,
        'after_free_bytes': after,
        'observed_free_change_bytes': after - before,
        'cloned_files': sum(r['action'] == 'share_duplicate_extents' for r in done),
        'removed_cache_files': sum(r['action'] == 'remove_regenerable_cache' for r in done),
        'removed_download_pieces': sum(
            r['action'] == 'remove_redundant_download_piece' for r in done
        ),
        'worktree_state_unchanged': True,
        'baseline_worktree_state': baseline,
        'all_unique_scientific_bytes_preserved': True,
        'journal_sha256': sha(OUT / 'cleanup-journal.jsonl'),
        'interpretation': 'Shared-volume capacity changes include unrelated concurrent activity.',
    }
    with (OUT / 'cleanup-result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(
        json.dumps({k: v for k, v in result.items() if k != 'baseline_worktree_state'}), flush=True
    )
