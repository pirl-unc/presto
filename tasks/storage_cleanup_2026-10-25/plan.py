"""Freeze metadata-compatible immutable duplicate and disposable-cache candidates."""

import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path

from audit import OUT, WORKTREES, sha


def attrs(path):
    return subprocess.check_output(['/usr/bin/xattr', '-lx', str(path)])


def open_files():
    result = subprocess.run(['/usr/sbin/lsof', '-nP', '-Fpn'], capture_output=True, text=True)
    assert result.returncode in (0, 1)
    return {line[1:] for line in result.stdout.splitlines() if line.startswith('n/')}


if __name__ == '__main__':
    inventory = json.loads((OUT / 'inventory.json').read_text())
    opened = open_files()
    modified = set()
    tracked = set()
    for root in WORKTREES:
        changed = subprocess.check_output(['git', 'diff', '--name-only', '-z', 'HEAD'], cwd=root)
        modified.update(str(root / p) for p in changed.decode().split('\0') if p)
        names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
        tracked.update(str(root / p) for p in names if p)
    operations = []
    skipped = []
    for group in inventory['duplicate_groups']:
        compatible = {}
        for record in group:
            path = Path(record['path'])
            info = path.stat()
            if str(path) in opened or str(path) in modified or path.is_symlink():
                skipped.append({'path': str(path), 'reason': 'open_modified_or_symlink'})
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
                skipped.append({'path': str(path), 'reason': 'ownership_type_or_links'})
                continue
            if (info.st_ino, info.st_size, info.st_mtime_ns) != (
                record['inode'],
                record['size_bytes'],
                record['mtime_ns'],
            ):
                skipped.append({'path': str(path), 'reason': 'changed_since_inventory'})
                continue
            attribute_digest = hashlib.sha256(attrs(path)).hexdigest()
            key = (
                info.st_dev,
                info.st_mode,
                info.st_uid,
                info.st_gid,
                info.st_flags,
                attribute_digest,
            )
            if key not in compatible:
                compatible[key] = record
            else:
                source = compatible[key]
                operations.append(
                    {
                        'source': source['path'],
                        'destination': str(path),
                        'sha256': record['sha256'],
                        'size_bytes': info.st_size,
                        'inode': info.st_ino,
                        'mtime_ns': info.st_mtime_ns,
                        'mode': info.st_mode,
                        'uid': info.st_uid,
                        'gid': info.st_gid,
                        'flags': info.st_flags,
                        'xattrs_sha256': attribute_digest,
                    }
                )
    caches = []
    now = time.time()
    for root in WORKTREES:
        for name in inventory['cache_directories']:
            directory = Path(name)
            if root not in directory.parents or any(
                p in {'.venv', 'site-packages'} for p in directory.parts
            ):
                continue
            for path in directory.rglob('*'):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or str(path) in tracked
                    or str(path) in opened
                ):
                    continue
                info = path.stat()
                if now - info.st_mtime < 3600:
                    continue
                ignored = subprocess.run(['git', 'check-ignore', '-q', '--', str(path)], cwd=root)
                if ignored.returncode != 0:
                    continue
                caches.append(
                    {
                        'path': str(path),
                        'size_bytes': info.st_size,
                        'allocated_bytes': info.st_blocks * 512,
                        'inode': info.st_ino,
                        'mtime_ns': info.st_mtime_ns,
                        'sha256': sha(path),
                    }
                )
    # Nested __pycache__ directories are possible: one deletion record per file.
    caches = list({r['path']: r for r in caches}.values())
    plan = {
        'created_epoch': now,
        'inventory_sha256': sha(OUT / 'inventory.json'),
        'clone_operations': operations,
        'cache_deletions': caches,
        'skipped': skipped,
        'clone_logical_bytes': sum(r['size_bytes'] for r in operations),
        'cache_allocated_bytes': sum(r['allocated_bytes'] for r in caches),
        'preserve_unique_evidence': True,
        'no_hardlinks_or_git_history_changes': True,
    }
    with (OUT / 'cleanup-plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    print(
        json.dumps(
            {
                k: v
                for k, v in plan.items()
                if k not in {'clone_operations', 'cache_deletions', 'skipped'}
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                'clone_files': len(operations),
                'cache_files': len(caches),
                'skipped_files': len(skipped),
            }
        )
    )
