"""Read-only Presto storage inventory and exact duplicate discovery."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
MAIN = Path('/Users/iskander/code/presto')
WORKTREES = [
    MAIN,
    Path('/private/tmp/presto-modular-model'),
    Path('/private/tmp/presto-pr66-completion'),
    Path('/private/tmp/presto-specificity-audit'),
]
ROOTS = [MAIN] + sorted(
    p for p in Path('/private/tmp').iterdir() if p.name.startswith('presto') and not p.is_symlink()
)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def state(root):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root)

    return {
        'head': git('rev-parse', 'HEAD').decode().strip(),
        'tracked_status': git('status', '--porcelain', '--untracked-files=no').decode(),
        'working_diff_sha256': hashlib.sha256(git('diff', '--binary')).hexdigest(),
        'staged_diff_sha256': hashlib.sha256(git('diff', '--cached', '--binary')).hexdigest(),
        'modified_paths': git('diff', '--name-only', '-z', 'HEAD').decode().split('\0'),
    }


if __name__ == '__main__':
    started = time.time()
    before = {str(root): state(root) for root in WORKTREES}
    inventory, summary = [], {}
    cache_directories = []
    for root in ROOTS:
        total = logical = count = 0
        categories = defaultdict(lambda: {'files': 0, 'logical_bytes': 0, 'allocated_bytes': 0})
        walks = (
            os.walk(root, followlinks=False) if root.is_dir() else [(root.parent, [], [root.name])]
        )
        for directory, directories, names in walks:
            directory = Path(directory)
            directories[:] = [
                d for d in directories if d != '.git' and not (directory / d).is_symlink()
            ]
            for name in names:
                path = directory / name
                if path.is_symlink():
                    continue
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                allocation = info.st_blocks * 512
                total += allocation
                logical += info.st_size
                count += 1
                category = path.suffix.lower() or '[no suffix]'
                categories[category]['files'] += 1
                categories[category]['logical_bytes'] += info.st_size
                categories[category]['allocated_bytes'] += allocation
                if info.st_size >= 1024**2 and not any(
                    p in {'.venv', 'site-packages', '__pycache__'} for p in path.parts
                ):
                    inventory.append(
                        {
                            'path': str(path),
                            'root': str(root),
                            'size_bytes': info.st_size,
                            'allocated_bytes': allocation,
                            'device': info.st_dev,
                            'inode': info.st_ino,
                            'nlink': info.st_nlink,
                            'mode': info.st_mode,
                            'mtime_ns': info.st_mtime_ns,
                        }
                    )
            for name in directories:
                if name in {'__pycache__', '.pytest_cache', '.ruff_cache'}:
                    cache_directories.append(str(directory / name))
        summary[str(root)] = {
            'allocated_bytes': total,
            'logical_bytes': logical,
            'files': count,
            'categories': dict(categories),
        }
    sizes = defaultdict(list)
    for record in inventory:
        if record['nlink'] == 1 and started - record['mtime_ns'] / 1e9 > 3600:
            sizes[record['size_bytes']].append(record)
    exact = defaultdict(list)
    for records in sizes.values():
        if len(records) < 2:
            continue
        for record in records:
            path = Path(record['path'])
            current = path.stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (
                record['inode'],
                record['size_bytes'],
                record['mtime_ns'],
            ):
                continue
            record['sha256'] = sha(path)
            current = path.stat()
            assert (current.st_ino, current.st_size, current.st_mtime_ns) == (
                record['inode'],
                record['size_bytes'],
                record['mtime_ns'],
            )
            exact[record['sha256']].append(record)
    duplicates = sorted(
        (records for records in exact.values() if len(records) > 1),
        key=lambda records: records[0]['size_bytes'] * (len(records) - 1),
        reverse=True,
    )
    result = {
        'started_epoch': started,
        'seconds': time.time() - started,
        'free_bytes': shutil.disk_usage(OUT).free,
        'worktree_state': before,
        'roots': summary,
        'large_files': inventory,
        'duplicate_groups': duplicates,
        'cache_directories': cache_directories,
        'duplicate_logical_bytes': sum(r[0]['size_bytes'] * (len(r) - 1) for r in duplicates),
        'scope_limit': (
            'APFS shared extents can already overlap; these are logical duplicate bytes, '
            'not promised recovery.'
        ),
    }
    with (OUT / 'inventory.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(
        json.dumps(
            {
                'roots': len(ROOTS),
                'large_files': len(inventory),
                'duplicate_groups': len(duplicates),
                'duplicate_logical_bytes': result['duplicate_logical_bytes'],
                'largest_roots': sorted(
                    ((k, v['allocated_bytes']) for k, v in summary.items()),
                    key=lambda r: r[1],
                    reverse=True,
                )[:12],
                'largest_duplicate_groups': [
                    {
                        'size_bytes': r[0]['size_bytes'],
                        'copies': len(r),
                        'paths': [x['path'] for x in r],
                    }
                    for r in duplicates[:12]
                ],
            },
            indent=2,
        )
    )
