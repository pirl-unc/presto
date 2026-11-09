"""Share identical APFS extents in four clean task-created checkouts."""

import ctypes
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

root = Path('/private/tmp/presto-modular-model')
plan = json.loads((root/'tasks/modular_model/intact-checkout-cow-plan.json').read_text())
clonefile = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True).clonefile
clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
clonefile.restype = ctypes.c_int
before_free = shutil.disk_usage(root).free
records = []

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def attrs(path):
    return subprocess.check_output(['/usr/bin/xattr', '-lx', str(path)])

for record in plan['candidate_files']:
    source = Path(record['source']); target = Path(record['destination'])
    original = target.stat()
    assert original.st_nlink == 1 and not target.is_symlink()
    assert stat.S_IMODE(original.st_mode) == stat.S_IMODE(source.stat().st_mode)
    assert attrs(source) == attrs(target)
    assert digest(source) == digest(target) == record['sha256']
    temporary = target.with_name(target.name + '.presto-cow-temporary')
    assert not temporary.exists()
    code = clonefile(os.fsencode(source), os.fsencode(temporary), 0)
    if code:
        raise OSError(ctypes.get_errno(), 'clonefile failed; no full-copy fallback')
    assert digest(temporary) == record['sha256']
    assert temporary.stat().st_ino != source.stat().st_ino
    current = target.stat()
    assert (current.st_ino,current.st_mtime_ns,current.st_size) == (original.st_ino,original.st_mtime_ns,original.st_size)
    assert digest(target) == record['sha256']
    os.utime(temporary, ns=(original.st_atime_ns,original.st_mtime_ns))
    os.replace(temporary,target)
    assert digest(target) == digest(source) == record['sha256']
    final = target.stat()
    assert final.st_ino != source.stat().st_ino and final.st_nlink == 1
    assert stat.S_IMODE(final.st_mode) == stat.S_IMODE(original.st_mode)
    assert final.st_mtime_ns == original.st_mtime_ns
    records.append({**record,'source_inode':source.stat().st_ino,'old_target_inode':original.st_ino,
                    'new_target_inode':final.st_ino,'verified':True})
# Explicit allowlist is easier to audit than inferring worktree roots from paths.
worktrees = ['/private/tmp/hitlist-intact-degradomics-20261024', '/private/tmp/hitlist-intact-kinetics-20261025',
             '/private/tmp/hitlist-goldberg-composition-20261024', '/private/tmp/hitlist-pbmc-composition-20261024']
for directory in worktrees:
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=directory,text=True)
after_free = shutil.disk_usage(root).free
result = {'passed':True,'n_files':len(records),'logical_bytes':sum(r['size_bytes'] for r in records),
          'before_free_bytes':before_free,'after_free_bytes':after_free,'observed_free_change_bytes':after_free-before_free,
          'independent_inodes':True,'all_bytes_unchanged':True,'files':records,
          'interpretation':'shared-volume free-space change may include unrelated activity; no original data or checkpoints deleted'}
(root/'tasks/modular_model/intact-checkout-cow-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'}))
