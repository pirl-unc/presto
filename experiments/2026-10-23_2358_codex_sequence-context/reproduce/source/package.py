"""Archive original checkpoints/arrays, preserving their exact executed bytes."""
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

FAMILY=Path(__file__).resolve().parents[1]
for arm in ['base','rate_context','response_termini','both','both_hybrid']:
    state=json.loads((FAMILY/'results'/arm/'status.json').read_text())
    assert state['status']!='running'
    if state['status']=='completed':assert json.loads((FAMILY/'results'/arm/'independent-verification.json').read_text())['all_pass']
files=sorted(p for p in (FAMILY/'results').rglob('*') if p.suffix in ['.npz','.pt'])
groups=[[]];size=0
for p in files:
    if groups[-1] and size+p.stat().st_size>32*1024**2:groups.append([]);size=0
    groups[-1].append(p);size+=p.stat().st_size
records={};archives=[]
for i,group in enumerate(groups):
    archive=FAMILY/f'results/raw-training-{i:02d}.tar.gz';assert not archive.exists()
    members=[]
    with archive.open('wb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for p in group:
            rel=str(p.relative_to(FAMILY));members.append(rel)
            records[rel]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            info=tar.gettarinfo(str(p),arcname=rel);info.mtime=info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
            with p.open('rb') as raw:tar.addfile(info,raw)
    archives.append(dict(path=str(archive.relative_to(FAMILY)),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=members))
    print(archive.name,len(group),archive.stat().st_size,flush=True)
(FAMILY/'results/raw-artifacts.json').write_text(json.dumps(dict(files=records,archives=archives,scope='Exact executed checkpoints/arrays; prediction gzip files are kept directly. Restore original paths with reproduce/unpack.py.'),indent=2)+'\n')
