"""Exact raw matrix archives in deterministic, bounded-size groups."""
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

FAMILY=Path(__file__).resolve().parents[1]


def main():
    assert (FAMILY/'results/summary.json').exists(),'Finish verification first'
    files=sorted([*(FAMILY/'results').glob('case-*.npz'),*(FAMILY/'results').glob('independent-case-*.npz')])
    assert len(files)==168
    groups=[[]];total=0
    for path in files:
        size=path.stat().st_size
        if total+size>32*1024**2 and groups[-1]:
            groups.append([]);total=0
        groups[-1].append(path);total+=size
    records={};archives=[]
    for i,group in enumerate(groups):
        output=FAMILY/f'results/raw-arrays-{i:02d}.tar.gz'
        assert not output.exists(),'Refuse to overwrite archive'
        members=[]
        with output.open('wb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as gz,tarfile.open(fileobj=gz,mode='w') as tar:
            for p in group:
                rel=str(p.relative_to(FAMILY));members.append(rel)
                records[rel]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                info=tar.gettarinfo(str(p),arcname=rel)
                info.mtime=info.uid=info.gid=0
                info.uname=info.gname='';info.mode=0o644
                with p.open('rb') as raw:tar.addfile(info,raw)
        archives.append(dict(path=str(output.relative_to(FAMILY)),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),members=members,bytes=output.stat().st_size))
        print(output.name,len(group),output.stat().st_size,flush=True)
    (FAMILY/'results/raw-artifacts.json').write_text(json.dumps(dict(scope='Exact raw arrays; original paths restored and byte-verified by reproduce/unpack.py. Local originals preserved.',files=records,archives=archives),indent=2)+'\n')


if __name__=='__main__':main()
