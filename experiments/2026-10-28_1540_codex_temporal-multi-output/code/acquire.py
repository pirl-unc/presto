"""Acquire fresh canonical sources on CPU; no model fitting or target evaluation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
import modal

FID = '2026-10-28_1540_codex_temporal-multi-output'
app = modal.App('presto-temporal-source-acquisition-20261028')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12').uv_pip_install('hitlist==2.0.40')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


@app.function(image=image,cpu=8,memory=98304,timeout=10800,max_containers=1,volumes={'/runs':volume})
def acquire():
    import importlib.metadata
    from hitlist.downloads import set_data_dir,fetch,FETCHABLE_DATASETS
    root=Path('/runs')/FID/'source'
    root.mkdir(parents=True,exist_ok=True)
    set_data_dir(root)
    receipt_path=root/'acquisition.json'
    receipt={'started_unix':time.time(),'status':'started','gpu':None,'cpu':8,'memory_mib':98304,'python':platform.python_version(),'hitlist':importlib.metadata.version('hitlist'),'sources':{}}
    receipt['environment']=subprocess.check_output(['python','-m','pip','freeze'],text=True).splitlines()
    def save():
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');volume.commit()
    save()
    try:
        for name in ('iedb','cedar'):
            print('Fetching canonical source',name,flush=True)
            path=fetch(name)
            with path.open(newline='') as stream:
                reader=csv.reader(stream);header1=next(reader);header2=next(reader)
            receipt['sources'][name]={'url':FETCHABLE_DATASETS[name]['url'],'path':str(path),'size_bytes':path.stat().st_size,'sha256':sha(path),'header1':header1,'header2':header2,'completed_unix':time.time()}
            save();print(name,'verified',path.stat().st_size,flush=True)
        receipt['status']='downloaded'
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:
        receipt['updated_unix']=time.time();save()
    return receipt


@app.local_entrypoint()
def main():
    family=Path(__file__).resolve().parents[1]
    invocation={'argv':['modal','run','--detach',str(Path(__file__).resolve())],'source_sha256':sha(Path(__file__)),'started_unix':time.time(),'mode':'CPU source download, no fit','requirements':['hitlist==2.0.40'],'requested_gpu':None,'cpu':8,'memory_mib':98304}
    (family/'reproduce/acquisition-launch.json').write_text(json.dumps(invocation,indent=2)+'\n')
    call=acquire.spawn()
    (family/'reproduce/acquisition-call.json').write_text(json.dumps({'call_id':call.object_id},indent=2)+'\n')
    result=call.get()
    (family/'results/acquisition.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'sources':{k:{x:v[x] for x in ('size_bytes','sha256')} for k,v in result['sources'].items()}},indent=2))
