"""Source attribution only: downloaded/indexed versus actually admitted TRAIN rows."""

from collections import Counter
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

import modal

FID='2026-10-29_1428_codex_assay-methods-censoring'
PARENT='2026-10-28_1540_codex_temporal-multi-output'
app=modal.App('presto-actual-source-ledger-20261029')
volume=modal.Volume.from_name('presto-checkpoints')
image=modal.Image.debian_slim(python_version='3.12').uv_pip_install('pyarrow==26.0.0')
if modal.is_local():
    f=Path(__file__).resolve().parents[1]
    image=image.add_local_file(f.parent/PARENT/'reproduce/manifests/wide-multi-s42.json','/bundle/manifest.json')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


@app.function(image=image,cpu=4,memory=8192,timeout=3600,volumes={'/volume':volume})
def run():
    import pyarrow.parquet as pq
    start=time.monotonic();root=Path('/volume')/PARENT
    export=root/'admission/canonical-ms-export.parquet'
    expected='0f6fe591e6f5c12975ec89bdbdc8e17b1b910b26e24e27a80ded09a6576e114e'
    assert sha(export)==expected
    fields=['source','has_ptm','mhc_class','mhc_allele_provenance','restriction_evidence']
    counts=Counter()
    for batch in pq.ParquetFile(export).iter_batches(columns=fields,batch_size=65536):
        for row in batch.to_pylist():counts[json.dumps(row,sort_keys=True)]+=1
    manifest=json.loads(Path('/bundle/manifest.json').read_text())
    assert sha('/bundle/manifest.json')=='5541024b53ce7d09ef6779639eb9fb1b7fd2f72c770f84ba311a5b294053b9a1'
    spec=next(s for s in manifest['sources'] if Path(s['path']).name=='hitlist_native_ms-train.jsonl')
    source=root/'revisions/kinetic-review-v2/prepared'/Path(spec['path']).name
    h=hashlib.sha256();train=Counter();studies=Counter();n=0
    with source.open('rb') as stream:
        for line in stream:
            h.update(line);r=json.loads(line);p=r['provenance'];raw=p['raw_record'];n+=1
            assert r['split']=='train'
            train[raw.get('source') or 'missing']+=1
            studies[raw.get('source') or 'missing',p['study_id']]+=1
    assert h.hexdigest()==spec['sha256'] and n==1641063
    result=dict(status='verified_source_ledger',requested_gpu=None,cpu=4,memory_mib=8192,
                export=dict(path=str(export),sha256=expected,rows=sum(counts.values())),
                export_metadata=[dict(**json.loads(k),n=v) for k,v in counts.items()],
                train=dict(path=str(source),sha256=spec['sha256'],rows=n,source_counts=dict(train)),
                train_studies=[dict(source=s,study_id=p,n=v) for (s,p),v in studies.items()],
                seconds=time.monotonic()-start,model_fitted=False,validation_test_used=False)
    out=Path('/volume')/FID/'source-ledger.json'
    assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def main():
    f=Path(__file__).resolve().parents[1];callpath=f/'reproduce/source-ledger-call.json'
    assert not callpath.exists()
    shutil.copyfile(__file__,f/'reproduce/source/source_ledger.py')
    launch=dict(argv=['python','-m','modal','run','--detach',__file__],source_sha256=sha(__file__),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                git_dirty=subprocess.check_output(['git','status','--short'],text=True),requested_gpu=None)
    (f/'reproduce/source-ledger-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    call=run.spawn();callpath.write_text(json.dumps(dict(call_id=call.object_id),indent=2)+'\n')
    result=call.get();(f/'results/source-ledger.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],train=result['train'],seconds=result['seconds'])))
