"""Raw-source metadata counts only, independent of measurements and predictions."""
from pathlib import Path
from collections import Counter
import hashlib,json,time
import modal

FID='2026-10-28_1540_codex_temporal-multi-output'
app=modal.App('presto-temporal-metadata-20261028')
volume=modal.Volume.from_name('presto-checkpoints')
image=modal.Image.debian_slim(python_version='3.12').uv_pip_install('pyarrow==26.0.0')

@app.function(image=image,cpu=4,memory=16384,timeout=3600,max_containers=1,volumes={'/runs':volume})
def inventory():
    import pyarrow.csv as pc
    import pyarrow.compute as compute
    root=Path('/runs')/FID/'source'
    receipt=json.loads((root/'acquisition.json').read_text())
    fields=['Reference | PMID','Reference | Date','Assay | Method','Assay | Response measured','Assay | Units','MHC Restriction | Name','MHC Restriction | Class','MHC Restriction | Evidence Code']
    counts=Counter()
    for source,s in receipt['sources'].items():
        names=[f'{a} | {b}' for a,b in zip(s['header1'],s['header2'],strict=True)]
        reader=pc.open_csv(s['path'],read_options=pc.ReadOptions(column_names=names,skip_rows=2,block_size=1<<24),parse_options=pc.ParseOptions(newlines_in_values=True),convert_options=pc.ConvertOptions(include_columns=fields,column_types={name:'string' for name in fields}))
        for batch in reader:
            subset=batch.filter(compute.equal(batch.column(fields.index('Reference | Date')),'2026'))
            columns=[subset.column(i).to_pylist() for i in range(len(fields))]
            counts.update((source,*row) for row in zip(*columns,strict=True))
        print(source,'metadata strata',len(counts),flush=True)
    result=[{'source':key[0],**dict(zip(fields,key[1:],strict=True)),'n':n} for key,n in counts.items()]
    path=root/'inventory-2026.json';path.write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result

@app.local_entrypoint()
def main():
    f=Path(__file__).resolve().parents[1]
    source=Path(__file__)
    (f/'reproduce/inventory-2026-launch.json').write_text(json.dumps({'argv':['modal','run','--detach',str(source.resolve())],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'started_unix':time.time(),'cpu':4,'memory_mib':16384,'gpu':None},indent=2)+'\n')
    c=inventory.spawn();(f/'reproduce/inventory-2026-call.json').write_text(json.dumps({'call_id':c.object_id})+'\n')
    result=c.get();(f/'results/inventory-2026.json').write_text(json.dumps(result,indent=2)+'\n')
    print('inventoried',len(result),'strata',flush=True)
