"""CPU audit and assembly; uses the frozen canonical runtime."""
from pathlib import Path
import json
import time
import modal
from common import NAME,E,ROOT,BASE,VOLUME,dump
app=modal.App('presto-processing-data-audit')
image=BASE
if modal.is_local():
    image=(image.add_local_dir(E/'reproduce/runtime','/runtime/presto').add_local_dir(E/'code','/bundle')
           .add_local_file(ROOT/'data/shared-presentation-20261117/proteins.fa.gz','/inputs/proteins.fa.gz'))

@app.function(image=image,cpu=8,memory=65536,timeout=14400,volumes={'/volume':VOLUME})
def run(action,stage):
    import pool_analysis
    root=Path('/volume')/NAME
    if action=='analyze':result=pool_analysis.analyze(root,stage)
    elif action=='assemble':result=pool_analysis.assemble(root,stage)
    elif action=='finalize':
        pool_analysis.analyze(root,stage);VOLUME.commit()
        result=pool_analysis.assemble(root,stage)
    elif action=='validation':
        try:result=pool_analysis.validation(root)
        finally:VOLUME.commit()
    elif action=='runtime':
        records=[]
        for which in (1,2):
            completed=json.loads((root/f'external/completed-stage{which}.json').read_text())
            assert completed['status']=='completed' and not completed['errors']
            jobs=[]
            for item in completed['results']:
                score=json.loads(Path(item['path']).read_text())
                assert len(score['scores'])==item['rows']
                jobs.append({k:score[k] for k in ('class','allele','version','query_sha256','new_queries','seconds')})
            records.append({'stage':which,'jobs':len(jobs),'queries_returned':sum(x['rows'] for x in completed['results']),
                'new_queries':sum(x['new_queries'] for x in jobs),'successful_score_file_seconds':sum(x['seconds'] for x in jobs),
                'max_worker_seconds':max(x['seconds'] for x in jobs),'worker_details':jobs})
        result={'stages':records,'requested_cpu_per_worker':2,'maximum_workers':16,'test_accessed':False,
                'scope':'elapsed times in successful score files; excludes failed/preempted segments; not coordinator wall time'}
        dump(root/'scoring-runtime-summary.json',result)
    else:raise ValueError(action)
    VOLUME.commit();return result

@app.local_entrypoint()
def main(action:str='analyze',stage:int=1):
    call=run.spawn(action,stage);dump(E/'reproduce'/f'data-{action}-stage{stage}-call.json',{'call':call.object_id})
    result=call.get();dump(E/'results'/f'data-{action}-stage{stage}.json',result)
    display={k:v for k,v in result.items() if k not in ('jobs','studies')}
    if action=='runtime':display['stages']=[{k:v for k,v in s.items() if k!='worker_details'} for s in result['stages']]
    print(json.dumps(display),flush=True)
