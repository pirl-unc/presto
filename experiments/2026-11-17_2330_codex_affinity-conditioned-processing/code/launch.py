"""Durable H100 comparison; frozen source and explicit hardware."""
from pathlib import Path
import json
import os
import time
import modal
from common import NAME,E,BASE,VOLUME,dump
GPU=os.environ.get('PRESTO_MODAL_GPU','H100!')
assert GPU=='H100!'
app=modal.App('presto-affinity-conditioned-processing')
image=BASE
if modal.is_local():image=image.add_local_dir(E/'reproduce/runtime','/runtime/presto').add_local_dir(E/'code','/bundle')

@app.function(image=image,gpu=GPU,cpu=4,memory=65536,timeout=43200,volumes={'/volume':VOLUME},retries=0,single_use_containers=True)
def fit(arm,seed,steps):
    import torch
    from fit import fit as run
    root=Path('/volume')/NAME;out=root/'fits'/f'{arm}-s{seed}-n{steps}';out.mkdir(parents=True,exist_ok=False)
    dump(out/'hardware.json',{'requested_gpu':GPU,'name':torch.cuda.get_device_name(),
        'memory_bytes':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__})
    VOLUME.commit()
    try:return run(root,out,arm,seed,steps,commit=VOLUME.commit)
    except Exception as e:
        dump(out/'failure.json',{'error':repr(e)});raise
    finally:VOLUME.commit()

@app.function(image=image,cpu=2,memory=4096,timeout=86400,volumes={'/volume':VOLUME})
def coordinate(arms,seeds,steps):
    root=Path('/volume')/NAME;batch=f'n{steps}-'+'_'.join(arms)+'-s'+'_'.join(map(str,seeds))
    journal=root/'calls'/f'{batch}.json'
    if journal.exists():
        entries=json.loads(journal.read_text());calls=[(r['arm'],r['seed'],modal.FunctionCall.from_id(r['call'])) for r in entries]
    else:
        calls=[(arm,seed,fit.spawn(arm,seed,steps)) for seed in seeds for arm in arms]
        dump(journal,[{'arm':a,'seed':s,'steps':steps,'call':c.object_id} for a,s,c in calls]);VOLUME.commit()
    results=[];errors=[]
    for arm,seed,c in calls:
        try:results.append(c.get())
        except Exception as e:errors.append({'arm':arm,'seed':seed,'error':repr(e)})
        dump(root/'calls'/f'{batch}-progress.json',{'results':results,'errors':errors});VOLUME.commit()
    report={'status':'failed' if errors else 'completed','results':results,'errors':errors,'test_accessed':False}
    dump(root/'calls'/f'{batch}-completed.json',report);VOLUME.commit()
    if errors:raise ValueError(errors)
    return report

@app.local_entrypoint()
def main(arms:str='paired-junction',seeds:str='42',steps:int=16):
    arms=arms.split(',');seeds=list(map(int,seeds.split(',')))
    call=coordinate.spawn(arms,seeds,steps)
    dump(E/'reproduce'/f'fit-n{steps}-coordinator.json',{'call':call.object_id,'arms':arms,'seeds':seeds,'steps':steps})
    print('coordinator',call.object_id,flush=True)
    report=call.get();dump(E/'results'/f'fits-n{steps}-completion.json',report);print(json.dumps(report),flush=True)
