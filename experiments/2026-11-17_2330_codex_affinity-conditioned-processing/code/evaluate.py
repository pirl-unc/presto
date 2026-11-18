"""Independent selected/terminal exports on identical sealed-validation populations."""
import gzip
import json
import time
from pathlib import Path
from dataclasses import replace
import modal
from common import NAME,BROAD,PRIOR,E,VOLUME,rows,sha,dump
from launch import image,GPU
app=modal.App('presto-processing-evaluation')


def supply_readouts(record):
    if record['endpoint']!='elution_observation':return
    supply=[];junction=[]
    for c in record['diagnostics']['candidates']:
        supply.append(sum(r['class_probability']*r['elution_terms']['processing'] for r in c['registers']))
        junction.append(sum(r['class_probability']*sum(r['elution_terms'].get('junction_processing',{}).values()) for r in c['registers']))
    # Known-class supply is candidate-independent, so subtracting it is exact.
    assert max(supply)-min(supply)<1e-5 and max(junction)-min(junction)<1e-5
    record.update(processing_supply=sum(supply)/len(supply),junction_processing=sum(junction)/len(junction))
    record['without_processing_logit']=record['prediction']-record['processing_supply']
    record['without_junction_logit']=record['prediction']-record['junction_processing']


@app.function(image=image,gpu=GPU,cpu=4,memory=65536,timeout=14400,volumes={'/volume':VOLUME},single_use_containers=True,retries=0)
def evaluate_run(run='parent',kind='selected'):
    import torch
    from fit import PARENT,PARENT_SHA
    from presto.data.modular import Observation,Context
    from presto.inference.modular import ModularPredictor
    from presto.training.modular_evaluation import evaluate
    from project_context import Projection
    assert kind in ('selected','last')
    root=Path('/volume')/NAME;broad=Path('/volume')/BROAD
    path=PARENT if run=='parent' else root/'fits'/run/'raw'/f'{kind}.pt'
    if run=='parent':assert kind=='selected' and sha(path)==PARENT_SHA
    else:
        fit_receipt=json.loads((root/'fits'/run/'receipt.json').read_text())
        assert fit_receipt['coverage']=='passed'
        if kind=='selected':assert sha(path)==fit_receipt['checkpoint_sha256']
    name=run+('-terminal' if kind=='last' else '')
    out=root/'evaluation'/name
    if (out/'receipt.json').exists():
        record=json.loads((out/'receipt.json').read_text());assert record['checkpoint_sha256']==sha(path)
        return {'run':name,'cached':True}
    if out.exists():out.rename(out.with_name(name+'-incomplete-'+str(time.time_ns())))
    out.mkdir(parents=True)
    start=time.monotonic();predictor=ModularPredictor(path,device='cuda');model=predictor.model.eval()
    assert not model.frozen_parameters and all(p.requires_grad for p in model.parameters())
    if kind=='last':assert predictor.checkpoint['metadata']['steps']==fit_receipt['steps']
    report={'run':name,'checkpoint_kind':kind,'checkpoint':str(path),'checkpoint_sha256':sha(path),
        'steps':predictor.checkpoint['metadata'].get('steps'),'data_sha256':{},'metrics':{},'context_projection':{},
        'hardware':{'requested_gpu':GPU,'name':torch.cuda.get_device_name(),'memory_bytes':torch.cuda.get_device_properties(0).total_memory},
        'test_accessed':False}
    populations={k:broad/'inputs'/f'{k}-validation.jsonl' for k in ('affinity','apc','broad','expanded')}
    populations['matched']=Path('/volume')/PRIOR/'inputs/matched-validation-v3.jsonl'
    populations['rank-matched']=root/'inputs/rank-matched-validation.jsonl'
    query=[]
    for population,source in populations.items():
        projection=Projection(model.config.to_dict());observations=[Observation.from_dict(projection.record(r)[0]) for r in rows(source)]
        assert all(r.split=='validation' for r in observations)
        metrics,records=evaluate(model,observations,list(range(len(observations))),predictor.scales,batch_size=64)
        for record in records:supply_readouts(record)
        report['metrics'][population]=metrics;report['data_sha256'][population]=sha(source);report['context_projection'][population]=projection.receipt()
        with gzip.open(out/f'{population}-predictions.jsonl.gz','wt') as stream:
            for record in records:stream.write(json.dumps(record,separators=(',',':'))+'\n')
        dump(out/f'{population}-metrics.json',metrics);VOLUME.commit()
        if population=='expanded':
            query=[replace(next(r for r in observations if r.mhc[0].mhc_class==cls),value=None,split='predict') for cls in ('I','II')]
    public=predictor.predict(query,unsupported_apc_policy='unknown')
    assert all(torch.isfinite(torch.tensor(r['logit'])) for r in public)
    affinity=[replace(r,endpoint='ic50',unit='nM') for r in query]
    response=predictor.predict(affinity+[replace(r,context=Context()) for r in affinity],unsupported_apc_policy='unknown')
    delta=max(abs(response[i]['transformed_value']-response[i+2]['transformed_value']) for i in range(2))
    assert delta==0
    report.update(seconds=time.monotonic()-start,public_inference=public,affinity_context_max_delta=delta)
    dump(out/'receipt.json',report);VOLUME.commit()
    return {'run':name,'seconds':report['seconds'],'checkpoint_sha256':report['checkpoint_sha256']}


@app.function(image=image,cpu=2,memory=4096,timeout=43200,volumes={'/volume':VOLUME})
def coordinate(runs):
    key='__'.join(runs);root=Path('/volume')/NAME/'calls';journal=root/f'eval-{key}.json'
    if journal.exists():
        entries=json.loads(journal.read_text());calls=[(r['run'],r['kind'],modal.FunctionCall.from_id(r['call'])) for r in entries]
    else:
        calls=[(run,kind,evaluate_run.spawn(run,kind)) for run in runs for kind in (['selected'] if run=='parent' else ['selected','last'])]
        dump(journal,[{'run':r,'kind':k,'call':c.object_id} for r,k,c in calls]);VOLUME.commit()
    result=[];errors=[]
    for run,kind,c in calls:
        try:result.append(c.get())
        except Exception as e:errors.append({'run':run,'kind':kind,'error':repr(e)})
        dump(root/f'eval-{key}-progress.json',{'results':result,'errors':errors});VOLUME.commit()
    if errors:raise ValueError(errors)
    return result

@app.local_entrypoint()
def main(runs:str='parent'):
    call=coordinate.spawn(runs.split(','));dump(E/'reproduce'/f'evaluation-{runs}-call.json',{'call':call.object_id,'runs':runs})
    result=call.get();dump(E/'results'/f'evaluation-{runs}-completion.json',result);print(json.dumps(result),flush=True)
