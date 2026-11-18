"""Canonical manifest runner for harmonized, state-absent and shuffled controls."""
from pathlib import Path
import hashlib,json,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def manifest(arm,inputs,parent,reference,out,seed=42):
    from presto.data.modular_sources import MANIFEST_VERSION
    context=json.loads((Path(inputs)/'cohort.json').read_text())['model_context']
    model=json.loads(Path(reference).read_text())['model']
    if arm!='state_absent':model.update(context,state_bottleneck=8,state_mode='harmonized')
    sources=[]
    for filename in ('train-shuffled-global.jsonl' if arm.startswith('shuffled_global') else 'train-shuffled.jsonl' if arm=='shuffled' else 'train.jsonl','validation.jsonl'):
        p=Path(inputs)/'data'/filename
        sources.append(dict(path=str(p),sha256=sha(p),source='hitlist_native_ms',population='Resolved current Hitlist APC arms; sequence proposals from prior pool only',label_policy='explicit positive MS versus synthetic same-protein background; HAP1 unknown PTM retained as sequence-only; legacy source IDs pinned',split_policy='source-protein split, protected related-sequence embargo and cross-partition sequence-family filtering'))
    result={'schema':MANIFEST_VERSION,'model':model,'sources':sources,
       'pretrained':{'path':str(parent),'sha256':sha(parent),'freeze':[x for x in model['endpoints'] if x!='elution_observation'],**({'allow_apc_expansion':True} if arm!='state_absent' else {})},
       'training':{'seed':seed,'batch_size':32,'max_steps':800,'max_epochs':12,'min_epochs':1,'patience':12,
          'learning_rate':.0003,'weight_decay':.0001,'identity_weight':0.,'balance_by':['mhc_class'],
          'checkpoint_every_steps':200,'validation_every_steps':200,'selection_endpoint':'elution_observation',
          'selection_reduction':'study_mean','train_evaluation_max_rows':512,
          'test_policy':'no_test_training_diagnostic','no_test_reason':'Development-only APC validation; canonical TEST remains sealed',
          'context_support_policy':'predictive','coverage_policy':'required'}}
    Path(out).write_text(json.dumps(result,indent=2));return result

def run(arm,inputs,parent,reference,out,device='cuda',seed=42):
    from presto.data.modular_sources import ObservationIndex
    from presto.data.modular import Observation
    from presto.training.modular_run import fit_index
    from presto.training.modular import load_checkpoint
    from presto.training.modular_evaluation import evaluate
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=out/'manifest.json';manifest(arm,inputs,parent,reference,spec,seed)
    index=ObservationIndex(spec)
    summary=fit_index(index,out/'raw',device=device)
    model,scales,_=load_checkpoint(out/'raw/selected.pt',device=device)
    metrics,records=evaluate(model,index,index.partitions['validation:assay'],scales,batch_size=32)
    (out/'validation-predictions.json').write_text(json.dumps(records))
    (out/'validation-metrics.json').write_text(json.dumps(metrics,indent=2))
    affinity=Path(inputs)/'affinity-validation.jsonl'
    if affinity.exists():
        rows=[Observation.from_dict(json.loads(line)) for line in affinity.open()]
        base,base_scales,_=load_checkpoint(parent,device=device)
        before,base_records=evaluate(base,rows,list(range(len(rows))),base_scales,batch_size=32)
        after,records=evaluate(model,rows,list(range(len(rows))),scales,batch_size=32)
        drift=[abs(a['prediction']-b['prediction']) for a,b in zip(base_records,records,strict=True)]
        (out/'affinity-predictions.json').write_text(json.dumps(records))
        (out/'affinity-check.json').write_text(json.dumps({'before':before,'after':after,'n':len(drift),'max_absolute_drift':max(drift),'mean_absolute_drift':sum(drift)/len(drift)},indent=2))
    if arm=='state_absent':
        base,base_scales,_=load_checkpoint(parent,device=device)
        m,r=evaluate(base,index,index.partitions['validation:assay'],base_scales,batch_size=32)
        (out/'reference-predictions.json').write_text(json.dumps(r));(out/'reference-metrics.json').write_text(json.dumps(m,indent=2))
    return {'arm':arm,'path':str(out),'steps':summary['steps'],'seconds':summary['elapsed_seconds'],'coverage':summary['coverage_gate']['status']}
