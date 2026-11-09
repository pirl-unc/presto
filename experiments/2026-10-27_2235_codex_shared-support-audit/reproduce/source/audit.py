"""Saved-output and weight audit; probes consume training rows only."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from presto.data.modular import ENDPOINTS, Observation
from presto.training.modular import load_checkpoint, batch_loss
from presto.models.modular_support import feature_report

ROOT = Path(__file__).resolve().parents[3]
FAMILY = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'experiments/2026-10-27_2100_codex_transfer-improvements'
RAW = ROOT / 'artifacts' / PRIOR.name / 'runs'

def read(path):
    return json.loads(Path(path).read_text())

def records(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(name, value):
    (FAMILY / 'results' / name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def stats(rows):
    exact=[x for x in rows if x['qualifier']=='=']
    y=np.array([x['target'] for x in exact]);p=np.array([x['prediction'] for x in exact])
    bounds=[x for x in rows if x['qualifier']!='=']
    rho=float(spearmanr(y,p).statistic) if len(y)>2 and np.ptp(y)>0 and np.ptp(p)>0 else None
    return dict(n=len(rows),exact=len(exact),spearman=rho,rmse=float(np.sqrt(np.mean((p-y)**2))) if len(y) else None,mean_error=float(np.mean(p-y)) if len(y) else None,prediction_sd=float(np.std(p)) if len(y) else None,target_sd=float(np.std(y)) if len(y) else None,bounds=len(bounds),bound_satisfaction=float(np.mean([x['loss']==0 for x in bounds])) if bounds else None)

def audit_checkpoint(name):
    directory=RAW/name/'raw'
    model,scales,payload=load_checkpoint(directory/'selected.pt')
    initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=True)
    last=torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
    rows=[Observation.from_dict(x['observation']) for x in records(PRIOR/'results'/name/'training-replay.jsonl.gz')]
    assert len(rows)==64 and all(x.split=='train' for x in rows)
    # Deterministic diagnostic gradients are not claimed to reproduce training steps.
    model.eval();model.zero_grad(set_to_none=True)
    loss,_=batch_loss(model,rows,model(rows),scales);loss.backward()
    coverage=payload['metadata']['coverage'];detail=[];structural=[]
    for n,p in model.named_parameters():
        a=initial['state_dict'][n];b=p.detach();z=last['state_dict'][n]
        g=p.grad
        detail.append(dict(parameter=n,numel=p.numel(),selected_changed=int((a!=b).sum()),last_changed=int((a!=z).sum()),selected_unchanged=int((a==b).sum()),selected_zero=int((b==0).sum()),relative_l2_update=float((b-a).norm()/(a.norm()+1e-12)),probe_gradient_nonzero=int(g.ne(0).sum()) if g is not None else 0,probe_gradient_norm=float(g.norm()) if g is not None else 0,**coverage[n]))
        if n.endswith('in_proj_bias'):
            width=p.numel()//3
            structural.append(dict(parameter=n,slice=[width,2*width],meaning='key projection bias adds a constant to logits across keys; cancels in softmax',probe_gradient_max=float(g[width:2*width].abs().max()),query_value_gradient_max=float(torch.cat([g[:width],g[2*width:]]).abs().max()),changed_elements=int((a[width:2*width]!=b[width:2*width]).sum())))
    output_matrix={n:dict(allocated=n in model.config.endpoints,support=model.support.get(n),dependency_training_evidence='see checkpoint tensor/feature coverage' if n in model.config.endpoints else 'not allocated; no trained output claimed') for n in ENDPOINTS}
    result=dict(condition=name,checkpoint=str(directory/'selected.pt'),checkpoint_sha256=sha(directory/'selected.pt'),selected_epoch=payload['metadata'].get('epoch'),parameters=detail,structural_redundancies=structural,input_features=feature_report(model),outputs=output_matrix,probe=dict(split='train',n=64,loss=float(loss.detach()),mode='eval-mode gradients; not training-history element evidence'),limitation='Existing checkpoint stores tensor-level and input-column support, not historical elementwise gradients. Initial/final movement cannot prove learning or rule out decay-only changes.')
    dump(name+'-weights.json',result)
    return dict(condition=name,parameters=sum(x['numel'] for x in detail),selected_unchanged=sum(x['selected_unchanged'] for x in detail),untrained_tensors=[x['parameter'] for x in detail if not x['gradient_and_update_steps']],structural_key_bias_elements=sum(x['slice'][1]-x['slice'][0] for x in structural),outputs=list(model.config.endpoints))

def failure_groups(name,split):
    path=PRIOR/'results'/name/f'{split}-predictions.jsonl.gz'
    rows=records(path);groups=defaultdict(list);residuals=[]
    for x in rows:
        kind=x['mhc_classes'][0];obs=x['observation'];mhc=obs['mhc'][0];raw=obs.get('provenance',{}).get('raw_record',{})
        restriction=mhc.get('provenance',{}).get('reported_restriction') or hashlib.sha256((mhc['groove1']+mhc['groove2']).encode()).hexdigest()[:12]
        for category,value in [('class',kind),('length',kind+':'+str(len(x['peptide']))),('mhc_pair',kind+':'+restriction),('qualifier',kind+':'+x['qualifier']),('target_bin',kind+':'+str(int(np.floor(x['target'])))),('study',kind+':'+str(raw.get('pmid') or raw.get('reference_iri') or 'unknown'))]:
            groups[(category,value)].append(x)
        residuals.append(dict(condition=name,split=split,observation_id=x['observation_id'],mhc_class=kind,restriction=restriction,peptide_length=len(x['peptide']),target=x['target'],prediction=x['prediction'],qualifier=x['qualifier'],error=x['prediction']-x['target'],loss=x['loss']))
    scores=[dict(condition=name,split=split,group=key[0],value=key[1],**stats(rr)) for key,rr in sorted(groups.items())]
    dump(name+'-'+split+'-failures.json',dict(source=str(path),source_sha256=sha(path),groups=scores,residuals=residuals))
    return scores

def main():
    started=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    names=[f'{v}-s{s}' for s in (42,43) for v in ('interaction-natural-joint','interaction-I','interaction-II')]
    checkpoint_summary=[audit_checkpoint(n) for n in names]
    rows=[]
    for n in names:
        for split in ('validation','test'):rows.extend(failure_groups(n,split))
    with (FAMILY/'results/failure-groups.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    dump('summary.json',dict(status='completed',checkpoints=checkpoint_summary,seconds=time.monotonic()-started,new_test_inference=False,new_training=False))
    print(json.dumps(checkpoint_summary,indent=2),flush=True)

if __name__=='__main__':main()
