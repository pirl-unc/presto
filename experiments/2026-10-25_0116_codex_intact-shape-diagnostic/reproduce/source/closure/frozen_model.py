"""Exact independent fragmentation and matched conditional score gradients."""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def data():
    records = {s:pq.read_table(ROOT/f"results/data/{s}/covariates.parquet").to_pylist() for s in ("train","val","test")}
    sequences = sorted({r['peptide_sequence'] for r in records['train']})
    index = {s:i for i,s in enumerate(sequences)}
    parent = records['train'][0]['parent_sequence']
    assert all(r['parent_sequence']==parent for rows in records.values() for r in rows)
    amino_acids = sorted(set(parent[:-1]))
    aa_index = torch.tensor([amino_acids.index(a) for a in parent[:-1]])
    result = {}
    for split,rows in records.items():
        result[split] = {'rows':rows,'sequence_index':torch.tensor([index[r['peptide_sequence']] for r in rows]),
                         'start':torch.tensor([r['start_1based']-1 for r in rows]),
                         'end':torch.tensor([r['end_1based'] for r in rows]),
                         'time':torch.tensor([r['time_minutes'] for r in rows]),
                         'weight':torch.tensor([r['weight'] for r in rows]),'n_sequences':len(sequences)}
    labels = pq.read_table(ROOT/'results/data/train/outcomes.parquet').to_pylist()
    assert [r['assignment_id'] for r in labels]==[r['assignment_id'] for r in records['train']]
    result['train']['target'] = torch.tensor([r['target'] for r in labels])
    return result,parent,amino_acids,aa_index,sequences


def masks(start,end,length):
    bonds = torch.arange(length-1)
    internal=(bonds[None,:]>=start[:,None]) & (bonds[None,:]<(end-1)[:,None])
    boundary=(bonds[None,:]==(start-1)[:,None]) | (bonds[None,:]==(end-1)[:,None])
    return internal,boundary


def log_events(rates,exposure,start,end,draws=0,generator=None,full_draws=False,return_sample=False):
    internal,boundary=masks(start,end,len(rates)+1)
    h=exposure[:,None]*rates[None,:]
    log_break=torch.log(-torch.expm1(-h))
    exact=(-h*internal+log_break*boundary).sum(1)
    means=None
    if draws:
        p=(-torch.expm1(-h)).detach()
        if full_draws:
            means=(torch.rand((draws,*p.shape),generator=generator)<p).double().mean(0)
        else:
            means=torch.binomial(torch.full_like(p,float(draws)),p,generator=generator)/draws
        means=torch.where(boundary,1.,torch.where(internal,0.,means))
        score=(means*log_break-(1-means)*h).sum(1)
        exact=exact.detach()+score-score.detach()
    if return_sample:
        return exact,means
    return exact


def initial(kind,n_residues):
    n=1 if kind=='uniform' else n_residues
    rate=(.001-1e-7)/(1-1e-7)
    tau=(10-.001)/(120-.001)
    return torch.tensor([math.log(rate/(1-rate))]*n+[math.log(tau/(1-tau))],requires_grad=True)


def decode(theta,kind,aa_index):
    category_rates=1e-7+(1-1e-7)*torch.sigmoid(theta[:-1])
    rates=category_rates.expand(len(aa_index)) if kind=='uniform' else category_rates[aa_index]
    tau=.001+(120-.001)*torch.sigmoid(theta[-1])
    return rates,tau,category_rates


def component(theta,kind,batch,aa_index,draws=0,generator=None,full_draws=False):
    rates,tau,_=decode(theta,kind,aa_index)
    exposure=batch['time']+tau
    events=log_events(rates,exposure,batch['start'],batch['end'],draws,generator,full_draws)
    parent=-exposure*rates.sum()
    return (events-parent)/math.log(10)


def profile(values,batch):
    numerator=torch.zeros(batch['n_sequences']).index_add(0,batch['sequence_index'],batch['weight']*values)
    denominator=torch.zeros(batch['n_sequences']).index_add(0,batch['sequence_index'],batch['weight'])
    assert torch.all(denominator>0)
    return numerator/denominator


def objective(theta,kind,train,aa_index,draws=0,generator=None,full_draws=False):
    physical=component(theta,kind,train,aa_index,draws,generator,full_draws)
    offsets=profile(train['target']-physical,train)
    prediction=physical+offsets[train['sequence_index']]
    mse=((prediction-train['target']).square()*train['weight']).sum()
    _,_,rates=decode(theta,kind,aa_index)
    penalty=.01*(rates.log()-rates.log().mean()).square().mean() if kind=='residue' else theta.new_zeros(())
    return mse+penalty,mse,penalty,offsets


def predict(theta,kind,train,query,aa_index):
    offsets=profile(train['target']-component(theta,kind,train,aa_index),train)
    return component(theta,kind,query,aa_index)+offsets[query['sequence_index']],offsets


def direct_controls(train,queries):
    seq=train['sequence_index'].numpy();y=train['target'].numpy();w=train['weight'].numpy();x=train['time'].numpy()/80
    n=train['n_sequences'];mass=np.bincount(seq,weights=w,minlength=n)
    meanx=np.bincount(seq,weights=w*x,minlength=n)/mass
    meany=np.bincount(seq,weights=w*y,minlength=n)/mass
    xc=x-meanx[seq];yc=y-meany[seq]
    sxx=np.bincount(seq,weights=w*xc*xc,minlength=n)
    sxy=np.bincount(seq,weights=w*xc*yc,minlength=n)
    assert np.all(sxx>0)
    global_slope=sxy.sum()/sxx.sum()
    latest=np.array([y[(seq==i)&(x==x[seq==i].max())].mean() for i in range(n)])
    models={'mean':(meany,np.zeros(n)), 'last':(latest,np.zeros(n)),
            'global_linear':(meany-global_slope*meanx,np.full(n,global_slope))}
    for alpha in (.001,.01,.1,1.):
        penalty=alpha/n
        matrix=np.diag(sxx+penalty)-np.full((n,n),penalty/n)
        slope=np.linalg.solve(matrix,sxy)
        assert np.max(np.abs(matrix@slope-sxy))<1e-10
        models[f'shrunk_{alpha:g}']=(meany-slope*meanx,slope)
    predictions={name:{split:intercepts[q['sequence_index'].numpy()]+slope[q['sequence_index'].numpy()]*q['time'].numpy()/80
                       for split,q in queries.items()} for name,(intercepts,slope) in models.items()}
    return models,predictions,latest


def dump_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
