"""Profile one scale per condition at each current parameter state."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import torch
from presto.models.independent_cut import log_fragment_expectation
from rational import expectations

FAMILY=Path(__file__).resolve().parents[1]
LN10=math.log(10.)
LAST_QUERY = {}


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def offset_optimum(base,target,weight,condition,n_conditions):
    numerator=base.new_zeros(n_conditions).index_add(0,condition,weight*(target-base))
    denominator=base.new_zeros(n_conditions).index_add(0,condition,weight)
    return numerator/denominator


def load():
    torch.set_num_threads(1)
    cfg=json.loads((FAMILY/'reproduce/launch.json').read_text())
    for p,expected in cfg['source_hashes'].items():assert sha(p)==expected,p
    loaded=Path(log_fragment_expectation.__code__.co_filename).resolve()
    assert sha(loaded)==cfg['source_hashes'][str(loaded)]
    source=Path(cfg['observation_family'])/'code/observation.py'
    spec=importlib.util.spec_from_file_location('registered_observation_source',source)
    observation=importlib.util.module_from_spec(spec);spec.loader.exec_module(observation)
    old,groups,feature,theta=observation.load_data()
    train=[g for g in groups if g['split']=='train']
    assert len(train)==7 and feature['allocated_dimensions']==38
    assert sum(len(g['rows']) for g in train)==90630
    gamma=theta.new_zeros(feature['allocated_dimensions'])
    initial=torch.cat([theta,gamma]).detach()
    blocks=[]
    for gi,g in enumerate(train):
        for t in sorted(set(g['t'].tolist())):
            idx=torch.nonzero(g['t']==t).flatten()
            blocks.append(dict(block_id=len(blocks),group=gi,condition=g['condition'],time=t,indices=idx,probability=float(g['w'][idx].sum())))
    assert len(blocks)==28 and abs(sum(b['probability'] for b in blocks)-1)<1e-12
    return cfg,observation,groups,train,feature,initial,blocks


def penalty(p,n_response):
    return .005*(p[1:40].square().sum()+p[40:40+n_response].square().sum())


def analytical_amount(theta,g,idx=slice(None)):
    return log_fragment_expectation(g['x']@theta,g['a'][idx],g['b'][idx],g['t'][idx],validate=False)/LN10


def fragment_amount(theta,eta,g,idx,*,check=True):
    times=g['t'][idx]
    assert bool((times==times[0]).all())
    time=float(times[0]);a=g['a'][idx];b=g['b'][idx]-1
    LAST_QUERY.clear();LAST_QUERY.update(condition=g['condition'],split=g['split'],time=time)
    y=expectations(g['x']@theta,eta,time,order=48,substeps=2)
    values=y[a,b]
    if not bool((torch.isfinite(values)&(values>0)).all()):
        raise FloatingPointError('Unresolved required fragment values; no flooring or row exclusion')
    logs=values.log()
    discrepancy=None
    if check:
        with torch.no_grad():
            ref=expectations(g['x']@theta,eta,time,order=48,substeps=4)[a,b]
            if not bool((torch.isfinite(ref)&(ref>0)).all()):
                raise FloatingPointError('Unresolved required four-step reference')
            discrepancy=float((logs.detach()-ref.log()).abs().max())
            if discrepancy>1e-5:
                raise FloatingPointError(f'Required log disagreement {discrepancy} > 1e-5')
    stats=dict(max_log_disagreement=discrepancy,minimum_required_log10=float(logs.detach().min()/LN10),maximum_required_log10=float(logs.detach().max()/LN10),full_negative_count=int((y.detach()<0).sum()))
    return logs/LN10,stats


def block_loss(p,train,n_response,block,*,eta=None,check=True):
    g=train[block['group']];idx=block['indices'];checks=[]
    if eta is None:
        selected=analytical_amount(p[:40],g,idx)
        with torch.no_grad():all_amount=analytical_amount(p[:40],g)
    else:
        # Reuse only the selected block's detached values at this exact state.
        selected,stats=fragment_amount(p[:40],eta,g,idx,check=check)
        checks.append(dict(time=block['time'],**stats))
        with torch.no_grad():
            all_amount=torch.empty_like(g['y']);all_amount[idx]=selected.detach()
            for t in sorted(set(g['t'].tolist())):
                if t==block['time']:continue
                other=torch.nonzero(g['t']==t).flatten()
                amount,stats=fragment_amount(p[:40],eta,g,other,check=check)
                all_amount[other]=amount;checks.append(dict(time=t,**stats))
    with torch.no_grad():
        base=all_amount+g['z']@p[40:40+n_response]
        beta=(g['w']*(g['y']-base)).sum()/g['w'].sum()
        weighted_residual=(g['w']*(base+beta-g['y'])).sum()
    prediction=selected+g['z'][idx]@p[40:40+n_response]+beta
    data=((g['w'][idx]/block['probability'])*(prediction-g['y'][idx]).square()).sum()
    stats=dict(profiled_offset=float(beta),condition_weighted_residual=float(weighted_residual),condition_numeric_checks=checks)
    return data+penalty(p,n_response),data,stats


def full_loss(p,train,n_response):
    data=p.new_zeros(())
    for g in train:
        base=analytical_amount(p[:40],g)+g['z']@p[40:40+n_response]
        residual=base-g['y'];residual=residual-(g['w']*residual).sum()/g['w'].sum()
        data+=(g['w']*residual.square()).sum()
    return data+penalty(p,n_response),data


def independent_values_jacobian(theta,g):
    x=g['x'].numpy();rates=np.exp(x@theta);a=g['a'].numpy();b=g['b'].numpy();t=g['t'].numpy()
    prefix=np.r_[0,np.cumsum(rates)]
    xp=np.vstack([np.zeros((1,40)),np.cumsum(rates[:,None]*x,axis=0)])
    value=-t*(prefix[b-1]-prefix[a]);jac=-t[:,None]*(xp[b-1]-xp[a])
    for mask,index in [(a>0,a-1),(b<=len(rates),b-1)]:
        ix=np.flatnonzero(mask);k=index[ix];h=rates[k]*t[ix]
        value[ix]+=np.log(-np.expm1(-h))
        factor=h*np.exp(-h)/(-np.expm1(-h))
        jac[ix]+=factor[:,None]*x[k]
    return value/LN10,jac/LN10


def independent_profiled(p,train,n_response,block=None):
    total=0.;gradient=np.zeros_like(p)
    selected=train if block is None else [train[block['group']]]
    for g in selected:
        amount,jac=independent_values_jacobian(p[:40],g)
        z=g['z'].numpy();w=g['w'].numpy();y=g['y'].numpy()
        base=amount+z@p[40:40+n_response]
        beta=w@(y-base)/w.sum()
        residual=base+beta-y
        if block is not None:
            ix=block['indices'].numpy();residual=residual[ix];jac=jac[ix];z=z[ix];w=w[ix]/block['probability']
        weighted=2*w*residual
        total+=w@(residual**2)
        gradient[:40]+=weighted@jac
        gradient[40:40+n_response]+=weighted@z
    regularizer=.005*(np.sum(p[1:40]**2)+np.sum(p[40:40+n_response]**2))
    gradient[1:40]+=.01*p[1:40];gradient[40:40+n_response]+=.01*p[40:40+n_response]
    return total+regularizer,gradient
