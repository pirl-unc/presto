"""Independent CTMC, complete-path scores, and real-training numerical checks."""

import json
import math
import resource
import time

import numpy as np
import torch
from scipy.linalg import expm, expm_frechet

from common import FAMILY, designs, dump, entry_objective, load_data, prepare_points
from presto.models.independent_cut import partition_contains
from presto.models.substrate_entry import log_entry_fragment_expectation, sample_entry_partitions


def ctmc(logs, log_entry, time_hours):
    n=len(logs);states=2**n;generator=np.zeros((states+1,states+1))
    entry=np.exp(log_entry);rates=np.exp(logs)
    generator[0,1]=entry;generator[0,0]=-entry
    directions=[]
    for bond,rate in enumerate(rates):
        d=np.zeros_like(generator)
        for mask in range(states):
            if not mask&(1<<bond):
                d[1+mask,1+(mask|(1<<bond))]+=rate;d[1+mask,1+mask]-=rate
        generator+=d;directions.append(d)
    d=np.zeros_like(generator);d[0,1]=entry;d[0,0]=-entry;directions.append(d)
    spans=[(a,b) for a in range(n+1) for b in range(a+1,n+2)]
    indicators=np.zeros((states+1,len(spans)));indicators[0,spans.index((0,n+1))]=1
    for mask in range(states):
        cuts=[bool(mask&(1<<j)) for j in range(n)]
        for column,(a,b) in enumerate(spans):
            indicators[1+mask,column]=((a==0 or cuts[a-1]) and (b==n+1 or cuts[b-1]) and not any(cuts[a:b-1]))
    transition=expm(generator*time_hours)
    amounts=transition[0]@indicators
    gradients=np.stack([expm_frechet(generator*time_hours,d*time_hours,compute_expm=False)[0]@indicators for d in directions])
    return spans,amounts,gradients


def main():
    started=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg,groups=load_data();groups=[prepare_points(g) for g in groups]
    assert not (FAMILY/'results/qualification.json').exists()
    math_rows=[]
    logs=np.log([.08,.18,.45,.9,1.4])
    for entry in (.03,.3,1.,5.):
        for t in (.1,.7,2.,6.,20.):
            spans,expected,jac=ctmc(logs,math.log(entry),t)
            starts=torch.tensor([a for a,b in spans]);ends=torch.tensor([b for a,b in spans]);times=torch.full((len(spans),),t,dtype=torch.float64)
            for order in (16,32,64,128):
                theta=torch.tensor([*logs,math.log(entry)],dtype=torch.float64,requires_grad=True)
                values=log_entry_fragment_expectation(theta[:-1],theta[-1],starts,ends,times,order=order).exp()
                coefficients=torch.linspace(.2,1.3,len(spans),dtype=torch.float64)
                gradient=torch.autograd.grad((values*coefficients).sum(),theta)[0].detach().numpy()
                mass=float((values*(ends-starts)).sum().detach())
                row=dict(entry_rate=entry,time=t,order=order,max_amount_error=float(np.max(np.abs(values.detach().numpy()-expected))),
                         max_log_amount_error=float(np.max(np.abs(np.log(values.detach().numpy())-np.log(expected)))),
                         max_gradient_error=float(np.max(np.abs(gradient-jac@coefficients.numpy()))),mass_error=abs(mass-6))
                math_rows.append(row)
    dump(FAMILY/'results/ctmc-qualification.json',math_rows)
    assert max(r['max_log_amount_error'] for r in math_rows if r['order']==128)<1e-6
    assert max(r['max_gradient_error'] for r in math_rows if r['order']==128)<1e-7
    assert max(r['mass_error'] for r in math_rows)<1e-11
    # Censored entry paths and full cut scores, using an independent CTMC gradient.
    spans,expected,jac=ctmc(logs,math.log(.3),2.)
    starts=torch.tensor([a for a,b in spans]);ends=torch.tensor([b for a,b in spans]);coeff=torch.linspace(.2,1.3,len(spans),dtype=torch.float64)
    estimates=[];amounts=[]
    for seed in range(32):
        theta=torch.tensor([*logs,math.log(.3)],dtype=torch.float64,requires_grad=True)
        draw=sample_entry_partitions(theta[:-1],theta[-1],2.,4096,generator=torch.Generator().manual_seed(20261023+seed))
        present=partition_contains(draw.cuts,starts,ends).double()
        assert torch.equal(present@(ends-starts).double(),torch.full((4096,),6.,dtype=torch.float64))
        reward=present@coeff;gradient=torch.autograd.grad((reward.detach()*draw.log_prob).mean(),theta)[0]
        estimates.append(gradient.tolist());amounts.append(present.mean(0).tolist())
    estimates=np.array(estimates);target=jac@coeff.numpy();se=estimates.std(0,ddof=1)/np.sqrt(len(estimates));z=np.abs(estimates.mean(0)-target)/se
    score=dict(estimates=estimates.tolist(),reference=target.tolist(),coordinate_standard_errors=se.tolist(),mean_z_scores=z.tolist(),mass_estimates=amounts,ctmc_amounts=expected.tolist(),seed_start=20261023,samples_per_group=4096,groups=32)
    dump(FAMILY/'results/path-score-qualification.json',score);assert z.max()<6
    # Numerical domain includes only training spans and known training parameter states.
    old=json.loads(open(cfg['previous_family']+'/results/fits.json').read())
    design,basis,audit=designs(groups,'sequence_context');assert torch.equal(basis,torch.eye(40,dtype=torch.float64))
    dump(FAMILY/'results/feature-audit.json',audit)
    numerical=[];timings=[]
    train=[(g,x) for g,x in zip(groups,design) if g['split']=='train']
    for state,weights in [('initial',old['sequence_context']['checkpoints']['0']),('previous_fit',old['sequence_context']['coefficients'])]:
        for entry in (.01,.1,1.,10.):
            records={}
            for order in (16,32,64,128,256):
                theta=torch.tensor([*weights,math.log(entry)],dtype=torch.float64,requires_grad=True);tick=time.perf_counter()
                loss,_=entry_objective(theta,groups,design,entry=True,order=order)
                gradient=torch.autograd.grad(loss,theta)[0]
                values=[]
                with torch.no_grad():
                    for g,x in train:
                        values.extend(log_entry_fragment_expectation(x@theta[:-1],theta[-1],g['point_starts'],g['point_ends'],g['point_times'],order=order,validate=False).tolist())
                records[order]=dict(loss=float(loss.detach()),gradient=gradient.tolist(),log_amounts=values)
                timings.append(dict(state=state,entry_rate=entry,order=order,forward_backward_and_point_check_seconds=time.perf_counter()-tick))
            reference=records[256]
            for order,row in records.items():
                numerical.append(dict(state=state,entry_rate=entry,order=order,objective=row['loss'],
                    max_log_amount_difference=float(np.max(np.abs(np.array(row['log_amounts'])-reference['log_amounts']))),
                    max_gradient_difference=float(np.max(np.abs(np.array(row['gradient'])-reference['gradient'])))))
            print(state,entry,[(r['order'],r['max_log_amount_difference']) for r in numerical if r['state']==state and r['entry_rate']==entry],flush=True)
    eligible=[q for q in cfg['quadrature_candidates'] if all(r['max_log_amount_difference']<cfg['log_amount_tolerance'] and r['max_gradient_difference']<cfg['gradient_tolerance'] for r in numerical if r['order']==q)]
    dump(FAMILY/'results/training-numerics.json',numerical);dump(FAMILY/'results/numerical-runtime.json',timings)
    assert eligible,'No qualified declared quadrature order; preserve and replan'
    result=dict(passed=True,selected_order=min(eligible),ctmc_max_error128=max(r['max_log_amount_error'] for r in math_rows if r['order']==128),
        score_mean_max_z=float(z.max()),test_not_evaluated=True,training_span_times=sum(len(g['point_starts']) for g,x in train),
        wall_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(FAMILY/'results/qualification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
