"""Pre-fit numerical, nesting, feature and sampled-estimator admission."""
import json
import resource
import time

import numpy as np
import torch
from core import AA, FAMILY, amount, dump, independent, load, objective, prepare, sha, variant
from sampled import estimate, toy_check


def main():
    started = time.perf_counter()
    cfg, old, groups, feature, initial = load()
    for path, h in json.loads((FAMILY/'reproduce/execution-code.json').read_text()).items(): assert sha(FAMILY/path)==h
    # Independently indexed sequence features; different construction than producer.
    for g in groups:
        seq = np.array(list(g['sequence'])); idx=np.arange(len(seq)-1)[:,None]+np.array([-2,-1,2,3])[None,:]
        valid=(idx>=0)&(idx<len(seq)); letters=np.full(idx.shape,'_',dtype='<U1'); letters[valid]=seq[idx[valid]]
        reference=(letters[:,:,None]==np.array(list(AA[1:]+'_'))[None,None,:]).astype(float).reshape(len(seq)-1,80)
        np.testing.assert_array_equal(reference,g['raw_rate_extra'])
        raw=np.zeros_like(g['raw_response_extra']);ins=feature['old_response']['instruments'].index(g['instrument'])
        for j,a in enumerate(AA[1:]):
            raw[:,ins*38+j]=[r['peptide_sequence'].startswith(a) for r in g['rows']]
            raw[:,ins*38+19+j]=[r['peptide_sequence'].endswith(a) for r in g['rows']]
        np.testing.assert_array_equal(raw,g['raw_response_extra'])
    dump(FAMILY/'results/features.json',feature)
    records=[];costs={}
    for arm in cfg['exact_arms']:
        gg,theta,gamma=variant(groups,initial,arm);train=[g for g in gg if g['split']=='train']
        # Zero extensions must preserve all old train AND development amounts/response.
        for g,original in zip(gg,groups):
            np.testing.assert_allclose(amount(theta,g).numpy(),old.log_amount(initial,original).numpy(),atol=1e-12,rtol=1e-12)
        for perturb in [0.,.02]:
            p=torch.cat([theta,gamma]);p=p+perturb*torch.sin(torch.arange(1,len(p)+1,dtype=torch.float64));p.requires_grad_()
            value,_=objective(p[:len(theta)],p[len(theta):],train);grad=torch.autograd.grad(value,p)[0]
            ref,rg,_=independent(p[:len(theta)].detach().numpy(),p[len(theta):].detach().numpy(),train)
            np.testing.assert_allclose(float(value.detach()),ref,atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(grad.numpy(),rg,atol=1e-9,rtol=1e-9)
            records.append(dict(arm=arm,perturbation=perturb,objective=ref,maximum_gradient_error=float(np.max(np.abs(grad.numpy()-rg))),parameters=len(p)))
            if arm=='base':
                ov,_=old.objective(p[:40],p[40:],groups);np.testing.assert_allclose(float(ov.detach()),ref,atol=1e-10,rtol=1e-10)
        timing=[]
        for _ in range(3):
            p=torch.cat([theta,gamma]).requires_grad_();tick=time.perf_counter();value,_=objective(p[:len(theta)],p[len(theta):],train);torch.autograd.grad(value,p);timing.append(time.perf_counter()-tick)
        estimate_seconds=max(timing)*cfg['steps']*1.5+30
        costs[arm]=dict(times=timing,estimated_seconds_with_margin=estimate_seconds,cap=cfg['caps']['exact_seconds'],pass_cap=estimate_seconds<=cfg['caps']['exact_seconds'])
        print(json.dumps(dict(arm=arm,parameters=len(theta)+len(gamma),cost=costs[arm])),flush=True)
    toy=toy_check(dict(toy_draws=32768,seed=9142,toy_max_standard_errors=6))
    dump(FAMILY/'results/conditional-toy.json',toy)
    gg,theta,gamma=variant(groups,initial,'both');train=[g for g in gg if g['split']=='train'];sampled=[]
    for perturb in [0.,.02]:
        p=torch.cat([theta,gamma]);p=p+perturb*torch.sin(torch.arange(1,len(p)+1,dtype=torch.float64));p.requires_grad_()
        v,_=objective(p[:len(theta)],p[len(theta):],train);exact=torch.autograd.grad(v,p)[0][:len(theta)]
        with torch.no_grad(): rr=prepare(p[:len(theta)],p[len(theta):],train)
        values=[];ticks=[]
        for rep in range(cfg['hybrid']['groups']):
            gen=torch.Generator().manual_seed(9142+rep+int(perturb>0)*1000);tick=time.perf_counter();gradient,work=estimate(rr,512,gen,cfg['hybrid']['proposal']);ticks.append(time.perf_counter()-tick)
            reg=.01*p[:len(theta)].detach().clone();reg[0]=0;values.append(gradient+reg)
        values=torch.stack(values);norm=float(exact.norm());mean=float((values.mean(0)-exact).norm())/norm;rms=float((values-exact).square().sum(1).mean().sqrt())/norm
        sampled.append(dict(perturbation=perturb,mean_relative_error=mean,rms_relative_error=rms,gradients=values.tolist(),exact=exact.tolist(),times=ticks,pass_gate=mean<=.2 and rms<=.5 and bool(torch.isfinite(values).all())))
        print(json.dumps({k:v for k,v in sampled[-1].items() if k not in ['gradients','exact','times']}),flush=True)
    timings=[]
    for rep in range(3):
        p=torch.cat([theta,gamma]).requires_grad_();tick=time.perf_counter();v,_=objective(p[:len(theta)],p[len(theta):],train);torch.autograd.grad(v,p)
        with torch.no_grad():rr=prepare(p[:len(theta)],p[len(theta):],train);estimate(rr,512,torch.Generator().manual_seed(1200+rep),cfg['hybrid']['proposal'])
        timings.append(time.perf_counter()-tick)
    estimate_seconds=max(timings)*cfg['steps']*1.5+30
    costs['both_hybrid']=dict(times=timings,estimated_seconds_with_margin=estimate_seconds,cap=cfg['caps']['hybrid_seconds'],pass_cap=estimate_seconds<=cfg['caps']['hybrid_seconds'])
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;assert peak<cfg['caps']['peak_rss_bytes']
    result=dict(exact_admitted=all(costs[a]['pass_cap'] for a in cfg['exact_arms']),hybrid_admitted=all(r['pass_gate'] for r in sampled) and costs['both_hybrid']['pass_cap'],numerical_checks=records,costs=costs,sampled_probes=sampled,seconds=time.perf_counter()-started,peak_rss_bytes=peak,source_hashes=cfg['source_hashes'])
    dump(FAMILY/'results/qualification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ['exact_admitted','hybrid_admitted','seconds','peak_rss_bytes','costs']}),flush=True)


if __name__=='__main__':main()
