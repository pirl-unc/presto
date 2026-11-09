"""Frozen source/exposure, mathematical, cost and gradient qualification."""
import itertools
import json
import math
import resource
import time
import numpy as np
import torch
from scipy.integrate import quad
from core import (FAMILY, audit, cfg, contains, draw, dump, load, objective_torch,
                  physical_numpy, physical_torch, prepare, proposal, sampled,
                  sha, spans_terms, torch_data)


def math_check(c):
    results=[]; tol=c['math_tolerances']
    for length in [2,4,6]:
        rate=np.array([.07,.13,.21,.1,.15])[:length-1]
        states=np.array(list(itertools.product([False,True],repeat=length-1)))
        codes={tuple(s):i for i,s in enumerate(states)}; q=np.zeros((len(states),len(states)))
        for i,state in enumerate(states):
            for j,k in enumerate(rate):
                if not state[j]:
                    after=state.copy();after[j]=True;q[i,codes[tuple(after)]]+=k;q[i,i]-=k
        starts=np.array([a for a in range(length) for b in range(a+1,length+1)])
        ends=np.array([b for a in range(length) for b in range(a+1,length+1)])
        indicator=contains(states,starts,ends).astype(float);terms=spans_terms(rate,starts,ends)
        for delta in [1e-4,.01,1.,100.]:
            ld=math.log(delta);lv,d=physical_numpy(ld,terms)
            initial=np.zeros(len(states));initial[0]=1
            occupancy=np.linalg.solve((delta*np.eye(len(states))-q).T,initial)
            ref=occupancy@indicator
            integral=[]
            for a,b in zip(starts,ends):
                def integrand(t):
                    v=math.exp(-(delta+rate[a:b-1].sum())*t)
                    if a:v*=-math.expm1(-rate[a-1]*t)
                    if b<length:v*=-math.expm1(-rate[b-1]*t)
                    return v
                integral.append(quad(integrand,0,np.inf,epsabs=1e-12,epsrel=1e-10,limit=300)[0])
            value=np.exp(lv)
            assert np.allclose(value,ref,rtol=tol['value_relative'],atol=tol['value_absolute'])
            assert np.allclose(value,integral,rtol=tol['value_relative'],atol=tol['value_absolute'])
            mass=float((ends-starts)@value);assert abs(mass/(length/delta)-1)<=tol['residue_relative']
            plus=physical_numpy(ld+1e-5,terms)[0];minus=physical_numpy(ld-1e-5,terms)[0]
            finite=(plus-minus)/2e-5
            assert np.allclose(d,finite,rtol=tol['gradient_relative'],atol=tol['gradient_absolute'])
            z=torch.tensor(ld,dtype=torch.float64,requires_grad=True)
            tv=physical_torch(z,tuple(torch.tensor(t) for t in terms))
            td=np.array([float(torch.autograd.grad(v,z,retain_graph=True)[0]) for v in tv])
            assert np.allclose(d,td,rtol=tol['gradient_relative'],atol=tol['gradient_absolute'])
            results.append(dict(length=length,delta=delta,maximum_resolvent_relative_error=float(np.max(abs(value/ref-1))),maximum_integral_error=float(np.max(abs(value-np.array(integral)))),residue_relative_error=abs(mass/(length/delta)-1),maximum_fd_error=float(np.max(abs(d-finite)))))
    return results


def toy_check(c):
    rate=np.array([.07,.13,.21,.1,.15]);a=np.array([0,1,2]);b=np.array([3,4,6]);delta=.03
    terms=spans_terms(rate,a,b);lv,d=physical_numpy(math.log(delta),terms)
    record=dict(rate=rate,starts=a,ends=b,terms=terms,logf=lv,logdelta=math.log(delta),probability=delta*np.exp(lv),derivative=d,centered_derivative=d-d.mean(),baseline=float(d.mean()),coefficient=np.array([.3,-.7,.4]))
    w=proposal(record)
    states=np.array(list(itertools.product([False,True],repeat=5)));indicator=contains(states,a,b)
    # Independent marginalized joint distribution via the branching generator.
    codes={tuple(s):i for i,s in enumerate(states)};q=np.zeros((32,32))
    for i,state in enumerate(states):
        for j,k in enumerate(rate):
            if not state[j]:
                after=state.copy();after[j]=True;q[i,codes[tuple(after)]]+=k;q[i,i]-=k
    initial=np.zeros(32);initial[0]=delta
    base=np.linalg.solve((delta*np.eye(32)-q).T,initial)
    density=indicator@(w/record['probability']);prob=base*density;valid=prob>0
    assert abs(prob.sum()-1)<1e-10
    estimate=(prob[valid]/density[valid])@(indicator[valid]@(record['coefficient']*record['centered_derivative']/record['probability']))+record['baseline']*record['coefficient'].sum()
    exact=record['coefficient']@d;assert abs(estimate-exact)<=c['math_tolerances']['toy_gradient_absolute']
    n=c['math_tolerances']['toy_draws'];gen=np.random.default_rng(c['seed']);cuts,chosen,age=draw(record,n,gen,w)
    observed=np.bincount(cuts.astype(int)@np.array([16,8,4,2,1]),minlength=32)
    standard=np.max(abs(observed-n*prob)/np.sqrt(n*prob*(1-prob)+1))
    assert standard<=c['math_tolerances']['toy_max_standard_errors']
    assert contains(cuts,a,b)[np.arange(n),chosen].all()
    age_checks=[]
    for j in range(len(a)):
        ages=age[chosen==j];ref=-d[j]/delta
        se=ages.std(ddof=1)/math.sqrt(len(ages));z=abs(ages.mean()-ref)/se
        assert z<=c['math_tolerances']['toy_max_standard_errors']
        age_checks.append(dict(span=[int(a[j]),int(b[j])],n=len(ages),expected_mean=ref,observed_mean=float(ages.mean()),standard_errors=z))
    return dict(expected_gradient=float(exact),gradient_error=float(abs(estimate-exact)),probabilities=prob.tolist(),counts=observed.tolist(),maximum_count_standard_errors=float(standard),ages=age_checks)


def main():
    start=time.perf_counter();c=cfg();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    receipt=json.loads((FAMILY/'reproduce/execution-code.json').read_text())
    for p,h in receipt['files'].items():assert sha(FAMILY/p)==h,p
    assert not (FAMILY/'results/qualification.json').exists()
    exposure=audit();print('source/exposure',exposure['summary'],flush=True)
    mathrows=math_check(c);dump(FAMILY/'results/math.json',mathrows)
    toy=toy_check(c);dump(FAMILY/'results/toy.json',toy);print('mathematical checks pass',flush=True)
    costs=[];caps={};groups_by_arm={}
    for arm in c['exact_arms']:
        groups=load(arm,synthetic=True);data=torch_data(groups,synthetic=True);times=[]
        for rep in range(c['preflight']['repetitions']):
            now=time.perf_counter();z=torch.tensor(math.log(.01),dtype=torch.float64,requires_grad=True);objective_torch(z,data).backward();times.append(time.perf_counter()-now)
        projected=math.ceil(max(times)*c['steps']*c['preflight']['cost_margin']+60)
        cap=max(c['preflight']['exact_cap_floor_seconds'],projected)
        caps[arm]=dict(seconds=cap,admitted=cap<=c['preflight']['exact_cap_ceiling_seconds'])
        costs.append(dict(arm=arm,repetition_seconds=times,projected=projected,**caps[arm]))
        groups_by_arm[arm]=groups
    _,_,records=prepare(math.log(.01),groups_by_arm['wider_rates'],synthetic=True)
    times=[]
    for rep in range(c['preflight']['repetitions']):
        now=time.perf_counter();sampled(records,c['sample_count'],np.random.default_rng(c['seed']+rep));times.append(time.perf_counter()-now)
    # Include full analytical objective/preparation cost with the synthetic target.
    now=time.perf_counter();prepare(math.log(.01),groups_by_arm['wider_rates'],synthetic=True);prepare_seconds=time.perf_counter()-now
    projected=math.ceil((max(times)+prepare_seconds)*c['steps']*c['preflight']['cost_margin']+60)
    cap=max(c['preflight']['hybrid_cap_floor_seconds'],projected)
    caps['wider_hybrid']=dict(seconds=cap,admitted=cap<=c['preflight']['hybrid_cap_ceiling_seconds'])
    costs.append(dict(arm='wider_hybrid',repetition_seconds=times,prepare_seconds=prepare_seconds,projected=projected,**caps['wider_hybrid']))
    dump(FAMILY/'results/preflight.json',costs);print('cost admission',caps,flush=True)
    probes=[];parity=[]
    for arm in c['exact_arms']:
        groups=load(arm);data=torch_data(groups)
        for offset in [0,-1,1]:
            ld=math.log(.01)+offset;loss,gradient,records=prepare(ld,groups)
            z=torch.tensor(ld,dtype=torch.float64,requires_grad=True);tl=objective_torch(z,data);tg=float(torch.autograd.grad(tl,z)[0])
            assert abs(float(tl)-loss)<=1e-10 and abs(tg-gradient)<=1e-10
            parity.append(dict(arm=arm,offset=offset,loss_error=abs(float(tl)-loss),gradient_error=abs(tg-gradient),gradient=gradient))
            if arm!='wider_rates':continue
            values=[];t0=time.perf_counter()
            for rep in range(c['gradient_groups']):
                g,_=sampled(records,c['sample_count'],np.random.default_rng(c['seed']+rep+1000*(offset+1)));values.append(g)
                assert time.perf_counter()-start<c['preflight']['qualification_seconds']
            values=np.array(values);mean=abs(values.mean()-gradient);rms=float(np.sqrt(np.mean((values-gradient)**2)))
            near=abs(gradient)<=c['near_zero_gradient']
            passed=(mean<=c['near_zero_mean_absolute_error'] and rms<=c['near_zero_rms_absolute_error']) if near else (mean/abs(gradient)<=c['mean_relative_gate'] and rms/abs(gradient)<=c['rms_relative_gate'])
            probes.append(dict(offset=offset,exact_gradient=gradient,estimates=values.tolist(),mean_absolute_error=float(mean),rms_absolute_error=rms,near_zero=near,mean_relative_error=float(mean/abs(gradient)) if gradient else None,rms_relative_error=rms/abs(gradient) if gradient else None,gate_pass=bool(passed),seconds=time.perf_counter()-t0))
            dump(FAMILY/'results/gradient-probes.json',probes);print('gradient probe',offset,passed,mean,rms,flush=True)
    dump(FAMILY/'results/exact-parity.json',parity)
    caps['wider_hybrid']['admitted'] &= all(p['gate_pass'] for p in probes)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rss<c['preflight']['peak_rss_bytes']
    summary=dict(status='completed',seconds=time.perf_counter()-start,peak_rss_bytes=rss,math_pass=True,source_pass=True,hybrid_gradient_gate_pass=all(p['gate_pass'] for p in probes),fit_caps=caps)
    dump(FAMILY/'results/qualification.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
