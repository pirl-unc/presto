"""Production-boundary algebra, gradients and discrete cut-set qualification."""

import itertools
import math
import time

import numpy as np
import torch

from model import ROOT,component,data,decode,dump_json,initial,log_events,masks,objective

assert __import__('json').loads((ROOT/'results/independent-admission.json').read_text())['passed']
datasets,parent,aas,aa_index,sequences=data()
train=datasets['train']
start_clock=time.monotonic()


def spans(length):
    pairs=np.array([(i,j) for i in range(length) for j in range(i+1,length+1)])
    return torch.tensor(pairs[:,0]),torch.tensor(pairs[:,1])


def independent(rates,u,start,end):
    out=[]
    for a,b in zip(start,end,strict=True):
        logp=-u*sum(rates[a:b-1])
        if a>0:logp+=math.log(-math.expm1(-rates[a-1]*u))
        if b<len(rates)+1:logp+=math.log(-math.expm1(-rates[b-1]*u))
        out.append(logp)
    return np.array(out)


small=np.array([.001,.003,.007,.02,.04]);a,b=spans(6)
enumeration=[]
for u in (.001,2.,20.,80.):
    probabilities=np.zeros(len(a));p=-np.expm1(-small*u)
    for cuts in itertools.product((False,True),repeat=5):
        mass=np.prod(np.where(cuts,p,1-p));boundaries=[0,*[i+1 for i,c in enumerate(cuts) if c],6]
        present=set(zip(boundaries[:-1],boundaries[1:],strict=True))
        probabilities+=mass*np.array([(int(i),int(j)) in present for i,j in zip(a,b,strict=True)])
    actual=log_events(torch.tensor(small),torch.full((len(a),),u),a,b).exp().numpy()
    error=float(np.max(np.abs(actual-probabilities)));assert error<1e-12
    assert np.max(np.abs(log_events(torch.tensor(small),torch.full((len(a),),u),a,b).numpy()-independent(small,u,a.numpy(),b.numpy())))<1e-12
    enumeration.append({'exposure':u,'max_error':error})
full_a,full_b=spans(len(parent));length=(full_b-full_a).numpy();mass_checks=[]
for rate in (1e-7,.001,.1,1.):
    for u in (.001,10.,100.,240.):
        rates=torch.full((len(parent)-1,),rate)
        counts=log_events(rates,torch.full((len(full_a),),u),full_a,full_b).exp().numpy()
        mass=float(counts@length);number=float(counts.sum());expected=1+(len(parent)-1)*(-math.expm1(-rate*u))
        assert abs(mass-len(parent))<1e-10 and abs(number-expected)<1e-10
        mass_checks.append({'rate':rate,'exposure':u,'mass_error':mass-len(parent),'count_error':number-expected})
gradients=[];score_checks=[]
for kind in ('uniform','residue'):
    for perturbed in (False,True):
        theta=initial(kind,len(aas))
        if perturbed:
            with torch.no_grad():theta.add_(.2*torch.sin(torch.arange(len(theta))))
        exact=torch.autograd.grad(objective(theta,kind,train,aa_index)[0],theta)[0].numpy()
        finite=[]
        for k in range(len(theta)):
            plus=theta.detach().clone();minus=theta.detach().clone();plus[k]+=1e-5;minus[k]-=1e-5
            finite.append(float((objective(plus,kind,train,aa_index)[0]-objective(minus,kind,train,aa_index)[0])/2e-5))
        error=float(np.max(np.abs(exact-finite)));assert error<1e-6
        gradients.append({'kind':kind,'perturbed':perturbed,'max_abs_error':error})
        rates,tau,_=decode(theta,kind,aa_index)
        expected=np.array([independent(rates.detach().numpy(),float(t+tau),np.array([int(a)]),np.array([int(b)]))[0]
                           +float((t+tau)*rates.sum()) for t,a,b in zip(train['time'],train['start'],train['end'],strict=True)])/math.log(10)
        assert np.max(np.abs(component(theta,kind,train,aa_index).detach().numpy()-expected))<1e-10
        if perturbed:continue
        for draws in (16,64):
            generator=torch.Generator().manual_seed(701+draws)
            samples=[]
            for repeat in range(128):
                loss=objective(theta,kind,train,aa_index,draws,generator)[0]
                samples.append(torch.autograd.grad(loss,theta)[0].numpy())
            samples=np.array(samples);mean=samples.mean(0);sem=samples.std(0,ddof=1)/math.sqrt(len(samples))
            z=np.abs(mean-exact)/(sem+1e-10);assert z.max()<6
            score_checks.append({'kind':kind,'draws':draws,'repeats':128,'max_sem_units':float(z.max()),
                                 'relative_mean_error':float(np.linalg.norm(mean-exact)/np.linalg.norm(exact)),
                                 'mean_gradient':mean.tolist(),'exact_gradient':exact.tolist(),'sem':sem.tolist()})
            print('qualified',kind,draws,score_checks[-1]['max_sem_units'],flush=True)
        internal,boundary=masks(train['start'],train['end'],len(parent))
        for compact in (True,False):
            result,means=log_events(rates,train['time']+tau,train['start'],train['end'],16,torch.Generator().manual_seed(444),not compact,True)
            assert torch.all(means[internal]==0) and torch.all(means[boundary]==1)
            assert torch.all((means>=0)&(means<=1))
            assert torch.max(torch.abs(result-log_events(rates,train['time']+tau,train['start'],train['end'])))<1e-12
unconditional=[];rng=np.random.default_rng(90210)
observed={(int(a),int(b)) for a,b in zip(train['start'],train['end'],strict=True)}
observed_indices=np.array([i for i,p in enumerate(zip(full_a.tolist(),full_b.tolist(),strict=True)) if p in observed])
for u in (10.,90.,130.):
    expected=log_events(torch.full((len(parent)-1,),.001),torch.full((len(full_a),),u),full_a,full_b).exp().numpy()
    for n in (256,2048,16384):
        before=time.monotonic();cuts=rng.random((n,len(parent)-1)) < -math.expm1(-.001*u)
        counts=np.zeros(len(full_a))
        for cut in cuts:
            edges=np.r_[0,np.flatnonzero(cut)+1,len(parent)];starts=edges[:-1];ends=edges[1:]
            assert np.diff(edges).sum()==len(parent)
            ids=starts*len(parent)-starts*(starts-1)//2+(ends-starts-1)
            counts[ids]+=1
        measured=counts/n;assert abs(measured@length-len(parent))<1e-10
        unconditional.append({'exposure':u,'draws':n,'max_abs_count_error':float(np.max(np.abs(measured-expected))),
                              'mean_abs_count_error':float(np.mean(np.abs(measured-expected))),
                              'observed_event_zero_hit_fraction':float(np.mean(counts[observed_indices]==0)),
                              'seconds':time.monotonic()-before})
result={'passed':True,'short_chain_enumeration':enumeration,'full_span_mass_checks':mass_checks,
        'finite_differences':gradients,'conditional_score_checks':score_checks,'unconditional_counts':unconditional,
        'seconds':time.monotonic()-start_clock,'n_parent_bonds':len(parent)-1,'n_spans':len(full_a),'residue_categories':aas}
dump_json(ROOT/'results/qualification.json',result)
print('qualification passed',result['seconds'],flush=True)
