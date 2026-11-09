"""Prospective toy/actual estimator identities, eta-zero parity and target-free cost."""
import json
import resource
import sys
import time
import numpy as np
import torch
from core import FAMILY,block_loss,dump,full_loss,independent_profiled,load,offset_optimum,sha,fragment_amount


def toy():
    x=torch.tensor([[1.,-1.,.3],[1.,2.,-.8],[1.,.5,.4],[1.,-3.,1.1],[1.,.2,-.7],[1.,1.2,.1],[1.,-.4,.6],[1.,.8,-.2],[1.,1.5,.9]],dtype=torch.float64)
    z=x[:,1:].square();w=torch.arange(1.,10.,dtype=torch.float64);w/=w.sum()
    c=torch.tensor([0,0,0,1,1,1,2,2,2]);blocks=torch.tensor([0,0,1,2,2,3,4,4,4]);y=torch.sin(torch.arange(9.,dtype=torch.float64))
    def base(p):
        a=x@p[:3]
        return a+.2*a.square()+z@p[3:5]
    p=torch.tensor([.2,-.1,.4,.3,-.2],dtype=torch.float64,requires_grad=True)
    b=base(p);beta=offset_optimum(b,y,w,c,3)
    full=(w*(b+beta[c]-y).square()).sum();gradient=torch.autograd.grad(full,p)[0]
    expected_loss=0.;expected_gradient=torch.zeros_like(p);wrong_gradient=torch.zeros_like(p)
    for k in range(5):
        ix=blocks==k;prob=w[ix].sum();q=p.detach().clone().requires_grad_();v=base(q)
        solved=offset_optimum(v.detach(),y,w,c,3)
        loss=((w[ix]/prob)*(v[ix]+solved[c[ix]]-y[ix]).square()).sum()
        expected_loss+=float(prob*loss.detach());expected_gradient+=prob*torch.autograd.grad(loss,q)[0]
        q=p.detach().clone().requires_grad_();v=base(q)[ix];offset=(w[ix]*(y[ix]-v)).sum()/w[ix].sum()
        wrong=((w[ix]/prob)*(v+offset-y[ix]).square()).sum();wrong_gradient+=prob*torch.autograd.grad(wrong,q)[0]
    np.testing.assert_allclose(expected_loss,float(full.detach()),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(expected_gradient,gradient,atol=1e-12,rtol=1e-12)
    counterexample=float((wrong_gradient-gradient).abs().max());assert counterexample>1e-4
    return dict(passed=True,loss=float(full.detach()),expected_gradient_error=float((expected_gradient-gradient).abs().max()),incorrect_timewise_profile_gradient_error=counterexample)


def main():
    started=time.perf_counter()
    for rel,h in json.loads((FAMILY/'reproduce/qualification-code.json').read_text()).items():assert sha(FAMILY/rel)==h,rel
    toy_result=toy();cfg,obs,groups,train,feature,initial,blocks=load();m=feature['allocated_dimensions'];assert initial.numel()==78
    states=[initial,initial+.04*torch.sin(torch.arange(1,79,dtype=initial.dtype))];records=[];arrays={}
    for si,state in enumerate(states):
        p=state.clone().requires_grad_();loss,data=full_loss(p,train,m);gradient=torch.autograd.grad(loss,p)[0]
        theta=state[:40].clone().requires_grad_();gamma=state[40:78].clone().requires_grad_();oldloss,_=obs.objective(theta,gamma,train)
        oldgrad=torch.cat(torch.autograd.grad(oldloss,(theta,gamma)))
        independent,ig=independent_profiled(state.numpy(),train,m)
        np.testing.assert_allclose(float(loss.detach()),[float(oldloss.detach()),independent],atol=1e-9,rtol=1e-8)
        np.testing.assert_allclose(gradient,oldgrad,atol=1e-9,rtol=1e-8);np.testing.assert_allclose(gradient,ig,atol=1e-9,rtol=1e-8)
        expected_loss=0.;expected_gradient=torch.zeros_like(p);br=[]
        for b in blocks:
            q=state.clone().requires_grad_();value,d,stats=block_loss(q,train,m,b);grad=torch.autograd.grad(value,q)[0]
            il,igg=independent_profiled(state.numpy(),train,m,b)
            np.testing.assert_allclose(float(value.detach()),il,atol=1e-9,rtol=1e-8);np.testing.assert_allclose(grad,igg,atol=1e-9,rtol=1e-8)
            expected_loss+=b['probability']*float(value.detach());expected_gradient+=b['probability']*grad
            br.append(dict(block_id=b['block_id'],loss=float(value.detach()),gradient=grad.tolist(),stats=stats))
        np.testing.assert_allclose(expected_loss,float(loss.detach()),atol=1e-9,rtol=1e-8);np.testing.assert_allclose(expected_gradient,gradient,atol=1e-9,rtol=1e-8)
        records.append(dict(state=si,full_loss=float(loss.detach()),data_loss=float(data.detach()),envelope_gradient_error=float((gradient-oldgrad).abs().max()),independent_gradient_error=float(np.max(np.abs(gradient.numpy()-ig))),expected_gradient_error=float((expected_gradient-gradient).abs().max()),blocks=br))
        arrays[f'state{si}']=state.numpy();arrays[f'gradient{si}']=gradient.numpy()
    dump(FAMILY/'results/actual-block-qualification.json',records)
    q=torch.cat([initial,initial.new_zeros(1)]).requires_grad_();tick=time.perf_counter()
    fl,_,stats=block_loss(q[:78],train,m,blocks[0],eta=torch.tanh(q[-1]));fg=torch.autograd.grad(fl,q)[0]
    np.testing.assert_allclose(float(fl.detach()),records[0]['blocks'][0]['loss'],atol=1e-9,rtol=1e-8)
    expected=np.array(records[0]['blocks'][0]['gradient']);np.testing.assert_allclose(fg[:78],expected,atol=1e-9,rtol=1e-8)
    assert torch.isfinite(fg[-1]) and abs(float(fg[-1]))>1e-10
    parity=dict(block_id=0,seconds=time.perf_counter()-tick,loss=float(fl.detach()),common_gradient_error=float(np.max(np.abs(fg[:78].numpy()-expected))),eta_gradient=float(fg[-1]),stats=stats)
    arrays['fragment_parity_gradient']=fg.numpy()
    # Target-free timing: all times for the longest parent; selected last time has
    # autograd, detached other times, then selected postcheck at unchanged parameters.
    gi=max(range(len(train)),key=lambda i:len(train[i]['sequence']));g=train[gi];times=sorted(set(g['t'].tolist()));t=times[-1];tick=time.perf_counter()
    q=torch.cat([initial,initial.new_zeros(1)]).requires_grad_();ix=torch.nonzero(g['t']==t).flatten()
    amount,stats=fragment_amount(q[:40],torch.tanh(q[-1]),g,ix,check=True);cost_checks=[dict(time=t,**stats)]
    with torch.no_grad():
        for other in times[:-1]:
            idx=torch.nonzero(g['t']==other).flatten();_,st=fragment_amount(q[:40],torch.tanh(q[-1]),g,idx,check=True);cost_checks.append(dict(time=other,**st))
    amount.mean().backward()
    with torch.no_grad():_,post=fragment_amount(q[:40],torch.tanh(q[-1]),g,ix,check=True)
    cost=dict(seconds=time.perf_counter()-tick,condition=g['condition'],length=len(g['sequence']),times=times,selected_time=t,checks=cost_checks,post=post,scope='No measured target consulted by this forward/backward cost probe; no optimizer update.')
    np.savez_compressed(FAMILY/'results/qualification.npz',**arrays)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result=dict(all_pass=True,optimizer_updates=0,toy=toy_result,n_training=90630,n_common_parameters=78,n_conditions=7,n_blocks=28,state_summaries=[{k:v for k,v in r.items() if k!='blocks'} for r in records],fragment_parity=parity,cost=cost,seconds=time.perf_counter()-started,peak_rss_bytes=rss if sys.platform=='darwin' else rss*1024)
    dump(FAMILY/'results/qualification-summary.json',result);dump(FAMILY/'results/response-features.json',feature);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
