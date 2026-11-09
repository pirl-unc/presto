"""Verify the unchanged objective before any biological optimization."""
import json
import resource
import sys
import time
import numpy as np
import torch
from core import FAMILY,analytical_amount,block_loss,dump,full_loss,independent_joint,load,offset_optimum,sha


def toy():
    x=torch.tensor([[1.,-1.,.3],[1.,2.,-.8],[1.,.5,.4],[1.,-3.,1.1],[1.,.2,-.7],[1.,1.2,.1],[1.,-.4,.6],[1.,.8,-.2],[1.,1.5,.9]],dtype=torch.float64)
    z=x[:,1:].square();w=torch.arange(1.,10.,dtype=torch.float64);w/=w.sum()
    c=torch.tensor([0,0,0,1,1,1,2,2,2]);block=torch.tensor([0,0,1,2,2,3,4,4,4])
    y=torch.sin(torch.arange(9.,dtype=torch.float64));p0=torch.tensor([.2,-.1,.4,.3,-.2],dtype=torch.float64)
    def base(p):
        a=x@p[:3]
        return a+.2*a.square()+z@p[3:5]
    beta=offset_optimum(base(p0),y,w,c,3).detach()
    p=torch.cat([p0,beta]).requires_grad_()
    joint=(w*(base(p)+p[5:][c]-y).square()).sum()
    grad=torch.autograd.grad(joint,p)[0]
    q=p0.clone().requires_grad_();b=offset_optimum(base(q),y,w,c,3)
    profiled=(w*(base(q)+b[c]-y).square()).sum();pg=torch.autograd.grad(profiled,q)[0]
    np.testing.assert_allclose(float(joint.detach()),float(profiled.detach()),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(grad[:5],pg,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(grad[5:],0,atol=1e-12)
    shift=torch.tensor([.7,-.2,.1],dtype=torch.float64)
    raised=(w*(base(p0)+(beta+shift)[c]-y).square()).sum()
    expected=(w*shift[c].square()).sum()
    np.testing.assert_allclose(float(raised-joint.detach()),float(expected),rtol=1e-12,atol=1e-12)
    weighted_loss=0.;weighted_grad=torch.zeros_like(p)
    for k in range(5):
        ids=block==k;prob=w[ids].sum();q=p.detach().clone().requires_grad_()
        loss=((w[ids]/prob)*(base(q)[ids]+q[5:][c[ids]]-y[ids]).square()).sum()
        weighted_loss+=float(prob*loss.detach());weighted_grad+=prob*torch.autograd.grad(loss,q)[0]
    np.testing.assert_allclose(weighted_loss,float(joint.detach()),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(weighted_grad,grad,rtol=1e-12,atol=1e-12)
    return dict(profiled_loss=float(profiled.detach()),shifted_offset_loss_increase=float(raised-joint.detach()),expected_increase=float(expected),envelope_max_error=float((grad[:5]-pg).abs().max()),block_max_gradient_error=float((weighted_grad-grad).abs().max()),passed=True)


def main():
    start=time.perf_counter()
    frozen=json.loads((FAMILY/'reproduce/qualification-code.json').read_text())
    for rel,expected in frozen.items():assert sha(FAMILY/rel)==expected,rel
    toy_result=toy()
    cfg,observation,groups,train,feature,initial,blocks=load();m=feature['allocated_dimensions']
    assert initial.numel()==85
    atol=cfg['gates']['algebra_atol'];rtol=cfg['gates']['algebra_rtol']
    p=initial.clone().requires_grad_();loss,data=full_loss(p,train,m);gradient=torch.autograd.grad(loss,p)[0]
    theta=initial[:40].clone().requires_grad_();gamma=initial[40:40+m].clone().requires_grad_()
    ploss,pdata=observation.objective(theta,gamma,train)
    pg=torch.autograd.grad(ploss,(theta,gamma));profile_gradient=torch.cat(pg)
    np.testing.assert_allclose(float(loss.detach()),float(ploss.detach()),atol=atol,rtol=rtol)
    np.testing.assert_allclose(gradient[:40+m],profile_gradient,atol=atol,rtol=rtol)
    np.testing.assert_allclose(gradient[40+m:],0,atol=atol,rtol=0)
    independent_loss,independent_gradient=independent_joint(initial.numpy(),train,m)
    np.testing.assert_allclose(independent_loss,float(loss.detach()),atol=atol,rtol=rtol)
    np.testing.assert_allclose(independent_gradient,gradient.numpy(),atol=atol,rtol=rtol)
    weighted_loss=0.;weighted_gradient=torch.zeros_like(initial);records=[]
    for b in blocks:
        q=initial.clone().requires_grad_();value,d,_=block_loss(q,train,m,b)
        grad=torch.autograd.grad(value,q)[0]
        weighted_loss+=b['probability']*float(value.detach());weighted_gradient+=b['probability']*grad
        records.append(dict(block_id=b['block_id'],condition=b['condition'],time=b['time'],n_rows=len(b['indices']),probability=b['probability'],loss=float(value.detach()),data_loss=float(d.detach()),gradient=grad.tolist()))
    np.testing.assert_allclose(weighted_loss,float(loss.detach()),atol=atol,rtol=rtol)
    np.testing.assert_allclose(weighted_gradient,gradient,atol=atol,rtol=rtol)
    dump(FAMILY/'results/block-qualification.json',records)
    np.savez_compressed(FAMILY/'results/objective-qualification.npz',initial=initial.numpy(),full_gradient=gradient.numpy(),profile_gradient=profile_gradient.numpy(),independent_gradient=independent_gradient,expected_block_gradient=weighted_gradient.numpy())
    rows=[]
    for gi,g in enumerate(train):
        a=analytical_amount(initial[:40],g).numpy()
        for j,r in enumerate(g['rows']):rows.append((r['curve_id'],gi,r['start'],r['end'],r['time'],float(g['w'][j]),float(g['y'][j]),float(a[j]),float(a[j]+initial[40+m+gi])))
    np.savez_compressed(FAMILY/'results/initial-training.npz',curve_id=np.array([r[0] for r in rows]),values=np.array([r[1:] for r in rows]))
    b=blocks[0];tick=time.perf_counter();q=torch.cat([initial,initial.new_zeros(1)]).requires_grad_()
    fl,fd,stats=block_loss(q[:85],train,m,b,eta=q[85]);fg=torch.autograd.grad(fl,q)[0]
    baseline=records[0]
    np.testing.assert_allclose(float(fl.detach()),baseline['loss'],atol=atol,rtol=rtol)
    np.testing.assert_allclose(fg[:85].numpy(),baseline['gradient'],atol=atol,rtol=rtol)
    assert torch.isfinite(fg[85])
    actual_fragment=dict(block_id=0,condition=b['condition'],time=b['time'],n_rows=len(b['indices']),loss=float(fl.detach()),common_gradient_max_error=float(np.max(np.abs(fg[:85].numpy()-baseline['gradient']))),eta_gradient=float(fg[85]),seconds=time.perf_counter()-tick,stats=stats)
    dump(FAMILY/'results/fragment-parity.json',dict(**actual_fragment,gradient=fg.tolist()))
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary=dict(status='complete',all_pass=True,toy=toy_result,n_training_observations=len(rows),n_conditions=len(train),n_blocks=len(blocks),n_rate_coefficients=40,n_response_coefficients=m,n_offsets=7,n_common_parameters=85,optimizer_updates=0,full_loss=float(loss.detach()),full_data_loss=float(data.detach()),profile_loss=float(ploss.detach()),profile_gradient_max_error=float((profile_gradient-gradient[:78]).abs().max()),independent_loss_error=abs(independent_loss-float(loss.detach())),independent_gradient_max_error=float(np.max(np.abs(independent_gradient-gradient.numpy()))),block_expected_loss_error=abs(weighted_loss-float(loss.detach())),block_expected_gradient_max_error=float((weighted_gradient-gradient).abs().max()),actual_fragment=actual_fragment,seconds=time.perf_counter()-start,peak_rss_bytes=rss if sys.platform=='darwin' else rss*1024,scope='Training-only objective and initial-point gradient identities; not a new predictive evaluation or biological fit. Existing loader computes development log targets but they are not consulted in this qualification or used for schedule/initialization selection.')
    dump(FAMILY/'results/qualification-summary.json',summary)
    dump(FAMILY/'results/response-features.json',feature)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
