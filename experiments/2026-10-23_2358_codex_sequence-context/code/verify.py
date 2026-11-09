"""Independent NumPy feature/readout/metric checks and optimizer replay."""
import argparse
import gzip
import json
import math
import time
from collections import defaultdict
import numpy as np
import torch
from core import AA,FAMILY,dump,independent,load,sha,variant
from independent_metrics import metrics


def decompose(rows):
    names,inv=np.unique([r['curve_id'] for r in rows],return_inverse=True)
    w=np.array([r['weight'] for r in rows]);e=np.array([r['centered_prediction']-r['centered_target'] for r in rows])
    weight=np.bincount(inv,weights=w,minlength=len(names));mean=np.bincount(inv,weights=w*e)/weight
    persistent=weight@mean**2;temporal=w@(e-mean[inv])**2
    np.testing.assert_allclose(persistent+temporal,w@e**2,atol=1e-12,rtol=1e-10)
    return dict(persistent_mse=float(persistent),temporal_mse=float(temporal),persistent_fraction=float(persistent/(persistent+temporal)))


def rebuild_features(groups,feature,arm):
    for g in groups:
        if arm in ['rate_context','both','both_hybrid']:
            seq=g['sequence'];raw=np.zeros((len(seq)-1,80))
            for j,d in enumerate([-2,-1,2,3]):
                for k,a in enumerate(AA[1:]+'_'):
                    raw[:,j*20+k]=[float((seq[i+d] if 0<=i+d<len(seq) else '_')==a) for i in range(len(seq)-1)]
            f=feature['rate'];extra=((raw-g['x'].numpy()@np.array(f['regression']))/np.array(f['scale']))@np.array(f['basis'])
            g['x']=torch.tensor(np.column_stack([g['x'].numpy(),extra]))
        if arm in ['response_termini','both','both_hybrid']:
            instruments=feature['old_response']['instruments'];raw=np.zeros((len(g['rows']),38*len(instruments)));offset=instruments.index(g['instrument'])*38
            for k,a in enumerate(AA[1:]):
                raw[:,offset+k]=[r['peptide_sequence'][0]==a for r in g['rows']];raw[:,offset+19+k]=[r['peptide_sequence'][-1]==a for r in g['rows']]
            f=feature['response'];extra=((raw-g['z'].numpy()@np.array(f['regression']))/np.array(f['scale']))@np.array(f['basis'])
            g['z']=torch.tensor(np.column_stack([g['z'].numpy(),extra]))
    return groups


def sampled_gradient(theta,gamma,groups,rng):
    generator=torch.Generator();generator.set_state(torch.tensor(rng,dtype=torch.uint8))
    result=np.zeros(len(theta))
    for g in groups:
        if g['split']!='train':continue
        x,z,y,w,a,b,t=[g[k].numpy() for k in ['x','z','y','w','a','b','t']]
        rates=np.exp(x@theta);rp=np.r_[0,np.cumsum(rates)]
        lv=-t*(rp[b-1]-rp[a])
        for mask,idx in [(a>0,a-1),(b<=len(rates),b-1)]:
            ii=np.flatnonzero(mask);lv[ii]+=np.log(-np.expm1(-rates[idx[ii]]*t[ii]))
        residual=lv/math.log(10)+z@gamma-y;residual-=w@residual/w.sum();coefficient=2*w*residual/math.log(10)
        for timepoint in sorted(set(t)):
            mask=t==timepoint;spans,inverse=np.unique(np.column_stack([a[mask],b[mask]]),axis=0,return_inverse=True);starts,ends=spans.T
            h=rates*timepoint;cutscore=h*np.exp(-h)/(-np.expm1(-h));prefix=np.vstack([np.zeros((1,len(theta))),np.cumsum(h[:,None]*x,axis=0)])
            score=-(prefix[ends-1]-prefix[starts]);logs=-timepoint*(rp[ends-1]-rp[starts])
            for valid,idx in [(starts>0,starts-1),(ends<=len(rates),ends-1)]:
                ii=np.flatnonzero(valid);k=idx[ii];score[ii]+=cutscore[k,None]*x[k];logs[ii]+=np.log(-np.expm1(-h[k]))
            mass=np.exp(logs);coeff=np.bincount(inverse,weights=coefficient[mask],minlength=len(spans))
            importance=np.abs(coeff)*np.linalg.norm(score,axis=1)
            proposal=.9*importance/importance.sum()+.1/len(spans) if importance.sum()>0 else np.full(len(spans),1/len(spans))
            chosen=torch.multinomial(torch.tensor(proposal),512,replacement=True,generator=generator).numpy()
            cuts=torch.rand((512,len(rates)),dtype=torch.float64,generator=generator).numpy() < -np.expm1(-h)
            positions=np.arange(len(rates))[None,:];sa=starts[chosen,None];sb=ends[chosen,None]
            cuts&=~((positions>=sa)&(positions<sb-1));cuts|=(positions==sa-1)|(positions==sb-1)
            cumsum=np.column_stack([np.zeros(512,dtype=int),np.cumsum(cuts,axis=1)])
            present=(cumsum[:,ends-1]-cumsum[:,starts])==0
            valid=starts>0;present[:,valid]&=cuts[:,starts[valid]-1]
            valid=ends<=len(rates);present[:,valid]&=cuts[:,ends[valid]-1]
            indicators=present.astype(float);density=indicators@(proposal/mass);assert np.all(density>0)
            result+=((indicators/density[:,None]).mean(0)*coeff/mass)@score
    result[1:]+=.01*theta[1:]
    return result


def main(arm):
    tick=time.perf_counter();cfg,_,groups,computed_feature,initial=load();out=FAMILY/'results'/arm
    for rel,h in json.loads((FAMILY/'reproduce/execution-code.json').read_text()).items():assert sha(FAMILY/rel)==h
    status=json.loads((out/'status.json').read_text());assert status['status']=='completed' and status['updates']==2000
    feature=json.loads((FAMILY/'results/features.json').read_text());assert feature==computed_feature
    producer,t0,g0=variant(groups,initial,arm);gg=rebuild_features([dict(g) for g in groups],feature,arm)
    for g,p in zip(gg,producer):
        np.testing.assert_allclose(g['x'],p['x'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(g['z'],p['z'],atol=1e-12,rtol=1e-12)
    checkpoint=torch.load(out/'checkpoint.pt',weights_only=True);p=checkpoint['parameters'].numpy();nx=checkpoint['rate_parameters'];updates=np.load(out/'updates.npz')
    np.testing.assert_array_equal(updates['before'][0],np.r_[t0.numpy(),g0.numpy()]);np.testing.assert_array_equal(updates['before'][1:],updates['after'][:-1]);np.testing.assert_array_equal(p,updates['after'][-1])
    m=np.zeros_like(p);v=np.zeros_like(p);max_update=0.;checks=[];rng=json.loads((out/'rng.json').read_text())
    for i,grad in enumerate(updates['gradient']):
        m=.9*m+.1*grad;v=.999*v+.001*grad*grad
        after=updates['before'][i]-(.03/(1-.9**(i+1)))*m/(np.sqrt(v)/math.sqrt(1-.999**(i+1))+1e-8)
        for actual,expected in [(m,updates['moment'][i]),(v,updates['square'][i]),(after,updates['after'][i])]:np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-9)
        max_update=max(max_update,float(np.max(np.abs(after-updates['after'][i]))))
        if i+1 in [1,100,1000,2000]:
            q=updates['before'][i];_,gradient,_=independent(q[:nx],q[nx:],[g for g in gg if g['split']=='train'])
            if arm=='both_hybrid':gradient[:nx]=sampled_gradient(q[:nx],q[nx:],gg,rng[str(i+1)])
            error=float(np.max(np.abs(gradient-grad)));np.testing.assert_allclose(gradient,grad,atol=1e-8,rtol=1e-7);checks.append(dict(step=i+1,maximum_gradient_error=error))
    evaluation=json.loads((out/'evaluation.json').read_text())
    grad_count=(np.abs(updates['gradient'])>1e-12).sum(0);update_count=(np.abs(updates['after']-updates['before'])>1e-12).sum(0)
    np.testing.assert_array_equal(grad_count,evaluation['gradient_counts']);np.testing.assert_array_equal(update_count,evaluation['update_counts']);assert bool((grad_count>0).all() and (update_count>0).all())==evaluation['all_allocated_materially_trained']
    value,gradient,amounts=independent(p[:nx],p[nx:],gg);np.testing.assert_allclose(value,evaluation['final_objective'],atol=1e-10,rtol=1e-10);np.testing.assert_allclose(gradient,evaluation['final_exact_gradient'],atol=1e-8,rtol=1e-7)
    with gzip.open(out/'predictions.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    assert len(rows)==105452;by_condition=defaultdict(list)
    for row in rows:by_condition[row['condition_id']].append(row)
    max_amount=0.
    for g,a in zip(gg,amounts):
        rr=by_condition[g['condition']];assert len(rr)==len(g['rows'])
        for saved,source in zip(rr,g['rows']):
            for key,x in source.items():assert saved[key]==x
        w=g['w'].numpy();np.testing.assert_array_equal(w,[r['weight'] for r in rr]);response=g['z'].numpy()@p[nx:];raw=a+response
        max_amount=max(max_amount,float(np.max(np.abs(a-np.array([r['log10_amount'] for r in rr])))))
        for key,x in [('log10_amount',a),('log10_response',response),('raw_prediction',raw),('centered_prediction',raw-w@raw/w.sum()),('centered_target',g['y'].numpy()-w@g['y'].numpy()/w.sum())]:np.testing.assert_allclose(x,[r[key] for r in rr],atol=1e-10,rtol=1e-10)
    calculated={}
    for split in ['train','val']:
        rr=[r for r in rows if r['split']==split];calculated[split]=dict(**metrics(rr),**decompose(rr))
        for key,x in calculated[split].items():np.testing.assert_allclose(x,evaluation['metrics'][split][key],atol=1e-10,rtol=1e-9)
    for parent,record in evaluation['per_parent'].items():
        rr=[dict(r) for r in rows if r['construct_id']==parent];weight=sum(r['weight'] for r in rr)
        for r in rr:r['weight']/=weight
        for key,x in dict(**metrics(rr),**decompose(rr)).items():np.testing.assert_allclose(x,record[key],atol=1e-10,rtol=1e-9)
    result=dict(all_pass=True,arm=arm,updates_replayed=2000,n_predictions_verified=len(rows),maximum_adam_error=max_update,selected_gradient_checks=checks,maximum_log10_amount_error=max_amount,all_allocated_materially_trained=evaluation['all_allocated_materially_trained'],metrics=calculated,seconds=time.perf_counter()-tick,scope='New raw features independently rebuilt; all final physical/source/metric values and Adam arithmetic replay. Four fixed gradient states; hybrid uses independent NumPy conditional score/importance arithmetic with PyTorch RNG draws. Other gradients not replayed. Pinned source loader reused; no new raw-source curation.')
    dump(out/'independent-verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm');main(parser.parse_args().arm)
