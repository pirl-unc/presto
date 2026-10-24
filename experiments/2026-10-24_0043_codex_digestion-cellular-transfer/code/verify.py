"""Independent source, feature, residence, metric and scalar Adam replay."""
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import pearsonr,spearmanr

F=Path(__file__).resolve().parents[1];AA='ACDEFGHIKLMNPQRSTVWY';LN10=math.log(10)
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(r) for r in Path(p).read_text().splitlines()]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def feature(seq,context=0.):
    index=np.array([AA.index(a) for a in seq]);n=len(seq)-1
    x=np.concatenate([np.ones((n,1)),np.eye(20)[index[:-1],1:],np.eye(20)[index[1:],1:],np.full((n,1),context)],axis=1)
    extended=[]
    for off in [-2,-1,2,3]:
        positions=np.arange(n)+off;valid=(positions>=0)&(positions<len(seq));categories=np.full(n,20)
        categories[valid]=index[positions[valid]]
        extended.append(np.eye(21)[categories,1:])
    return x,np.concatenate(extended,axis=1)

def check_transform(c):
    saved=read(Path(c['digestion_family'])/'results/features.json')['rate']
    source=read(Path(c['digestion_recipe'])/'examples.json');conditions={r['condition_id']:r for r in source if r['split']=='train'}
    parents=Counter(r['construct_id'] for r in conditions.values());xs=[];es=[]
    for _,r in sorted(conditions.items()):
        ctx=r['context'];enzyme=ctx['substrate_concentration_um']/ctx['enzyme_substrate_denominator']
        x,e=feature(r['parent_sequence'],math.log(enzyme/.2))
        weight=1/(len(parents)*parents[r['construct_id']]*len(x))
        xs.append(x*math.sqrt(weight));es.append(e*math.sqrt(weight))
    x=np.concatenate(xs);e=np.concatenate(es);reg=np.linalg.lstsq(x,e,rcond=1e-12)[0]
    residual=e-x@reg;gram=residual.T@residual;scale=np.sqrt(np.maximum(np.diag(gram),0));scale=np.where(scale>1e-10,scale,1.)
    eigen,vec=np.linalg.eigh(gram/scale[:,None]/scale[None,:]);basis=vec[:,eigen>max(eigen[-1],1.)*1e-10]
    for j in range(basis.shape[1]):
        nonzero=np.flatnonzero(abs(basis[:,j])>1e-10)
        if basis[nonzero[0],j]<0:basis[:,j]*=-1
    errors={k:float(np.max(abs(v-np.array(saved[k])))) for k,v in [('regression',reg),('scale',scale),('basis',basis)]}
    # A degenerate eigenspace may rotate: the saved basis is the coordinate authority;
    # compare its projector and its eigen equation in addition to raw reconstruction.
    projector=float(np.max(abs(basis@basis.T-np.array(saved['basis'])@np.array(saved['basis']).T)))
    assert errors['regression']<1e-8 and errors['scale']<1e-8 and projector<1e-8
    return {k:np.array(saved[k]) for k in ['regression','scale','basis']},dict(raw_errors=errors,projector_error=projector)

def reference(c,arm):
    transform,diagnostics=check_transform(c)
    base=Path(c['dataset_path']);parents={p['parent_sequence_sha256']:p for p in rows(base/'parents.jsonl') if p['split'] in c['allowed_splits']}
    seqarm='wider_rates' if arm=='wider_hybrid' else arm
    if seqarm in ['adjacent_rates','wider_rates']:
        oldarm='base' if seqarm=='adjacent_rates' else 'rate_context'
        cp=torch.load(Path(c['digestion_family'])/f'results/{oldarm}/checkpoint.pt',weights_only=True)
        theta=cp['parameters'][:cp['rate_parameters']].numpy()
    groups=[]
    for e in rows(base/'examples.jsonl'):
        if e['split'] not in c['allowed_splits']:continue
        parent=parents[e['parent_sequence_sha256']];seq=parent['sequence'];x,extra=feature(seq)
        if seqarm=='uniform_rates' or arm=='uniform_composition':rate=np.repeat(.01,len(seq)-1)
        else:
            if seqarm=='wider_rates':x=np.concatenate([x,((extra-x@transform['regression'])/transform['scale'])@transform['basis']],axis=1)
            rate=np.exp(x@theta)
        raw=np.array([q['source_log10_intensity'] for q in e['candidates']]);target=raw-raw.mean()
        assert np.allclose(target,[q['target_centered_log10'] for q in e['candidates']],atol=1e-12,rtol=0)
        spans=[]
        for i,q in enumerate(e['candidates']):
            for start in q['start_positions_0based']:
                end=start+len(q['sequence']);assert seq[start:end]==q['sequence'];spans.append((start,end,i))
        groups.append(dict(source=e,rate=rate,spans=spans,target=target,n=len(target)))
    return groups,diagnostics

def values(ld,g):
    delta=math.exp(ld);logs=[];derivatives=[]
    for a,b,j in g['spans']:
        u=delta+sum(g['rate'][a:b-1]);left=g['rate'][a-1] if a else None;right=g['rate'][b-1] if b<=len(g['rate']) else None
        if left is not None and right is not None:
            log=math.log(left)+math.log(right)+math.log(2*u+left+right)-math.log(u)-math.log(u+left)-math.log(u+right)-math.log(u+left+right)
            mean=1/u+1/(u+left+right)+(u+left)/(2*u+left+right)/(u+right)+(u+right)/(2*u+left+right)/(u+left)
        elif left is not None or right is not None:
            k=left if left is not None else right;log=math.log(k)-math.log(u)-math.log(u+k);mean=1/u+1/(u+k)
        else:log=-math.log(u);mean=1/u
        logs.append(log);derivatives.append(-delta*mean)
    logs=np.array(logs);derivatives=np.array(derivatives);owner=np.array([s[2] for s in g['spans']])
    combined=np.array([logsumexp(logs[owner==i]) for i in range(g['n'])]);p=combined/LN10;p-=p.mean()
    return p,logs,derivatives,np.exp(logs-combined[owner]),owner

def reference_gradient(ld,groups):
    train=[g for g in groups if g['source']['split']=='train'];gradient=loss=0.;records=[]
    for g in train:
        p,logf,d,fraction,owner=values(ld,g);error=p-g['target'];error-=error.mean()
        coefficient=2*error[owner]*fraction/(len(train)*g['n']*LN10)
        gradient+=coefficient@d;loss+=np.mean(error**2)/len(train)
        records.append(dict(g,logf=logf,derivative=d,coefficient=coefficient))
    return float(gradient),float(loss),records

def sampled_gradient(ld,records,rng_state,k):
    rng=np.random.default_rng();rng.bit_generator.state=rng_state;total=0.
    for r in records:
        d=r['derivative'];baseline=d.mean();contrast=d-baseline;coef=r['coefficient']
        score=np.abs(coef*contrast);w=.9*score/score.sum()+.1/len(score) if score.sum()>0 else np.ones(len(score))/len(score)
        chosen=rng.choice(len(w),size=k,p=w);a=np.array([s[0] for s in r['spans']]);b=np.array([s[1] for s in r['spans']]);rate=r['rate']
        left=np.array([rate[v-1] if v else 0. for v in a]);right=np.array([rate[v-1] if v<=len(rate) else 0. for v in b]);inside=np.array([sum(rate[i:j-1]) for i,j in zip(a,b)])
        u=inside[chosen]+math.exp(ld);l=left[chosen];rr=right[chosen]
        exponential=-np.log1p(-rng.random((k,3)));age=exponential[:,0]/u;order=rng.random(k)<(u+l)/(2*u+l+rr)
        for i in range(k):
            if l[i] and rr[i]:age[i]+=exponential[i,1]/(u[i]+l[i]+rr[i])+exponential[i,2]/(u[i]+(rr[i] if order[i] else l[i]))
            elif l[i] or rr[i]:age[i]+=exponential[i,1]/(u[i]+l[i]+rr[i])
        cuts=rng.random((k,len(rate))) < -np.expm1(-age[:,None]*rate[None,:])
        for i,s in enumerate(chosen):
            cuts[i,a[s]:b[s]-1]=False
            if a[s]:cuts[i,a[s]-1]=True
            if b[s]<=len(rate):cuts[i,b[s]-1]=True
        present=np.ones((k,len(a)),dtype=bool)
        for j,(start,end) in enumerate(zip(a,b)):
            present[:,j]=~np.any(cuts[:,start:end-1],axis=1)
            if start:present[:,j]&=cuts[:,start-1]
            if end<=len(rate):present[:,j]&=cuts[:,end-1]
        prob=np.exp(ld+r['logf']);density=np.sum(present*(w/prob)[None,:],axis=1)
        total+=np.mean(np.sum(present*(coef*contrast/prob)[None,:],axis=1)/density)+baseline*coef.sum()
    return float(total)

def metric(rr):
    p=np.array([r['prediction'] for r in rr]);y=np.array([r['target'] for r in rr]);w=np.array([r['weight'] for r in rr]);w/=sum(w)
    ok=np.ptp(p)>1e-12 and np.ptp(y)>1e-12
    return dict(n=len(rr),mse=float(sum(w*(p-y)**2)),rmse=float(math.sqrt(sum(w*(p-y)**2))),pearson=float(pearsonr(p,y).statistic) if ok else None,spearman=float(spearmanr(p,y).statistic) if ok else None,prediction_sd=float(np.std(p)),target_sd=float(np.std(y)),kl=float(sum(w*np.array([r['candidate_kl'] for r in rr]))))

def main():
    tick=time.perf_counter();arm=sys.argv[1];c=read(F/'reproduce/launch.json');out=F/'results'/arm
    for p,h in c['source_hashes'].items():assert digest(p)==h,p
    for p,h in read(F/'reproduce/verification.json')['files'].items():assert digest(F/p)==h,p
    groups,transform=reference(c,arm);uniform=arm=='uniform_composition'
    ld=0. if uniform else read(out/'checkpoint.json')['logdelta'];observed=[json.loads(s) for s in gzip.open(out/'predictions.jsonl.gz','rt')]
    keyed={(r['example_id'],r['measurement_id']):r for r in observed};replayed=[];maximum=0.
    assert len(keyed)==len(observed)==sum(g['n'] for g in groups)
    for g in groups:
        e=g['source'];pred,logf,_,_,owner=values(ld,g)
        if uniform:pred=np.zeros(g['n'])
        lt=g['target']*LN10-logsumexp(g['target']*LN10);lp=pred*LN10-logsumexp(pred*LN10)
        count=sum(h['source']['split']==e['split'] for h in groups)
        for j,q in enumerate(e['candidates']):
            actual=keyed[(e['example_id'],q['measurement_id'])]
            expected=dict(split=e['split'],example_id=e['example_id'],parent=e['parent_sequence_sha256'],component_id=e['component_id'],condition=e['condition'],peptide=q['sequence'],measurement_id=q['measurement_id'],peptide_row_id=q['peptide_row_id'],starts=q['start_positions_0based'],source_log10_intensity=q['source_log10_intensity'],target=float(g['target'][j]),prediction=float(pred[j]),weight=1/(count*g['n']),candidate_kl=float(g['n']*math.exp(lt[j])*(lt[j]-lp[j])),log_residence_amounts=logf[owner==j].tolist() if not uniform else None)
            for k,v in expected.items():
                if isinstance(v,float):
                    err=abs(v-actual[k]);maximum=max(maximum,err);assert err<1e-9,(k,err)
                elif k=='log_residence_amounts' and v is not None:assert np.allclose(v,actual[k],rtol=1e-10,atol=1e-10)
                else:assert v==actual[k],k
            replayed.append(expected)
    expected_metrics=read(out/'evaluation.json')
    def check_metrics(rows,ref):
        actual=metric(rows)
        for k,v in actual.items():
            if v is None:assert ref[k] is None
            else:assert abs(v-ref[k])<1e-9,(k,v,ref[k])
    for split in c['allowed_splits']:check_metrics([r for r in replayed if r['split']==split],expected_metrics['overall'][split])
    for p,v in expected_metrics['per_parent'].items():check_metrics([r for r in replayed if r['parent']==p],v)
    exposure=read(F/'results/exposure.json')['parents']
    for key,v in expected_metrics['exposure_strata'].items():
        split,field,yes=key.split(':');ids={r['parent'] for r in exposure if r['split']==split and bool(r[field])==(yes=='True')}
        check_metrics([r for r in replayed if r['parent'] in ids],v)
    optimizer_error=0.;gradient_checks=[]
    if not uniform:
        states=read(out/'optimizer.json');assert len(states)==c['steps'];m=v=0.;old=math.log(c['initial_delta'])
        for i,s in enumerate(states,1):
            gradient=s['gradient'];m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
            after=old-c['learning_rate']*(m/(1-.9**i))/(math.sqrt(v/(1-.999**i))+c['epsilon'])
            error=max(abs(s['before']-old),abs(s['after']-after),abs(s['moment']-m),abs(s['square']-v));optimizer_error=max(optimizer_error,error)
            assert error<1e-9,(i,error);old=s['after']
            if i in [1,100,1000,2000]:
                exact,loss,records=reference_gradient(s['before'],groups)
                replay=sampled_gradient(s['before'],records,read(out/'rng.json')[str(i)],c['sample_count']) if arm=='wider_hybrid' else exact
                assert abs(replay-gradient)<1e-9,(i,replay,gradient)
                gradient_checks.append(dict(step=i,gradient_error=abs(replay-gradient),reference_gradient=exact,loss=loss))
        assert abs(old-ld)<1e-12
    result=dict(status='verified',arm=arm,predictions=len(replayed),maximum_prediction_field_error=maximum,feature_transform=transform,optimizer_updates=0 if uniform else len(states),maximum_optimizer_error=optimizer_error,fixed_gradient_checks=gradient_checks,source_target_reconstruction='original source log10 intensities centered per example; no historical test',metrics_verified=True,seconds=time.perf_counter()-tick)
    write(out/'independent-verification.json',result);print(result,flush=True)

if __name__=='__main__':main()
