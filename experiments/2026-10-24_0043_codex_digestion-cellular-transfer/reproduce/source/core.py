"""Frozen sequence rates and steady-input independent-cut residence amounts."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
LN10 = math.log(10)

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cfg():
    return json.loads((FAMILY / 'reproduce/launch.json').read_text())

def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]

def raw_features(sequence):
    x = np.zeros((len(sequence)-1, 40)); x[:, 0] = 1
    extra = np.zeros((len(sequence)-1, 80))
    for i in range(len(sequence)-1):
        for off, residue in [(0,sequence[i]), (19,sequence[i+1])]:
            j = AA.index(residue)
            if j: x[i, off+j] = 1
        for k, offset in enumerate([-2,-1,2,3]):
            j = i+offset
            residue = sequence[j] if 0 <= j < len(sequence) else '_'
            if residue != 'A': extra[i, 20*k+(AA[1:]+'_').index(residue)] = 1
    return x, extra

def load_rates():
    c = cfg(); parent = Path(c['digestion_family'])
    f = json.loads((parent/'results/features.json').read_text())['rate']
    transform = {k: np.array(f[k]) for k in ['regression','scale','basis']}
    theta = {}
    for arm, key in [('base','adjacent_rates'),('rate_context','wider_rates')]:
        cp = torch.load(parent/f'results/{arm}/checkpoint.pt', weights_only=True)
        theta[key] = cp['parameters'][:cp['rate_parameters']].numpy()
    return transform, theta

def rates(sequence, transform, theta, arm):
    x, e = raw_features(sequence)
    if arm == 'uniform_rates': return np.full(len(x), .01)
    if arm in ['wider_rates','wider_hybrid']:
        r = (e-x@transform['regression'])/transform['scale']
        x = np.column_stack([x, r@transform['basis']])
        arm = 'wider_rates'
    return np.exp(x@theta[arm])

def spans_terms(rate, starts, ends):
    # Direct interval sums avoid long-parent prefix-subtraction cancellation.
    internal = np.array([rate[a:b-1].sum() for a,b in zip(starts,ends)])
    left = np.array([rate[a-1] if a else 0. for a in starts])
    right = np.array([rate[b-1] if b <= len(rate) else 0. for b in ends])
    return internal, left, right

def physical_numpy(logdelta, terms):
    s,l,r = terms; delta = math.exp(logdelta); u = s+delta
    value = -np.log(u); derivative = -1/u
    for k in [l,r]:
        mask = k>0
        value[mask] += np.log(k[mask])-np.log(u[mask]+k[mask])
        derivative[mask] -= 1/(u[mask]+k[mask])
    mask = (l>0)&(r>0)
    value[mask] += np.log(2*u[mask]+l[mask]+r[mask])-np.log(u[mask]+l[mask]+r[mask])
    derivative[mask] += 2/(2*u[mask]+l[mask]+r[mask])-1/(u[mask]+l[mask]+r[mask])
    return value, derivative*delta

def physical_torch(logdelta, terms):
    s,l,r = terms; u = s+logdelta.exp(); value = -u.log()
    for k in [l,r]:
        positive = k>0; safe = torch.where(positive,k,torch.ones_like(k))
        value = value + torch.where(positive,safe.log()-(u+safe).log(),torch.zeros_like(u))
    both = (l>0)&(r>0)
    return value + torch.where(both,(2*u+l+r).log()-(u+l+r).log(),torch.zeros_like(u))

def audit():
    c=cfg()
    for p,h in c['source_hashes'].items(): assert sha(p)==h, p
    data=Path(c['dataset_path']); manifest=json.loads((data/'manifest.json').read_text())
    assert sha(data/'manifest.json')==c['dataset_manifest_sha256']
    for name,meta in manifest['files'].items(): assert sha(data/name)==meta['sha256'],name
    digestion=Path(c['digestion_recipe']); dm=json.loads((digestion/'manifest.json').read_text())
    assert sha(digestion/'examples.json')==dm['files']['examples.json']['sha256']
    parents=[p for p in lines(data/'parents.jsonl') if p['split'] in c['allowed_splits']]
    examples=[e for e in lines(data/'examples.jsonl') if e['split'] in c['allowed_splits']]
    old=json.loads((digestion/'examples.json').read_text())
    old=[r for r in old if r['split'] in ['train','val']]
    transform,theta=load_rates(); rows=[]
    for p in parents:
        sequence=p['sequence']; pid=p['parent_sequence_sha256']
        assert hashlib.sha256(sequence.encode()).hexdigest()==pid
        peps={q['sequence'] for e in examples if e['parent_sequence_sha256']==pid for q in e['candidates']}
        x,e=raw_features(sequence); z=(e-x@transform['regression'])/transform['scale']
        remainder=z-z@transform['basis']@transform['basis'].T
        out=dict(parent=pid,split=p['split'],component_id=p['component_id'],length=len(sequence),feature_outside_span_bonds=int((np.linalg.norm(remainder,axis=1)>1e-6).sum()),max_feature_residual=float(np.linalg.norm(remainder,axis=1).max()))
        for split in ['train','val']:
            constructs={r['construct_id']:r['parent_sequence'] for r in old if r['split']==split}
            oldpeps={r['peptide_sequence'] for r in old if r['split']==split}
            components={r['component_id'] for r in old if r['split']==split}
            out[split+'_exact']=[n for n,s in constructs.items() if s==sequence]
            out[split+'_contained_construct']=[n for n,s in constructs.items() if s in sequence]
            out[split+'_parent_in_construct']=[n for n,s in constructs.items() if sequence in s]
            out[split+'_shared_peptides']=sorted(peps&oldpeps)
            out[split+'_component_match']=p['component_id'] in components
        rows.append(out)
    summary={split:dict(parents=sum(r['split']==split for r in rows),examples=sum(e['split']==split for e in examples),candidates=sum(len(e['candidates']) for e in examples if e['split']==split),training_construct_overlap=sum(r['split']==split and bool(r['train_exact'] or r['train_contained_construct'] or r['train_parent_in_construct']) for r in rows),training_shared_peptide_parents=sum(r['split']==split and bool(r['train_shared_peptides']) for r in rows),outside_feature_span_bonds=sum(r['feature_outside_span_bonds'] for r in rows if r['split']==split)) for split in c['allowed_splits']}
    assert summary['train']['parents']==72 and summary['validation']['parents']==15
    result=dict(summary=summary,parents=rows,scope='Sequence-only audit; no target arithmetic; not homology/specimen independence',source_hashes=c['source_hashes'])
    dump(FAMILY/'results/exposure.json',result)
    return result

def load(arm, synthetic=False):
    c=cfg(); assert (FAMILY/'results/exposure.json').exists()
    data=Path(c['dataset_path']); transform,theta=load_rates()
    parents={p['parent_sequence_sha256']:p for p in lines(data/'parents.jsonl') if p['split'] in c['allowed_splits']}
    raw=[e for e in lines(data/'examples.jsonl') if e['split'] in c['allowed_splits']]
    groups=[]
    for e in raw:
        p=parents[e['parent_sequence_sha256']]; seq=p['sequence']; rate=rates(seq,transform,theta,arm)
        starts=[]; ends=[]; owner=[]; target=[]
        for i,q in enumerate(e['candidates']):
            for a in q['start_positions_0based']:
                assert seq[a:a+len(q['sequence'])]==q['sequence']
                starts.append(a);ends.append(a+len(q['sequence']));owner.append(i)
            if not synthetic: target.append(q['target_centered_log10'])
        terms=spans_terms(rate,starts,ends)
        groups.append(dict(source=e,sequence=seq,rate=rate,starts=np.array(starts),ends=np.array(ends),owner=np.array(owner),terms=terms,y=np.array(target),split=e['split'],n=len(e['candidates'])))
    return groups

def predict_numpy(logdelta,g):
    logf,d=physical_numpy(logdelta,g['terms']); n=g['n']
    amount=np.array([logsumexp(logf[g['owner']==j]) for j in range(n)])
    fraction=np.exp(logf-amount[g['owner']])
    prediction=amount/LN10; prediction-=prediction.mean()
    return prediction,logf,d,fraction

def prepare(logdelta,groups,synthetic=False):
    records=[]; objective=0.;gradient=0.
    train=[g for g in groups if g['split']=='train']; count=len(train)
    for g in train:
        prediction,logf,d,fraction=predict_numpy(logdelta,g)
        target=np.zeros(g['n']) if synthetic else g['y']
        error=prediction-target; error-=error.mean()
        coefficient=2*error[g['owner']]*fraction/(count*g['n']*LN10)
        baseline=float(d.mean()); centered=d-baseline
        record=dict(g,logf=logf,probability=np.exp(logdelta+logf),coefficient=coefficient,derivative=d,centered_derivative=centered,baseline=baseline,logdelta=logdelta)
        assert np.all(record['probability']>0) and np.all(record['probability']<=1+1e-10)
        objective+=np.mean(error**2)/count; gradient+=coefficient@d; records.append(record)
    return objective,gradient,records

def torch_data(groups,synthetic=False):
    data=[]
    for g in groups:
        if g['split']!='train':continue
        data.append(dict(terms=tuple(torch.tensor(t,dtype=torch.float64) for t in g['terms']),owner=torch.tensor(g['owner']),n=g['n'],y=torch.zeros(g['n'],dtype=torch.float64) if synthetic else torch.tensor(g['y'],dtype=torch.float64)))
    return data

def objective_torch(logdelta,data):
    loss=logdelta.new_zeros(())
    for g in data:
        logf=physical_torch(logdelta,g['terms']); owner=g['owner']
        maximum=logf.new_full((g['n'],),-math.inf).scatter_reduce(0,owner,logf,reduce='amax',include_self=True)
        amount=maximum+logf.new_zeros(g['n']).index_add(0,owner,(logf-maximum[owner]).exp()).log()
        pred=amount/LN10; error=pred-pred.mean()-g['y']; error=error-error.mean()
        loss+=error.square().mean()/len(data)
    return loss

def proposal(record):
    score=np.abs(record['coefficient']*record['centered_derivative']); n=len(score)
    return .9*score/score.sum()+.1/n if score.sum()>0 else np.full(n,1/n)

def draw(record,count,generator,w):
    # NumPy generator state is saved for independent replay, no global RNG.
    chosen=generator.choice(len(w),size=count,p=w)
    delta=math.exp(record['logdelta']); s,l,r=record['terms']; u=s[chosen]+delta; left=l[chosen];right=r[chosen]
    waiting=-np.log1p(-generator.random((count,3)))
    age=waiting[:,0]/u
    both=(left>0)&(right>0); one=(left>0)^(right>0)
    left_first=generator.random(count)<(u+left)/(2*u+left+right)
    age[both]+=waiting[both,1]/(u[both]+left[both]+right[both])+waiting[both,2]/(u[both]+np.where(left_first[both],right[both],left[both]))
    age[one]+=waiting[one,1]/(u[one]+left[one]+right[one])
    cuts=generator.random((count,len(record['rate']))) < -np.expm1(-age[:,None]*record['rate'][None,:])
    positions=np.arange(len(record['rate']))[None,:]
    a=record['starts'][chosen,None]; b=record['ends'][chosen,None]
    cuts &= ~((positions>=a)&(positions<b-1)); cuts |= (positions==a-1)|(positions==b-1)
    return cuts,chosen,age

def contains(cuts,starts,ends):
    prefix=np.column_stack([np.zeros(len(cuts),dtype=int),np.cumsum(cuts,axis=1)])
    inside=prefix[:,ends-1]-prefix[:,starts]
    left=np.ones((len(cuts),len(starts)),dtype=bool);right=left.copy()
    idx=starts>0;left[:,idx]=cuts[:,starts[idx]-1]
    idx=ends<=cuts.shape[1];right[:,idx]=cuts[:,ends[idx]-1]
    return (inside==0)&left&right

def sampled(records,count,generator):
    gradient=0.; diagnostics=dict(partitions=0,bond_draws=0,presence_terms=0)
    for r in records:
        w=proposal(r);cuts,chosen,age=draw(r,count,generator,w)
        present=contains(cuts,r['starts'],r['ends']);assert present[np.arange(count),chosen].all()
        density=present@(w/r['probability']);assert np.all(density>0) and np.isfinite(density).all()
        counts=np.mean(present/density[:,None],axis=0)
        gradient+=(counts*r['coefficient']/r['probability'])@r['centered_derivative']+r['baseline']*r['coefficient'].sum()
        diagnostics['partitions']+=count;diagnostics['bond_draws']+=cuts.size;diagnostics['presence_terms']+=present.size
    return float(gradient),diagnostics

def metrics(rows):
    y=np.array([r['target'] for r in rows]);p=np.array([r['prediction'] for r in rows]);w=np.array([r['weight'] for r in rows]);w/=w.sum()
    valid=bool(np.ptp(p)>1e-12 and np.ptp(y)>1e-12)
    return dict(n=len(rows),mse=float(w@((p-y)**2)),rmse=float(np.sqrt(w@((p-y)**2))),pearson=float(pearsonr(p,y).statistic) if valid else None,spearman=float(spearmanr(p,y).statistic) if valid else None,prediction_sd=float(p.std()),target_sd=float(y.std()),kl=float(w@np.array([r['candidate_kl'] for r in rows])))

def evaluate(logdelta,groups,uniform=False):
    rows=[]
    for g in groups:
        e=g['source'];p=np.zeros(g['n']) if uniform else predict_numpy(logdelta,g)[0]
        if uniform:amounts=np.full(len(g['owner']),np.nan)
        else:amounts=physical_numpy(logdelta,g['terms'])[0]
        lp=p*LN10-logsumexp(p*LN10);t=g['y'];lt=t*LN10-logsumexp(t*LN10)
        count=sum(h['split']==g['split'] for h in groups)
        for i,q in enumerate(e['candidates']):
            rows.append(dict(split=g['split'],example_id=e['example_id'],parent=e['parent_sequence_sha256'],component_id=e['component_id'],condition=e['condition'],peptide=q['sequence'],measurement_id=q['measurement_id'],peptide_row_id=q['peptide_row_id'],starts=q['start_positions_0based'],source_log10_intensity=q['source_log10_intensity'],target=float(t[i]),prediction=float(p[i]),weight=1/(count*g['n']),candidate_kl=float(g['n']*np.exp(lt[i])*(lt[i]-lp[i])),log_residence_amounts=amounts[g['owner']==i].tolist() if not uniform else None))
    overall={s:metrics([r for r in rows if r['split']==s]) for s in cfg()['allowed_splits']}
    parents={p:dict(split=next(r['split'] for r in rows if r['parent']==p),**metrics([r for r in rows if r['parent']==p])) for p in sorted({r['parent'] for r in rows})}
    exposure=json.loads((FAMILY/'results/exposure.json').read_text())['parents'];strata={}
    for key in ['train_exact','train_contained_construct','train_shared_peptides','val_exact','val_contained_construct','val_shared_peptides']:
        for split in cfg()['allowed_splits']:
            for status in [True,False]:
                ids={r['parent'] for r in exposure if r['split']==split and bool(r[key])==status};subset=[r for r in rows if r['parent'] in ids]
                if subset:strata[f'{split}:{key}:{status}']=dict(parents=len(ids),**metrics(subset))
    return rows,dict(overall=overall,per_parent=parents,exposure_strata=strata)
