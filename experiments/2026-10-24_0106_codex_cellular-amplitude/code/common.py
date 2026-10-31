"""Artifact and metric utilities for the frozen prediction calibration."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import pearsonr,spearmanr
FAMILY=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_rows(p):return [json.loads(s) for s in gzip.open(p,'rt')]
def save_rows(p,rows):
    with gzip.open(p,'wt') as h:
        for row in rows:h.write(json.dumps(row,allow_nan=False)+'\n')
def scaled(rows,beta,model):
    result=[dict(r,unscaled_prediction=r['prediction'],prediction=beta*r['prediction'],beta=beta,model=model) for r in rows]
    groups={}
    for r in result:groups.setdefault(r['example_id'],[]).append(r)
    for rr in groups.values():
        p=np.array([r['prediction'] for r in rr])*np.log(10);y=np.array([r['target'] for r in rr])*np.log(10)
        lp=p-logsumexp(p);ly=y-logsumexp(y)
        for r,t,q in zip(rr,ly,lp):r['candidate_kl']=float(len(rr)*np.exp(t)*(t-q))
    return result

def metrics(rr):
    p=np.array([r['prediction'] for r in rr]);y=np.array([r['target'] for r in rr]);w=np.array([r['weight'] for r in rr]);w/=w.sum()
    mse=float(w@((p-y)**2));null=float(w@(y*y));variable=np.ptp(p)>1e-12 and np.ptp(y)>1e-12
    return dict(n=len(rr),mse=mse,rmse=float(np.sqrt(mse)),weighted_r2=1-mse/null if null else None,pearson=float(pearsonr(p,y).statistic) if variable else None,spearman=float(spearmanr(p,y).statistic) if variable else None,prediction_sd=float(p.std()),target_sd=float(y.std()),forward_kl=float(w@np.array([r['candidate_kl'] for r in rr])))
def evaluate(rows,exposure):
    overall={s:metrics([r for r in rows if r['split']==s]) for s in ['train','validation']}
    parents={p:dict(split=next(r['split'] for r in rows if r['parent']==p),**metrics([r for r in rows if r['parent']==p])) for p in sorted({r['parent'] for r in rows})}
    strata={}
    for key in ['train_exact','train_contained_construct','train_shared_peptides','val_exact','val_contained_construct','val_shared_peptides']:
        for split in ['train','validation']:
            for status in [True,False]:
                ids={r['parent'] for r in exposure if r['split']==split and bool(r[key])==status};subset=[r for r in rows if r['parent'] in ids]
                if subset:strata[f'{split}:{key}:{status}']=dict(parents=len(ids),**metrics(subset))
    return dict(overall=overall,per_parent=parents,exposure_strata=strata)
