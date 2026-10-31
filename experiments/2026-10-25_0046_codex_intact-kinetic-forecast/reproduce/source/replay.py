"""Independent NumPy inference, optimizer recurrence and metric replay."""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1];start_clock=time.monotonic()
config=json.loads((ROOT/'reproduce/launch.json').read_text())
assert json.loads((ROOT/'results/evaluate-resource.json').read_text())['passed']
predicted=pq.read_table(ROOT/'results/evaluation/predictions.parquet').to_pylist()
data={s:pq.read_table(ROOT/f'results/data/{s}/covariates.parquet').to_pylist() for s in ('train','val','test')}
labels={s:pq.read_table(ROOT/f'results/data/{s}/outcomes.parquet').to_pylist() for s in data}
targets={r['assignment_id']:r['target'] for rows in labels.values() for r in rows}
train=data['train'];sequences=sorted({r['peptide_sequence'] for r in train});index={s:i for i,s in enumerate(sequences)}
parent=train[0]['parent_sequence'];aa=sorted(set(parent[:-1]));aa_index=np.array([aa.index(a) for a in parent[:-1]])
seq=np.array([index[r['peptide_sequence']] for r in train]);y=np.array([targets[r['assignment_id']] for r in train]);w=np.array([r['weight'] for r in train])
mass=np.bincount(seq,weights=w,minlength=len(sequences));x=np.array([r['time_minutes']/80 for r in train])
hot=np.eye(len(sequences))[seq];latest=np.array([y[(seq==i)&(x==x[seq==i].max())].mean() for i in range(len(sequences))])
mean=np.array([np.average(y[seq==i],weights=w[seq==i]) for i in range(len(sequences))])
models={'mean':(mean,np.zeros(len(sequences))),'last':(latest,np.zeros(len(sequences)))}
design=np.c_[hot,x];beta=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
models['global_linear']=(beta[:-1],np.full(len(sequences),beta[-1]))
for alpha in (.001,.01,.1,1.):
    design=np.c_[hot,x,hot*x[:,None]]
    penalty=np.zeros((len(sequences),design.shape[1]));penalty[:,len(sequences)+1:]=np.eye(len(sequences))*np.sqrt(alpha/len(sequences))
    beta=np.linalg.lstsq(np.r_[design*np.sqrt(w[:,None]),penalty],np.r_[y*np.sqrt(w),np.zeros(len(sequences))],rcond=None)[0]
    models[f'shrunk_{alpha:g}']=(beta[:len(sequences)],beta[len(sequences)]+beta[len(sequences)+1:])
saved=np.load(ROOT/'results/controls/parameters.npz');maximum=0.
for name,(intercept,slope) in models.items():
    assert np.max(np.abs(intercept-saved[name+'_intercept']))<2e-10
    assert np.max(np.abs(slope-saved[name+'_slope']))<2e-10
    for row in (r for r in predicted if r['model']==name):
        expected=intercept[index[row['peptide_sequence']]]+slope[index[row['peptide_sequence']]]*row['time_minutes']/80
        maximum=max(maximum,abs(expected-row['prediction']));assert abs(expected-row['prediction'])<2e-10


def forward(theta,kind,rows):
    category=1e-7+(1-1e-7)*expit(theta[:-1]);tau=.001+(120-.001)*expit(theta[-1])
    rates=np.full(len(parent)-1,category[0]) if kind=='uniform' else category[aa_index]
    result=[]
    for row in rows:
        i,j=row['start_1based']-1,row['end_1based'];u=row['time_minutes']+tau
        logp=-u*sum(rates[i:j-1])
        if i:logp+=math.log(-math.expm1(-u*rates[i-1]))
        if j<len(parent):logp+=math.log(-math.expm1(-u*rates[j-1]))
        result.append((logp+u*sum(rates))/math.log(10))
    return np.array(result)


bond=np.arange(len(parent)-1);a=np.array([r['start_1based']-1 for r in train]);b=np.array([r['end_1based'] for r in train]);times=np.array([r['time_minutes'] for r in train])
inside=(bond[None,:]>=a[:,None])&(bond[None,:]<(b-1)[:,None]);edge=(bond[None,:]==(a-1)[:,None])|(bond[None,:]==(b-1)[:,None])


def analytic_gradient(theta,kind):
    sigmoid=expit(theta);category=1e-7+(1-1e-7)*sigmoid[:-1];tau=.001+(120-.001)*sigmoid[-1]
    rates=np.full(len(parent)-1,category[0]) if kind=='uniform' else category[aa_index]
    u=times+tau;h=u[:,None]*rates[None,:]
    value=(h*(~inside)+np.log(-np.expm1(-h))*edge).sum(1)/math.log(10)
    offset=np.bincount(seq,weights=w*(y-value),minlength=len(sequences))/mass
    residual=value+offset[seq]-y
    rate_derivative=(u[:,None]*(~inside)+edge*u[:,None]/np.expm1(h))/math.log(10)
    gradient_rates=(2*w*residual)@rate_derivative
    category_gradient=np.array([gradient_rates.sum()]) if kind=='uniform' else np.bincount(aa_index,weights=gradient_rates,minlength=len(aa))
    if kind=='residue':category_gradient+=.02*(np.log(category)-np.log(category).mean())/len(category)/category
    category_gradient*= (1-1e-7)*sigmoid[:-1]*(1-sigmoid[:-1])
    derivative_time=(rates[None,:]*(~inside)+edge*rates[None,:]/np.expm1(h)).sum(1)/math.log(10)
    tau_gradient=2*np.sum(w*residual*derivative_time)*(120-.001)*sigmoid[-1]*(1-sigmoid[-1])
    return np.r_[category_gradient,tau_gradient]


optimizer_errors=[];exact_errors=[];n_values=0
for name in config['fit_conditions']:
    kind=name.split('-')[1];checkpoint=np.load(ROOT/f'results/fits/{name}/checkpoints.npz')
    theta=checkpoint['theta'];first=checkpoint['exp_avg'];second=checkpoint['exp_avg_sq'];rng=checkpoint['rng_state']
    assert theta.shape[0]==801 and np.array_equal(checkpoint['step'],np.arange(801))
    assert np.array_equal(checkpoint['sequences'],sequences)
    raw_gradient=(first[1:]-.9*first[:-1])/.1
    assert np.max(np.abs(second[1:]-(.999*second[:-1]+.001*raw_gradient**2)))<2e-12
    step=np.arange(1,801)[:,None]
    expected=theta[:-1]-.03*(first[1:]/(1-.9**step))/(np.sqrt(second[1:]/(1-.999**step))+1e-8)
    error=float(np.max(np.abs(theta[1:]-expected)));assert error<2e-12
    optimizer_errors.append({'model':name,'max_parameter_update_error':error})
    if '-exact-' in name:
        errors=[np.max(np.abs(analytic_gradient(theta[k],kind)-raw_gradient[k])) for k in range(800)]
        assert max(errors)<2e-10
        exact_errors.append({'model':name,'n_steps':800,'max_gradient_error':float(max(errors))})
        assert np.all(rng==rng[0])
    else:assert np.all(np.any(rng[1:]!=rng[:-1],axis=1))
    physical=forward(theta[-1],kind,train)
    offsets=np.array([np.average(y[seq==i]-physical[seq==i],weights=w[seq==i]) for i in range(len(sequences))])
    assert np.max(np.abs(offsets-checkpoint['offset'][-1]))<2e-10
    for split,rows in data.items():
        values=forward(theta[-1],kind,rows)+offsets[[index[r['peptide_sequence']] for r in rows]]
        archived={r['assignment_id']:r for r in predicted if r['model']==name and r['split']==split}
        assert len(archived)==len(rows)
        for row,value in zip(rows,values,strict=True):
            error=abs(value-archived[row['assignment_id']]['prediction']);maximum=max(maximum,error);assert error<2e-10;n_values+=1


def metric(rows,space):
    y=np.array([r['target' if space=='ratio' else 'target_change'] for r in rows]);p=np.array([r['prediction' if space=='ratio' else 'prediction_change'] for r in rows]);w=np.array([r['weight'] for r in rows]);w/=w.sum()
    mse=float(np.average((p-y)**2,weights=w));variance=float(np.average((y-np.average(y,weights=w))**2,weights=w))
    def correlation(a,b):
        if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
        a=a-a.mean();b=b-b.mean();return float(a@b/np.sqrt((a@a)*(b@b)))
    return {'n_observations':len(rows),'mse':mse,'rmse':math.sqrt(mse),'mae':float(np.average(abs(p-y),weights=w)),
            'r2':1-mse/variance if variance>0 else None,'pearson':correlation(y,p),'spearman':correlation(rankdata(y),rankdata(p)),
            'prediction_sd':float(np.average((p-np.average(p,weights=w))**2,weights=w))**.5,'target_sd':variance**.5}


metric_error=0.;summary=json.loads((ROOT/'results/evaluation/summary.json').read_text())
for item in summary['all_models']:
    rows=[r for r in predicted if r['model']==item['model'] and r['split']==item['split']]
    actual=metric(rows,item['space'])
    for field,value in actual.items():
        if value is None:assert item[field] is None
        else:
            error=abs(item[field]-value);metric_error=max(metric_error,error);assert error<2e-12,(item['model'],field,error)
for row in predicted:
    assert row['target']==targets[row['assignment_id']]
    assert row['last_train_mean']==latest[index[row['peptide_sequence']]]
    assert row['target_change']==row['target']-row['last_train_mean']
    assert row['prediction_change']==row['prediction']-row['last_train_mean']
source_hashes=json.loads((ROOT/'results/pre-outcome-prediction-hashes.json').read_text())
for path,h in source_hashes['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
result={'passed':True,'n_verified_physical_predictions':n_values,'n_control_models':len(models),
        'max_prediction_error':maximum,'n_metric_records':len(summary['all_models']),'max_metric_error':metric_error,
        'optimizer_recurrence':optimizer_errors,'exact_gradient_replay':exact_errors,
        'stochastic_gradient_samples_independently_replayed':False,
        'stochastic_checks':'pre-fit_unbiased_score_qualification_plus_all_optimizer_recurrences_and_fresh_generator_states',
        'seconds':time.monotonic()-start_clock}
(ROOT/'results/independent-prediction-replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
