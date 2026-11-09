"""Independent source/physics reconstruction and constrained scalar least squares."""
import gzip
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import pearsonr,spearmanr
F=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def gz(p):return [json.loads(r) for r in gzip.open(p,'rt')]
def metric(rr):
    y=np.array([r['target'] for r in rr]);p=np.array([r['prediction'] for r in rr]);w=np.array([r['weight'] for r in rr]);w=w/sum(w)
    error=sum(w*(y-p)**2);null=sum(w*y*y);varies=np.ptp(p)>1e-12 and np.ptp(y)>1e-12
    return dict(n=len(rr),mse=float(error),rmse=math.sqrt(error),weighted_r2=float(1-error/null) if null else None,pearson=float(pearsonr(p,y).statistic) if varies else None,spearman=float(spearmanr(p,y).statistic) if varies else None,prediction_sd=float(np.std(p)),target_sd=float(np.std(y)),forward_kl=float(sum(w*np.array([r['candidate_kl'] for r in rr]))))
def equal(actual,expected,tolerance=1e-9):
    for k,v in expected.items():
        if isinstance(v,float):assert abs(actual[k]-v)<tolerance,(k,actual[k],v)
        elif isinstance(v,list) and k=='log_residence_amounts':assert np.allclose(v,actual[k],rtol=1e-10,atol=1e-10)
        else:assert actual[k]==v,(k,actual[k],v)

def main():
    tick=time.perf_counter();c=read(F/'reproduce/launch.json');prior=Path(c['prior_family']);pc=read(prior/'reproduce/launch.json')
    for p,h in c['source_hashes'].items():assert sha(p)==h,p
    for p,h in pc['source_hashes'].items():assert sha(p)==h,p
    for p,h in read(F/'reproduce/verification.json')['files'].items():assert sha(F/p)==h,p
    # Reuse the already independently checked physical reference, not the producer.
    spec=importlib.util.spec_from_file_location('prior_independent_transfer',prior/'code/verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    summary=read(F/'results/summary.json');allrows={};fits={};ninput=0;noutput=0;maxerror=0.;transforms={}
    for arm in ['uniform_composition']+c['arms']:
        groups,transform=old.reference(pc,arm);transforms[arm]=transform
        ld=0. if arm=='uniform_composition' else read(prior/f'results/{arm}/checkpoint.json')['logdelta']
        source_rows=[]
        for g in groups:
            e=g['source'];p,logs,_,_,owner=old.values(ld,g)
            if arm=='uniform_composition':p=np.zeros(g['n'])
            n=sum(h['source']['split']==e['split'] for h in groups);y=g['target'];lp=p*math.log(10)-logsumexp(p*math.log(10));ly=y*math.log(10)-logsumexp(y*math.log(10))
            for j,q in enumerate(e['candidates']):
                source_rows.append(dict(split=e['split'],example_id=e['example_id'],parent=e['parent_sequence_sha256'],component_id=e['component_id'],condition=e['condition'],peptide=q['sequence'],measurement_id=q['measurement_id'],peptide_row_id=q['peptide_row_id'],starts=q['start_positions_0based'],source_log10_intensity=q['source_log10_intensity'],target=float(y[j]),prediction=float(p[j]),weight=1/(n*g['n']),candidate_kl=float(g['n']*np.exp(ly[j])*(ly[j]-lp[j])),log_residence_amounts=logs[owner==j].tolist() if arm!='uniform_composition' else None))
        actual={ (r['example_id'],r['measurement_id']):r for r in gz(prior/f'results/{arm}/predictions.jsonl.gz')}
        assert len(actual)==len(source_rows)
        for r in source_rows:equal(actual[(r['example_id'],r['measurement_id'])],r);ninput+=1
        coefficients=[(arm,1.)]
        if arm!='uniform_composition':
            train=[r for r in source_rows if r['split']=='train'];w=np.array([r['weight'] for r in train]);w/=sum(w);p=np.array([r['prediction'] for r in train]);y=np.array([r['target'] for r in train])
            norm=float(np.dot(p*np.sqrt(w),p*np.sqrt(w)))
            signed=float(np.linalg.lstsq((p*np.sqrt(w))[:,None],y*np.sqrt(w),rcond=None)[0][0]) if norm else None
            beta=max(0.,signed) if signed is not None else 0.
            derivative=float(2*np.sum(w*p*(beta*p-y)))
            assert (beta>0 and abs(derivative)<1e-10) or (beta==0 and derivative>=-1e-10)
            fit=summary['fits'][arm];assert abs(fit['beta']-beta)<1e-10
            if signed is None:assert fit['signed_unconstrained'] is None
            else:assert abs(fit['signed_unconstrained']-signed)<1e-10
            equal(fit,dict(beta=beta,signed_unconstrained=signed,identified=bool(norm>0),cross_moment=float(sum(w*p*y)),prediction_second_moment=norm,target_second_moment=float(sum(w*y*y)),training_rows=len(train),training_parents=len({r['parent'] for r in train}),training_weight_sum=float(sum(r['weight'] for r in train)),derivative=derivative,loss_at_zero=float(sum(w*y*y)),loss_at_one=float(sum(w*(p-y)**2)),loss_at_beta=float(sum(w*(beta*p-y)**2))))
            fits[arm]=dict(beta=beta,signed=signed,coefficient_error=abs(fit['beta']-beta),kkt_derivative=derivative,at_boundary=beta==0)
            coefficients.append((arm+'_calibrated',beta))
        for name,beta in coefficients:
            expected=[dict(r,unscaled_prediction=r['prediction'],prediction=beta*r['prediction'],beta=beta,model=name) for r in source_rows]
            byexample={}
            for r in expected:byexample.setdefault(r['example_id'],[]).append(r)
            for rr in byexample.values():
                y=np.array([r['target'] for r in rr])*math.log(10);p=np.array([r['prediction'] for r in rr])*math.log(10);ly=y-logsumexp(y);lp=p-logsumexp(p)
                for i,r in enumerate(rr):r['candidate_kl']=float(len(rr)*math.exp(ly[i])*(ly[i]-lp[i]))
            output={(r['example_id'],r['measurement_id']):r for r in gz(F/f'results/{name}/predictions.jsonl.gz')};assert len(output)==len(expected)
            for r in expected:
                row=output[(r['example_id'],r['measurement_id'])];equal(row,r);maxerror=max(maxerror,abs(row['prediction']-r['prediction']));noutput+=1
            allrows[name]=expected;e=read(F/f'results/{name}/evaluation.json')
            for split in c['allowed_splits']:equal(e['overall'][split],metric([r for r in expected if r['split']==split]))
            for parent,m in e['per_parent'].items():equal(m,metric([r for r in expected if r['parent']==parent]))
            exposure=read(prior/'results/exposure.json')['parents']
            for key,m in e['exposure_strata'].items():
                split,field,flag=key.split(':');ids={r['parent'] for r in exposure if r['split']==split and bool(r[field])==(flag=='True')}
                equal(m,metric([r for r in expected if r['parent'] in ids]))
    for name,comparisons in summary['comparisons'].items():
        for control,value in comparisons.items():
            differences=[]
            for parent in sorted({r['parent'] for r in allrows[name] if r['split']=='validation'}):
                a=metric([r for r in allrows[name] if r['parent']==parent])['mse'];b=metric([r for r in allrows[control] if r['parent']==parent])['mse'];differences.append(a-b)
                equal(next(r for r in value['per_parent'] if r['parent']==parent),dict(parent=parent,calibrated_mse=a,control_mse=b,difference=a-b))
            assert value['parents_improved']==sum(d < -1e-12 for d in differences)
            assert value['parents_worsened']==sum(d > 1e-12 for d in differences)
            assert value['parents_tied']==sum(abs(d)<=1e-12 for d in differences)
    passes={arm:bool(metric([r for r in allrows[arm+'_calibrated'] if r['split']=='validation'])['mse']<min(metric([r for r in allrows[ctrl] if r['split']=='validation'])['mse'] for ctrl in ['uniform_composition','uniform_rates_calibrated'])) for arm in ['adjacent_rates','wider_rates']}
    assert passes==summary['sequence_prior_passes_both_controls'] and (not any(passes.values()))==summary['stop_extending_same_mapp_bridge']
    result=dict(status='verified',input_source_predictions_rebuilt=ninput,output_predictions_verified=noutput,maximum_prediction_error=maxerror,least_squares_kkt=fits,feature_transforms=transforms,all_metrics_and_parent_strata_verified=True,decision_verified=True,seconds=time.perf_counter()-tick,scope='Reused pinned independent physical reference; new calibration verifier is independent of producer and metrics')
    dump(F/'results/independent-verification.json',result);print(result,flush=True)
if __name__=='__main__':main()
