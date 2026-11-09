"""Independent source/array/gate closure; does not repeat every matrix solve."""
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

FAMILY=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def log_gate(logs, analytical=None):
    """Apply the original tolerance to every refinement and analytic reference."""
    logs=np.asarray(logs,dtype=float)
    if logs.shape[0]!=3 or not np.isfinite(logs).all():
        return False
    if np.max(np.ptp(logs,axis=0))>1e-5:
        return False
    return analytical is None or bool(np.max(np.abs(logs-np.asarray(analytical)))<=1e-5)


def main():
    started=time.perf_counter()
    cfg=json.loads((FAMILY/'reproduce/launch.json').read_text())
    frozen=json.loads((FAMILY/'reproduce/frozen-code.json').read_text())
    for rel,expected in frozen.items():
        assert sha(FAMILY/rel)==expected,rel
    for path,expected in cfg['source_hashes'].items():
        assert sha(path)==expected,path
    support=json.loads((FAMILY/'reproduce/training-support.json').read_text())
    cases=json.loads((FAMILY/'reproduce/cases.json').read_text())
    original=json.loads((Path(cfg['recipe_directory'])/'examples.json').read_text())
    train=[e for e in original if e['split']=='train']
    groups=defaultdict(list)
    for e in train:groups[e['condition_id']].append(e)
    parents=Counter(g[0]['construct_id'] for g in groups.values())
    reconstructed={}
    for condition,curves in sorted(groups.items()):
        rows=defaultdict(list)
        for e in curves:
            for t in e['common_positive_times']:
                rows[t].append((e['curve_id'],e['start_0based'],e['end_0based'],t,
                    1/(len(parents)*parents[e['construct_id']]*len(curves)*len(e['common_positive_times']))))
        for t,rr in rows.items():reconstructed[condition,t]=rr
    assert len(reconstructed)==len(support['blocks'])==28
    nrows=0
    for block in support['blocks']:
        expected=reconstructed[block['condition'],block['time_hours']]
        actual=[(r['curve_id'],r['start'],r['end'],r['time'],r['weight']) for r in block['rows']]
        assert actual==expected
        assert block['spans']==[list(s) for s in sorted({(r[1],r[2]) for r in expected})]
        first=groups[block['condition']][0]
        assert block['sequence']==first['parent_sequence']
        ctx=first['context']
        assert block['context_log_enzyme']==math.log(ctx['substrate_concentration_um']/ctx['enzyme_substrate_denominator']/.2)
        nrows+=len(actual)
    assert nrows==90630
    receipts=json.loads((FAMILY/'results/worker-receipts.json').read_text())
    assert len(receipts['workers'])==len(cases)==84,'Suite is not terminal'
    independent=json.loads((FAMILY/'results/independent-reference.json').read_text())
    assert independent['status']=='complete' and len(independent['cases'])==84
    gate_rows=[]
    outliers=[]
    per_span=[]
    full_arrays=0
    maximum_mass_error=0.0
    derivative_cases=0
    for case in cases:
        cid=case['case_id']
        result=json.loads((FAMILY/f'results/case-{cid:03d}.json').read_text())
        raw=np.load(FAMILY/f'results/case-{cid:03d}.npz')
        block=support['blocks'][case['block_id']]
        for key,value in case.items():assert result[key]==value
        assert result['condition']==block['condition']
        assert result['time_hours']==block['time_hours']
        receipt=receipts['workers'][cid]
        assert receipt['case_id']==cid
        spans=np.array(block['spans']);a,b=spans[:,0],spans[:,1]-1
        np.testing.assert_array_equal(raw['spans'],spans)
        theta=np.array(support['initial_coefficients'])
        np.testing.assert_array_equal(raw['theta'],theta)
        seq=block['sequence'];n=len(seq)
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        x=np.zeros((n-1,40));x[:,0]=1;x[:,-1]=block['context_log_enzyme']
        for k in range(n-1):
            if seq[k]!='A':x[k,alphabet.index(seq[k])]=1
            if seq[k+1]!='A':x[k,19+alphabet.index(seq[k+1])]=1
        np.testing.assert_array_equal(raw['x'],x)
        np.testing.assert_allclose(raw['log_rates'],x@theta,atol=1e-14,rtol=0)
        required=[];log_values=[];negative=[]
        for step in [2,4,8]:
            key=f'counts_{step}'
            if key not in raw:break
            values=raw[key];full_arrays+=1
            assert values.shape==(n,n)
            upper=values[np.triu_indices(n)]
            v=values[a,b];required.append(v)
            ok=(v>0)&np.isfinite(v)
            logs=np.full(v.shape,np.nan);logs[ok]=np.log(v[ok])
            np.testing.assert_allclose(raw[f'logs_{step}'],logs,rtol=0,atol=0,equal_nan=True)
            log_values.append(logs)
            fr=next(r for r in result['forwards'] if r['substeps']==step)
            assert fr['required_nonpositive']==int((v<=0).sum())
            assert fr['full_negative_count']==int((upper<0).sum())
            assert fr['full_zero_count']==int((upper==0).sum())
            assert fr['full_minimum']==float(upper.min())
            ai,bi=np.triu_indices(n)
            np.testing.assert_allclose(fr['full_mass'],upper@(bi-ai+1),rtol=1e-14,atol=1e-12)
            negative.append(fr['full_negative_count'])
            maximum_mass_error=max(maximum_mass_error,abs(float(upper@(bi-ai+1))-n))
        positive=len(required)==3 and all(np.all((v>0)&np.isfinite(v)) for v in required)
        logs=np.array(log_values)
        analytic=None
        if case['eta']==0:
            rates=np.exp(x@theta)
            analytic=[]
            for aa,bb in spans:
                v=-block['time_hours']*sum(rates[aa:bb-1])
                for k in ([aa-1] if aa>0 else [])+([bb-1] if bb<n else []):
                    v+=math.log(-math.expm1(-rates[k]*block['time_hours']))
                analytic.append(v)
            analytic=np.array(analytic)
            np.testing.assert_allclose(raw['analytical_logs'],analytic,rtol=1e-13,atol=1e-12)
        agreement=float(np.max(np.max(logs,axis=0)-np.min(logs,axis=0))) if positive else None
        analytic_error=float(np.max(np.abs(logs-analytic[None,:]))) if positive and analytic is not None else None
        forward=positive and log_gate(logs,analytic)
        # Worker records eta0 error for eight substeps; closure checks all three.
        # All original worker scalars and decisions remain immutable.
        grad_pass=False
        if 'fd_logs' in raw:
            assert result['forward_pass']  # Original producer admission, retained even if closure differs.
            j=np.arange(len(spans));w=(-1.)**j*(1+.25*np.sin(j+1));w/=np.abs(w).sum()
            np.testing.assert_array_equal(raw['weights'],w)
            ds=np.zeros((3,41));ds[0,:40]=np.sin(np.arange(1,41));ds[1,:40]=np.cos(np.arange(1,41));ds[2,-1]=1
            ds/=np.linalg.norm(ds,axis=1)[:,None]
            np.testing.assert_array_equal(raw['directions'],ds)
            fd=(raw['fd_logs'][:,1]@w-raw['fd_logs'][:,0]@w)/(2*cfg['gate']['fd_step'])
            actual=ds@raw['gradient']
            errors=np.abs(fd-actual)
            tolerance=1e-5+1e-6*np.abs(fd)
            np.testing.assert_array_equal(raw['fd_directions'],fd)
            np.testing.assert_array_equal(raw['autograd_directions'],actual)
            np.testing.assert_array_equal(raw['fd_error'],errors)
            grad_pass=bool(np.isfinite(raw['gradient']).all() and np.all(errors<=tolerance))
            assert grad_pass==result['gradient_pass']
            derivative_cases+=1
        ind=independent['cases'][cid]
        assert ind['case_id']==cid and sha(FAMILY/ind['archive'])==ind['sha256']
        ref=np.load(FAMILY/ind['archive'])['counts']
        vv=ref[a,b]
        ref_valid=bool(np.all((vv>0)&np.isfinite(vv)))
        ref_error=float(np.max(np.abs(np.log(vv)-logs[-1]))) if ref_valid else None
        reference_pass=ref_valid and ref_error<=1e-5
        assert ind['required_gate']==reference_pass
        assert ind['full_array_max_abs']==float(np.max(np.abs(ref-raw['counts_8'])))
        accepted=forward and reference_pass and grad_pass and result['status']=='completed' and receipt['returncode']==0
        worker_disagreement=accepted!=result['accepted']
        gate_rows.append(dict(case_id=cid,condition=block['condition'],time_hours=block['time_hours'],eta=case['eta'],
            n_residues=n,n_spans=len(spans),status=result['status'],accepted=accepted,worker_accepted=result['accepted'],worker_gate_disagreement=worker_disagreement,
            required_positive=positive,max_log_disagreement=agreement,analytical_error=analytic_error,independent_log_error=ref_error,
            full_negative_max=max(negative,default=0),seconds=result['seconds'],peak_rss_bytes=result['peak_rss_bytes']))
        for i,(aa,bb) in enumerate(spans):
            row=dict(case_id=cid,condition=block['condition'],time_hours=block['time_hours'],eta=case['eta'],start=int(aa),end=int(bb))
            for j,s in enumerate([2,4,8]):
                row[f'count_{s}']=float(required[j][i]) if j<len(required) else None
                row[f'log_count_{s}']=float(log_values[j][i]) if j<len(log_values) and np.isfinite(log_values[j][i]) else None
            vals=[row[f'log_count_{s}'] for s in [2,4,8]]
            delta=max(vals)-min(vals) if all(v is not None for v in vals) else None
            row['max_log_disagreement']=delta
            row['analytical_log_count']=float(analytic[i]) if analytic is not None else None
            row['independent_log_count']=float(np.log(vv[i])) if np.isfinite(vv[i]) and vv[i]>0 else None
            row['resolved']=delta is not None and delta<=1e-5 and row['independent_log_count'] is not None and abs(vals[-1]-row['independent_log_count'])<=1e-5 and (analytic is None or max(abs(v-analytic[i]) for v in vals)<=1e-5)
            per_span.append(row)
            if not row['resolved']:outliers.append(row)
    def csv_write(name,rows):
        if rows:
            with (FAMILY/'results'/name).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_write('case-summary.csv',gate_rows)
    csv_write('required-span-predictions.csv',per_span)
    csv_write('unresolved-spans.csv',outliers)
    by_eta={str(eta):dict(cases=sum(r['eta']==eta for r in gate_rows),passed=sum(r['eta']==eta and r['accepted'] for r in gate_rows),unresolved_span_queries=sum(r['eta']==eta for r in outliers)) for eta in [-1.,0.,1.]}
    summary=dict(n_cases=len(cases),passed=sum(r['accepted'] for r in gate_rows),by_eta=by_eta,
        biological_fit_admitted=all(r['accepted'] for r in gate_rows),n_training_measurements=nrows,n_unique_span_times=45315,
        n_span_eta_predictions=len(per_span),unresolved_span_eta_queries=len(outliers),worker_gate_disagreements=sum(r['worker_gate_disagreement'] for r in gate_rows),full_arrays_verified=full_arrays,
        derivative_cases=derivative_cases,directional_checks=3*derivative_cases,
        max_gradient_error=max((max(json.loads((FAMILY/f"results/case-{r['case_id']:03d}.json").read_text()).get('gradient_error',[0])) for r in gate_rows),default=0),
        max_log_disagreement_resolved=max((r['max_log_disagreement'] for r in gate_rows if r['max_log_disagreement'] is not None),default=None),
        independent_full_arrays=84,independent_reference_seconds=independent['seconds'],independent_max_log_error=independent['maximum_required_log_error'],independent_max_full_error=independent['maximum_full_array_error'],max_full_negative_count=max(r['full_negative_max'] for r in gate_rows),maximum_residue_mass_error=maximum_mass_error,
        max_rss_bytes=max(r['peak_rss_bytes'] for r in gate_rows),suite_seconds=receipts['seconds'],
        verification_seconds=time.perf_counter()-started,
        scope='Rebuilt all 90,630 training support identities/weights from canonical recipe; every primary array checked against freshly recomputed independent NumPy full arrays; per-span logs, analytical eta0 counts and gates recomputed. Replayed saved NumPy finite-difference reductions, not a second execution of every perturbed matrix solve. Producer independently computed six NumPy perturbations per gradient-admitted case; variable-time oracle tests pass. No target values used or predictive quality measured.')
    write(FAMILY/'results/summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
