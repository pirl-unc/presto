"""Stable-sum training residual partition; no optimization or held-out metrics."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time

FAMILY=Path(__file__).resolve().parents[1]
SOURCE_FIELDS=['curve_id','condition_id','construct_id','biological_replicate_raw','peptide_sequence','start','end','time','reference_time','reference_signal','signal','target','split','n_times','instrument','weight','centered_target']


def read(path):
    rows={};skipped=0
    with gzip.open(path,'rt') as stream:
        for line in stream:
            r=json.loads(line)
            if r['split']!='train':skipped+=1;continue
            key=(r['curve_id'],r['time']);assert key not in rows;rows[key]=r
    return rows,skipped


def partition(values,weights):
    weight=math.fsum(weights);mean=math.fsum(w*r for r,w in zip(values,weights))/weight
    total=math.fsum(w*r*r for r,w in zip(values,weights))
    between=weight*mean*mean;within=math.fsum(w*(r-mean)**2 for r,w in zip(values,weights))
    cross=2*mean*math.fsum(w*(r-mean) for r,w in zip(values,weights))
    assert math.isclose(total,between+within,abs_tol=1e-10,rel_tol=1e-9)
    return dict(weight=weight,residual_mean=mean,total_sum=total,persistent_sum=between,temporal_sum=within,cross_term=cross)


def toy():
    values=[-2.,1.,4.];weights=[.1,.2,.7];r=partition(values,weights)
    wrong=math.fsum(weights)*(sum(values)/len(values))**2+math.fsum(w*(v-sum(values)/len(values))**2 for v,w in zip(values,weights))
    assert r['persistent_sum']>0 and r['temporal_sum']>0 and abs(wrong-r['total_sum'])>.01
    return dict(**r,wrong_unweighted_total=wrong,passed=True)


def reduce(records):
    weight=math.fsum(r['weight'] for r in records)
    total=math.fsum(r['total_sum'] for r in records)/weight
    persistent=math.fsum(r['persistent_sum'] for r in records)/weight
    temporal=math.fsum(r['temporal_sum'] for r in records)/weight
    return dict(weight=weight,n_curves=len(records),n_rows=sum(r['n_rows'] for r in records),total_mse=total,persistent_mse=persistent,temporal_mse=temporal,persistent_fraction=persistent/total,temporal_fraction=temporal/total,rmse=math.sqrt(total),identity_error=abs(total-persistent-temporal),cross_term_sum=math.fsum(r['cross_term'] for r in records))


def main():
    start=time.perf_counter();cfg=json.loads((FAMILY/'reproduce/launch.json').read_text())
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Registered 120-second cap')));signal.alarm(cfg['runtime_limits']['seconds'])
    assert not (FAMILY/'results/summary.json').exists()
    for rel,h in json.loads((FAMILY/'reproduce/source-hashes.json').read_text()).items():assert hashlib.sha256((FAMILY/rel).read_bytes()).hexdigest()==h,rel
    for source in cfg['inputs'].values():assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    ref,skipped=read(cfg['inputs'][cfg['reference']]['path']);assert len(ref)==90630
    summaries={};all_records=[];source_audits=[]
    for name in [*cfg['inputs'],'constant']:
        rows,excluded=(ref,skipped) if name=='constant' else read(cfg['inputs'][name]['path'])
        assert set(rows)==set(ref)
        curves=defaultdict(list)
        for key,r in rows.items():
            for field in SOURCE_FIELDS:assert r[field]==ref[key][field],(name,key,field)
            curves[r['curve_id']].append(r)
        assert abs(math.fsum(r['weight'] for r in rows.values())-1)<1e-12
        records=[]
        for cid,rr in sorted(curves.items()):
            assert len({r['condition_id'] for r in rr})==len({r['construct_id'] for r in rr})==1
            values=[(0. if name=='constant' else r['centered_prediction'])-r['centered_target'] for r in rr]
            row=dict(model=name,curve_id=cid,condition_id=rr[0]['condition_id'],construct_id=rr[0]['construct_id'],n_rows=len(rr),**partition(values,[r['weight'] for r in rr]))
            records.append(row)
        subsets={}
        for label,field in [('conditions','condition_id'),('constructs','construct_id')]:
            gg=defaultdict(list)
            for r in records:gg[r[field]].append(r)
            subsets[label]={key:reduce(rr) for key,rr in sorted(gg.items())}
        summaries[name]=dict(overall=reduce(records),**subsets);all_records.extend(records)
        source_audits.append(dict(model=name,training_rows=len(rows),excluded_nontraining_rows=excluded,curves=len(curves),all_training_identities_values_weights_match=True))
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert rss<=cfg['runtime_limits']['rss_bytes']
        print(json.dumps(dict(model=name,**summaries[name]['overall'])),flush=True)
    with gzip.open(FAMILY/'results/curve-residuals.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_records[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_records)
    result=dict(all_pass=True,toy=toy(),models=summaries,source_audits=source_audits,optimizer_updates=0,seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    (FAMILY/'results/summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    signal.alarm(0)


if __name__=='__main__':main()
