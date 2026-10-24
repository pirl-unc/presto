"""Independent NumPy group arithmetic from original frozen prediction files."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np

FAMILY=Path(__file__).resolve().parents[1]


def rows(source):
    with gzip.open(source,'rt') as f:return [r for line in f if (r:=json.loads(line))['split']=='train']


def main():
    start=time.perf_counter();cfg=json.loads((FAMILY/'reproduce/launch.json').read_text());saved=json.loads((FAMILY/'results/summary.json').read_text())
    for v in cfg['inputs'].values():assert hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256']
    ref=rows(cfg['inputs'][cfg['reference']]['path']);ref=sorted(ref,key=lambda r:(r['curve_id'],r['time']))
    with gzip.open(FAMILY/'results/curve-residuals.csv.gz','rt') as f:records={(r['model'],r['curve_id']):r for r in csv.DictReader(f)}
    identities=['curve_id','condition_id','construct_id','biological_replicate_raw','peptide_sequence','start','end','time','reference_time','reference_signal','signal','target','split','n_times','instrument','weight','centered_target']
    errors=[];counts=0
    for name in [*cfg['inputs'],'constant']:
        rr=ref if name=='constant' else sorted(rows(cfg['inputs'][name]['path']),key=lambda r:(r['curve_id'],r['time']))
        assert len(rr)==len(ref)==90630
        for a,b in zip(rr,ref):assert all(a[k]==b[k] for k in identities)
        weight=np.array([r['weight'] for r in rr]);residual=np.array([(0. if name=='constant' else r['centered_prediction'])-r['centered_target'] for r in rr])
        unique,first,inverse=np.unique([r['curve_id'] for r in rr],return_index=True,return_inverse=True)
        w=np.bincount(inverse,weights=weight);mean=np.bincount(inverse,weights=weight*residual)/w
        total=np.bincount(inverse,weights=weight*residual**2);between=w*mean**2;within=np.bincount(inverse,weights=weight*(residual-mean[inverse])**2)
        cross=2*mean*np.bincount(inverse,weights=weight*(residual-mean[inverse]));sizes=np.bincount(inverse)
        np.testing.assert_allclose(total,between+within,atol=1e-10,rtol=1e-9)
        out=[]
        matched=[records[(name,c)] for c in unique]
        expected=dict(weight=w,residual_mean=mean,total_sum=total,persistent_sum=between,temporal_sum=within,cross_term=cross,n_rows=sizes)
        for key,values in expected.items():
            actual=np.array([float(record[key]) for record in matched])
            errors.append(float(np.max(np.abs(actual-values))))
            np.testing.assert_allclose(actual,values,atol=1e-10,rtol=1e-9)
        for i,record in enumerate(matched):
            idx=int(first[i])
            assert record['condition_id']==rr[idx]['condition_id'] and record['construct_id']==rr[idx]['construct_id']
            out.append(dict(condition_id=record['condition_id'],construct_id=record['construct_id']))
        for label,field in [('overall',None),('conditions','condition_id'),('constructs','construct_id')]:
            groups=[None] if field is None else sorted({r[field] for r in out})
            for group in groups:
                ix=np.ones(len(unique),dtype=bool) if field is None else np.array([r[field]==group for r in out])
                z=saved['models'][name][label] if field is None else saved['models'][name][label][group]
                norm=w[ix].sum();tm=total[ix].sum()/norm;bm=between[ix].sum()/norm;wm=within[ix].sum()/norm
                computed=dict(weight=norm,n_curves=int(ix.sum()),n_rows=int(sizes[ix].sum()),total_mse=tm,persistent_mse=bm,temporal_mse=wm,persistent_fraction=bm/tm,temporal_fraction=wm/tm,rmse=np.sqrt(tm))
                for key,value in computed.items():np.testing.assert_allclose(value,z[key],atol=1e-10,rtol=1e-9)
        counts+=len(unique)
    result=dict(all_pass=True,models=len(saved['models']),training_rows_per_model=90630,curve_records_verified=counts,maximum_absolute_component_error=max(errors),seconds=time.perf_counter()-start,scope='Independent NumPy grouping/sums from every original training prediction; no producer imports, fitting, development/test metrics or new biological source curation.')
    (FAMILY/'results/independent-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
