"""Three exact train-only nonnegative scalar calibrations; no physical refit."""
import resource
import time
from pathlib import Path
import numpy as np
from common import FAMILY,dump,evaluate,read,read_rows,save_rows,scaled,sha

def main():
    tick=time.perf_counter();c=read(FAMILY/'reproduce/launch.json');prior=Path(c['prior_family'])
    for p,h in c['source_hashes'].items():assert sha(p)==h,p
    for p,h in read(FAMILY/'reproduce/execution-code.json')['files'].items():assert sha(FAMILY/p)==h,p
    assert not (FAMILY/'results/summary.json').exists()
    exposure=read(prior/'results/exposure.json')['parents'];results={};fits={}
    for arm in ['uniform_composition']+c['arms']:
        original=read_rows(prior/f'results/{arm}/predictions.jsonl.gz')
        assert {r['split'] for r in original}==set(c['allowed_splits'])
        variant=scaled(original,1.,arm);out=FAMILY/'results'/arm;out.mkdir()
        save_rows(out/'predictions.jsonl.gz',variant);evaluation=evaluate(variant,exposure);dump(out/'evaluation.json',evaluation);results[arm]=evaluation
        if arm=='uniform_composition':continue
        train=[r for r in original if r['split']=='train'];w=np.array([r['weight'] for r in train]);w/=w.sum()
        p=np.array([r['prediction'] for r in train]);y=np.array([r['target'] for r in train])
        cross=float(w@(p*y));square=float(w@(p*p));target_square=float(w@(y*y))
        assert square>=0 and np.isfinite([cross,square,target_square]).all()
        signed=cross/square if square else None;beta=max(0.,signed) if signed is not None else 0.
        fit=dict(beta=beta,signed_unconstrained=signed,identified=bool(square>0),cross_moment=cross,prediction_second_moment=square,target_second_moment=target_square,training_rows=len(train),training_parents=len({r['parent'] for r in train}),training_weight_sum=float(sum(r['weight'] for r in train)),derivative=2*(beta*square-cross),loss_at_zero=target_square,loss_at_one=target_square-2*cross+square,loss_at_beta=target_square-2*beta*cross+beta*beta*square)
        fits[arm]=fit;name=arm+'_calibrated';variant=scaled(original,beta,name);out=FAMILY/'results'/name;out.mkdir()
        save_rows(out/'predictions.jsonl.gz',variant);dump(out/'coefficient.json',fit);evaluation=evaluate(variant,exposure);dump(out/'evaluation.json',evaluation);results[name]=evaluation
        assert time.perf_counter()-tick<=c['caps']['producer_seconds'];assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=c['caps']['peak_rss_bytes']
    comparisons={}
    for arm in c['arms']:
        name=arm+'_calibrated';comparison={}
        for control in ['uniform_composition','uniform_rates_calibrated',arm]:
            records=[]
            for parent,m in results[name]['per_parent'].items():
                if m['split']!='validation':continue
                reference=results[control]['per_parent'][parent]
                records.append(dict(parent=parent,calibrated_mse=m['mse'],control_mse=reference['mse'],difference=m['mse']-reference['mse']))
            comparison[control]=dict(validation_mse_difference=results[name]['overall']['validation']['mse']-results[control]['overall']['validation']['mse'],parents_improved=sum(r['difference'] < -1e-12 for r in records),parents_worsened=sum(r['difference'] > 1e-12 for r in records),parents_tied=sum(abs(r['difference'])<=1e-12 for r in records),per_parent=records)
        comparisons[name]=comparison
    passes={arm:bool(results[arm+'_calibrated']['overall']['validation']['mse'] < min(results[control]['overall']['validation']['mse'] for control in ['uniform_composition','uniform_rates_calibrated'])) for arm in ['adjacent_rates','wider_rates']}
    summary=dict(status='completed',fits=fits,metrics={a:r['overall'] for a,r in results.items()},comparisons=comparisons,sequence_prior_passes_both_controls=passes,stop_extending_same_mapp_bridge=not any(passes.values()),seconds=time.perf_counter()-tick,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(FAMILY/'results/summary.json',summary);print(summary,flush=True)
if __name__=='__main__':main()
