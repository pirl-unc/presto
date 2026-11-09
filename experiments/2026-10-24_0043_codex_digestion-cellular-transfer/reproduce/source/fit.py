"""Fixed 2000-update single-parameter biological transfer fit."""
import gzip
import json
import math
import resource
import sys
import time
import numpy as np
import torch
from core import (FAMILY, cfg, dump, evaluate, load, objective_torch, prepare,
                  sampled, sha, torch_data)


def main():
    start=time.perf_counter();arm=sys.argv[1];c=cfg()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    receipt=json.loads((FAMILY/'reproduce/fitting.json').read_text())
    for p,h in receipt['files'].items():assert sha(FAMILY/p)==h,p
    for p,h in c['source_hashes'].items():assert sha(p)==h,p
    data_root=__import__('pathlib').Path(c['dataset_path'])
    for p,h in json.loads((data_root/'manifest.json').read_text())['files'].items():assert sha(data_root/p)==h['sha256']
    q=json.loads((FAMILY/'results/qualification.json').read_text());cap=q['fit_caps'][arm]
    assert q['status']=='completed' and cap['admitted']
    out=FAMILY/'results'/arm;out.mkdir()
    groups=load(arm);data=torch_data(groups)
    parameter=torch.tensor(math.log(c['initial_delta']),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([parameter],lr=c['learning_rate'],betas=tuple(c['betas']),eps=c['epsilon'])
    gen=np.random.default_rng(c['seed']);history=[];states=[];rng={}
    status=dict(arm=arm,status='running',updates=0)
    dump(out/'status.json',status)
    try:
        for step in range(1,c['steps']+1):
            optimizer.zero_grad();before=float(parameter.detach())
            if step in [1,100,1000,2000]:rng[str(step)]=gen.bit_generator.state
            if arm=='wider_hybrid':
                value,exact,records=prepare(before,groups);gradient,work=sampled(records,c['sample_count'],gen)
                parameter.grad=torch.tensor(gradient,dtype=torch.float64)
            else:
                value_tensor=objective_torch(parameter,data);value_tensor.backward()
                value=float(value_tensor.detach());gradient=float(parameter.grad);exact=gradient;work=None
            assert math.isfinite(value) and math.isfinite(gradient)
            optimizer.step();after=float(parameter.detach());assert math.isfinite(after)
            state=optimizer.state[parameter]
            states.append(dict(step=step,before=before,gradient=gradient,after=after,moment=float(state['exp_avg']),square=float(state['exp_avg_sq'])))
            if step==1 or step%100==0:
                elapsed=time.perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                history.append(dict(step=step,training_loss=value,logdelta=after,delta=math.exp(after),gradient=gradient,reference_gradient=exact,seconds=elapsed,work=work))
                status.update(updates=step,seconds=elapsed,peak_rss_bytes=rss)
                dump(out/'status.json',status);dump(out/'history.json',history);dump(out/'optimizer.json',states);dump(out/'rng.json',rng)
                print(arm,step,value,after,elapsed,flush=True)
            assert time.perf_counter()-start<=cap['seconds'], 'registered runtime cap'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<c['preflight']['peak_rss_bytes'], 'registered RSS cap'
        final=float(parameter.detach());rows,metrics=evaluate(final,groups)
        with gzip.open(out/'predictions.jsonl.gz','wt') as h:
            for row in rows:h.write(json.dumps(row,allow_nan=False)+'\n')
        dump(out/'evaluation.json',metrics)
        dump(out/'checkpoint.json',dict(logdelta=final,delta=math.exp(final),initial_logdelta=math.log(c['initial_delta']),steps=c['steps'],arm=arm,trainable_parameters=1,frozen_sequence_parameters=0 if arm=='uniform_rates' else 40 if arm=='adjacent_rates' else 120))
        dump(out/'optimizer.json',states);dump(out/'rng.json',rng)
        status.update(status='completed',updates=c['steps'],seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,maximum_gradient=max(abs(s['gradient']) for s in states),parameter_change=final-math.log(c['initial_delta']))
        dump(out/'status.json',status)
    except BaseException as error:
        status.update(status='failed',error=repr(error),updates=len(states),seconds=time.perf_counter()-start)
        dump(out/'status.json',status);dump(out/'optimizer.json',states);dump(out/'rng.json',rng)
        raise

if __name__=='__main__':main()
