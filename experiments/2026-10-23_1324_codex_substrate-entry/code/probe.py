"""Qualify continuous-path hybrid gradients before actual sampled training."""

import json
import math
import resource
import time

import numpy as np
import torch

from common import FAMILY, designs, dump, entry_objective, load_data, prepare_points, sha
from qualify import ctmc
from sampled import event_gradients, training_gradient


def main():
    start=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=json.loads((FAMILY/'reproduce/probe.json').read_text())
    for name,h in config['source_hashes'].items():assert sha(FAMILY/name)==h
    cfg,groups=load_data();groups=[prepare_points(g) for g in groups];design,basis,audit=designs(groups,'sequence_context')
    logs=np.log([.08,.18,.45,.9,1.4]);spans,mass,jac=ctmc(logs,math.log(.3),2.)
    starts=torch.tensor([a for a,b in spans]);ends=torch.tensor([b for a,b in spans]);coeff=torch.linspace(.2,1.3,len(spans),dtype=torch.float64)
    toy=[]
    with torch.no_grad():
        for seed in range(32):
            full,reduced,_=event_gradients(torch.tensor(logs),torch.tensor(math.log(.3),dtype=torch.float64),torch.eye(5,dtype=torch.float64),2.,starts,ends,coeff,4096,torch.Generator().manual_seed(20261023+seed))
            toy.append(dict(full=full.tolist(),integrated=reduced.tolist()))
    target=jac@coeff.numpy();checks={}
    for method in ('full','integrated'):
        estimates=np.array([r[method] for r in toy]);se=estimates.std(0,ddof=1)/np.sqrt(len(estimates));z=np.abs(estimates.mean(0)-target)/se
        assert np.max(z)<6
        checks[method]=dict(reference=target.tolist(),mean=estimates.mean(0).tolist(),standard_error=se.tolist(),max_z=float(z.max()))
    dump(FAMILY/'results/integrated-score-qualification.json',dict(checks=checks,estimates=toy,continuous_ctmc_reference=True))
    theta=torch.zeros(41,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():theta[0]=math.log(.01)
    loss,_=entry_objective(theta,groups,design,entry=True,order=config['order']);reference=torch.autograd.grad(loss,theta)[0].detach().numpy()
    rows=[]
    for count in config['counts']:
        tick=time.perf_counter();estimates=[];work={}
        with torch.no_grad():
            for rep in range(config['groups']):
                full,integrated,w=training_gradient(theta,groups,design,order=config['order'],count=count,generator=torch.Generator().manual_seed(20261023+count*1000+rep))
                estimates.append(dict(full=full.tolist(),integrated=integrated.tolist()))
                for k,v in w.items():work[k]=work.get(k,0)+v
        statistics={}
        for method in ('full','integrated'):
            values=np.array([r[method] for r in estimates]);difference=values-reference
            statistics[method]=dict(rms_relative_error=float(np.sqrt(np.mean(np.sum(difference**2,axis=1)))/np.linalg.norm(reference)),
                mean_gradient_error=float(np.linalg.norm(values.mean(0)-reference)),reference_norm=float(np.linalg.norm(reference)),
                mean=values.mean(0).tolist(),coordinate_standard_error=(values.std(0,ddof=1)/np.sqrt(len(values))).tolist())
        record=dict(count=count,groups=config['groups'],statistics=statistics,estimates=estimates,work=work,seconds=time.perf_counter()-tick)
        rows.append(record);print(count,{k:v['rms_relative_error'] for k,v in statistics.items()},record['seconds'],flush=True)
    dump(FAMILY/'results/training-gradient-probes.json',dict(state='initial',reference=reference.tolist(),rows=rows))
    dump(FAMILY/'results/probe-runtime.json',dict(wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
