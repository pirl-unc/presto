"""Fixed-budget exact factorial and prospectively named sampled hybrid."""
import argparse
import json
import resource
import time
import numpy as np
import torch
from core import FAMILY, dump, evaluate, load, objective, prepare, sha, variant
from sampled import estimate


def main(arm):
    started=time.perf_counter();cfg,old,groups,feature,initial=load()
    for path,h in json.loads((FAMILY/'reproduce/execution-code.json').read_text()).items():assert sha(FAMILY/path)==h
    gate=json.loads((FAMILY/'results/qualification.json').read_text())
    hybrid=arm=='both_hybrid';assert gate['hybrid_admitted' if hybrid else 'exact_admitted']
    out=FAMILY/'results'/arm;assert not out.exists();out.mkdir()
    state=dict(arm=arm,status='running',updates=0);dump(out/'status.json',state)
    try:
        gg,t0,g0=variant(groups,initial,arm);train=[g for g in gg if g['split']=='train'];nx=len(t0)
        p=torch.nn.Parameter(torch.cat([t0,g0]));optimizer=torch.optim.Adam([p],lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['epsilon'])
        generator=torch.Generator().manual_seed(cfg['hybrid']['seed']);selected=[1,100,1000,2000]
        n=len(p);arrays={k:np.empty((cfg['steps'],n)) for k in ['before','gradient','after','moment','square']};grad_count=np.zeros(n,dtype=int);update_count=np.zeros(n,dtype=int)
        history=[];rng={};work_total={};tick=time.perf_counter();cap=cfg['caps']['hybrid_seconds' if hybrid else 'exact_seconds']
        for i in range(cfg['steps']):
            optimizer.zero_grad();v,data=objective(p[:nx],p[nx:],train);assert bool(torch.isfinite(v));v.backward()
            if hybrid:
                if i+1 in selected:rng[str(i+1)]=generator.get_state().tolist()
                with torch.no_grad():
                    rr=prepare(p[:nx],p[nx:],train);gradient,work=estimate(rr,512,generator,cfg['hybrid']['proposal']);reg=.01*p[:nx].clone();reg[0]=0;p.grad[:nx]=gradient+reg
                for key in ['partitions','bond_draws','presence_terms']:work_total[key]=work_total.get(key,0)+work[key]
            assert bool(torch.isfinite(p.grad).all());arrays['before'][i]=p.detach().numpy();arrays['gradient'][i]=p.grad.numpy();grad_count+=np.abs(arrays['gradient'][i])>1e-12
            optimizer.step();arrays['after'][i]=p.detach().numpy();arrays['moment'][i]=optimizer.state[p]['exp_avg'].numpy();arrays['square'][i]=optimizer.state[p]['exp_avg_sq'].numpy();update_count+=np.abs(arrays['after'][i]-arrays['before'][i])>1e-12
            if (i+1)%100==0:
                row=dict(step=i+1,preupdate_loss=float(v.detach()),preupdate_data_mse=float(data.detach()),seconds=time.perf_counter()-tick);history.append(row);print(arm,json.dumps(row),flush=True)
                state.update(updates=i+1,seconds=row['seconds']);dump(out/'status.json',state)
                assert time.perf_counter()-started<cap,'Registered runtime cap exceeded'
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<cfg['caps']['peak_rss_bytes'],'Registered memory cap exceeded'
        np.savez_compressed(out/'updates.npz',**arrays)
        dump(out/'rng.json',rng);dump(out/'history.json',history)
        v,data=objective(p[:nx],p[nx:],train);gradient=torch.autograd.grad(v,p)[0]
        result=evaluate(old,p[:nx].detach(),p[nx:].detach(),gg,out)
        result.update(final_objective=float(v.detach()),final_data_mse=float(data.detach()),final_exact_gradient=gradient.tolist(),gradient_counts=grad_count.tolist(),update_counts=update_count.tolist(),all_allocated_materially_trained=bool((grad_count>0).all() and (update_count>0).all()),parameters=n,rate_parameters=nx,response_parameters=n-nx,work=work_total)
        dump(out/'evaluation.json',result)
        torch.save(dict(parameters=p.detach(),rate_parameters=nx,initial=torch.cat([t0,g0]),optimizer=optimizer.state_dict(),rng=generator.get_state(),arm=arm,steps=cfg['steps']),out/'checkpoint.pt')
        state.update(status='completed',updates=cfg['steps'],seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert state['seconds']<cap and state['peak_rss_bytes']<cfg['caps']['peak_rss_bytes']
        dump(out/'status.json',state);print(json.dumps(dict(status=state,metrics=result['metrics'])),flush=True)
    except BaseException as e:
        state.update(status='failed',error=repr(e),seconds=time.perf_counter()-started);dump(out/'status.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=['base','rate_context','response_termini','both','both_hybrid']);main(parser.parse_args().arm)
