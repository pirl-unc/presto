"""Frozen paired biological fits; preserve every failure and final prediction."""
import argparse
import gzip
import json
import os
import resource
import signal
import sys
import threading
import time
import numpy as np
import torch
from core import FAMILY,LAST_QUERY,analytical_amount,block_loss,dump,fragment_amount,load,penalty,sha
from rational import expectations


def evaluate(p,arm,groups,train,feature,observation,out):
    m=feature['allocated_dimensions'];amounts=[];responses=[];values=[];checks=[]
    eta=torch.tanh(p[-1]) if arm=='fragment' else None
    for g in groups:
        if arm=='analytical':
            amount=analytical_amount(p[:40],g)
        else:
            amount=torch.full_like(g['y'],float('nan'))
            for t in sorted(set(g['t'].tolist())):
                idx=torch.nonzero(g['t']==t).flatten()
                dump(out/'evaluation-current.json',dict(condition=g['condition'],split=g['split'],time=t))
                value,stats=fragment_amount(p[:40],eta,g,idx,check=True)
                amount[idx]=value
                checks.append(dict(condition=g['condition'],split=g['split'],time=t,n_rows=len(idx),**stats))
                dump(out/'evaluation-checks.json',checks)
        assert torch.isfinite(amount).all()
        response=g['z']@p[40:40+m]
        amounts.append(amount);responses.append(response);values.append(amount+response)
    rows=observation.predictions(groups,values,amounts,responses)
    offsets={g['condition']:float((g['w']*(g['y']-value)).sum()/g['w'].sum()) for g,value in zip(groups,values) if g['split']=='train'}
    for row in rows:
        beta=offsets.get(row['condition_id'])
        row['profiled_training_offset']=beta
        row['profiled_training_prediction']=None if beta is None else row['raw_prediction']+beta
    with gzip.open(out/'predictions.jsonl.gz','wt') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    metrics={split:observation.metrics([r for r in rows if r['split']==split]) for split in ['train','val']}
    per_condition={}
    # The existing metrics helper requires split weights sum to one. Normalize only
    # diagnostic per-condition weights; preserve original global weights in dumps.
    for g in groups:
        rr=[dict(r) for r in rows if r['condition_id']==g['condition']]
        total=sum(r['weight'] for r in rr)
        for r in rr:r['weight']/=total
        per_condition[g['condition']]=observation.metrics(rr)
    train_rows=[r for r in rows if r['split']=='train']
    joint=sum(r['weight']*(r['profiled_training_prediction']-r['target'])**2 for r in train_rows)
    regularizer=float(penalty(p,m))+(0.005*float(eta.square()) if eta is not None else 0.)
    rates=torch.cat([(g['x']@p[:40]).exp() for g in groups])
    prediction=np.array([r['raw_prediction'] for r in rows])
    response=np.array([r['log10_response'] for r in rows])
    amount=np.array([r['log10_amount'] for r in rows])
    return dict(metrics=metrics,per_condition=per_condition,training_data_mse=joint,training_objective=joint+regularizer,regularizer=regularizer,
        n_predictions=len(rows),eta=None if eta is None else float(eta),minimum_rate_per_hour=float(rates.min()),maximum_rate_per_hour=float(rates.max()),
        minimum_log10_amount=float(amount.min()),maximum_log10_amount=float(amount.max()),response_range=float(np.ptp(response)),prediction_range=float(np.ptp(prediction)),
        all_final_queries_resolved=True,n_final_numeric_blocks=len(checks),maximum_final_log_disagreement=max((c['max_log_disagreement'] for c in checks),default=0.),
        development_scope='Profiled offsets use development labels solely for within-condition shape evaluation; no absolute out-of-sample signal calibration.')


def fit(arm):
    start=time.perf_counter()
    cfg=json.loads((FAMILY/'reproduce/fitting.json').read_text())
    frozen=json.loads((FAMILY/'reproduce/fitting-code.json').read_text())
    for rel,expected in frozen.items():assert sha(FAMILY/rel)==expected,rel
    assert cfg['biological_fit_allowed'] and arm in cfg['arms']
    assert sha(FAMILY/'results/qualification-summary.json')==cfg['qualification_summary_sha256']
    out=FAMILY/'results'/arm
    assert not out.exists(),'Refuse to overwrite an attempted arm'
    out.mkdir()
    state=dict(arm=arm,status='running',phase='load',updates=0,started_at=time.time())
    dump(out/'status.json',state)
    done=threading.Event()
    def exceeded(signum,frame):
        raise TimeoutError(f"{cfg['runtime_limits']['seconds_per_arm']}-second arm cap" if signum==signal.SIGALRM else '4-GiB peak RSS cap')
    signal.signal(signal.SIGALRM,exceeded);signal.signal(signal.SIGUSR1,exceeded)
    signal.setitimer(signal.ITIMER_REAL,cfg['runtime_limits']['seconds_per_arm'])
    def watch():
        while not done.wait(.1):
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
            if rss>cfg['runtime_limits']['rss_bytes']:
                os.kill(os.getpid(),signal.SIGUSR1);return
    threading.Thread(target=watch,daemon=True).start()
    history=[];before=[];gradients=[];after=[];moments=[];squares=[]
    p=None;optimizer=None;gradient_counts=None;update_counts=None;material_gradient=None;material_update=None
    def save():
        if p is not None:
            torch.save(dict(parameters=p.detach().cpu(),optimizer=None if optimizer is None else optimizer.state_dict(),last_gradient=None if p.grad is None else p.grad.detach().cpu(),updates=state['updates'],arm=arm,config=cfg),out/'checkpoint.pt')
        if before:
            np.savez_compressed(out/'updates.npz',parameters_before=np.stack(before),gradients=np.stack(gradients),parameters_after=np.stack(after),adam_mean=np.stack(moments),adam_square=np.stack(squares),block_ids=np.array([h['block_id'] for h in history]))
        dump(out/'history.json',history)
        dump(out/'status.json',state)
    try:
        _,observation,groups,train,feature,initial,blocks=load();m=feature['allocated_dimensions']
        np.testing.assert_array_equal(initial.numpy(),np.array(cfg['initial_common_parameters']))
        np.testing.assert_allclose([b['probability'] for b in blocks],cfg['block_probabilities'],atol=0,rtol=0)
        q=initial if arm=='analytical' else torch.cat([initial,initial.new_zeros(1)])
        p=torch.nn.Parameter(q.clone())
        torch.manual_seed(cfg['seed'])
        optimizer=torch.optim.Adam([p],lr=cfg['optimizer']['lr'],betas=tuple(cfg['optimizer']['betas']),eps=cfg['optimizer']['eps'],weight_decay=0.)
        gradient_counts=torch.zeros_like(p,dtype=torch.long);update_counts=gradient_counts.clone();material_gradient=gradient_counts.clone();material_update=gradient_counts.clone()
        dump(out/'initial.json',dict(parameters=p.detach().tolist(),common_sha256=sha(FAMILY/'results/qualification.npz'),n_parameters=p.numel()))
        state['phase']='training'
        for step,bid in enumerate(cfg['block_schedule']):
            tick=time.perf_counter();b=blocks[bid];state.update(current_block=bid,current_step=step+1)
            optimizer.zero_grad(set_to_none=True)
            eta=torch.tanh(p[-1]) if arm=='fragment' else None
            total,data,pre=block_loss(p[:78],train,m,b,eta=eta,check=True)
            if eta is not None:total=total+.005*eta.square()
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite training loss')
            total.backward()
            if not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Nonfinite training gradient')
            old=p.detach().clone();grad=p.grad.detach().clone()
            optimizer.step()
            if not bool(torch.isfinite(p).all()):raise FloatingPointError('Nonfinite updated parameters')
            delta=p.detach()-old
            gradient_counts+=(grad!=0);update_counts+=(delta!=0)
            material_gradient+=(grad.abs()>1e-12);material_update+=(delta.abs()>1e-12)
            before.append(old.numpy());gradients.append(grad.numpy());after.append(p.detach().numpy().copy())
            opt=optimizer.state[p];moments.append(opt['exp_avg'].numpy().copy());squares.append(opt['exp_avg_sq'].numpy().copy())
            entry=dict(step=step+1,block_id=bid,condition=b['condition'],time=b['time'],pre_total_loss=float(total.detach()),pre_data_loss=float(data.detach()),eta_before=None if eta is None else float(eta.detach()),eta_after=None if eta is None else float(torch.tanh(p[-1]).detach()),gradient_l2=float(grad.norm()),update_l2=float(delta.norm()),pre_numerics=pre)
            history.append(entry);state['updates']=step+1
            del total,data
            if arm=='fragment':
                with torch.no_grad():
                    _,post=fragment_amount(p[:40],torch.tanh(p[-1]),train[b['group']],b['indices'],check=True)
                entry['post_numerics']=post
            entry['seconds']=time.perf_counter()-tick
            if (step+1)%10==0:
                state['elapsed_seconds']=time.perf_counter()-start;save()
                print(json.dumps(dict(arm=arm,step=step+1,eta=entry['eta_after'],last_block_loss=entry['pre_total_loss'],elapsed=state['elapsed_seconds'])),flush=True)
        state['phase']='evaluation';save()
        with torch.no_grad():evaluation=evaluate(p.detach(),arm,groups,train,feature,observation,out)
        dump(out/'evaluation.json',evaluation)
        state.update(status='completed',phase='complete',qualified_final=True)
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc),qualified_final=False)
        if isinstance(exc,FloatingPointError) and arm=='fragment' and p is not None:
            try:
                query=LAST_QUERY.copy()
                g=next(g for g in groups if g['condition']==query['condition'])
                idx=torch.nonzero(g['t']==query['time']).flatten();t=query['time']
                with torch.no_grad():
                    arrays={f'counts_{n}':expectations(g['x']@p[:40],torch.tanh(p[-1]),t,order=48,substeps=n).numpy() for n in [2,4]}
                np.savez_compressed(out/'failure-values.npz',parameters=p.detach().numpy(),starts=g['a'][idx].numpy(),ends=g['b'][idx].numpy(),time_hours=t,**arrays)
                state['failure_query']=dict(condition=g['condition'],split=g['split'],time=t,n_rows=len(idx))
            except Exception as diagnostic_error:
                state['failure_snapshot_error']=str(diagnostic_error)
    finally:
        done.set();signal.setitimer(signal.ITIMER_REAL,0)
        state['seconds']=time.perf_counter()-start
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        state['peak_rss_bytes']=rss if sys.platform=='darwin' else rss*1024
        if gradient_counts is not None:
            state['coverage']=dict(nonzero_gradient_counts=gradient_counts.tolist(),nonzero_update_counts=update_counts.tolist(),material_gradient_counts=material_gradient.tolist(),material_update_counts=material_update.tolist(),all_allocated_materially_trained=bool(((material_gradient>0)&(material_update>0)).all()))
            state['qualified_final']=state.get('qualified_final',False) and state['coverage']['all_allocated_materially_trained']
        save()
        print(json.dumps({k:v for k,v in state.items() if k!='coverage'}),flush=True)
    return 0 if state['status']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=['analytical','fragment'])
    sys.exit(fit(parser.parse_args().arm))
