"""Independent optimizer arithmetic, physics readouts, and metric closure."""
import argparse
from collections import defaultdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.stats import pearsonr,spearmanr
import torch
from core import FAMILY,dump,independent_profiled,load,sha


def metrics(rows):
    p=np.array([r['centered_prediction'] for r in rows]);y=np.array([r['centered_target'] for r in rows]);w=np.array([r['weight'] for r in rows])
    assert abs(w.sum()-1)<1e-12
    curves=defaultdict(list)
    for r in rows:curves[r['curve_id']].append(r)
    rp=[];ry=[];rw=[]
    for curve in curves.values():
        ref=next(r for r in curve if r['time']==r['reference_time'])
        for r in curve:
            if r is ref:continue
            rp.append(r['raw_prediction']-ref['raw_prediction']);ry.append(r['target']-ref['target']);rw.append(r['weight']*len(curve)/(len(curve)-1))
    rp,ry,rw=map(np.array,(rp,ry,rw));assert abs(rw.sum()-1)<1e-12
    def cor(a,b,fn):return None if np.ptp(a)<1e-12 or np.ptp(b)<1e-12 else float(fn(a,b).statistic)
    return dict(shape_mse=float(w@(p-y)**2),shape_rmse=float(np.sqrt(w@(p-y)**2)),shape_pearson=cor(p,y,pearsonr),shape_spearman=cor(p,y,spearmanr),prediction_sd=float(p.std()),target_sd=float(y.std()),ratio_rmse=float(np.sqrt(rw@(rp-ry)**2)),ratio_pearson=cor(rp,ry,pearsonr),ratio_spearman=cor(rp,ry,spearmanr))


def verify(arm):
    started=time.perf_counter();cfg,_,groups,train,feature,initial,blocks=load();m=feature['allocated_dimensions']
    contract=json.loads((FAMILY/'reproduce/verification.json').read_text())
    for path,expected in contract['reference_hashes'].items():assert sha(path)==expected,path
    fitcfg=json.loads((FAMILY/'reproduce/fitting.json').read_text())
    for rel,expected in json.loads((FAMILY/'reproduce/fitting-code.json').read_text()).items():assert sha(FAMILY/rel)==expected,rel
    out=FAMILY/'results'/arm;status=json.loads((out/'status.json').read_text());assert status['status']=='completed' and status['updates']==200
    checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True);p=checkpoint['parameters'].numpy()
    assert checkpoint['config']==fitcfg
    updates=np.load(out/'updates.npz');before=updates['parameters_before'];grad=updates['gradients'];after=updates['parameters_after']
    np.testing.assert_array_equal(before[0],np.r_[initial.numpy(),0.] if arm=='fragment' else initial.numpy())
    np.testing.assert_array_equal(before[1:],after[:-1]);np.testing.assert_array_equal(p,after[-1])
    np.testing.assert_array_equal(updates['block_ids'],fitcfg['block_schedule'])
    mean=np.zeros_like(p);square=np.zeros_like(p);max_update=0.;max_gradient=0.
    for i,g in enumerate(grad):
        mean=.9*mean+.1*g;square=.999*square+.001*g*g
        candidate=before[i]-(.03/(1-.9**(i+1)))*mean/(np.sqrt(square)/math.sqrt(1-.999**(i+1))+1e-8)
        np.testing.assert_allclose(mean,updates['adam_mean'][i],atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(square,updates['adam_square'][i],atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(candidate,after[i],atol=1e-12,rtol=1e-10)
        max_update=max(max_update,float(np.max(np.abs(candidate-after[i]))))
        if arm=='analytical':
            b=blocks[updates['block_ids'][i]]
            _,gg=independent_profiled(before[i],train,m,b)
            np.testing.assert_allclose(gg,g,atol=1e-8,rtol=1e-7)
            max_gradient=max(max_gradient,float(np.max(np.abs(gg-g))))
    coverage=status['coverage']
    np.testing.assert_array_equal(coverage['material_gradient_counts'],(np.abs(grad)>1e-12).sum(0))
    np.testing.assert_array_equal(coverage['material_update_counts'],(np.abs(after-before)>1e-12).sum(0))
    assert coverage['all_allocated_materially_trained']
    history=json.loads((out/'history.json').read_text());assert len(history)==200
    max_centered_residual=0.
    for i,h in enumerate(history):
        b=blocks[updates['block_ids'][i]];g=train[b['group']]
        max_centered_residual=max(max_centered_residual,abs(h['pre_numerics']['condition_weighted_residual']))
        if arm=='fragment':
            checks=h['pre_numerics']['condition_numeric_checks']
            assert sorted(r['time'] for r in checks)==sorted(set(g['t'].tolist()))
            assert all(r['max_log_disagreement']<=1e-5 for r in checks)
            assert h['post_numerics']['max_log_disagreement']<=1e-5
    assert max_centered_residual<1e-10
    source=Path(cfg['admission_family'])/'code/reference.py';spec=importlib.util.spec_from_file_location('independent_fragment_reference',source);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    coefficients=json.loads((Path(cfg['admission_family'])/'reproduce/coefficients.json').read_text())
    with gzip.open(out/'predictions.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
    assert len(rows)==105452 and {r['split'] for r in rows}=={'train','val'}
    by_condition=defaultdict(list)
    for r in rows:by_condition[r['condition_id']].append(r)
    max_amount=0.;max_response=0.;independent_amounts=[];full_reference={};joint=0.;offset_gap=0.
    for gi,g in enumerate(groups):
        rr=by_condition[g['condition']];assert len(rr)==len(g['rows'])
        for saved,expected in zip(rr,g['rows']):
            for key,value in expected.items():assert saved[key]==value,(g['condition'],key)
        np.testing.assert_array_equal([r['weight'] for r in rr],g['w'].numpy())
        x=reference.features(g['sequence'],float(g['x'][0,-1]));logs=x@p[:40]
        amount=np.empty(len(rr))
        for ti,t in enumerate(sorted(set(g['t'].tolist()))):
            ix=np.flatnonzero(g['t'].numpy()==t);spans=np.column_stack([g['a'].numpy()[ix],g['b'].numpy()[ix]])
            if arm=='analytical':lv=reference.analytical_logs(logs,spans,t)
            else:
                matrix=reference.counts(logs,float(np.tanh(p[-1])),t,8,coefficients)
                full_reference[f'g{gi}_t{ti}']=matrix
                vv=matrix[spans[:,0],spans[:,1]-1];assert np.all(np.isfinite(vv)&(vv>0))
                lv=np.log(vv)
            amount[ix]=lv/math.log(10)
        saved_amount=np.array([r['log10_amount'] for r in rr]);err=float(np.max(np.abs(saved_amount-amount))*math.log(10));assert err<=1e-5
        max_amount=max(max_amount,err);independent_amounts.extend(amount.tolist())
        raw=g['raw_response'];z=(raw/np.array(feature['scale']))@np.array(feature['basis']);response=z@p[40:78]
        max_response=max(max_response,float(np.max(np.abs(response-np.array([r['log10_response'] for r in rr])))))
        np.testing.assert_allclose(response,[r['log10_response'] for r in rr],atol=1e-10,rtol=1e-10)
        pred=np.array([r['raw_prediction'] for r in rr]);target=g['y'].numpy();w=g['w'].numpy()
        np.testing.assert_allclose(pred,saved_amount+response,atol=1e-10,rtol=1e-10)
        pc=pred-w@pred/w.sum();yc=target-w@target/w.sum();offset=w@(target-pred)/w.sum()
        np.testing.assert_allclose(pc,[r['centered_prediction'] for r in rr],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(yc,[r['centered_target'] for r in rr],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(offset,[r['profiled_offset'] for r in rr],atol=1e-10,rtol=1e-10)
        if g['split']=='train':
            beta=offset
            np.testing.assert_allclose([r['profiled_training_offset'] for r in rr],beta,atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose([r['profiled_training_prediction'] for r in rr],pred+beta,atol=1e-10,rtol=1e-10)
            joint+=w@(pred+beta-target)**2;offset_gap+=w.sum()*(float(rr[0]['profiled_training_offset'])-offset)**2
    evaluated=json.loads((out/'evaluation.json').read_text())
    computed={split:metrics([r for r in rows if r['split']==split]) for split in ['train','val']}
    for split in computed:
        for key,value in computed[split].items():np.testing.assert_allclose(value,evaluated['metrics'][split][key],atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(joint,evaluated['training_data_mse'],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(joint-computed['train']['shape_mse'],offset_gap,atol=1e-10,rtol=1e-10)
    assert offset_gap < 1e-20
    regularizer=.005*(p[1:40]@p[1:40]+p[40:78]@p[40:78])
    if arm=='fragment':regularizer+=.005*np.tanh(p[-1])**2
    np.testing.assert_allclose(regularizer,evaluated['regularizer'],atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(joint+regularizer,evaluated['training_objective'],atol=1e-10,rtol=1e-10)
    directional=[]
    if arm=='fragment':
        for step in contract['fragment_gradient_steps']:
            i=step-1;b=blocks[updates['block_ids'][i]];g=train[b['group']];ix=b['indices'].numpy();x=reference.features(g['sequence'],float(g['x'][0,-1]));spans=np.column_stack([g['a'].numpy()[ix],g['b'].numpy()[ix]]);z=g['z'].numpy()[ix];y=g['y'].numpy()[ix];w=g['w'].numpy()[ix]/b['probability']
            # Solve the whole-condition offset independently at this state.
            # Hold it fixed under individual-block finite differences: allowing
            # it to vary would test a different, biased block-gradient estimator.
            point0=before[i];all_amount=np.empty(len(g['rows']))
            for tt in sorted(set(g['t'].tolist())):
                ii=np.flatnonzero(g['t'].numpy()==tt)
                matrix=reference.counts(x@point0[:40],float(np.tanh(point0[-1])),tt,8,coefficients)
                vv=matrix[g['a'].numpy()[ii],g['b'].numpy()[ii]-1];assert np.all(vv>0)
                all_amount[ii]=np.log10(vv)
            full_weight=g['w'].numpy();base=all_amount+g['z'].numpy()@point0[40:78]
            beta=full_weight@(g['y'].numpy()-base)/full_weight.sum()
            history=json.loads((out/'history.json').read_text())
            np.testing.assert_allclose(beta,history[i]['pre_numerics']['profiled_offset'],atol=1e-9,rtol=1e-9)
            def loss(point):
                eta=np.tanh(point[-1]);mat=reference.counts(x@point[:40],eta,b['time'],8,coefficients);v=mat[spans[:,0],spans[:,1]-1];assert np.all(v>0)
                residual=np.log10(v)+z@point[40:78]+beta-y
                total=w@(residual**2)+.005*(point[1:40]@point[1:40]+point[40:78]@point[40:78]+eta*eta)
                return total,residual
            _,residual=loss(point0);linear=2*(w*residual)@z+.01*point0[40:78]
            np.testing.assert_allclose(linear,grad[i,40:78],atol=1e-8,rtol=1e-7)
            directions=np.zeros((3,79));directions[0,:40]=np.sin(np.arange(1,41));directions[1,:40]=np.cos(np.arange(1,41));directions[2,-1]=1;directions/=np.linalg.norm(directions,axis=1)[:,None]
            for di,d in enumerate(directions):
                h=contract['fd_step'];fd=(loss(before[i]+h*d)[0]-loss(before[i]-h*d)[0])/(2*h);actual=grad[i]@d;error=abs(fd-actual);tol=1e-5+1e-6*abs(fd);assert error<=tol
                directional.append(dict(step=step,block_id=int(updates['block_ids'][i]),direction=di,independent_fd=float(fd),stored_gradient=float(actual),absolute_error=float(error),tolerance=float(tol)))
        np.savez_compressed(out/'independent-reference-arrays.npz',**full_reference)
    np.savez_compressed(out/'independent-amounts.npz',log10_amount=np.array(independent_amounts))
    result=dict(all_pass=True,arm=arm,updates_replayed=200,maximum_adam_update_error=max_update,analytical_gradient_checks=200 if arm=='analytical' else 0,maximum_analytical_gradient_error=max_gradient,fragment_directional_checks=directional,n_predictions_verified=len(rows),maximum_required_natural_log_error=max_amount,maximum_response_error=max_response,metrics=computed,training_data_mse=joint,training_offset_optimization_gap=offset_gap,all_allocated_materially_trained=True,maximum_condition_weighted_residual=max_centered_residual,seconds=time.perf_counter()-started,scope='All source identities/targets/weights checked against the unchanged pinned observation loader; all final physical amounts independently recomputed with direct analytical or NumPy eight-step fragment equations. All200 Adam updates replayed; analytical gradients all200, fragment whole-condition offsets and fixed-offset block gradients at three fixed steps/nine directions plus38linear components at each. Other fragment gradients not independently recomputed.')
    dump(out/'independent-verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=['analytical','fragment']);verify(parser.parse_args().arm)
