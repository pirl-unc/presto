"""Frozen required-output and directional-gradient numerical admission."""
import argparse
import json
import os
import resource
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from independent_cut import ordered_bond_features
from project import dump
from rational import expectations
from reference import analytical_logs, counts, features

FAMILY = Path(__file__).resolve().parents[1]


def readout_weights(n):
    i = np.arange(n)
    w = (-1.)**i * (1+0.25*np.sin(i+1))
    return w / np.abs(w).sum()


def directions():
    d = np.zeros((3,41))
    d[0,:40] = np.sin(np.arange(1,41))
    d[1,:40] = np.cos(np.arange(1,41))
    d[2,-1] = 1
    return d / np.linalg.norm(d, axis=1)[:,None]


def worker(case_id):
    cfg = json.loads((FAMILY/'reproduce/launch.json').read_text())
    support = json.loads((FAMILY/'reproduce/training-support.json').read_text())
    cases = json.loads((FAMILY/'reproduce/cases.json').read_text())
    case = cases[case_id]
    block = support['blocks'][case['block_id']]
    torch.set_num_threads(1)
    result = dict(**case, status='running', accepted=False, condition=block['condition'],
        time_hours=block['time_hours'], n_residues=len(block['sequence']),
        n_required_spans=len(block['spans']), n_measurements=len(block['rows']),
        phase='forward', forwards=[])
    output = FAMILY/f'results/case-{case_id:03d}.json'
    archive = FAMILY/f'results/case-{case_id:03d}.npz'
    start = time.perf_counter()
    done = threading.Event()

    def exceeded(signum, frame):
        raise TimeoutError('120-second cap' if signum == signal.SIGALRM else '4-GiB peak RSS cap')

    signal.signal(signal.SIGALRM, exceeded)
    signal.signal(signal.SIGUSR1, exceeded)
    signal.setitimer(signal.ITIMER_REAL, cfg['runtime_limits']['seconds'])

    def watch():
        while not done.wait(0.1):
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != 'darwin':
                rss *= 1024
            if rss > cfg['runtime_limits']['rss_bytes']:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    threading.Thread(target=watch, daemon=True).start()
    raw = {}
    try:
        theta = np.array(support['initial_coefficients'])
        x = ordered_bond_features(block['sequence'], context=block['context_log_enzyme'])
        xn = features(block['sequence'], block['context_log_enzyme'])
        np.testing.assert_array_equal(x.numpy(), xn)
        logs = x@torch.tensor(theta)
        eta = torch.tensor(case['eta'], dtype=torch.float64)
        spans = np.array(block['spans'], dtype=int)
        a,b = spans[:,0],spans[:,1]-1
        weights = readout_weights(len(a))
        raw.update(theta=theta,x=xn,log_rates=logs.numpy(),spans=spans,weights=weights,
            directions=directions())
        size = len(block['sequence'])
        triangle = np.triu_indices(size)
        all_resolved = True
        for steps in cfg['substeps']:
            tick = time.perf_counter()
            with torch.no_grad():
                y = expectations(logs,eta,block['time_hours'],order=48,substeps=steps).numpy()
            v = y[a,b]
            valid = np.isfinite(v)&(v>0)
            lv = np.full(v.shape,np.nan)
            lv[valid] = np.log(v[valid])
            full = y[triangle]
            lengths = triangle[1]-triangle[0]+1
            raw[f'counts_{steps}'] = y.copy()
            raw[f'logs_{steps}'] = lv
            result['forwards'].append(dict(substeps=steps,seconds=time.perf_counter()-tick,
                required_nonpositive=int((v<=0).sum()), required_nonfinite=int((~np.isfinite(v)).sum()),
                full_negative_count=int((full<0).sum()),full_zero_count=int((full==0).sum()),
                full_minimum=float(full.min()),full_mass=float(full@lengths),
                required_minimum=float(v.min()), required_maximum=float(v.max())))
            all_resolved = all_resolved and bool(valid.all())
            np.savez_compressed(archive,**raw)
            dump(output,result)
        result['required_positive_finite'] = all_resolved
        if all_resolved:
            lv = np.stack([raw[f'logs_{s}'] for s in cfg['substeps']])
            delta = lv.max(0)-lv.min(0)
            raw['log_disagreement'] = delta
            result['max_log_disagreement'] = float(delta.max())
            result['n_unresolved_precision'] = int((delta>cfg['gate']['log_max_abs']).sum())
        else:
            result['max_log_disagreement'] = None
            result['n_unresolved_precision'] = None
        if case['eta'] == 0:
            expected = analytical_logs(xn@theta,spans,block['time_hours'])
            raw['analytical_logs'] = expected
            result['max_analytical_log_error'] = float(np.max(np.abs(raw['logs_8']-expected))) if all_resolved else None
        forward_pass = all_resolved and result['max_log_disagreement']<=cfg['gate']['log_max_abs']
        if case['eta'] == 0:
            forward_pass = forward_pass and result['max_analytical_log_error'] is not None and result['max_analytical_log_error']<=cfg['gate']['log_max_abs']
        result['forward_pass'] = forward_pass
        np.savez_compressed(archive,**raw)
        dump(output,result)
        if not forward_pass:
            result.update(status='unresolved_required_values',phase='complete',gradient_status='not_admitted')
            return
        result['phase'] = 'gradient'
        tick = time.perf_counter()
        param = torch.tensor(np.r_[theta,case['eta']],requires_grad=True)
        values = expectations(x@param[:40],param[40],block['time_hours'],order=48,substeps=2)
        readout = (values[a,b].log()*torch.tensor(weights)).sum()
        grad = torch.autograd.grad(readout,param)[0].detach().numpy()
        raw['gradient'] = grad
        result['autograd_seconds'] = time.perf_counter()-tick
        del values,readout,param
        coefficient = json.loads((FAMILY/'reproduce/coefficients.json').read_text())
        delta = cfg['gate']['fd_step']
        fd_logs = []
        tick = time.perf_counter()
        for direction in directions():
            pair = []
            for sign in [-1,1]:
                v = np.r_[theta,case['eta']]+sign*delta*direction
                y = counts(xn@v[:40],v[-1],block['time_hours'],8,coefficient)
                target = y[a,b]
                if not np.isfinite(target).all() or not (target>0).all():
                    raise FloatingPointError('Unresolved independent finite-difference log values')
                pair.append(np.log(target))
            fd_logs.append(pair)
        fd_logs = np.array(fd_logs)
        fd = (fd_logs[:,1]@weights-fd_logs[:,0]@weights)/(2*delta)
        actual = directions()@grad
        error = np.abs(fd-actual)
        tolerance = cfg['gate']['fd_absolute']+cfg['gate']['fd_relative']*np.abs(fd)
        raw.update(fd_logs=fd_logs,fd_directions=fd,autograd_directions=actual,fd_error=error,fd_tolerance=tolerance)
        result.update(independent_fd_seconds=time.perf_counter()-tick,
            gradient_error=error.tolist(),gradient_tolerance=tolerance.tolist(),
            gradient_pass=bool(np.isfinite(grad).all() and np.all(error<=tolerance)),
            gradient_status='evaluated',phase='complete',status='completed')
        result['accepted'] = result['gradient_pass']
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),accepted=False)
    finally:
        done.set()
        signal.setitimer(signal.ITIMER_REAL,0)
        if raw:
            np.savez_compressed(archive,**raw)
        result['seconds'] = time.perf_counter()-start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_rss_bytes'] = rss if sys.platform=='darwin' else rss*1024
        dump(output,result)
        print(json.dumps({k:result.get(k) for k in ['case_id','block_id','eta','status','accepted','seconds','max_log_disagreement','max_analytical_log_error']}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('case_id',type=int)
    worker(p.parse_args().case_id)
