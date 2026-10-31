"""Frozen independent metric arithmetic from the preceding verified experiment."""
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr,spearmanr

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
