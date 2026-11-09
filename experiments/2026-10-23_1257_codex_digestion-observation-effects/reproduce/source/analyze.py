"""Fixed training-only response, shape and baseline-association diagnostics."""

import csv
import gzip
import json
import math
import resource
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from layers import FAMILY, dump

LAYERS = ("corrected", "restored", "unthresholded")


def read(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def summary(rows, prediction):
    y = np.array([r["target"] for r in rows]); p = np.array([r[prediction] for r in rows]); w = np.array([r["weight"] for r in rows])
    w = w/w.sum()
    return dict(n=len(rows), macro_rmse=float(np.sqrt(w @ ((p-y)**2))), pooled_rmse=float(np.sqrt(np.mean((p-y)**2))),
                pearson=None if np.ptp(p)==0 else float(pearsonr(y,p).statistic),
                spearman=None if np.ptp(p)==0 else float(spearmanr(y,p).statistic), prediction_sd=float(p.std()), target_sd=float(y.std()))


def shape_summary(rows, value, key):
    pairs=defaultdict(list)
    for r in rows:pairs[r["condition_id"], r["peptide_IL_sequence"], str(r[key])].append(r[value])
    return {str(t): dict(n=len(rows), pooled_fraction=sum(r[value]>t for r in rows)/len(rows),
                        equal_parent_fraction=sum(r["weight"]*(r[value]>t) for r in rows)/sum(r["weight"] for r in rows),
                        paired_events=len(pairs), both_labels_violate=sum(len(v)>=2 and all(x>t for x in v) for v in pairs.values()))
            for t in (0.0,0.3,0.5)}


def main():
    start=time.perf_counter();cfg=json.loads((FAMILY/'reproduce/launch.json').read_text())
    previous=cfg['previous_family'];saved={}
    with gzip.open(previous+'/results/predictions.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line)
            if r['split']=='train' and r['variant']=='sequence_context':saved[r['curve_id'],r['time']]=r
    curves=read(FAMILY/'results/curve-layers.jsonl.gz')
    ncurves=Counter(r['condition_id'] for r in curves)
    conditions={r['condition_id']:r['construct_id'] for r in curves};nconditions=Counter(conditions.values());nparents=len(nconditions)
    pairs=defaultdict(list)
    for curve in curves:pairs[curve['condition_id'],curve['peptide_IL_sequence']].append(curve)
    assert all(len(v)==2 for v in pairs.values())
    ratios=[];triplets=[]
    for curve in curves:
        points={p['time']:p for p in curve['points']};times=curve['common_positive_times'];ref=times[0];c=curve['condition_id'];parent=curve['construct_id']
        mate=next(r for r in pairs[c,curve['peptide_IL_sequence']] if r['curve_id']!=curve['curve_id']);other={p['time']:p for p in mate['points']}
        baseweight=1/(nparents*nconditions[parent]*ncurves[c]);boundaries=int(curve['start_0based']>0)+int(curve['end_0based']<curve['parent_length'])
        for layer in LAYERS:
            assert all(points[t][layer]>0 and other[t][layer]>0 for t in times)
            for t in times[1:]:
                target=math.log10(points[t][layer]/points[ref][layer]);repeat=math.log10(other[t][layer]/other[ref][layer]);pred=saved[curve['curve_id'],t]
                if layer=='corrected':assert math.isclose(target,pred['target'],rel_tol=1e-10,abs_tol=1e-12)
                weight=baseweight/(len(times)-1);assert math.isclose(weight,pred['weight'],rel_tol=1e-12)
                ratios.append(dict(layer=layer,curve_id=curve['curve_id'],condition_id=c,construct_id=parent,peptide_IL_sequence=curve['peptide_IL_sequence'],
                    biological_replicate_raw=curve['biological_replicate_raw'],time=t,reference_time=ref,target=target,other_label_prediction=repeat,
                    no_change=0.0,weight=weight,upper_bound=boundaries*math.log10(t/ref),excess=target-boundaries*math.log10(t/ref),
                    reference_subtracted_fraction=points[ref]['background']/points[ref]['restored'],
                    source_label_signature_changed=points[t]['retained_source_label_signature']!=points[ref]['retained_source_label_signature'],
                    saved_corrected_model_prediction=pred['prediction'],saved_corrected_model_squared_error=(pred['prediction']-pred['target'])**2))
            for a,b,d in zip(times,times[1:],times[2:]):
                logs={t:math.log10(points[t][layer]) for t in (a,b,d)}
                excess=logs[a]+(b-a)/(d-a)*(logs[d]-logs[a])-logs[b]
                triplets.append(dict(layer=layer,curve_id=curve['curve_id'],condition_id=c,construct_id=parent,peptide_IL_sequence=curve['peptide_IL_sequence'],
                    times=[a,b,d],chord_minus_middle=excess,weight=baseweight/(len(times)-2)))
    for name,rows in [('ratios',ratios),('triplets',triplets)]:
        p=FAMILY/'results'/f'{name}.jsonl.gz';assert not p.exists()
        with gzip.open(p,'wt') as out:
            for r in rows:out.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    metrics=[];shapes={}
    for layer in LAYERS:
        rows=[r for r in ratios if r['layer']==layer];tris=[r for r in triplets if r['layer']==layer]
        assert math.isclose(sum(r['weight'] for r in rows),1,rel_tol=1e-10)
        assert math.isclose(sum(r['weight'] for r in tris),1,rel_tol=1e-10)
        for parent in ['all',*sorted(nconditions)]:
            subset=rows if parent=='all' else [r for r in rows if r['construct_id']==parent]
            for predictor in ('no_change','other_label_prediction'):metrics.append(dict(layer=layer,parent=parent,predictor=predictor,**summary(subset,predictor)))
        shapes[layer]=dict(envelope=shape_summary(rows,'excess','time'),concavity=shape_summary(tris,'chord_minus_middle','times'))
    # Pooled percentages and counts must reproduce the earlier diagnostic exactly.
    old=json.loads(open(previous+'/results/training-shape-constraints.json').read())
    for kind in ('envelope','concavity'):
        for t,r in shapes['corrected'][kind].items():
            prior=old[kind]['by_threshold'][t]
            assert r['pooled_fraction']==prior['fraction'] and r['both_labels_violate']==prior['both_labels_violate']
    transitions={}
    for kind,allrows,value,key in [('envelope',ratios,'excess','time'),('concavity',triplets,'chord_minus_middle','times')]:
        index={(r['curve_id'],str(r[key]),r['layer']):r for r in allrows};baseline=[r for r in allrows if r['layer']=='corrected']
        transitions[kind]={}
        for layer in LAYERS[1:]:
            counter=Counter();weight=Counter()
            for r in baseline:
                a=r[value]>.3;b=index[r['curve_id'],str(r[key]),layer][value]>.3;label=f'{int(a)}_to_{int(b)}';counter[label]+=1;weight[label]+=r['weight']
            transitions[kind][layer]=dict(counts=dict(counter),weights=dict(weight))
    corrected=[r for r in ratios if r['layer']=='corrected'];strata=[];bins=cfg['reference_subtracted_fraction_bins']
    for lo,hi in zip(bins,bins[1:]):
        subset=[r for r in corrected if lo<=r['reference_subtracted_fraction']<hi or hi==1 and r['reference_subtracted_fraction']==1]
        if not subset:continue
        w=sum(r['weight'] for r in subset)
        strata.append(dict(lower=lo,upper=hi,n=len(subset),weight_mass=w,
            saved_model_rmse=math.sqrt(sum(r['weight']*r['saved_corrected_model_squared_error'] for r in subset)/w),
            repeatability_rmse=summary(subset,'other_label_prediction')['macro_rmse'],
            envelope_fraction=sum(r['weight']*(r['excess']>.3) for r in subset)/w))
    changed={}
    for flag in (False,True):
        rows=[r for r in corrected if r['source_label_signature_changed']==flag]
        changed[str(flag)]=dict(n=len(rows),weight_mass=sum(r['weight'] for r in rows),metrics=summary(rows,'other_label_prediction') if rows else None)
    dump(FAMILY/'results/metrics.json',metrics);dump(FAMILY/'results/shapes.json',shapes);dump(FAMILY/'results/shape-transitions.json',transitions)
    dump(FAMILY/'results/baseline-strata.json',strata);dump(FAMILY/'results/source-label-change.json',changed)
    with (FAMILY/'results/comparison.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    dump(FAMILY/'results/analysis-runtime.json',dict(training_only=True,no_new_fit=True,ratio_rows=len(ratios),triplet_rows=len(triplets),
        wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(overall=[r for r in metrics if r['parent']=='all'],shapes=shapes,strata=strata,source_changes=changed)),flush=True)


if __name__=='__main__':main()
