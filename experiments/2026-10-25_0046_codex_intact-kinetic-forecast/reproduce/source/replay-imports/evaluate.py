"""Once-opened temporal outcomes, frozen comparisons and complete predictions."""

import datetime
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import pearsonr,spearmanr

from model import ROOT,dump_json

config=json.loads((ROOT/'reproduce/launch.json').read_text())
conditions=config['fit_conditions']
prediction_paths=[ROOT/'results/controls/predictions.parquet',*[ROOT/f'results/fits/{c}/predictions.parquet' for c in conditions]]
for name in ['admit','audit','qualify','controls',*conditions]:
    assert json.loads((ROOT/f'results/{name}-resource.json').read_text())['passed']
receipt={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in prediction_paths}
assert len(receipt)==15
dump_json(ROOT/'results/pre-outcome-prediction-hashes.json',{'saved_before_predictive_outcomes':True,'files':receipt,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
out=ROOT/'results/evaluation';out.mkdir()
dump_json(out/'outcome-access.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'validation100_then_test120_once_for_frozen_conditions'})
predictions=[r for p in prediction_paths for r in pq.read_table(p).to_pylist()]
covariates={};targets={}
for split in ('train','val','test'):
    for row in pq.read_table(ROOT/f'results/data/{split}/covariates.parquet').to_pylist():covariates[row['assignment_id']]=row
    for row in pq.read_table(ROOT/f'results/data/{split}/outcomes.parquet').to_pylist():targets[row['assignment_id']]=row['target']
last={(r['split'],r['assignment_id']):r['prediction'] for r in predictions if r['model']=='last'}
joined=[]
for row in predictions:
    c=covariates[row['assignment_id']];anchor=last[row['split'],row['assignment_id']]
    joined.append({**c,**row,'target':targets[row['assignment_id']],'last_train_mean':anchor,
                   'target_change':targets[row['assignment_id']]-anchor,'prediction_change':row['prediction']-anchor})
pq.write_table(pa.Table.from_pylist(joined),out/'predictions.parquet')


def metrics(y,p,w):
    y=np.array(y);p=np.array(p);w=np.array(w);w=w/w.sum()
    mse=float(w@((y-p)**2));variance=float(w@((y-w@y)**2))
    valid=len(y)>=2 and np.ptp(y)>0 and np.ptp(p)>0
    return {'n_observations':len(y),'mse':mse,'rmse':mse**.5,'mae':float(w@np.abs(y-p)),
            'r2':1-mse/variance if variance>0 else None,'pearson':float(pearsonr(y,p).statistic) if valid else None,
            'spearman':float(spearmanr(y,p).statistic) if valid else None,
            'prediction_sd':float(w@((p-w@p)**2))**.5,'target_sd':variance**.5}


model_names=list(dict.fromkeys(r['model'] for r in predictions));groups={}
overall=[];strata=[]
for model in model_names:
    for split in ('train','val','test'):
        rows=[r for r in joined if r['model']==model and r['split']==split]
        groups[model,split]=rows
        for space in ('ratio','change'):
            yt='target' if space=='ratio' else 'target_change';yp='prediction' if space=='ratio' else 'prediction_change'
            overall.append({'model':model,'split':split,'space':space,**metrics([r[yt] for r in rows],[r[yp] for r in rows],[r['weight'] for r in rows])})
        for axis in ('peptide_sequence','replicate_label_raw','time_minutes'):
            for value in sorted({r[axis] for r in rows}):
                subset=[r for r in rows if r[axis]==value]
                strata.append({'model':model,'split':split,'axis':axis,'value':str(value),
                               **metrics([r['target'] for r in subset],[r['prediction'] for r in subset],[r['weight'] for r in subset])})
lookup={(r['model'],r['split'],r['space']):r for r in overall}
control_names=[m for m in model_names if not m.startswith('fit-')]
exact_names=[m for m in model_names if '-exact-' in m]
comparator=min(control_names,key=lambda m:lookup[m,'val','ratio']['rmse'])
physical=min(exact_names,key=lambda m:lookup[m,'val','ratio']['rmse'])
gains={split:100*(1-lookup[physical,split,'ratio']['rmse']/lookup[comparator,split,'ratio']['rmse']) for split in ('val','test')}
rng=np.random.default_rng(31415);bootstrap={}
for split in ('val','test'):
    physical_rows={r['assignment_id']:r for r in groups[physical,split]};rows=groups[comparator,split]
    sequences=sorted({r['peptide_sequence'] for r in rows});index=np.array([sequences.index(r['peptide_sequence']) for r in rows])
    base=np.array([(r['prediction']-r['target'])**2 for r in rows]);experiment=np.array([(physical_rows[r['assignment_id']]['prediction']-r['target'])**2 for r in rows])
    weights=np.array([r['weight'] for r in rows]);values=[]
    for _ in range(1000):
        counts=np.bincount(rng.integers(len(sequences),size=len(sequences)),minlength=len(sequences))
        w=weights*counts[index];w/=w.sum();values.append(100*(1-np.sqrt(w@experiment)/np.sqrt(w@base)))
    bootstrap[split]={'unit':'peptide_not_specimen','n_resamples':1000,'gain_percent_95_interval':np.quantile(values,[.025,.975]).tolist()}
gates={'validation_rmse_gain_at_least_5pct':gains['val']>=5,'test_rmse_gain_at_least_5pct':gains['test']>=5}
for split in ('val','test'):
    m=lookup[physical,split,'ratio'];gates[split+'_positive_correlations']=m['pearson'] is not None and m['pearson']>0 and m['spearman']>0
    gates[split+'_prediction_variation']=m['prediction_sd']>=.1*m['target_sd']
fits=[json.loads((ROOT/f'results/fits/{c}/summary.json').read_text()) for c in conditions]
sampling=[]
for kind in ('uniform','residue'):
    reference=lookup[f'fit-{kind}-exact-0-42','test','ratio']['rmse']
    for draws in (16,64):
        names=[f'fit-{kind}-score-{draws}-{seed}' for seed in (42,43,44)]
        values=[lookup[name,'test','ratio']['rmse'] for name in names]
        times=[r['fit_seconds'] for r in fits if r['model'] in names]
        exact_time=next(r['fit_seconds'] for r in fits if r['model']==f'fit-{kind}-exact-0-42')
        sampling.append({'kind':kind,'draws':draws,'test_rmse_mean':float(np.mean(values)),'test_rmse_seed_sd':float(np.std(values,ddof=1)),
                         'exact_test_rmse':reference,'fit_seconds_mean':float(np.mean(times)),'runtime_ratio_to_exact':float(np.mean(times)/exact_time)})
summary={'comparator_selected_on_validation':comparator,'physical_selected_on_validation':physical,
         'gain_percent':gains,'gates':gates,'primary_gate_passed':all(gates.values()),'bootstrap':bootstrap,
         'sampling_comparisons':sampling,'all_models':overall,'training_scope':'same_source_future_time_ratios_for_training_anchored_peptides',
         'APC_concentration_validated':False,'pure_simulator_RL':False}
dump_json(out/'summary.json',summary)
pq.write_table(pa.Table.from_pylist(overall),out/'metrics.parquet');pq.write_table(pa.Table.from_pylist(strata),out/'strata.parquet')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
names=[*control_names,*exact_names];labels=[n.removeprefix('fit-').replace('-exact-0-42','') for n in names]
axes[0].barh(labels,[lookup[n,'test','ratio']['rmse'] for n in names]);axes[0].set_xlabel('Test RMSE (log10 ratio)');axes[0].set_title('Future-time prediction')
rows=groups[physical,'test'];axes[1].scatter([r['target_change'] for r in rows],[r['prediction_change'] for r in rows],s=12,alpha=.6)
axes[1].axline((0,0),slope=1,color='gray',linestyle='--');axes[1].set_xlabel('Measured change from last fitting time');axes[1].set_ylabel('Predicted change');axes[1].set_title('Temporal signal beyond peptide anchors')
for name in exact_names:
    trace=np.load(ROOT/f'results/fits/{name}/checkpoints.npz')['trace'];axes[2].plot(trace[:,0],trace[:,2],label=name.split('-')[1])
axes[2].set_xlabel('Update');axes[2].set_ylabel('Training MSE');axes[2].set_title('Exact physical models');axes[2].legend()
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='all_models'}),flush=True)
