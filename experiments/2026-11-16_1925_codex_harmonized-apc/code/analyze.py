"""Paired protein-cluster intervals and class/source metrics from saved predictions."""
from pathlib import Path
from collections import defaultdict
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

def main():
 a=argparse.ArgumentParser();a.add_argument('--runs',default='results/runs');a.add_argument('--data',default='results/inputs/data/validation.jsonl');a.add_argument('--out',default='results/analysis-v1');args=a.parse_args()
 runs=Path(args.runs);out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
 raw=[json.loads(x) for x in Path(args.data).open()];meta={r['observation_id']:r for r in raw}
 loaded={}
 for folder in sorted(runs.glob('*')):
  if not folder.is_dir():continue
  for prefix,key in [('validation',folder.name),('reference','reference')]:
   p=folder/f'{prefix}-predictions.json'
   if p.exists():loaded[key]={r['observation_id']:r['prediction'] for r in json.load(p.open())}
 if not loaded:raise ValueError('No prediction dumps')
 ids=sorted(meta);table=[]
 for id in ids:
  r=meta[id];p=r['provenance'];row={'id':id,'pair':p['pair_id'],'protein':r['group_id'],'label':r['value'],'class':r['mhc'][0]['mhc_class'],'pmid':p['pmid'],'condition':p['condition_id']}
  for name,scores in loaded.items():row[name]=scores[id]
  table.append(row)
 d=pd.DataFrame(table);d.to_parquet(out/'aligned-predictions.parquet',index=False)
 groups={'pooled':d,**{f'class_{c}':g for c,g in d.groupby('class')},**{f'PMID{p}_class_{c}':g for (p,c),g in d.groupby(['pmid','class'])}}
 metrics=[];intervals=[];rng=np.random.default_rng(20261116)
 def score(g,key):
  pairs=g.pivot(index='pair',columns='label',values=key);diff=pairs[1.]-pairs[0.]
  logits=g[key].to_numpy();y=g.label.to_numpy()
  return {'auroc':roc_auc_score(y,logits),'auprc':average_precision_score(y,logits),'loss':float(np.mean(np.logaddexp(0,logits)-y*logits)), 'pair_win':float(((diff>0)+.5*(diff==0)).mean())}
 for group,g in groups.items():
  for key in loaded:metrics.append({'group':group,'model':key,'rows':len(g),'pairs':g.pair.nunique(),'proteins':g.protein.nunique(),**score(g,key)})
  if group not in ('pooled','class_I','class_II'):continue
  cluster=[v.index.to_numpy() for _,v in g.groupby('protein')]
  real=next((k for k in loaded if k.startswith('harmonized')),None)
  if real is None:continue
  comparisons=[k for k in loaded if k!=real]
  samples={c:defaultdict(list) for c in comparisons}
  # Repeated protein draws preserve all dependent observations. Unique instance
  # pair IDs prevent duplicated bootstrap pairs from colliding in the pivot.
  for _ in range(1000):
   indices=np.concatenate([cluster[i] for i in rng.integers(len(cluster),size=len(cluster))]);boot=d.loc[indices].copy()
   # Compute win rate directly from per-pair wins and cluster multiplicity.
   y=boot.label.to_numpy();rs=boot[real].to_numpy();real_auc=roc_auc_score(y,rs)
   counts=boot[boot.label==1].groupby('pair').size()
   original=g.pivot(index='pair',columns='label',values=real);rv=original[1]-original[0];rwin=((rv>0)+.5*(rv==0)).loc[counts.index].to_numpy()
   for c in comparisons:
    cv=g.pivot(index='pair',columns='label',values=c);delta=cv[1]-cv[0];cwin=((delta>0)+.5*(delta==0)).loc[counts.index].to_numpy()
    samples[c]['auroc'].append(real_auc-roc_auc_score(y,boot[c]))
    samples[c]['pair_win'].append(float(np.average(rwin-cwin,weights=counts)))
  for c,vals in samples.items():
   rscore,cscore=score(g,real),score(g,c)
   for metric,v in vals.items():intervals.append({'group':group,'model':real,'comparator':c,'metric':metric,'difference':rscore[metric]-cscore[metric],'ci95':np.quantile(v,[.025,.975]).tolist(),'bootstrap':'1000 paired source-protein cluster resamples; development cohort only'})
 pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
 (out/'paired-bootstrap.json').write_text(json.dumps(intervals,indent=2))
 print(pd.DataFrame(metrics).query("group in ['pooled','class_I','class_II']").to_string(index=False))
 print(json.dumps(intervals,indent=2))
if __name__=='__main__':main()
