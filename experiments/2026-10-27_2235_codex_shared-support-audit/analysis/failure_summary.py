"""Close saved-output diagnostics without running or selecting any model."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
F=Path(__file__).resolve().parents[1];ROOT=F.parents[1];PRIOR=ROOT/'experiments/2026-10-27_2100_codex_transfer-improvements'
def read(p):return json.loads(p.read_text())
def load(p):
 with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 inputs=ROOT/'artifacts'/PRIOR.name/'upload/inputs';training=[json.loads(x) for x in (inputs/'joint-train.jsonl').open()]
 sequence_sets={c:set() for c in ['I','II']};grooves={c:set() for c in ['I','II']};studies={c:set() for c in ['I','II']}
 for r in training:
  c=r['mhc'][0]['mhc_class'];sequence_sets[c].update(r['peptide'][i:i+9] for i in range(len(r['peptide'])-8));grooves[c].add((r['mhc'][0]['groove1'],r['mhc'][0]['groove2']));raw=r['provenance']['raw_record'];studies[c].add(str(raw.get('pmid') or raw.get('reference_iri') or 'unknown'))
 fig,axes=plt.subplots(2,2,figsize=(11,8));summary=[];strata=[]
 for j,c in enumerate(['I','II']):
  for seed in [42,43]:
   name=f'interaction-natural-joint-s{seed}';rows=[r for r in load(PRIOR/'results'/name/'test-predictions.jsonl.gz') if r['mhc_classes']==[c]]
   ex=[r for r in rows if r['qualifier']=='='];y=np.array([r['target'] for r in ex]);p=np.array([r['prediction'] for r in ex]);bins=np.floor(y).astype(int);xs=[];ys=[]
   for b in sorted(set(bins)):
    mask=bins==b
    if mask.sum()<10:continue
    xs.append(float(y[mask].mean()));ys.append(float(p[mask].mean()))
   axes[0,j].plot(xs,ys,'o-',label=f'seed {seed}')
   g=defaultdict(list)
   for r in rows:
    o=r['observation'];m=o['mhc'][0];peptide=r['peptide'];raw=o['provenance']['raw_record'];study=str(raw.get('pmid') or raw.get('reference_iri') or 'unknown')
    keys={'nine_mer_overlap':str(any(peptide[i:i+9] in sequence_sets[c] for i in range(len(peptide)-8))), 'seen_mhc_pair':str((m['groove1'],m['groove2']) in grooves[c]),'seen_study':str(study in studies[c]),'candidate_count':str(1 if c=='I' else len(peptide)-8)}
    for k,v in keys.items():g[k,v].append(r)
   for (category,value),rr in sorted(g.items()):
    exact=[r for r in rr if r['qualifier']=='='];yy=np.array([r['target'] for r in exact]);pp=np.array([r['prediction'] for r in exact]);rho=float(spearmanr(yy,pp).statistic) if len(yy)>2 and np.ptp(yy)>0 and np.ptp(pp)>0 else None
    strata.append(dict(condition=name,mhc_class=c,split='test',group=category,value=value,n=len(rr),n_exact=len(exact),spearman=rho,rmse=float(np.sqrt(((yy-pp)**2).mean())) if len(yy) else None))
   bounds=[r for r in rows if r['qualifier']!='='];satisfied=sum(r['loss']==0 for r in bounds)
   independent=read(PRIOR/'results'/f'{name}-independent-verification.json')['metrics'][f'in_class:test:{c}']
   allele=independent['subgroups']['by_mhc_pair'];valid=[a for a in allele.values() if a.get('n',0)>=20 and a.get('spearman') is not None];weighted=sum(a['n']*a['spearman'] for a in valid)/sum(a['n'] for a in valid)
   summary.append(dict(condition=name,mhc_class=c,test=independent['metrics'],bounds=len(bounds),satisfied_bounds=satisfied,alleles_at_least20=len(valid),negative_alleles=sum(a['spearman']<0 for a in valid),macro_allele_spearman=float(np.mean([a['spearman'] for a in valid])),weighted_allele_spearman=weighted,prediction_target_sd_ratio=float(p.std()/y.std()),source_study_overlap='see strata; study is not an independent holdout'))
  axes[0,j].plot([-1,6],[-1,6],'k--',alpha=.5,label='calibrated identity');axes[0,j].set(title=f'MHC-{c}: test calibration by measured bin',xlabel='Mean measured log10(nM)',ylabel='Mean predicted log10(nM)');axes[0,j].legend()
  values=[r for r in summary if r['mhc_class']==c];axes[1,j].bar(['seed 42','seed 43'],[100*r['satisfied_bounds']/r['bounds'] for r in values],color=['#3581b8','#63ac92']);axes[1,j].set(ylim=(0,100),title=f'MHC-{c}: original censor bounds satisfied',ylabel='Percent of bounds')
 fig.suptitle('Shared-v5 failure audit · existing IC50 development test outputs');fig.tight_layout();fig.savefig(F/'results/failure-audit.png',dpi=180);fig.savefig(F/'results/failure-audit.pdf');plt.close(fig)
 write(F/'results/failure-summary.json',dict(source_family=PRIOR.name,no_new_fit=True,no_new_test_inference=True,summary=summary,strata=strata))
 with (F/'results/novelty-register-strata.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(strata[0]),lineterminator='\n');w.writeheader();w.writerows(strata)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
