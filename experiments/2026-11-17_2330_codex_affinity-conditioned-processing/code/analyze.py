"""Independent decoded metrics and paired protein-family/condition bootstraps."""
from pathlib import Path
import argparse,gzip,hashlib,json,math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score
from analysis_shared import complete_pairs,paired_draws
E=Path(__file__).resolve().parents[1]
B=E.parent/'2026-11-17_1155_codex_shared-presentation-expansion'


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,np.generic):return clean(value.item())
    return value


def write(path,value):path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def flatten(r,model,population):
    o=r['observation'];p=o['provenance'];assert r['split']=='validation'
    assert len(set(r['mhc_classes']))==1
    return {'model':model,'population':population,'id':r['observation_id'],
        'class':r['mhc_classes'][0],'endpoint':r['endpoint'],'target':r['target'],'prediction':r['prediction'],
        'loss':r['loss'],'qualifier':r['qualifier'],'peptide':r['peptide'],'pair':p.get('pair_id'),
        'group':r['group_id'],'protein':p.get('source_protein'),'study':str(p.get('study_id') or p.get('pmid')),
        'condition':p.get('condition_id') or p.get('sample_id') or '',
        'candidates':len({(m['groove1'],m['groove2']) for m in o['mhc']}),
        'mhc_signature':hashlib.sha256(json.dumps(sorted({(m['groove1'],m['groove2']) for m in o['mhc']})).encode()).hexdigest(),
        'reported_mhc':str(p.get('raw_record',{}).get('mhc_allele_set','')),
        'flank_n_known':o['context'].get('source_n') is not None,
        'flank_c_known':o['context'].get('source_c') is not None,
        **{k:r[k] for k in ('processing_supply','junction_processing','without_processing_logit','without_junction_logit') if k in r}}


def load():
    data=[]
    for receipt in sorted((E/'results/evaluation').glob('*/receipt.json')):
        model=receipt.parent.name
        declared=json.loads(receipt.read_text())
        for population in sorted(declared['metrics']):
            path=receipt.parent/(population+'-predictions.jsonl.gz')
            if not path.exists():raise ValueError(f'incomplete local collection: {path}')
            count=0
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    data.append(flatten(json.loads(line),model,population))
                    count+=1
            assert count==sum(v['n'] for v in declared['metrics'][population].values()), (model,population,count)
    frame=pd.DataFrame(data)
    assert not frame.duplicated(['model','population','id']).any()
    # Same fixed ancestry sensitivity as the previous family, no new target selection.
    exposure=json.loads((B/'results/fixed-validation-ancestor-exposure.json').read_text())['affinity-validation.jsonl']['exposed']
    clean_affinity=frame[(frame.population=='affinity') & ~frame.id.isin({x['id'] for x in exposure})].copy()
    clean_affinity['population']='affinity_ancestry_screened'
    return pd.concat([frame,clean_affinity],ignore_index=True)


def metrics(frame):
    report=[]
    for (model,pop,ep,cls),g in frame.groupby(['model','population','endpoint','class']):
        if ep=='elution_observation':
            for score in ('prediction','processing_supply','without_processing_logit','without_junction_logit'):
                if g[score].isna().any():continue
                s=g[score].to_numpy();y=g.target.to_numpy()
                record={'model':model,'population':pop,'endpoint':ep,'class':cls,'readout':score,'n':len(g),
                    'auroc':roc_auc_score(y,s),'auprc':average_precision_score(y,s),
                    'loss':float(g.loss.mean()) if score=='prediction' else None}
                if complete_pairs(g):
                    pairs=g.pivot(index='pair',columns='target',values=score);delta=pairs[1]-pairs[0]
                    record.update(pairs=len(pairs),pair_win=float(((delta>0)+.5*(delta==0)).mean()))
                report.append(record)
        else:
            exact=g[g.qualifier=='='];y=exact.target.to_numpy();s=exact.prediction.to_numpy()
            record={'model':model,'population':pop,'endpoint':ep,'class':cls,'n':len(g),'loss':float(g.loss.mean()),'exact_n':len(exact)}
            if len(exact)>1:
                truth=y<=np.log10(500);pred=s<=np.log10(500)
                record.update(spearman=float(spearmanr(y,s).statistic),pearson=float(pearsonr(y,s).statistic),
                    rmse_log10_nm=float(np.sqrt(np.mean((s-y)**2))),accuracy=accuracy_score(truth,pred),
                    balanced_accuracy=balanced_accuracy_score(truth,pred) if len(set(truth))>1 else None,
                    precision=precision_score(truth,pred,zero_division=0),recall=recall_score(truth,pred,zero_division=0),
                    f1=f1_score(truth,pred,zero_division=0),binding_auroc=roc_auc_score(truth,-s) if len(set(truth))>1 else None,
                    binding_auprc=average_precision_score(truth,-s) if truth.any() else None)
            report.append(record)
    return report


def sensitivity(frame):
    """Descriptive strata; do not select the matcher or checkpoint from these."""
    table=pd.DataFrame(json.loads((E/'results/rank-matched-validation-pairs.json').read_text()))
    ranks=dict(zip(table.pair_id,table.positive_ba_rank))
    definitions={r['pair_id']:r for r in table.to_dict('records')}
    report=[]
    for (model,pop,cls),g in frame[frame.population.isin(['matched','rank-matched'])].groupby(['model','population','class']):
        assert complete_pairs(g)
        slices=[('candidates','one',g[g.candidates==1]),('candidates','multiple',g[g.candidates>1])]
        for key,cell in g[g.candidates==1].groupby('mhc_signature'):
            slices.append(('single-biological-MHC',key,cell))
        if pop=='rank-matched':
            g=g.copy();g['positive_rank']=g.pair.map(ranks);assert g.positive_rank.notna().all()
            for lower,upper in ((0,.5),(.5,2),(2,10),(10,100.00001)):
                slices.append(('positive-BA-rank',f'[{lower},{upper})',g[(g.positive_rank>=lower)&(g.positive_rank<upper)]))
            predicates={
                'max .1pp AND .1log-rank':lambda p:p['assessment']['max_rank_gap']<=.1 and p['assessment']['max_log_rank_gap']<=.1,
                'hybrid plus max .1log-nM':lambda p:p['assessment']['max_log_nm_gap']<=.1,
                'tight rank plus max .1log-nM':lambda p:p['assessment']['max_rank_gap']<=.1 and p['assessment']['max_log_rank_gap']<=.1 and p['assessment']['max_log_nm_gap']<=.1,
                'all printed BA ranks tied':lambda p:p['assessment']['max_rank_gap']==0,
                'positive has stronger BA nM':lambda p:p['positive_ba_nm']<p['negative_ba_nm'],
                'background has stronger BA nM':lambda p:p['positive_ba_nm']>p['negative_ba_nm'],
                'BA nM tied':lambda p:p['positive_ba_nm']==p['negative_ba_nm'],
            }
            for label,predicate in predicates.items():
                slices.append(('residual-affinity-sensitivity',label,g[g.pair.map(lambda key:predicate(definitions[key]))]))
        for category,value,cell in slices:
            if cell.empty:continue
            assert complete_pairs(cell)
            for score in ('prediction','processing_supply'):
                pairs=cell.pivot(index='pair',columns='target',values=score);delta=pairs[1]-pairs[0]
                report.append({'model':model,'population':pop,'class':cls,'stratum':category,'value':value,
                    'reported_mhc':sorted(set(cell.reported_mhc)-{''}),'pairs':len(pairs),'readout':score,
                    'auroc':roc_auc_score(cell.target,cell[score]),'pair_win':float(((delta>0)+.5*(delta==0)).mean()),
                    'interpretation':'descriptive; small allele cells are not independent confirmation'})
    return report


def bootstrap(frame,real,other,draws=2000):
    reports=[];skipped=[]
    for pop in ('expanded','apc','broad','matched','rank-matched'):
        for cls in ('I','II'):
            first=frame[(frame.population==pop)&(frame.model==real)&(frame['class']==cls)]
            second=frame[(frame.population==pop)&(frame.model==other)&(frame['class']==cls)]
            if first.empty or second.empty:continue
            assert set(first.id)==set(second.id)
            joined=first.merge(second[['id','target','peptide','prediction','processing_supply']],on='id',suffixes=('_real','_other'),validate='one_to_one')
            assert (joined.target_real==joined.target_other).all() and (joined.peptide_real==joined.peptide_other).all()
            joined['target']=joined.target_real
            for cluster in ('protein-family','condition'):
                a=joined.copy()
                if cluster=='protein-family':
                    if not a.group.astype(str).str.startswith('ENSP').all():
                        skipped.append({'population':pop,'class':cls,'cluster':cluster,'reason':'source has study-component groups, not validated protein-family groups'});continue
                    a['bootstrap_group']=a.group
                else:a['bootstrap_group']=a.study+'|'+a.condition.replace('','<unknown-condition>')
                names=sorted(a.bootstrap_group.unique())
                if len(names)<2:
                    skipped.append({'population':pop,'class':cls,'cluster':cluster,'reason':'fewer than two independent clusters'});continue
                seed=int.from_bytes(hashlib.sha256(json.dumps([20261117,pop,cls,cluster,names]).encode()).digest()[:8],'big')
                indices=np.random.default_rng(seed).integers(len(names),size=(draws,len(names)))
                for score in ('prediction','processing_supply'):
                    scored=a.copy()
                    scored['prediction_real']=a[score+'_real'];scored['prediction_other']=a[score+'_other']
                    samples=np.concatenate([paired_draws(scored,indices[i:i+32],names) for i in range(0,draws,32)])
                    paired=complete_pairs(scored)
                    for column,metric in enumerate(('auroc','pair_win')):
                        if metric=='pair_win' and not paired:continue
                        finite=np.isfinite(samples[:,column]);assert finite.sum()>=.95*draws
                        if metric=='auroc':point=roc_auc_score(a.target,scored.prediction_real)-roc_auc_score(a.target,scored.prediction_other)
                        else:
                            pairs=scored.pivot(index='pair',columns='target',values=['prediction_real','prediction_other'])
                            delta={key:pairs[key][1]-pairs[key][0] for key in ('prediction_real','prediction_other')}
                            point=float((((delta['prediction_real']>0)+.5*(delta['prediction_real']==0))-((delta['prediction_other']>0)+.5*(delta['prediction_other']==0))).mean())
                        reports.append({'model':real,'comparator':other,'population':pop,'class':cls,'cluster':cluster,'clusters':len(names),
                            'readout':score,'metric':metric,'difference':point,'ci95':np.quantile(samples[finite,column],[.025,.975]).tolist(),
                            'finite_draws':int(finite.sum()),'complete_pairs':paired,'unknown_condition_rows':int((a.condition=='').sum()),
                            'scope':'pointwise paired cluster bootstrap conditional on fitted models and development cohort; no multiplicity adjustment; excludes checkpoint/seed selection uncertainty'})
    return {'comparisons':reports,'skipped':skipped}


def main():
    p=argparse.ArgumentParser();p.add_argument('--compare');p.add_argument('--against',default='parent');p.add_argument('--bootstrap-only',action='store_true');args=p.parse_args()
    out=E/'results/analysis';out.mkdir(exist_ok=True)
    frame=pd.read_parquet(out/'predictions.parquet') if args.bootstrap_only else load()
    if not args.bootstrap_only:
        frame.to_parquet(out/'predictions.parquet',index=False);report=metrics(frame)
        write(out/'metrics.json',report);pd.DataFrame(report).to_csv(out/'metrics.csv',index=False)
        strata=sensitivity(frame);write(out/'matching-sensitivity.json',strata)
        pd.DataFrame(strata).to_csv(out/'matching-sensitivity.csv',index=False)
        print('independent rows',len(frame),'models',sorted(frame.model.unique()))
    if args.compare:
        result=bootstrap(frame,args.compare,args.against)
        write(out/f'bootstrap-{args.compare}-minus-{args.against}.json',result)
        print('bootstrap comparisons',len(result['comparisons']))

if __name__=='__main__':main()
