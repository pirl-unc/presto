"""Pure matching/assembly over immutable protected proposals and BA caches."""
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import math
import shutil
import time
from common import NAME,PRIOR,BROAD,sha,rows,dump,job_key,allele_name
from presto.data.affinity_matching import AffinityScore,MatchingPolicy,assess_match,candidate_key
from presto.data.modular import Observation, AA
from presto.training.processing_pairs import validate_pair

POLICY=MatchingPolicy()


def load_scores(root,stage=1):
    scores={};receipts=[]
    for s in range(1,stage+1):
        completion=json.loads((root/'external'/f'completed-stage{s}.json').read_text())
        assert completion['status']=='completed' and not completion['errors']
        for entry in completion['results']:
            path=Path(entry['path']);record=json.loads(path.read_text())
            key=(record['class'],record['allele']);target=scores.setdefault(key,{})
            for p,v in record['scores'].items():
                if p in target:assert target[p]==v
                target[p]=v
            receipts.append({'path':str(path),'sha256':sha(path),'class':key[0],'allele':key[1]})
    return scores,receipts


def profile(r,peptide,scores):
    return {key:AffinityScore(scores[(r['class'],allele)][peptide]['ba_nm'],
                              scores[(r['class'],allele)][peptide]['ba_rank'])
            for key,allele in r['alleles'].items()}


def best_match(r,scores,original,stage):
    positive=profile(r,r['peptide'],scores)
    for options in [r['options'][:8]]+([r['options'][8:]] if stage==2 else []):
        accepted=[]
        for q in options:
            if any(q['peptide'] not in scores[(r['class'],a)] for a in r['alleles'].values()):continue
            if any(original['context'].get(k) is not None and q['context'].get(k) is None for k in ('source_n','source_c','n_terminus','c_terminus')):continue
            negative=profile(r,q['peptide'],scores);assessment=assess_match(positive,negative,POLICY)
            if assessment.accepted:accepted.append((assessment.distance,q['peptide'],q,negative,assessment))
        if accepted:
            _,_,q,negative,assessment=min(accepted,key=lambda x:(x[0],x[1]))
            return q,positive,negative,assessment
    return None


def make_background(original,proposal,choice,lineage):
    q,positive,negative,assessment=choice
    oid='processing-background:'+hashlib.sha256((original['observation_id']+'|'+q['peptide']+'|hybrid-all-v1').encode()).hexdigest()
    n=copy.deepcopy(original)
    n.update(observation_id=oid,peptide=q['peptide'],value=0.)
    for k in ('source_n','source_c','n_terminus','c_terminus'):
        n['context'][k]=q['context'][k] if original['context'].get(k) is not None else None
    cls=proposal['class']
    n['provenance'].update(original_measurement_id=oid,measurement_namespace='synthetic_processing_matched',
        synthetic=True,label_kind='generated_background',parent_positive_id=original['observation_id'],
        processing_pair_positive_id=original['observation_id'],source_protein=proposal['protein'],
        generation_policy='Existing protected same-reference-protein/same-length native TRAIN background; all-candidate BA rank matching; context knownness matched',
        affinity_match={'schema':'presto.affinity_match.v1','policy':asdict(POLICY),
            'predictor':{'name':'NetMHCpan' if cls=='I' else 'NetMHCIIpan','version':'4.2c' if cls=='I' else '4.3e'},
            **lineage,'positive_scores':{k:asdict(v) for k,v in positive.items()},
            'background_scores':{k:asdict(v) for k,v in negative.items()},'assessment':asdict(assessment)})
    n['provenance'].pop('measurement_lineage',None)
    n['provenance'].pop('generated_position',None)
    return n


def local_native(source,expected):
    path=Path('/tmp/presto-processing-native.jsonl')
    if not path.exists():shutil.copyfile(source,path)
    assert sha(path)==expected
    return path


def analyze(root,stage=1):
    start=time.monotonic();receipt=json.loads((root/'pools/receipt.json').read_text())
    assert sha(root/'pools/proposals.jsonl')==receipt['proposal_sha256']
    assert sha(receipt['source'])==receipt['source_sha256']
    scores,score_receipts=load_scores(root,stage)
    dump(root/f'pools/scores-stage{stage}-manifest.json',score_receipts)
    score_sha=sha(root/f'pools/scores-stage{stage}-manifest.json')
    counts=Counter();studies=defaultdict(Counter);extra=defaultdict(set);stats=defaultdict(list)
    with local_native(receipt['source'],receipt['source_sha256']).open('rb') as native:
        for r in rows(root/'pools/proposals.jsonl'):
            native.seek(r['offset']);original=json.loads(native.readline());assert original['observation_id']==r['id']
            choice=best_match(r,scores,original,stage);cls=r['class'];counts[cls+':eligible']+=1
            if choice is None:
                counts[cls+':unmatched']+=1
                if stage==1:
                    for allele in set(r['alleles'].values()):
                        extra[(cls,allele)].update(q['peptide'] for q in r['options'][8:] if q['peptide'] not in scores[(cls,allele)])
                continue
            q,pp,np,assessment=choice;counts[cls+':matched']+=1;studies[cls][r['study']]+=1
            stats[cls].append({'positive':min(v.nm for v in pp.values()),'background':min(v.nm for v in np.values()),
                'positive_rank':min(v.rank for v in pp.values()),'background_rank':min(v.rank for v in np.values()),
                'rank_gap':assessment.max_rank_gap,'log_rank_gap':assessment.max_log_rank_gap,'candidates':len(pp)})
    from audit import auc,quantiles
    balance={cls:{'pairs':len(items),'monoallelic':sum(v['candidates']==1 for v in items),
        'ba_auc':auc([-math.log10(v['positive']) for v in items],[-math.log10(v['background']) for v in items]),
        'rank_auc':auc([-v['positive_rank'] for v in items],[-v['background_rank'] for v in items]),
        'ba_pair_win':sum((v['positive']<v['background'])+.5*(v['positive']==v['background']) for v in items)/len(items),
        'positive_nm':quantiles([v['positive'] for v in items]),'rank_gap':quantiles([v['rank_gap'] for v in items])} for cls,items in stats.items()}
    jobs=[]
    for (cls,allele),peptides in sorted(extra.items()):
        if not peptides:continue
        path=root/'pools/queries-stage2'/(job_key(cls,allele)+'.json');dump(path,sorted(peptides))
        jobs.append({'class':cls,'allele':allele,'path':str(path),'sha256':sha(path),'queries':len(peptides)})
    report={'status':'audited','stage':stage,'counts':dict(counts),'studies':dict(studies),'balance':balance,
        'jobs':jobs,'total_queries':sum(j['queries'] for j in jobs),'score_sha256':score_sha,
        'policy':asdict(POLICY),'seconds':time.monotonic()-start,'test_accessed':False}
    dump(root/f'pools/analysis-stage{stage}.json',report)
    return report


def assemble(root,stage):
    """Verify original identity/context, write only auxiliary backgrounds once."""
    from presto.data.presentation import read_gencode_proteins,junction
    start=time.monotonic();receipt=json.loads((root/'pools/receipt.json').read_text())
    analysis=json.loads((root/f'pools/analysis-stage{stage}.json').read_text())
    assert analysis['status']=='audited'
    source=Path(receipt['source']);assert sha(source)==receipt['source_sha256']
    source=local_native(source,receipt['source_sha256'])
    scores,_=load_scores(root,stage)
    proteins=read_gencode_proteins('/inputs/proteins.fa.gz')
    protected_backgrounds={r['peptide'] for r in rows(source) if r['value']==0}
    observed={r['peptide'] for r in rows(source) if r['value']==1}
    assert not protected_backgrounds & observed
    lineage={'source_sha256':receipt['source_sha256'],'score_sha256':analysis['score_sha256']}
    out=root/'inputs/processing-backgrounds-train.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
    table=root/'inputs/processing-pairs-train.jsonl';counts=Counter();studies=defaultdict(Counter)
    seen=set();positives=set()
    with source.open('rb') as native,out.open('x') as stream,table.open('x') as pairs:
        for r in rows(root/'pools/proposals.jsonl'):
            native.seek(r['offset']);p=json.loads(native.readline());assert p['observation_id']==r['id']
            choice=best_match(r,scores,p,stage)
            if choice is None:continue
            q,pp,np,assessment=choice;cls=r['class']
            assert q['peptide'] in protected_backgrounds and q['peptide'] not in observed
            sequence=proteins[r['protein']].sequence
            position=sequence.find(q['peptide'])
            while position>=0 and junction(sequence,position,len(q['peptide']))!=q['context']:
                position=sequence.find(q['peptide'],position+1)
            assert position>=0,'background reference context cannot be reproduced'
            n=make_background(p,r,choice,lineage);n['provenance']['generated_position']=position
            validate_pair(Observation.from_dict(p),Observation.from_dict(n))
            assert n['observation_id'] not in seen and p['observation_id'] not in positives
            seen.add(n['observation_id']);positives.add(p['observation_id'])
            stream.write(json.dumps(n,separators=(',',':'))+'\n')
            record={'positive_id':p['observation_id'],'negative_id':n['observation_id'],'class':cls,
                'protein':r['protein'],'study':r['study'],'condition':p['provenance'].get('condition_id'),
                'candidates':len(pp),'positive_ba_nm':min(x.nm for x in pp.values()),
                'negative_ba_nm':min(x.nm for x in np.values()),'positive_ba_rank':min(x.rank for x in pp.values()),
                'negative_ba_rank':min(x.rank for x in np.values()),'assessment':asdict(assessment)}
            pairs.write(json.dumps(record,separators=(',',':'))+'\n')
            counts[cls]+=1;studies[cls][r['study']]+=1
    assert all(counts[c]==analysis['counts'][c+':matched'] for c in ('I','II'))
    result={'status':'prepared','stage':stage,'pairs':dict(counts),'studies':dict(studies),'policy':asdict(POLICY),
        'sources':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [out,table]],
        'lineage':lineage,'ordinary_training_unchanged':True,'auxiliary_rows_excluded_from_ordinary_replay':True,
        'reference_contexts_verified':sum(counts.values()),'all_background_identities_in_protected_native_train':True,
        'seconds':time.monotonic()-start,'test_accessed':False}
    dump(root/'processing-preparation.json',result)
    return result


def validation(root):
    """New cohort ID; frozen old candidates and upgraded original APC context."""
    from audit import auc,quantiles
    prior=Path('/volume')/PRIOR;broad=Path('/volume')/BROAD
    originals={r['observation_id']:r for r in rows(broad/'inputs/apc-validation.jsonl')}
    scores={};hashes=[]
    for path in sorted((prior/'external').glob('*.json')):
        d=json.loads(path.read_text());scores[(d['class'],d['allele'])]=d['scores']
        hashes.append({'path':str(path),'sha256':sha(path)})
    lineage={'source_sha256':sha(broad/'inputs/apc-validation.jsonl'),
             'score_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}
    target=root/'inputs/rank-matched-validation.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
    if (root/'rank-matched-validation.json').exists():
        raise ValueError('completed validation cohort exists; refuse overwrite')
    if target.exists():
        target.rename(target.with_name('rank-matched-validation-incomplete-'+str(time.time_ns())+'.jsonl'))
    pairs=[];counts=Counter()
    with target.open('x') as f:
        for old in rows(prior/'inputs/matching-candidates.jsonl'):
            p=originals[old['positive_id']];cls=old['mhc_class']
            mapping={candidate_key(m):allele_name(cls,m['provenance']['catalog_alleles']) for m in p['mhc']}
            if any((cls,a) not in scores or p['peptide'] not in scores[(cls,a)] for a in mapping.values()):
                counts[cls+':unscored']+=1;continue
            r={'id':p['observation_id'],'peptide':p['peptide'],'class':cls,'alleles':mapping,
               'protein':old['protein'],'options':old['options']}
            # VAL historical pool is fixed at up to128 options; use all in one step.
            positive=profile(r,r['peptide'],scores);choices=[]
            for original_option in old['options']:
                q=copy.deepcopy(original_option)
                for side in ('source_n','source_c'):
                    if q['context'][side] is not None and set(q['context'][side])-set(AA):
                        q['context'][side]=None;counts[cls+':ambiguous_candidate_flanks']+=1
                if any(q['peptide'] not in scores[(cls,a)] for a in mapping.values()):continue
                if any(p['context'].get(k) is not None and q['context'].get(k) is None for k in ('source_n','source_c','n_terminus','c_terminus')):continue
                negative=profile(r,q['peptide'],scores);assess=assess_match(positive,negative,POLICY)
                if assess.accepted:choices.append((assess.distance,q['peptide'],q,negative,assess))
            if not choices:counts[cls+':unmatched']+=1;continue
            _,_,q,negative,assess=min(choices,key=lambda x:(x[0],x[1]))
            n=make_background(p,r,(q,positive,negative,assess),lineage)
            n['provenance'].pop('processing_pair_positive_id')
            n['provenance']['generation_policy']='Frozen protected historical validation candidate pool; all-candidate BA-rank match; source context knownness matched'
            pid='rank-match:'+hashlib.sha256((p['observation_id']+'|'+q['peptide']).encode()).hexdigest()
            p=copy.deepcopy(p)
            for row in (p,n):
                row['provenance']['pair_id']=pid
                assert row['split']=='validation';Observation.from_dict(row).validate()
                f.write(json.dumps(row,separators=(',',':'))+'\n')
            pairs.append({'pair_id':pid,'positive_id':p['observation_id'],'negative_id':n['observation_id'],
                'class':cls,'protein':old['protein'],'study':p['provenance'].get('study_id') or p['provenance'].get('pmid'),
                'condition':p['provenance'].get('condition_id'),'candidates':len(positive),
                'positive_ba_nm':min(v.nm for v in positive.values()),'negative_ba_nm':min(v.nm for v in negative.values()),
                'positive_ba_rank':min(v.rank for v in positive.values()),'negative_ba_rank':min(v.rank for v in negative.values()),
                'assessment':asdict(assess)})
            counts[cls+':matched']+=1
    dump(root/'rank-matched-validation-pairs.json',pairs)
    result={'status':'prepared','cohort':'hybrid-rank-all-v1','counts':dict(counts),'policy':asdict(POLICY),
            'sha256':sha(target),'path':str(target),'pair_sha256':sha(root/'rank-matched-validation-pairs.json'),
            'old_benchmark_unchanged':True,'test_accessed':False,'lineage':lineage,
            'candidate_contract':'unchanged historical pool, up to128 per positive before observed-peptide exclusions; differs from broad TRAIN first8/adaptive32 existing negatives'}
    dump(root/'rank-matched-validation.json',result)
    return result
