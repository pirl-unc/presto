"""Read-only full-corpus and existing-cache matching audit; never reads TEST outcomes."""
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import time

import modal

NAME = '2026-11-17_2330_codex_affinity-conditioned-processing'
E = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/experiment')
R = E.parents[1] if modal.is_local() else Path('/')
S = R / 'artifacts/harmonized-apc-source'
OLD = '2026-11-16_2035_codex_joint-apc-capacity'
BROAD = '2026-11-17_1155_codex_shared-presentation-expansion'
app = modal.App('presto-processing-matching-audit')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12').env({'PYTHONPATH': '/bundle'})
if modal.is_local():
    image = image.add_local_file(S / 'data/affinity_matching.py', '/bundle/affinity_matching.py')


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def allele_name(cls, alleles):
    if cls == 'I':
        return alleles[0].replace('*', '')
    parts = [x.replace('HLA-', '').replace('*', '').replace(':', '') for x in alleles]
    dr = next((x for x in parts if x.startswith('DRB')), None)
    return dr[:4] + '_' + dr[4:] if dr else 'HLA-' + '-'.join(parts)


def quantiles(values):
    v = sorted(values)
    return {'n':len(v), **({str(p):v[round((len(v)-1)*p)] for p in (0,.1,.5,.9,1)} if v else {})}


def auc(pos, neg):
    if not pos or not neg:
        return None
    data = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    total = 0.; negatives = 0; i = 0
    while i < len(data):
        j = i
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        p = sum(label for _, label in data[i:j]); n = j-i-p
        total += p * (negatives + .5*n); negatives += n; i=j
    return total/(len(pos)*len(neg))


@app.function(image=image,cpu=4,memory=32768,timeout=7200,volumes={'/volume':volume})
def audit():
    from affinity_matching import AffinityScore, MatchingPolicy, assess_match
    start=time.monotonic(); root=Path('/volume')/NAME
    root.mkdir(exist_ok=True)
    if (root/'audit.json').exists():
        return json.loads((root/'audit.json').read_text())
    prior=Path('/volume')/OLD; broad=Path('/volume')/BROAD
    census={}; all_queries=defaultdict(set)
    for name in ['expanded-train.jsonl','expanded-validation.jsonl','apc-validation.jsonl']:
        stats=defaultdict(Counter);studies=defaultdict(set); seqs=defaultdict(set);pool=defaultdict(set)
        h=hashlib.sha256()
        with (broad/'inputs'/name).open('rb',buffering=1024*1024) as stream:
            for line in stream:
                h.update(line); r=json.loads(line)
                assert r['split'] in {'train','validation'}
                if r['endpoint']!='elution_observation':continue
                classes={m.get('class_override') or m['mhc_class'] for m in r['mhc']}
                assert len(classes)==1;cls=classes.pop();c=r['context'];p=r['provenance'];key=cls
                stats[key]['rows']+=1
                if r['value'] !=1:continue
                stats[key]['positives']+=1;studies[key].add(str(p.get('pmid') or p.get('study_id')));seqs[key].add(r['peptide'])
                stats[key]['monoallelic' if len(r['mhc'])==1 else 'multiallelic']+=1
                stats[key]['both_flanks_known']+=int(c.get('source_n') is not None and c.get('source_c') is not None)
                stats[key]['both_flanks_nonempty']+=int(bool(c.get('source_n')) and bool(c.get('source_c')))
                stats[key]['one_or_more_flanks_known']+=int(c.get('source_n') is not None or c.get('source_c') is not None)
                stats[key]['protein_terminus']+=int(c.get('n_terminus') is True or c.get('c_terminus') is True)
                stats[key]['reported_source_protein']+=int(bool(p.get('source_protein')))
                stats[key]['source_candidates_gt1']+=int(p.get('source_candidate_count',1)>1)
                supported=(8<=len(r['peptide'])<=14) if cls=='I' else (9<=len(r['peptide'])<=30)
                stats[key]['predictor_length_supported']+=int(supported)
                for m in r['mhc']:
                    alleles=m.get('provenance',{}).get('catalog_alleles',[])
                    if not alleles:stats[key]['missing_external_allele_names']+=1;continue
                    allele=allele_name(cls,alleles)
                    if supported and name=='expanded-train.jsonl':all_queries[(cls,allele)].add(r['peptide'])
        expected=json.loads((broad/'preparation.json').read_text())['files'][name]['sha256']
        assert h.hexdigest()==expected,(name,'changed source')
        census[name]={'sha256':h.hexdigest(),'classes':{k:{**dict(v),'studies':len(studies[k]),'unique_peptides':len(seqs[k])} for k,v in stats.items()}}
        print(json.dumps({'census':name,**census[name]}),flush=True)
    policies={'legacy_nm':MatchingPolicy('log10_nm',.1,'winner',None)}
    for gap in [.1,.25,.5,1.]:
        for profile in ['winner','all']:
            policies[f'rank_{gap}_{profile}']=MatchingPolicy('ba_rank',gap,profile,None)
    for profile in ['winner','all']:
        policies[f'rank_hybrid_{profile}']=MatchingPolicy('ba_rank',.5,profile,.3)
    matched=[]; summaries=[]; cache_census={}
    for split,prefix,namespace in [('validation','','external'),('train','train-','external-train')]:
        scores={}
        for path in sorted((prior/namespace).glob('*.json')):
            d=json.loads(path.read_text());scores[(d['class'],d['allele'])]=d['scores']
        cache_census[split]={'allele_files':len(scores),'peptide_allele_scores':sum(map(len,scores.values()))}
        if not scores:continue
        candidates=prior/'inputs'/(prefix+'matching-candidates.jsonl')
        accum=defaultdict(list);eligible=Counter();missing=Counter()
        for r in rows(candidates):
            cls=r['mhc_class'];alleles=sorted({allele_name(cls,a) for a in r['alleles']})
            if any((cls,a) not in scores or r['peptide'] not in scores[(cls,a)] for a in alleles):
                missing[cls]+=1;continue
            eligible[cls]+=1
            def profile(pep):
                return {a:AffinityScore(scores[(cls,a)][pep]['ba_nm'],scores[(cls,a)][pep]['ba_rank']) for a in alleles}
            pp=profile(r['peptide']);choices={key:[] for key in policies}
            for b in r['options']:
                if any(b['peptide'] not in scores[(cls,a)] for a in alleles):raise ValueError('incomplete candidate cache')
                bp=profile(b['peptide'])
                for key,policy in policies.items():
                    q=assess_match(pp,bp,policy)
                    if q.accepted:choices[key].append((q.distance,b['peptide'],q,bp))
            for key,options in choices.items():
                if not options:continue
                _,pep,q,bp=min(options,key=lambda v:(v[0],v[1]));winner=q.positive_winner
                item={'split':split,'class':cls,'policy':key,'pair_id':r['pair_id'],'positive_id':r['positive_id'],'background':pep,'winner':winner,'candidate_count':len(alleles),'positive_nm':pp[winner].nm,'negative_nm':bp[winner].nm,'positive_rank':pp[winner].rank,'negative_rank':bp[winner].rank,**asdict(q)}
                matched.append(item);accum[(key,cls)].append(item)
        for (key,cls),items in sorted(accum.items()):
            summaries.append({'split':split,'class':cls,'policy':key,'eligible':eligible[cls],'retained':len(items),'fraction':len(items)/eligible[cls], 'monoallelic':sum(x['candidate_count']==1 for x in items),'positive_nm':quantiles([x['positive_nm'] for x in items]),'positive_rank':quantiles([x['positive_rank'] for x in items]),'max_profile_nm_gap':quantiles([x['max_log_nm_gap'] for x in items]),'ba_auc':auc([-x['positive_nm'] for x in items],[-x['negative_nm'] for x in items]),'rank_auc':auc([-x['positive_rank'] for x in items],[-x['negative_rank'] for x in items]),'ba_pair_win':sum((x['positive_nm']<x['negative_nm'])+.5*(x['positive_nm']==x['negative_nm']) for x in items)/len(items)})
        cache_census[split]['unsupported_positive_rows']=dict(missing)
    (root/'audit-matching-pairs.json').write_text(json.dumps(matched,separators=(',',':'))+'\n')
    report={'status':'passed','seconds':time.monotonic()-start,'census':census,'existing_external_cache':cache_census,'policies':{k:asdict(p) for k,p in policies.items()},'matching_summary':summaries,'native_training_positive_query_counts':{c:sum(len(p) for (cl,a),p in all_queries.items() if cl==c) for c in ['I','II']},'native_training_allele_jobs':dict(Counter(c for c,a in all_queries)),'test_accessed':False}
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n');volume.commit()
    return report


@app.local_entrypoint()
def main():
    call=audit.spawn()
    (E/'reproduce/audit-call.json').write_text(json.dumps({'call':call.object_id},indent=2)+'\n')
    print('audit call',call.object_id,flush=True)
    result=call.get()
    (E/'results/audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
