"""Source-qualified development cohort; current metadata, historical sequence proposals only."""
from pathlib import Path
from dataclasses import replace
from collections import Counter,defaultdict
import csv,gzip,hashlib,json,random,time,os
import pandas as pd
import pyarrow.parquet as pq
from presto.data.modular_hitlist import HumanMHCCatalog,AdmissionError
from presto.data.modular_hitlist_ms import ms_observation,study_factor_vocabulary,has_resolved_condition,sample_candidate_set
from presto.data.modular_sources import file_sha256,write_observations
E=Path(__file__).resolve().parents[1]; ROOT=E.parents[1]
POOL=Path('/Users/iskander/code/presto-apm/experiments/2026-11-15_0900_claude_state-bottleneck-v2/data/pool/pairs.parquet')
SAMPLES=ROOT/'experiments/2026-11-16_1835_codex_hitlist-apc-evidence/results/samples.csv'
EMBARGO=ROOT/'experiments/2026-11-07_1815_codex_presentation-cohort/results/paired-v1/held-sequence-embargo.txt.gz'
CATALOG=Path('/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv')
OUT=Path(os.environ.get('PRESTO_APC_OUTPUT',str(E/'results/cohort-v2')));OUT.mkdir(parents=True,exist_ok=True)
def stable(x):return hashlib.sha256(('harmonized-apc-20261116:'+x).encode()).hexdigest()
def clean(x):return None if x is None or (not isinstance(x,(str,list,dict)) and pd.isna(x)) else x
class Similarity:
 def __init__(self,peptides):
  self.peptides=set(peptides);self.cores={p[i:i+8] for p in self.peptides for i in range(len(p)-7)};self.halves=defaultdict(list)
  for p in self.peptides:
   m=len(p)//2
   for side,part in ((0,p[:m]),(1,p[m:])):self.halves[(len(p),side,part)].append(p)
 def reason(self,p):
  if p in self.peptides:return 'exact'
  if any(p[i:i+8] in self.cores for i in range(len(p)-7)):return 'core8'
  m=len(p)//2
  for side,part in ((0,p[:m]),(1,p[m:])):
   if any(sum(a!=b for a,b in zip(p,q,strict=True))<=1 for q in self.halves.get((len(p),side,part),())):return 'hamming1'
def main():
 start=time.time();counts=Counter();source=E/'results/source-inputs.parquet';sha=file_sha256(source)
 raw=[{k:clean(v) for k,v in r.items()} for r in pq.read_table(source).to_pylist()]
 vocabulary=study_factor_vocabulary(raw)
 samples={}
 for r in pd.read_csv(SAMPLES,keep_default_na=False).to_dict('records'):
  key=(int(r['pmid']),r['condition_id']);samples.setdefault(key,[]).append(r)
 pool=pq.read_table(POOL).to_pylist();proposals={};canonical_proteins=set()
 for p in pool:
  if p['parent_in_canonical_train']:canonical_proteins.add(json.loads(p['pos_context'])['protein'])
  key=p['peptide'];proposals.setdefault(key,p)
 del pool
 with gzip.open(EMBARGO,'rt') as f:protected=Similarity(x.strip() for x in f)
 positive_blacklist=set(pq.read_table('/Users/iskander/.hitlist/observations.parquet',columns=['peptide'])['peptide'].to_pylist())
 with CATALOG.open() as f:catalog=HumanMHCCatalog(csv.DictReader(f))
 kept=[];seen=set();forbidden_cache={}
 for ordinal,r in enumerate(raw):
  counts['input_rows']+=1
  if not has_resolved_condition(r):counts['unresolved_condition']+=1;continue
  if r['pmid']==33592498 and r['mhc_class']=='II' and 'parental' in r['condition_id']:counts['parental_II_background']+=1;continue
  p=proposals.get(r['peptide'])
  if p is None:counts['no_sequence_proposal']+=1;continue
  if not 8<=len(r['peptide'])<=25:counts['length_outside_pilot']+=1;continue
  def excluded(seq):
   if seq not in forbidden_cache:forbidden_cache[seq]=protected.reason(seq)
   return forbidden_cache[seq]
  if excluded(r['peptide']):counts['protected_positive']+=1;continue
  sample=samples.get((r['pmid'],r['condition_id']),[])
  try:
   if not r['mhc_allele_set']:
    if len(sample)!=1:raise AdmissionError('nonunique_sample_join')
    r=sample_candidate_set(r,sample[0],reference=f'{SAMPLES}#{r["pmid"]}:{r["condition_id"]}')
   row=ms_observation(r,catalog,source_sha256=sha,ordinal=ordinal,study_factors=vocabulary,
     dra_reference='HLA-DRA*01:01 reference alpha for source-reported DR beta; not measured typing',
     legacy_source_reference=f'pinned current Hitlist export {sha}; PMID {r["pmid"]}; no indexed contributor assertion',
     unspecified_modification_policy='sequence_only_unknown_ptm' if r['pmid']==40113210 else None)
  except (AdmissionError,ValueError) as error:counts['admission:'+getattr(error,'reason',str(error))]+=1;continue
  key=(r['peptide'],r['pmid'],r['condition_id'],r['mhc_class'],r['mhc_allele_set'])
  if key in seen:counts['duplicate_peptide_arm_candidates']+=1;continue
  seen.add(key)
  pos=json.loads(p['pos_context']);options=json.loads(p['options']);decoy=next((q for q in options if q['peptide'] not in positive_blacklist and not excluded(q['peptide'])),None)
  if decoy is None:counts['no_qualified_decoy']+=1;continue
  protein=pos['protein'];split='validation' if protein not in canonical_proteins and int(stable(protein)[:8],16)%5==0 else 'train'
  group=(r['pmid'],r['condition_id'],r['mhc_class']);pair=stable('|'.join(map(str,key)))[:24]
  context=replace(row.context,**{k:pos[k] for k in ('source_n','source_c','n_terminus','c_terminus')})
  common={'pmid':str(r['pmid']),'study_id':str(r['pmid']),'condition_id':r['condition_id'],'pair_id':pair,'source_protein':protein,'candidate_policy':'current reported candidates; exact-condition sample typing only when missing','canonical_parent_exposure':protein in canonical_proteins,'canonical_study_exposure':'not inferred from cross-study sequence proposals'}
  row=replace(row,split=split,group_id=protein,context=context,provenance={**row.provenance,**common})
  negid='generated:'+pair
  neg=replace(row,observation_id=negid,peptide=decoy['peptide'],value=0.,context=replace(context,**decoy['context']),
   provenance={**common,'original_measurement_id':negid,'measurement_namespace':'synthetic_same_protein','label_kind':'generated_background','parent_positive_id':row.observation_id,'generation_policy':'one same-protein same-length proposal excluding all cached MS positives and protected related sequences; not a measured negative','raw_record':{'pmid':str(r['pmid'])}})
  kept.append((group,split,row,neg))
 print('admitted pairs',len(kept),flush=True)
 # Keep all observations of a source protein on one side, then embargo sequence relatives.
 held=Similarity({r.peptide for _,split,*rs in kept if split=='validation' for r in rs})
 kept=[k for k in kept if k[1]=='validation' or not any(held.reason(r.peptide) for r in k[2:])]
 # Stable per-arm budgets; held-out labels did not determine selection.
 grouped=defaultdict(list)
 for group,split,*rs in kept:grouped[(group,split)].append(rs)
 selected=[]
 for (group,split),pairs in sorted(grouped.items()):
  limit=35 if split=='validation' else 140
  selected.extend((group,split,*rs) for rs in sorted(pairs,key=lambda rs:stable(rs[0].observation_id))[:limit])
 # Cap each study/class so the large monoallelic atlas cannot dominate this pilot.
 study_groups=defaultdict(list)
 for item in selected:study_groups[(item[0][0],item[0][2],item[1])].append(item)
 selected=[]
 for (_,_,split),items in sorted(study_groups.items()):
  selected.extend(sorted(items,key=lambda item:stable(item[2].observation_id))[:300 if split=='validation' else 1200])
 # Deduplicate original outcomes even if source export repeats a condition assignment.
 used=set();selected=[p for p in selected if not (p[2].observation_id in used or used.add(p[2].observation_id))]
 out=OUT/'data';out.mkdir(exist_ok=True)
 rows=[r for _,_,*rs in selected for r in rs]
 for split in ('train','validation'):
  write_observations(out/f'{split}.jsonl',[r for r in rows if r.split==split])
 # Shuffle whole APC context within study/cell/class; pairs retain the same label.
 buckets=defaultdict(list)
 for i,(group,split,pos,neg) in enumerate(selected):
  if split=='train':buckets[(group[0],group[2],pos.context.cell_type)].append(i)
 shuffled=list(selected);rng=random.Random(20261116)
 for indices in buckets.values():
  contexts=[selected[i][2].context for i in indices];rng.shuffle(contexts)
  for i,ctx in zip(indices,contexts,strict=True):
   group,split,pos,neg=selected[i]
   def change(r):return replace(r,context=replace(r.context,components=ctx.components,cytokines=ctx.cytokines,interventions=ctx.interventions,cell_type=ctx.cell_type),provenance={**r.provenance,'context_control':'within study/cell/class permutation'})
   shuffled[i]=(group,split,change(pos),change(neg))
 write_observations(out/'train-shuffled.jsonl',[r for _,split,*rs in shuffled if split=='train' for r in rs])
 states=defaultdict(set);cells=set();cytos=set();factors=set()
 for r in rows:
  if r.split=='train':
   if r.context.cell_type:cells.add(r.context.cell_type)
   cytos.update(r.context.cytokines);factors.update(r.context.interventions)
   for k,v in r.context.interventions.items():states[k].add(v.state)
 # All vocabulary comes from TRAIN. Validation-only factors are deferred explicitly.
 # In practice the stratified arm budgets leave all admitted conditions represented.
 report={'counts':dict(counts),'pairs':len(selected),'rows':len(rows),'by_arm':[{ 'pmid':g[0],'condition':g[1],'class':g[2],'split':s,'pairs':sum(1 for a,b,*_ in selected if a==g and b==s)} for g,s in sorted({(a,b) for a,b,*_ in selected})],
 'model_context':{'cell_types':sorted(cells),'cytokines':sorted(cytos),'interventions':sorted(factors)},'training_intervention_states':{k:sorted(v) for k,v in states.items()},'inputs':{str(p):file_sha256(p) for p in (source,POOL,SAMPLES,EMBARGO,CATALOG)},'data':{p.name:file_sha256(p) for p in out.glob('*.jsonl')},'seconds':time.time()-start,'test_policy':'sealed; identity-only embargo','source_policy':'HAP1 sequences may have unspecified PTMs; known modified rows excluded; legacy measurement IDs retained without inventing contributor IDs'}
 (OUT/'cohort.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('counts','pairs','rows','model_context','seconds')},indent=2),flush=True)
if __name__=='__main__':main()
