"""Admit literal purified assays with frozen cross-endpoint leakage protection."""
from pathlib import Path
from collections import Counter,defaultdict
from dataclasses import replace
import csv,gzip,hashlib,json,shutil,time
import pyarrow.parquet as pq
from presto.data.modular import Observation
from presto.data.modular_hitlist import HumanMHCCatalog,AdmissionError,purified_assay_observation
from presto.data.modular_preparation import prepare_observations
from presto.data.modular_lineage import measurement_key,semantic_fingerprint
from presto.data.modular_sources import file_sha256,ObservationIndex
ROOT=Path(__file__).resolve().parents[3];FAMILY=Path(__file__).resolve().parents[1]
COHORT=ROOT/'data/modular_cohorts/2026-10-27-purified-multi-assay'
SOURCE=Path('/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z')
SOURCES={'binding.parquet':'2b3f6ea2dd647e7fc54d214eef420feaebc4e81fab97ffc0565a2de1f8695ba0','other_assays.parquet':'c40f5227ded408830885ecba8336dbdd701172df9573b0d32e5e8a5f8fc947c2'}
CATALOG=Path('/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv')
HISTORY=Path('/Users/iskander/.hitlist/presto-corrected-human-qualification/public-downloads/v1/bundles/2026-10-16T08-32-18Z/membership.tsv')
OLD=ROOT/'artifacts/2026-10-27_2100_codex_transfer-improvements/upload/inputs'
POP='literal purified human quantitative pMHC assays; canonical peptide and verified complete HLA pair'
LABEL='endpoint-specific physical measurements; original numeric bounds; missing exported qualifier exact with missingness retained'
SPLIT='peptide/original-assay connected components; frozen historical and IC50 splits inherited; conflicting components deferred'
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
 start=time.monotonic();COHORT.mkdir(parents=True,exist_ok=False)
 assert file_sha256(CATALOG)=='ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1'
 assert file_sha256(HISTORY)=='8959e60c0d9b8049aec25095b9c354d456b48f97ba8fb31d7b786e2861113e99'
 with CATALOG.open() as f:catalog=HumanMHCCatalog(csv.DictReader(f))
 history=defaultdict(set);assay_splits=defaultdict(set);old_keys={};old_sources=[]
 with HISTORY.open() as f:
  for x in csv.DictReader(f,delimiter='\t'):
   if x['peptide']:history[x['peptide']].add({'val':'validation'}.get(x['source_partition'],x['source_partition']))
 for split in ('train','validation','test'):
  old=OLD/f'joint-{split}.jsonl';new=COHORT/f'ic50-original-{split}.jsonl';shutil.copy2(old,new)
  for line in old.open():
   row=Observation.from_dict(json.loads(line));history[row.peptide].add(split);key=measurement_key(row);old_keys[key]=row;assay_splits[key[:2]].add(split)
  old_sources.append(dict(path=new.name,sha256=file_sha256(new),source='hitlist-v4-iedb',population='original admitted human IC50 cohort; frozen unchanged',label_policy=LABEL,split_policy=SPLIT))
 rows=[];counts=Counter();total=0
 disposition=FAMILY/'results/quantitative-source-dispositions.jsonl.gz'
 with gzip.open(disposition,'wt') as log:
  for filename,digest in SOURCES.items():
   assert file_sha256(SOURCE/filename)==digest
   ordinal=0
   for batch in pq.ParquetFile(SOURCE/filename).iter_batches(batch_size=4096):
    for raw in batch.to_pylist():
     total+=1
     try:
      row=purified_assay_observation(raw,catalog,source_sha256=digest,ordinal=ordinal)
      key=measurement_key(row)
      if key in old_keys:
       old=old_keys[key]
       reason='original_ic50_outcome_retained' if semantic_fingerprint(row)==semantic_fingerprint(old) else 'conflict_with_frozen_ic50_outcome'
       counts[reason]+=1;log.write(json.dumps(dict(source_file=filename,ordinal=ordinal,disposition='preserved_existing' if reason.startswith('original') else 'deferred',reason=reason))+'\n')
      else:
       row=replace(row,observation_id=f'hitlist-v4-{filename[:-8]}:{ordinal}',provenance={**row.provenance,'source_file':filename})
       rows.append(row);counts['candidate:'+row.endpoint]+=1
     except AdmissionError as e:
      counts[e.disposition+':'+e.reason]+=1
      log.write(json.dumps(dict(source_file=filename,ordinal=ordinal,disposition=e.disposition,reason=e.reason))+'\n')
     ordinal+=1
   print(filename,'scanned',ordinal,'candidates',len(rows),flush=True)
 parents=list(range(len(rows)));seen={}
 def find(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 for i,row in enumerate(rows):
  for key in [('peptide',row.peptide),('assay',*measurement_key(row)[:2])]:
   if key in seen:parents[find(i)]=find(seen[key])
   else:seen[key]=i
 groups=defaultdict(list)
 for i in range(len(rows)):groups[find(i)].append(i)
 accepted=[];group_audit=[]
 for members in groups.values():
  peptides=sorted({rows[i].peptide for i in members});assays=sorted({measurement_key(rows[i])[:2] for i in members})
  old_splits={s for p in peptides for s in history[p]}|{s for a in assays for s in assay_splits[a]}
  digest=hashlib.sha256(json.dumps([peptides,assays]).encode()).hexdigest();bucket=int(hashlib.sha256(('presto-multi-assay-v1:'+digest).encode()).hexdigest(),16)%100
  split=next(iter(old_splits)) if len(old_splits)==1 else ('train' if bucket<80 else 'validation' if bucket<90 else 'test')
  group_audit.append(dict(component=digest,peptides=peptides,assays=assays,n=len(members),historical_splits=sorted(old_splits),split=split if len(old_splits)<2 else None))
  for i in members:
   row=rows[i]
   if len(old_splits)>1:counts['deferred:conflicting_frozen_component:'+row.endpoint]+=1;continue
   accepted.append(replace(row,split=split,group_id='multi-assay-component:'+digest,provenance={**row.provenance,'partition_basis':'frozen_history' if old_splits else 'novel_component_hash'}))
 write(FAMILY/'results/multi-assay-components.json',group_audit)
 print('partitioned',len(accepted),'rows',flush=True)
 prepare_observations(accepted,COHORT/'prepared',population=POP,label_policy=LABEL,split_policy=SPLIT)
 # Repartition admitted canonical rows by endpoint for explicit condition manifests.
 admitted=[]
 for path in (COHORT/'prepared').glob('*.jsonl'):
  if path.name in ('dispositions.jsonl','contradictions.jsonl'):continue
  for line in path.open():
   value=json.loads(line)
   if 'observation_id' in value:admitted.append(Observation.from_dict(value))
 streams={};census=Counter();source_specs=[]
 try:
  for row in admitted:
   key=(row.endpoint,row.source,row.split)
   if key not in streams:streams[key]=(COHORT/('-'.join(key)+'.jsonl')).open('w')
   streams[key].write(json.dumps(row.to_dict(),allow_nan=False)+'\n');census[(row.endpoint,row.split,row.mhc[0].mhc_class,row.qualifier)]+=1
 finally:
  for stream in streams.values():stream.close()
 for key,stream in streams.items():
  path=Path(stream.name);source_specs.append(dict(path=path.name,sha256=file_sha256(path),source=key[1],population=POP,label_policy=LABEL,split_policy=SPLIT))
 endpoints=sorted({'ic50',*(row.endpoint for row in admitted)})
 manifest=dict(schema='presto.modular.manifest.v1',model=dict(endpoints=endpoints,mhc_classes=['I','II'],species=['human'],identity_tasks=[],dropout=.1),training=dict(identity_weight=0,balance_by=[]),sources=old_sources+source_specs)
 write(COHORT/'manifest.json',manifest)
 index=ObservationIndex(COHORT/'manifest.json')
 summary=dict(status='admitted',raw_rows=total,candidate_rows=len(rows),partition_admitted=len(accepted),deduplicated_new_rows=len(admitted),total_rows=len(index),counts=dict(counts),census=[dict(endpoint=k[0],split=k[1],mhc_class=k[2],qualifier=k[3],n=v) for k,v in sorted(census.items())],endpoints=endpoints,cohort=str(COHORT),manifest_sha256=file_sha256(COHORT/'manifest.json'),source_hashes=SOURCES,source_catalog_sha256=file_sha256(CATALOG),historical_membership_sha256=file_sha256(HISTORY),seconds=time.monotonic()-start,synthetic_labels=0,pretraining=False,original_ic50_files_byte_identical=True)
 write(FAMILY/'results/data-admission.json',summary);write(FAMILY/'results/used-catalog-aliases.json',catalog.aliases());print(json.dumps({k:v for k,v in summary.items() if k not in ('counts','census')},indent=2),flush=True)
if __name__=='__main__':main()
