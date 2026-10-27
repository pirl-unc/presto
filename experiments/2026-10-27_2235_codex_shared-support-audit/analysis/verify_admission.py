"""Verify admitted rows against raw exports and catalog without using their adapter."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,math,re,time
import pyarrow.parquet as pq
F=Path(__file__).resolve().parents[1];ROOT=F.parents[1]
COHORT=ROOT/'data/modular_cohorts/2026-10-27-purified-multi-assay'
SOURCE=Path('/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z')
CATALOG=Path('/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv')
MAPPING={'half maximal inhibitory concentration (IC50)':'ic50','dissociation constant KD':'kd','half maximal effective concentration (EC50)':'ec50','dissociation constant KD (~IC50)':'kd_via_ic50','dissociation constant KD (~EC50)':'kd_via_ec50','half life':'half_life','off rate':'koff'}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def main():
 started=time.monotonic();manifest=json.loads((COHORT/'manifest.json').read_text());admission=json.loads((F/'results/data-admission.json').read_text());expected={};seen_ids=set();peptides=defaultdict(set);assays=defaultdict(set);lineage=set();census=Counter();pairs={};inputs=defaultdict(list);known={}
 with CATALOG.open() as f:
  for c in csv.DictReader(f):
   if c['source']=='imgt' and c['species']=='Homo sapiens' and c['is_functional']=='True' and all(c[k]=='False' for k in ('is_null','is_questionable','is_pseudogene')) and c['groove_status']=='ok' and not c['groove_flags']:
    known.setdefault(c['two_field_allele'].replace('HomoSapiens-','HLA-',1),set()).add((c['mhc_class'],c['chain'],c['groove1'],c['groove2']))
 for spec in manifest['sources']:
  path=COHORT/spec['path'];assert sha(path)==spec['sha256']
  for line in path.open():
   r=json.loads(line);p=r['provenance'];raw=p['raw_record'];ordinal=p['source_ordinal']-(0 if 'source_file' in p else 1);file=p.get('source_file','binding.parquet')
   assert r['observation_id'] not in seen_ids;seen_ids.add(r['observation_id']);assert r['source']==spec['source']
   assert p['source_sha256']==admission['source_hashes'][file]
   key=(file,ordinal);fingerprint=digest(raw)
   if key in expected:assert expected[key]==fingerprint
   expected[key]=fingerprint
   assert r['endpoint']==MAPPING[raw['response_measured']]
   assert r['value']==raw['quantitative_value'] and r['value']>0 and math.isfinite(r['value'])
   assert r['unit']==raw['measurement_units']
   q=raw['measurement_inequality'] or '=';assert r['qualifier']==q
   text=raw['quantitative_measurement'].strip();number=re.sub(r'^[<>=]+\s*','',text);assert float(number)==r['value']
   assert r['peptide']==raw['peptide'] and not set(r['peptide'])-set('ACDEFGHIKLMNPQRSTVWY')
   assert raw['has_ptm'] is False and not raw['peptide_modifications'] and not raw['is_non_peptide_ligand']
   assert raw['source'] in ['iedb','cedar'] and raw['assay_method'].startswith('purified MHC/')
   assert p['original_measurement_id']==raw['assay_iri']
   identity=(p['measurement_namespace'],p['original_measurement_id'],p.get('original_replicate_id'),r['endpoint']);assert identity not in lineage;lineage.add(identity)
   peptides[r['peptide']].add(r['split']);assays[identity[:2]].add(r['split'])
   m=r['mhc'][0];names=m['provenance']['catalog_alleles'];assert not m['provenance']['partner_imputation'];assert m['mhc_class']==raw['mhc_class'];assert m['species1']==m['species2']=='human'
   declared=raw['mhc_restriction'].split('/');declared=[n if n.startswith('HLA-') else 'HLA-'+n for n in declared];assert names==declared
   for name in names:assert len(known[name])==1
   values=[next(iter(known[n])) for n in names]
   if m['mhc_class']=='I':assert len(values)==1 and values[0]==('I','alpha',m['groove1'],m['groove2'])
   else:
    by_chain={v[1]:v for v in values};assert set(by_chain)=={'alpha','beta'} and by_chain['alpha'][2]==m['groove1'] and by_chain['beta'][3]==m['groove2']
   pairs[(m['groove1'],m['groove2'])]=names
   census[(r['endpoint'],r['split'],m['mhc_class'],q)]+=1
   if r['split']=='train' and q=='=':
    unitfactor={'pM':.001,'nM':1,'uM':1000,'mM':1e6,'M':1e9,'min':1,'1/s':1}[r['unit']]
    inputs[(r['endpoint'],r['peptide'],m['groove1'],m['groove2'])].append(math.log10(r['value']*unitfactor))
 assert all(len(v)==1 for v in peptides.values()) and all(len(v)==1 for v in assays.values())
 matched=0
 for filename in admission['source_hashes']:
  assert sha(SOURCE/filename)==admission['source_hashes'][filename]
  ordinal=0
  for batch in pq.ParquetFile(SOURCE/filename).iter_batches(batch_size=4096):
   for raw in batch.to_pylist():
    if (filename,ordinal) in expected:
     assert digest(raw)==expected[filename,ordinal];matched+=1
    ordinal+=1
 assert matched==len(expected)
 disagreements=defaultdict(list)
 for k,v in inputs.items():
  if len(v)>1:disagreements[k[0]].append(dict(n=len(v),range=max(v)-min(v),sd=(sum((x-sum(v)/len(v))**2 for x in v)/len(v))**.5))
 import numpy as np
 result=dict(source_ordinal_conventions={'frozen_original_ic50':'one-based; verified from pinned original preparer','new_quantitative_sources':'zero-based'},status='passed',rows=len(seen_ids),raw_source_rows_verified=matched,unique_peptides=len(peptides),unique_groove_pairs=len(pairs),cross_endpoint_assay_leaks=0,peptide_leaks=0,original_outcome_duplicates=0,census=[dict(endpoint=k[0],split=k[1],mhc_class=k[2],qualifier=k[3],n=v) for k,v in sorted(census.items())],repeated_input_training_assays={k:dict(groups=len(v),groups_over_1log=sum(x['range']>1 for x in v),median_range=float(np.median([x['range'] for x in v])),interpretation='Independent readouts; assay conditions are not matched. Not an irreducible-noise estimate.') for k,v in disagreements.items()},seconds=time.monotonic()-started)
 (F/'results/admission-independent-verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['census','repeated_input_training_assays']},indent=2))
if __name__=='__main__':main()
