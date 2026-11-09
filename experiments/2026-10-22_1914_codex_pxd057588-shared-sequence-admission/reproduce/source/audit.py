"""Consume shared PXD057588 and audit exact mouse reference support."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import datacache
import pyarrow.parquet as pq

FAMILY=Path(__file__).resolve().parents[1]
ROOT=FAMILY.parents[1]
RESULTS=FAMILY/'results'
PACKAGE=Path('/private/tmp/hitlist-v2-external-18222-audited')
PREVIOUS=ROOT/'experiments/2026-10-22_1845_codex_mapp-parent-reference'
EXPECTED='daff7930e193c6dbde1fe84c065b024187232147fa006d9348e297766e95e69c'

def load_helper():
 p=PREVIOUS/'reproduce/source/audit.py'
 launch=json.loads((PREVIOUS/'reproduce/launch.json').read_text())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==launch['code_sha256']
 spec=importlib.util.spec_from_file_location('frozen_mapp_audit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 shutil.copy2(p,FAMILY/'reproduce/source/mapp_audit.py')
 return m

def main():
 tick=time.monotonic();h=load_helper()
 assert not (RESULTS/'summary.json').exists()
 assert h.sha(PACKAGE/'manifest.json')==EXPECTED
 assert json.loads((RESULTS/'package-verification.json').read_text())['passed']
 manifest=json.loads((PACKAGE/'manifest.json').read_text())
 reference=json.loads((FAMILY/'reproduce/reference_asset.json').read_text())
 assets={name:{'url':(PACKAGE/name).as_uri(),'sha256':r['sha256'],'size':r['size_bytes'],'decompress':False} for name,r in manifest['files'].items() if name.startswith('external/pxd057588/')}
 assets['shared_manifest.json']={'url':(PACKAGE/'manifest.json').as_uri(),'sha256':EXPECTED,'size':(PACKAGE/'manifest.json').stat().st_size,'decompress':False}
 assets['mouse-reviewed-2026_03-isoforms.fasta']={k:reference[k] for k in ['url','sha256','size','decompress']}
 seed={name:str(PACKAGE/name) for name in assets if name.startswith('external/')};seed['shared_manifest.json']=str(PACKAGE/'manifest.json');seed['mouse-reviewed-2026_03-isoforms.fasta']='/private/tmp/presto-mouse-reviewed-isoforms.fasta'
 h.dump(FAMILY/'reproduce/input_assets.json',{'assets':assets,'seed_paths':seed})
 key=hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest()
 installed=datacache.install_bundle(ROOT/'data/availability_sources'/key,assets,source_paths=seed)
 inode_report={}
 for name,p in installed.items():
  assert h.sha(p)==assets[name]['sha256'];inode_report[name]=Path(p).stat().st_ino!=Path(seed[name]).stat().st_ino
 assert all(inode_report.values()),'do not share mutable source inodes'
 h.dump(RESULTS/'input_paths.json',{k:str(v) for k,v in installed.items()})
 h.dump(FAMILY/'reproduce/launch.json',{'agent_model':'Codex / GPT-6','started':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,'python':sys.version,'executable':sys.executable,'cwd':str(ROOT),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],text=True),'environment_overrides':{},'code_sha256':h.sha(__file__),'helper_sha256':h.sha(PREVIOUS/'reproduce/source/audit.py'),'shared_manifest_sha256':EXPECTED,'reference_sha256':reference['sha256'],'input_independent_inodes':inode_report,'device':'local CPU','model':None,'training':None,'pretraining':None,'loss':None,'synthetic_training':None,'predictive_split':None,'no_prediction_dump_reason':'shared-source sequence admission audit; no fitted model'})
 shutil.copy2(__file__,FAMILY/'reproduce/source/audit.py')
 def table(name):return pq.read_table(installed['external/pxd057588/typed/'+name+'.parquet']).to_pylist()
 rows=table('peptide_rows');assert len(rows)==35772 and len({r['peptide_row_id'] for r in rows})==35772
 seqs=sorted({r['sequence_raw'] for r in rows});assert len(seqs)==30587
 assert all(set(s)<=set('ACDEFGHIKLMNPQRSTVWYUO') and len(s)>=6 for s in seqs)
 peptides={str(i):s for i,s in enumerate(seqs)};seq_ids={s:i for i,s in peptides.items()}
 ref=h.parse_fasta(installed['mouse-reviewed-2026_03-isoforms.fasta']);records={r['accession']:r for r in ref};assert len(records)==len(ref)
 h.jsonl(RESULTS/'reference_entries.jsonl.gz',[{k:v for k,v in r.items() if k!='sequence'} for r in ref])
 print('scanning',len(seqs),'sequences against',len(ref),'mouse entries',flush=True)
 found=h.scan(ref,peptides);components=h.components(found)
 hits=[];parent_sets={pid:{acc for acc,_ in found[pid]} for pid in peptides}
 for pid in peptides:
  for acc,start in found[pid]:
   r=records[acc];seq=r['sequence'];end=start+len(peptides[pid])-1
   hits.append({'sequence_id':pid,'accession':acc,'start1':start,'end1':end,'left':seq[start-2] if start>1 else '-','right':seq[end] if end<len(seq) else '-','parent_sequence_sha256':r['sequence_sha256']})
 h.csvout(RESULTS/'matches.csv.gz',hits)
 matched_parent_ids={a for ps in parent_sets.values() for a in ps}
 h.jsonl(RESULTS/'matched_parents.jsonl.gz',[r for r in ref if r['accession'] in matched_parent_ids])
 sequence_rows=[{'sequence_id':pid,'sequence':p,'length':len(p),'reference_identity_count':len(parent_sets[pid]),'reference_full_sequence_count':len({records[a]['sequence_sha256'] for a in parent_sets[pid]}),'reference_occurrence_count':len(found[pid]),'component':components.get(pid,'')} for pid,p in peptides.items()]
 h.csvout(RESULTS/'sequences.csv.gz',sequence_rows)
 mapping=[]
 for r in rows:
  pid=seq_ids[r['sequence_raw']];acc=r['parent_label_raw'];parents=parent_sets[pid]
  direct=[start for a,start in found[pid] if a==acc]
  mapping.append({**r,'sequence_id':pid,'source_parent_status':'missing_id' if acc not in records else 'exact_sequence_match' if direct else 'resolved_id_peptide_absent','source_parent_positions_start1':direct,'reference_identity_count':len(parents),'reference_full_sequence_count':len({records[a]['sequence_sha256'] for a in parents}),'reference_occurrence_count':len(found[pid]),'additional_parent_identities':len(parents-{acc}),'component':components.get(pid,'')})
 h.jsonl(RESULTS/'source_mapping.jsonl.gz',mapping)
 strata=[]
 for t in ['all',7,8,9,10,11,12]:
  selected=[r for r in mapping if t=='all' or r['table_number']==t]
  ids={r['sequence_id'] for r in selected};parents={a for pid in ids for a in parent_sets[pid]};comp=Counter(components[pid] for pid in ids if pid in components)
  strata.append({'table':t,'source_rows':len(selected),'distinct_sequences':len(ids),'exact_reference_matched_rows':sum(r['reference_identity_count']>0 for r in selected),'source_parent_status':dict(Counter(r['source_parent_status'] for r in selected)),'single_parent_identity_rows':sum(r['reference_identity_count']==1 for r in selected),'single_full_parent_sequence_rows':sum(r['reference_full_sequence_count']==1 for r in selected),'multiple_full_parent_sequence_rows':sum(r['reference_full_sequence_count']>1 for r in selected),'additional_parent_rows':sum(r['additional_parent_identities']>0 for r in selected),'matched_parent_identities':len(parents),'matched_full_parent_sequences':len({records[a]['sequence_sha256'] for a in parents}),'components':len(comp),'largest_component_sequences':max(comp.values(),default=0)})
 source_label_support=defaultdict(lambda:{9:set(),11:set()})
 verified_label_support=defaultdict(lambda:{9:set(),11:set()})
 all_parent_support=defaultdict(lambda:{9:set(),11:set()})
 fullseq_support=defaultdict(lambda:{9:set(),11:set()})
 for r in mapping:
  t=r['table_number']
  if t not in [9,11]:continue
  source_label_support[r['parent_label_raw']][t].add(r['peptide_row_id'])
  if r['source_parent_status']=='exact_sequence_match':verified_label_support[r['parent_label_raw']][t].add(r['peptide_row_id'])
  for acc in parent_sets[r['sequence_id']]:
   all_parent_support[acc][t].add(r['peptide_row_id']);fullseq_support[records[acc]['sequence_sha256']][t].add(r['peptide_row_id'])
 support={};support_rows=[]
 for kind,groups in [('literal_label',source_label_support),('verified_source_label',verified_label_support),('all_reference_identities',all_parent_support),('all_reference_full_sequences',fullseq_support)]:
  shared={k:v for k,v in groups.items() if v[9] and v[11]};multi={k:v for k,v in shared.items() if len(v[9])>=2 and len(v[11])>=2}
  support[kind]={'shared_parent_keys':len(shared),'multiple_ligand_and_digest_keys':len(multi),'distinct_ligand_rows_shared':len(set().union(*(v[9] for v in shared.values()))),'distinct_digest_rows_shared':len(set().union(*(v[11] for v in shared.values()))),'distinct_ligand_rows_multi':len(set().union(*(v[9] for v in multi.values()))),'distinct_digest_rows_multi':len(set().union(*(v[11] for v in multi.values())))}
  for parent,groupset in groups.items():support_rows.append({'definition':kind,'parent_key':parent,'table9_rows':sorted(groupset[9]),'table11_rows':sorted(groupset[11])})
 h.jsonl(RESULTS/'mhc_digest_parent_support.jsonl.gz',support_rows)
 channels=table('channels');claims=table('source_assertions');quantities=table('quantity_measurements');results=table('quantity_results')
 assert len(results)==309126 and len(quantities)==509850
 assert len({r['result_id'] for r in results})==len(results)
 assert {r['peptide_row_id'] for r in results}=={r['peptide_row_id'] for r in rows}
 result_ids={r['result_id'] for r in results}
 assert all(r['result_id'] in result_ids for r in quantities)
 project=json.loads(Path(installed['external/pxd057588/assets/PXD057588/project.json']).read_text())
 context={'organisms_source_project':project['organisms'],'organism_parts_source_project':project['organismParts'],'channel_records':channels,'source_assertions':claims,'source_registry':manifest['sources']['pxd057588']['registry'],'shared_typed_peptide_fields':list(rows[0]),'shared_typed_channel_fields':list(channels[0]),'shared_typed_assertion_predicates':sorted({r['predicate'] for r in claims}),'material_link_precision_counts':dict(Counter(r['material_link_precision'] for r in rows)),'quantity_representations':dict(Counter(r['representation'] for r in quantities)),'notes':['Mouse project species is raw source evidence; no study-to-all-families propagation.','Table9/11 source row and assay-channel scopes remain distinct; parent match does not establish a specimen pair.','Table11 rows are measured digest peptide signals, not independent precursor-protein abundance estimates.','Exact-source reference lookup does not prove unique biological origin or search-reference equivalence.']}
 h.dump(RESULTS/'context_and_quantity_contract.json',context)
 for name,r in manifest['files'].items():
  assert h.sha(PACKAGE/name)==r['sha256']
 assert h.sha(PACKAGE/'manifest.json')==EXPECTED
 h.dump(RESULTS/'summary.json',{'reference_entries':len(ref),'reference_residues':sum(r['length'] for r in ref),'reference_distinct_full_sequences':len({r['sequence_sha256'] for r in ref}),'source_rows':len(rows),'unique_source_sequences':len(seqs),'exact_position_alternatives':len(hits),'strata':strata,'mhc_digest_support':support,'elapsed_seconds':time.monotonic()-tick,'status':'complete_scan_verification_pending','interpretation':'reference-scoped sequence and observed parent support; no specimen pairing, training admission or concentration claim'})
 h.dump(RESULTS/'output_hashes.json',{p.name:h.sha(p) for p in RESULTS.iterdir() if p.is_file() and p.name!='output_hashes.json'})
 print(json.dumps({'reference_entries':len(ref),'position_alternatives':len(hits),'support':support,'elapsed_seconds':time.monotonic()-tick}),flush=True)

if __name__=='__main__':main()
