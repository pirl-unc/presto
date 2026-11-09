"""Independent source-cell, mouse-reference and shared-parent replay."""
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

from Bio import SeqIO
import openpyxl
from openpyxl.utils.cell import coordinate_to_tuple
import pyarrow.parquet as pq

FAMILY=Path(__file__).resolve().parents[1];ROOT=FAMILY.parents[1];R=FAMILY/'results'
PREVIOUS=ROOT/'experiments/2026-10-22_1845_codex_mapp-parent-reference'

def load_helper():
 p=PREVIOUS/'reproduce/source/verify.py';expected=json.loads((PREVIOUS/'results/independent-verification.json').read_text())['verifier_sha256'];assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 spec=importlib.util.spec_from_file_location('frozen_parent_verifier',p);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);shutil.copy2(p,FAMILY/'reproduce/source/mapp_verify.py');return h

def main():
 tick=time.monotonic();h=load_helper();assert not (R/'independent-verification.json').exists()
 paths=json.loads((R/'input_paths.json').read_text());assets=json.loads((FAMILY/'reproduce/input_assets.json').read_text())['assets']
 for name,p in paths.items():assert h.sha(p)==assets[name]['sha256'] and Path(p).stat().st_size==assets[name]['size']
 hashes=json.loads((R/'output_hashes.json').read_text())
 for name,value in hashes.items():assert h.sha(R/name)==value
 def table(name):return pq.read_table(paths['external/pxd057588/typed/'+name+'.parquet']).to_pylist()
 rows=table('peptide_rows');by_id={r['peptide_row_id']:r for r in rows}
 assert len(by_id)==len(rows)==35772
 books={}
 for r in rows:
  asset=r['asset_path']
  if asset not in books:
   workbook=openpyxl.load_workbook(paths['external/pxd057588/assets/'+asset],read_only=True,data_only=True);assert len(workbook.worksheets)==1;books[asset]=list(workbook.worksheets[0].values);workbook.close()
  for key,expected in [('sequence_cell',r['sequence_raw']),('parent_label_cell',r['parent_label_raw'])]:
   row,col=coordinate_to_tuple(r[key]);assert books[asset][row-1][col-1]==expected
 seq_rows=h.readcsv(R/'sequences.csv.gz');peptides={r['sequence_id']:r['sequence'] for r in seq_rows};assert set(peptides.values())=={r['sequence_raw'] for r in rows} and len(peptides)==30587
 seq_ids={v:k for k,v in peptides.items()}
 reference={}
 for r in SeqIO.parse(paths['mouse-reviewed-2026_03-isoforms.fasta'],'fasta'):
  acc=r.id.split('|')[1];assert acc not in reference;reference[acc]={'sequence':str(r.seq),'header':r.description,'entry':r.id.split('|')[2]}
 inventory=h.readjsonl(R/'reference_entries.jsonl.gz');assert len(inventory)==len(reference) and {r['accession'] for r in inventory}==set(reference)
 for r in inventory:
  expected=reference[r['accession']];assert r['header']==expected['header'] and r['entry']==expected['entry'] and r['length']==len(expected['sequence']) and r['sequence_sha256']==hashlib.sha256(expected['sequence'].encode()).hexdigest()
 sequences={a:r['sequence'] for a,r in reference.items()};hits=h.scan(sequences,peptides)
 emitted=h.readcsv(R/'matches.csv.gz');observed={(r['sequence_id'],r['accession'],int(r['start1'])) for r in emitted};assert observed==hits and len(observed)==len(emitted)
 per_peptide=defaultdict(lambda:defaultdict(list))
 for pid,acc,start in sorted(hits):per_peptide[pid][acc].append(start)
 for r in emitted:
  seq=sequences[r['accession']];s,e=int(r['start1']),int(r['end1']);assert seq[s-1:e]==peptides[r['sequence_id']] and e-s+1==len(peptides[r['sequence_id']]);assert r['left']==(seq[s-2] if s>1 else '-') and r['right']==(seq[e] if e<len(seq) else '-');assert r['parent_sequence_sha256']==hashlib.sha256(seq.encode()).hexdigest()
 components=h.graph_components(hits)
 for r in seq_rows:
  pid=r['sequence_id'];expected={'sequence_id':pid,'sequence':peptides[pid],'length':len(peptides[pid]),'reference_identity_count':len(per_peptide[pid]),'reference_full_sequence_count':len({sequences[a] for a in per_peptide[pid]}),'reference_occurrence_count':sum(len(p) for p in per_peptide[pid].values()),'component':components.get(pid,'')};assert r=={k:str(v) for k,v in expected.items()}
 parents=h.readjsonl(R/'matched_parents.jsonl.gz');assert {r['accession'] for r in parents}=={a for _,a,_ in hits}
 for r in parents:assert r['sequence']==sequences[r['accession']]
 mappings=h.readjsonl(R/'source_mapping.jsonl.gz');assert len(mappings)==len(rows) and {r['peptide_row_id'] for r in mappings}==set(by_id)
 for r in mappings:
  raw=by_id[r['peptide_row_id']];assert all(r[k]==v for k,v in raw.items());pid=seq_ids[raw['sequence_raw']];acc=raw['parent_label_raw'];positions=h.direct_positions(sequences[acc],peptides[pid]) if acc in sequences else []
  expected={**raw,'sequence_id':pid,'source_parent_status':'missing_id' if acc not in sequences else 'exact_sequence_match' if positions else 'resolved_id_peptide_absent','source_parent_positions_start1':positions,'reference_identity_count':len(per_peptide[pid]),'reference_full_sequence_count':len({sequences[a] for a in per_peptide[pid]}),'reference_occurrence_count':sum(len(p) for p in per_peptide[pid].values()),'additional_parent_identities':len(set(per_peptide[pid])-{acc}),'component':components.get(pid,'')};assert r==expected
 summary=json.loads((R/'summary.json').read_text());assert summary['reference_entries']==len(reference) and summary['reference_residues']==sum(map(len,sequences.values())) and summary['reference_distinct_full_sequences']==len(set(sequences.values())) and summary['exact_position_alternatives']==len(hits)
 for stat in summary['strata']:
  selected=[r for r in mappings if stat['table']=='all' or r['table_number']==stat['table']];ids={r['sequence_id'] for r in selected};ps={a for pid in ids for a in per_peptide[pid]};cs=Counter(components[pid] for pid in ids if pid in components)
  expected={'table':stat['table'],'source_rows':len(selected),'distinct_sequences':len(ids),'exact_reference_matched_rows':sum(bool(per_peptide[r['sequence_id']]) for r in selected),'source_parent_status':dict(Counter(r['source_parent_status'] for r in selected)),'single_parent_identity_rows':sum(r['reference_identity_count']==1 for r in selected),'single_full_parent_sequence_rows':sum(r['reference_full_sequence_count']==1 for r in selected),'multiple_full_parent_sequence_rows':sum(r['reference_full_sequence_count']>1 for r in selected),'additional_parent_rows':sum(r['additional_parent_identities']>0 for r in selected),'matched_parent_identities':len(ps),'matched_full_parent_sequences':len({sequences[a] for a in ps}),'components':len(cs),'largest_component_sequences':max(cs.values(),default=0)};assert stat==expected
 groups=defaultdict(lambda:{9:set(),11:set()})
 for r in rows:
  t=r['table_number']
  if t not in [9,11]:continue
  pid=seq_ids[r['sequence_raw']];acc=r['parent_label_raw'];rowid=r['peptide_row_id'];groups['literal_label',acc][t].add(rowid)
  if acc in per_peptide[pid]:groups['verified_source_label',acc][t].add(rowid)
  for a in per_peptide[pid]:
   groups['all_reference_identities',a][t].add(rowid);groups['all_reference_full_sequences',hashlib.sha256(sequences[a].encode()).hexdigest()][t].add(rowid)
 support_rows=h.readjsonl(R/'mhc_digest_parent_support.jsonl.gz');assert len(support_rows)==len(groups)
 for r in support_rows:assert r['table9_rows']==sorted(groups[r['definition'],r['parent_key']][9]) and r['table11_rows']==sorted(groups[r['definition'],r['parent_key']][11])
 for kind,stat in summary['mhc_digest_support'].items():
  shared=[v for (k,_),v in groups.items() if k==kind and v[9] and v[11]];multi=[v for v in shared if len(v[9])>=2 and len(v[11])>=2]
  expected={'shared_parent_keys':len(shared),'multiple_ligand_and_digest_keys':len(multi),'distinct_ligand_rows_shared':len(set().union(*(v[9] for v in shared))),'distinct_digest_rows_shared':len(set().union(*(v[11] for v in shared))),'distinct_ligand_rows_multi':len(set().union(*(v[9] for v in multi))),'distinct_digest_rows_multi':len(set().union(*(v[11] for v in multi)))};assert stat==expected
 shutil.copy2(__file__,FAMILY/'reproduce/source/verify.py')
 receipt={'passed':True,'input_hashes':len(paths),'output_hashes':len(hashes),'original_source_cells':2*len(rows),'source_mapping_rows':len(mappings),'reference_entries':len(reference),'complete_scan_positions':len(hits),'sequence_rows':len(seq_rows),'parent_support_rows':len(support_rows),'strata':len(summary['strata']),'source_negative_candidates':sum(r['source_parent_status']=='resolved_id_peptide_absent' for r in mappings),'elapsed_seconds':time.monotonic()-tick,'code_sha256':h.sha(__file__),'helper_sha256':h.sha(PREVIOUS/'reproduce/source/verify.py'),'scope':'independent BioPython + complete Aho-Corasick scan, original workbook source cells, direct source-parent lookahead, all support strata and relations'}
 (R/'independent-verification.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
