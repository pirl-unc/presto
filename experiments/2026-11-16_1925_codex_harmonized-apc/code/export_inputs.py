from pathlib import Path
import json,time,hashlib
import pyarrow.parquet as pq
from hitlist.export import generate_training_table
E=Path(__file__).resolve().parents[1]
POOL=Path('/Users/iskander/code/presto-apm/experiments/2026-11-15_0900_claude_state-bottleneck-v2/data/pool/pairs.parquet')
PMIDS={40113210,33592498,32161166,35561310,32488085,36215666,31844290,32938616,33968037}
pool=pq.read_table(POOL).to_pandas();pool=pool[pool.pmid.isin(PMIDS)]
cols=['peptide','pmid','mhc_class','mhc_species','source','sample_label','sample_attribution','mhc_restriction','mhc_allele_set','mhc_allele_provenance','restriction_evidence','cell_line_name','cell_type','is_ms_observation','assay_modality','ms_outcome_status','has_ptm','peptide_modifications','is_non_peptide_ligand','is_potential_contaminant','condition_id','condition_status','condition_evidence','condition_reference','condition_control','condition_control_for','condition_background','condition_combination','condition_knockout_genes','condition_knockdown_genes','condition_overexpression_genes','condition_genetic_variants','condition_transfection','condition_transduction','condition_cytokines','condition_drugs','condition_stimulation']
t=time.time();print('export start',len(set(pool.peptide)),flush=True)
d=generate_training_table(include_evidence='ms',peptide=sorted(set(pool.peptide)),columns=cols)
d=d[d.pmid.isin(PMIDS)];d.to_parquet(E/'results/source-inputs.parquet',index=False)
print('saved',len(d),'seconds',time.time()-t,flush=True)
(E/'results/export-receipt.json').write_text(json.dumps({'rows':len(d),'seconds':time.time()-t,'columns':list(d),'pmids':sorted(PMIDS),'cache':'/Users/iskander/.hitlist/observations.parquet','hitlist_commit':'43e226621434a483141764ed54faba725bbb3869'},indent=2))
