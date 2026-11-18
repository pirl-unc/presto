from pathlib import Path
from collections import defaultdict,Counter
from dataclasses import replace
import json,gzip,hashlib,sys
import torch
from presto.data.modular import Observation
from presto.training.modular import load_checkpoint
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E/'code'));from prepare import Similarity
EMBARGO=E/'results/protected-sequences.txt.gz'
D=E/'results/cohort-v2';rows={split:[Observation.from_dict(json.loads(x)) for x in (D/'data'/f'{split}.jsonl').open()] for split in ('train','validation')}
train,val=rows['train'],rows['validation'];vseq={r.peptide for r in val};held=Similarity(vseq)
assert not {r.group_id for r in train}&{r.group_id for r in val}
assert not any(held.reason(r.peptide) for r in train)
with gzip.open(EMBARGO,'rt') as f:protected=Similarity(x.strip() for x in f)
assert not any(protected.reason(p) for p in {r.peptide for r in train+val})
assert not any(r.provenance['canonical_parent_exposure'] for r in val)
pairs=defaultdict(list)
for r in train+val:pairs[r.provenance['pair_id']].append(r)
for pair in pairs.values():
 assert len(pair)==2 and sorted(r.value for r in pair)==[0.,1.]
 a,b=pair
 assert a.group_id==b.group_id and a.split==b.split and a.mhc==b.mhc and len(a.peptide)==len(b.peptide)
 for key in ('components','cell_type','cytokines','interventions'):assert getattr(a.context,key)==getattr(b.context,key)
model,_,_=load_checkpoint(E/'results/runs-v2/harmonized-s42/raw/selected.pt')
parent,_,_=load_checkpoint(E/'results/parent-reference.pt')
for n in model.frozen_parameters:assert torch.equal(model.get_parameter(n),parent.get_parameter(n)),n
probe=[]
for cls in ('I','II'):
 probe.extend([r for r in val if r.mhc[0].mhc_class==cls][:4])
model.eval();qualified=model.predict(probe);ordinary=model(probe)
for a,b in zip(qualified,ordinary,strict=True):torch.testing.assert_close(a.value,b.value)
report={'status':'passed','rows':{k:len(v) for k,v in rows.items()},'pairs':len(pairs),'protected_sequences':len(protected.peptides),'cross_partition_exact_core8_hamming1_matches':0,'validation_canonical_presentation_parent_exposures':0,'matched_pair_checks':'labels, source protein, split, length, MHC candidate set, APC context','frozen_affinity_parameters_identical':len(model.frozen_parameters),'qualified_prediction_api_rows':len(probe),'test_outcomes_accessed':False}
(E/'results/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
