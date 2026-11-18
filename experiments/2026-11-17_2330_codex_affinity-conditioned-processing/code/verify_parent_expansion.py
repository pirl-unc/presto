"""Actual preferred-checkpoint transfer preserves both-class CPU predictions."""
from pathlib import Path
import json
import hashlib
import torch
from presto.data.modular import Observation
from presto.models.modular import ModularPresto,ModularConfig
from presto.training.modular import load_checkpoint,initialize_from_checkpoint
E=Path(__file__).resolve().parents[1];R=E.parents[1]
B=R/'experiments/2026-11-17_1155_codex_shared-presentation-expansion'
p=B/'results/fits/expanded-state-sqrt-s42-n6000/raw/selected.pt'
expected='1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
old,scales,payload=load_checkpoint(p,device='cpu');old.eval()
new=ModularPresto(ModularConfig.from_dict({**old.config.to_dict(),'elution_junction_processing':True})).eval()
transfer=initialize_from_checkpoint(new,p,allow_junction_expansion=True)
rows=[Observation.from_dict(json.loads(s)) for s in (B/'results/example-queries.jsonl').read_text().splitlines()]
with torch.no_grad():
 a,b=old(rows),new(rows)
deltas=[abs(float(x.value-y.value)) for x,y in zip(a,b,strict=True)]
assert max(deltas)==0
assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
report={'status':'passed','parent_sha256':expected,'rows':len(rows),'classes':[r.mhc[0].mhc_class for r in rows],
        'max_logit_delta':max(deltas),'all_existing_parameters_bitwise_equal':True,
        'new_parameters':sum(v.numel() for k,v in new.named_parameters() if k.startswith('elution_junction.')),
        'parent_parameters':sum(v.numel() for v in old.parameters()),'torch':torch.__version__,
        'source_context':new.config.source_context,'state_width':new.config.state_bottleneck,
        'new_head_untrained':True,'test_accessed':False}
(E/'results/parent-expansion-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
