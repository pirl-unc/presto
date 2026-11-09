import importlib.util,json,math,resource,time
from pathlib import Path
import numpy as np,torch
r=Path('/private/tmp/presto-modular-model/experiments/2026-10-24_1007_codex_erap1-length-response')
spec=importlib.util.spec_from_file_location('frozen_trimming',r/'reproduce/training/n_terminal_trimming.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1)
p=dict(np.load(r/'results/path-inputs.npz'));origins=json.loads((r/'results/origins.json').read_text());rng=np.random.default_rng(42);direction=rng.normal(size=23);checks=[]
t={k:torch.as_tensor(v,dtype=torch.float64 if k in ['counts','log_birth'] else None) for k,v in p.items()}
for probe in [np.zeros(23),.2*direction,-.2*direction]:
 for label,scale in [('wt',0.),('ko',-np.logaddexp(0,-probe[-1]))]:
  rates=probe[:20]+scale;survival=-np.logaddexp(0,-rates);hazard=np.logaddexp(0,rates[p['target_residue']]);hazard[p['target_monomer']]=0
  reference=p['log_birth']+p['counts']@survival-hazard[p['owner']]
  state_survival=survival[p['state_residue']];before=np.r_[0,np.cumsum(state_survival)[:-1]]
  bad=p['log_birth']+(before[p['terminal_index']]-before)-hazard[p['owner']]
  local=np.empty(len(reference))
  for o in origins:
   a,b=o['path_start'],o['path_end'];v=state_survival[a:b];prefix=np.r_[0,np.cumsum(v)[:-1]]
   local[a:b]=p['log_birth'][a:b]+prefix[-1]-prefix-hazard[o['query_id']]
  actual=m.log_state_trimming_paths(t['log_birth'],torch.tensor(rates)[t['state_residue']],t['terminal_index'],t['state_monomer']).numpy()
  checks.append({'probe':len(checks)//2,'condition':label,'numpy_global_vs_counts':float(abs(bad-reference).max()),'numpy_local_vs_counts':float(abs(local-reference).max()),'actual_torch_vs_counts':float(abs(actual-reference).max()),'actual_torch_vs_local':float(abs(actual-local).max())})
out={'checks':checks,'scope':'Separate producer error from independent NumPy global-sum error after the gate failure. Frozen original executed producer.'}
(r/'results/rounding-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
