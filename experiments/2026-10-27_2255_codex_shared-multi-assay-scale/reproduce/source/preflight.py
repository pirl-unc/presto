"""Admit each frozen manifest; one declared training-only software update."""
from pathlib import Path
import copy,json,time
import torch
from presto.data.modular_sources import ObservationIndex,file_sha256
from presto.training.modular_run import preflight
from presto.training.modular import train_step,coverage_summary,save_checkpoint
from presto.models.presto import Presto
F=Path(__file__).resolve().parents[1];ROOT=F.parents[1];UPLOAD=ROOT/'artifacts'/F.name/'upload'
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);started=time.monotonic();cache={};reports={}
 for variant in ('ic50-exact','ic50-bounds','multi-assay','multi-assay-wide'):
  path=UPLOAD/'inputs'/f'{variant}-s42.json';document=json.loads(path.read_text());key=tuple((s['path'],s['sha256']) for s in document['sources'])
  if key not in cache:cache[key]=ObservationIndex(path)
  index=copy.copy(cache[key]);index.manifest=document;index.manifest_path=path;index.manifest_sha256=file_sha256(path)
  config,training,scales,admission=preflight(index)
  torch.manual_seed(42);model=Presto(config);reports[variant]=dict(parameters=sum(p.numel() for p in model.parameters()),admission=admission,scales={k:vars(v) for k,v in scales.items()},training=training,partitions={k:len(v) for k,v in index.partitions.items()})
  print(variant,reports[variant]['parameters'],reports[variant]['partitions'],flush=True)
  if variant=='multi-assay-wide':
   strata={}
   for i in index.partitions['train:assay']:
    row=index[i];k=(row.endpoint,row.mhc[0].mhc_class)
    if len(strata.get(k,[]))<2:strata.setdefault(k,[]).append(row)
   rows=[r for values in strata.values() for r in values]
   step=train_step(model,torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4),rows,scales)
   coverage=coverage_summary([step]);assert not coverage['untrained_parameters']
   probe=dict(n=len(rows),split='train',updates=1,observation_ids=[r.observation_id for r in rows],loss=step['loss'],counts=step['counts'],coverage=coverage,interpretation='software smoke check; no validation/test inference and no predictive-quality claim')
   destination=ROOT/'artifacts'/F.name/'software-preflight';destination.mkdir(parents=True,exist_ok=True);save_checkpoint(destination/'probe.pt',model,scales,metadata=probe)
   (F/'results/software-probe.json').write_text(json.dumps(probe,indent=2)+'\n')
 (F/'results/preflight.json').write_text(json.dumps(dict(status='passed',conditions=reports,seconds=time.monotonic()-started),indent=2)+'\n')
if __name__=='__main__':main()
