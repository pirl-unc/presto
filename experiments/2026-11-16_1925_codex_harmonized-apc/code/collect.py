from pathlib import Path
import argparse,hashlib,json
from concurrent.futures import ThreadPoolExecutor
import modal
p=argparse.ArgumentParser();p.add_argument('--prefix',default='2026-11-16_1925_codex_harmonized-apc');p.add_argument('--out',default='results/runs');p.add_argument('--seed',type=int,default=42);p.add_argument('--arms',default='state_absent,harmonized,shuffled');a=p.parse_args()
v=modal.Volume.from_name('presto-checkpoints');out=Path(a.out)
files=['hardware.json','manifest.json','validation-predictions.json','validation-metrics.json','affinity-check.json','affinity-predictions.json','raw/selected.pt','raw/summary.json','raw/history.json','raw/coverage.json','raw/admission.json','raw/resolved-config.json','raw/completed.json']
jobs=[(f'{arm}-s{a.seed}',f) for arm in a.arms.split(',') for f in files]
if 'state_absent' in a.arms.split(','):jobs += [(f'state_absent-s{a.seed}',f'reference-{name}.json') for name in ('predictions','metrics')]
def get(job):
 arm,name=job;p=out/arm/name;p.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
 with p.open('wb') as f:
  for b in v.read_file(f'{a.prefix}/{arm}/{name}'):f.write(b);h.update(b);size+=len(b)
 return {'path':str(p),'bytes':size,'sha256':h.hexdigest()}
with ThreadPoolExecutor(max_workers=6) as executor:receipt=list(executor.map(get,jobs))
(out/('collection-'+a.arms.replace(',','-')+'.json')).write_text(json.dumps(receipt,indent=2));print('Collected',len(receipt),'files')
