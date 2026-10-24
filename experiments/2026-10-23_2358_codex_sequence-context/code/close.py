"""Freeze final artifacts only after every predeclared arm verifies."""
import hashlib
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import torch

FAMILY=Path(__file__).resolve().parents[1]
for name in ['base','rate_context','response_termini','both','both_hybrid']:
    s=json.loads((FAMILY/f'results/{name}/status.json').read_text());v=json.loads((FAMILY/f'results/{name}/independent-verification.json').read_text())
    assert s['status']=='completed' and s['updates']==2000 and v['all_pass'] and v['n_predictions_verified']==105452
for receipt,key in [('execution-code.json',None),('verification.json','source_hashes'),('fitting.json','input_hashes')]:
    r=json.loads((FAMILY/'reproduce'/receipt).read_text());r=r[key] if key else r
    for p,h in r.items():assert hashlib.sha256((FAMILY/p).read_bytes()).hexdigest()==h,p
archive=json.loads((FAMILY/'results/raw-artifacts.json').read_text())
for p,r in archive['files'].items():assert hashlib.sha256((FAMILY/p).read_bytes()).hexdigest()==r['sha256']
for r in archive['archives']:assert hashlib.sha256((FAMILY/r['path']).read_bytes()).hexdigest()==r['sha256']
for p in (FAMILY/'code').glob('*.py'):
    target=FAMILY/'reproduce/source'/p.name
    if target.exists():assert target.read_bytes()==p.read_bytes(),p
    else:shutil.copyfile(p,target)
closure=dict(all_pass=True,updates=10000,predictions=5*105452,raw_files=len(archive['files']),source_commit='010be23818400d47a55d6ce0f386b95c222ce8ec',prior_ci=dict(run=40056708162,conclusion='success'),environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,machine=platform.machine()),scope='Five fixed biological fits complete; development evidence only. No calibrated APC concentration model.')
(FAMILY/'results/closure.json').write_text(json.dumps(closure,indent=2)+'\n')
files={str(p.relative_to(FAMILY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FAMILY.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.ruff_cache' not in p.parts and p.suffix not in ['.npz','.pt','.pyc'] and p.name!='closure-hashes.json'}
(FAMILY/'reproduce/closure-hashes.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print(json.dumps(dict(closure=closure,closure_files=len(files))))
