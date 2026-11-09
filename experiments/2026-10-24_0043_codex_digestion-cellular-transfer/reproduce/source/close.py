"""Verify completed evidence and record immutable closure hashes."""
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
import numpy as np
import scipy
import torch

f=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
summary=read(f/'results/report-summary.json');suite=read(f/'results/fit-suite.json')
assert suite['status']=='completed'
for r in suite['results']:
    if r.get('status')=='not_admitted':continue
    assert r['returncode']==0
    assert read(f/'results'/r['arm']/'independent-verification.json')['status']=='verified'
for receipt in ['execution-code','fitting','verification']:
    for p,h in read(f/f'reproduce/{receipt}.json')['files'].items():assert sha(f/p)==h,(receipt,p)
config=read(f/'reproduce/launch.json')
for p,h in config['source_hashes'].items():assert sha(Path(p))==h,p
dump(f/'results/environment.json',dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,threads=config['environment']))
for p in (f/'code').glob('*.py'):shutil.copy2(p,f/'reproduce/source'/p.name)
closure=dict(all_pass=True,completed_fit_arms=[r['arm'] for r in suite['results'] if r.get('returncode')==0],sampled_fit_performed=False,optimizer_updates=summary['verified_optimizer_updates'],predictions=summary['verified_prediction_count'],scope='Fixed cellular transfer comparison; no model promotion or concentration qualification')
dump(f/'results/closure.json',closure)
files={str(p.relative_to(f)):sha(p) for p in sorted(f.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.ruff_cache' not in p.parts and p.name!='closure-hashes.json'}
dump(f/'reproduce/closure-hashes.json',dict(files=files))
print(json.dumps(dict(**closure,hashed_files=len(files))))
