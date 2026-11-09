"""Run the frozen arms sequentially; retain both terminal receipts."""
import json
import subprocess
import sys
import time
from core import FAMILY,dump,sha

cfg=json.loads((FAMILY/'reproduce/fitting.json').read_text())
assert not (FAMILY/'results/fit-receipts.json').exists()
for rel,expected in json.loads((FAMILY/'reproduce/fitting-code.json').read_text()).items():assert sha(FAMILY/rel)==expected,rel
receipts=[];start=time.perf_counter()
for arm in cfg['arms']:
    with (FAMILY/f'launch_logs/fit-{arm}.txt').open('w') as out:
        try:
            result=subprocess.run([sys.executable,str(FAMILY/'code/fit.py'),arm],stdout=out,stderr=subprocess.STDOUT,timeout=cfg['runtime_limits']['seconds_per_arm']+60)
            entry=dict(arm=arm,returncode=result.returncode)
        except subprocess.TimeoutExpired:
            entry=dict(arm=arm,returncode=None,parent_timeout=True)
    receipts.append(entry);dump(FAMILY/'results/fit-receipts.json',dict(arms=receipts,seconds=time.perf_counter()-start))
    print(json.dumps(entry),flush=True)
    if entry['returncode']!=0:break
