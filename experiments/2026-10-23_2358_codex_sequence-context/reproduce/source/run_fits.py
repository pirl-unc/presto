"""Run the prospectively fixed five-arm suite without retries."""
import json
import subprocess
import time
from core import FAMILY,dump,sha

started=time.perf_counter()
receipt=json.loads((FAMILY/'reproduce/fitting.json').read_text())
results=[]
for arm in receipt['arms']:
    for rel,h in receipt['input_hashes'].items():assert sha(FAMILY/rel)==h
    with (FAMILY/f'launch_logs/fit-{arm}.txt').open('x') as stream:
        proc=subprocess.run(['sh',str(FAMILY/'reproduce/fit.sh'),arm],stdout=stream,stderr=subprocess.STDOUT)
    results.append(dict(arm=arm,returncode=proc.returncode))
    dump(FAMILY/'results/fit-receipts.json',dict(arms=results,seconds=time.perf_counter()-started))
    print(json.dumps(results[-1]),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
