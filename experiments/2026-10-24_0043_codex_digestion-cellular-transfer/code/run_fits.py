"""Sequential frozen exact/hybrid matrix; no failed-arm retries."""
import gzip
import json
import subprocess
import time
from core import FAMILY,cfg,dump,evaluate,load,sha


def main():
    start=time.perf_counter();c=cfg();q=json.loads((FAMILY/'results/qualification.json').read_text())
    assert not (FAMILY/'results/fit-suite.json').exists()
    for p,h in json.loads((FAMILY/'reproduce/fitting.json').read_text())['files'].items():assert sha(FAMILY/p)==h
    rows,metrics=evaluate(0.,load('uniform_rates'),uniform=True)
    out=FAMILY/'results/uniform_composition';out.mkdir()
    dump(out/'evaluation.json',metrics)
    with gzip.open(out/'predictions.jsonl.gz','wt') as h:
        for row in rows:h.write(json.dumps(row,allow_nan=False)+'\n')
    results=[]
    for arm in c['exact_arms']+[c['hybrid_arm']]:
        cap=q['fit_caps'][arm]
        if not cap['admitted']:
            results.append(dict(arm=arm,status='not_admitted',cap=cap));continue
        now=time.perf_counter()
        with (FAMILY/f'launch_logs/{arm}.txt').open('w') as h:
            child=subprocess.Popen([c['python'],str(FAMILY/'code/fit.py'),arm],stdout=h,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=cap['seconds']+60)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:code=child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();code=child.wait()
        results.append(dict(arm=arm,returncode=code,seconds=time.perf_counter()-now))
        dump(FAMILY/'results/fit-suite.json',dict(status='running',results=results,seconds=time.perf_counter()-start))
        print(results[-1],flush=True)
    dump(FAMILY/'results/fit-suite.json',dict(status='completed',results=results,seconds=time.perf_counter()-start))

if __name__=='__main__':main()
