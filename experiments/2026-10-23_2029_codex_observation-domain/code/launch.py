"""Fresh bounded CPU workers; refuse to overwrite any existing outcome."""
import json
import subprocess
import sys
import time
from pathlib import Path
from project import dump,sha

FAMILY=Path(__file__).resolve().parents[1]


def main():
    cfg=json.loads((FAMILY/'reproduce/launch.json').read_text())
    frozen=json.loads((FAMILY/'reproduce/frozen-code.json').read_text())
    for rel,expected in frozen.items():
        assert sha(FAMILY/rel)==expected,rel
    cases=json.loads((FAMILY/'reproduce/cases.json').read_text())
    for case in cases:
        assert not (FAMILY/f"results/case-{case['case_id']:03d}.json").exists(),'Existing outcome; do not overwrite'
    started=time.perf_counter()
    receipts=[]
    for case in cases:
        cid=case['case_id']
        tick=time.perf_counter()
        path=FAMILY/f'launch_logs/case-{cid:03d}.log'
        with path.open('w') as out:
            try:
                p=subprocess.run([sys.executable,str(FAMILY/'code/admission.py'),str(cid)],stdout=out,stderr=subprocess.STDOUT,timeout=cfg['runtime_limits']['seconds']+15)
                status=dict(case_id=cid,returncode=p.returncode,seconds=time.perf_counter()-tick)
            except subprocess.TimeoutExpired:
                status=dict(case_id=cid,returncode=None,parent_timeout=True,seconds=time.perf_counter()-tick)
        receipts.append(status)
        dump(FAMILY/'results/worker-receipts.json',dict(seconds=time.perf_counter()-started,workers=receipts))
        result=FAMILY/f'results/case-{cid:03d}.json'
        if result.exists():
            info=json.loads(result.read_text())
            print(json.dumps({k:info.get(k) for k in ['case_id','eta','status','accepted','seconds','max_log_disagreement']}),flush=True)
        else:
            print(json.dumps(status),flush=True)
    print(json.dumps(dict(suite_seconds=time.perf_counter()-started,workers=len(receipts))),flush=True)


if __name__=='__main__':
    main()
