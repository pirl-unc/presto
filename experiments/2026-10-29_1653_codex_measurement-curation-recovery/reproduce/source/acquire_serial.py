"""Resume a public-source batch serially, honoring server backoff and stopping on persistent limits."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
import datacache
import requests

F=Path(__file__).resolve().parents[1]
ROOT=Path('/Users/iskander/code/presto/artifacts')/F.name/'sources'

def sha(path):
    with path.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()

def main():
    result_path=F/'results/acquisition-strazar-resume.json'
    assert not result_path.exists()
    source=F/'reproduce/acquisition-strazar-manifest.json'
    prior=json.loads((F/'results/acquisition-strazar.json').read_text())
    completed={r['id'] for r in prior if r['status']=='downloaded'}
    sources=[s for s in json.loads(source.read_text())['sources'] if s['id'] not in completed]
    (F/'reproduce/acquisition-strazar-resume-launch.json').write_text(json.dumps(dict(argv=['python',__file__],source_sha256=sha(Path(__file__)),manifest_sha256=sha(source),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),datacache_version=importlib.metadata.version('datacache'),workers=1,minimum_interval_seconds=5,rate_limit_attempts=3,started_unix=time.time()),indent=2)+'\n')
    shutil.copyfile(__file__,F/'reproduce/source/acquire_serial.py')
    results=[]
    for spec in sources:
        path=ROOT/spec['id']/spec['filename'];path.parent.mkdir(parents=True,exist_ok=True)
        result=None
        for attempt in range(3):
            time.sleep(5)
            try:
                datacache.fetch_file(spec['url'],destination=str(path),raw=True,record_provenance=True,timeout=(30,90),max_retries=0)
                if path.stat().st_size!=spec['expected_bytes']:raise ValueError('Source size mismatch')
                result=dict(spec,status='downloaded',path=str(path),bytes=path.stat().st_size,sha256=sha(path));break
            except requests.HTTPError as error:
                if error.response.status_code==429:
                    retry=error.response.headers.get('Retry-After','60')
                    delay=max(60,float(retry)) if retry.isdigit() else 120
                    print(json.dumps(dict(id=spec['id'],status='rate_limited',attempt=attempt+1,backoff_seconds=delay)),flush=True)
                    result=dict(spec,status='rate_limited',retry_after=retry,error=str(error))
                    if attempt<2:time.sleep(delay)
                else:result=dict(spec,status='unavailable',error=str(error));break
            except Exception as error:result=dict(spec,status='unavailable',error=repr(error));break
        results.append(result);result_path.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k in ('id','status','bytes','error')}),flush=True)
        if result['status']=='rate_limited':break
    print(json.dumps(dict(completed=len(completed)+sum(r['status']=='downloaded' for r in results),pending=len(sources)-len(results))),flush=True)
if __name__=='__main__':main()
