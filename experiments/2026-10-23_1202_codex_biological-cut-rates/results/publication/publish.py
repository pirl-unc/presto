"""Fresh read, compare, patch only body, and exact readback for existing threads."""
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def get(path):
    return json.loads(subprocess.check_output(["gh","api",path],text=True))

for entry in json.loads((ROOT/"updates.json").read_text()):
    path=f"repos/{entry['repository']}/{entry['kind']}/{entry['number']}"
    current=get(path)
    assert current['state']=='open'
    marker=entry['marker']
    assert marker not in (current['body'] or ''), 'Already updated; inspect rather than duplicate'
    prefix=entry['repository'].split('/')[-1]+'-'+str(entry['number'])
    before=ROOT/(prefix+'-before.json');before.write_text(json.dumps(current,indent=2)+'\n')
    body=(current['body'] or '').rstrip()+entry['section']
    payload=ROOT/(prefix+'-request.json');payload.write_text(json.dumps({'body':body},indent=2)+'\n')
    fresh=get(path)
    assert fresh['body']==current['body'] and fresh['updated_at']==current['updated_at'], 'Concurrent edit; rebase update'
    subprocess.run(['gh','api','-X','PATCH',path,'--input',str(payload)],check=True,stdout=subprocess.DEVNULL)
    after=get(path)
    assert after['body']==body and after['state']=='open'
    if entry['kind']=='pulls':assert after['draft'] is True
    (ROOT/(prefix+'-readback.json')).write_text(json.dumps(after,indent=2)+'\n')
    print(prefix, 'exact body/open state verified',flush=True)
