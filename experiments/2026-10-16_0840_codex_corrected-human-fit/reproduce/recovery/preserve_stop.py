from pathlib import Path
import subprocess
code = r'''
from pathlib import Path
import hashlib,json,os,signal,shutil,tarfile,tempfile,time
from datetime import datetime,timezone
import datacache
pid=4
assert Path('/proc/4/cmdline').read_bytes()==b'/usr/local/bin/python\x00/opt/qualification/fit.py\x00'
os.kill(pid,signal.SIGSTOP)
work=next(Path('/tmp').glob('corrected-human-fit-*'))
assert work.is_dir()
reason={"status":"manually_interrupted_before_optimization","reason":"Released canonical affinity targets clamp at 50000 nM, while this prospective experiment requires original log10(nM) targets. Frozen IC50 rows above the cap: train 157 (39 exact), val 31 (8 exact), test 23 (7 exact). Preserve recipe and source artifacts; do not reinterpret this as a completed fit or a timeout.","utc":datetime.now(timezone.utc).isoformat(),"process_status":Path('/proc/4/status').read_text(),"observed_files":sorted(p.name for p in work.iterdir()),"n_checkpoint_files":len(list(work.glob('*.pt'))),"n_prediction_files":len(list(work.glob('*predictions.csv'))),"log_limit":"Python buffered training stdout is not flushed by this manual process termination; preserve the existing bytes without claiming a complete log."}
assert reason['n_checkpoint_files']==0 and reason['n_prediction_files']==0
root=Path('/opt/qualification')
metadata=json.loads((root/'invocation.json').read_text())
def pin(p):
 p=Path(p);return {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
def builder(inputs,outputs):
 with tempfile.TemporaryDirectory(prefix='preserved-fit-') as temp:
  dest=Path(temp)
  for p in work.rglob('*'):
   if p.is_file():
    out=dest/'work'/p.relative_to(work);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
  for name in ('fit.py','baselines.py','diagnostic.py','common.py','launch.json','invocation.json','producer.json','mhc-sources.json','frozen-inputs.json','consumer-install.json'):
   shutil.copyfile(root/name,dest/name)
  (dest/'interruption.json').write_text(json.dumps(reason,indent=2)+'\n')
  inventory={p.relative_to(dest).as_posix():pin(p) for p in sorted(dest.rglob('*')) if p.is_file()}
  Path(outputs['inventory.json']).write_text(json.dumps(inventory,indent=2)+'\n')
  with tarfile.open(outputs['run.tar.gz'],'w:gz') as archive:
   for name in inventory:archive.add(dest/name,arcname=name,recursive=False)
  Path(outputs['summary.json']).write_text(json.dumps({'invocation':metadata,**reason,'files':{name:pin(outputs[name]) for name in ('run.tar.gz','inventory.json')}},indent=2)+'\n')
paths=datacache.materialize('/data/presto-qualification/corrected-human-fit-v1/interrupted-first-attempt',{'invocation':{'path':str(root/'invocation.json'),'sha256':pin(root/'invocation.json')['sha256']}},transform={'version':'preserve-pre-optimization-target-policy-mismatch-v1','options':{'reason':reason['reason'],'original_commit':metadata['commit']}},outputs={n:{} for n in ('run.tar.gz','inventory.json','summary.json')},builder=builder)
receipt={'files':{n:{'path':str(p),**pin(p)} for n,p in paths.items()},'summary':json.loads(Path(paths['summary.json']).read_text())}
Path('/data/presto-qualification/corrected-human-fit-v1/interrupted-first-attempt/receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
time.sleep(1)
os.kill(pid,signal.SIGTERM)
os.kill(pid,signal.SIGCONT)
'''
subprocess.run(['/Users/iskander/code/shared-virtual-env/bin/modal','container','exec','--no-pty','ta-01M51YVCB8ZPDG81J16XGN3HXR','--','python','-u','-c',code],check=True)
