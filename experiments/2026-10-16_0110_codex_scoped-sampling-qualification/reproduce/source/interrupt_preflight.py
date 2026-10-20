from pathlib import Path
import hashlib,json,os,signal,shutil,tempfile,time,tarfile
from datetime import datetime,timezone
from datacache import materialize
root=next(Path('/tmp').glob('scoped-sampling-*'))
config=json.loads((root/'preflight/config.json').read_text())
assert config['data_preflight_only'] is True
assert not any(root.rglob('checkpoint.pt'))
command=Path('/proc/4/cmdline').read_bytes().split(b'\0')
assert b'/opt/qualification/fit.py' in command
actual=json.loads(Path('/opt/qualification/invocation.json').read_text())
def pin(path):
 path=Path(path)
 return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size_bytes':path.stat().st_size}
with tempfile.TemporaryDirectory(prefix='stop-evidence-') as temporary:
 work=Path(temporary)
 files={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}
 files.update({p.name:p for p in Path('/opt/qualification').iterdir() if p.name in ('fit.py','baselines.py','launch.json','invocation.json','supported_outputs.json','consumer-install.json')})
 for name,path in files.items():
  target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
 record={'status':'manual_stop_requested_before_fitting','reason':'Shared-volume occurrence SQLite caused slow random I/O; #102 moves only disposable scratch to local storage. No condition fit began.','recorded_utc':datetime.now(timezone.utc).isoformat(),'invocation':actual,'original_process_command':[x.decode() for x in command if x],'original_preflight_config':config,'partial_log_note':'The running Python process buffered stdout; the zero-byte log is preserved as observed.','sqlite_not_archived':'Incomplete disposable collapse database is not a reusable dataset; original frozen public source remains available.'}
 (work/'interruption.json').write_text(json.dumps(record,indent=2)+'\n')
 inventory={p.relative_to(work).as_posix():pin(p) for p in work.rglob('*') if p.is_file()}
 (work/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 archive=Path(temporary)/'partial-run.tar.gz'
 with tarfile.open(archive,'w:gz') as output:
  for name in sorted(inventory):output.add(work/name,arcname=name,recursive=False)
 sources={name:{'path':str(work/name),'sha256':pin(work/name)['sha256']} for name in ('partial-run.tar.gz','inventory.json','interruption.json')}
 def builder(inputs,outputs):
  for name,path in inputs.items():shutil.copyfile(path,outputs[name])
 paths=materialize('/data/presto-qualification/scoped-sampling-v1/interrupted-shared-scratch',sources,transform={'version':'pre-fit-interruption-evidence-v1'},outputs={name:{} for name in sources},builder=builder)
 print(json.dumps({'files':{name:{'path':str(path),**pin(path)} for name,path in paths.items()}},indent=2),flush=True)
 time.sleep(1)
 os.kill(4,signal.SIGTERM)
 print('Sent SIGTERM to the verified preflight process after preserving evidence',flush=True)
 time.sleep(1)
