import hashlib,json,shutil,tempfile
from pathlib import Path
import modal
from datacache import materialize
volume=modal.Volume.from_name('presto-data',create_if_missing=False)
raw=b''.join(volume.read_file('presto-qualification/corrected-human-fit-v1/interrupted-first-attempt/receipt.json'))
receipt=json.loads(raw)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='stopped-fit-recovery-') as temp:
 sources={}
 for name,pin in receipt['files'].items():
  p=Path(temp)/name
  with p.open('wb') as f:
   for chunk in volume.read_file(pin['path'].removeprefix('/data/')):f.write(chunk)
  assert digest(p)==pin['sha256'] and p.stat().st_size==pin['size_bytes']
  sources[name]={'path':str(p),'sha256':pin['sha256']}
 def builder(inputs,outputs):
  for n,p in inputs.items():shutil.copyfile(p,outputs[n])
 paths=materialize('/Users/iskander/.hitlist/presto-corrected-human-fit/interrupted-first-attempt',sources,transform={'version':'stopped-target-policy-fit-recovery-v1','options':{'original_commit':receipt['summary']['invocation']['commit']}},outputs={n:{} for n in sources},builder=builder)
receipt['local_paths']={n:str(p) for n,p in paths.items()}
f=Path('/private/tmp/presto-corrected-human-fit/experiments/2026-10-16_0840_codex_corrected-human-fit/results/interrupted-first-attempt.json');f.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
