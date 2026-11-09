"""Verify the actual shared source generation without changing its files."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

FAMILY=Path(__file__).resolve().parents[1]
ROOT=FAMILY.parents[1]
PACKAGE=Path('/private/tmp/hitlist-v2-external-18222-audited')
EXPECTED='daff7930e193c6dbde1fe84c065b024187232147fa006d9348e297766e95e69c'
HITLIST=Path('/private/tmp/hitlist-external-quant-v2')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 start=time.monotonic()
 assert not (FAMILY/'results/package-verification.json').exists()
 assert sha(PACKAGE/'manifest.json')==EXPECTED
 manifest=json.loads((PACKAGE/'manifest.json').read_text())
 actual={str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*') if p.is_file() and p!=PACKAGE/'manifest.json'}
 assert actual==set(manifest['files'])
 for name,record in manifest['files'].items():
  p=PACKAGE/name;assert p.stat().st_size==record['size_bytes'] and sha(p)==record['sha256'],name
 first_pass=time.monotonic()-start
 print('all',len(actual),'package file hashes verified in',first_pass,flush=True)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=HITLIST,text=True).strip()=='bac9c7f104d475c7b2f703681f7667102924c1e1'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=HITLIST,text=True).strip()
 env=dict(os.environ);env['PYTHONPATH']=str(HITLIST)
 command=[sys.executable,'-c',"from pathlib import Path; import hitlist; from hitlist.external_evidence import verify_external_evidence_package; assert Path(hitlist.__file__).resolve().parent == Path('/private/tmp/hitlist-external-quant-v2/hitlist'); m=verify_external_evidence_package('/private/tmp/hitlist-v2-external-18222-audited'); print(m['schema_version'], m['catalog_schema_version'],m['source_ids'])"]
 with (FAMILY/'launch_logs/native-package.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert sha(PACKAGE/'manifest.json')==EXPECTED
 receipt={'passed':True,'manifest_sha256':EXPECTED,'package_path':str(PACKAGE),'files':len(actual),'manifest_bytes':sum(x['size_bytes'] for x in manifest['files'].values()),'first_hash_pass_seconds':first_pass,'total_seconds':time.monotonic()-start,'native_command':command,'native_environment_overrides':{'PYTHONPATH':str(HITLIST)},'native_checkout':str(HITLIST),'native_git_commit':'bac9c7f104d475c7b2f703681f7667102924c1e1','native_git_dirty':False,'python':sys.version,'pyarrow':importlib.metadata.version('pyarrow'),'script_sha256':sha(__file__),'presto_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'presto_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)}
 (FAMILY/'results/package-verification.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 shutil.copy2(PACKAGE/'manifest.json',FAMILY/'reproduce/shared-package-manifest.json')
 shutil.copy2(__file__,FAMILY/'reproduce/source/verify_package.py')
 print(json.dumps({'passed':True,'files':len(actual),'seconds':receipt['total_seconds']}),flush=True)

if __name__=='__main__':main()
