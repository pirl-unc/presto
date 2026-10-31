"""Use the repository's documented anonymous FTP route for public bulk PSM files."""
from pathlib import Path
import ftplib
import hashlib
import json
import shutil
import subprocess
import time
from urllib.parse import parse_qs,urlparse
import modal

FID='2026-10-29_1653_codex_measurement-curation-recovery'
app=modal.App('presto-public-psm-acquisition-20261029')
volume=modal.Volume.from_name('presto-checkpoints')
image=modal.Image.debian_slim(python_version='3.12')
if modal.is_local():
    F=Path(__file__).resolve().parents[1]
    image=image.add_local_file(F/'reproduce/acquisition-strazar-manifest.json','/bundle/manifest.json')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


@app.function(image=image,cpu=1,memory=2048,timeout=3600,max_containers=1,volumes={'/volume':volume})
def run():
    root=Path('/volume')/FID/'massive-ftp';root.mkdir(parents=True,exist_ok=True)
    receipt=root/'receipt.json';assert not receipt.exists()
    sources=json.loads(Path('/bundle/manifest.json').read_text())['sources']
    results=[];started=time.monotonic()
    try:
        with ftplib.FTP('massive.ucsd.edu',timeout=30) as ftp:
            ftp.login();ftp.voidcmd('TYPE I')
            for spec in sources:
                remote=parse_qs(urlparse(spec['url']).query)['file'][0]
                assert remote.startswith('f.MSV000089225/search/PSMreports/Expi293_monoallelic_HLA2_psmReports/')
                remote='/'+remote[2:]
                path=root/spec['filename'];part=path.with_suffix('.partial')
                with part.open('wb') as stream:ftp.retrbinary('RETR '+remote,stream.write,blocksize=1<<20)
                assert part.stat().st_size==spec['expected_bytes']
                part.replace(path)
                result=dict(spec,status='downloaded',transport='official_anonymous_ftp',ftp_path=remote,path=str(path),sha256=sha(path),bytes=path.stat().st_size)
                results.append(result);receipt.write_text(json.dumps(results,indent=2)+'\n')
                if len(results)%10==0:volume.commit();print('Acquired public PSM files',len(results),flush=True)
    except Exception as error:
        results.append(dict(status='acquisition_stopped',error=repr(error),files_acquired=len(results)))
        receipt.write_text(json.dumps(results,indent=2)+'\n')
    volume.commit()
    return dict(results=results,seconds=time.monotonic()-started,requested_gpu=None,cpu=1,memory_mib=2048)


@app.local_entrypoint()
def main():
    f=Path(__file__).resolve().parents[1];callpath=f/'reproduce/massive-ftp-call.json';assert not callpath.exists()
    shutil.copyfile(__file__,f/'reproduce/source/massive_ftp.py')
    (f/'reproduce/massive-ftp-launch.json').write_text(json.dumps(dict(argv=['python','-m','modal','run','--detach',__file__],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(Path(__file__)),manifest_sha256=sha(f/'reproduce/acquisition-strazar-manifest.json'),requested_gpu=None,started_unix=time.time()),indent=2)+'\n')
    call=run.spawn();callpath.write_text(json.dumps(dict(call_id=call.object_id))+'\n')
    result=call.get();(f/'results/acquisition-strazar-ftp.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='results'}))
