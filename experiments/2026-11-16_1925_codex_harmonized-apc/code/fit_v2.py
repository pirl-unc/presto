from pathlib import Path
import json,os,sys,time,shutil
import modal
E=Path(__file__).resolve().parents[1]
GPU=os.environ.get('PRESTO_MODAL_GPU','H100!')
if GPU!='H100!':raise ValueError('Frozen hardware contract requires H100!')
app=modal.App('presto-harmonized-apc-v2-20261116')
volume=modal.Volume.from_name('presto-checkpoints')
image=(modal.Image.debian_slim(python_version='3.12').uv_pip_install('torch==2.14.1','numpy==2.5.4','pyyaml==6.0.3','mhcgnomes==3.64.6','mhcseqs==2.6.13','tqdm==4.70.1','psutil==7.2.2')
 .env({'PYTHONPATH':'/runtime:/bundle','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}))
if modal.is_local():
 image=(image.add_local_dir(E/'reproduce/runtime','/runtime/presto')
  .add_local_file(E/'code/run.py','/bundle/run.py')
  .add_local_dir(E/'results/cohort-v2','/inputs')
  .add_local_file(E/'reproduce/reference-manifest.json','/bundle/reference.json')
  .add_local_file('/private/tmp/presto-hapc-reference.pt','/parent.pt'))
@app.function(image=image,gpu=GPU,cpu=4,memory=32768,timeout=10800,retries=0,volumes={'/volume':volume},single_use_containers=True)
def fit(arm,seed=42):
 import torch
 from run import run
 root=Path('/volume/2026-11-16_1925_codex_harmonized-apc-v2')
 out=root/f'{arm}-s{seed}';out.mkdir(parents=True,exist_ok=True)
 (out/'hardware.json').write_text(json.dumps({'requested_gpu':GPU,'name':torch.cuda.get_device_name(),'memory_bytes':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__},indent=2));volume.commit()
 try:
  result=run(arm,'/inputs','/parent.pt','/bundle/reference.json',out,seed=seed)
  volume.commit();return result
 finally:volume.commit()
@app.local_entrypoint()
def main(arm:str='all',seed:int=42):
 arms=['state_absent','harmonized','shuffled'] if arm=='all' else [arm]
 calls=[fit.spawn(a,seed) for a in arms]
 print(json.dumps({'calls':[c.object_id for c in calls],'arms':arms,'seed':seed}),flush=True)
 for call in calls:print(json.dumps(call.get()),flush=True)
