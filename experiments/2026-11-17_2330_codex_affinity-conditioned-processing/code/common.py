"""Pinned paths and hardware for the registered processing follow-up."""
from pathlib import Path
import hashlib
import json
import modal

NAME='2026-11-17_2330_codex_affinity-conditioned-processing'
PRIOR='2026-11-16_2035_codex_joint-apc-capacity'
BROAD='2026-11-17_1155_codex_shared-presentation-expansion'
E=Path(__file__).resolve().parents[1] if modal.is_local() else Path('/experiment')
ROOT=E.parents[1] if modal.is_local() else Path('/')
SOURCE=ROOT/'artifacts/harmonized-apc-source'
VOLUME=modal.Volume.from_name('presto-checkpoints')
BASE=(modal.Image.debian_slim(python_version='3.12')
 .uv_pip_install('torch==2.14.1','numpy==2.5.4','pyyaml==6.0.3','mhcgnomes==3.64.6','mhcseqs==2.6.13','tqdm==4.70.1','psutil==7.2.2')
 .uv_pip_install('pyarrow==26.0.0','pandas==3.1.0','pyahocorasick==2.3.1')
 .env({'PYTHONPATH':'/runtime:/bundle','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}))

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def rows(path):
 with Path(path).open() as f:
  for line in f:yield json.loads(line)

def dump(path, data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n');tmp.replace(path)

def allele_name(cls, alleles):
 if cls=='I':return alleles[0].replace('*','')
 parts=[x.replace('HLA-','').replace('*','').replace(':','') for x in alleles]
 dr=next((x for x in parts if x.startswith('DRB')),None)
 return dr[:4]+'_'+dr[4:] if dr else 'HLA-'+'-'.join(sorted(parts))

def job_key(cls,allele):return hashlib.sha256(f'{cls}:{allele}'.encode()).hexdigest()[:20]
