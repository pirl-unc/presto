"""All native positives, protected source pools, pinned allele-specific BA scoring."""
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import tempfile
import tarfile
import time
import modal
from common import NAME,PRIOR,BROAD,E,ROOT,SOURCE,VOLUME,BASE,sha,rows,dump,allele_name,job_key

app=modal.App('presto-broad-processing-affinity')
prep_image=BASE
external_image=(modal.Image.from_registry('debian:bookworm-slim',add_python='3.12')
 .run_commands('apt-get update','apt-get install -y tcsh gawk')
 .env({'PYTHONPATH':'/bundle','NETMHC_BUNDLE_HOME':'/net','PYTHONDONTWRITEBYTECODE':'1'}))
if modal.is_local():
 prep_image=(prep_image.add_local_dir(ROOT/'experiments'/BROAD/'reproduce/runtime-pathway','/runtime/presto')
  .add_local_dir(E/'code','/bundle')
  .add_local_file(SOURCE/'data/affinity_matching.py','/bundle/affinity_matching.py')
  .add_local_file(ROOT/'data/shared-presentation-20261117/proteins.fa.gz','/inputs/proteins.fa.gz'))
 external_image=(external_image.add_local_dir(E/'code','/bundle')
  .add_local_file(ROOT/'artifacts/affinity-conditioned-processing/external-tools.tar.gz','/bundle/external-tools.tar.gz'))


def profile_for(r):
 from affinity_matching import candidate_key
 cls=r['mhc'][0].get('class_override') or r['mhc'][0]['mhc_class']
 mapping={}
 for m in r['mhc']:
  alleles=m.get('provenance',{}).get('catalog_alleles')
  if not alleles:raise ValueError('missing external allele name')
  mapping[candidate_key(m)]=allele_name(cls,alleles)
 return cls,mapping


@app.function(image=prep_image,cpu=8,memory=65536,timeout=7200,volumes={'/volume':VOLUME})
def prepare():
 from presto.data.presentation import read_gencode_proteins,junction
 start=time.monotonic();root=Path('/volume')/NAME;out=root/'pools';out.mkdir(parents=True,exist_ok=True)
 if (out/'receipt.json').exists():return json.loads((out/'receipt.json').read_text())
 source=Path('/volume')/BROAD/'inputs/expanded-train.jsonl'
 report=json.loads((Path('/volume')/BROAD/'preparation.json').read_text())
 assert sha(source)==report['files']['expanded-train.jsonl']['sha256']
 assert sha('/inputs/proteins.fa.gz')==report['source_files']['proteins.fa.gz']
 proteins=read_gencode_proteins('/inputs/proteins.fa.gz')
 pools=defaultdict(dict);positives=[];counts=Counter();original_negative={}
 with source.open('rb',buffering=1024**2) as f:
  while True:
   offset=f.tell();line=f.readline()
   if not line:break
   r=json.loads(line);assert r['split']=='train'
   p=r['provenance'];protein=p['source_protein'];key=(protein,len(r['peptide']))
   if r['value']==0:
    position=p['generated_position'];sequence=proteins[protein].sequence
    assert sequence[position:position+len(r['peptide'])]==r['peptide']
    context=junction(sequence,position,len(r['peptide']))
    pools[key].setdefault(r['peptide'],context)
    original_negative[p['parent_positive_id']]=r['peptide']
   else:
    assert r['value']==1
    cls,mapping=profile_for(r);counts['all_positives:'+cls]+=1
    if not ((8<=len(r['peptide'])<=14) if cls=='I' else (9<=len(r['peptide'])<=30)):
     counts['predictor_length_excluded:'+cls]+=1;continue
    positives.append({'id':r['observation_id'],'peptide':r['peptide'],'class':cls,'alleles':mapping,'protein':protein,'offset':offset,'study':str(p['study_id'])})
 queries=defaultdict(set);option_counts=Counter();proposal=out/'proposals.jsonl'
 with proposal.open('w',buffering=1024**2) as f:
  for r in positives:
   own=original_negative[r['id']];pool=pools[(r['protein'],len(r['peptide']))]
   ordered=sorted(pool,key=lambda pep:hashlib.sha256((r['id']+'|'+pep).encode()).hexdigest())
   chosen=[own]+[p for p in ordered if p!=own][:31]
   r['options']=[{'peptide':p,'context':pool[p]} for p in chosen]
   option_counts[r['class']]+=min(8,len(chosen));counts['eligible:'+r['class']]+=1
   for allele in set(r['alleles'].values()):queries[(r['class'],allele)].update([r['peptide'],*chosen[:8]])
   f.write(json.dumps(r,separators=(',',':'))+'\n')
 jobs=[]
 for (cls,allele),peptides in sorted(queries.items()):
  path=out/'queries-stage1'/(job_key(cls,allele)+'.json');dump(path,sorted(peptides))
  jobs.append({'class':cls,'allele':allele,'path':str(path),'sha256':sha(path),'queries':len(peptides)})
 receipt={'status':'prepared','source':str(source),'source_sha256':sha(source),'proposal_sha256':sha(proposal),'counts':dict(counts),'initial_options_by_class':dict(option_counts),'jobs':jobs,'total_queries':sum(j['queries'] for j in jobs),'seconds':time.monotonic()-start,'positive_cap':None,'study_cap':None,'candidate_policy':'first8 of up to32 deterministic existing protected backgrounds from the same exact reference protein and length, own original background included; no new sequence identities','test_accessed':False}
 dump(out/'receipt.json',receipt);VOLUME.commit();return receipt



def ensure_external():
 if not Path('/net/netMHCpan-4.2/data/version').exists():
  with tarfile.open('/bundle/external-tools.tar.gz') as archive:archive.extractall('/net',filter='data')
 for package,version in [('netMHCpan-4.2','4.2c'),('netMHCIIpan-4.3','4.3e')]:
  assert version in (Path('/net')/package/'Linux_x86_64/data/version').read_text()


@app.function(image=external_image,cpu=2,memory=12288,timeout=1800,volumes={'/volume':VOLUME})
def probe():
 ensure_external();report={}
 with tempfile.TemporaryDirectory() as td:
  for cls,package,wrapper,allele,pep,version in [('I','netMHCpan-4.2','netMHCpan','HLA-A02:01','SLYNTVATL','4.2c'),('II','netMHCIIpan-4.3','netMHCIIpan','DRB1_0101','ACDEFGHIKLMNPQR','4.3e')]:
   path=Path(td)/'peptides';path.write_text(pep+'\n')
   cmd=[f'/net/{package}/{wrapper}','-p' if cls=='I' else '-f',str(path),'-a',allele,'-BA']+([] if cls=='I' else ['-inptype','1'])
   result=subprocess.run(cmd,env=dict(os.environ,NETMHC_BUNDLE_TMPDIR=td),capture_output=True,text=True,timeout=900)
   report[cls]={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
   dump(Path('/volume')/NAME/'external/package-probe-v2.json',report);VOLUME.commit()
   assert result.returncode==0 and ('version '+version) in result.stdout,(cls,report[cls])
 return {cls:{'passed':v['returncode']==0} for cls,v in report.items()}

@app.function(image=external_image,cpu=2,memory=12288,timeout=14400,max_containers=16,volumes={'/volume':VOLUME},retries=0)
def score(job,stage):
 ensure_external()
 start=time.monotonic();cls,allele=job['class'],job['allele'];assert stage in (1,2)
 assert sha(job['path'])==job['sha256'];peptides=json.loads(Path(job['path']).read_text())
 root=Path('/volume')/NAME/'external'/f'stage{stage}';target=root/(job_key(cls,allele)+'.json')
 if target.exists():
  prior=json.loads(target.read_text());assert prior['query_sha256']==job['sha256'];return {'path':str(target),'cached':True,'rows':len(prior['scores'])}
 package,wrapper,version=('netMHCpan-4.2','netMHCpan','4.2c') if cls=='I' else ('netMHCIIpan-4.3','netMHCIIpan','4.3e')
 cached={};cache_sources=[];wanted=set(peptides)
 for base in [Path('/volume')/PRIOR/'external',Path('/volume')/PRIOR/'external-train',Path('/volume')/NAME/'external/stage1']:
  p=base/(job_key(cls,allele)+'.json')
  if p==target or not p.exists():continue
  old=json.loads(p.read_text());assert old['class']==cls and old['allele']==allele and old['version']==version
  for pep,v in old['scores'].items():
   if pep in wanted:
    if pep in cached:assert cached[pep]==v
    cached[pep]=v
  cache_sources.append({'path':str(p),'sha256':sha(p)})
 missing=sorted(wanted-set(cached));chunks=[]
 with tempfile.TemporaryDirectory() as td:
  for offset in range(0,len(missing),5000):
   batch=missing[offset:offset+5000];digest=hashlib.sha256(('\n'.join(batch)).encode()).hexdigest()
   chunk=root/'chunks'/job_key(cls,allele)/(digest+'.json')
   if chunk.exists():
    values=json.loads(chunk.read_text());assert set(values)==set(batch);cached.update(values);chunks.append(str(chunk));continue
   path=Path(td)/'peptides';path.write_text('\n'.join(batch)+'\n')
   cmd=[f'/net/{package}/{wrapper}', '-p' if cls=='I' else '-f',str(path),'-a',allele,'-BA']+([] if cls=='I' else ['-inptype','1'])
   result=subprocess.run(cmd,env=dict(os.environ,NETMHC_BUNDLE_TMPDIR=td),capture_output=True,text=True,timeout=3600)
   if result.returncode or ('version '+version) not in result.stdout:
    dump(root/'failures'/(job_key(cls,allele)+'.json'),{'job':job,'returncode':result.returncode,'stdout':result.stdout[-6000:],'stderr':result.stderr[-6000:]});VOLUME.commit();raise ValueError('external prediction failed: '+allele)
   header=None;values={}
   for line in result.stdout.splitlines():
    cells=line.split()
    if cells and cells[0]=='Pos' and 'Peptide' in cells:header=cells
    elif header and cells and cells[0].isdigit():
     d=dict(zip(header,cells));pep=d['Peptide']
     def value(*names):
      for name in names:
       if name in d:return float(d[name])
      raise ValueError((names,header))
     v={'ba_nm':value('Aff(nM)','Affinity(nM)','Affinity'),'ba_rank':value('%Rank_BA'),'el_score':value('Score_EL'),'el_rank':value('%Rank_EL')}
     assert all(math.isfinite(x) for x in v.values()) and v['ba_nm']>0 and 0<=v['ba_rank']<=100
     assert pep not in values;values[pep]=v
   assert set(values)==set(batch),(cls,allele,len(values),len(batch),result.stdout[-1000:])
   dump(chunk,values);VOLUME.commit();cached.update(values);chunks.append(str(chunk))
 assert set(cached)==wanted
 record={'class':cls,'allele':allele,'version':version,'query_sha256':job['sha256'],'scores':cached,'reuse_sources':cache_sources,'new_queries':len(missing),'chunks':chunks,'seconds':time.monotonic()-start}
 dump(target,record);VOLUME.commit();return {'path':str(target),'rows':len(cached),'new_queries':len(missing),'seconds':record['seconds']}


@app.function(image=prep_image,cpu=4,memory=32768,timeout=43200,volumes={'/volume':VOLUME})
def coordinate(stage=1):
 assert stage in (1,2)
 probe.remote()
 root=Path('/volume')/NAME
 receipt=prepare.remote() if stage==1 else json.loads((root/'pools/analysis-stage1.json').read_text())
 VOLUME.reload()
 dump(root/'external/coordinator-start.json',{'stage':stage,'jobs':len(receipt['jobs']),'queries':receipt['total_queries']});VOLUME.commit()
 journal=root/f'external/calls-stage{stage}.json'
 if journal.exists():
  prior=json.loads(journal.read_text());assert [r['job'] for r in prior]==receipt['jobs']
  calls=[(r['job'],modal.FunctionCall.from_id(r['call'])) for r in prior]
 else:
  calls=[(j,score.spawn(j,stage)) for j in receipt['jobs']]
  dump(journal,[{'job':j,'call':c.object_id} for j,c in calls]);VOLUME.commit()
 results=[];errors=[]
 for j,c in calls:
  try:results.append(c.get())
  except Exception as e:errors.append({'job':j,'error':repr(e)})
  dump(root/f'external/progress-stage{stage}.json',{'completed':len(results),'total':len(calls),'errors':errors});VOLUME.commit()
 record={'status':'failed' if errors else 'completed','results':results,'errors':errors,'stage':stage,'test_accessed':False}
 dump(root/f'external/completed-stage{stage}.json',record);VOLUME.commit()
 if errors:raise ValueError(f'{len(errors)} external jobs failed; receipts preserved')
 return record


@app.local_entrypoint()
def main(action:str='prepare',stage:int=1):
 call=(prepare.spawn() if action=='prepare' else coordinate.spawn(stage))
 dump(E/'reproduce'/f'pool-{action}-stage{stage}-call.json',{'call':call.object_id,'action':action})
 print('call',call.object_id,flush=True);result=call.get();dump(E/'results'/f'pool-{action}-stage{stage}.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','results')}),flush=True)
