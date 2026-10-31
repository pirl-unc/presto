"""Fixed-weight compute/gradient benchmark; real sequences, synthetic energy."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from presto.models.peptide_availability import PeptideAvailabilityPolicy

FAMILY=Path(__file__).resolve().parents[1];ROOT=FAMILY.parents[1];R=FAMILY/'results'
CONTEXT=[0.25,-0.5]
BANDS=[(64,128),(256,512),(1024,2048),(2048,4096)]
CONDITIONS=[('exact',0),('dense',8),('lazy',8),('dense',64),('lazy',64)]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def gradient(model):return torch.cat([p.grad.detach().flatten() for p in model.parameters()]).cpu().numpy().copy()

def energy(state,starts,lengths):
 mean=(state.prefix[starts+lengths]-state.prefix[starts])/lengths[:,None]
 return .7*(mean[:,3]+mean[:,4]-mean[:,0])+.3*torch.sin(2*torch.pi*starts/state.protein_length)+3*(lengths/state.max_length-.4).square()

def exact_values(policy,state):
 pieces=list(policy.enumerate(state,chunk_size=256))
 starts,lengths,logq=(torch.cat([p[i] for p in pieces]) for i in range(3))
 cost=energy(state,starts,lengths)
 return starts,lengths,logq,cost

def dense_sample(policy,state,count,generator):
 positions=torch.arange(len(state.start_log_probs))
 conditional,lengths,_=policy.conditional_log_probs(state,positions)
 starts=torch.multinomial(state.start_log_probs.detach().exp(),count,replacement=True,generator=generator)
 offsets=torch.multinomial(conditional.detach().exp()[starts],1,generator=generator).squeeze(-1)
 return starts,lengths[offsets],state.start_log_probs[starts]+conditional[starts,offsets]

def step(policy,record,method,count,seed,*,trace=False):
 work=defaultdict(int)
 def hook(name):
  def record_work(_module,args):work[name]+=args[0].numel()//args[0].shape[-1]
  return record_work
 handles=[policy.start_head[0].register_forward_pre_hook(hook('start')),policy.length_head[0].register_forward_pre_hook(hook('conditional'))]
 begin=time.perf_counter()
 try:
  policy.zero_grad(set_to_none=True)
  state=policy.prepare(record['sequence'],CONTEXT)
  if method=='exact':
   starts,lengths,logq,cost=exact_values(policy,state)
   loss=(logq.exp()*(logq+cost)).sum();queries=len(logq);expected_rows=len(state.start_log_probs)*policy.width
  else:
   generator=torch.Generator().manual_seed(seed)
   if method=='dense':starts,lengths,logq=dense_sample(policy,state,count,generator);expected_rows=len(state.start_log_probs)*policy.width
   else:
    sampled=policy.sample(state,count,generator=generator);starts,lengths,logq=sampled.starts,sampled.lengths,sampled.log_probs;expected_rows=sampled.conditional_network_rows
   costs=logq+energy(state,starts,lengths);baseline=(costs.sum()-costs)/(count-1)
   loss=((costs-baseline).detach()*logq).mean();queries=count
  loss.backward();grad=gradient(policy)
  elapsed=time.perf_counter()-begin
 finally:
  for handle in handles:handle.remove()
 assert work['start']==len(record['sequence']) and work['conditional']==expected_rows
 assert np.isfinite(grad).all() and np.any(grad)
 data={'seconds':elapsed,'start_network_rows':work['start'],'conditional_network_rows':work['conditional'],'valid_span_count':state.valid_span_count,'cost_queries':queries,'loss_or_surrogate':float(loss.detach()),'gradient_norm':float(np.linalg.norm(grad))}
 if method!='exact' and trace:
  data.update({'starts':starts.tolist(),'lengths':lengths.tolist(),'selected_log_probs':logq.detach().tolist()})
 return data,grad

def fixed_span_check(policy,record):
 policy.zero_grad(set_to_none=True);state=policy.prepare(record['sequence'],CONTEXT)
 starts,lengths,dense=dense_sample(policy,state,64,torch.Generator().manual_seed(9901))
 costs=dense+energy(state,starts,lengths);centered=(costs-(costs.sum()-costs)/63).detach()
 (centered*dense).mean().backward();gd=gradient(policy)
 policy.zero_grad(set_to_none=True);other=policy.prepare(record['sequence'],CONTEXT);lazy=policy.log_prob(other,starts,lengths);(centered*lazy).mean().backward();gl=gradient(policy)
 torch.testing.assert_close(lazy,dense,atol=3e-6,rtol=2e-5)
 np.testing.assert_allclose(gl,gd,atol=3e-6,rtol=2e-4)
 return {'max_abs_logp_error':float((lazy-dense).detach().abs().max()),'max_abs_gradient_error':float(np.max(np.abs(gl-gd))),'relative_gradient_error':float(np.linalg.norm(gl-gd)/max(np.linalg.norm(gd),1e-30))}

def main():
 tick=time.monotonic();assert not (R/'summary.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 inputs=json.loads((FAMILY/'reproduce/input_manifest.json').read_text());selected=[]
 for source,receipt in inputs.items():
  path=ROOT/receipt['path'];assert sha(path)==receipt['sha256']
  with gzip.open(path,'rt') as h:records=[json.loads(line) for line in h]
  for low,high in BANDS:
   eligible=[r for r in records if low<=len(r['sequence'])<high]
   assert eligible,('empty registered band',source,low,high)
   chosen=min(eligible,key=lambda r:(r['sequence_sha256'],r['accession']))
   selected.append({'source':source,'band':[low,high],'accession':chosen['accession'],'sequence':chosen['sequence'],'sequence_sha256':chosen['sequence_sha256'],'length':len(chosen['sequence'])})
 dump(R/'selected_parents.json',selected)
 model_path=ROOT/'models/peptide_availability.py';shutil.copy2(model_path,FAMILY/'reproduce/source/peptide_availability.py');shutil.copy2(__file__,FAMILY/'reproduce/source/launch.py')
 launch={'agent_model':'Codex / GPT-6','argv':sys.argv,'cwd':str(ROOT),'executable':sys.executable,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],text=True),'environment_overrides':{k:os.environ.get(k) for k in ['PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']},'torch_threads':4,'torch_interop_threads':1,'dtype':'float32','warmup':4,'measured':12,'conditions':CONDITIONS,'supports':[[1,40],[1,53]],'gradient_draws':128,'model_seed':42,'context':CONTEXT,'optimizer':None,'training_updates':0,'assay_labels':None,'predictive_split':None,'no_prediction_dump_reason':'fixed-weight computational experiment; full mathematical reference arrays and draw traces are retained','model_code_sha256':sha(model_path),'launcher_sha256':sha(__file__),'device':'local CPU','synthetic_contract':'fixed span energy, not biological labels'}
 dump(FAMILY/'reproduce/launch.json',launch)
 timings=[];checks=[];references={};probes={};seed_order=np.random.default_rng(20261022)
 for index,record in enumerate(selected):
  for width in [40,53]:
   case=f'{record["source"]}-{record["band"][0]}-w{width}'
   torch.manual_seed(42);policy=PeptideAvailabilityPolicy(max_length=width,context_size=2)
   checkpoint=R/f'weights-w{width}.pt'
   if not checkpoint.exists():torch.save(policy.state_dict(),checkpoint)
   state=policy.prepare(record['sequence'],CONTEXT);starts,lengths,logq,cost=exact_values(policy,state)
   expected=sum(min(width,record['length']-s) for s in range(record['length']))
   assert len(logq)==expected and len(set(zip(starts.tolist(),lengths.tolist())))==expected
   torch.testing.assert_close(logq.exp().sum(),torch.tensor(1.),atol=2e-6,rtol=0)
   np.savez_compressed(R/(case+'-reference.npz'),starts=starts.numpy(),lengths=lengths.numpy(),logq=logq.detach().numpy(),energy=cost.numpy())
   exact_record,exact_grad=step(policy,record,'exact',0,0)
   references[case]={'exact_gradient':exact_grad,'exact_loss':exact_record['loss_or_surrogate']}
   check=fixed_span_check(policy,record);checks.append({'case':case,'length':record['length'],'valid_spans':expected,**check})
   for repetition in range(16):
    for condition_index in seed_order.permutation(len(CONDITIONS)):
     method,count=CONDITIONS[condition_index];seed=400000+index*1000+width*10+repetition+count*1000000
     result,_=step(policy,record,method,count,seed,trace=method!='exact')
     timings.append({'case':case,'source':record['source'],'length':record['length'],'max_length':width,'method':method,'samples':count,'repetition':repetition,'warmup':repetition<4,'seed':seed,**result})
   if record['band'][0]==2048 and width==40:
    for count in [8,64]:
     estimates=[]
     for draw in range(128):
      _,g=step(policy,record,'lazy',count,9000000+index*10000+count*100+draw);estimates.append(g)
     estimates=np.stack(estimates);probes[case+f'-k{count}']={'exact':exact_grad,'estimates':estimates}
   print(case,record['length'],'residues; equivalence passed',flush=True)
 with gzip.open(R/'timings_and_draws.jsonl.gz','wt') as out:
  for row in timings:out.write(json.dumps(row,allow_nan=False)+'\n')
 np.savez_compressed(R/'exact_gradients.npz',**{k:v['exact_gradient'] for k,v in references.items()})
 probe_summary={}
 for key,p in probes.items():
  np.savez_compressed(R/(key+'-gradients.npz'),**p)
  mean=p['estimates'].mean(0);reference=p['exact'];se=p['estimates'].std(0,ddof=1)/np.sqrt(len(p['estimates']))
  probe_summary[key]={'draws':len(p['estimates']),'relative_l2_error':float(np.linalg.norm(mean-reference)/np.linalg.norm(reference)),'cosine':float(np.dot(mean,reference)/(np.linalg.norm(mean)*np.linalg.norm(reference))),'coordinate_se_l2':float(np.linalg.norm(se)),'exact_gradient_norm':float(np.linalg.norm(reference)),'absolute_l2_error':float(np.linalg.norm(mean-reference))}
 summaries=[]
 for case in references:
  for method,count in CONDITIONS:
   rows=[r for r in timings if r['case']==case and r['method']==method and r['samples']==count and not r['warmup']]
   summaries.append({'case':case,'method':method,'samples':count,'length':rows[0]['length'],'max_length':rows[0]['max_length'],'median_seconds':float(np.median([r['seconds'] for r in rows])),'min_seconds':min(r['seconds'] for r in rows),'max_seconds':max(r['seconds'] for r in rows),'mean_conditional_network_rows':float(np.mean([r['conditional_network_rows'] for r in rows])),'start_network_rows':rows[0]['start_network_rows'],'valid_span_count':rows[0]['valid_span_count'],'cost_queries':rows[0]['cost_queries'],'measured_calls':len(rows)})
 sample_checks=[]
 for case in references:
  for count in [8,64]:
   for repetition in range(16):
    a,b=[next(r for r in timings if r['case']==case and r['method']==method and r['samples']==count and r['repetition']==repetition) for method in ['dense','lazy']]
    sample_checks.append(a['starts']==b['starts'] and a['lengths']==b['lengths'])
 dump(R/'summary.json',{'conditions':summaries,'fixed_span_checks':checks,'gradient_probes':probe_summary,'seed_matched_sample_agreement':{'pairs':len(sample_checks),'identical':sum(sample_checks)},'parameter_count':sum(p.numel() for p in policy.parameters()),'parameter_tensors':len(list(policy.parameters())),'elapsed_seconds':time.monotonic()-tick,'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3),'no_fit':True,'status':'benchmark_complete_independent_replay_pending'})
 dump(R/'output_hashes.json',{p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name!='output_hashes.json'})
 print(json.dumps({'seconds':time.monotonic()-tick,'cases':len(references),'timed_calls':len(timings),'sample_agreement':sum(sample_checks),'sample_pairs':len(sample_checks),'gradient_probes':probe_summary}),flush=True)

if __name__=='__main__':main()
