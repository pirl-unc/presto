"""Independent NumPy probability/gradient and raw-work replay."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

FAMILY=Path(__file__).resolve().parents[1];ROOT=FAMILY.parents[1];R=FAMILY/'results'
ALPHABET='ACDEFGHIKLMNPQRSTVWYUO'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm_log(x):
 m=np.max(x);return x-m-np.log(np.exp(x-m).sum())

def network(x,weights,prefix):
 w=np.asarray(weights[prefix+'.0.weight'],dtype=np.float64);b=np.asarray(weights[prefix+'.0.bias'],dtype=np.float64);o=np.asarray(weights[prefix+'.2.weight'],dtype=np.float64)
 hidden=np.tanh(x@w.T+b);return (hidden@o.T).reshape(-1),hidden,o

def backward(x,hidden,out,upstream):
 z=upstream[:,None]*out*(1-hidden**2)
 return [z.T@x,z.sum(0),upstream[None,:]@hidden]

def independent(policy_record,width,weights):
 sequence=policy_record['sequence'];size=len(sequence);context=np.asarray([.25,-.5])
 token=np.asarray([ALPHABET.index(a) for a in sequence]+[22]);residues=np.eye(23)[token];prefix=np.vstack([np.zeros((1,22)),np.cumsum(residues[:-1,:22],axis=0)])
 starts=np.repeat(np.arange(size),np.minimum(width,size-np.arange(size)));lengths=np.concatenate([np.arange(1,min(width,size-i)+1) for i in range(size)]);ends=starts+lengths
 positions=np.arange(size);counts=np.minimum(width,size-positions);previous=np.where(positions>0,positions-1,size)
 start_x=np.column_stack([residues[previous],residues[positions],positions/max(size-1,1),counts/width,np.broadcast_to(context,(size,2))])
 a,ha,oa=network(start_x,weights,'start_head');log_start=norm_log(a+np.log(counts))
 mean=(prefix[ends]-prefix[starts])/lengths[:,None]
 length_x=np.column_stack([residues[starts],residues[ends-1],residues[np.where(starts>0,starts-1,size)],residues[ends],mean,lengths/width,starts/max(size-1,1),ends/size,np.broadcast_to(context,(len(starts),2))])
 b,hb,ob=network(length_x,weights,'length_head');offsets=np.r_[0,np.cumsum(counts)[:-1]];maximum=np.maximum.reduceat(b,offsets);log_den=maximum+np.log(np.add.reduceat(np.exp(b-maximum[starts]),offsets));log_cond=b-log_den[starts];logq=log_start[starts]+log_cond
 cost=.7*(mean[:,3]+mean[:,4]-mean[:,0])+.3*np.sin(2*np.pi*starts/size)+3*(lengths/width-.4)**2
 q=np.exp(logq);terms=logq+cost;loss=float(np.dot(q,terms));cond_expectation=np.add.reduceat(np.exp(log_cond)*terms,offsets)
 grad_start=np.exp(log_start)*(cond_expectation-loss);grad_length=q*(terms-cond_expectation[starts])
 grads=backward(start_x,ha,oa,grad_start)+backward(length_x,hb,ob,grad_length)
 return starts,lengths,logq,cost,np.concatenate([g.reshape(-1) for g in grads]),loss

def main():
 tick=time.monotonic();assert not (R/'independent-verification.json').exists()
 hashes=json.loads((R/'output_hashes.json').read_text())
 for name,value in hashes.items():assert sha(R/name)==value
 config=json.loads((FAMILY/'reproduce/launch.json').read_text());assert sha(FAMILY/'reproduce/source/peptide_availability.py')==config['model_code_sha256']
 selected=json.loads((R/'selected_parents.json').read_text());inputs=json.loads((FAMILY/'reproduce/input_manifest.json').read_text())
 for source,receipt in inputs.items():
  path=ROOT/receipt['path'];assert sha(path)==receipt['sha256']
  with gzip.open(path,'rt') as h:original=[json.loads(line) for line in h]
  for row in [r for r in selected if r['source']==source]:
   low,high=row['band'];chosen=min((r for r in original if low<=len(r['sequence'])<high),key=lambda r:(r['sequence_sha256'],r['accession']))
   assert row['sequence']==chosen['sequence'] and row['accession']==chosen['accession'] and row['length']==len(chosen['sequence'])
   assert row['sequence_sha256']==hashlib.sha256(row['sequence'].encode()).hexdigest()
 summary=json.loads((R/'summary.json').read_text());exact={k:v for k,v in np.load(R/'exact_gradients.npz').items()};checks=[]
 weights={width:{k:v.numpy() for k,v in torch.load(R/f'weights-w{width}.pt',weights_only=True).items()} for width in [40,53]}
 exact_logs={}
 for row in selected:
  for width in [40,53]:
   case=f'{row["source"]}-{row["band"][0]}-w{width}';starts,lengths,logq,cost,grad,loss=independent(row,width,weights[width]);saved={k:v for k,v in np.load(R/(case+'-reference.npz')).items()}
   np.testing.assert_array_equal(starts,saved['starts']);np.testing.assert_array_equal(lengths,saved['lengths']);np.testing.assert_allclose(logq,saved['logq'],atol=3e-6,rtol=0);np.testing.assert_allclose(cost,saved['energy'],atol=5e-7,rtol=0);np.testing.assert_allclose(grad,exact[case],atol=3e-6,rtol=2e-4)
   exact_logs[case]={(int(s),int(n)):float(v) for s,n,v in zip(starts,lengths,logq)}
   assert abs(np.exp(logq).sum()-1)<1e-12
   checks.append({'case':case,'spans':len(starts),'max_logq_error':float(np.max(np.abs(logq-saved['logq']))),'max_gradient_error':float(np.max(np.abs(grad-exact[case]))),'exact_loss_numpy':loss})
 with gzip.open(R/'timings_and_draws.jsonl.gz','rt') as h:timings=[json.loads(line) for line in h]
 assert len(timings)==1280
 for r in timings:
  size,width=r['length'],r['max_length'];assert r['start_network_rows']==size and r['seconds']>0 and np.isfinite(r['seconds'])
  assert r['valid_span_count']==sum(min(width,size-s) for s in range(size))
  if r['method']=='exact':assert r['cost_queries']==r['valid_span_count'] and r['conditional_network_rows']==size*width
  else:
   assert len(r['starts'])==len(r['lengths'])==len(r['selected_log_probs'])==r['samples']==r['cost_queries']
   expected=len(set(r['starts']))*width if r['method']=='lazy' else size*width;assert r['conditional_network_rows']==expected
   for start,length,value in zip(r['starts'],r['lengths'],r['selected_log_probs']):assert abs(value-exact_logs[r['case']][start,length])<3e-6
 for stat in summary['conditions']:
  rows=[r for r in timings if r['case']==stat['case'] and r['method']==stat['method'] and r['samples']==stat['samples'] and not r['warmup']];assert len(rows)==stat['measured_calls']==12
  assert float(np.median([r['seconds'] for r in rows]))==stat['median_seconds']
  assert min(r['seconds'] for r in rows)==stat['min_seconds'] and max(r['seconds'] for r in rows)==stat['max_seconds']
  assert float(np.mean([r['conditional_network_rows'] for r in rows]))==stat['mean_conditional_network_rows']
 for key,stat in summary['gradient_probes'].items():
  data={k:v for k,v in np.load(R/(key+'-gradients.npz')).items()};g=data['estimates'].astype(np.float64);reference=data['exact'].astype(np.float64);mean=g.mean(0);err=np.linalg.norm(mean-reference);se=g.std(0,ddof=1)/np.sqrt(len(g))
  expected={'draws':len(g),'relative_l2_error':err/np.linalg.norm(reference),'cosine':np.dot(mean,reference)/(np.linalg.norm(mean)*np.linalg.norm(reference)),'coordinate_se_l2':np.linalg.norm(se),'exact_gradient_norm':np.linalg.norm(reference),'absolute_l2_error':err}
  for name,value in expected.items():np.testing.assert_allclose(value,stat[name],rtol=3e-6,atol=1e-8)
  assert np.array_equal(data['exact'],exact[key.rsplit('-k',1)[0]])
 pairs=[]
 for a in [r for r in timings if r['method']=='dense']:
  b=next(r for r in timings if r['method']=='lazy' and all(r[k]==a[k] for k in ['case','samples','repetition','seed']));pairs.append(a['starts']==b['starts'] and a['lengths']==b['lengths'])
 assert {'pairs':len(pairs),'identical':sum(pairs)}==summary['seed_matched_sample_agreement']
 shutil.copy2(__file__,FAMILY/'reproduce/source/verify.py')
 output={'passed':True,'source_records':len(selected),'input_hashes':len(inputs),'output_hashes':len(hashes),'reference_cases':len(checks),'complete_span_probabilities':sum(r['spans'] for r in checks),'fixed_weight_numpy_gradient_cases':len(checks),'timed_calls':len(timings),'seed_matched_sample_pairs':len(pairs),'gradient_probe_groups':len(summary['gradient_probes']),'checks':checks,'elapsed_seconds':time.monotonic()-tick,'code_sha256':sha(__file__),'scope':'independent NumPy feature/MLP/normalization/analytic gradient; raw timing/work and source selection replay'}
 (R/'independent-verification.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='checks'}),flush=True)

if __name__=='__main__':main()
