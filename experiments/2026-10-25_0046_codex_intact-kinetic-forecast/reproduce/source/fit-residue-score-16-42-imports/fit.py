"""One frozen exact/conditional-score fit with every optimizer checkpoint."""

import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from model import ROOT,component,data,decode,dump_json,initial,log_events,objective,predict

stage=sys.argv[1]
_,kind,method,draw_text,seed_text=stage.split('-')
draws,seed=int(draw_text),int(seed_text)
assert method in ('exact','score') and (method=='exact')==(draws==0)
assert json.loads((ROOT/'results/qualification.json').read_text())['passed']
assert (ROOT/'code/model.py').read_bytes()==(ROOT/'reproduce/source/qualify-imports/model.py').read_bytes()
config=json.loads((ROOT/'reproduce/launch.json').read_text())
assert stage in config['fit_conditions']
datasets,parent,aas,aa_index,sequences=data();train=datasets['train']
directory=ROOT/'results/fits'/stage;directory.mkdir(parents=True)
theta=initial(kind,len(aas));original=theta.detach().numpy().copy()
optimizer=torch.optim.Adam([theta],lr=.03,betas=(.9,.999),eps=1e-8)
generator=torch.Generator().manual_seed(seed)
steps=800;dimension=len(theta);rng_size=len(generator.get_state())
states=np.zeros((steps+1,dimension));moments=np.zeros_like(states);squares=np.zeros_like(states)
offsets=np.zeros((steps+1,len(sequences)));rng_states=np.empty((steps+1,rng_size),dtype=np.uint8)
trace=np.zeros((steps+1,6));coverage=np.zeros(dimension);started=time.monotonic()


def record(step,gradient_norm):
    with torch.no_grad():
        loss,mse,penalty,intercepts=objective(theta,kind,train,aa_index)
    states[step]=theta.detach().numpy();offsets[step]=intercepts.numpy();rng_states[step]=generator.get_state().numpy()
    if step:
        state=optimizer.state[theta]
        assert int(state['step'].item())==step
        moments[step]=state['exp_avg'].numpy();squares[step]=state['exp_avg_sq'].numpy()
    trace[step]=step,float(loss),float(mse),float(penalty),gradient_norm,time.monotonic()-started


record(0,0.)
for step in range(1,steps+1):
    optimizer.zero_grad(set_to_none=True)
    loss,_,_,_=objective(theta,kind,train,aa_index,draws,generator)
    loss.backward()
    assert torch.isfinite(theta.grad).all() and torch.isfinite(loss)
    gradient=theta.grad.detach().numpy().copy();coverage+=np.abs(gradient)
    optimizer.step();record(step,float(np.linalg.norm(gradient)))
    if step%200==0:print(stage,step,'loss',trace[step,1],flush=True)
fit_seconds=time.monotonic()-started
np.savez_compressed(directory/'checkpoints.npz',theta=states,exp_avg=moments,exp_avg_sq=squares,
                    offset=offsets,rng_state=rng_states,trace=trace,sequences=np.array(sequences),
                    aa_categories=np.array(aas),step=np.arange(steps+1))
assert np.all(coverage>0) and np.all(states[-1]!=original)
prediction_rows=[]
for split,batch in datasets.items():
    with torch.no_grad():
        prediction,intercepts=predict(theta,kind,train,batch,aa_index)
        physical=component(theta,kind,batch,aa_index)
    for i,row in enumerate(batch['rows']):
        prediction_rows.append({'model':stage,'split':split,'assignment_id':row['assignment_id'],
                                'prediction':float(prediction[i]),'physical_log10_ratio':float(physical[i]),
                                'training_offset':float(intercepts[batch['sequence_index'][i]])})
pq.write_table(pa.Table.from_pylist(prediction_rows),directory/'predictions.parquet')
rates,tau,categories=decode(theta,kind,aa_index)
hessian=torch.autograd.functional.hessian(lambda p:objective(p,kind,train,aa_index)[0],theta).detach().numpy()
eigenvalues=np.linalg.eigvalsh((hessian+hessian.T)/2)
if method=='exact':
    spans=np.array([(i,j) for i in range(len(parent)) for j in range(i+1,len(parent)+1)])
    values=[]
    with torch.no_grad():
        for minutes in (0,20,40,60,80,100,120):
            counts=log_events(rates,torch.full((len(spans),),minutes+float(tau)),torch.tensor(spans[:,0]),torch.tensor(spans[:,1])).exp().numpy()
            assert abs(counts@(spans[:,1]-spans[:,0])-len(parent))<1e-10
            values.append(counts)
    np.savez_compressed(directory/'unvalidated_expected_span_counts.npz',spans=spans,times=np.arange(0,121,20),expected_counts=np.array(values))
stats={'model':stage,'kind':kind,'method':method,'draws':draws,'seed':seed,'updates':steps,
       'n_parameters':dimension,'n_profiled_training_offsets':len(sequences),'rates_by_category':categories.detach().tolist(),
       'tau_minutes':float(tau.detach()),'raw_parameters':states[-1].tolist(),'gradient_absolute_totals':coverage.tolist(),
       'all_parameters_changed':bool(np.all(states[-1]!=original)), 'hessian_eigenvalues':eigenvalues.tolist(),
       'fit_seconds':fit_seconds,'training_objective':float(trace[-1,1]),'training_mse':float(trace[-1,2]),
       'last_50_step_objective_change':float(trace[-1,1]-trace[-51,1]),
       'conditional_Bernoulli_draw_equivalents':steps*draws*len(train['rows'])*(len(parent)-1),
       'actual_binomial_count_draws':steps*len(train['rows'])*(len(parent)-1) if draws else 0,
       'exact_forward_example_scores':(2*steps+1)*len(train['rows']),
       'sampler':'independent_binomial_sufficient_statistics_of_complete_conditional_cut_sets' if draws else None,
       'prediction_sha256':hashlib.sha256((directory/'predictions.parquet').read_bytes()).hexdigest(),
       'checkpoint_sha256':hashlib.sha256((directory/'checkpoints.npz').read_bytes()).hexdigest(),
       'model_source_sha256':hashlib.sha256((ROOT/'code/model.py').read_bytes()).hexdigest(),
       'versions':{p:importlib.metadata.version(p) for p in ('numpy','torch','pyarrow')},
       'predictive_outcomes_read':False}
dump_json(directory/'summary.json',stats)
print(json.dumps(stats),flush=True)
