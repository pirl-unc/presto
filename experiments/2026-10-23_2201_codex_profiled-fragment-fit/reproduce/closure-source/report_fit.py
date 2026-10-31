"""Close actual biological outcomes with frozen cross-family comparisons."""
import csv
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import FAMILY,dump
from verify_fit import metrics as independent_metrics


def main():
    cfg=json.loads((FAMILY/'reproduce/launch.json').read_text())
    previous=__import__('pathlib').Path(cfg['previous_family']);results={};rows=[]
    for arm in ['analytical','fragment']:
        out=FAMILY/'results'/arm;state=json.loads((out/'status.json').read_text())
        assert state['status']!='running','Do not close a live arm'
        if state['status']=='completed':
            v=json.loads((out/'independent-verification.json').read_text());assert v['all_pass']
            e=json.loads((out/'evaluation.json').read_text())
            results[arm]=dict(status=state,evaluation=e,verification=v)
            for split,metrics in e['metrics'].items():rows.append(dict(arm=arm,split=split,**metrics))
        else:results[arm]=dict(status=state)
    if rows:
        with (FAMILY/'results/comparison.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    table=[];runtime=[];offsets=[];old={}
    for arm,r in results.items():
        oe=json.loads((previous/'results'/arm/'evaluation.json').read_text());old[arm]=oe
        for label,e in [(f'Prior joint-offset {arm}',oe),(f'New profiled-offset {arm}',r.get('evaluation'))]:
            if e is None:
                table.append(f'| {label} | FAILED | — | — | — | — |');continue
            tr=e['metrics']['train'];v=e['metrics']['val']
            table.append(f"| {label} | {tr['shape_rmse']:.5f} | {v['shape_rmse']:.5f} | {v['shape_pearson']:.4f} / {v['shape_spearman']:.4f} | {v['ratio_rmse']:.5f} | {v['prediction_sd']:.4f} / {v['target_sd']:.4f} |")
        if 'evaluation' not in r:continue
        e=r['evaluation'];st=r['status'];vr=r['verification'];n=len(st['coverage']['material_gradient_counts'])
        runtime.append(f"| {arm} | {st['seconds']:.2f} | {st['peak_rss_bytes']/1024**3:.3f} | {n} / {n} | {e['eta'] if arm=='fragment' else 0} |")
        offsets.append(f"| {arm} | {e['training_objective']:.6f} | {e['training_data_mse']:.6f} | {vr['training_offset_optimization_gap']:.3g} |")
    done={a:r for a,r in results.items() if 'evaluation' in r}
    if len(done)==2:
        a=done['analytical']['evaluation']['metrics']['val'];f=done['fragment']['evaluation']['metrics']['val']
        conclusion=f"Exact condition-offset solving gives development shape RMSE **{a['shape_rmse']:.5f} analytical / {f['shape_rmse']:.5f} fragment**, versus **{old['analytical']['metrics']['val']['shape_rmse']:.5f} / {old['fragment']['metrics']['val']['shape_rmse']:.5f}** with learned offsets. The added length effect {'improves' if f['shape_rmse']<a['shape_rmse'] else 'does not improve'} the matched analytical result."
    else:conclusion='A registered arm failed; preserve its failure and the completed comparison without selecting a rescued checkpoint.'
    per_condition_table=[]
    if len(done)==2:
        val_rows={}
        for arm in done:
            with gzip.open(FAMILY/'results'/arm/'predictions.jsonl.gz','rt') as stream:
                val_rows[arm]=[row for line in stream if (row:=json.loads(line))['split']=='val']
        development_conditions={r['condition_id'] for r in val_rows['analytical']}
        for condition in sorted(development_conditions):
            for arm in done:
                rr=[dict(r) for r in val_rows[arm] if r['condition_id']==condition]
                weight=sum(r['weight'] for r in rr)
                for r in rr:r['weight']/=weight
                calculated=independent_metrics(rr)
                recorded=done[arm]['evaluation']['per_condition'][condition]
                for key,value in calculated.items():
                    if value is None:assert recorded[key] is None
                    else:np.testing.assert_allclose(value,recorded[key],atol=1e-10,rtol=1e-9)
            av=done['analytical']['evaluation']['per_condition'][condition]
            fv=done['fragment']['evaluation']['per_condition'][condition]
            assert av['n_observations']==fv['n_observations']
            label=condition.rsplit(':',1)[-1]
            per_condition_table.append(f"| {label} | {av['n_observations']} | {av['shape_rmse']:.5f} | {fv['shape_rmse']:.5f} | {av['ratio_rmse']:.5f} | {fv['ratio_rmse']:.5f} |")
    labels=['Constant','Time quadratic','Prior exact: 2,000 full batches','Prior hybrid: 2,000 full batches'];scores=[1.28060,1.27383,1.26510,1.26279]
    for arm in ['analytical','fragment']:labels.append(f'Joint offsets: {arm}');scores.append(old[arm]['metrics']['val']['shape_rmse'])
    for arm,r in done.items():labels.append(f'Profiled offsets: {arm}');scores.append(r['evaluation']['metrics']['val']['shape_rmse'])
    fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ax[0].scatter(scores,np.arange(len(scores)),color=['#777777']*6+['#2873a5','#af5832'][:len(done)])
    ax[0].set(yticks=np.arange(len(scores)),yticklabels=labels,xlabel='Development log10-signal shape RMSE',title='Biological comparison');ax[0].invert_yaxis()
    for arm in ['analytical','fragment']:
        h=json.loads((FAMILY/'results'/arm/'history.json').read_text())
        if h:
            values=np.array([r['pre_total_loss'] for r in h]);n=min(25,len(values));smooth=np.convolve(values,np.ones(n)/n,mode='valid');ax[1].plot(np.arange(n,len(values)+1),smooth,label=arm)
    ax[1].set(xlabel='Update',ylabel='25-block moving average objective',title='Training, fixed final checkpoint');ax[1].legend(frameon=False)
    fig.savefig(FAMILY/'results/comparison.png',dpi=160)
    q=json.loads((FAMILY/'results/qualification-summary.json').read_text());fit=json.loads((FAMILY/'reproduce/fitting.json').read_text())
    report=f'''# Biological fragment fits with exactly solved condition offsets

{conclusion}

The full APC peptide-availability/concentration goal is not achieved. These are
relative MS signal-shape predictions on purified human standard 20S digestion,
not cell concentrations, a complete Presto fit, or a new RL comparison.

## Biological outcomes

| Condition | Train shape RMSE | Development shape RMSE | Development Pearson / Spearman | Development temporal-ratio RMSE | Development prediction / target SD |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Constant-condition scale gives development shape RMSE 1.28060 and temporal-ratio
RMSE 1.16704. Earlier time quadratic gives 1.27383 / 1.10889. Earlier 2,000-full-batch
exact and sampled-hybrid fits give shape RMSE 1.26510 / 1.26279, Spearman .2452 /
.2464 and ratio RMSE 1.16391 / 1.15644. Those are contextual controls, with different
optimization budgets; the preceding 200-draw joint-offset arms are matched controls.
No one-seed, three-development-parent difference establishes robust superiority.
Prediction SD need not equal target SD; matching spread by construction would
not establish predictive quality. Judge it together with error and correlation.

![Biological comparison](results/comparison.png)

The already recorded per-condition metrics show the three development parents
separately; every reported per-condition metric is independently recomputed from
the original prediction dumps before report generation. These are descriptive,
not additional checkpoint-selection criteria.

| Development condition | Observations | Analytical shape RMSE | Fragment shape RMSE | Analytical temporal-ratio RMSE | Fragment temporal-ratio RMSE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(per_condition_table)}

## Dataset and meaning of predictions

Unchanged canonical Hitlist recipe manifest
`5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615`,
MSV000092631/PXD044463: 90,630 training observations from six constructs/seven
conditions and 14,822 development observations from three other parents.
Original positive background-corrected MS signals, eligibility, instruments,
modified-form/technical contributor ambiguity and source mapping are unchanged.
Weights give equal parent, then condition, curve and common positive time.
No new negatives, censor inequalities, synthetic biological targets or curator.

Inputs are full parent sequence, digestion time, enzyme concentration context and
instrument identity. Forty coefficients encode log cut hazards using an intercept,
19 left/19 right residue contrasts and log enzyme context. Thirty-eight supported
instrument/length/composition coordinates estimate relative MS response. The
fragment arm adds one scalar eta=tanh(rho), physically bounded to (-1,1), with
hazard `exp(x@theta)*(current_fragment_length/20)^eta`. All fragment lengths 1..L
remain, including monomers/nonbinders. Outputs are expected fragment copies per
initial precursor and log10 signal = log10 expected copies + response + scale.
No normalization over observed candidates. Eta-zero uses the exact independent-cut
formula. No MHC, TCR, APC machinery KO, transport, cellular clearance or species
transfer module receives supervision here. MS response is not independently calibrated.

Development scales are solved using development labels for the existing
scale-invariant shape diagnostic. Thus this is not absolute out-of-sample signal
prediction. MSE uses hierarchical weights; correlations/spread are pooled unweighted.
Development is already inspected. Original test parents are deliberately excluded
from this optimization diagnostic, with no new blind-test claim. All train and
development predictions are retained; nM/binder metrics do not apply.

## The one change: eliminate seven nuisance optimizer variables

At each selected condition/time block, evaluate every admitted time/span in that
condition at the current parameters, and compute
`beta = sum(w*(target-base_prediction))/sum(w)`. Detach beta for the selected
probability-weighted block loss. The weighted expectation over all 28 blocks equals
the full profiled-objective gradient by the envelope identity. A single-block
finite difference must hold beta fixed; differentiating a new block-specific
optimum tests a different objective. Neither stale nor separate timewise offsets
are used. Only 78 common parameters (79 with raw eta) are allocated to Adam.
Selected differentiable values are reused detached only at the exact same state.

Identical original 40-coordinate ratio-trained initializer, zero response,
200 fixed seed-42 block draws, Adam lr .03, betas .9/.999, eps 1e-8, no clipping,
weight decay or dropout. Penalty is .005 squared nonintercept rate/response
coefficients, plus .005 eta squared for the fragment arm. No new pretraining.
Final checkpoint only; no outcome-driven extension/retry/reset or test rescue.

| Arm | Final training objective | Profiled training data MSE | Residual offset optimization gap |
|---|---:|---:|---:|
{chr(10).join(offsets)}

Initial objective was {q['state_summaries'][0]['full_loss']:.6f}.
The former joint-offset gaps were .196090 analytical and .129245 fragment MSE.
Removing that gap tests optimization, not a causal claim about fragmentation or
proof that other representation/generalization limits have disappeared.

## Verification, runtime and artifacts

Before fitting, toy and all-28-block actual analytical identities passed at the
original initializer and a predetermined sinusoidal perturbation. A toy
counterexample rejects separate timewise offsets. Actual eta-zero fragment
common-gradient parity and positive finite 2/4-step readouts passed. Every queried
condition/time is checked during fitting, the selected block is checked after
each update, and all final train/development queries are checked. Required-log
disagreement must be <=1e-5, with no clipping, floors, dropped rows or eta reset.
Tiny negative unobserved tails remain unqualified and are recorded.

Independent verification replays all 200 Adam steps per arm and every analytical
gradient. Fragment probes at fixed steps 1/100/200 independently solve whole-condition
offsets with NumPy eight-step physics and test nine fixed-offset rate/eta directions
plus all 38 response components per state. Other fragment gradients are not
independently recomputed. Every final physical amount, source-linked target/weight,
scale, training objective and global metric is independently checked. Exact scope
and numerical errors are in each `independent-verification.json`.

| Arm | End-to-end seconds | Peak RSS GiB | Materially trained / allocated parameters | Eta |
|---|---:|---:|---:|---:|
{chr(10).join(runtime)}

Local CPU float64/complex128, one thread, no GPU/Modal/environment change. Before
training, the longest-parent target-free update cost was {q['cost']['seconds']:.2f} s.
The frozen cap is {fit['runtime_limits']['seconds_per_arm']} s / 4 GiB per arm, derived
from 200 worst-parent costs with 50% margin plus 300 s evaluation and rounding.
This is a prospective cap, not an outcome-driven extension of the previous run.
Descriptive runtimes include numerical guards and artifact/evaluation work;
independent analytical verification briefly overlaps fragment fitting.

Agent/model: Codex / GPT-6; source commit a6c6559 with recorded dirty state. Exact
source/data hashes, commands and source snapshots are under `reproduce/`.
`launch.json`/`launch.sh` preserve zero-update qualification; separately frozen
`fitting.json`/`fit.sh` record the subsequent fits. To rerun, restore the recorded
source/data paths or explicitly document equivalent relocation, then run in a
fresh family copy without attempted-arm outputs. Verification commands are
`sh reproduce/verify.sh analytical` and `sh reproduce/verify.sh fragment`;
`sh reproduce/report.sh` regenerates the summary. Raw checkpoint/state/reference
arrays are archived byte-for-byte as `results/raw-training-*.tar.gz`; restore them
with `python reproduce/unpack.py`. All 105,452 predictions per completed arm remain
directly available as `results/<arm>/predictions.jsonl.gz`.

Status: terminal outcomes and independent verification recorded. The next decision
must follow these biological results; numerical precision or parameter updates
alone do not justify model promotion. Existing cellular MAPP failure, precursor
mapping and independent MS calibration remain unresolved. No new length-dependent
sampler is admitted and no APC concentration model is validated.
'''
    (FAMILY/'README.md').write_text(report)
    dump(FAMILY/'results/report-summary.json',dict(conclusion=conclusion,arms=results,previous_joint_offset_metrics={a:e['metrics'] for a,e in old.items()}))
    print(conclusion,flush=True)


if __name__=='__main__':main()
