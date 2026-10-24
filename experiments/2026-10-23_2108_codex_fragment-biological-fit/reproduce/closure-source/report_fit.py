"""Biological outcomes first; retain the distinct optimizer contracts."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import FAMILY,dump


def main():
    cfg=json.loads((FAMILY/'reproduce/fitting.json').read_text())
    results={};rows=[]
    for arm in cfg['arms']:
        out=FAMILY/'results'/arm;state=json.loads((out/'status.json').read_text())
        assert state['status']!='running'
        if state['status']=='completed':
            verified=json.loads((out/'independent-verification.json').read_text());assert verified['all_pass']
            evaluation=json.loads((out/'evaluation.json').read_text())
            results[arm]=dict(state=state,verified=verified,evaluation=evaluation)
            for split,met in evaluation['metrics'].items():rows.append(dict(arm=arm,split=split,**met))
        else:results[arm]=dict(state=state)
    if rows:
        with (FAMILY/'results/comparison.csv').open('w') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    done={k:v for k,v in results.items() if 'verified' in v}
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    labels=['Constant signal','Time quadratic','Prior exact: 2,000 full batches','Prior hybrid: 2,000 full batches']
    scores=[1.28060,1.27383,1.26510,1.26279]
    for arm,value in done.items():labels.append(f'{arm}: 200 blocks');scores.append(value['evaluation']['metrics']['val']['shape_rmse'])
    axes[0].scatter(scores,np.arange(len(scores)),s=50,color=['#777777']*4+['#3877ac','#b15b39'][:len(done)])
    axes[0].set(yticks=np.arange(len(scores)),yticklabels=labels,xlabel='Development log10-signal shape RMSE',title='Biological prediction (lower is better)');axes[0].invert_yaxis()
    for arm in cfg['arms']:
        history=json.loads((FAMILY/'results'/arm/'history.json').read_text())
        if history:
            losses=np.array([h['pre_total_loss'] for h in history]);n=min(25,len(losses));smooth=np.convolve(losses,np.ones(n)/n,mode='valid')
            axes[1].plot(np.arange(n,len(losses)+1),smooth,label=arm)
            if arm=='fragment':axes[2].plot([h['step'] for h in history],[h['eta_after'] for h in history],color='#b15b39')
    axes[1].set(xlabel='Update',ylabel='25-block moving average',title='Sampled-block objective');axes[1].legend(frameon=False)
    axes[2].axhline(0,color='#777777',linestyle='--');axes[2].set(xlabel='Update',ylabel='Physical eta = tanh(raw eta)',title='Learned length dependence')
    fig.savefig(FAMILY/'results/comparison.png',dpi=160)
    if len(done)==2:
        av=done['analytical']['evaluation']['metrics']['val']['shape_rmse'];fv=done['fragment']['evaluation']['metrics']['val']['shape_rmse']
        conclusion=f"The length-effect arm {'improves' if fv<av else 'does not improve'} matched development shape RMSE: **{fv:.5f} versus {av:.5f}**. Its score is {'better' if fv<1.28060 else 'worse'} than the constant control (1.28060) and {'better' if fv<1.26510 else 'worse'} than the earlier longer exact fit (1.26510)."
    else:conclusion='The matched comparison is incomplete because a registered arm failed. Its failure and all completed results remain below; no model is promoted.'
    if len(done)==2:
        conclusion += ' It also trails the time-quadratic control (1.27383) and earlier hybrid fit (1.26279). This is no new best biological model.'
    table=[];optim=[];runtime=[]
    for arm,res in results.items():
        state=res['state']
        if arm not in done:
            table.append(f"| {arm} | FAILED after {state['updates']} updates | — | — | — | — |")
            continue
        v=res['evaluation'];tr=v['metrics']['train'];dv=v['metrics']['val'];vr=res['verified']
        table.append(f"| {arm} | {tr['shape_rmse']:.5f} | {dv['shape_rmse']:.5f} | {dv['shape_pearson']:.4f} / {dv['shape_spearman']:.4f} | {dv['ratio_rmse']:.5f} | {dv['prediction_sd']:.4f} / {dv['target_sd']:.4f} |")
        optim.append(f"| {arm} | {v['joint_train_objective']:.6f} | {v['joint_train_data_mse']:.6f} | {vr['training_offset_optimization_gap']:.6f} | {v['eta'] if arm=='fragment' else 0} |")
        runtime.append(f"| {arm} | {state['seconds']:.2f} | {state['peak_rss_bytes']/1024**3:.3f} | {len(state['coverage']['material_gradient_counts'])} / {len(state['coverage']['material_gradient_counts']) if state['coverage']['all_allocated_materially_trained'] else 'incomplete'} |")
    q=json.loads((FAMILY/'results/qualification-summary.json').read_text())
    text=f'''# Bounded biological fragment-length comparison

{conclusion}

This is a real biological-data fit, with a fixed 200-update matched comparison.
It does not establish APC peptide concentrations. The original MAPP failure and
independent MS calibration gaps remain unresolved.

## Outcomes

| Arm | Train shape RMSE | Development shape RMSE | Development Pearson / Spearman | Development temporal-ratio RMSE | Development prediction / target SD |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

All RMSEs above use log10 original MS signal or its same-peptide temporal ratio,
not nM. Shape MSE weights parent/condition/curve/time equally in that hierarchy;
correlations and SD are pooled unweighted observations. Development condition
scales are profiled using development labels to reproduce the existing shape
metric. These are not absolute out-of-sample signal predictions. Only three
previously inspected development parents are available; there is no new blind test.

Earlier controls on the same endpoint: constant 1.28060, response-only 1.27936,
time-quadratic 1.27383, exact kinetics+response 1.26510, sampled hybrid 1.26279.
The latter two used 2,000 full-batch updates and analytically profiled offsets;
they are contextual references, not matched optimizer-budget comparisons.

![Biological comparison](results/comparison.png)

## What was trained

Both arms use full precursor sequence, elapsed digestion time, enzyme concentration
context and source-backed instrument identity. Forty log-linear cut-rate
coefficients encode an intercept, 19 left and 19 right residue contrasts and log
enzyme context. Thirty-eight supported instrument/length/composition coordinates
model relative MS response, and seven condition offsets model nuisance scale.
The fragment arm adds exactly one scalar: eta=tanh(raw_eta), initialized 0 and
prospectively bounded to (-1,1).

The physical law is `cut hazard = exp(x@theta)*(current_fragment_length/20)^eta`.
All fragments 1..L, including monomers and nonbinders, remain in the branching
process. It predicts expected fragment copies per initial parent. The observation
model predicts log10 MS signal from log10 count + shared response + condition
scale. Counts are not normalized over observed peptides. The analytical arm fixes
eta 0 and uses the closed-form independent-cut expectation.

These are standalone experimental processing/observation models, not a fit of
all Presto modules. MHC/TCR, APC machinery KO, transport, cellular clearance and
species-transfer heads receive no supervision in this experiment. The enzyme is
purified human standard 20S; its origin is not an APC sample. Response coefficients
and offsets are nuisance estimates, not independently calibrated detectability.

## Optimization and its limits

Both arms start from the identical original 40-coefficient ratio-trained checkpoint,
zero response and weighted-optimal initial condition offsets. Fixed seed 42,
identical 200 block draws (all 28 blocks visited), Adam lr 0.03, betas 0.9/0.999,
eps 1e-8, no decay/dropout/clipping. Draw probabilities equal each condition/time
block's total training weight; divide within-block weights by that probability.
The expectation equals the full weighted objective. Add 0.01/2 squared
nonintercept rate/response coefficients once per update; the fragment arm also
penalizes physical eta with 0.01/2. Offsets are unpenalized. Final checkpoint only;
no development selection, schedule extension, reinitialization or test rescue.

| Arm | Final joint training objective | Joint training data MSE | MSE from nonoptimal offsets | Final eta |
|---|---:|---:|---:|---:|
{chr(10).join(optim)}

Initial joint objective was {q['full_loss']:.6f}, with data MSE {q['full_data_loss']:.6f}.
The offset gap is exactly sum(condition_weight * (learned_offset - optimal_offset)^2).
It diagnoses incomplete nuisance optimization; it does not by itself identify the
cause of development prediction errors. Joint finite-step SGD is not equivalent
to the earlier full-batch profiled optimizer, even though its objective and expected
gradient are coherent. Do not interpret trained coordinates or numerical precision
as predictive success.

## Data and admission

Unchanged canonical Hitlist digestion recipe manifest 5f616170… from
MSV000092631/PXD044463: 90,630 training and 14,822 development observations, same
complete-case/source mapping, modified-form/technical contributor ambiguity,
instruments, original background-corrected positive signals and hierarchical
weights. Original test parents remain excluded. No raw-data curator, new negative
labels, support filter or quantity reinterpretation was introduced. Full source
and initializer hashes are in `reproduce/launch.json`. The existing loader computes
development log targets but they are never used for initialization, schedule or
checkpoint selection. Development is already inspected data.

Before fitting, all 84 required numerical input states passed and the offset optimum,
envelope gradients and weighted minibatch identities qualified on toy and actual
training data. Independent NumPy actual-gradient error was
{q['independent_gradient_max_error']:.3g}; actual-block eta 0 common-gradient parity
error {q['actual_fragment']['common_gradient_max_error']:.3g}. The actual initial eta
gradient is {q['actual_fragment']['eta_gradient']:.6g}, so this added mechanism is
not locally disconnected. These checks were prerequisites, not biological results.

## Numerical safety, runtime and verification

Every selected fragment block is checked before and after its update with 2/4-step
CRAM48 log agreement<=1e-5, finite positive required values and no clipping/flooring.
Final evaluation checks all training/development queries. Tiny negative unobserved
tails remain recorded and unqualified. A numerical/resource failure stops that
arm without dropping rows or resetting eta. `qualified_final` in checkpoint status
means numerical validity and coordinate training support, not a biological quality
screen or concentration promotion.

| Arm | End-to-end seconds | Peak RSS GiB | Materially trained / allocated coordinates |
|---|---:|---:|---:|
{chr(10).join(runtime)}

CPU float64/complex128, one thread, no Modal/GPU or environment changes; per-arm
caps 3600 s/4 GiB. Timings include loading, numerical guards, artifact writes and final
evaluation. Analytical verification briefly overlaps fragment fitting, so these
are descriptive runtimes, not isolated hardware benchmarks.

Independent verification replays every Adam update and all analytical-arm
biological gradients, recomputes all final physical amounts using direct formulas
or the separate NumPy eight-step fragment solver, checks every source-linked
prediction and recomputes every global shape/ratio metric. Fragment gradient checks
use fixed updates 1/100/200, nine rate/eta directions and all 45 response/offset
components at those states; other fragment gradients are not independently rerun.
See each arm's `independent-verification.json` for exact errors and scope.

## Reproduction and handoff

Agent/model: Codex / GPT-6. Code source commit 120d55a with recorded dirty status; exact source and commands
under `reproduce/`. Run qualification, then the separately frozen `fit.sh` in a
fresh copy without existing attempted-arm output directories. No output overwrite
is allowed. All 200 states/gradients/optimizer moments, final checkpoint and full
train/development prediction dumps are retained under `results/<arm>/`.
`fitting.json` fixes the complete draw schedule and every nondefault setting;
`fitting-code.json` pins executed sources. `verification.json` records the independent
reference contract. Binding-threshold/nM metrics do not apply to this MS endpoint.

`launch.json` and `launch.sh` preserve the initial qualification entry point;
`fitting.json` is the separately frozen authorization/configuration for the actual
fits, executed by `sh reproduce/fit.sh`. The unchanged original launch receipt
therefore says zero updates; completed fitting receipts record all 400 updates.
Run `sh reproduce/verify.sh analytical`, `sh reproduce/verify.sh fragment`, then
`sh reproduce/report.sh` to reproduce closure. Source paths are pinned to the
recorded worktree and Hitlist package; recreate those paths or explicitly record
an equivalent relocation before rerunning. Required external source files are
hash checked. No independent MS calibration, concentration targets, IC50 labels,
inequalities or synthetic biological labels were added.

Raw arrays/checkpoints are stored byte-for-byte in `results/raw-training-*.tar.gz`.
Run `python reproduce/unpack.py` to restore and verify original paths. Full
prediction dumps remain directly available as `results/<arm>/predictions.jsonl.gz`.
`reproduce/closure-hashes.json` pins the published evidence bundle.

Next experiment: [solve condition offsets exactly at every current parameter
state](../agents/codex/plans/fragment-profiled-minibatches.md), retaining the same
physical models, 200 draws, optimizer settings and original data. This targets
an observed training optimization gap; it does not assume the gap explains weak
development prediction. No new RL fit is admitted by this comparison.

Status: both arm outcomes and independent verification are recorded; publication
receipt is separate. Numerical precision, optimization, biological prediction and
concentration calibration remain separate claims. Next decision must follow these
biological outcomes and the explicit offset gap; no further sampler tuning is
justified by a numerical pass alone. Full APC availability goal remains active.
'''
    (FAMILY/'README.md').write_text(text)
    dump(FAMILY/'results/report-summary.json',dict(conclusion=conclusion,arms=results,old_references=dict(constant=1.28060,exact_2000_fullbatch=1.26510,hybrid_2000_fullbatch=1.26279)))
    print(conclusion,flush=True)


if __name__=='__main__':main()
