"""Report the predeclared sequence-context factorial and sampled comparison."""
import csv
import json
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import FAMILY,dump


def main():
    cfg=json.loads((FAMILY/'reproduce/launch.json').read_text());features=json.loads((FAMILY/'results/features.json').read_text())
    arms=cfg['exact_arms']+['both_hybrid'];results={};rows=[];table=[];runtime=[]
    labels={'base':'Adjacent cleavage + composition response','rate_context':'Wider cleavage context','response_termini':'Peptide termini response','both':'Both feature additions','both_hybrid':'Both, sampled hybrid'}
    for arm in arms:
        out=FAMILY/'results'/arm;s=json.loads((out/'status.json').read_text());assert s['status']=='completed'
        v=json.loads((out/'independent-verification.json').read_text());assert v['all_pass'];e=json.loads((out/'evaluation.json').read_text())
        results[arm]=dict(status=s,evaluation=e,verification=v)
        for split,m in e['metrics'].items():rows.append(dict(arm=arm,split=split,**m))
        t=e['metrics']['train'];d=e['metrics']['val']
        table.append(f"| {labels[arm]} | {e['parameters']} | {t['shape_rmse']:.5f} | {d['shape_rmse']:.5f} | {d['shape_pearson']:.4f} / {d['shape_spearman']:.4f} | {d['ratio_rmse']:.5f} |")
        runtime.append(f"| {arm} | {s['seconds']:.2f} | {s['peak_rss_bytes']/1024**3:.3f} | {e['all_allocated_materially_trained']} |")
    with (FAMILY/'results/comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    with gzip.open(FAMILY/'results/base/predictions.jsonl.gz','rt') as stream:
        parent_labels={}
        for line in stream:
            row=json.loads(line)
            parent_labels.setdefault(row['construct_id'],set()).add(row['condition_id'].rsplit(':',1)[-1])
    parent_table=[]
    for parent,r in results['base']['evaluation']['per_parent'].items():
        if r['split']!='val':continue
        values=[results[a]['evaluation']['per_parent'][parent]['shape_rmse'] for a in arms]
        parent_table.append('| '+', '.join(sorted(parent_labels[parent]))+' | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    residual_table=[]
    for arm in arms:
        t=results[arm]['evaluation']['metrics']['train'];v=results[arm]['evaluation']['metrics']['val']
        residual_table.append(f"| {arm} | {t['persistent_mse']:.5f} | {t['temporal_mse']:.5f} | {v['persistent_mse']:.5f} | {v['temporal_mse']:.5f} |")
    base=results['base']['evaluation']['metrics']['val'];wide=results['rate_context']['evaluation']['metrics']['val'];both=results['both']['evaluation']['metrics']['val'];sample=results['both_hybrid']['evaluation']['metrics']['val']
    conclusion=f"Wider cleavage context changes development shape RMSE from {base['shape_rmse']:.5f} to {wide['shape_rmse']:.5f}, and Spearman from {base['shape_spearman']:.4f} to {wide['shape_spearman']:.4f}. Combining both additions gives {both['shape_rmse']:.5f}; the matched conditional sampled hybrid gives {sample['shape_rmse']:.5f}."
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    score=[results[a]['evaluation']['metrics']['val']['shape_rmse'] for a in arms]
    axs[0].scatter(score,np.arange(len(arms)),color=['#555555','#24799a','#b47a24','#377842','#965b9a']);axs[0].axvline(1.28060,color='#888888',linestyle='--',label='Constant control');axs[0].set(yticks=np.arange(len(arms)),yticklabels=[labels[a] for a in arms],xlabel='Development log10 MS signal-shape RMSE');axs[0].invert_yaxis();axs[0].legend(frameon=False,loc='lower right')
    for arm in arms:
        history=json.loads((FAMILY/'results'/arm/'history.json').read_text());axs[1].plot([r['step'] for r in history],[r['preupdate_loss'] for r in history],label=arm)
    axs[1].set(xlabel='Fixed full-batch updates',ylabel='Training objective');axs[1].legend(frameon=False,fontsize=8)
    fig.savefig(FAMILY/'results/comparison.png',dpi=160)
    q=json.loads((FAMILY/'results/qualification.json').read_text())
    with gzip.open(FAMILY/'results/base/predictions.jsonl.gz','rt') as stream:
        development_conditions={r['condition_id'] for line in stream if (r:=json.loads(line))['split']=='val'}
    extrap=[]
    for kind in ['rate','response']:
        f=features[kind];val={c:r for c,r in f['extrapolation'].items() if c in development_conditions}
        extrap.append(dict(kind=kind,added=f['added_dimensions'],absent_raw_columns=f['absent_raw_columns'],all_condition_extrapolation=f['extrapolation'],development_conditions=val))
    dump(FAMILY/'results/report-summary.json',dict(conclusion=conclusion,arms=results,features=extrap))
    text=f'''# Sequence context and peptide-terminal response on biological digestion

{conclusion}

This is development evidence on three already inspected proteins, not a new blind
biological test or a calibrated intracellular-concentration model. All five
predeclared fits are retained. No architecture, checkpoint, penalty or seed was
selected from development outcomes. The full availability goal remains active.

## Actual biological outcomes

| Model | Parameters | Train shape RMSE | Development shape RMSE | Development Pearson / Spearman | Development temporal-ratio RMSE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Constant-condition scale: development shape RMSE1.28060, temporal-ratio RMSE1.16704.
Prior adjacent-context exact/hybrid shape RMSE1.26510/1.26279. The new base arm
reproduces the old exact result. The previous fragment-length model gave1.26272
with a different200-minibatch budget. These comparisons do not make the current
one-seed result robust across biological studies. Every per-example prediction
is retained; MSE is hierarchical weighted, correlations and SD are pooled/unweighted.

![Biological results and fixed-budget training](results/comparison.png)

| Development parent | Base | Wider rates | Terminal response | Both exact | Both hybrid |
|---|---:|---:|---:|---:|---:|
{chr(10).join(parent_table)}

The error decomposition keeps original source-curve identities. Persistent means
are not pure MS effects, and temporal residuals are not pure processing effects.
No per-peptide fitted offset is added to erase between-peptide information.

| Model | Training persistent MSE | Training temporal MSE | Development persistent MSE | Development temporal MSE |
|---|---:|---:|---:|---:|
{chr(10).join(residual_table)}

## Interpretation of this factorial

The wider-rate-context arm is the preferred next representation hypothesis: it
improves both development shape and temporal error. Terminal response additions
improve training fit but worsen development shape error, both with and without
wider rates. Do not add those response features to the preferred model merely
because the full-feature model is more flexible. Wider rates improve all three
development parents, but most absolute improvement comes from H4; the other
changes are small. No confidence interval or independent-study replication is
claimed. Current results motivate a prospectively specified transfer test from
purified-digestion rate features to cellular MAPP, retaining the latter's capture/
observation and total-yield limitations, rather than another sampler extension.

## Dataset and prediction meaning

Unchanged canonical Hitlist manifest
`5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615`,
MSV000092631/PXD044463 purified human standard20S digestion. Same90,630training
observations from six constructs/seven conditions and14,822development observations
from three other parents. Original positive background-corrected log10MS signals,
source identities, contributor/form ambiguity, instruments, sequence mapping and
parent/condition/curve/time weights remain. No new filters, negatives, censor
conversion, synthetic biological labels or parallel source curator.

Original test parents stay excluded: this is a representation diagnostic using
already inspected development parents, with no untouched-test claim. Development
condition scales are solved from development labels to evaluate relative signal
shape. Absolute signal, total yield and peptide concentration are not identified.
All train/development observations are preserved; binding thresholds/nM do not apply.

Inputs are full parent sequence, candidate span, time, enzyme context and
instrument. Analytical independent-cut expectations represent copies per initial
precursor for all lengths1..L, including monomers/nonbinders. Log10copies plus
relative response plus one profiled condition scale predicts the original MS
signal. No normalization over observed candidates. No new APC species/cell/KO,
transport, clearance, pMHC or TCR labels train; these modules are not allocated.
Independent MS response calibration and cellular validation remain unresolved.

## Nested representations and fixed training

Keep the original40rate coefficients and38instrument-specific length/composition
response coordinates. Add residue indicators at bond positionsi-2,i-1,i+2,i+3,
wherei/i+1 are already modeled, including explicit parent-boundary indicators.
There are80raw extra rate columns. Add first/last peptide residue contrasts per
instrument to response; external parent flanks never enter that response layer.
These are small linear sequence models, not a learned neural encoder.

Residualize new features against old features using training sequences/metadata
only; center response contrasts within condition. Normalize by training RMS and
retain eigenvalues>max(largest,1)*1e-10. New allocated dimensions: rates
{features['rate']['added_dimensions']}, response{features['response']['added_dimensions']}.
Unsupported/dependent raw directions are removed before training. All transforms,
names/eigenvalues and omitted/out-of-training-span directions are preserved in
`results/features.json`. Held-out directions outside the supported span are
projected away, an explicit prediction limitation rather than serving qualification.
Zero additions reproduce the old model for every training/development query.

Each arm uses the same original ratio-trained40rate initializer, zero response and
zero added rates, 2,000full-batch Adam updates, lr.03, betas.9/.999, eps1e-8. Loss is
weighted condition-centered MSE plus .005squared coefficients excluding the kinetic
intercept. No dropout, clipping, weight decay, schedule, new pretraining or
outcome-driven retry/extension. Final checkpoint only. All allocated-coordinate
material gradients/updates are reported, without claiming causal identification.

The fifth arm uses the full both representation, selected before results, and
replaces only kinetic data gradients with the established K512conditional complete-
partition importance estimator (seed42,90%gradient-norm proposal/10%uniform).
Exact probabilities, residuals, conditional scores and response gradients remain.
This is a sampled hybrid; it is not a pure simulator-only RL model and has no
score-saving claim. No new GRPO/clipping/reuse/reference-KL change is made.

## Qualification, verification and runtime

Independent raw feature indexing, exact nesting, old-objective equality and NumPy
value/Jacobian checks pass at initialization and predetermined .02sin perturbations.
Enumerated conditional-mixture toy identity passes. Sixteen K512groups at each
fixed full-model state pass mean-relative-error<=.2 and RMS<=.5, without sampler
tuning. Pre-fit measured costs admit600s per exact arm and1800s hybrid,4GiB each.
Qualification took{q['seconds']:.2f}s. The initial source-freeze invocation used the
wrong relative directory and failed before qualification/fitting; its log and the
invocation-only correction are preserved. No executed biological fit was replaced.

Independent closure rebuilds new raw sequence features and final NumPy physical
expectations, source IDs/targets/weights, all global/per-parent metrics and both
residual components. It replays all2,000Adam arithmetic updates per arm and four
fixed gradient states (1/100/1000/2000). Hybrid replay uses independent NumPy
conditional score/importance arithmetic with PyTorch RNG draws. Other gradients
are not independently replayed; the pinned source loader is reused.

| Arm | End-to-end seconds | Peak RSS GiB | All allocated coefficients materially trained |
|---|---:|---:|---:|
{chr(10).join(runtime)}

Local CPU float64, one thread, no Modal/GPU/shared-environment changes. Descriptive
runtimes include artifact/evaluation work and limited overlap with independent
verification, not isolated benchmark comparisons. Scientific interpretation must
follow biological metrics, not numerical gates or parameter coverage.

## Reproduction and decision scope

Agent/model: Codex / GPT-6; source commit{cfg['source_commit']} with recorded dirty
state. `reproduce/plan.md`, `launch.json`, `execution-code.json`, source snapshots,
`fitting.json` and `verification.json` preserve exact contracts and hashes.
Run `sh reproduce/launch.sh` for qualification; fitting command/environment is
frozen in`reproduce/fitting.json` and invoked through`code/run_fits.py` in a fresh
family copy. Fitters refuse existing arm destinations. Run`sh reproduce/verify.sh
ARM` and`sh reproduce/report.sh` for closure. Absolute input paths require the
pinned preceding families/Hitlist recipe, or an explicitly documented relocation.
Raw checkpoints/updates are archived with a byte-verifying restore helper; direct
`results/ARM/predictions.jsonl.gz` files retain105,452source-linked rows per arm.

Do not promote a general APC concentration model from these scores. This family
tests whether ordered sequence representation warrants further study; independent
protein/study transfer, calibrated observation response and cellular production/
clearance remain required. Separate literature discovery filed [Hitlist788](https://github.com/pirl-unc/hitlist/issues/788)
for native intracellular peptidome perturbation panels. No new source workbook was
admitted or used in these fits; its censored ratios and cross-paper reuse require
canonical qualification. Existing calibration/precursor/intensity issues retain ownership.
'''
    (FAMILY/'README.md').write_text(text);print(conclusion,flush=True)


if __name__=='__main__':main()
