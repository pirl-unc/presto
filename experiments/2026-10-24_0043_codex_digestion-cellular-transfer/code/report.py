"""Close the bounded cellular transfer family without outcome-driven refitting."""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import FAMILY,cfg,dump


def main():
    c=cfg();root=FAMILY.parents[1];q=json.loads((FAMILY/'results/qualification.json').read_text());suite=json.loads((FAMILY/'results/fit-suite.json').read_text());assert suite['status']=='completed'
    labels=dict(uniform_composition='Uniform observed composition',uniform_rates='Uniform cleavage',adjacent_rates='Frozen adjacent rates',wider_rates='Frozen wider rates',wider_hybrid='Frozen wider / sampled')
    arms=['uniform_composition']+[r['arm'] for r in suite['results'] if r.get('returncode')==0]
    tables=[];evaluations={};verifications=[]
    for arm in arms:
        out=FAMILY/'results'/arm;v=json.loads((out/'independent-verification.json').read_text());assert v['status']=='verified';verifications.append(v)
        e=json.loads((out/'evaluation.json').read_text());evaluations[arm]=e
        status=dict(seconds=0,peak_rss_bytes=0) if arm=='uniform_composition' else json.loads((out/'status.json').read_text())
        cp=dict(delta=None,logdelta=None) if arm=='uniform_composition' else json.loads((out/'checkpoint.json').read_text())
        row=dict(arm=arm,label=labels[arm],delta=cp['delta'],logdelta=cp['logdelta'],seconds=status['seconds'],peak_gib=status['peak_rss_bytes']/1024**3)
        for split in c['allowed_splits']:
            row.update({split+'_'+key:value for key,value in e['overall'][split].items()})
        tables.append(row)
    with (FAMILY/'results/comparison.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    summary=dict(conditions=tables,not_fitted=[r for r in suite['results'] if r.get('status')=='not_admitted'],verified_prediction_count=sum(v['predictions'] for v in verifications),verified_optimizer_updates=sum(v['optimizer_updates'] for v in verifications),qualification=q)
    dump(FAMILY/'results/report-summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
    xs=list(range(len(tables)));axes[0].bar(xs,[r['validation_rmse'] for r in tables],color=['#9ca3af','#64748b','#60a5fa','#0d9488'][:len(tables)])
    axes[0].set_xticks(xs,[r['label'].replace(' ','\n',1) for r in tables],fontsize=8);axes[0].set_ylabel('Validation centered log10 RMSE');axes[0].set_title('Captured-peptide composition; 15 parents')
    for x,r in zip(xs,tables):axes[0].text(x,r['validation_rmse'],f"{r['validation_rmse']:.3f}",ha='center',va='bottom',fontsize=9)
    for arm in arms[1:]:
        h=json.loads((FAMILY/f'results/{arm}/history.json').read_text());axes[1].plot([r['step'] for r in h],[r['training_loss'] for r in h],label=labels[arm])
    axes[1].axhline(tables[0]['train_mse'],color='#9ca3af',ls='--',label='Uniform composition');axes[1].set_xlabel('Fixed Adam update');axes[1].set_ylabel('Training centered-log10 MSE');axes[1].set_title('One learned scalar per rate model');axes[1].legend(fontsize=8)
    fig.savefig(FAMILY/'results/comparison.png',dpi=160);plt.close(fig)
    def fmt(x,d=4):return 'undefined' if x is None else f'{x:.{d}f}'
    table='| Model | Train RMSE | Validation RMSE | Validation Pearson / Spearman | Prediction SD | Delta | Seconds |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in tables:table+=f"| {r['label']} | {r['train_rmse']:.5f} | {r['validation_rmse']:.5f} | {fmt(r['validation_pearson'])} / {fmt(r['validation_spearman'])} | {r['validation_prediction_sd']:.5f} | {fmt(r['delta'],6)} | {r['seconds']:.2f} |\n"
    baseline=tables[0]['validation_rmse'];byarm={r['arm']:r for r in tables};control=byarm['uniform_rates']['validation_rmse']
    transfer=[r for r in tables if r['arm'] in ['adjacent_rates','wider_rates'] and r['validation_rmse']<min(baseline,control)]
    decision=('At least one frozen sequence prior improves both no-sequence controls on this development endpoint.' if transfer else 'Neither frozen sequence prior improves both no-sequence controls on this development endpoint.')
    decision+=' These are already inspected validation parents, not a newly blind result. No APC concentration model is promoted.'
    decision+=' All physical fits have varying outputs; wider-context validation prediction SD exceeds the target SD. The failure is not constant-output collapse.'
    parent_lines=['| Validation parent | Uniform composition | Uniform cleavage | Adjacent | Wider |','|---|---:|---:|---:|---:|']
    for p,v in evaluations['uniform_composition']['per_parent'].items():
        if v['split']!='validation':continue
        vals=[evaluations[a]['per_parent'][p]['rmse'] for a in ['uniform_composition','uniform_rates','adjacent_rates','wider_rates']]
        parent_lines.append('| '+p[:12]+' | '+' | '.join(f'{v:.5f}' for v in vals)+' |')
    text=f"""# Frozen digestion-to-cellular transfer: biological outcomes

{decision}

All three exact fits completed the fixed 2,000 updates. The sampled estimator passed
its numerical gates, but its predeclared cost gate rejected training; this family
therefore does **not** provide a fitted exact-versus-RL biological comparison.

## Observed biological prediction

{table}

![Validation comparison and fixed training](results/comparison.png)

RMSE/MSE weight parents, conditions and candidate peptides equally within each
level. Pearson/Spearman and prediction/targetSD are pooled, unweighted. Validation
target SD is {tables[0]['validation_target_sd']:.5f}. Exact constant controls have
undefined correlations. Secondary `kl` in JSON/CSV is **KL(target || prediction)**,
not the reverse KL used as primary loss in older MAPP families. Historical test
scores are not recomputed or compared as matched outcomes here.

{chr(10).join(parent_lines)}

No new sequence weights, treatment effects or peptide-specific offsets can adapt
to MAPP. These outcomes test the frozen cleavage prior plus the specified removal
and observation assumptions; failure cannot uniquely identify which is wrong.
The model predicts identical composition across UT/TNF+IFN for the same parent,
so it does not learn treatment response. See per-parent and exposure strata JSON.

## Source and endpoint contract

Agent/model: Codex / GPT-6. Source commit{c['source_commit']} with registered dirty
state. Canonical Hitlist MAPPmin2 manifest{c['dataset_manifest_sha256']}, PXD009092,
human HEK293 proteasome-associated captured fragments.72 training parents / 144 examples /
918 candidate measurements; 15 validation parents / 30 examples / 112 measurements. The
14 historical test parents are excluded from target construction, fitting and eval;
there are no test prediction dumps because those partitions were already exposed
and this prospective diagnostic explicitly excludes them. Validation also has
historical exposure, so this is no new blind biological test.

Preserve original nonimputed published quantities, MS/MS/PEP eligibility, source
run filters, coordinate/parent alternatives and contributor/material limitations.
No new curation, negatives, zero imputation, clipping or censor reinterpretation.
All targets are original log10MS intensities centered per parent/condition. No
nM/binder threshold is applicable. No pMHC/TCR/KO/species/transport labels train.
No native-peptidome workbook from Hitlist788 enters this family. This captured
pool is not free cytosol or a professional APC measurement.

The sequence-only exposure audit precedes target arithmetic and finds no exact
parent/construct containment or shared measured peptides against digestion
**training** in either split. The prior digestion development sequences are
separately flagged in `results/exposure.json`; no shared-component flag is proof
of homology or specimen independence. All transferred bond features lie in the
supported algebraic span; that does not establish biological transfer support.

## Physical model and information flow

Full parent sequence -> fixed per-bond cleavage rates -> expected amount for each
contiguous fragment under steady precursor input and common removal -> occurrence
sum -> centered log10amount -> centered observed log10MS target.

For independent-cut expectation C_s(t), F_s(delta)=integral C_s(t)exp(-delta*t)dt.
F is residence amount per unit precursor influx. All lengths 1..L are in physical
support, including short/nonbinding products; compute needed spans analytically
without enumerating the complete long-parent O(L^2) table. The residue identity
is sum length(s)F_s=L/delta. Observation normalization/centering is downstream of
physical F. Unknown peptide-specific capture/MS response is assumed equal within
the observed set; purified-study MS-response coefficients are not transferred.

Three rate models: uniform .01; frozen 40adjacent-residue/enzyme-context coefficients;
frozen 120wider-context coefficients from2358. Enzyme context fixed reference 0.
The frozen features and rates were fit on purified standard20S timecourses; prior
development evidence motivated wider context. These source models may be wrong
for cellular proteasome composition/substrate handling. Only log(delta) trains;
its units inherit an uncalibrated rate/time transfer and must not be interpreted
as a measured cellular half-life. Absolute amount additionally needs calibrated
precursor influx, removal and volume, which current data do not identify.

## Training and sampling contract

Each fit initializes delta=.01, runs 2,000 full-batch Adam updates at lr .03,
betas .9/.999, eps 1e-8, and uses equal-parent/condition/candidate centered-log10MSE.
No penalty, bound, schedule, clipping, restart, outcome-based checkpoint selection,
new pretraining or additional fitted context. All scalar optimizer states are
retained; no unallocated holistic module is claimed trained. Uniform observed
composition is an unfitted zero prediction control.

The prospective wider hybrid retains exact forward probabilities/residuals and
conditional log-derivatives. It samples complete partitions conditional on chosen
spans, including sampled ages under Exp(delta), with 90%gradient-norm/10%uniform
proposal and explicit q/p correction. Shared derivative baseline subtraction has
an exact correction. K512 per example, seed42. This is a sampled hybrid, not pure
simulator-only RL/GRPO. It passes three fixed-state gradient checks but is NOT fit:
its original cost projection 8598 seconds exceeds the 7200 second ceiling. Do not
infer an RL accuracy result or relax that recorded decision after seeing scores.

## Qualification, independent verification and resource contract

All spans for lengths 2/4/6 at delta 1e-4/.01/1/100 match independent branching-state
resolvents, numerical integration, finite differences and Torch derivatives.
Residue conservation passes. A 65,536-draw enumerated conditional-age/partition toy
passes its probability/count/mean checks and exact importance-gradient identity.
Actual initial logdelta and +/-1:16 K512 groups, relative mean errors about.00113/
.00117/.00045 and RMS.00483/.00466/.00638; all pass unchanged. None triggered the
registered absolute near-zero gate.

The first qualification fails while writing a NumPy bool after the first actual
gradient probe. Preserve original code/log/results under attempts/serialization.
Only serialization/scalar diagnostic handling and resume from ORIGINAL cost
receipts change. All original caps/decisions/seeds/settings remain; no biological
fit failed or restarted. Corrected qualification 61.58 seconds, about1.08 GiB RSS.
Exact runtime caps 690/517/525 seconds, all 4 GiB. Local CPU float64, one thread, no
Modal/GPU/package/shared-environment changes. Limited verifier overlap means
reported end-to-end durations are not isolated hardware benchmarks.

Independent verifier rebuilds source targets from original intensities, raw
sequence features and the training-derived transform, physical span amounts,
all {summary['verified_prediction_count']} final rows, global/per-parent/exposure
metrics, all {summary['verified_optimizer_updates']} Adam arithmetic updates and
four fixed exact gradient states per fit. Other gradients are not independently
recomputed. Source recipe is already canonically qualified; this consumer does
not repeat upstream workbook ingestion.

## Reproduction and decision scope

`reproduce/launch.json`, registration/plan, execution/fitting/verification receipts,
and source snapshots freeze settings, inputs, code and actual commands.
Run `sh reproduce/launch.sh` in a fresh family copy, then `sh reproduce/fit.sh`;
fitters refuse existing output directories. Verify each completed arm with
`python code/verify.py ARM`, then run`python code/report.py`. Use the pinned Python
and one-thread environment in the shell launchers; external input paths need an
explicit relocation. Raw optimizer/checkpoint JSON and compressed source-linked
prediction dumps are retained directly; no separate large-output archive.

Status: biological fits and independent verification complete.
Next step: use the result to test or improve the cellular observation/processing
bridge with independent support; do not change this family's seeds/caps/features.
Open questions: capture/MS response, cellular production/removal, precursor
abundance, machinery contrasts, independent cellular data. Full APC availability
and concentration goal remains active.
"""
    (FAMILY/'README.md').write_text(text)
    entry=f"""

## {FAMILY.name} — Frozen digestion rates tested on cellular captured-peptide composition

- **Agent/model:** Codex / GPT-6; source{c['source_commit']}, registered dirty state. [Report, source snapshots and reproduction](./{FAMILY.name}/README.md).
- **Source/curation:** PXD009092 human HEK293 proteasome-associated fragments; unchanged canonical Hitlist MAPPmin2 manifest{c['dataset_manifest_sha256']}.72 train parents / 144 examples / 918 measurements;15 historically inspected validation parents / 30 examples / 112 measurements. Original14testparents excluded; no new test metrics/dumps by prospective design. No source curator, new negatives/imputation/censor conversion or synthetic biological labels. Original source/run/parent/material limitations retained. No applicable nM/binder metrics.
- **Model/training:** Continuous precursor input plus independent cleavage/common removal, F_s=int C_s(t)exp(-delta*t)dt over all 1..Lspans. Occurrence-summed centered log10F predicts centered log10MS. Uniform rates versus frozen 40adjacent or120wider digestion coefficients; no transfer of MS-response terms. Only log(delta) learns, initialized log(.01),2,000 full-batch Adam .03/.9/.999/1e-8, equalparent/condition/candidateMSE, no penalty/clipping/selection/restarts/new pretraining. No treatment/KO/species/TCR/pMHC/transport branch trains. Uncalibratedcapture/MSresponse and imported time scale preclude concentration/half-life claims.

{table}

- **Decision:** {decision} These numbers use validation, not the older14parenttestcontract. ForwardKL(target||prediction) is secondary; older reverseKL-trained models are contextual, not matched objective controls. All train/validation predictions and global/per-parent/exposure metrics retained.
- **Sampled comparison:** Conditional-age complete-partition estimator passes enumerated toy and16 K512 groups at 3 fixed states (meanrelative<=.00118,RMS<=.00639). It retains exact forward/conditional scores and is hybrid, not simulator-onlyRL. Original target-independent projection8598s exceeds7200 s ceiling; sampled training is not launched. This family supplies no fitted RL accuracy comparison. No prospective budget extension.
- **Exposure/verification:** No digestion-training construct/peptide overlap or unsupported algebraic bond directions in either MAPPsplit; prior digestion development overlaps separately reported. No homology/specimenindependenceclaim. Independent feature-transform/source/physical/metric checks verify{summary['verified_prediction_count']}predictionrows,{summary['verified_optimizer_updates']}Adamupdates and4fixedgradients perfit. Every allocated scalar receives updates; frozen imported sequence coefficients intentionally remain frozen. Allsmall-parentresolvent/integration/finite-difference/massconservation checks pass.
- **Resources/reproducibility:** Local CPU float64, one thread, noModal/GPU/environmentchanges. Exactcaps690/517/525s,4GiB; raw optimizer/checkpointJSON,predictionGZ,source/configreceipts directly stored. Initial NumPy boolserialization failure preserved; same seed/settings resume with ORIGINALcostcaps. No biologicalfitretry. Correctedqualification61.58s,1.08GiB. Runtimes include artifacts and limited verifieroverlap, not isolatedbenchmarks. Independentcalibration, precursor/clearance, nativecellularsource788 and APCvalidation remain open; fullgoalactive.
"""
    log=root/'experiments/experiment_log.md';s=log.read_text();marker='## '+FAMILY.name+' —'
    if marker in s:
        index=s.index(marker)
        assert '\n## ' not in s[index+len(marker):], 'Later registry entries require scoped reconciliation'
        s=s[:index].rstrip()
    log.write_text(s+entry)
    print(json.dumps(summary,allow_nan=False),flush=True)

if __name__=='__main__':main()
