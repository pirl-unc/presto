"""Report the complete preregistered amplitude diagnostic and its stop decision."""
import csv
import json
import hashlib
import platform
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from common import FAMILY,dump,read,sha

def main():
    c=read(FAMILY/'reproduce/launch.json');s=read(FAMILY/'results/summary.json');v=read(FAMILY/'results/independent-verification.json')
    assert s['status']=='completed' and v['status']=='verified'
    for receipt in ['execution-code','verification']:
        for p,h in read(FAMILY/f'reproduce/{receipt}.json')['files'].items():assert sha(FAMILY/p)==h,p
    for p,h in c['source_hashes'].items():assert sha(p)==h,p
    labels=dict(uniform_composition='Uniform composition',uniform_rates='Uniform cleavage',adjacent_rates='Adjacent sequence rates',wider_rates='Wider sequence rates')
    table='| Model | Beta | Train RMSE | Validation RMSE | Validation Pearson / Spearman | Validation prediction SD |\n|---|---:|---:|---:|---:|---:|\n';records=[]
    for arm,m in s['metrics'].items():
        base=arm.removesuffix('_calibrated');calibrated=arm.endswith('_calibrated');beta=s['fits'][base]['beta'] if calibrated else 1.
        name=labels[base]+(' + calibration' if calibrated else '')
        row=dict(model=arm,label=name,beta=beta)
        for split,metric in m.items():row.update({split+'_'+k:value for k,value in metric.items()})
        records.append(row)
        p=m['validation']['pearson'];r=m['validation']['spearman'];correlation='undefined' if p is None else f'{p:.4f} / {r:.4f}'
        table+=f"| {name} | {beta:.6f} | {m['train']['rmse']:.5f} | {m['validation']['rmse']:.5f} | {correlation} | {m['validation']['prediction_sd']:.5f} |\n"
    with (FAMILY/'results/comparison.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);arms=c['arms'];x=np.arange(3)
    axes[0].bar(x-.18,[s['metrics'][a]['validation']['rmse'] for a in arms],.36,label='Original',color='#94a3b8')
    axes[0].bar(x+.18,[s['metrics'][a+'_calibrated']['validation']['rmse'] for a in arms],.36,label='Train-only calibration',color='#0d9488')
    axes[0].axhline(s['metrics']['uniform_composition']['validation']['rmse'],ls='--',color='#111827',label='Uniform composition')
    axes[0].set_xticks(x,['Uniform cleavage','Adjacent rates','Wider rates'],fontsize=9);axes[0].set_ylabel('Validation centered log10 RMSE');axes[0].legend(fontsize=8);axes[0].set_title('Calibration helps; uniform cleavage wins')
    for a in arms:
        p=s['comparisons'][a+'_calibrated']['uniform_rates_calibrated']['per_parent']
        axes[1].plot(np.arange(len(p)),[r['difference'] for r in p],marker='o',ms=3,label=labels[a])
    axes[1].axhline(0,color='#111827',lw=.7);axes[1].set_ylabel('MSE difference vs calibrated uniform cleavage');axes[1].set_xlabel('Validation parent (fixed hash order)');axes[1].set_title('Negative favors the compared model');axes[1].legend(fontsize=8)
    fig.savefig(FAMILY/'results/comparison.png',dpi=160);plt.close(fig)
    stop=s['stop_extending_same_mapp_bridge'];assert stop
    context='The calibrated uniform-cleavage control wins. Neither imported sequence prior beats both controls, so the prospective stop rule is triggered: stop extending this bridge on the same MAPP development set.'
    text=f"""# Train-only amplitude calibration of frozen cellular predictions

{context}

Uniform-cleavage calibration reduces validation RMSE from 0.36972 to **0.26711**,
versus 0.28603 for uniform observed composition (about 6.6% lower RMSE).
This is modest, nonconstant prediction on previously inspected cellular data.
It is not evidence that digestion-trained sequence preferences transfer, nor that
physical peptide amounts or APC concentrations have been validated.

## Complete outcomes

{table}

![All calibration conditions and parent effects](results/comparison.png)

The calibrated uniform-cleavage control improves 10 of 15 validation parents over
uniform composition, worsens 4 and ties 1. Its weighted R2 relative to uniform is
0.12793; prediction SD is 0.13255 versus target SD 0.35130. The adjacent model loses
to that control on 9 parents, and the wider model loses on 11. Wider calibration
still slightly loses to uniform composition overall. All beta values are positive;
no boundary/zero-denominator case occurs. A positive scalar preserves each model's
rank ordering: this diagnostic fixes amplitude, not peptide ordering.

MSE/RMSE and R2 use equal parent, condition and candidate weighting. Correlations
and SD are pooled and unweighted. Constant-control correlations are undefined.
Secondary forward KL is KL(target || prediction), not the reverse KL objective
in older MAPP fits. Per-parent and exposure-stratified metrics and full predictions
are retained for all seven conditions. Original test scores are not revisited.

## Inputs, targets and data scope

Agent/model: Codex / GPT-6. Source commit `{c['source_commit']}` with recorded dirty
state. This family consumes the verified prediction dumps from
`2026-10-24_0043_codex_digestion-cellular-transfer`, unchanged canonical Hitlist
MAPPmin2 manifest `df78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80`.
Source: PXD009092, human HEK293 proteasome-associated captured fragments.
Training: 72 parents, 144 parent/condition examples, 918 candidate measurements.
Validation: 15 parents, 30 examples, 112 measurements. Both original conditions
UT and TNF+IFN are retained, with unresolved specimen/run independence.

Targets remain original source log10 MS intensities centered within each example.
All source quantities, eligibility, mapping alternatives and measurement IDs are
unchanged. No new source curation, negatives, synthetic biological labels, censor
conversion, imputation or clipping. This is a captured pool, not free cytosol or
a professional APC validation. No nM/binder-threshold metrics apply.

The 14 historical test parents are excluded from fitting and evaluation by the
prospective plan; hence no test prediction dumps or new test metrics. Validation
also has historical exposure. Calibration was proposed after examining the earlier
transfer failure, so the result is developmental evidence, not a fresh blind test.
Prior source-overlap and feature-support flags are preserved without claiming
homology or specimen independence.

## Model, fitting and meaning

The physical models are entirely frozen: full parent sequence -> uniform or
purified-digestion-trained cleavage rates -> continuous precursor input/common
removal -> occurrence-summed centered log10 residence amount p. The earlier fitted
removal parameter and all 40/120 imported sequence coefficients remain unchanged.
No purified-study MS-response coefficients transfer. Uniform cleavage still uses
span length, parent boundaries and occurrence multiplicity; it has no learned
amino-acid preference.

For each physical arm, fit exactly one nonnegative scalar using training only:

    beta = max(0, sum(w*p*y) / sum(w*p*p))
    empirical prediction = beta*p

No intercept, penalty, optimizer, hyperparameter search, new pretraining or
validation-derived calibration. The original unscaled predictions and unfitted
uniform composition remain controls. Signed unconstrained coefficients and all
training cross moments are saved. New parameters learned: three scalar beta
values. No physical rates, APC machinery, species, treatment effect, transport,
MHC or TCR component trains. This diagnostic performs no RL/sampled fitting.

The multiplier is an empirical observation-link correction, not an independently
identified detector efficiency or biological rate. It changes predicted captured
MS differences without validating the underlying physical amount distribution.
All models retain zero predicted UT/TNF+IFN change for an identical parent.
Absolute concentration, precursor influx/abundance and cellular clearance remain
unidentified. The unchanged broader goal still requires those biological tests.

## Independent verification and resources

The independent verifier rebuilds **{v['input_source_predictions_rebuilt']} input
predictions** from canonical source targets/weights and the pinned independent
physical reference, including sequence features and their training transforms.
It obtains beta through weighted least squares and checks the nonnegative KKT
condition; maximum coefficient difference is 1.12e-16. All **{v['output_predictions_verified']}
output predictions**, global/per-parent/exposure metrics and the stop decision
verify; maximum prediction discrepancy is {v['maximum_prediction_error']:.3g}.
The prior independent physical implementation is reused, not a newly written
third physical solver. Original workbook curation is not repeated downstream.

Local CPU, one-thread NumPy/SciPy, unchanged shared environment; no Modal or GPU.
Producer {s['seconds']:.3f} seconds, peak {s['peak_rss_bytes']/1024**2:.2f} MiB, below
the prospectively declared 60-second/2-GiB budget. Independent verification
{v['seconds']:.3f} seconds. No failed scientific run, retry or outcome-based change.

## Reproduction and next action

`reproduce/launch.json`, `plan.md`, execution/verification receipts and source
snapshots freeze inputs, settings, source hashes and exact invocation. Run
`sh reproduce/launch.sh` in a fresh family copy, then the pinned Python with
`code/verify.py` and `code/report.py`. Use the recorded one-thread environment.
The producer refuses an existing summary; explicit source-path relocation is
needed outside this workspace. Raw coefficient/moment JSON and compressed
per-example train/validation dumps are directly retained.

Status: completed and independently verified; the same-MAPP transfer stop rule is
in force. Keep calibrated uniform cleavage as a provisional development control,
not a promoted APC model or replacement for the historical test-set baseline.
Next: canonical native cellular peptidomics admission under Hitlist788 and broader
independent MS-response/precursor/machinery support. Do not launch another sampler
extension or tune these sequence priors against the same development parents.
Open questions: processing, capture/detection, production/clearance, precursor
abundance, perturbation response and independent cellular validation.
"""
    (FAMILY/'README.md').write_text(text)
    entry=f"""

## {FAMILY.name} — Calibration helps the non-sequence control; stop the MAPP transfer bridge

- **Date / agent:** {c['registered_at']}; Codex / GPT-6; source `{c['source_commit']}`, recorded dirty state. [Report and frozen reproduction bundle]({FAMILY.name}/README.md).
- **Source / curation:** Unchanged canonical MAPPmin2, PXD009092 human HEK293 captured proteasome fragments; manifest `df78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80`. Verified 0043 predictions: 72 train parents/144 examples/918 rows, 15 historically inspected validation parents/30 examples/112 rows. Original 14 test parents explicitly excluded: no new test metrics/dumps. No new curator, filtering, negatives, synthetic labels, censor reinterpretation or source independence claim. No applicable affinity/binder metrics.
- **Training / outputs:** All physical rates and prior removal parameters frozen. Fit one nonnegative scalar beta per physical arm by exact training-weighted least squares, no optimizer/pretraining/penalty/search or validation calibration. `beta * centered log10 residence amount` maps to original centered log10 captured-MS intensity. Three new scalar parameters; no physical/APC/KO/treatment/species/MHC/TCR/transport training. Original unscaled and uniform composition controls retained; no sampled fit in this analytic diagnostic.

{table}

- **Decision:** {context} Uniform calibration improves validation RMSE about 6.6% versus uniform composition and helps 10/15 parents (4 worse,1 tie); this is modest geometry-based captured-signal prediction. Adjacent loses to calibrated uniform on 9 parents, wider on 11. Positive beta preserves ranking; no proof of detector calibration, physical abundance or APC concentrations. Validation selection history remains explicit. Keep the best control provisional for this development contract only, not the older test contract.
- **Metrics:** Hierarchical-weighted MSE/RMSE/R2, pooled Pearson/Spearman/SD, forward KL(target||prediction), per-parent/exposure strata and all train/validation predictions retained. Calibrated uniform validation R2 .12793 and predictionSD .13255 versus targetSD .35130. No new untouched test.
- **Verification / runtime:** Independently reconstruct 4,120 input source/physical predictions using the pinned prior reference; weighted least-squares/KKT verify all3 beta values; all7,210 output rows, metrics/strata and stop decision replay (max prediction error1.78e-15). Producer {s['seconds']:.3f}s/{s['peak_rss_bytes']/1024**2:.2f}MiB, verifier {v['seconds']:.3f}s; declared60s/2GiB. Local CPU one-thread, no Modal/GPU/environment change or failed-run retry. Source/config receipts, coefficients/cross moments, plot/CSV and compressed predictions directly stored.
- **Next:** Enforce the stop rule and prioritize canonical native cellular source788, independent response95/762 and precursor/machinery evidence. Full APC availability/concentration goal remains active; no further tuning of this bridge on the same MAPP development set.
"""
    log=FAMILY.parents[1]/'experiments/experiment_log.md';existing=log.read_text();marker='## '+FAMILY.name+' —'
    if marker in existing:
        i=existing.index(marker);assert '\n## ' not in existing[i+len(marker):];existing=existing[:i].rstrip()
    log.write_text(existing+entry)
    dump(FAMILY/'results/environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,environment=c['environment']))
    for p in (FAMILY/'code').glob('*.py'):shutil.copy2(p,FAMILY/'reproduce/source'/p.name)
    dump(FAMILY/'results/closure.json',dict(all_pass=True,source_inputs_rebuilt=v['input_source_predictions_rebuilt'],output_predictions_verified=v['output_predictions_verified'],new_scalar_fits=3,physical_optimizer_updates=0,sampled_fit=False,stop_rule_enforced=True,goal_complete=False))
    files={str(p.relative_to(FAMILY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FAMILY.rglob('*')) if p.is_file() and not {'__pycache__','.ruff_cache'}&set(p.parts) and p.name!='closure-hashes.json'}
    dump(FAMILY/'reproduce/closure-hashes.json',dict(files=files));print(json.dumps(dict(closure_files=len(files),decision=context)))
if __name__=='__main__':main()
