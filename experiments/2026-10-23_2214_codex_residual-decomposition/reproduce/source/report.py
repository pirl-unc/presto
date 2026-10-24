"""Report the arithmetic finding without assigning biological causality."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FAMILY=Path(__file__).resolve().parents[1]


def main():
    s=json.loads((FAMILY/'results/summary.json').read_text());v=json.loads((FAMILY/'results/independent-verification.json').read_text());assert s['all_pass'] and v['all_pass']
    names=['constant','joint_analytical','profiled_analytical','joint_fragment','long_exact','long_hybrid']
    labels=['Constant condition scale','Analytical: joint offsets, 200','Analytical: exact offsets, 200','Fragment: joint offsets, 200','Earlier exact: 2,000 full batches','Earlier hybrid: 2,000 full batches']
    rows=[dict(model=n,**s['models'][n]['overall']) for n in names]
    with (FAMILY/'results/comparison.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    between=np.array([r['persistent_mse'] for r in rows]);within=np.array([r['temporal_mse'] for r in rows]);positions=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');ax.barh(positions,between,label='Persistent curve-mean residual',color='#376b94');ax.barh(positions,within,left=between,label='Within-curve temporal residual',color='#c67e42')
    ax.set(yticks=positions,yticklabels=labels,xlabel='Training MSE in log10 MS signal',title='Where the fixed models miss the training measurements');ax.invert_yaxis();ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=1);fig.savefig(FAMILY/'results/decomposition.png',dpi=160)
    old=s['models']['joint_analytical']['overall'];new=s['models']['profiled_analytical']['overall']
    delta={key:new[key]-old[key] for key in ['total_mse','persistent_mse','temporal_mse']}
    table='\n'.join(f"| {label} | {r['total_mse']:.6f} | {r['persistent_mse']:.6f} | {r['temporal_mse']:.6f} | {100*r['persistent_fraction']:.2f}% |" for label,r in zip(labels,rows))
    parent_fractions=[r['persistent_fraction'] for r in s['models']['profiled_analytical']['constructs'].values()]
    conclusion=f"Persistent curve-mean residuals account for **{100*new['persistent_fraction']:.2f}%** of the profiled analytical model's training MSE. Solving condition offsets changes persistent MSE by **{delta['persistent_mse']:+.6f}** and temporal MSE by **{delta['temporal_mse']:+.6f}** relative to the matched joint-offset fit."
    report=f'''# Training residuals: persistent curve means dominate remaining error

{conclusion}

This is a posthoc arithmetic diagnostic of completed, independently verified
fits. It is not another model, predictive evaluation, biological noise ceiling,
MS calibration, or evidence of APC concentrations. The live profiled-fragment
fit is deliberately excluded; no running checkpoint or development target is
used to select this analysis.

| Frozen model | Total MSE | Persistent curve-mean MSE | Within-curve temporal MSE | Persistent share |
|---|---:|---:|---:|---:|
{table}

The new analytical fit's total training improvement comes from temporal error
reduction; persistent curve-mean error slightly worsens. The longer exact/hybrid
fits still have about 76% persistent error. This is not concentrated in one
construct: the new analytical model's persistent share ranges from
{100*min(parent_fractions):.2f}% to {100*max(parent_fractions):.2f}% across all six
training constructs. Original condition/construct summaries are retained.

![Training MSE decomposition](results/decomposition.png)

## Exact meaning of the two components

For each original curve, let residual r_i be the condition-centered predicted
minus measured log10 MS signal; retain original weight w_i. Let W_c be the sum of
weights and m_c the weighted mean residual in that curve. Then

`total MSE = sum_c W_c*m_c^2 + sum_i w_i*(r_i-m_c)^2`.

The cross term is zero. The first component is a mismatch persistent across the
curve's observed times; the second changes within that curve. No added curve
intercept is fitted into the predictive model. Curves include source biological
labels, and source molecular-form/technical ambiguity persists. They are not
necessarily independent unique molecular peptides.

**These are not processing-versus-detectability components.** Wrong kinetic
amplitude, source preparation, MS response and contributor effects can all cause
persistent errors; time-varying measurement effects can cause temporal errors.
The decomposition does not identify a causal mechanism or a noise ceiling.

## Data, comparisons and boundaries

All six model comparisons use the same 90,630 original training measurements across
23,054 curves, seven conditions and six constructs. Canonical Hitlist manifest
`5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615`,
MSV000092631/PXD044463 purified human standard 20S digestion. Original positive
corrected MS values, target units, row identities, instrument labels, source
contributors and equal parent/condition/curve/time weights are unchanged. No
negative labels, censor reinterpretation, new biological synthesis or curator.

Four prior fixed fits and the completed profiled analytical fit are compared
with the constant condition-scale control. The 200-update fits share draw
schedule/initializer/Adam settings; the earlier exact/hybrid used 2,000 full
batches and are contextual references. This diagnostic performs **zero optimizer
or pretraining updates** and adds no assay loss. Original signal maps to each
existing model's saved condition-centered log10-signal prediction. No affinity,
nM, binding-threshold, MHC/TCR or APC machinery outcomes are evaluated here.

Complete input JSONL files include development records. They are parsed and
skipped by their original split before quantities are consulted; all reported
arithmetic uses training only. Original test is excluded. There are no new
validation/test predictions or metrics because this is a training residual
diagnostic, not generalization evidence. Existing full prediction dumps, pinned
by path/hash, remain the per-example evidence; no model outputs are overwritten.

## Verification, resources and reproduction

Stable Python sums produce the original curve records. A separate NumPy
bincount implementation reads every original prediction, independently verifies
all measurement identities/targets/weights across files and reconstructs all
138,324 curve records, condition/construct summaries and fractions. Maximum
absolute component disagreement is {v['maximum_absolute_component_error']:.3g}.
A nonuniform-weight toy checks both components and rejects the unweighted-mean
substitution. Producer {s['seconds']:.2f} s, peak RSS {s['peak_rss_bytes']/1024**3:.3f}
GiB; independent verifier {v['seconds']:.2f} s. Local CPU, no GPU/Modal or environment
change; the producer stayed within the registered 120 s / 4 GiB cap. These are
arithmetic runtimes, not model-training speed comparisons.

Agent/model Codex / GPT-6; source a6c6559 with recorded dirty state. Frozen plan,
exact source hashes, command/environment and source snapshots are in `reproduce/`.
Run `sh reproduce/launch.sh` only in a fresh result directory; then run
`python code/verify.py` and `python code/report.py` using the recorded shared
Python environment and one-thread settings. Source paths include the completed
analytical arm of the still-live 2201 family; original hashes must match. The
source-frozen producer refuses an existing summary. `results/curve-residuals.csv.gz`
contains every curve's original identifiers, weights and component sums;
`results/summary.json` and `comparison.csv` hold the complete reductions.

## Decision

Finish the live paired fragment fit unchanged. Then require the next hypothesis
to address persistent cross-curve signal error as well as temporal behavior.
Do not assume a more elaborate trajectory sampler will solve the dominant error,
or silently add per-peptide nuisance offsets that erase abundance information.
A more expressive response or cleavage context can be a predictive hypothesis,
but independent known-load/form/transfer data are still required to interpret it
as detectability or availability. The existing Hitlist calibration and precursor
requests remain relevant; this diagnostic finds no new missing source column.
No causal attribution, new biological winner or APC concentration claim follows.
Full availability goal remains active.
'''
    (FAMILY/'README.md').write_text(report)
    (FAMILY/'results/report-summary.json').write_text(json.dumps(dict(conclusion=conclusion,profiled_minus_joint=delta,all_pass=True),indent=2)+'\n')
    print(conclusion)


if __name__=='__main__':main()
