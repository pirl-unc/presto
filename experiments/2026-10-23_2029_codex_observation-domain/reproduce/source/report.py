"""Report the registered numerical family without interpreting it as prediction."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAMILY=Path(__file__).resolve().parents[1]


def main():
    summary=json.loads((FAMILY/'results/summary.json').read_text())
    cfg=json.loads((FAMILY/'reproduce/launch.json').read_text())
    rows=list(csv.DictReader((FAMILY/'results/case-summary.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for eta,color in [(-1.,'#3877ac'),(0.,'#2d8a68'),(1.,'#b15b39')]:
        rr=[r for r in rows if float(r['eta'])==eta]
        n=np.array([int(r['n_residues']) for r in rr])
        sec=np.array([float(r['seconds']) for r in rr])
        valid=[r for r in rr if r['max_log_disagreement']]
        axes[0].scatter(n,sec,label=f'eta {eta:+g}',color=color,s=24,alpha=.75)
        if valid:
            axes[1].scatter([float(r['time_hours']) for r in valid],
                [max(float(r['max_log_disagreement']),1e-16) for r in valid],color=color,s=24,alpha=.75)
        axes[2].bar(eta, sum(r['accepted']=='True' for r in rr),color=color,width=.65)
    axes[0].set(xlabel='Parent length (residues)',ylabel='Total worker time (s)',title='Numerical check cost')
    axes[0].legend(frameon=False)
    axes[1].set(xlabel='Digestion time (h)',ylabel='Maximum required-log disagreement',yscale='log',title='2 / 4 / 8 substeps')
    axes[1].axhline(1e-5,color='#333333',linestyle='--',linewidth=1,label='Registered limit')
    axes[1].legend(frameon=False)
    axes[2].set(xticks=[-1,0,1],xlabel='Fragment-length effect eta',ylabel='Admitted condition/time cases',ylim=(0,30),title='Passes out of 28 per effect')
    axes[2].axhline(28,color='#555555',linewidth=1,linestyle=':')
    fig.suptitle('Numerical admission on all training peptide queries; no biological fit',fontsize=12)
    fig.savefig(FAMILY/'results/admission.png',dpi=160)
    gate='ADMITTED for a separately frozen bounded fit' if summary['biological_fit_admitted'] else 'NOT ADMITTED for biological fitting'
    table='\n'.join(f"| {eta} | {v['passed']}/{v['cases']} | {v['unresolved_span_queries']} |" for eta,v in summary['by_eta'].items())
    readme=f'''# Full-observation fragment solver admission

**{gate}.** This checks the numerical solver over actual training support, not
biological predictive accuracy. The separate sampled estimator still fails all
length-dependent full-parent cases under its frozen K512 contract.

Codex / GPT-6, October 23, 2026. Source commit `{cfg['git_commit']}` with recorded
dirty status and exact copied sources in `reproduce/`. The preceding adaptive
sampling family is unchanged. This family implements the [registered plan](reproduce/plan.md).

## Model and source

Full branching fragments of every length 1..L, including monomers and nonbinders.
A bond in a current fragment cuts at
`exp(x_bond @ theta) * (current_fragment_length / 20)^eta`.
The 40 theta inputs are an intercept, 19 left-residue contrasts, 19 right-residue
contrasts, and log enzyme concentration context. The original fitted initializer
is reused without fitting. Eta is fixed at -1, 0, +1. The output is the expected
number of copies of each position-specific fragment per initial parent. No MHC
selection, transport, clearance, MS response, conditioning on observed products,
probability floors, or new biological labels are introduced.

Canonical Hitlist recipe manifest `5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615`,
from the shared-observation family. Complete unchanged training support: **90,630
measurements**, 45,315 unique span/time queries, seven conditions, six parent
constructs, lengths 123–533, times 1/2/4/24 h. Source support, row identities and
weights are independently reconstructed. Every required span is covered; no easy
subset, length filtering, negative-label imputation or quantity-based selection.
All source/recipe/initializer/feature hashes are in `reproduce/launch.json`.

The entire source JSON is parsed, including excluded split records. Projection
accesses identity, support and context fields, and counts development support;
**measured quantity values are deserialized with the file but never consulted by
the projection or numerical gates**. The frozen initializer comes from
prior training. No development/test quantity, loss or prediction is computed.
Original test parents remain excluded. Existing 14,822 development observations
are not a new blind test. Canonical single-source-span mapping is asserted; no
new repeated-origin disambiguation is made.

## Numerical contract and outcomes

Fixed CRAM48 two/four/eight substeps, CPU float64/complex128, one thread. All
required values must be positive and finite, with maximum pairwise natural-log
disagreement <=1e-5; eta0 also matches independently summed analytical log counts.
Every full array is preserved, including negative or zero unobserved tails.
This is a prospective required-output criterion; it **does not pass the earlier
blanket positivity gate** or qualify arbitrary unobserved peptide queries.

| eta | Admitted condition/time cases | Unresolved required span/time queries |
|---|---:|---:|
{table}

Signed readout covers every sorted unique required span: alternating weights of
magnitude `1+0.25*sin(i+1)`, normalized by total absolute weight. Two-step autograd
produces all 41 parameter gradients. Independent NumPy eight-step central
finite differences (h=1e-5) test normalized sine/cosine theta directions and eta;
tolerance is `1e-5 + 1e-6*abs(FD)`. Undefined/unresolved forward values prevent a
derivative/admission, without clipping or dropping rows.

**{summary['directional_checks']} directional checks** across {summary['derivative_cases']}
gradient-evaluated cases; maximum absolute discrepancy
{summary['max_gradient_error']:.6g}. This finite set of directions/initial parameter
states is an empirical numerical check, not a universal error bound or a guarantee
throughout training. Every future training update must pass its registered
required-output check, and the final checkpoint must cover all rows.

Maximum full-array negative count per call: {summary['max_full_negative_count']:,}.
Maximum absolute residue-mass discrepancy: {summary['maximum_residue_mass_error']:.6g}.
Full arrays and per-call minima/mass remain in raw case archives; negative tails
are numerical artifacts, not negative molecular concentrations.

![Admission summary](results/admission.png)

## Runtime and verification

All {summary['n_cases']} registered fresh-worker cases are accounted for;
{summary['passed']} are admitted. Suite elapsed {summary['suite_seconds']:.2f} s;
maximum worker peak RSS {summary['max_rss_bytes']/1024**3:.3f} GiB. Each worker has
120 s / 4 GiB caps. Runtime includes forward refinements, autograd, six independent
perturbed solves and artifact writes, not a training update. No GPU/Modal was
requested or used; shared environment unchanged. Hardware sysctl identification
was sandbox-denied and that return code is preserved; platform/software recorded.

Two focused oracle tests plus one closure-gate counterexample test pass: nine variable-time/eta cases versus an independent
dense generator and three finite-difference directions for a full-support signed
readout. Initial pytest collection failed without the family PYTHONPATH; the
original log and corrected explicit invocation are retained. This is not a full
repository-test or stylistic-lint claim.

Closure reconstructs all source support/weights, checks {summary['full_arrays_verified']}
full primary arrays, recomputes every span log/gate and eta0 analytical reference,
and replays saved FD reductions. All 84 unperturbed full arrays are independently
recomputed with the NumPy eight-step solver: maximum full-array absolute discrepancy
{summary['independent_max_full_error']:.6g}, maximum required-log discrepancy
{summary['independent_max_log_error']:.6g}, elapsed {summary['independent_reference_seconds']:.2f} s.
The worker stores eta0 analytical discrepancy
for eight substeps; closure independently enforces that comparison for all three
refinements. Changed admission decisions: {summary['worker_gate_disagreements']}.
It does **not** rerun every perturbed matrix solve;
the producer uses a separate NumPy implementation qualified against the dense
oracle. Exact artifacts and verification scope are in `results/summary.json`.

## Reproduction and artifacts

Run `sh reproduce/launch.sh` in a fresh copy with no existing case outcomes. It
refuses to overwrite results. `reproduce/frozen-code.json` pins launcher, solver,
reference, support, cases and configuration; `reproduce/source/` contains snapshots.
Source paths are recorded; canonical recipe must be present and match its hash.
Environment: `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`;
Python `{cfg['python']}`. Tests additionally use the absolute family `code/` as
PYTHONPATH. Run `reproduce/reference-verify.sh`, then `reproduce/verify.sh`;
`reproduce/test.sh` and `reproduce/report.sh` reproduce tests and reporting.

- `results/case-summary.csv`: all case decisions, cost and negativity.
- `results/required-span-predictions.csv`: all {summary['n_span_eta_predictions']:,}
  required span/eta queries and three solver predictions.
- `results/unresolved-spans.csv` when nonempty: every unresolved query.
- `results/case-*.json/.npz`: full arrays, parameters, logs, gradients and FD logs.
- `results/independent-reference.json` and `independent-case-*.npz`: freshly
  recomputed independent full-output references.
- `results/raw-arrays-*.tar.gz`, `raw-artifacts.json`, `reproduce/unpack.py`:
  byte-verified archives/restoration of all 168 raw numerical array files.
- `results/worker-receipts.json`, `launch_logs/`: all worker returns and failures.

No fitting/pretraining or optimizer updates, assay-label losses, binding thresholds
or synthetic biological targets. No held-out predictive validation/test split is
used: this is a numerical admission experiment. Numerical prediction dumps are
retained; Spearman/Pearson/RMSE for biological observations and binder metrics do
not apply.

## Decision and handoff

Status: numerical family closed after independent artifact verification; publication
receipt is recorded separately. {gate}.

Next step: {'freeze the matched analytical-vs-fragment biological execution contract, qualifying the offset/minibatch objective first' if summary['biological_fit_admitted'] else 'inspect the preserved offending queries and replan; do not substitute a smaller biological cohort'}.
A numerical pass does not justify more sampler tuning by itself: biological value
of the length effect remains the next scientific test. No new Hitlist metadata gap
arose in this numerical work. Full APC availability/concentration, machinery,
clearance, independent MS calibration and cellular validation remain open.
'''
    (FAMILY/'README.md').write_text(readme)
    print(gate)


if __name__=='__main__':main()
