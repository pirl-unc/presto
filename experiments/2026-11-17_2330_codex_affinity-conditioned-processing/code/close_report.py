"""Generate the final common-cohort decision tables from verified exports."""
from pathlib import Path
import json
import pandas as pd

E = Path(__file__).resolve().parents[1]
D = pd.read_csv(E / 'results/analysis/metrics.csv')
G = pd.read_csv(E / 'results/analysis/retention-guards.csv')
RUNS = ['parent'] + [f'{arm}-s{seed}-n2000{end}' for arm in ('continued','paired','paired-junction') for seed in (42,43) for end in ('','-terminal')]

def metric(run,pop,cls,key='auroc',readout='prediction'):
    return D[(D.model==run)&(D.population==pop)&(D['class']==cls)&(D.readout==readout)][key].item()

def label(run):
    if run=='parent':return 'Parent'
    kind='terminal' if run.endswith('-terminal') else 'selected'
    seed=run.split('-s')[1][:2]
    arm=run.split('-s')[0]
    return f'{arm}, seed {seed}, {kind}'

def table(populations):
    lines=['| Model | Ordinary I / II AUROC | Matched I / II AUROC | Matched I / II pair wins | Retention guards |','|---|---:|---:|---:|---|']
    for run in RUNS:
        pop=populations
        ordinary=' / '.join(f'{metric(run,"expanded",c):.4f}' for c in ('I','II'))
        auc=' / '.join(f'{metric(run,pop,c):.4f}' for c in ('I','II'))
        win=' / '.join(f'{metric(run,pop,c,"pair_win"):.4f}' for c in ('I','II'))
        guard='reference' if run=='parent' else 'pass' if G[G.model==run].passed.all() else 'fails ordinary I'
        lines.append(f'| {label(run)} | {ordinary} | {auc} | {win} | {guard} |')
    return '\n'.join(lines)

header='''# Processing decision and complete comparison

2026-11-18 UTC · Codex / GPT-6 · Core `0c21bc4da1066ebfadf713dca478d655c31d4b8b`.

**Retain the previous general-purpose checkpoint. Keep the new matching and paired-training infrastructure, and expose the boundary-aware selected checkpoint as a provisional processing candidate.** There is a replicated pooled-AUROC gain, but no convincing class-II paired-ordering gain and no uniformly superior model. This is a completed comparison, not a claim that processing or isolated interventions are solved.

The provisional candidate is `paired-junction-s42-n2000/raw/selected.pt`, selected at update 500 by equal-class, equal-study ordinary presentation loss. Its SHA256 is `e4898af3a7a442a999177273687ea4e4ce5d4216e908f2e435b2d73226e56963`. Seed 43 is the replicate. Both selected boundary checkpoints pass all 13 operational retention guards. Candidate choice remains a development decision; TEST is sealed. The stronger seed-42 terminal result is retained as a fixed-endpoint sensitivity, not substituted for the selected checkpoint.

The default remains the previous expanded-state-sqrt seed-42 checkpoint, SHA256 `1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68`. [Usage](USAGE.md) supports both without changing the installed model or adding an external NetMHC inference dependency.

## New hybrid-rank benchmark

209 class-I and 137 class-II pairs, respectively 4 and 3 studies; all are monoallelic. Same source protein, peptide length, condition and biological MHC candidates. Every candidate satisfies the absolute 0.5-percentile-point and relative 0.3-log-rank calipers; BA only, no EL matching. Rows and matching policy are fixed across models. Ordinary validation has 390 I and 597 II positive/background pairs, a different population.

'''
text=header+table('rank-matched')+'''

Both selected boundary replicates improve new matched AUROC from .6263 to .6437/.6431 (I), and .5482 to .5605/.5607 (II). Ordinary class I declines by .00494/.00410, while class II is retained or improves. These modest class-I declines pass the prespecified .005 point guard; that is not a statistical noninferiority result.

Class-I pair wins increase from .6890 to .7129/.6986. Class-II pair wins are .5839/.5912 versus .5839 for the parent. At the fixed terminal endpoint the boundary module gives larger AUROC gains but class-II pair wins are unchanged/decreased. Four checkpoints fail the ordinary class-I guard, as shown; none fails an affinity guard. Increasing matched-loss training is therefore not a free improvement.

## Unchanged historical nM-matched benchmark

340 I and 158 II pairs. This is the benchmark where the previous broader-training comparison regressed from .6342 to .6142 for class I. It is not Claude's original 10,998-pair cohort, and its results cannot be compared numerically to that report. The new matcher changes which pairs qualify; parent .6142 versus .6263 across the old/new benchmarks is a population difference, not a model gain.

'''+table('matched')+'''

## Paired uncertainty and component interpretation

All 24 prespecified parent/continuation/objective/module contrasts have 2,000 paired protein-family and condition bootstrap draws where the grouping is valid. Complete pair membership, common predictions/labels and grouping semantics were verified. Intervals are pointwise, conditional on this development cohort and fitted models; they omit selection uncertainty and multiplicity correction. Study-component IDs are not treated as protein clusters.

For the selected boundary candidate minus parent on the new benchmark, condition-cluster AUROC differences are +.01738, 95% CI [.01041, .02507] for I and +.01225, CI [-.00661, .03316] for II. Class-II has only eight condition clusters. Its positive AUROC point estimate replicates, but condition-level uncertainty crosses zero in both seeds. Class-II pair-win uncertainty also crosses zero. The seed-42 terminal AUROC intervals are positive for both classes, but that is insufficient to erase the cross-seed and pair-ordering limitations.

Continued ordinary training improves ordinary class-II AUROC but leaves affinity-matched performance approximately unchanged. Matched supply supervision improves the supply readout and pooled conditional ranking. The extra boundary module improves pooled ranking further, but does not establish a corresponding class-II pair-ordering benefit. Full decomposition and fixed stricter-caliper, allele, rank and BA-difference-sign sensitivities are in `results/analysis/`; small strata are descriptive, never replacement primary endpoints.

Input controls remove or shuffle APC signatures and source contexts while preserving labels, peptides and MHC. Whole-pair, role-preserving flank shuffle retains the marginal positive/background flank distributions; it tests source linkage, not all flank information. Unknown-input controls also change availability. These are fixed-checkpoint sensitivity tests, not trained ablations or isolated causal effects. Exact original-input predictions must reproduce the independent evaluations before interpreting a control.

## Measured affinity and verification

Original and ancestry-screened IC50 Spearman/Pearson/log10(nM) RMSE and 500-nM classification metrics are retained, along with all other available assay endpoints, losses and per-example predictions. For the selected boundary candidate, ancestry-screened I Spearman/RMSE are .68676/1.03526 versus .68141/1.04258 for parent; II .49483/.84218 versus .48996/.83597. All supported parameters were trainable; affinity retention is empirical, not a consequence of freezing. The original affinity cohort has 346 ancestor-exposed rows and is not substituted for its screened sensitivity.

All six 2,000-update fits and three 16-update integrations pass coverage. Ordinary draw order/source weights/scalers agree across arms, auxiliary pairs remain TRAIN-only, and the boundary head receives real gradients/updates. Thirteen model evaluations on six cohorts independently reproduce 1,066 numeric checks across 78 combinations. Core tests: 202 passed; bootstrap/control tests: six passed. CPU label-free class-I/class-II presentation and IC50 inference succeeds. Input-control checks and final lifecycle receipts are linked in the README. No TEST metrics or predictions were produced because the split remains sealed.

## Limits and next data generation

BA is the accepted working affinity estimate. Residual BA within-pair discrimination, limited studies, monoallelic validation, unobserved backgrounds, and unknown external-predictor training overlap remain limitations. Same-protein/condition pairs do not identify protein-expression or absolute-yield effects. The matched subset has no quantitative expression annotations and sparse clean drug/KO comparisons; actual PMID-linked support is in `results/PROCESSING_CONTEXT.md`.

The user subsequently requested Hitlist 2.1.12. Its refreshed loader-based exports and dependency verification are a separate source generation recorded in `tasks/hitlist-2.1.12-refresh/`. This family's frozen 2.1.9 data and model runtime are unchanged. Any next fit must regenerate protected TRAIN/VAL inputs and current annotation audits from that new generation; swapping source/version labels onto these results is invalid. A next bounded model comparison should target final within-pair presentation ordering and ordinary retention explicitly, rather than enlarging a head based only on pooled AUROC.
'''
(E/'DECISION.md').write_text(text)
(E/'results/analysis/comparison-rank.md').write_text(table('rank-matched')+'\n')
(E/'results/analysis/comparison-historical.md').write_text(table('matched')+'\n')
