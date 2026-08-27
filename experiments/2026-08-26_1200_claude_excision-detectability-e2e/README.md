# End-to-End Validation: Excision + MS Detectability Branches

- **Agent**: `claude`
- **Date**: 2026-08-26
- **Kind**: validation run, not a result. It answers "does every branch train together
  in the canonical path", not "is the model better".
- **Design spec**: [`tasks/protease_detectability_spec.md`](../../tasks/protease_detectability_spec.md)
- **Execution plan**: [`tasks/todo.md`](../../tasks/todo.md) Stages 0-3
- **Reproduce**: [`reproduce/launch.sh`](reproduce/launch.sh), [`reproduce/launch.json`](reproduce/launch.json)

## Question

Presto gained four things in this change set: a hitlist-backed data source that supplies
flanks, an expanded latent DAG, a machinery-conditioned excision head, and a non-MHC
shotgun corpus supervising MS detectability. Do they train together, in one
`presto train unified` invocation, with every loss receiving support?

This is deliberately a smoke-scale run on CPU. No claim is made about model quality; the
decision gate for that is the Stage 4 factorial, which has not been run.

## Contract

**Data**

| | |
|---|---|
| MHC source | `hitlist` curated indexes (`--data-source hitlist`), class I, `HLA-A*02:01` |
| Non-MHC source | hitlist `bulk_proteomics`, HeLa, all four protease arms |
| T-cell / TCR / processing | merged TSV (hitlist does not carry them) |
| Caps | binding 1200, elution 1200, tcell 300, vdjdb 200, processing 200, stability 200, kinetics 50, bulk 1500 observed |
| Cap sampling | `head` |

**Model**

| | |
|---|---|
| Latent topology | `expanded` (12 design-doc latents, own query + segment scope each) |
| Size | `d_model=32`, 2 layers, 4 heads — 510,251 parameters |
| Epochs / batch | 1 / 32 |
| Device | CPU |

## Observed

Loaded corpus:

- **Resolved MHC sequences: 215/219 alleles** (mhcseqs=215, index fallback=0)
- Flank coverage from hitlist: **binding 63.5%**, **elution 88.5%** — the merged TSV
  supplies 0%, so this is the first Presto run with real junction context
- Bulk corpus: 3,000 records = 1,500 observed + 1,500 excision negatives, across
  trypsin 1,100 / chymotrypsin 143 / GluC 132 / LysC 125
- **Total samples: 9,012** (MHC branch 6,012 + shotgun branch 3,000)
- Train batches 226, val batches 57

Training ran to completion. **Best val loss 1.6547**; the total dropped monotonically
from 4.98 to 2.18 over 226 batches.

**25 tasks received support**, across every branch. Per-task loss:

| task | train | val |
|---|---:|---:|
| binding | 2.7577 | 1.8248 |
| presentation | 0.6573 | 0.4409 |
| ms_detectability | 0.6620 | 0.6344 |
| elution | 0.6365 | 0.3406 |
| tcell | 0.5222 | 0.2432 |
| excision | 0.2694 | 0.2584 |

Mean rows per batch carrying each of the new labels: `excision` 17.0,
`ms_detectability` 9.0 — so the shotgun branch is contributing real gradient, not a
handful of stragglers. Full per-task table in `results/task_losses.json`.

## Held-out metrics

The unified trainer now emits them (`training/holdout_eval.py`, wired after training).
From a smaller companion run with the same configuration — **13 tasks scored**, every
branch represented:

| task | metric | n |
|---|---|---:|
| excision | AUPRC 0.9889 | 132 |
| immunogenicity | AUPRC 0.9821 | 34 |
| presentation | AUPRC 0.9740 | 110 |
| elution | AUPRC 0.9512 | 110 |
| tcell | AUPRC 0.8866 | 34 |
| ms_detectability | AUPRC 0.8080 | 65 |
| binding_ic50 | Spearman 0.2554 | 25 |
| binding | Spearman -0.0606 | 136 |

Read these as plumbing evidence, not model quality — one epoch at `d_model=32` on one
allele. Two are worth calling out honestly:

- **`excision` AUPRC 0.9889 is close to measuring its own constraint.** Negatives are
  generated as cleavage-rule violations and the in-vitro profiles are pinned, so a high
  score here largely confirms the rule is wired through to the logit. The informative
  version is `pin_profiles=False`.
- **`binding` Spearman is negative** at this scale — one epoch on a 32-dim model over
  136 held-out measurements. Expected, and precisely why the Stage 4 factorial and a
  real baseline comparison are the gate rather than this run. (Checked whether a
  target-space mismatch in the dump explained it: it did not. That bug was real and is
  fixed, but Spearman is invariant to monotone transforms, so it could not have been
  the cause — the value moved to -0.0765 and stayed negative.)

## What this validates

1. Both corpora co-train in one batch stream — MHC rows with no excision label and
   shotgun rows with no MHC coexist, each masked out of the other's losses.
2. The expanded topology trains: 12 latents, all reachable, gradients everywhere.
3. The excision and detectability losses receive support from real data.
4. The hitlist path supplies flanks end to end, into a model whose flank segments have
   been wired since February and never fed.

## What it does not validate

- **Nothing about quality.** One epoch, `d_model=32`, one allele, CPU.
- The Stage 4 factorial (arms A-E) is the decision gate and has not been run.
- Detectability targets here are the depth-graded signal over observed peptides only.
  True in-silico-digest negatives need [hitlist#361](https://github.com/pirl-unc/hitlist/issues/361).
- Bulk rows carry no flanks (coordinates yes, protein sequence no), so `s_N` is
  unsupervised on that branch and excision rests on `s_C`.

## Two defects found and fixed while getting here

1. **`train unified` silently dropped every MHC row.** Sequence resolution was gated on
   `if args.index_csv:`, which defaults to `None`, but the resolver is mhcseqs-first
   with the CSV index only as a fallback. A default invocation resolved 0 alleles, then
   `filter_unresolved_mhc` dropped 100% of binding/elution/tcell/stability rows. The
   coverage report printed `resolved=0/N`, which reads as a data problem rather than a
   skipped branch. Guard removed.
2. **`~/code` on `sys.path` disables MHC resolution.** That directory holds a
   `mhcseqs/` repo folder with no `__init__.py`; Python treats it as an empty namespace
   package that shadows the installed one, and `mhcseqs.__file__` becomes `None`. Both
   packages are installed editable, so run from a neutral directory with no
   `PYTHONPATH`. Recorded in `tasks/lessons.md`.

## Artifacts

- `results/run.log` — full training log
- `results/metrics.csv`, `results/metrics.jsonl` — per-epoch, per-task losses
- `results/config.json` — resolved run configuration
- `results/mhc_sequence_coverage{,_post_filter}.{json,csv}` — allele resolution audit
- `summary.json`, `val_predictions.csv`, `val_metrics.csv` — written by the held-out
  pass for runs made after that wiring landed
