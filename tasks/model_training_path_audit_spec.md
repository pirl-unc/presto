# Model shape and executed training audit — 2026-10-20

## Question and scope

Explain the actual canonical Presto graph, the separate new TCR wrapper, and
which pathways the failed bounded IC50 fit supervised. Separate architectural
capacity, configured trainability, measured gradient exposure, parameter
movement, and demonstrated predictive quality. Trace source labels through
collation, output routing, masking, weights and losses. Do not infer that a
module learned useful biology merely because its weights changed.

Work in the isolated PR #129 checkout, preserving the original dirty checkout.
No production model changes, fresh data acquisition, GPU jobs, new fits,
held-out selection, or changes to historical artifacts are part of this audit.

## Plan

1. Read the implemented inputs, segment encoder, latent DAG, readouts and heads.
   Record actual tensor dimensions under defaults and the historical run config.
   Resolve the class-I/class-II groove-half naming and explicit TCR boundary.
2. Trace every supported assay family to its target, loss, mask, head and latent
   parents. Check whether the bounded cohort contains each source type, whether
   examples survive ingestion, and whether they enter optimizer batches.
3. Extend the already registered training-only IC50 forensic family with a
   read-only parameter/support audit. Compare initial/selected checkpoint
   tensors, optimizer slots when present, logged direct output gradients, and
   training-only loss/support ledgers. Preserve all numerical tables and exact
   commands. Do not call parameter movement proof of useful learning.
4. Inspect initialization, numerical scales, support normalization, objective
   coupling, masking and freeze settings for specific underfitting hypotheses.
   Distinguish verified defects, observations and prospective causal tests.
5. Write a code-linked report with an input/output map, information-flow
   diagram, assay supervision table, and trained/untrained component inventory.
   Include the TCR extension's actual current limitations. Give a prioritized
   training-only diagnostic plan without launching a new fit.

## Verification

- Source citations must point to inspected code at the recorded commit.
- Every historical count must identify its population (source/cohort/train
  split/exposure), rather than mixing unique records with repeated batches.
- Checkpoint names and shapes must agree before comparisons. Use safe Torch
  loading, no optimizer step and no validation/test inference.
- Diagnostic source, invocation, summaries and limitations belong in
  `experiments/2026-10-20_1458_codex_ic50-collapse-audit/`.
- Check arithmetic and generated table consistency, report exact execution
  versions and runtime, and run documentation checks for any new documentation.
- Existing final code suite has completed: 2,396 passed and four skipped.
  Documentation/audit-only additions do not require another full model suite.

## Deliverable

A repository report and concise but substantive user explanation, with exact
assay support and module evidence. Record what remains unknown and the smallest
experiments that could distinguish the leading explanations for underfit.
