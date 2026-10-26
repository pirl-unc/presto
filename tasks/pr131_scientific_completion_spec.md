# PR 131 scientific completion — 2026-10-26

The user explicitly requires scientific milestones before merge. Keep PR 131
draft; software integration and green CI do not complete issues 130/132.

## Ordered gates

1. Finish and merge PR 129, bringing forward its needed Hitlist 2.x import fix.
   Preserve its wrapper/audit as an independently documented research API.
2. Establish the existing full-cohort IC50 convergence/replication comparison,
   following `experiments/agents/codex/plans/full-ic50-fitting.md`, amended to
   local CPU by the later local-feasibility decision. Preserve the four fixed
   neural conditions, exact training cohort, historical split exposure, matched
   controls and class-specific validation gates. Add censor-aware training only
   as the separately declared next condition. Do not retune from test labels.
3. Qualify auxiliary inference and retained assay performance; extend other
   directly measured endpoints in the support order in issue 130. Passing one
   assay subset is not completion of the full model.
4. Establish qualified APC product/selection and shared MS observation evidence
   with independent transfer. Existing negative availability/response studies
   remain failed. Source audits and varying outputs do not satisfy these gates.
5. Admit the authorized TCRvdb release and qualify paired scoring/retrieval with
   the declared leakage and reconstruction controls. The current author README
   still requires requesting access; no file was found in project/attachments.
   A local path/connected source has been requested from the user. Do not invent
   a parser, substitute a restricted derivative, or contact authors implicitly.
6. Verify integrated training/inference/support, full software regression and
   final-head CI, then reconcile all scientific criteria before merging.

## Immediate implementation and verification

Before the full IC50 comparison, canonical prediction dumps must retain the
original observation/input provenance, resolved field diagnostics and class-II
register weights. Convert tensor diagnostics to ordinary JSON values at the
reporting boundary only. Keep existing numeric predictions, losses, optimization
and checkpoint selection unchanged. Add a real multi-register regression that
fails on the old evaluator, preserves raw input metadata and compares the saved
scalar/loss against the direct model. Verify train/eval mode restoration and
strict JSON serialization.

Register the experiment, preserve original source pins, snapshot launcher and
runtime files, and record the exact local hardware/dependencies and invocation.
Use local CPU/four numerical threads/one interop thread, sequential fits, with
no package installs or cloud compute. Enforce memory/disk admission and preserve
terminal failures. Do not run a large source materialization alongside fitting.
Every completed condition needs predictions, metrics, coverage, selected-checkpoint
fresh-process replay and independent metric verification. Report partial running
conditions honestly; no successful family closure until all declared arms finish.

## Known limits

The IC50 held-outs were inspected historically and share studies with training.
They are not a fresh independent clinical/biological validation cohort. The
full-cohort comparison cannot qualify APC concentrations, MS calibration, TCR
specificity, auxiliary benefits or all-species inference. Each keeps its own gate.
