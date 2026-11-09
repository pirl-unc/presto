# Canonical native peptide response targets, then biological prediction

Prospective specification after source correction replay 0156; before recipe
implementation or any fitting on these native extract measurements. Source
corrections are Hitlist PR793 (ddc6581), with delivery still pending. This work
continues issues788/791; do not create a competing Presto source curator.

## Biological question and measured endpoint

Can a sequence-dependent processing model predict which endogenous peptides
increase or decrease under chemical perturbation better than no-change and
condition-only controls? The measured endpoint is each original positive native
extract treated/control isotope-channel ratio, exact or explicitly bounded.
It is not an intracellular molar concentration, APC compartment, MHC ligand
distribution, genetic knockout or measured enzyme residual activity.

The next fit must use these real endpoints. Mathematical sampler validation and
source conservation are prerequisites, not the biological acceptance criterion.
Retain the prior MAPP stop rule; do not reopen that development set for tuning.

## Shared source and reference work

1. Extend the existing Hitlist native family and registered recipe interface.
   Preserve every source disposition and queryable context; pin the corrected
   package and exact code/release state. Initial source candidate counts are
   7,933 exact and 4,662 bounded slots, before grouping or further eligibility.
2. Expose source flanks and parent/gene assertions at their original grain. The
   primary tables contain protein descriptions, not a resolved accession set.
   Do not silently treat names as full protein sequences or gene IDs as origins.
3. Reuse Hitlist's exact sequence-reference occurrence machinery with a pinned
   public human reference already available to the shared source system. Keep
   its date, format, hash, taxonomy, accession/sequence versions, candidate full
   sequences and every occurrence. A modern reference is not the original IPI
   search library. Source name/gene/flank agreement is a separate assertion.
   Preserve multiple parents and absent mappings. No I/L substitution or guessed
   polymorphism repair; no duplication of the label for alternative origins.
4. Sequence-only diagnostics can cover unresolved-parent records, but full-parent
   processing fits require an explicitly declared mapping contract and coverage.
   Any new gap belongs in the existing upstream issue, not a local workaround.

## Target and dependence contract

- Admit only reported, supported sequence identities with positive finite exact
  ratios or positive explicit bounds from treated/control slots. Retain supported
  peptidoforms separately. Unknown/tentative/unsupported identities, controls,
  derived means, overlap, missingness and spreadsheet errors remain in disposition
  records and do not become targets or negatives. Do not impose an MHC length or
  affinity filter; short/nonbinding processing products remain in scope.
- Keep numeric targets, inequality, raw source cell and every contributor; never
  average bounds into exact values or infer censoring from a numeric threshold.
- Build reuse relationships on all source rows before training admission. The
  830 candidate groups are possible reuse, not adjudicated specimen identities.
  For matching-context copies with identical source observations, retain all
  provenance and at most one effective observation weight. Conflicting contexts
  are deferred pending explicit source adjudication; no preferred copy selected
  by its target value. Preserve the 278 different-label groups for review.
- Represent charge states and repeated ratios as contributors to a common
  peptide-form/condition response group. Preserve raw values rather than inventing
  physical replicate identities. Normalize their total training weight so extra
  charges or republished rows cannot create independent biological sample mass.
- Shared-control IDs remain source-label proxies. Preserve unknown sample/run/
  isotope-channel/denominator identity and do not report specimen-independent
  confidence intervals. Controls can support descriptive measurement checks,
  but cannot be presented as biological response supervision.
- Define fixed hierarchical reductions before fitting: condition, parent component,
  peptide form, contributor. Keep group-weighted and pooled diagnostics distinct.
  Report effective groups and source slots separately, with explicit count units.
- The recipe must export examples, contributor membership, every excluded/deferred
  disposition, context/reference alternatives, split components, reduction weights,
  coverage and manifests. Re-materialization through the public API must verify
  every original quantity, bound direction, weight sum and group membership.

## Partition and evaluation contract

Construct connected components across every source peptide before filtering:
identical sequence, all shared exact full-parent alternatives, and possible reused
measurements stay together. Preserve source name/gene assertions for additional
conservative links without declaring those assertions proven origins. Freeze
deterministic train/validation/test assignments before reading model outcomes;
inspect size/support feasibility without selecting splits by target performance.
Do not promise homology, specimen or external-study independence from these links.

Fit scalers/features and any context category vocabulary only on training data.
Audit unseen cell/compound/dose/duration/parent features and allocated parameter
support. Do not silently use untrained coefficients. Reserve test predictions for
the predeclared final comparison. Exact-label RMSE/MAE, Pearson/Spearman, prediction
spread, group-level error and direction-of-change metrics accompany one-sided
censored loss/satisfaction; neither nM nor a 500 nM binding cutoff applies here.

## Next model comparison, to freeze after recipe feasibility

First establish zero log response, train-mean, condition-only and parsimonious
sequence/context regression controls. This is a real peptide response benchmark,
not a replacement for the requested generative concentration model.

For mapped parents, use the same complete 1..L fragment support as the existing
processing model, continuous precursor input, sequence-dependent cut rates and
removal. Predict a treated/control availability ratio per peptide form, aggregating
origin alternatives under an explicit rule before one measurement loss. Chemical
context can modulate learned rates; it is not a supplied knockout or enzyme truth.
Same-form detector response cancellation is an explicit modeling assumption, not
independent MS calibration. Precursor changes remain a possible confounder.

Freeze one differentiable and one sampled arm with the same physical model,
initialization, features, endpoint, loss/reductions and final-checkpoint rule.
Start from the already qualified independent-cut model; do not add another
fragment-length mechanism before an informative biological comparison. The
sampled arm must state every exact component it retains and record gradient
error, score evaluations and actual runtime. No claim of pure RL or speedup when
using an exact-assisted hybrid. Specify numerical/resource limits prospectively;
retain any failure without outcome-driven retries or moving gates.

Run bounded local CPU fits, preserve per-example validation/test predictions and
all baseline comparisons, and close the canonical experiment log. A win must
beat the simple biological controls on the same held-out contract. Report failures
directly. Neither source readiness nor nonzero gradients can substitute for this.

## Exit from this stage

Deliver a canonical reproducible recipe and at least one completed real biological
comparison with the source limits visible. The full goal still requires validated
APC machinery/context behavior, precursor-relative amounts, independently supported
observation effects and concentration-relevant evidence; this response experiment
does not redefine that goal or declare it complete.

## Located reference inputs (revalidate bytes before admission)

The previously audited shared package `/private/tmp/hitlist-v2-mapp-reference-second`
contains these existing public inputs under `external/pxd009092/assets/references/`:

- `human-reviewed-2026_03-isoforms.fasta`, manifest SHA-256
  `8c4459cc1e190caad54150511e5a0b5779863557f88387e5a86647a48d19a9d4`,
  29,338,408 bytes. Prefer this explicitly modern human model reference for exact
  parent alternatives; it is not the original IPI search library.
- `human-reviewed-2023.dat.gz`, manifest SHA-256
  `defd9fd7fdc3be6f36b5b57b1a2c29978b956612e1437ec15f7065b055bbe9d3`,
  108,046,188 bytes. Historical file contains reference-local accession metadata;
  do not silently transfer aliases between reference generations.

`hitlist/sequence_reference_evidence.py` already provides `read_reference`,
`exact_occurrences` and `reference_relations`. Native protein descriptions must
not be passed as if they were source accession identifiers. Keep original
name/gene/flank assertions separately and add no guessed origin assignments.
