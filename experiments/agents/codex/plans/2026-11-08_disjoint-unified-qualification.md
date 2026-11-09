# Newly disjoint unified qualification and MHC-II mechanism plan

Date: 2026-11-08
Owner: Codex / GPT-5
Status: planned alongside the active parity retrain; no cohort outcomes accessed

## Objective

Produce the next defensible unified Presto candidate: one streamlined,
assay-aware model that improves MHC-I and MHC-II physical affinity while
retaining presentation and operational-processing performance, then adds
assay-specific T-cell immunogenicity tracks and compares eligible class-I rows
with PRIME 2.1. Superiority claims must come from newly disjoint whole-study
cohorts, not the exposed Stage B development set or the NetMHC training
archives.

The active reference/parity retrain remains the first decision. Do not implement
the class-II adapter, recut an evaluation cohort, or train T-cell tracks until
both 5,000-step receipts and their preregistered parent decision are closed.

## Why MHC-II currently lags

The current evidence points to four separable constraints rather than an
inability to improve:

1. NetMHCIIpan has a much closer data contract to its own public BA archive.
   Presto's current exact-IC50 development comparison has only 1,170 MHC-II
   rows, while the parity arm adds 62,539 archive rows as TRAIN with unknown
   assay method and many censored values. Those rows improve coverage but cannot
   teach protocol-specific offsets.
2. The shared candidate interaction concatenates alpha and beta groove residues
   into one attention memory. It enumerates every 9-mer register correctly, but
   does not expose ordered chain-specific interaction summaries to the readout.
   Prior register-temperature/pooling changes were negative, so another pooling
   tweak is not the next mechanism.
3. Class-II registers are latent in these affinity rows. Without measured cores,
   the model must learn both register choice and affinity from sparse,
   assay-heterogeneous outcomes. Register entropy is therefore a diagnostic, not
   occupancy truth.
4. The active Stage B checkpoint is selected on equal-study elution loss, not
   class-II affinity. This protects presentation but can select a checkpoint
   before the best MHC-II affinity step. The next mechanism comparison must use
   the same selection contract for adapter and control and report final-step as
   well as selected-step metrics.

These are testable hypotheses. The plan does not promise that one adapter will
beat NetMHCIIpan; it defines the shortest fair experiment that can show whether
chain structure closes the residual gap.

## Phase 0 — close the active parity retrain

- Verify completed reference/parity receipts, selected and last checkpoint
  hashes, 5,000 steps, coverage, resource metrics, and zero TEST rows.
- Run the exact matched evaluator on all 51,066 development rows and publish the
  timestamped report with reusable affinity, calibration, ROC/PR, and study-delta
  figures.
- Apply the frozen parent gate without retuning it. If parity does not improve
  MHC-II exact-IC50 Spearman by at least 0.05 while preserving the other tasks,
  retain reference as the continuation parent and treat the BA rows as an
  informative negative/partial result.

## Phase 1 — genuinely new affinity data

### Source inventory

Run a metadata-first Modal inventory of the newest canonical Hitlist/IEDB/CEDAR
exports and any source supplements released after the snapshots used by Stage B
and the NetMHC archives. Preserve publication, contributor, assay, MHC-pair,
physical endpoint, units, qualifier, and original-measurement identities. Do not
look at model predictions or use outcome values to select studies.

Admit only literal physical affinity endpoints (IC50, KD, EC50 and explicit
proxies kept separate). Do not relabel qualitative binding, stability, elution,
or generated backgrounds as affinity. Resolve complete alpha/beta pairs for
DP/DQ and preserve the recorded DRA assumption only for eligible DR rows.

### Split and overlap contract

- Build connected components over publication/contributor aliases and original
  measurement identities.
- Reserve whole post-snapshot studies for TEST before outcome inspection,
  prioritizing support for MHC-I/MHC-II, exact values, assay-method cells, and
  allele diversity. A dedicated calibration split uses separate studies.
- TEST > calibration > development > TRAIN priority; embargo lower-priority
  exact peptides and shared 9-mer cores. Record every removed row.
- Audit against every current Presto source, the projected NetMHC archives,
  historical development/TEST cohorts, and comparator training identities when
  available. Post-cutoff study provenance is the primary defense against hidden
  comparator training overlap.
- If there is not enough genuinely new exact class-II affinity support for a
  nondegenerate test, state that explicitly and do not manufacture one from
  consumed studies.

Freeze membership, row counts, hashes, endpoint/class/assay cells, and a blinded
outcome token before any fit. Comparator predictions and model predictions are
forbidden during membership selection.

## Phase 2 — one class-II chain-aware mechanism

### Architecture

Add one opt-in, default-off `class2_chain_adapter` to the existing Presto model:

1. retain the current complete-peptide encoder, 9-mer candidate enumeration,
   joint-groove interaction, normalized register marginalization, and assay
   decoder;
2. for MHC-II only, attend each candidate's ordered core representation to the
   alpha groove and beta groove separately;
3. concatenate the ordered alpha/beta interaction summaries and pass them
   through a small residual adapter whose final projection is initialized to
   zero;
4. add the residual to the existing candidate vector. At initialization, the
   adapter-on model must be numerically identical to adapter-off for every
   endpoint and both MHC classes;
5. keep assay output identities and offsets unchanged. No new generic affinity
   head, register temperature, class-specific trainer, or alternate inference
   backend is allowed.

Test exact neutral start, alpha/beta order sensitivity, MHC-I invariance,
variable-register gradients, checkpoint/resume/freeze, complete parameter
coverage, and old-checkpoint compatibility before training.

### Matched experiment

Use the Phase-0 selected parent and the expanded Phase-1 TRAIN data. Run
adapter-off and adapter-on for seeds 42 and 43, two concurrent requested
`H100!` workers at most. Freeze one fixed update budget after the common loader
reports the new census. Both arms use identical sampling, losses, schedules,
initial common weights, and checkpoints.

Checkpoint selection is identical between arms: a preregistered development
score that gives equal weight to equal-study elution loss and MHC-II exact-IC50
loss after train-only scaling. Also report fixed final-step results and the
historical EL-only selected step as nonselecting diagnostics. The new TEST stays
locked until both seeds, the adapter decision, all development predictions, and
all software gates are frozen.

Promote the adapter to the unified architecture only if both seeds improve
development MHC-II exact-IC50 Spearman by at least 0.03, mean improvement is at
least 0.05, MHC-I declines by no more than 0.01, and presentation and operational
processing equal-study AUROC each decline by no more than 0.01. Ties favor the
adapter-off model.

Open the new affinity TEST once. A comparator-win claim requires a positive
Presto-minus-comparator point estimate on identical rows and a paired
whole-study bootstrap interval excluding zero. Report pooled and equal-study
Spearman/Pearson/RMSE, qualifier-aware 500-nM metrics, censor satisfaction,
calibration, assay cells, alleles, lengths, and register entropy/core diagnostics.

## Phase 3 — assay-aware T-cell continuation and PRIME

### Cohort

Create a new whole-study T-cell cohort with source-observed negatives,
mixed-outcome studies, complete class-I restriction for the PRIME comparison,
and enough positives/negatives per study to define AUROC/AUPRC. Separate tracks
remain separate outputs at minimum for activation, IFNg ELISPOT, IFNg ICS, TNFa
ICS, cytotoxicity/proliferation when supported, and multimer binding. Preserve
species, host, disease, cell source, CD4/CD8 context, stimulation protocol,
assay method, and qualitative/ordinal intensity. Do not use random peptides as
immunogenicity negatives.

Use whole-study TRAIN/development/calibration/TEST partitions with peptide/core
embargoes and original-measurement deduplication. Fit calibration only on the
dedicated calibration studies (or nested TRAIN folds if a track lacks support),
with a separate temperature/intercept per explicit assay track. Never pool all
T-cell assays into one calibration category.

### Model and comparator

Continue the Phase-2 selected unified checkpoint with the existing fixed-parent
assay-aware T-cell layer. Start from binary-only losses; conditional ordinal
intensity remains auxiliary and must re-earn admission. Preserve affinity,
presentation, and processing heads in the same checkpoint and re-run their
regression gates after T-cell training.

PRIME 2.1 is operational through `mhctools.PRIME` using the hash-pinned local
MixMHCpred 3.0 isolation recorded in
`2026-11-08_2225_codex_prime-mhctools-smoke`. Score PRIME only on eligible
class-I, 8–14-aa, supported-allele TEST rows after the candidate and calibration
are frozen. Use PRIME percentile rank with the correct lower-is-better direction
and preserve its raw score/rank per peptide-allele pair.

Primary T-cell metrics are equal-study and equal-track AUROC/AUPRC on observed
binary outcomes. Also report Brier score, calibration slope/intercept, ECE,
precision/recall at preregistered thresholds, per-study/track support, and paired
study bootstrap intervals versus PRIME. PRIME-ineligible rows remain Presto-only
evaluation; they are never counted as comparator failures.

## Phase 4 — final unified retrain and reporting

Once mechanism and T-cell choices are frozen, retrain one streamlined model on
all eligible TRAIN plus calibration data, keeping TEST locked. Use no ensemble
unless a separately preregistered compact-distillation experiment shows a
serving benefit; the default deliverable is one checkpoint and one inference
path with explicit assay-aware outputs.

Generate a timestamped state report at every promoted checkpoint. Each report
must include:

- source commit, dirty state, model/checkpoint hashes, hardware and runtime;
- complete TRAIN/development/calibration/TEST dataset contracts and hashes;
- assay-label to output/calibrator mappings, loss and sampling weights;
- per-example prediction artifact locations and hashes;
- NetMHCpan, NetMHCIIpan, MHCflurry and PRIME versions/receipts;
- exact-value, threshold, discrimination and calibration figures generated only
  from saved per-example tables;
- explicit pass/fail gates and limitations.

Large/raw/licensed artifacts stay on the Modal volume or existing licensed local
installations. Do not add the approximately 900-MB local bundle or licensed
binaries to Git.

## PR and stopping policy

The current successor PR closes Phase 0, the reusable evaluator/reporting code,
and the PRIME integration receipt. Class-II mechanism code and new-cohort
admission belong in a subsequent reviewable commit/PR only after the Phase-0
decision. Stop and re-plan on source insufficiency, leakage, nonneutral adapter
initialization, unsupported assay cells, or any task-regression gate; do not add
posthoc arms to rescue a failed comparison.
