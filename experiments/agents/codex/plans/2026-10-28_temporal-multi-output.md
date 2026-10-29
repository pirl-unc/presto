# Shared multi-output training with prospective 2026 study holdouts

Owner: Codex / GPT-6. Requested 2026-10-28. This specification precedes code/data
membership changes. Family: `2026-10-28_1540_codex_temporal-multi-output`.

## Objective and accepted decisions

Use one shared-v5 model and observation-manifest trainer, starting from tested
wide 192/3/6/768. The user requests a primary multi-output recipe, broader
IEDB/Hitlist measurements, whole 2026 study holdouts and per-assay plus ligand
metrics. This supersedes retaining IC50-only as the primary development workflow;
it does not retroactively change the previous failed multi-output promotion gate.
Preserve all closed experiments and original dirty workspaces/shared environment.

## Stage A: source and metadata inventory

- Inspect canonical Hitlist APIs/releases and source freshness. Fetch current
  public IEDB/CEDAR through canonical APIs into task-owned storage; use Modal CPU
  staging for large sources if appropriate. No arbitrary row cap.
- Inventory publication year, study, class/species, assay, candidate-allele scope,
  positive/negative support and original contributor identities. Verify dates from
  reference metadata, never PMID magnitude or ingestion date. Do not select on
  numeric outcomes or model predictions.
- Preserve physical endpoint identity/units/censoring. Use declared DRA*01:01 for
  otherwise eligible allele-resolved ordinary human DR missing alpha typing.
  Retain source-reported typing and reference-assumption provenance. Reported
  alpha/construct identities take precedence; DP/DQ and serotypes need resolution.
- Admit canonical MS positives with their single/multiple candidate MHCs; never
  invent a cognate allele. Keep measured negatives separate from synthetic
  background. Include compatible available evidence and record exclusion reasons.
- Freeze exact source hashes and support inventory before training. If suitable
  2026 studies are unavailable, do not relabel old data as 2026; report the gap and
  continue independent source/model work. Update the execution contract below
  after metadata inventory, before any fit.

## Stage B: prospective membership and training

Select a small number of whole 2026 studies deterministically for validation/test
based on metadata coverage, not favorable outcomes; train on remaining eligible
measurements. Whole-paper supplements and duplicate contributor aliases follow
study assignment. Audit and remove train/heldout measurement alias leakage. Freeze
sequence/component exclusion policy and report a sequence-novel subset alongside
study-level results where necessary. Old fits do not initialize these models;
previous test data may join training under this newly requested temporal contract.

All endpoints/classes use wide shared-v5 and the same trainer. Quantitative outputs
retain exact-TRAIN-only scales and one-sided squared loss; ligand labels use the
common binary endpoint, preserving multi-allelic aggregation. Additive observation
components must not be claimed as separately identified from a single MS outcome.
Protect affinity supervision with explicit endpoint mixture/loss weights; report
unique observations and repeated optimizer exposures. Fix two seeds, a bounded
budget covering the expanded source meaningfully, validation cadence/checkpoint
objective, and all loss weights before fitting. Use at most two H100! workers and
record actual hardware. Build the live report before launch; poll status separately
from archive collection. Do not expand into an uncontrolled architecture sweep.

Ligand AUROC/AUPRC: use supported measured negatives when present. For positive-only
MS, background is explicitly synthetic/unobserved, never confirmed negative. Freeze
candidate origin (same-protein when canonical mapping permits, otherwise separately
reported proteome population), length distribution, ratio, seed, positive exclusion,
sample context and split grouping. Report ligand-versus-background discrimination,
class/study/allele-set strata and length-only control. No hidden positive relabeling.

## Stage C: implementation, validation and closure

Admission/evaluation changes belong in canonical shared modules. Experiment-local
scripts orchestrate preparation/launch/report and cannot supply another model or
trainer. Verify original targets, dates/assumptions, membership, candidate aggregation,
sampler counts, bounds, resume and installed inference. Freeze complete package,
launcher, manifests, invocation and dependency pins before remote fits.

Close all fixed fits: original archive hashes, per-example validation/test dumps,
independent metrics and saved-TRAIN-probe serving replay. Report each output/class/
study loss, exact Spearman/Pearson/RMSE, concentration-threshold classification,
ligand AUROC/AUPRC, support/uncertainty, runtime, input/parameter coverage and untrained
weights. Publish shared multi-output recipe/checkpoint contract, docs, canonical
registry and model reference. PR131 stays draft pending scientific milestones130/132.

## Frozen execution details

Pending source/metadata inventory. No predictive fit has started for this family.

## Shared scalability changes, specified before implementation

The prior trainer materializes predictions for the entire training corpus after
selection. At the expanded MS scale this duplicates large provenance/diagnostic
records without adding held-out evidence. Add an opt-in deterministic random
TRAIN diagnostic evaluation limit (default unlimited), recording available and
evaluated counts. It must never limit training membership or validation/test.
Add opt-in equal-endpoint validation-loss reduction (default historical
observation mean), so MS row counts do not determine checkpoint selection alone.
Existing recipes and resume behavior retain their defaults. The new recipe's
exact diagnostic count and selection reduction will be frozen after inventory.

CPU preparation attempt 1 was cancelled while still PENDING with no task ID after
five minutes. Attempt 2 uses the successful acquisition resource shape (8 CPUs,
96 GiB reservation); both invocation receipts and the original source are retained.
This changes no predictive contract. The source-date scan is now executing.

## Ligand comparison implementation contract

Canonical background sampler: exact Hitlist protein mappings; equal probability
among mapped protein IDs, otherwise all length-eligible proteins in the frozen
human Ensembl-112 FASTA. Match each positive's length, candidate MHC set, context,
study and split. Exclude every known positive ligand sequence and all peptides
assigned to other splits. Keep the source protein and position and positive parent
ID for each generated comparator. A deterministic exhaustion search makes missing
background support explicit. These are unobserved comparators, never measured
non-binders. The shared training recipe initially leaves source-context encodings
disabled, so mapping availability cannot create a positive/negative missingness
shortcut. Full ordered peptide context and both groove sequences remain active.
Report same-protein and fallback-proteome comparisons separately. Freeze train/eval
ratios and parent membership after admission, before fitting.

## Accepted user clarification

The refreshed 2026 exports have no direct IC50, KD or EC50 rows. On 2026-10-28 the
user explicitly chose **additional older assay-study holdouts for broader
evaluation**. Reserve whole older studies to cover quantitative output/class
combinations, selected by admissible metadata counts before fitting. Keep the
2026 ligand/new-study results separate from the older-study assay evaluation.
Earlier software/architecture development used historical data; the older cohort
is a new from-scratch study holdout, not a claim of previously unseen research.
No old checkpoint initializes the models. Train on the remaining compatible data,
with whole-source measurement aliases and peptide leakage exclusions recorded.

## Additional measured output identities

The canonical cached-source metadata inventory (without target values) contains
69,916 purified qualitative binding rows, 1,152 50%-dissociation-temperature rows,
53 off-rate rows and 34 on-rate rows, before human/catalog admission. Extend the
same typed adapter and endpoint registry with `qualitative_binding` (source-reported
binary outcome, no inferred 500 nM threshold), `td50` (degrees C, distinct from
melting temperature `tm`), and `kon` (log10(1/(M*s))). Preserve units/censoring and
all source fields. Never relabel KD proxies as direct IC50 or KD. Only allocate
heads with admitted TRAIN labels. Sparse outputs retain explicit limited-support
status; adding a head is not a claim of predictive qualification.

## Preparation recovery (no predictive fit has started)

Attempt 2 stopped at Hitlist's default 16 GiB contributor scratch SQLite ceiling,
not GPU memory or an arbitrary training cap. The canonical builder correctly
removed its partial scratch output and retained the source downloads. Preserve
its logs/receipt; retry the same source/admission semantics with explicit 128 GiB
scratch budget, 16 GiB free reserve and 256 GiB requested ephemeral disk. Record
actual starting disk capacity. Reduce progress redraw frequency to 30 seconds.

Attempt 3's 256 GiB explicit disk request was rejected before worker creation:
Modal requires at least 512 GiB for an explicit disk allocation. Attempt 4 requests
512 GiB and asserts the 128 GiB provenance budget environment before building.
The preserved attempt receipts distinguish requested resources from actual workers.

The cached other-assay index also contains 23,504 human MS rows with reported
negative status and exact allele scope. Add admission of explicit source-reported
MS negatives under the same human/catalog/chemical/source-identity checks. Keep
reported-negative evaluation separate from sampled-unobserved backgrounds; never
silently replace reported negatives with decoys. Verify fresh-corpus support and
study identity before freezing the cohort.

## Study-selection algorithm (before fit and numerical metric inspection)

Use connected components of source-study aliases, including original contributors.
Prespecified 2026 test papers are PMIDs 41534530 (II MS), 42069060 (I MS) and
42048449 (I KD-via-IC50), subject to actual admission. They were chosen for explicit
MHC typing and assay coverage, not model performance. A missing admission remains
an explicit unavailable cohort. Other 2026 papers remain eligible for training.

For older supplemental test and validation, greedily cover endpoint/class strata
using exact-label counts (physical values are not inspected). Target 64 exact
observations per quantitative stratum, 32 examples of each qualitative-binding
label, and 512 positive ligands per class. An additional reported-MS-outcome stratum
uses the minimum of within-study positive/negative counts, target 32 per class.
Only components whose known publication years are all before 2026 are eligible.
Choose maximum sum of fractional coverage gains divided by log2(2 + component
row count), breaking ties by stable component ID. Stop at coverage or 12 papers
per split. Supplemental test selection preserves at least two other contributing
components where it consumes support (train plus validation); validation preserves
at least one. Preserve at least min(64, half the remaining stratum labels) in train.
Sparse outputs may have unavailable held-out strata; never invent support.

Assign complete study components first, then embargo exact peptide identity with
test > validation > train priority. Drop overlapping lower-priority rows and record
every disposition. All held-out paper aliases remain excluded from train. Only
outputs with final training labels receive model heads. Report any unsupported
held-out output explicitly. Freeze resulting membership and counts before fitting.

## Fixed shared training and evaluation recipe

Two seeds (42, 43), no pretrained weights, one wide shared-v5 architecture
(192 width, 3 layers, 6 attention heads, 768 feedforward width, dropout 0.1).
Candidate groove pairs and full ordered ligand context use the common encoder.
Source-protein context and unsupported cellular interventions are not allocated;
their raw annotations are retained for stratification, not claimed as trained effects.

Relative label frequencies: 60% elution, 20% IC50, 15% other physical outputs,
5% reported qualitative binding, renormalizing when a family is absent. Within
each family, endpoint/class weights are proportional to the square root of unique
TRAIN counts. Canonical weighted quotas cover every unique row at least once.
Run one complete weighted epoch or 6,000 updates, whichever is larger, batch 128;
freeze the resulting integer budget after admission. AdamW 3e-4, weight decay
1e-4; mean per-observation losses, exact-TRAIN-only normalization, one-sided
squared censor loss and binary cross-entropy. Validation every 1,000 updates;
equal-endpoint validation loss selects the checkpoint. Disable early stopping
within this fixed budget. Full validation/test; only TRAIN diagnostic prediction
dumps are limited to a deterministic 16,384-row sample.

Backgrounds use seed 20261028, one comparator per TRAIN positive and five per
validation/test positive, with replacement across replicates. Both initialization
seeds use identical membership. The complete known-positive peptide blacklist
includes held-out sources only to conservatively exclude known ligands from
generated backgrounds; no held-out values or predictions enter optimization.
Same-protein comparisons are the primary ligand benchmark; fallback human-proteome
comparisons and source-reported nondetections are reported separately.

## Remote launcher recovery and final cohort gate

Admission attempt 4 failed during remote module import, before the worker function
or any data admission ran: a local checkout-relative upload path was evaluated
under `/root/admit.py`. Stop that app and restrict all three launcher upload paths
to `modal.is_local()`. Preserve the failed call, invocation and source snapshot.
Verify each module imports with remote path semantics before retrying admission.
The source build is complete and will not be repeated.

After materialization, the same CPU preparation worker runs the independent
source/target conservation, whole-study and peptide separation, background lineage,
ratio and TRAIN scale audit. Its passed receipt gates GPU launch. Return the small
selection/preflight/manifest metadata for local review; large row files remain on
the volume with hashes. Live validation shows both seeds side by side and separates
same-protein, fallback-proteome and source-reported ligand comparisons.

## Worker-local storage before preparation and fitting

The canonical observation index performs many guarded random reads. Mounted-volume
metadata latency is unnecessary for these reads. Before the queued preparation
worker starts, change only its storage placement: mount the durable volume at
`/volume`, stage the completed admission SQLite plus reference metadata under the
same canonical `/runs/FAMILY` paths on local disk, prepare and independently audit
there, then publish the complete prepared directory to the volume. All generated
manifest paths and bytes remain unchanged. Request512GiB ephemeral disk explicitly;
record staging/publishing bytes and elapsed time. Preserve the earlier queued
invocation, which never launched a preparation worker.

For each GPU, stage and hash-check frozen prepared source files on local disk under
those same `/runs/FAMILY` paths. The common trainer, source/manifest hashes and data
membership remain unchanged. Outputs/checkpoints stay on the durable volume. This
is an I/O placement change, not a dataset cap, precision change, or alternate trainer.

## Admission recovery after extended GIL/heartbeat failure

At18:39 UTC Modal warned that attempt5 may be marked unhealthy after17minutes of
failed worker heartbeats. Direct profiling found the main thread holding the GIL
in `RecordBatch.to_pylist()` at admit.py:247 during the metadata-only vocabulary
pass; the 5,440,009-row canonical export is complete and readable. Preserve a
consistent read-only SQLite backup of the completed binding/other admission and
known-positive index, source status, canonical export and failed attempt logs.
Stop the original only after the backup is durable. Recover from the verified
stage, not by redownloading sources or rebuilding the canonical indexes.

Investigate Arrow dictionary conversion on a tiny read-only sample. Project only
fields actually read by the vocabulary collector. Preserve every original field
for full MS admission, using a semantically identical dictionary-decoded conversion
if the probe confirms equivalence. Any replacement must match original row values,
source IDs and curation outputs. Keep the common scientific source snapshot fixed.
Regenerate early per-row dispositions if the interrupted gzip stream was unclosed;
check their aggregate counts against the saved stage receipt. No GPU fit or data
partition has started, so this recovery changes no predictive evaluation contract.

Recovery disposition at18:46 UTC: the original worker advanced to canonical MS
admission before the snapshot command ran. Its stage assertion prevented creating
or changing any backup/database; no app was stopped or relaunched. The original
local admission monitor ended on DNS lookup failure, but the detached remote call
remains active and publishes counts. The waiting preparation controller uses that
saved remote call directly, so its dependency remains valid. Profiling established
a costly full-column conversion but did not prove it caused network heartbeat
errors; keep these observations distinct. Reconnect logs and continue the existing
job rather than executing the now-stale recovery plan.

## Local-database recovery justified by measured admission throughput

Between source counters32,768 at1793213191 and131,072 at1793213512, admission
processed about306rows/s. The remaining5.31million rows would exceed the original
four-hour CPU invocation timeout. Before hitting that limit, make a consistent
read-only SQLite backup while the original continues. A snapshot can include a
partial MS pass. Recover on local512GiB disk with a1GiB SQLite page cache.

Revalidate all source rows in original order with the same frozen canonical
adapters. If the identical source-derived observation ID already exists, require
its semantic fingerprint and lineage to match, reuse it, and recompute counters.
Existing different-ID duplicates/conflicts follow the original rules. This avoids
inventing a resume ordinal from a nontransactional status file. Regenerate every
per-row disposition and recompute the metadata vocabulary from only the columns
its existing pure function actually reads. Reuse the complete canonical MS export
and known-positive index; do not rebuild sources. Preserve attempt5 as a separate
durable directory when publishing the successful replacement. The pending
preparation call must depend on the replacement admission call. No fit has started.

## Saved-prediction reporting specification

The following post-fit analyses do not change the frozen checkpoint or seed
selection. Require the independent prediction-verification receipt before reporting.
Use saved validation/test dumps only, with no additional held-out forward passes.

- Show every allocated endpoint and both MHC classes, including zero held-out
  support. Separate 2026 from older/mixed study components. Keep original counts,
  generated backgrounds, exact/bounded targets and optimizer exposures distinct.
- Plot direct IC50 Spearman and ligand AUROC/AUPRC for both seeds. Separate
  same-protein, fallback-proteome and reported study outcomes. Do not compare pooled
  scores to the older development cohort as if membership were the same.
- Tabulate prediction spread, mean error, censor satisfaction and scores by peptide
  length, candidate count and exact catalog allele set (minimum20 observations for
  detailed strata). A multi-allelic outcome is one bag, never replicated per allele.
  Compare exact-value RMSE to the endpoint's exact-TRAIN mean in the same target
  space. Constant-score ligand AUPRC equals the evaluation prevalence; retain that
  reference beside AUPRC rather than interpreting raw AP across different ratios.
- Summarize saved class-II register entropy and maximum weights per candidate,
  explicitly without measured register truth or physical-occupancy claims.
- Extract selected-checkpoint coordinate coverage, inactive inputs, public-serving
  replay and runtime. Distinguish structural padding/constant coordinates from
  exposed coordinates without updates, and quantify selected versus final exposure.
- Add uncertainty to the validation-selected seed's principal metrics using fixed
  seed20261028 and500 bootstrap replicates. Whole-study-component intervals require
  at least4 components; otherwise mark cross-study uncertainty unavailable. A
  separate within-cohort peptide-cluster interval may be shown with at least20
  peptide clusters, keeping each generated comparator with its positive parent's
  peptide. Such intervals are conditional on the observed studies and model;
  neither they nor a two-seed range establish uncertainty over new training seeds.
- Preserve JSON/CSV tables plus standalone plots under this experiment directory.
  Include the exact analysis invocation and source hash. Automated tables are
  evidence for the final writeup, not automatic claims of scientific qualification.

## Completed admission and local controller reconnect

The full5,440,009-row MS scan finished. Local admission/preparation CLI controllers
then exited with DNS errors outside their saved-call retry output; no preparation
call ID exists. Retrieve the original admission call and require its returned
`admitted` result (not only live status before final publication). Preserve the
attempt3 preparation source snapshot and failed log. Start preparation again from
that saved, completed dependency, without rebuilding or readmitting any source.
The continuation controller still waits for the single preparation call file.
There are no predictive fits to restart. The patched transport helper additionally
recognizes Modal's own ConnectionError class, verified against the installed SDK.

Before GPU submission, persist each successful call ID immediately in a partial
submission receipt. Publish the canonical two-call file only after both calls
exist, so the collector cannot mistakenly close a one-seed comparison. Refuse
automatic resubmission when any partial receipt exists; recover its saved IDs if
needed. This addresses recurrent local transport failure without changing either
fit, its inputs, or the maximum of two concurrently requested GPUs.

## Observed CPU preemption: durable stage and fit recovery

Preparation attempt4 materialized the cohort and reached canonical preflight,
then Modal preempted its worker and automatically restarted the same input.
Its first call-graph entry reports that attempt's failure; a pending result poll
is not proof that the overall call failed. Do not submit a second preparation
while the automatic retry is active. Capture completed, closed materialized files
if feasible; let the current retry continue while implementing future recovery.

Persist the complete materialization receipt and files before canonical preflight.
On a retry, validate hashes and reuse that materialization, then rerun the common
preflight and independent audit. Completed preparation should also return its
verified durable result idempotently. Preserve exact membership/order/backgrounds;
no new numerical or cohort policy. New CPU invocations may use Modal's documented
nonpreemptible option, recording its3x CPU/memory price multiplier, while GPUs
remain at most two H100! workers (nonpreemptible GPU is unsupported).

Before launching GPUs, bind retry to the existing canonical `resume.pt` machinery.
Preserve model/optimizer/RNG/order/position/selection/evaluation ledger. Persist
experiment observer state at the same durable snapshot boundary, including seen
TRAIN IDs, exposure counters, TRAIN replay probes and selected-exposure diagnostics.
Preemption recovery must roll those diagnostics back to the matching common
snapshot, never mix post-checkpoint observations into restored training state.
Completed artifacts should return the same receipt without another held-out pass.
Test interruption equivalence and phase recovery before launching actual fits.

Official behavior checked in Modal docs on2026-10-28:
https://modal.com/docs/guide/preemption and
https://modal.com/docs/examples/long-training .

## Post-preparation composition audit

The original metadata table counts `reported_candidates`, which exists for MS
bags but not physical-affinity restrictions. Add a separate metadata-only audit
of the exact manifest sources using canonical `catalog_alleles` for every endpoint.
Hash-check each input, skip generated comparators for original-observation tables,
and preserve split, endpoint, class, qualifier, study, exact candidate set, and
individual allele membership separately. Individual membership can overlap;
candidate-set and endpoint/class tables count each observation once. Save counts,
manifest/source hashes and analysis-code hash without peptide sequences or target
values. Run alongside the existing jobs using their staged read-only files;
do not modify fitted data, selected studies, model code or checkpoints. Verify
original/comparator totals against the independent preparation receipt before
using the new composition tables in the final report.

## Primary-source kinetic mismatch: stop and refreeze before fitting

The six TRAIN `kon` labels have an implausible scale. Four come from PMID21257290;
the primary article explicitly describes association kinetics in hour^-1, while
the exported rows label values528.84,704,500,500 as nM^-1s^-1. These are not the
same physical quantity. Keep the original export and all existing receipts;
stop the two preflight-only GPU calls (neither has an optimizer update or held-out
prediction), and preserve their call IDs, source snapshots, status and timing.

Inspect primary-source identity/units for the remaining sparse kinetic rates,
without reading peptide sequences or making biological design recommendations.
Record explicit original assay IDs and evidence for quarantine. Do not guess
corrected units, values or a bimolecular rate from a first-order curve.

Derive a revised materialized cohort from the verified prepared files. Preserve
every original study split, peptide embargo, background draw and all unaffected
row bytes. Remove only explicitly reviewed ambiguous measurements; retain their
complete dispositions and original cohort artifacts. Recompute TRAIN support,
scales, quotas, model output allocation and exact budgets through the common path;
drop any now-unsupported output. Independently verify the exact row-removal delta,
unchanged membership/order/targets outside that list and revised background totals.
Do CPU validation before submitting replacement H100! fits. Freeze a separate
revision contract and launch receipt; never overwrite the stopped attempt's record.
No model-output-based curation or test-guided selection has occurred.

The completed review quarantines20TRAIN records: six association labels (four
first-order scope conflicts from PMID21257290, two unresolved scale/scope conflicts
against PMID9831134), fourteen off-rates from PMID21257290 with incompatible
exported time conversion. The earlier note misread half-life assay1820987: its
export already has a `>` qualifier, consistent with the primary table. Preserve
all twelve half-lives from that paper and all unrelated outcomes. There are no
held-out changes. `kon` loses all support; `koff` retains8TRAIN outcomes.

Store assay-identity decisions, observed values/units/qualifiers and primary
references in a packaged shared JSON review registry. The common purified-assay
adapter defers those identities for every future source preparation, with explicit
reason codes and no numerical repair. Normalize only HTTP/HTTPS for the known
canonical IEDB assay namespace. Tests must cover each reviewed identity, changed
future labels requiring re-review, unrelated/neighboring assays, both IEDB/CEDAR
source routes, and unchanged literal unit arithmetic. Ensure the review registry
is included in release wheels. The reviewed cohort builder consumes this same
registry and verifies its exact20-row removal against original source hashes.

To avoid repeating the same long corpus-wide preflight on both CPU and GPU,
the revision's CPU gate will independently prove the exact removal delta,
all unchanged source hashes and retained row bytes/order, original split/background
inheritance, supported heads, and recomputed TRAIN scales/quotas. Mark canonical
preflight explicitly `pending_in_trainers`; do not copy the old passed status onto
the revised cohort. Both replacement workers still execute the unmodified common
full preflight before their first optimizer step. The launcher gate requires the
specific verified-delta receipt before allowing this declared state. There is no
cache or bypass of the common trainer. Preserve the entire original result and
reproduction directories before replacing current metadata with revision2.

The live report should show the reviewed TRAIN population (original and generated
rows separately), rather than leaving superseded pre-partition counts as its main
census. Publish only small existing composition summaries, retain two fit columns,
and label submitted jobs separately from optimizer progress. This is display-only;
no manifest, training worker or prediction contract changes.
