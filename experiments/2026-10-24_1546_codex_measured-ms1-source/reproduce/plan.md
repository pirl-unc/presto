# Qualify measured peptide MS1 response as an independent observation constraint

October24,2026; Codex / GPT-6. Prior HAP1 held-out elution prediction is positive
for a small sequence model but gains no benefit from processing augmentation.
All existing HAP1/native/macrophage/MAPP tests remain closed to tuning. This audit
targets independent detector-response evidence required by the original architecture,
not a replacement of the APC availability objective.

## Hypothesis and source selection

Measured, nominally equimolar synthetic peptide pools may provide much broader
sequence-dependent MS1-response supervision than the already qualified22-sequence
PXD004886 slice. Chemically synthesized peptides measured on an instrument are
experimental observations, distinct from fabricated numerical examples.

Primary lead: Abdul-Khalek et al.2023 DOI10.1016/j.csbj.2023.07.027 / PMC10407266,
which derives MS1 intensities from PXD004732, PXD010595 and PXD021013. Prefer the
nontryptic human-peptide arm when supported, while preserving tryptic context.
The review/reanalysis is a discovery route; source acquisitions and planned pools
must identify measurement origin. A second candidate, Jarnuczak2016
DOI10.1021/acs.jproteome.6b00048 / PXD003472, estimates response from known protein
amounts and digestion. It cannot silently substitute for known peptide loads.
Existing Hitlist762 owns response-data admission. Check current implementation and
issues before proposing additional shared work; do not create a duplicate request.

## Read-only qualification before training

1. Retrieve original public article XML and linked supplementary/index metadata;
   preserve URLs/bytes/checksums/access status/license and source version.
2. Locate author-deposited numeric data/code and PRIDE file manifests, planned pool
   composition and acquisition relationships. Inventory actual files rather than
   assuming a publication's dataset description is ready-to-train data.
3. Inspect bounded original processed peptide tables if available (up to100MiB
   per file,500MiB total). Preserve raw input data under data/, not in experiment
   artifacts; no large raw-MS download or spectra reprocessing in this audit.
4. Determine whether nominal peptide amount, measured/certified amount, synthesis
   purity/yield, form/charge, pool, replicate/run, instrument/acquisition/LC,
   integration, PEP/FDR, search settings and omitted/zero/failed rows are supplied.
   Enzyme specified for an in-silico search is not evidence of laboratory digestion.
5. Separate experimental measurements from artificial proof-of-concept datasets.
   Preserve selection on recurrence/CV/intensity and all original dispositions.
   Report whether targets are conditional on successful identification and whether
   missing planned peptides can be distinguished from unqueried peptides.
6. Check original-vs-derived numeric conservation where a published derivative is
   available. Retain all dependencies needed for peptide/pool/run grouped splits;
   do not make train/test partitions or inspect new model outcomes in this audit.
7. Record exact remaining gap/admission criteria in Hitlist762. A positive finding
   permits a separately registered canonical Hitlist source/recipe and model test,
   not immediate application of fitted coefficients to APC data.

## Modeling contract to prepare, not execute here

Proposed smallest downstream test predicts log10 measured MS1 intensity from
sequence plus supplied nominal load and acquisition/pool context, against a
load/context-only reference. Whether the response should instead be represented
as within-pool contrasts depends on verified load metadata. All assay gains and
nominal-load uncertainty remain explicit. No guaranteed absolute concentration
or transfer between instruments/matrices is assumed. A separate detectability
head requires a known planned candidate universe; absent IDs alone are insufficient.
Only after independent response prediction succeeds should a frozen response term
be combined with processing/selection and tested on a new biological cohort.
Exact versus sampled processing must use the same independently anchored observer.

## Bounds and closure

Local source audit; no optimizer, pretraining, loss, synthetic training labels,
validation/test prediction dump or binding metrics. No shared environment changes,
cloud compute, author contact or access-control bypass. Public network requests
timeout60s; each inspection capped900s/4GiB. Cap bounded retrieval to500MiB and
record partial coverage explicitly; giant PRIDE archives remain indexed only.
Freeze plan/invocations/code/source hashes, original access failures and derived
summary/verification in a timestamped experiment. Close negative or partial
admission without calling broad search exhaustive. Availability goal stays active.
