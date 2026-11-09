# Nontryptic known-amount calibration screen

**Closed source screen: separate nominal-dose and raw-response data exist for
18 selected peptides, but training admission remains incomplete.** The prospective
support threshold passes. Exact acquisition joins and conflicting isotope metadata
require shared curation in [Hitlist849](https://github.com/pirl-unc/hitlist/issues/849).
This does not establish a calibrated APC availability model or an RL advantage.

## Question and source contract

Date: 2026-10-25. Agent/model: Codex/GPT-6. Starting Presto commit
`a9ae724b026ae3f278ef11e11b31e3f26ef9e11d`; unrelated tracked task notes and
historical untracked inputs were preserved. No model code changed.

After the frozen organ transfer failed its no-change comparator, ask whether a
broader nontryptic standard panel can constrain the observation component. This
screen extends the prior nine-standard nominal-dose source; it does not reopen
F3, F2 or native biological held-outs or select data using prediction quality.

Primary study: [Stopfer et al., PNAS 2021](https://doi.org/10.1073/pnas.2111173118),
PMID34497125, PMC8449407, PRIDE PXD024917. Eighteen selected human tumor-antigen
sequences were chosen as predicted HLA-A*02:01 binders. They are not an unbiased
peptide sample, measured binding labels or proof of every sample's allele typing.
The paper distinguishes SKMEL5 melanoma/binimetinib experiments from calibration
spikes and patient biopsies. This screen inspects the calibration tables only.

Raw sources live once at
`data/nontryptic_calibration_sources/2026-10-25_1816_codex_nontryptic-calibration-screen/`.
The official PMC inventory, metadata JSON and article XML establish exact media
URLs and MD5s. Retrieved supplements: S1 panel, S3 quantities, S6 isotope/method
definitions, S7 filenames and the seven-page methods PDF. See fetch receipts for
URLs, SHA256, byte counts and timing. No raw spectra were downloaded.

## What is measured

| Route | Source | Sequences | Reported dose rows | Positive signal cells | Blank signal slots |
|---|---|---:|---:|---:|---:|
| Free 1H peptide added after HLA-peptide purification | S3A | 18 | 71 | 213 | 0 |
| Embedded 2H hipMHC added before immunoprecipitation | S3B | 9 | 45 | 126 | 9 |

S3A mostly contains 0.1/1/10/100 fmol. GLFDQHFRL[+7], A16:B18, reports only
1/10/100 fmol: the omitted low-dose row is not a zero and has no source coordinates.
S3B contains 0.1/1/10/100/1000 fmol. All 18 S3A sequences have at least three
positive dose levels, satisfying the unchanged prospective minimum of 12 sequences.
Columns C/D/E are ordinal measurement slots; they do not identify acquisitions.
All 348 actual source slots, including nine blanks, independently match raw OOXML.

The response is summed selected fragment-ion signal under targeted SureQuant/
IS-PRM, not MS1 response. S3C contains selected-transition counts, not intensities.
S1 affinity values are predictions. Free-peptide content is reported as assigned
by amino-acid analysis, HPLC purity >97%, with isotopologue balancing by LC/MS.
Embedded-complex concentrations use reported stable-complex ELISA after exchange.
Neither has independently certified amount uncertainty in the retrieved evidence.
The 4H trigger and its dose belong to acquisition metadata, not biological inputs.

The free-spike route can constrain downstream response in that matrix. Embedded
spikes additionally include capture/purification. Different isotope and spike
designs do not isolate causal recovery loss on their own. Neither route directly
measures intracellular free-peptide amounts, precursor-relative production or
APC concentrations. Source-reported ratio compression also prevents assuming
perfect signal/dose proportionality without a separate registered calibration fit.

## Unresolved metadata

- S7 names `Iso18_DirectSpike_TransitionResults.xlsx` and, with that literal
  spelling, `Iso18_hipMHCspiikein_TransitionResults.xlsx`. Retrieve these through
  shared Hitlist ingestion to establish acquisition links, quantified transitions,
  aggregation/interference and missingness. Numbered raw filenames alone do not
  prove C = run1, D = run2, E = run3.
- S6A A10:C10 lists valine with a 4.007-Da isotope offset, whereas S3A uses
  V[+6] and S6B reports corresponding nominal shifts. Comparing sums of S6A
  masses with S6B's nominal shifts flags 54 of 72 isotopologue records. This is
  one systematic source inconsistency, not 54 independent chemical failures.
- One additional S6B record is uncheckable by S6A alone: B99 includes labeled
  methionine, omitted from S6A. S6B E99 does report an isotope assignment for M.
  The remaining 17 records agree within the stated 0.15-Da nominal-rounding check.
  No canonical chemical identity was chosen or source value repaired.
- S6C supplies method/trigger transitions. Their identity with the exact ions
  summed in S3 is unproven. Acquisition, aliquot, dose, matrix, target selection,
  protocol and isotope assertions must stay separate from biological observations.

## Verification and preserved failures

Four workbooks/eight sheets: **18,911 stored cells** independently replayed through
OOXML and openpyxl, including cached formulas and original blanks. Most S1 cells
are formatting-only; this count is not a count of measurements. Official MD5s
and download SHA256 receipts agree. PDF text was extracted with two installed
parsers and pages2–5 visually checked; extraction/tokenization differences remain
recorded. Source workbooks/PDF were never rewritten or re-exported.

Ten guarded stages and every attempt are retained:

| Stage | Outcome |
|---|---|
| fetch1–4 | Eight official payloads retrieved and validated |
| inspect1 | Missing `fitz` import; stopped before source access |
| verify1 | Existing PDF tools and independent cell readers pass |
| analyze1 | Isotope inventory passes; overstrict full-dose-grid assertion fails |
| diagnose1 | Original block layout identifies the omitted dose row |
| reconcile1 | Dose extraction passes; isotope audit fails on absent S6A methionine |
| audit1 | All 348 cells and all 72 isotope records audited, unknowns retained |

See `reproduce/dependency-recovery.md`, `layout-recovery.md`, `semantic-recovery.md`
and `identity-audit-recovery.md` for sequential corrections. Executed sources,
failed receipts and partial outputs remain immutable. The initial source-preparation
lint warning and a prelaunch transformation assertion are also disclosed in the
dependency-recovery note. The final source metadata audit passes; earlier failed
attempts do not become successful retroactively.

Local CPU, one numerical thread, Python3.12.6. No Modal/GPU, installations or
shared-environment modifications. The limits were 20min/stage, 3GiB startup
free+speculative memory and process-tree RSS, 1GiB disk reserve plus 32MiB startup
margin, and 50ms resource polling with OS child-peak checks. Per-stage measurements
and aggregate runtime are in `results/closure-summary.json`. Browser discovery
has no guarded timing/byte evidence; do not infer total resource compliance from
the guarded fetches.

The single inclusive discovery budget is exhausted: two searches, two unsuccessful
reCAPTCHA article opens, eight official payload requests = 12 operations. No further
scientific sources were requested in this family. GitHub CI/issue operations are
software coordination, not additional source discovery.

## Training and reproduction

No training/pretraining, synthetic labels, fitting loss/weights, model parameters,
predictive validation/test split, predictions or biological metrics apply to this
source-only screen. IC50/500nM classification metrics are inapplicable. No model is
preferred or promoted. The source support decision does not satisfy model quality.

Original invocation/config/environment is frozen in `reproduce/launch.json` and
each `reproduce/*-launch.json`; exact requests and source/input hashes are under
`reproduce/requests/`. The four fetch calls used `reproduce/launch.sh fetch` with
attempts1–4 and corresponding request files. The original inspect/verify calls
used that launcher, attempt1. Later stages used `code/launch_analysis.py`,
`launch_diagnose.py`, `launch_reconcile.py`, and `launch_audit.py` with respectively
`analyze`, `diagnose`, `reconcile`, `audit`, all attempt1 and the matching request
file. All were prefixed `PYTHONSAFEPATH=1` and the configured absolute Python path.
Frozen copies are in `reproduce/source/`; guards refuse overwriting an attempt.
For independent replay, create a fresh experiment directory and register the
copied frozen source/config with new output paths; retain the exact raw hashes.
Do not rerun these immutable output directories in place.

Status: source screen closed, nominal dose/signal support qualified, training
metadata admission incomplete. Next step: shared Hitlist849 source ingestion and
an explicit empirical-response recipe retaining form uncertainty and peptide-level
dependence, followed by a separately registered held-out calibration comparison.
Open questions: sequence/protocol transfer, native observation calibration and
machinery-conditioned processing remain unresolved. The full precursor-relative
APC availability goal and matched differentiable/discrete biological comparison
remain active; prior biological held-outs stay closed.
