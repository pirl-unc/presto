# Focused Affinity Improvement Plan

Date: 2026-03-08

## Goal

Make the binding model quantitatively credible before scaling back to the full
Presto task stack.

Success means:
- exact `IC50` training on a focused class-I panel produces the correct
  allele-specific ordering on fit-supported peptides and generalization probes
- weak/non-binding probes for incompatible alleles move toward the right scale
  rather than only the right sign
- the improvements survive small changes in seed and panel composition
- adding presentation/immunogenicity later does not break affinity

## Current State

What is working:
- the model now consumes groove-segment sequences, not allele-ID embeddings
- the direct affinity path now sees `interaction + peptide + mhc_a + mhc_b + groove`
- exact `IC50` focused training with strict A0201/A2402 batch balance gets the
  correct direction on `SLLQHLIGL`, `FLRYLLFGI`, and `NFLIKFLLI`

What is not good enough:
- quantitative gaps are still too small for some probes
- `A*24:02` is still not pushed weak enough on the A0201-favored cases
- the focused artifact logger still emphasizes `KD_nM` instead of the actual
  supervised `IC50_nM` output

## Design Principles

1. Match the training contract to the question.
   - binding debugging should use exact quantitative affinity rows
   - the primary metric should be the exact assay output being supervised

2. Add complexity one factor at a time.
   - no combining new synthetics, new ranking losses, and new panels in one step

3. Keep the MHC input biologically grounded.
   - use groove segments as the canonical MHC input
   - avoid switching to learned allele-ID embeddings as the primary model path

4. Use pretraining only to improve encoder initialization, not to blur the main objective.
   - the requested MHC-only pretraining should be short, explicit, and easy to ablate

## Phase 0: MHC Encoder Warm Start

### Objective

Run one epoch of MHC-only pretraining on all available indexed MHC sequences so
the shared encoder starts with a stronger notion of:
- class: `I` vs `II`
- species category used by the network

This is not meant to solve affinity. It is a cheap initialization step to make
the groove encoder less random before affinity fitting.

### Data

Source:
- `data/mhc_index.csv`

Examples:
- all valid groove-parsed rows currently accepted by the runtime/index path
- include all species available in the index
- keep parsing/index species distinct in the source data, but map to the
  network's current species-category targets at label time

Inputs:
- `groove_half_1`
- `groove_half_2`
- `mhc_class`
- species category label derived from the canonical resolver/bucket mapping

Exclusions:
- rows with unusable groove extraction
- rows with invalid sequence content

### Model path

Reuse the same MHC/groove token path already used by Presto:
- groove segments
- segment embeddings
- groove positional embeddings
- shared stream encoder

Use lightweight heads only:
- class head
- species-category head

Do not involve:
- peptide input
- affinity head
- processing/presentation/immunogenicity heads

### Training contract

- 1 epoch only
- balanced batches across:
  - class
  - species category
- standard cross-entropy losses
- save checkpoint as an initialization checkpoint for the focused affinity runs

### Verification

- class accuracy well above chance
- species-category accuracy well above chance
- no degradation in the focused affinity path relative to random init on a
  tiny smoke run

### Risk

If this pretraining is too strong or too long, it can over-shape the encoder
toward coarse classification instead of interaction learning. That is why this
phase is explicitly one epoch and must remain ablatable.

## Phase 1: Clean Focused IC50 Baseline

### Objective

Establish a stable exact-assay baseline before adding any additional training
pressure.

### Dataset

Use all available exact `IC50` rows for:
- `HLA-A*02:01`
- `HLA-A*24:02`

Contract:
- exact rows only (`qualifier = 0`)
- no synthetic negatives
- no row cap
- peptide-family train/val split
- strict per-batch allele balance during training

### Output target

Primary supervised target:
- `assays["IC50_nM"]`

Diagnostics to log every epoch:
- `IC50_nM` for:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- `KD_nM` and `binding_affinity_probe_kd` only as secondary debugging outputs

### Required logging

Add to the focused artifact set:
- `IC50_nM`
- `IC50_log10`
- train/val loss
- gradient norms for:
  - affinity head
  - binding core
  - shared trunk

### Acceptance

- correct direction on all three tracked peptides under `IC50_nM`
- stable loss curve
- nonzero shared-trunk gradients

## Phase 2: Calibration and Ceiling Audit

### Objective

Determine whether the remaining under-separation is caused by target scaling and
weak-binder compression.

### Experiments

Run the exact same focused A0201/A2402 baseline with:
1. `max_affinity_nM = 50_000`
2. `max_affinity_nM = 500_000`
3. unclipped exact values in the focused path

Keep everything else fixed:
- same data
- same batch balance
- same loss mode
- same seed set

### Outputs to compare

- final and best-epoch `IC50_nM` on tracked probes
- validation loss
- distribution of predictions for weak binders

### Acceptance

Use the scaling that:
- preserves stable optimization
- improves weak-binder separation
- does not break fit-supported peptides

## Phase 3: Multi-Allele Generalization Panel

### Objective

Test whether the groove-segment path is learning a general class-I binding model
instead of a narrow A0201-vs-A2402 separator.

### Panel

Build a small exact-IC50 panel of abundant, motif-distinct class-I alleles.

Candidate strategy:
- include A0201, A2402
- add several alleles with clearly different anchor preferences and enough exact
  `IC50` support

Selection criteria:
- direct exact `IC50` support is sufficient
- motifs are diverse
- species/class remain fixed to reduce confounds at first

### Contract

- exact `IC50` only
- no synthetics
- no ranking losses initially
- strict per-batch allele balancing across the whole panel

### Acceptance

- fit-supported probes for multiple alleles move in the right direction
- A0201/A2402 performance does not regress badly

## Phase 4: One-at-a-Time Additional Pressures

Only start this phase if the clean multi-allele baseline is stable.

### 4A. Same-allele / different-peptide ranking

Purpose:
- sharpen peptide specificity within an allele

Keep:
- no synthetic negatives
- no same-peptide cross-allele ranking yet

Acceptance:
- improves fit-supported peptide separation
- does not damage generalization probes

### 4B. Conservative synthetic negatives

Purpose:
- provide clean weak-binding pressure where real negatives are sparse

Start only with:
- anchor-broken decoys from strong exact binders

Do not start with:
- broad random scramble
- mixed synthetic modes

Acceptance:
- improves weak-binder calibration
- does not create obvious false-negative artifacts

### 4C. Same-peptide / different-allele ranking

Purpose:
- only if the prior two steps are already working

This stays last because it previously amplified allele priors instead of peptide
specificity.

## Phase 5: Architecture Escalation if Needed

Only do this if the cleaner data/training contract still plateaus.

Potential changes:
- make the affinity head consume more explicit pocket-level peptide/groove
  features
- reduce over-smoothing in the affinity readout
- expose anchor-position interaction features directly

Do not do this until the exact-assay baseline, scaling audit, and multi-allele
baseline are complete.

## Phase 6: Scale Back to Full Presto

Once focused affinity is stable:
1. add more alleles if not already done
2. add stability outputs
3. add processing
4. add presentation
5. add immunogenicity last

Rule:
- every added component must be shown not to damage affinity

## Questions This Plan Answers

### Are these allele embeddings or groove segments?

Groove segments.

The model uses MHC groove-half sequences with segment/position embeddings. The
allele name is currently used for resolution, batching, and diagnostics, not as
a learned identity embedding in the main path.

### Would adding more alleles tell us if this is working?

Yes.

Two alleles are necessary for debugging but insufficient to prove the model is
learning general groove logic. A small motif-diverse panel is the next correct
test after the focused A0201/A2402 baseline is stable.

### Should we try synthetic negatives or contrastive terms one at a time?

Yes.

That is mandatory. The previous runs already showed that:
- same-peptide cross-allele ranking can hurt
- synthetics can help one probe and hurt another

So each pressure must be added in isolation with the exact same baseline held
fixed.
