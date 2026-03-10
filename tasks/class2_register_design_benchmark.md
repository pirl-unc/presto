# Class-II Register Design Benchmark Plan

Date: 2026-03-08

## Goal

Find a way to add MHC-II quantitative affinity learning without damaging the
class-I affinity behavior that is finally starting to work.

The benchmark should answer two questions in order:

1. Which binding-core / register designs preserve or improve class-I binding on
   the known-good small class-I quantitative panel?
2. Of the class-I-safe designs, which also work well once class-II quantitative
   affinity rows are introduced?

This benchmark is explicitly about the **binding mechanism**, not the full
presentation / immunogenicity stack.

## Non-Negotiable Constraints

- Preserve the current winning training contract as much as possible:
  - MHC warm-start pretraining on class / species
  - quantitative affinity data only
  - no same-peptide cross-allele ranking
  - use peptide-ranking as the first regularizer if ranking is enabled
- Evaluate class-I preservation before adding class II.
- Compare designs head-to-head under the same optimization recipe unless the
  design itself requires a different inference procedure.
- Judge designs on real assay outputs (`IC50_nM`, `KD_nM`, `EC50_nM`), not only
  probe-only latent readouts.

## Fixed Benchmark Recipe

Unless a design explicitly requires otherwise, start from:

- initialization:
  - 1 epoch MHC-only warm-start pretraining
- optimizer:
  - current focused affinity optimizer and scheduler defaults
- losses:
  - affinity loss mode `full`
  - peptide-ranking enabled
  - same-peptide allele-ranking disabled
- augmentations:
  - start with no synthetic negatives
  - only add synthetic negatives in a second-pass ablation
- batching:
  - balanced across alleles within the chosen panel
- data:
  - quantitative affinity rows only

## Benchmark Datasets

### Stage A: Class-I Preservation Set

Small motif-diverse class-I panel that already behaved sensibly:

- `HLA-A*02:01`
- `HLA-A*24:02`
- `HLA-A*03:01`
- `HLA-A*11:01`
- `HLA-A*01:01`
- `HLA-B*07:02`
- `HLA-B*44:02`

Primary data slice:
- exact `IC50` first

Secondary class-I expansion:
- all quantitative `IC50`, `KD`, `EC50`

Key class-I probes:
- `SLLQHLIGL`
- `FLRYLLFGI`
- `NFLIKFLLI`

### Stage B: Class-I + Class-II Joint Quantitative Affinity

Add class-II quantitative rows after class-I screening:

- all class-II quantitative rows
- initial class-II focus alleles:
  - `HLA-DRB1*01:01`
  - `HLA-DRB1*04:01`
  - `HLA-DRB1*15:01`
  - `HLA-DRB1*07:01`
  - representative DQ pairs with enough data

Initial class-II slice order:
- exact `IC50` first
- then all quantitative `IC50`, `KD`, `EC50`

## Design Axes

The designs should be treated as combinations of three mostly independent axes:

1. groove positional encoding
2. core/register inference mechanism
3. class sharing strategy

This avoids testing arbitrary bundles without learning which component matters.

## Axis 1: Groove Positional Encoding Variants

### G0. Sequential per-half baseline

Current design:
- groove half 1: learned sequential positions
- groove half 2: learned sequential positions

Purpose:
- baseline only

Expected issue:
- too weak for pocket-aware register inference

### G1. Peptide-like within-half triple encoding

For each groove half:
- start-distance embedding
- end-distance embedding
- fractional-position MLP

Purpose:
- cheap stronger absolute position baseline

Pros:
- easy drop-in upgrade
- likely better than pure sequential positions

Cons:
- still does not explicitly represent pocket/landmark semantics

### G2. Landmark / pocket-distance encoding

For each groove residue:
- distance to a small set of learned or canonical landmarks
- optionally bucketed distances

Possible landmark choices:
- 4 per half
- 9 total contact landmarks
- learned landmark queries over groove residues
- profile-coordinate landmarks from a class-specific groove MSA / HMM

Important clarification:
- these landmarks should not be treated as single universal residues that are
  conserved across all MHC-I and MHC-II alleles
- non-classical molecules and non-model species break that assumption
- in practice, "landmark" should usually mean one of:
  - a stable coordinate in a groove profile alignment
  - a coarse structural region boundary
  - a learned query over homologized groove coordinates

Purpose:
- encode where residues sit relative to pocket-like regions

Pros:
- better biological inductive bias

Cons:
- requires choosing or learning landmarks cleanly

### G3. Groove-slot embeddings from the MHC sequence

Derive `K` slot vectors from the groove sequence:
- slot queries cross-attend to groove residues
- each slot is an allele-specific contact embedding

These slots are themselves the learned landmarks, so this design does not
require any hard-coded universal motif residues.

Purpose:
- make peptide alignment target explicit slots, not raw residue indices

Pros:
- natural fit for register models
- better abstraction than raw residue positions

Cons:
- larger change than G1/G2

### G4. Pairwise peptide-slot relative bias

Augment G2/G3 with explicit relative features:
- peptide position vs slot index
- peptide terminal distance vs slot location
- optional class-conditioned relative bias

Purpose:
- help the model distinguish anchor-like alignments from implausible ones

Pros:
- directly relevant to register assignment

Cons:
- only useful once slot/register structure exists

## Axis 2: Core / Register Inference Variants

### R0. Fixed 9-mer contiguous window

Current baseline:
- contiguous 9-mer
- sliding start
- terminal PFR summaries

Best for:
- baseline only

### R1. Variable contiguous window

Enumerate `(start, core_len)`:
- class I: allow `8, 9, 10, 11`
- class II: strongly favor `9`

Class-specific priors:
- class I penalizes long terminal PFRs and partial occupancy
- class II allows long terminal PFRs

Pros:
- easiest extension of current code

Cons:
- class-I bulges are still approximated as terminal-overhang behavior

### R2. Shared proposal + class-specific refinement

Shared candidate proposal:
- contiguous `(start, core_len)` lattice

Class-specific refinement:
- class-I rescoring head
- class-II rescoring head

Pros:
- low-risk
- good migration path

Cons:
- still limited if proposal lattice is too simple

### R3. Explicit groove-slot monotonic alignment

Align peptide positions to groove slots with monotonic constraints:
- terminal flanks allowed
- no internal insertions yet

Class I:
- near-full occupancy prior

Class II:
- contiguous 9-slot occupancy plus PFR prior

Pros:
- first truly unified register model

Cons:
- still misses class-I bulges

### R4. Groove-slot alignment with class-I insertion/bulge state

Extend R3 with explicit insertion states:
- peptide positions may skip slot assignment in central regions

Class I:
- insertion/bulge states enabled

Class II:
- insertion state heavily penalized or disabled

Pros:
- best biologic unification in a still-structured model

Cons:
- more complex inference

### R5. Soft monotonic slot attention

Learn a soft assignment of peptide positions to slots:
- monotonicity regularization
- compactness regularization
- class-specific occupancy penalties

Pros:
- simpler to implement than DP/CRF

Cons:
- easier to learn blurry or degenerate alignments

### R6. Full peptide-to-groove residue alignment

Explicit pairwise alignment between peptide positions and groove residues.

Pros:
- most expressive

Cons:
- too many degrees of freedom
- highest risk of poor optimization
- not recommended as the first explicit alignment model

## Axis 3: Class Sharing Variants

### C0. Fully separate class-I and class-II binding modules

Shared trunk only.

Pros:
- lowest risk to class-I performance

Cons:
- least elegant

### C1. Shared proposal, class-specific calibration heads

Shared register/core mechanism.

Separate:
- class-I affinity calibration
- class-II affinity calibration

Pros:
- likely necessary even in unified models

### C2. Shared proposal, class-specific refinement and calibration

Shared proposal / alignment lattice.

Separate:
- refinement scoring heads
- calibration heads

Pros:
- likely best practical compromise

### C3. Fully shared binding module

Everything shared, class only enters as context.

Pros:
- elegant

Cons:
- highest risk of class interference

## Recommended Design Set To Actually Test

Do not test every cross-product. Start with these:

1. `Baseline`
- `G0 + R0 + C1`
- current contiguous 9-mer model with class-specific calibration

2. `Cheap positional upgrade`
- `G1 + R1 + C1`
- first low-cost upgrade

3. `Proposal/refinement compromise`
- `G1 + R2 + C2`

4. `First explicit unified register model`
- `G3 + R3 + C2`

5. `Unified register + class-I bulges`
- `G3 + R4 + C2`

6. `Pocket-aware upgrade`
- `G2/G4 + R4 + C2`

Optional later:

7. `Soft alignment variant`
- `G3 + R5 + C2`

Do not start with:
- `R6`
- `C3`

## Active Benchmark Matrix

These are the concrete near-term models to implement and compare head-to-head.

1. `M0 Baseline`
- `G0 + R0 + C1`
- current model family

2. `M1 Low-cost upgrade`
- `G1 + R1 + C1`
- variable contiguous windows
- class-conditioned penalties against class-I partial cores / long terminal flanks

3. `M2 Shared proposal + class-specific refinement`
- `G1 + R2 + C2`

4. `M3 First explicit unified register model`
- `G3 + R3 + C2`

5. `M4 Unified register with class-I bulges`
- `G3 + R4 + C2`

6. `M5 Pocket-aware register model`
- `G2 + G4 + R4 + C2`

7. Optional `M6 Soft alignment`
- `G3 + R5 + C2`

Order of implementation:
- `M0` -> `M1` -> `M2` -> `M3` -> `M4`
- only build `M5` and `M6` if simpler variants plateau

## First Executable Sweep

Run the first head-to-head sweep on the executable designs only:

- `M0 = G0 + R0 + C1`
  - `groove_pos_mode=sequential`
  - `binding_core_lengths=9`
  - `binding_core_refinement=shared`
- `M1 = G1 + R1 + C1`
  - `groove_pos_mode=triple`
  - `binding_core_lengths=8,9,10,11`
  - `binding_core_refinement=shared`
- `M2 = G1 + R2 + C2`
  - `groove_pos_mode=triple`
  - `binding_core_lengths=8,9,10,11`
  - `binding_core_refinement=class_specific`

Stage-A launch contract:
- seeds: `41, 42, 43`
- epochs: `12`
- batch size: `128`
- alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- warm start:
  - `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- data filter:
  - `measurement_profile=direct_affinity_only`
  - `measurement_type_filter=ic50`
  - `qualifier_filter=exact`
- training recipe:
  - `affinity_loss_mode=full`
  - `binding_peptide_contrastive_weight=0.5`
  - `binding_contrastive_weight=0.0`
  - `synthetic_negatives=false`
  - `balanced_batches=true`
- probe peptides:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`

Live leaderboard rule:
- evaluate each run at its best validation-loss epoch
- primary rank:
  - number of probe orderings correct with at least `1.5x` separation
- secondary rank:
  - mean absolute log-ratio margin on the three probes
- tertiary rank:
  - best validation loss

Update cadence:
- poll `summary.json` and `probe_affinity_over_epochs.csv` from the Modal
  checkpoint volume every few minutes
- rewrite a local leaderboard after each successful poll
- report a new leader only when the ordering score or validation tie-breaker
  improves

## What Counts As A Usable Landmark

Avoid assuming there are universal motif residues shared by all groove
sequences. The local parser already shows the limitation of motif thinking:
- Cys-based anchors are useful for full-chain groove extraction
- they are not reliable universal runtime landmarks inside the extracted groove
- some alleles and species lose, shift, or fragment those motifs

Prefer these landmark definitions, in order:

1. **Homologized groove coordinates**
- because groove extraction already produces roughly aligned halves
- simplest landmarks:
  - half start
  - half end
  - fractional position
  - coarse floor / helix region buckets

2. **Profile-alignment coordinates**
- build class-specific groove-only MSAs or profile HMMs
- define landmarks as profile columns or column groups
- robust to residue substitution

3. **Learned slot landmarks**
- use learned queries to derive contact slots directly from groove residues
- likely best for the unified register model

4. **Pocket pseudosequence positions**
- use only as an auxiliary benchmark for classical human alleles
- do not make this the universal main representation

## Stage A Screening Rules: Class-I Only

A design survives Stage A only if it preserves class-I behavior on the small
panel.

Required:
- does not regress the known class-I probes materially
- keeps correct ordering on:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- keeps validation stable
- trains without obvious collapse / degenerate alignments

Track:
- per-epoch probe `IC50_nM`
- best checkpoint, not only final epoch
- alignment/register diagnostics where available
- gradient norms

Primary quantitative metrics:
- validation loss on the held-out class-I quantitative set
- per-allele Spearman on `log10(nM)` for exact rows
- pairwise ordering accuracy / concordance within allele
- weak-binder calibration:
  - fraction predicted `>500 nM`
  - fraction predicted `>5000 nM`

Primary biologic probes:
- `SLLQHLIGL`: expect `A*02:01 << A*24:02`
- `FLRYLLFGI`: expect `A*02:01 << A*24:02`
- `NFLIKFLLI`: expect `A*24:02 << A*02:01`

Required class-I preservation rule:
- a design is rejected if it materially regresses the best current class-I
  checkpoint on probe ordering or collapses weak-binder calibration

## Stage B Screening Rules: Joint Class-I + Class-II

Only Stage-A survivors enter.

Required:
- class-I probes do not materially regress
- class-II affinity training is numerically stable
- class-II register/core outputs are interpretable

Track:
- class-I probe panel
- class-II DR/DQ probes
- register posterior sharpness
- expected PFR lengths
- class-specific calibration quality

Primary quantitative metrics:
- separate class-I and class-II validation losses
- separate class-I and class-II per-allele Spearman on `log10(nM)`
- class-I regression relative to Stage A winner
- class-II exact-row ranking/concordance

Primary structural diagnostics:
- posterior entropy over register/core states
- expected N/C flank lengths
- class-I occupancy vs insertion usage
- class-II contiguous-core preference

Rejection rule:
- if a design improves class-II but materially regresses class-I probe behavior,
  it is not acceptable as the shared canonical binding mechanism

## Evaluation Protocol

### Stage A: Class-I preservation benchmark

For each model `M0..M5`:
- train on the small class-I panel with the fixed recipe
- run at least 3 seeds
- keep best checkpoint by validation loss
- save:
  - summary JSON
  - per-epoch probe CSV
  - probe plots
  - if applicable, register diagnostic plots

Output table columns:
- model id
- groove encoding
- register mechanism
- class sharing mode
- best val loss
- mean Spearman
- pairwise ordering accuracy
- `SLLQHLIGL` ratio `A24/A02`
- `FLRYLLFGI` ratio `A24/A02`
- `NFLIKFLLI` ratio `A02/A24`
- pass/fail

### Stage B: Joint class-I + class-II benchmark

Take only Stage-A passers.

For each surviving model:
- train on combined class-I + class-II quantitative data
- keep class-balanced and allele-balanced batching
- evaluate separately on:
  - class-I validation
  - class-II validation

Output table columns:
- model id
- class-I best val loss
- class-II best val loss
- class-I Spearman
- class-II Spearman
- class-I probe preservation score
- class-II register interpretability score
- pass/fail

## Implementation Order

1. Add benchmark harness for class-I-only model-family comparison.
2. Implement minimal `G1 + R1 + C1` variant first.
3. Implement `G1 + R2 + C2`.
4. Implement `G3 + R3 + C2`.
5. Implement `G3 + R4 + C2`.
6. Promote class-I-safe variants into joint class-I + class-II training.

## Expected Best Near-Term Path

Most likely near-term winner:
- `G3 + R4 + C2`

Meaning:
- groove-slot embeddings derived from the groove sequence
- explicit monotonic register alignment
- class-I insertion/bulge state
- shared proposal with class-specific refinement and calibration

Most likely fastest low-risk baseline:
- `G1 + R1 + C1`

## Deliverables

- one benchmark script that can switch among these design families
- one result table for Stage A
- one result table for Stage B
- recommendation for the canonical class-I/class-II affinity mechanism
