# Shared processing and its effect on presentation

This experiment adds an optional boundary module and matched processing training
to the canonical Presto model. The architecture is implemented in clean core
`0c21bc4da1066ebfadf713dca478d655c31d4b8b`, frozen in `reproduce/runtime/`.
Predictive qualification is separate from implementation; see the experiment
README for the current run status and decision.

## Inputs and shared representation

The shared sequence encoder has two transformer layers, four attention heads and
128-dimensional residue representations. It encodes peptide, typed MHC groove
sequences and available source flanks. Class I uses the full peptide binding
representation; class II evaluates candidate nine-residue registers with the
existing `shared_core_context_v2` representation. This experiment does not change
register enumeration or the typed-MHC candidate mixture.

The inherited harmonized APC encoder maps 87 inputs to a 64-dimensional state:
cell identity, supported processing factors, cytokine and intervention annotations,
and explicit knownness. That state feeds class-scoped supply and MHC/register
selection effects. Missing conditions remain unknown. An annotation's presence
does not establish an isolated knockout or drug response; current quantitative
expression and intervention contrasts remain limited by source attribution.

| Path | Inputs | Output influence |
| --- | --- | --- |
| Intrinsic binding | Peptide/register and biological MHC | IC50 and the binding contribution to presentation |
| Generic processing | Peptide and separately encoded source flanks, knownness and termini | Additive supply contribution |
| Shared APC state | Cell/context annotations, class, peptide/source and binding representations as scoped | Supply shift and register selection |
| New boundary module | Ordered boundary residues, side, class, terminus flags, shared APC64 state | Additive correction to supply |
| MS observation | Peptide and supported acquisition covariates | Separate observation contribution |

All supported parameters train jointly. Presentation gradients may update the
shared encoder and binding route; measured-affinity replay and held-out affinity
metrics test retention. The intrinsic affinity calculation itself receives no
APC or source-flank input.

## What changed about flanks

**Flanks were already used.** The broad source adapter supplies up to 15 residues
on either side, with ambiguity masked and protein-end flags kept distinct from
missing context. Both flanks are known for 170,339 of 209,695 class-I positives
and 149,939 of 178,179 class-II positives. The original transformer encodings are
order-aware, but downstream processing uses their pooled sequence summaries.

The optional boundary module explicitly preserves cut-relative residue positions.
For the N boundary it reads the last six N-flank residues and first six peptide
residues; for the C boundary it reads the last six peptide residues and first six
C-flank residues. Residue states, presence masks, class, N/C side, true-terminus
value/knownness and APC state enter one shared MLP. Short sequences are padded at
the outer edges, keeping the cut aligned. Unknown exterior context gives exactly
zero correction for that boundary; a known empty flank at a true protein end is
a valid input.

The new MLP adds 207,360 parameters to the 646,307-parameter parent. Its scalar
readout is initialized to zero, preserving parent parameters, RNG and predictions
at transfer. It learns through both ordinary presentation and matched-pair losses.
The same weights serve both MHC classes and both boundaries, with class/side/state
inputs allowing different responses. This shared representation is not an assertion
that class-I and class-II proteolysis use identical enzymes or mechanisms.

## How processing changes the output

For each known class and MHC candidate, the staged presentation contribution is:

```text
binding + processing supply + APC register selection

processing supply = generic processing + APC supply shift
                    + N-boundary correction + C-boundary correction
```

The existing MHC-candidate aggregation then combines candidate contributions, and
the separate MS observation contribution produces the final observed-MS logit.
The inherited checkpoint has no measured stability training added in this family.
The boundary correction is shared across a peptide's registers, so it changes
supply, not the relative register preference directly.

The auxiliary loss on each matched positive/background pair is:

```text
softplus(-(supply_positive - supply_background))
```

Its weight is 0.5; every update adds four complete pairs from each class to the
unchanged 64-row ordinary replay. Supply is read from differentiable model
diagnostics before aggregation. Binding, register selection and the MS observation
offset do not enter that auxiliary readout. Existing positive observation IDs are
reused; auxiliary backgrounds are excluded from ordinary sampling and its TRAIN
diagnostic evaluation. Pair sampling and IDs are saved for exact resume.

This directly supervises a contribution that affects the deployed presentation
logit. It still estimates a conditional processing proxy: observed MS ligands
versus unobserved source backgrounds also reflect detection and other unmeasured
factors. The scalar is not a physical cleavage rate or calibrated surface yield.

## What the matched benchmark can establish

Each pair holds source protein, peptide length, typed MHC candidates and APC
condition fixed, and approximately balances every candidate's predicted BA rank.
It asks whether the model can order an observed ligand above a similarly binding
source background. Within-pair wins are the direct ordering measure; pooled AUROC
also includes comparisons across different source/condition pairs. Both are
reported for both classes, with residual BA discrimination as a sanity check.

Holding source protein and condition fixed intentionally limits what this benchmark
can say about between-protein expression or total condition-specific yield. Those
require the broader presentation cohorts and, for causal claims, attributable
intervention/control data. Hiding or shuffling APC/flank inputs is an inference
sensitivity test, not proof of a specific knockout or chemical effect.

NetMHCpan/NetMHCIIpan BA are matching tools, not the deployed affinity heads and
not sources of biological positive labels. The model predicts class I and class II
without invoking either external program at inference. The sealed TEST outcomes
are not used in this development family.
