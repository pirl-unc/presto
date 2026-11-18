# Model information flow and executed training audit

> **Historical specification.** This page documents the former default model,
> its APIs or proposed extensions. New training uses the
> [shared observation model](modular_model.md) and
> [one manifest interface](unified_training.md). Statements about a default,
> canonical path or runnable training command below apply to the historical
> implementation at commit `9dd71348124faa49a4c6a60872385c86895d50b0`, not new training.


Audited 2026-10-20 by Codex / GPT-6 against PR #129 at
`4be42102b8b6477d4abc9ecb7570a68184f468e5`. The canonical model implementation is
identical to the failed fit's v0.1.16 implementation except for a docstring.
The input/target pipeline has subsequently received corrections; this report
does not retroactively apply them to the historical fit.

**Presto is a shared sequence encoder, a graph of task-specific attention
representations, and many assay readouts. The failed fit trained only part of
that graph, and did not fit IC50 well even on its training data.** Labels
exist for several measurement families, but do not independently identify every
named biological process. The new paired-TCR wrapper is a separate, untrained
candidate-pair scorer.

Evidence: the
[registered forensic family](https://github.com/pirl-unc/presto/tree/codex/specificity-and-fit-audit-20261020/experiments/2026-10-20_1458_codex_ic50-collapse-audit)
contains full parameter, optimizer-moment, label-support and output-gradient
tables. It performs no new fitting or held-out evaluation. The earlier
[failed qualification](https://github.com/pirl-unc/presto/tree/main/experiments/2026-10-16_1030_codex_effective-target-fit)
contains the predictive metrics and original artifacts.

## 1. What enters the model

The canonical sequence inputs are variable-length integer tensors, one row
per example:

| Input | Shape | Meaning |
| --- | --- | --- |
| `pep_tok` | B × peptide length | Supplied peptide amino acids |
| `mhc_a_tok` | B × groove-1 length | First groove segment |
| `mhc_b_tok` | B × groove-2 length | Second groove segment |
| `flank_n_tok` | B × N-flank length | Optional adjacent source-protein residues |
| `flank_c_tok` | B × C-flank length | Optional adjacent source-protein residues |

MHC names resolve upstream to sequence. The two MHC tensor names are misleading
if read as two whole biological chains:

| MHC class | Groove 1 | Groove 2 |
| --- | --- | --- |
| I | Alpha1 domain of the heavy chain | Alpha2 domain of the same heavy chain |
| II | Alpha1 domain of the alpha chain | Beta1 domain of the beta chain |

Beta2-microglobulin is not a class-I token input. Allele-specific groove residues
are represented; a categorical allele ID is not embedded. Alleles that produce
identical encoded groove sequences are indistinguishable through these inputs.
Class-II pairing must be preserved during resolution. Missing partners are
explicit missing segments in the canonical model, not invented sequences.

Additional downstream inputs include explicit MHC class, optional molecular
species overrides, processing species, coarse cellular provenance, machinery,
and protein-boundary flags. These do not enter residue embeddings. Without a
class override, class I/II probabilities are inferred from the two MHC segment
type predictions. This class prediction describes the supplied MHC molecule;
it is not an inference of a TCR's restriction.

Observed labels, source IDs and assay descriptors are outside the predictive
input path. Assay descriptors select which fixed output track receives a loss.
The normal row trainer does not pass the antigen-species target back into
`species_of_origin` as an input override.

The historical run supplied peptide and resolved MHC grooves, disabled source
flanks, used the older species/context consumer, and contained no TCR inputs.
Changes in missing-flank parameters therefore do not demonstrate learning from
observed source-protein context.

Source: [input assembly](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/data/collate.py#L493),
[groove preparation](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/data/groove.py#L180),
[forward inputs/class routing](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/models/presto.py#L2998).

## 2. How sequence information flows

All five segments use the same amino-acid embedding and Transformer weights,
with segment identity and positions. The base attention mask prevents residues
in one segment from attending to another segment. Peptide positions encode
distance from both ends and fractional position; groove positions use separate
sequential embeddings. Missingness and padding are explicit.

The early layer is pooled for MHC identity auxiliaries. Later task queries
cross-attend to allowed residues and upstream representations:

| Representation | Direct sequence access | Other inputs |
| --- | --- | --- |
| Processing I/II | Peptide and both source flanks | Terminal/length features, coarse processing context, molecular context |
| Binding affinity/stability | Candidate peptide core and both grooves | Groove summary, molecular context; peptide-flanking-region summaries |
| Presentation I/II | None | Corresponding processing representation and both binding query representations |
| MS detectability | Peptide | No separately qualified acquisition/abundance model |
| Species of origin | Peptide | Optional explicit API override, unused in normal row training |
| Foreignness | None | Projection of species-of-origin representation |
| Recognition CD8/CD4 | Peptide | Foreignness representation |
| Immunogenicity CD8/CD4 | None | Binding affinity/stability representations and corresponding recognition representation |

The misleadingly named `apc_cell_type_context_vec` is constructed from MHC
class, molecular-species probabilities and inferred chain compatibility. It is
not a cell-type annotation. Actual coarse cellular context uses separate
processing embeddings.

Binding enumerates candidate peptide windows, runs eight learned attention
queries with two cross-attention/MLP layers, scores windows, and averages
candidate representations by a learned softmax posterior. The shipped window
set is `[9]` for both classes; shorter peptides use a full-length fallback.
Longer peptides contribute flanking-region summaries. Class II adds a
flanking-region score adjustment. There were no core-position labels in the
fit, so core localization was learned indirectly, if at all.

The optional direct peptide–MHC residue interaction block is **off** in this
fit. Peptide/MHC integration occurs through the binding queries. The eight
query vectors are projected and flattened, followed by window fusion and
normalization; a further MLP compresses them for the assay heads.

| Dimension | Constructor defaults | Historical fit |
| --- | ---: | ---: |
| Residue width | 256 | 32 |
| Base Transformer layers | 4 | 1 |
| Base attention heads | 8 | 4 |
| Queries per binding representation | 8 | 8 |
| Binding query layers | 2 | 2 |
| Flattened pMHC representation | 512 | 256 |
| Affinity/stability assay vector | 256 | 32 |
| Total parameters in historical fit | — | 409,850 |

Thus “one layer” describes the base encoder, not the entire network. The
historical 256-dimensional pMHC vector is compressed to 32 features before
the affinity readout. No direct segment residual bypass was enabled.

Presentation consumes binding representations, not the reported scalar KD.
Similarly, presentation consumes processing representations, not the scalar
processing prediction. A supervised downstream representation can coexist with
an untouched upstream scalar head.

**Known semantic gap:** canonical recognition is still peptide-plus-foreignness,
with MHC class used for the final CD4/CD8 mixture. The individual recognition
branches do not see the presenting groove. The desired
[I/O contract](model_io_contract.md) calls for presented-pMHC and repertoire
context; that part is not implemented. There is no selection-system MHC set.

Source: [latent access/dependency tables](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/models/presto.py#L240),
[segment encoder](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/models/presto.py#L1446),
[binding windows](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/models/presto.py#L2102).

## 3. What comes out

The API computes many outputs on every forward pass, even when no labels
support them:

- Quantitative peptide–MHC assay predictions: KD, IC50, EC50, kon, koff,
  dissociation half-life and melting temperature. KD proxy outputs can alias KD.
  Concentrations are log10(nM) internally; rates are log10(rate), half-life is
  log10(minutes), and Tm is normalized Celsius.
- Processing, excision, class-specific binding/presentation, elution/MS,
  repertoire recognition, immunogenicity and T-cell response logits/scores.
- Fixed binding and T-cell assay panels. They sweep one descriptor axis at a
  time, rather than every joint protocol configuration.
- pMHC-only receptor-evidence outputs, which predict evidence that some
  receptor was reported. They do not consume a receptor sequence.
- MHC class/type/species auxiliaries, source/foreignness outputs, core-position
  diagnostics and intermediate representations.

A returned sigmoid is a numerical score, not evidence of calibration.
Aliases and repeated proxy targets are not independent measurements.

The affinity path itself is unusually elaborate for a small IC50 fit:

```text
32-feature affinity vector
  ├─ kinetic heads: koff, kon_intrinsic, kon_chaperone
  │    └─ derived KD + learned KD bias → core KD
  ├─ independent affinity probe → probe KD
  └─ IC50 / EC50 residual MLPs

mixed KD = (1 - sigmoid(g)) × core KD + sigmoid(g) × probe KD
IC50 = bounded(bounded(mixed KD) + IC50 residual)
```

Several stages apply smooth range bounds. The kinetic names do not establish
identifiability: there were no kon labels, and no measurements separately
identifying the two association-rate components. The construction permits
compensation among kinetic rates, KD bias, probe mixture and assay residuals.
Half-life also receives a learned stability residual in addition to its koff
relationship. A useful final concentration prediction would not by itself
validate those internal kinetic interpretations.

Source: [affinity forward](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/models/presto_modules.py#L323),
[kinetic module](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/models/pmhc.py#L391),
[assay derivation](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/models/heads.py#L464).

## 4. Which assays actually trained the failed fit

The frozen cohort had 9,053 observations; **7,241 were training observations**.
The following counts partition those training observations exactly:

| Source measurement | Training observations | Effective supervision |
| --- | ---: | --- |
| Direct IC50 | 2,074: 1,599 exact, 475 censored | IC50, generic KD, affinity probe, binding panel |
| Direct KD | 375: 144 exact, 231 censored | KD-specific and generic KD, probe, panel |
| Direct EC50 | 89: 74 exact, 15 censored | EC50, generic KD, probe, panel |
| KD reported via IC50 proxy | 935 | Generic KD, probe, panel under merged grouping |
| KD reported via EC50 proxy | 116 | Generic KD, probe, panel under merged grouping |
| koff | 19 | Log-rate regression and shared binding path |
| Dissociation half-life | 434: 412 exact, 22 censored | Censor-aware half-life regression |
| Tm | 111 | Normalized-temperature regression |
| Elution/MS | 3,088, all positive | Bag elution, presentation, duplicate MS alias, excision-condition proxies |
| kon, measured processing, T-cell, TCR response/evidence, gold core positions | 0 | No corresponding direct labels |

Elution's 3,088 bags contain 10,039 candidate molecule instances. Noisy-OR
combines instance probabilities into bag probabilities. Candidate instances
do not each inherit positive elution labels. This fit had no negative elution
bags, no generated negatives and no bulk-MS supervision.

MHC class/type/species auxiliaries and source-origin/foreignness targets were
also active on all 7,241 rows. Every origin label was human, and every foreignness
target was zero. These are constant-target auxiliary problems in this cohort.
The historical molecular-species labels included seven murine assignments from
host context; subsequent role corrections are documented separately.

The source loaders support more families than this Hitlist cohort supplied.
Even the broader merged path does not use every source assay: qualitative
binding, structures, equilibrium association constants and unknown endpoints
are reported as unsupported. Missing labels are masked rather than made zero.

Source:
[row losses](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/training/supervision.py#L59),
[MIL losses](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/training/mil.py#L163),
[assay-source policy](assay_learning_scheme.md).

## 5. Which parameters were actually updated

Every canonical parameter was admitted to AdamW; this fit did not use curriculum
freezing. Comparing initialization with the selected epoch-1 checkpoint:

- 476 parameter tensors, containing 409,850 scalar parameters.
- 108 tensors, containing **94,578 parameters (23.1%)**, have no optimizer state.
- 116 tensors, containing 95,453 parameters, are entirely unchanged.
- 359 tensors have nonzero Adam second moments, evidence of past nonzero
  gradients. This does not establish useful learning.

The checkpoint is epoch 1 (100 steps). The output-gradient ledger spans both
epochs (200 steps); these are deliberately separate evidence windows.

| Component | Observed status | Interpretation |
| --- | --- | --- |
| Shared residue encoder, binding queries/core windows, affinity and stability readouts | Changed; nonzero optimizer moments | Optimized, but IC50 remained underfit |
| Kinetic heads, probe, KD bias, IC50/EC50 residuals | Changed | Receives concentration/kinetics losses; internal decomposition not identified |
| Processing I/II query representations | Changed | Indirectly trained through presentation and excision proxies |
| Scalar processing I/II heads | Exactly unchanged; no optimizer state | Returned processing scores are untrained readouts |
| Presentation and elution | Changed | Positive-only/proxy supervision; no demonstrated discrimination |
| MS detectability | Changed through elution | Zero direct detectability labels; not independently identified |
| Recognition CD4/CD8 queries and heads | Exactly unchanged; no optimizer state | Untrained |
| Immunogenicity MLPs and heads | Exactly unchanged; no optimizer state | Untrained |
| T-cell assay head/panel | Exactly unchanged; no optimizer state | Untrained |
| pMHC-only receptor-evidence heads | Exactly unchanged; no optimizer state | Untrained |
| Class-specific binding calibration scalars | Exactly unchanged; no optimizer state | No active downstream scalar-output supervision reached them |
| Excision parameters | Mixed | Some indirect updates, some untouched/fixed-rule rows; no direct measured excision labels |
| MHC identity and source-origin/foreignness heads | Changed | Auxiliary supervision, with the support limitations above |
| New paired-TCR wrapper | Not in historical checkpoint | No receptor training has occurred |

An unchanged head can still emit different numbers after its shared inputs
change. Conversely, a tensor can change through AdamW weight decay without a
useful task gradient. The full tables retain element-level moment/change counts
and distinguish these cases; a changed embedding table does not imply every
category row was supervised.

## 6. What this explains about the underfit

The poor training fit is established: exact-IC50 training RMSE was **1.338**
log10(nM), versus **1.216** for the training mean and **1.089** for ridge.
It is not merely a held-out-generalization problem.
The baselines fit exact IC50 values only, while Presto also optimizes censored
and other task losses. This objective mismatch can affect the preferred mean
prediction; the comparison does not isolate architecture or optimizer failure.

Verified implementation/optimization concerns:

1. **Concentration objectives overlap.** An IC50 row trains its family-specific
   head, generic KD, the probe and the descriptor-panel objective. Direct KD
   rows train the same KD output through both generic and specific losses.
   The source endpoint remains preserved, but effective targets are still
   shared across measurement families. Describing this as complete endpoint
   separation would be incorrect.
2. **Elution is counted twice as an identical output.** `ms_logit` is literally
   `elution_logit`, and both appear in the MIL objective with base weight one.
   Presentation and the excision-condition panel add further losses from the
   same positive-only observations. These are four task slots, not four
   independent measurements.
3. **Task means, not observation counts, determine weighting.** Active losses
   are averaged with base weights and learned uncertainty factors. In epoch 1,
   an average batch has 18.31 IC50 labels, while a batch containing koff has
   about 1.14 koff labels. Both tasks receive roughly .07 normalized weight
   when present, despite only 19 unique koff training rows. Mean koff squared
   error is 29.61, versus IC50 2.19. Loss magnitudes do not prove gradient
   dominance, but justify measuring it.
4. **Constant-target auxiliary tasks reward shared offsets.** Source origin,
   foreignness and positive-only presentation tasks can improve without
   learning fine sequence discrimination. They share the base encoder.
5. **The affinity path is already narrow at initialization.** In the class-I
   replay, prediction SD starts at .00281 and ends at .00799 log10(nM), while
   target/bound SD is 1.52658. Thus output variation increases slightly even
   while intermediate pMHC variation contracts from .17315 to .05270.
   The model did not begin with a useful varied predictor that was simply
   destroyed.
6. **Internal components compensate.** Core/probe correlation is -.999313
   on the selected class-I training replay. Their weighted SDs are .01107 and
   .02156; the mixture SD is .01051. Cancellation is real, but both components
   are already too narrow. Removing cancellation alone is not a demonstrated cure.
7. **The optimization budget was short and selection was broad.** Fresh random
   initialization, width 32, 200 total steps, no pretraining; complete aggregate
   validation selected epoch 1. The selected checkpoint therefore represents
   100 steps, and selection was not specifically for IC50 RMSE.

The direct IC50 output received finite nonzero gradients in all 200 batches.
Its learned uncertainty multiplier stayed around .95/.94; it was not disabled.
None of the above observations alone establishes the cause of contraction.

The next causal tests should be registered training-only comparisons:

1. Fit a fixed small exact-IC50 training subset to test memorization, with a
   simple direct readout and the existing affinity readout under matched inputs.
2. Isolate loss routing and head complexity as separate factors: IC50-only
   versus current multitask loss; direct versus compound affinity readout.
3. Once fitting works, add independent measurement families one at a time.
   Separately remove alias duplication and generic endpoint reuse.
4. Track per-task gradients on the shared encoder, query diversity, core-window
   entropy, representation variation, local head sensitivity and train error.
   Then test training duration/width if the smaller controls justify it.

No new fitting was launched here. This report narrows the experiment design;
it does not silently change a historical recipe or claim a causal fix.

## 7. How the paired-TCR extension fits

Conceptually it scores five supplied sequence components:
`TCR alpha, TCR beta, peptide, MHC groove 1, MHC groove 2`, with class/masks and
optional backbone context. pMHC is an input candidate, not a target generated
by the network.

Each full supplied TCR chain has a separate Transformer encoder with amino-acid
embeddings and sinusoidal positions. Defaults are two layers, four heads and
width 128. Masked mean pooling produces one vector per chain; concatenation and
an MLP produce a 128-feature receptor vector. A projection maps Presto's pMHC
interaction representation to 128 features. A final MLP consumes receptor,
pMHC, their elementwise product and absolute difference, and returns one
functional-response logit per pair.

There is currently no CDR annotation, V/J feature channel, structural contact
model or TCR-to-pMHC residue cross-attention. Full-chain mean pooling can dilute
local specificity information; a binding-oriented pMHC bottleneck may discard
features needed for receptor recognition. These are modeling hypotheses, not
measured failures of an already-trained TCR model.

The wrapper accepts supplied sequences up to 512 residues without truncation.
Input length does not prove full-length identity. Observed/reconstructed
provenance must be explicit. The normalized CSV's `mhc_a`/`mhc_b` describe source
restriction/chain identity, whereas tensor names describe groove halves; this
boundary should be made explicit in any source adapter.

Training data must represent a measured receptor–candidate-pMHC pair, with
paired chains, peptide, class-correct alleles, functional outcome, QC and
assay/study provenance. A tested non-validating pair is an assay-scoped negative;
an untested pair or failed-QC result is not. No one-of-N epitope assumption is
required, so cross-reactivity is expressible as several positive pair rows.

Class/allele/peptide restriction can be investigated by scoring appropriate
candidate pairs, provided training and validation actually span those
contrasts. A fixed-allele or class-I-only dataset cannot establish class-II or
cross-allele generalization. There is no TCR-only restriction classifier,
peptide decoder or trained ligand-discovery system. The actual TCRvdb release
is still unavailable for schema/coverage inspection; the normalized loader is
not its source adapter.

The backbone is frozen by default during wrapper training, with explicit joint
fine-tuning available. Freezing an underfit pMHC model does not make its
representation qualified for receptor matching. No wrapper weights have been
trained or validated.

Source:
[wrapper](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/models/tcr_specificity.py),
[normalized evidence contract](https://github.com/pirl-unc/presto/blob/4be42102b8b6477d4abc9ecb7570a68184f468e5/data/tcr_specificity.py).
