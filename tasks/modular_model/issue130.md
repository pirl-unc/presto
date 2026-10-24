## Goal

Build a small, understandable Presto model that learns from the measurements we
actually have. Start with a demonstrably fitted peptide–MHC IC50 predictor, then
add independently supervised assay outputs without breaking it. Every allocated
trainable module must have a justified loss path and measured training coverage.
Every public output must have an explicit data and validation contract.

The implementation plan is detailed; the **model spec is deliberately short**:

> Sequence inputs → shared encoder → supervised class/species predictions
> → explicit defaults/overrides → class-aware peptide registers
> → shared fusion and register pooling → directly supervised assay heads.
> Cellular heads also see APC state; paired-TCR heads also see the receptor.

**Revision, October 20:** preserve explicit MHC-II register handling; infer
class/species before assay prediction; expose role-specific defaults and
overrides; prioritize cell lines with known processing defects; and specify how
paired TCRs enter the same model. This supersedes the original known-class-only,
whole-peptide-pooling proposal. These are design changes, not new fit results.

**Implementation authorization:** the user requested a holistic specification
covering all available data and subsets, then implementation from the frozen
issue. The source/subset and MS-observation contracts below are part of that
specification, not optional replacements for them.

This is a proposed implementation/qualification plan, not evidence that the
redesign already works. Do not launch a new fit or reinterpret an old checkpoint
merely by filing this issue.

Related work: #117 (underfit), #46 (I/O contract), #48 (coverage), #53 (real-data
qualification), #115/#123 (species roles), #127 (donor-linked Atlas), and #129
(TCR wrapper and forensic audit). These issues remain open on their own merits.
This issue coordinates a simpler implementation; it does not silently satisfy
the broader biological claims in #46.

## 1. Current state: what is implemented, what was trained, what failed

Evidence pinned to the October 20 audit:

- [Full architecture and training-path report](https://github.com/pirl-unc/presto/blob/6196143048fa9eaf6257b7826f1adb5a14f9c0a9/docs/model_training_audit.md)
- [Reproducible checkpoint/representation audit](https://github.com/pirl-unc/presto/tree/6196143048fa9eaf6257b7826f1adb5a14f9c0a9/experiments/2026-10-20_1458_codex_ic50-collapse-audit)
- [Original failed fit and held-out metrics](https://github.com/pirl-unc/presto/blob/6196143048fa9eaf6257b7826f1adb5a14f9c0a9/experiments/2026-10-16_1030_codex_effective-target-fit/README.md)
- [Public frozen fit artifacts](https://github.com/pirl-unc/presto/releases/tag/effective-target-fit-qualification-v1)

The canonical model has a shared segment encoder, a graph of processing,
binding, presentation, origin/foreignness, detectability, recognition and
immunogenicity representations, and many assay/panel heads. The failed bounded
run used a fresh width-32, one-base-layer model with 409,850 parameters, 200
optimizer steps and no pretraining. Aggregate validation selected epoch 1:
the selected checkpoint represents **100 steps**.

The frozen cohort contains 9,053 observations; training contains 7,241:

| Training measurement | Count |
| --- | ---: |
| Direct IC50 | 2,074: 1,599 exact and 475 censored |
| Direct KD | 375 |
| Direct EC50 | 89 |
| KD via IC50 / EC50 proxy | 935 / 116 |
| koff | 19 |
| Dissociation half-life | 434 |
| Tm | 111 |
| Elution/MS | 3,088, all positive |
| kon, measured processing, T-cell, TCR response/evidence, gold core positions | 0 |

These are historical selected-training counts, not a census of every upstream
record available today. Elution has 10,039 candidate MHC instances across its
3,088 bags; those instances are not 10,039 independent positive measurements.

Observed failures and limitations:

1. Exact-IC50 training RMSE is **1.338 log10(nM)**, worse than the train mean
   (**1.216**) and ridge (**1.089**). Test RMSE is **1.310**, versus ridge
   **1.138**, despite Spearman **0.337**. Held-out predictions identify no
   binders at 500 nM. The baselines optimize exact IC50 only, whereas Presto
   also optimizes censored and other objectives; that mismatch is part of the
   investigation.
2. On 877 class-I training rows, prediction SD is **0.0080 log10(nM)** versus
   target/bound SD **1.5266**. Predictions were already narrow at initialization.
   pMHC representation variation contracts about threefold. Core/probe terms
   partly cancel, with correlation **−0.9993**. These observations locate
   problems but do not establish a single cause.
3. IC50 gradients are finite and nonzero in all 200 optimizer batches. The
   learned uncertainty multiplier remains approximately .95/.94, not zero.
4. At the selected checkpoint, **94,578 parameters (23.1%)** lie in tensors
   without optimizer state. Recognition, immunogenicity, T-cell and
   receptor-evidence branches remain at initialization.
5. Processing representations receive downstream gradients, but the scalar
   processing prediction heads remain untrained. Detectability receives only
   indirect elution supervision. A named representation is not an independently
   identified biological quantity.
6. An IC50 row also trains generic KD, an affinity probe and a binding panel.
   Direct KD has overlapping generic/specific losses. `ms_logit` and
   `elution_logit` are the same tensor but separate MIL objectives.
7. Task-mean weighting gives a sparse koff task substantial weight when present.
   Positive-only elution and constant human-origin/nonforeign auxiliaries can
   reduce loss through common offsets rather than sequence discrimination.
8. Canonical recognition still lacks presenting-MHC sequence input. The separate
   paired-TCR wrapper is untrained, and the authorized TCRvdb release is not yet
   available for schema/coverage inspection.

Source freshness is a separate question. The old fit used the pinned corrected
human Hitlist 1.78.0 export. A newer installed/package version or a different
Atlas/bulk resource is not automatically a replacement training cohort. Compare
artifact manifests and semantic membership before changing data.

## 2. Minimal model specification

### Inputs

Required sequence inputs:

1. `peptide`: full supplied peptide amino-acid sequence.
2. `mhc_groove1`: first resolved groove segment.
3. `mhc_groove2`: second resolved groove segment.

Class and molecular species are inferred first, with optional supplied
annotations and overrides. Residue encoding itself remains sequence-only.

| Class | Groove 1 | Groove 2 |
| --- | --- | --- |
| I | Heavy-chain alpha1 | Same heavy-chain alpha2 |
| II | Alpha-chain alpha1 | Beta-chain beta1 |

Use `mhcseqs` as the sequence-resolution/extraction authority. Preserve original
and resolved allele identities and class-II pairing in provenance. There is no
allele-ID embedding, class-I beta2m input, invented partner, or silent sequence
truncation. Initial training requires both resolved grooves; incomplete pairs
remain in the exclusion report rather than receiving guessed sequences.
Use the 20 canonical amino acids plus padding in this first model. Quarantine
ambiguous/invalid residue rows with explicit counts rather than silently
changing their sequences or tokenizing an allele name as amino acids.

Additional biological inputs are scoped by endpoint: APC/immune/repertoire
species, APC machinery states and, where relevant, source-protein context or
paired TCR sequences. Source/dataset/study/donor/cell-line IDs remain provenance
and grouping keys, not predictive embeddings. Assay descriptors route measured
labels to outputs; the model does not classify the assay method.

### Stage 1: identity predictions, followed by context resolution

Use one shared amino-acid embedding and Transformer: width 128, two layers,
four attention heads, feed-forward width 512, fixed sinusoidal positions,
masked mean pooling. Apply it independently to each sequence.

- Predict MHC class with `Linear(concat(groove1_vec, groove2_vec), 256 -> 2)`.
- Predict each MHC segment's species with one shared `Linear(128 -> S)`,
  where the supported species vocabulary `S` is frozen in the checkpoint.
- These are directly supervised identity outputs, with curated molecular
  identity labels. A host/APC species is not an MHC species label.
- A peptide-origin predictor is optional only after qualified, varied source
  labels exist. A short peptide can have several valid species matches;
  all-human origin labels cannot qualify a general species predictor.
- Add TCR-chain species prediction when the receptor module and reliable
  receptor-origin labels are admitted. Do not copy an APC label onto a receptor.

Keep `predicted`, `supplied`, `resolved`, confidence/support, and resolution
source separate. The resolver is deterministic and has no trainable parameters:

1. Explicit per-role override wins over supplied source metadata.
2. Supplied source metadata wins over that role's supported identity prediction.
3. An explicit, versioned default policy may fill unresolved contextual roles.
   The standard convenience policy defaults `immune_species` from a consistent
   MHC molecular-species prediction, then defaults APC and repertoire species
   from `immune_species`. Each role can be overridden independently.
4. A conflicting/mixed MHC-species pair does not yield a confident common
   species. Unsupported predictions/defaults remain unknown.

Defaults in step 3 are **user-requested assumptions**, not recovered biological
facts. Preserve the original missing fields and flag each assumed value; never
use a cascade-generated value as an auxiliary training label. A strict policy
disables these convenience defaults. Antigen origin and the repertoire's
selection MHC are not copied from the presenting molecule. A generic in-vivo
host field does not establish APC identity. Human MHC in a nonhuman APC remains
expressible without changing the molecular labels.

This explicitly updates #46's earlier prohibition on implicit cross-role
defaults: the new defaults are visible, overridable and provenance-preserving.
It does not relax the prohibition on inventing source annotations.

Retain probabilities when identity is inferred. A categorical class override
sets the effective class distribution to one-hot, while preserving the raw
classifier prediction for auditing. Stop assay gradients at the resolved
identity probabilities: identity heads learn from identity labels, rather than
being repurposed as arbitrary hidden assay features. The shared encoder still
receives both identity and assay gradients through their own paths.

The first stage supplies **both** interpreted predictions and richer encoder
representations. Downstream heads consume the representations directly; the
class/species probabilities are not an information bottleneck. Reuse encodings
within one forward pass and fine-tune the encoder with admitted assay losses.
Good identity accuracy alone does not prove useful specificity features.

For paired TCRs, an optional sequence-only auxiliary predicts observed
CD4/CD8 association from phenotyped-cell labels, separately from receptor
species. It must not infer its target from the candidate MHC class. Preserve
actual assay-cell coreceptor state (CD4, CD8, both, neither or unknown) as a
separate overridable context field. A sequence-derived lineage prior is not
proof of the coreceptors expressed by a transduced or otherwise altered cell.
Qualify the auxiliary across held-out donors and receptor clusters before
using it to fill missing context. See the
[paired-chain lineage study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6684766/).

### Stage 2: explicit registers and a small shared assay model

**Class I:** one candidate containing the full measured peptide, with its
termini and length preserved. Do not replace a longer class-I peptide with a
free choice of unrelated nine-residue substrings.

**Class II:** enumerate every contiguous nine-residue core, starting at
`0 .. L-9`; a 15-mer has seven candidates. Encode each core with positions
restarting at its first residue, so P1–P9 are relative to the proposed register.
This first version covers canonical nine-residue cores; report shorter class-II
peptides and noncanonical/gapped-core support as unsupported, not silently
padded or truncated. An uncertain class on a peptide lacking a valid class-II
candidate requires explicit resolution or an unsupported result.

For each candidate retain peptide-flanking-region (PFR) context:

- Mean 20-amino-acid composition over up to the three nearest residues on
  each side of the core; empty regions have zero vectors.
- `log1p` of the full N-PFR length, C-PFR length and measured peptide length.
- Class-I PFR vectors/lengths are zero because its candidate is the full peptide.

These are fixed features derived from the measured peptide, **not**
source-protein flanks outside that peptide. Position-aware core encoding and
within-peptide flank features retain the register problem while keeping the
network small. This design choice is informed by
[NN-align](https://pmc.ncbi.nlm.nih.gov/articles/PMC2753847/); it is not a claim
that a nine-residue-only model covers every observed binding mode
([noncanonical-core study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5588850/)).

One concrete network, with direct heads and one register-pooling operation:

```text
E(sequence) = shared sequence-only encoder and masked pooling -> 128

For each permitted class c and candidate register r:
    z[c,r] = GELU(Linear(concat(
        E(candidate_peptide), E(groove1), E(groove2),
        PFR_features[43], branch_class_onehot[2],
        resolved_molecular_species[2*S]), (429 + 2*S) -> 128))

    v[c,r,endpoint] = z[c,r]                         # purified binding
        or shared_context_fusion(z[c,r], allowed_context)  # cellular/TCR
    register_logit[c,r,endpoint] = shared_Linear(v[c,r,endpoint], 128 -> 1)

    w[c,:,endpoint] = masked_softmax(register_logit[c,:,endpoint])
    pooled[c,endpoint] = sum_r w[c,r,endpoint] * v[c,r,endpoint]

v[endpoint] = sum_c resolved_class_probability[c] * pooled[c,endpoint]
prediction[endpoint] = endpoint_Linear(v[endpoint], 128 -> 1)
```

Cache groove encodings per example; do not re-encode them per register.
Padding and candidate masks exclude nonexistent residues/registers. Normalize
register weights within each class before class mixing, so seven class-II
windows do not create seven times the class prior. Known class uses only its
branch; class I has one candidate with weight one. Shared register pooling
learns from class-II peptide losses, without requiring core-position labels.
Its weights are learned attention, not calibrated physical occupancies or
validated structural register predictions. Report weight entropy and gradients
to detect collapse or uniform averaging.

`shared_context_fusion` is one optional `Linear -> GELU` to width 128, with
checkpoint-declared feature slots and endpoint masks. Construct it only when
supervised cellular or receptor data are admitted. TCR slots and machinery
features without training support are not allocated as dormant parameters.
This is not a graph of separate processing/recognition/foreignness scalar heads.

All residue encodings remain independent of other sequences and metadata.
Fusion combines them downstream. Dropout is 0.1 for the regularized full fit
and zero for the memorization diagnostic.

**Initial assay head set: `{IC50}`**, alongside the supervised class/species
identity heads. Later measured endpoint heads are allocated only with admitted
data. One whole-peptide measurement produces one loss **after** register
aggregation; never label every candidate core as an independently measured
binder. The encoder, fusion, pooling and active heads must all show appropriate
training gradients.

Retain explicit registers and small supervised identity heads; remove kinetic
factorization, KD/probe mixtures, residual stacks, learned task uncertainty,
descriptor panels, duplicated outcome aliases and unused biological branches.
Thresholding IC50 is a decision rule, not a new calibrated probability head.

### APC machinery defaults and scope

Resolve a versioned reference machinery profile from APC species, with cell
type/state refinement when known. Species alone is a prior about a reference
system, not evidence that every component is expressed or functional in a
particular cell line.

Represent a set of independent component records, each with:

```text
component / ortholog_group
state: reference | intact | reduced | lost | unknown
optional measured activity_or_abundance and its scale
evidence/provenance, confidence, source sample/condition
```

Explicit per-component overrides and sample-specific measured loss beat the
reference profile. Keep multiple concurrent defects; do not collapse all genes
to the single "most severe" mechanism group. Distinguish knockout, knockdown,
inhibition, rescued expression and unknown functional validation in provenance;
do not invent a numeric residual activity for a categorical perturbation.
An empty perturbation list is not proof of an intact control.

Support ontology-aligned components such as peptide supply, trimming, loading
and class-II editing components only where data support them. Keep distinct
component identities even if a first supported predictor must pool sparse
features by mechanism. Untrained effects receive an unsupported status rather
than a fabricated condition-dependent response.

| Output family | Allowed downstream biological context |
| --- | --- |
| Purified peptide–MHC affinity/stability | pMHC sequences, class, molecular species; relevant measured reaction context if admitted |
| Cellular peptide presentation / observed elution | Above plus APC species, machinery, stimulus, source sequence/flanks, protein-boundary and supported abundance context |
| Receptor–supplied-pMHC compatibility | Paired TCR sequences/species and register-specific pMHC; no invented APC processing dependence |
| Cellular TCR functional response | Above plus supplied APC/immune context and antigen exposure/delivery state |

Intrinsic binding must be invariant to APC machinery changes. In particular,
a cellular component loss must not blank a groove sequence or set the molecular
affinity to zero. Cellular presentation may change while binding of the supplied
peptide to the supplied purified molecule is unchanged. Preserve whether antigen
was endogenous, protein supplied, peptide pulsed or a preformed pMHC: these
routes expose different dependencies and cannot share an unconditional
"processing defect = negative" rule.

### Separate MS observation from peptide production and presentation

Non-MHC proteomics must train a useful reusable component. Distinguish three
questions:

1. **Digest product/availability:** which peptide is produced from a parent
   protein under the specified digestion? Inputs include peptide and mapped
   source flanks/boundaries, protease, missed-cleavage rules and supported
   preparation conditions.
2. **Cellular pMHC availability:** which peptide is presented under the APC
   state? This uses pMHC register representations trained by affinity/stability
   outcomes, plus source context and component-specific processing/loading
   state. An IC50 scalar is not a presentation probability.
3. **MS observation response:** how likely is an available peptide/peptidoform
   to be identified under the acquisition/search conditions? Inputs include
   peptide sequence, supported chemical modifications and prospective
   acquisition features. Amount/abundance and recovery belong in the observation
   context, with missingness and measurement scale recorded.

The small network therefore has two observation routes:

```text
Non-MHC MS:
  peptide + source context + digest -> digest/availability features --+
  peptide + chemical/acquisition context -> shared MS features -------+-> bulk observation

MHC elution MS:
  pMHC + source context + APC state -> cellular availability features -+
  peptide + chemical/acquisition context -> SAME MS features ----------+-> elution observation
```

Implement each new feature block as one `Linear -> GELU` to width 128 over
shared sequence vectors and declared context slots. Retain the existing small
cellular fusion as the biological availability block. Digest and MS blocks are
separate: the MS block must not learn a direct enzyme, allele, APC-knockout,
species, TCR or dataset-ID shortcut. Digest type belongs to the digest-product
block; actual acquisition conditions may affect detection. Known MS sample
preparation is distinct from in-vivo APC processing.

For the initial binary observation endpoints use a small additive combiner:

```text
bulk_observation_logit   = digest_score + ms_response_score + bulk_offset
elution_observation_logit = cellular_score + ms_response_score + elution_offset
```

Each score is a linear readout of its corresponding feature block; the two
MS routes share the same `ms_response_score` readout. Offsets include one
endpoint intercept and only admitted amount/recovery/acquisition covariates
with explicit missing masks. No arbitrary dataset-ID intercepts. One observed
outcome gets one BCE loss at the final logit. These are statistical observation
scores, not a multiplication of calibrated processing/binding/detection
probabilities. Quantitative intensities and detection-limit censored outcomes
need their own explicitly scaled measured-endpoint heads; never relabel them
as affinity or silently binarize them.
Use bias-free component score readouts and one intercept per observation
endpoint, avoiding redundant component/intercept offsets. Do not allocate an
additional free coefficient for every component in the initial combiner.

Affinity and stability measurements supervise their own heads and the pMHC
representation consumed by cellular prediction. They are not multiplied
together as independent probabilities. Direct processing/digest-yield labels,
if actually available, can additionally supervise the relevant component;
ordinary peptide identifications are not isolated cleavage measurements.

**Identifiability is a data requirement.** A bulk detection depends jointly on
digestion, amount and MS response. Arbitrary latent decompositions can fit the
same outcome. Use independent available-peptide/standard measurements,
quantitative abundance, controlled digest comparisons and acquisition
replicates where available to anchor the MS component. Without those anchors,
return a qualified *within-search observability* output and mark separate
digest/MS scores as indirectly trained diagnostics, not calibrated biological
probabilities. The model may learn useful shared features without identifying
each process separately. This limitation is established in
[peptide-detectability work](https://www.ccs.neu.edu/home/radivojac/papers/li_jproteomeres_2010.pdf).

The frozen scoped non-MHC adapter already supplies one observed/not-observed
vote per verified search scope, distinguishing MS/MS identification, matching
and eligible unobserved candidates. Its endpoint is
`observed_in_selected_search_scope`, not intrinsic detectability
([adapter](https://github.com/pirl-unc/presto/blob/6196143048fa9eaf6257b7826f1adb5a14f9c0a9/data/scoped_bulk.py#L1)).
The raw bulk table lacks this denominator and emits no detectability labels.
Do not fabricate negatives from an arbitrary proteome, count fractions as
replicates, or treat wrong-enzyme generated controls as measured detector
failures. Search-scoped zero means "not identified in this search," not
"cannot be produced/detected."

Preserve parent protein/occurrence ambiguity, modification sites, precursor
charge, enzyme specificity, eligible length/mass ranges, searched FASTA,
acquisition/search settings, identification evidence/FDR, abundance, run and
replicate identities. Predict at a declared peptide/peptidoform/charge scope;
do not use an observed post-identification charge/intensity as an input that
would be unavailable for an unobserved candidate. Unsupported modifications
remain explicit exclusions, never silently stripped from their labels.

Before transferring the MS component into elution, evaluate by held-out parent
protein/study/acquisition and by enzyme/length/chemical domain. Tryptic digestion
is not the generation process of native MHC ligands. Retain MS anchor replay
and monitor its quality during joint fitting; freezing only a detector head
does not freeze its function when the shared encoder continues changing.
Loss of an anchor task's quality blocks promotion. A single shared MS block
with documented domain support is preferable to an unbounded set of
dataset-specific correction heads.

### Outputs and target space

- Regression endpoints return their declared continuous value; binary endpoints
  return one logit and a sigmoid score; identity endpoints return categorical
  logits/probabilities. Only qualified outputs may claim calibrated probabilities.
- Use explicit internal names such as `ic50_log10_nM`, `kd_log10_nM`,
  `koff_log10_per_s`, `half_life_log10_min`, `tm_celsius`.
- Standardize each regression target with fixed **training-only** location and
  scale; save these constants in the checkpoint and invert them for reporting.
  Compute concentration/rate/duration transforms before standardization.
- New primary target policy: log10 of valid positive physical values, with
  exact values and censor boundaries preserved; no silent 50,000 nM clipping.
  Invalid units/nonpositive values are rejected with source IDs and reasons.
  An endpoint needs at least two distinct exact training values to establish
  its scale; otherwise defer the head. Explicit numerical scale floors are
  recorded, not inferred from validation/test.
- This target policy is a new versioned contract. Recompute baselines on the
  same admitted rows/target space. Historical clipped-target runs remain
  immutable and are not treated as directly comparable experiments.
- Use linear outputs without sigmoid/range clamps in the regression path.
  Public APIs return values with units and support status; do not silently
  produce inactive/unqualified outputs.

## 3. One coherent observation-to-loss contract

One normalized row describes **one measured outcome**:

```text
observation_id, source/version, study/assay/replicate identity,
peptide, original/resolved MHC pair, mhc_class,
endpoint, raw_value, raw_unit, qualifier, transformed_target,
quality/provenance, biological context, split/group IDs
```

Metadata can be rich while predictive inputs remain small.

| Source outcome | Model head / disposition |
| --- | --- |
| Direct IC50 | IC50 only |
| Direct KD | KD only |
| Direct EC50 | EC50 only |
| KD reported via IC50 / EC50 proxy | Separate explicitly named proxy heads if admitted; never relabel as direct KD or direct IC50/EC50 |
| koff, half-life, Tm | Own direct scalar heads if admitted; no imposed kinetic identity |
| kon | No head until suitable data exist |
| Elution/MS | Separate future observation task; one loss per bag, no MS alias or processing/presentation pseudo-targets |
| Search-qualified non-MHC MS | One bulk-observation outcome per peptide/peptidoform and eligible search scope; trains digest and shared MS components |
| Independent available-peptide/MS standards | MS-response endpoint only when actual inputs/labels support that interpretation |
| Direct processing/digest measurements | Own measured endpoint on the appropriate component; no label invented from observed products |
| Molecular identity / phenotyped TCRs | Class/species and optional coreceptor-association auxiliaries from their actual annotations |
| T-cell / paired-TCR functional response | Separate future measured-response task after source/QC qualification |
| Structure, unknown endpoints, unsupported qualitative outcomes | Retain and report as excluded; do not coerce into concentration or negatives |

For standardized prediction `p` and standardized exact target/bound `b`:

- Exact: `(p - b)^2`.
- Upper bound (`value <= b`): `max(0, p - b)^2`.
- Lower bound (`value >= b`): `max(0, b - p)^2`.

Propagate original qualifier direction through monotone transformations.
Bounds are not exact values. Missing targets contribute no loss. Report exact
regression and censor satisfaction separately.

**Weighting:** deterministic shuffled passes over all admitted training
observations, with a mean over active observation losses. One scalar measurement
contributes once to its selected endpoint. No equal-task normalization that
turns one rare measurement into the weight of a large common family; no hidden
oversampling or learned uncertainty weights. Endpoint standardization prevents
unit scales alone from determining influence. Report realized endpoint counts,
loss shares and shared-encoder gradients.

Identity annotations are separate supervised outcomes, not aliases of an assay
value. Pretrain class/species heads and the shared encoder on curated molecular
labels, then retain explicit identity supervision during assay fitting:
`L_total = L_assay + 0.1 * L_identity`, with `L_identity` the mean CE over valid
class and per-chain species labels in one separately sampled identity batch per
assay update. Freeze the identity sampler, vocabulary, splits and replay budget;
report identity accuracy/calibration and its gradient share. Do not repeat one
MHC identity label for every peptide simply because it occurs in many assays.
Assay labels never supervise species/class, and inferred defaults never become
identity targets. The fixed 0.1 coefficient is a declared starting choice, not
a learned uncertainty scheme or a proven optimum.

Deduplicate identical exported measurements through source lineage before
splitting, including overlap between datasets. Preserve genuine independent
replicates and their grouping; report contradictory repeated measurements.
If a source reuses one readout in multiple rows/allele candidates, retain one
measurement objective or an explicit group/bag, not several independent labels.
Do not resolve contradictions using validation/test targets or current model
predictions.

### One model family across sources and subsets

There is one canonical preparation, model, loss-routing and checkpoint path.
Source adapters populate a typed observation schema; they do not create custom
models or quietly union unrelated cached files.

| Available data | Inputs needed | What it can train |
| --- | --- | --- |
| MHC sequence/reference catalog | Resolved groove pair and molecular labels | Encoder, class/species heads; no peptide/affinity labels required |
| Annotated receptor sequences / phenotyped cells | TCR chains and observed species/coreceptor labels | Encoder and admitted TCR auxiliaries; no pMHC specificity claim |
| Direct binding/stability assays | Peptide, resolved grooves, endpoint/value/unit/qualifier | pMHC registers/fusion and that endpoint's head |
| Qualified non-MHC bulk MS | Peptide, digest, search denominator, source context when available, acquisition evidence | Digest + shared MS + bulk observation |
| Raw bulk positives without search scope | Observed products and source evidence | Positive-only/auxiliary use only under an explicit policy; no fabricated binary denominator |
| MHC-elution MS, including donor/Atlas sources | Peptide, candidate MHC set, APC/sample context and observation scope | Cellular availability + shared MS + one elution observation loss |
| Matched cell-line perturbations | Above plus component state and control/parent/replicate linkage | Context effects on measured cellular outcomes |
| Direct processing or independent detector measurements | Relevant sequence/process/acquisition context and actual measured outcome | Corresponding component anchor |
| T-cell assays without a paired receptor | pMHC and relevant cell/exposure context | Separate population-level measured response; never the paired-TCR head |
| Validated paired TCR response | Both TCR chains, peptide, MHC pair, measured outcome and relevant context | Receptor/pMHC fusion and measured functional-response head |
| Unvalidated database associations / structural records | Provenance, quality and any valid molecular annotations | Explicitly admitted evidence/identity/structure tasks only; not automatically functional labels |
| Generated controls | Exact generation provenance | Declared controls or optional auxiliary objective; never independent biological validation |

Unavailable TCRvdb access and incomplete source context remain admission
constraints for those records, not reasons to stop fitting other supported
subsets. All source records receive an admitted/excluded/deferred disposition.
Retain unsupported fields/outcomes in provenance rather than corrupting their
meaning to force them into a head.

**Subset construction is deterministic.** The run declares sources, endpoints,
identity tasks, MHC classes, species, machinery components, enzymes and
acquisition features, plus any pretrained checkpoint. Build the union of
modules reachable from these admitted losses. Missing TCR/MHC inputs are
inapplicable to identity-only/bulk-only rows, not dummy sequence tokens.
Missing labels carry masks and never become zeros.

- A binding-only run allocates pMHC modules and its supported identity/assay
  heads; it does not allocate a random detector or receptor predictor.
- A bulk-only run needs no MHC, APC or TCR. It trains bulk observation and shared
  MS features, with isolated-detector qualification only if anchors exist.
- An elution-only run may train the combined observation model. Separate
  affinity/detection outputs remain unsupported unless inherited from qualified
  checkpoints or directly supervised; internal features can be indirectly trained.
- A TCR-auxiliary-only run needs no candidate pMHC. A specificity run requires
  candidate pMHC and receptor labels.
- A mixed run uses the union of these same modules and routes each row/bag once.
  The `"all"` selection means all admitted source families, not every nominal
  head or every unqualified database row.

Use per-output status: `not_allocated`, `random/untrained`, `indirectly_trained`,
`directly_supervised`, `pretrained_frozen`, and separately `validated` with a
named evaluation domain. Random outputs are never served. A qualified frozen
component is allowed and explicitly excluded from trainable-gradient coverage;
if its shared upstream encoder changes, it is no longer functionally frozen.
An assay-only subset can use supplied class/species without allocating fresh
identity classifiers; inferred-identity mode requires trained identity weights
or identity data. Never pretend single-species assay rows teach a general
species classifier.

The same prepared observation is used in training, prediction, evaluation and
export. Required fields depend on the requested endpoint, and unsupported
requests fail with the missing capability/input instead of returning a random
score. A source manifest records exact files/hashes, populations, label policy,
units, missingness, dataset overlaps and frozen row dispositions.

Within a declared cohort, default assay weighting remains observation-weighted.
For mixed-source training freeze source selection, sampling/exposure budgets,
loss coefficients, sampling corrections and replay before launch; report
unique observations and repeated exposures separately. Do not let millions of
bulk rows silently redefine an IC50 milestone, or give one rare kinetic row
equal weight to an entire affinity family. Use staged pretraining plus declared
replay/noninferiority gates; any alternative source balancing is an explicitly
registered condition. Large-corpus ingestion must stream/batch observations
and candidate bags rather than materialize every register of every row at once.

Checkpoint the architecture/version, active endpoints and modules, context
vocabularies/default policy, target scalers, source/support manifests, inherited
weights, optimizer state and qualification status. Resume/inference validates
these contracts and fails on incompatible schema or mismatched module sets.

## 4. Data and feature admission before training

- [ ] Pin source bundle URLs/hashes, exporter/runtime versions, MHC catalog,
  normalization code, membership, source endpoint and target-policy receipts.
- [ ] Compare the latest available Hitlist export with the frozen diagnostic
  source. Use a new cohort only when its manifest is available and its actual
  fields/lineage pass admission. The existing training subset is sufficient for
  the initial fitting diagnostic; do not block that on an export refresh.
- [ ] Apply current role/missingness corrections, including #115/#123 where
  relevant. Keep assay-applicability distinctions such as purified versus
  cellular binding in provenance.
- [ ] Export a census per endpoint/class/allele/study/qualifier/split: raw,
  admitted, deduplicated, excluded, exact, censored, missing, and units.
- [ ] Separately qualify molecular identity labels from sequence/reference
  metadata, with both MHC classes and supported species represented. Deduplicate
  identical groove sequences, report conflicting taxonomy and hold out related
  sequence groups for identity validation. Declare which assay-held-out alleles
  were seen during identity pretraining; sequence exposure is not unseen-allele
  generalization even when no assay label was used.
- [ ] Inventory attributed processing-defect/control samples, paired parental
  lines, clones, rescues, biological replicates, MHC expression and quantitative
  outcomes. Missing arm identity or a familiar cell-line name alone does not
  establish a matched comparison or current machinery state.
- [ ] Check target arithmetic independently on known source rows before model
  construction. Never judge a transform by coverage counts alone.
- [ ] Split before fitting preprocessing. Group identical observations,
  replicates and reused lineage; primary evaluation is peptide-disjoint.
  Report study/donor/specimen overlap and unknown identities. Add study/allele
  holdouts as separate generalization questions when adequately supported.
- [ ] Preserve exact admission failures, including unresolved grooves/partners,
  invalid values and unsupported outcomes. Do not rescue them through fallback
  sources or invented labels.

The historical test set has already been inspected. Further results on it are
exploratory. A new blind claim requires a genuinely unused cohort and a frozen
evaluation policy; reshuffling previously trained/evaluated rows does not
create an independent external test.

## 5. Implementation and experimental sequence

### A. Implement the small model and loss path

- [ ] Add a versioned small-model checkpoint type and one supported prediction
  path. Preserve historical checkpoint loading for reproduction.
- [ ] Reuse sequence resolution, lineage, splitting, output coverage and
  experiment infrastructure. Replace the predictive graph/loss routing; do not
  create a second ad hoc trainer or bypass canonical evaluation.
- [ ] Unit-test padding/batch invariance, sequence sensitivity, class-II partner
  handling, target transforms, censor direction, output routing, missing labels
  and checkpoint round-trip.
- [ ] Verify class-II core counts/positions (9 residues -> one core; 15 -> seven),
  core-relative positions, both PFRs, empty-flank semantics, short-peptide
  rejection, candidate masks, within-class normalization, and exactly one
  measured-peptide loss. Verify class I preserves the full peptide.
- [ ] Verify identity prediction/override paths independently, preserving raw
  predictions and provenance. Test human MHC in nonhuman APCs, mixed molecular
  species, independent receptor/repertoire roles and acyclic default resolution.
- [ ] Verify known component loss overrides species defaults, concurrent defects
  remain separate, unknown does not mean control, and biological context changes
  leave residue embeddings and purified-binding outputs invariant.
- [ ] Verify bulk-only, identity-only, binding-only, elution-only, receptor-only
  auxiliary, paired-response and mixed batches through the same preparation,
  model, loss and checkpoint paths. Missing modalities do not create negatives.
- [ ] Verify MS features are invariant to digest type/MHC/APC/TCR changes for
  fixed peptide chemistry and acquisition; digest features respond to enzyme
  context, and cellular features respond only to their allowed biological state.
  Adding a TCR leaves fixed-checkpoint pMHC assay outputs unchanged.
- [ ] Verify one final MS observation loss, shared detector parameter identity
  across bulk/elution routes, proxy evidence status, qualified candidate masks
  and correct gradient reachability for each admitted source.
- [ ] Test class-only/species-only and detector-only anchor training without
  allocating unused assay modules. Report any indirect component as such.
- [ ] Assert that changing nonpredictive metadata leaves a fixed prediction
  unchanged. Changing one endpoint label must not add losses to other heads.
- [ ] Make unsupported-head requests fail explicitly. Training support and
  validation qualification are separate checkpoint fields.

### B. Pass a training-only memorization gate

- [ ] First qualify the identity stage with supervised sequence/reference
  labels; record its own bounded pretraining recipe before execution. Require
  held-out identity metrics by class/species, a majority baseline, confidence
  calibration and an explicit unsupported/OOD policy. Never establish identity
  prediction quality from the human-only assay cohort.
- [ ] Deterministically choose 32 unique, internally consistent exact-IC50
  input tuples from training only, with both classes represented and at
  least one log10 unit of target range. Fix IDs and seed before execution.
  Exclude conflicting identical-input labels from this diagnostic only and
  report them; keep them in the full-data noise audit.
- [ ] Include class-II peptides with multiple candidate registers and varied
  target values. Report fitting separately for class I and II. If adequate
  class-II training data are unavailable, withhold the two-class support claim.
- [ ] Train the entire small model, not just its head: AdamW, constant learning
  rate `3e-4`, zero weight decay/dropout, full batch, at most 2,000 steps.
  Record a resource/time ceiling before launch and preserve a budget failure.
- [ ] Prespecified diagnostic gate: training RMSE <=0.10 log10(nM), Spearman
  >=0.95, and prediction/target SD ratio between 0.8 and 1.2, overall and in each
  sufficiently varied class subset. A class-I-only pass cannot hide class-II
  failure. No generalization or accurate-core-localization claim follows.
- [ ] Record nonzero class-II gradients through register pooling and candidate
  fusion. Synthetic position/shift fixtures verify code semantics, not a
  biological claim that changing peptide flanks leaves affinity invariant.
- [ ] Save initial/intermediate/final losses, predictions, source IDs, parameter
  and gradient summaries. Reload the checkpoint and reproduce predictions.
- [ ] If it fails, stop expansion. Check transforms, masks, optimizer membership,
  gradients and input dependence first. Change one factor at a time under a
  new registered condition; do not hide the failure with a longer multitask run.

### C. Establish a useful IC50 baseline

- [ ] Start with all admitted exact-IC50 training rows, then add censored IC50
  as a separate controlled condition with unchanged exact-value reporting.
- [ ] Train constant-mean and existing length/composition/class ridge baselines
  on the identical training rows and new target space. Retain the old baseline
  artifacts as historical results rather than reusing their numbers.
- [ ] Fixed first full-fit recipe: the architecture above, batch size 64,
  AdamW `lr=3e-4`, weight decay `1e-4`, dropout .1, constant learning rate;
  complete epochs, maximum 100, validation patience 15 after at least 10 epochs.
  Use seeds 42, 43 and 44; register the exact budget before launch. No arbitrary
  100-batch truncation, aggregate multitask checkpoint selection or test tuning.
  Start each full-cohort condition from the same qualified identity-pretrained
  encoder/head checkpoint and freshly initialized assay modules, rather than
  the memorized 32-row checkpoint; fit scaling on its own full training set.
- [ ] Evaluate both supplied-identity and predicted-identity modes. Report
  identity/default mistakes separately from assay-regression errors. A supplied
  class run alone does not qualify the sequence-only inference cascade.
- [ ] Select by validation exact-IC50 RMSE; report censored performance alongside
  it. Baseline and model selection never use test labels.
- [ ] Require clear fitting on the complete training set, then a validation
  improvement over both mean and ridge: the upper bound of the paired
  peptide-bootstrap 95% RMSE-difference interval must be below zero. Preserve
  all seeds; do not cherry-pick a seed using test performance.
- [ ] Track prediction SD, SD/target-SD, quantiles, range, correlation and
  calibration by class/allele where supported. Flag SD/target-SD <0.1 as
  contraction requiring investigation; variance by itself is never success.
- [ ] Require a separately reported class-II result against matched baselines;
  pooled improvement dominated by class I does not qualify class-II prediction.
  Freeze class-specific uncertainty/support requirements before fitting.
- [ ] Where held-out binder/nonbinder support is adequate, require nontrivial
  500 nM decisions rather than the current all-nonbinder result. Report counts,
  precision/recall/F1, balanced accuracy, AUROC, AUPRC and uncertainty, with
  conservative treatment of ambiguous censored bounds.
- [ ] Run a training-label-permutation control under the same fitting budget.
  It may memorize training data but must not reproduce real-label validation
  improvement. This is a leakage/shortcut check, not a source of new labels.

Do not manufacture output variation through a variance reward, arbitrary
rescaling, noise injection or fitting a calibrator on held-out labels.

### D. Diagnose and add other measurements incrementally

Use a bounded comparison to separate architecture from objective effects:

| Condition | Architecture | Loss/data scope |
| --- | --- | --- |
| Historical reference | Archived model | Archived objectives/target policy; descriptive only |
| Controlled legacy IC50 arm | Legacy architecture | The new matched IC50-only cohort/target policy |
| Proposed primary baseline | Small model | Same IC50-only cohort/target policy |
| Incremental multitask candidate | Small model | IC50 plus one separately routed measured family |

Reusing old run metrics as a matched causal ablation is invalid because the
target policy, budget and objective differ. Avoid a large factorial sweep.
Match the identity-supervision corpus/replay policy and pretraining budget
between prospective architecture arms as well; otherwise name the changed
training regime and do not attribute the difference solely to architecture.

- [ ] Admit direct KD, half-life, Tm, EC50 and the two distinct proxy families
  one at a time, with fresh per-head support and baseline reports. The order is
  fixed from the pretraining census by exact training support, not validation
  outcomes. Treat koff's current 19 rows as sparse/exploratory; do not give them
  equal-task weight or claim well-identified kinetics.
- [ ] Construct only heads with admitted support and defined training scaling.
  Each extension trains the shared encoder/fusion and all active heads jointly.
- [ ] For promotion, the added head must pass its prespecified endpoint quality
  screen and IC50 must remain noninferior: paired validation upper 95% RMSE
  difference versus the IC50-only model <=0.02 log10(nM). Report precision
  limitations rather than silently relaxing a failed/underpowered gate.
- [ ] If a family harms IC50 or cannot be qualified, keep its data and results,
  but leave its head out of the serving checkpoint. Do not preserve a random or
  unsupported head merely to advertise output breadth.
- [ ] Compare only matched source/target/split/hardware contracts. Any later
  change to sampling, weighting, architecture or optimization gets its own
  declared condition and mechanism checks.

### E. Admit MS and TCR functionality when their evidence supports it

**First cellular milestone: known processing defects with matched controls.**
This is a planned source of informative supervision, not a speculative distant
extension. It follows the bounded fitting gate; dataset qualification can
proceed alongside identity/IC50 work. Expected improvement must be measured.

The current adapter already requests `apm_genes_perturbed`, `condition_category`,
`sample_attribution`, `is_control_arm` and `sample_label`
([source](https://github.com/pirl-unc/presto/blob/6196143048fa9eaf6257b7826f1adb5a14f9c0a9/data/hitlist_source.py#L166)).
An empty gene list can mean either a genuine control or an unresolved arm.
Study-level perturbation rollups must never be copied onto every sample.
Existing mechanism-group collapse is insufficient for concurrent component loss.

Prioritize independently identifiable control/defect/rescue comparisons in a
shared background. Primary-source examples worth checking against Hitlist's
actual export and sample manifests include an
[isogenic antigen-processing component panel](https://pmc.ncbi.nlm.nih.gov/articles/PMC12090245/),
[class-II DO comparisons](https://pmc.ncbi.nlm.nih.gov/articles/PMC6398211/) and
[DM-dependent quantitative peptidomes](https://pmc.ncbi.nlm.nih.gov/articles/PMC5943503/).
These references show suitable comparison designs; this issue does not claim
their complete quantitative data are already in the admitted export.

- [ ] Preserve study, sample, parent line, clone, intervention/control/rescue
  links, component identities, validation evidence, concurrent state, MHC
  identities/expression, source mappings and biological replicates.
- [ ] Preserve quantitative peptide/protein measurements, normalization,
  acquisition batch, detection limits and outcome missingness. A peptide
  undetected in one MS run is not automatically a measured biological negative.
- [ ] Choose the primary outcome from what is actually measured: quantitative
  presentation/abundance when supported, or an explicitly qualified observed
  elution ranking/PU task when only detections are available. Preserve
  multi-allele uncertainty; one MS detection remains one bag-level observation.
- [ ] Group matched arms during sampling and keep parent/clone/study groups
  together in evaluation splits. Cell-line IDs are grouping keys, not
  embeddings. Unmatched known-defect lines can supplement the corpus, but do
  not by themselves identify an intervention effect.
- [ ] Supply source-protein flanks/boundaries for processing-dependent outputs
  when resolved; retain unknown/multiple mappings. Keep biological source
  processing separate from sample-preparation protease digestion.
- [ ] Train the same small contextual model on measured cellular outcomes.
  Do not manufacture scalar TAP/processing/detectability targets from an MS
  label. Do not force every machinery loss to reduce every peptide: observed
  repertoires can change in both directions, and alternate presentation remains.
- [ ] Evaluate held-out within-comparison changes, effect direction/ranking,
  per-component/per-allele performance and overall cellular prediction, with
  replicate/parent/study uncertainty. Avoid normalization that erases total
  presentation changes if the measured outcome includes them.
- [ ] Compare the context-aware model with the same model given constant or
  withheld machinery context on identical observations/splits/budgets. The
  control changes context information, not dataset size or labels. A genuine
  gain must survive held-out biological backgrounds; report narrow support
  where only one background exists.
- [ ] File upstream requests only for confirmed missing export fields after
  inspecting the latest source schema. Prioritize missing component state,
  control/parent linkage, quantitative outcomes and detection-limit provenance.

The historical 3,088 all-positive MS observations do not independently support
ordinary positive/negative discrimination. Before any ranking task, require
a documented candidate/exposure universe and measured controls or a declared
case-control/positive-unlabeled objective. Generated background is not a measured
negative. No simultaneous presentation/processing/detectability targets copied
from the same observation. Evaluate paired differences from the measured
predictions; do not silently count every observation again as an extra contrast
loss without a separately declared objective.

**Paired TCR:** the intended source is the independently retested resource in
[Messemaker et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12132471/) and the
[author repository](https://github.com/schumacherlab/TCRvdb), not a VDJdb
confidence-score filter. Obtain the authorized TCRvdb release, verify its exact schema,
QC/outcome policy, alpha/beta pairing, full-length/reconstruction provenance and
peptide/allele/class coverage.

The paired model has five external sequence components:
`TCR_alpha, TCR_beta, peptide, MHC_groove1, MHC_groove2`.
Encode each TCR chain with the same sequence-only encoder, retaining chain
identity by its feature slot. Full-chain input does not imply complete
experimental sequences when a source supplies only CDR3/V/J; reconstruction
must be explicit and never silently substituted for observed full length.

For each pMHC candidate register, the shared context fusion receives the two
TCR vectors, its pMHC candidate vector, a full-peptide vector and allowed
species/context. The full-peptide vector preserves information beyond the
nine-residue core/local PFR composition. Apply register pooling after receptor
conditioning; do not discard alternative class-II registers using the affinity
model's single best core before considering the TCR. Pooling weights describe
the predictor's attention, not measured occupancy.

Initially add one directly supervised **functional-response** head if that is
what the validated data measure. A sigmoid gives a response score for that
measurement context; it is not automatically TCR binding KD or calibrated
recognition of purified pMHC. Pure receptor/pMHC compatibility and cellular
functional response have different allowed context, as specified above.
Jointly fine-tune the shared encoder/fusion and all active heads with retained
identity/affinity supervision, subject to prespecified noninferiority checks.
Do not silently reuse the failed frozen backbone and call it trained end to end.

For a **TCR-only query**, the service must receive or select a declared candidate
library of peptides paired with MHC molecules, evaluate receptor–pMHC scores,
then return a ranked list with candidate coverage and support. pMHC remains an
input to the pair scorer. The receptor does not uniquely determine one peptide
or allele; top-ranked candidates are not a de novo epitope sequence or a
universal restriction assignment. Evaluate retrieval separately from pair
classification. Class/allele restriction claims require a suitably varied,
independently evaluated candidate panel, not just correctly classifying the
supplied MHC molecule in stage 1.

Use only measured QC-passing positive/non-validating outcomes in their assay
scope. Unknown, unassayed and QC-failed pairs are not negatives. Audit
receptor-cluster, peptide and study leakage separately. A narrow allele/class
panel does not establish general restriction prediction. CDR-aware pooling or
residue cross-attention can be evaluated later if this simpler baseline and
coverage show a specific limitation; do not add them speculatively.

These extensions do not add free-standing recognition, processing,
immunogenicity or kinetic-component outputs without independent evidence.

## 6. What “all of the model trains” must mean

- [ ] Every instantiated trainable module is reachable from an admitted active
  loss, is present in the optimizer, receives finite nonzero gradients over the
  appropriate training window, and records parameter updates.
- [ ] Every active head receives its own measured labels every complete epoch.
  Report unique observations separately from repeated optimizer exposures.
- [ ] Identity heads receive actual identity supervision; register pooling
  receives class-II assay gradients; machinery features receive attributed
  condition variation; TCR features receive receptor-specific labels.
  Reference profiles and default-filled fields do not establish any of these.
  Deterministic resolvers/masks have no parameters to train.
- [ ] Log per-module and per-head gradient/update coverage and optimizer state
  alongside output support. Persistent zero-gradient branches fail the run
  preflight/closure checks until removed or explicitly justified.
- [ ] Distinguish genuine gradients from weight-decay-only movement. Do not
  claim unused embedding categories, padding rows or unobserved positional
  entries were learned; these fixed/unused entries must be identified explicitly.
- [ ] Check sequence sensitivity on training examples and report within-allele
  peptide discrimination and within-peptide allele discrimination where such
  measured contrasts exist. Unmeasured substitutions are sensitivity probes,
  not biological ground truth.
- [ ] Remove dormant branches and unsupported head requests from the normal
  serving model. A fixed algebraic inverse transform is not an untrained head.

Training coverage, useful training fit, held-out performance and biological
interpretation are four different claims. Publish evidence for each separately.

## 7. Deliverables and completion criteria

- [ ] One short canonical model spec and one shared train/inference preparation
  path. Update #46's desired-versus-supported inventory explicitly; preserve
  role-specific context boundaries for future biology.
- [ ] Identity outputs, resolved species/class, per-field default/override
  provenance, component machinery state and predicted-only versus supplied-
  identity qualification. No loss of underlying source annotations.
- [ ] Explicit MHC-II registers and class-specific training/validation evidence;
  diagnostics do not claim experimentally validated core localization.
- [ ] A normalized, pinned dataset/target contract with source census,
  exclusions, deduplication, lineage and split leakage reports.
- [ ] A reproducible memorization pass and a complete-data IC50 learning result
  with useful variation, matched baselines and prespecified validation gates.
- [ ] A serving checkpoint whose allocated trainable modules were actually
  optimized and whose exposed endpoints each have direct support and qualified
  evaluation. An initially IC50-only assay release, with qualified identity
  outputs, is a complete valid milestone; cellular and TCR claims need their
  own subsequent evidence.
- [ ] Endpoint-specific per-example train/validation/test predictions with
  source IDs, raw/transformed values, units, qualifiers, allele/class and
  residuals; exact/censored metrics and physical-threshold diagnostics.
- [ ] An attributed cell-line control/defect cohort and measured contextual
  comparison against the identical context-withheld baseline; unsupported
  components/backgrounds remain explicit.
- [ ] Non-MHC digestion/MS and MHC-elution observation routes sharing one MS
  component, with acquisition/abundance provenance, search-scope labels and
  independently anchored versus indirect detectability status.
- [ ] A deterministic source/subset capability manifest and one training entry
  point. Every included source/endpoint has declared inputs, target, loss,
  evidence tier, dependency modules and evaluation split policy.
- [ ] When authorized TCR data arrive, five-sequence paired scoring and an
  explicitly scoped candidate-library ranking evaluation, with observed versus
  reconstructed chain provenance and receptor/pMHC split audits.
- [ ] A declared test-use policy and an honest exploratory label where the
  historical test set is reused. No claim of external independence without it.
- [ ] Frozen initialization/selected checkpoints, optimizer/config/source pins,
  launcher/env, dependency versions, loss and gradient traces, runtime/hardware
  and failed attempts. Register every family under `experiments/` and close it
  in `experiments/experiment_log.md`; large artifacts use pinned datacache.
- [ ] Run focused and full software checks, then verify the selected checkpoint
  in a fresh process. Software correctness alone does not pass model quality.

For GPU experiments, follow the repository's explicit `H100!` default, freeze
the budget before launch, and record observed hardware. The initial source and
arithmetic audits are CPU work; this issue does not authorize unbounded compute.

Upstream data work remains tracked in
[Hitlist #678](https://github.com/pirl-unc/hitlist/issues/678),
[#721](https://github.com/pirl-unc/hitlist/issues/721),
[#722](https://github.com/pirl-unc/hitlist/issues/722), and
[#723](https://github.com/pirl-unc/hitlist/issues/723).
Use available valid measurements now; richer exports should enable justified
extensions rather than become an excuse for an unfitted baseline.
