# Model input/output contract

This is the single authoritative I/O design for Presto. It distinguishes the
**desired contract** from the **implemented interface** below. Architecture notes,
training notes and historical experiments do not override it.
[Issue #46](https://github.com/pirl-unc/presto/issues/46) records the audited drift
and remaining implementation work. Status here describes code, not demonstrated
scientific validity.

## 1. One intended design

Presto predicts peptide processing, MHC binding, presentation, repertoire
recognition and immune response, and separately predicts the assays that observe
them.

**Sequence-only encoding; biologically conditioned downstream predictions.**
Residue encoders consume amino acids, positions, segment identity and explicit
missingness/boundaries. They never receive species, cellular state, assay IDs,
dataset IDs or labels. A segment's representation does not depend on whether
another segment was supplied. Sequence integration and biological context occur
downstream through explicitly scoped components.

Biological state is not measurement apparatus. APC cytokine exposure may change
processing; cytokine secretion measured in responding T cells is an outcome.
Antigen delivery/expression may change biology; detection chemistry describes a
measurement. Split mixed source columns into their actual roles, rather than
classifying everything called “culture”, “format” or “stimulation” as forbidden.

Assay descriptors may parameterize fixed output tracks and select losses. A
given track must predict the same function regardless of which assay metadata
accompanies an example. No assay ID is fed back into the predictive input path.

### Desired semantic inputs

These are roles, not a claim that every field is already accepted by the API.

| Input | Meaning | Permitted downstream use | Current support |
|---|---|---|---|
| Peptide | Required amino-acid sequence | Task-scoped sequence paths | Implemented |
| N/C flanks | Adjacent source-protein sequence | Processing and junctions | Implemented |
| N/C boundary status | Observed protein end versus unknown context | Structural window construction | Implemented; carried through row/MIL/presentation paths |
| Presenting MHC molecule | Class-correct groove pair, resolved from sequence/alleles | Binding, presentation, presented-pMHC recognition | Binding/presentation implemented; recognition context incomplete |
| APC MHC set | Co-expressed molecules with class-II partner identities | Per-molecule predictions and bag aggregation | MIL exists; shared serving preparation incomplete |
| Repertoire-system MHC set | MHC shaping the queried organism's selected repertoire | Repertoire recognition and response | Not implemented |
| APC species | Species of the presenting cell | APC biology | Legacy host field only; independent schema pending |
| Molecular MHC species | Origin of each MHC molecule | Molecular inference/routing | Inference and explicit override; source-target separation incomplete |
| Individual TCR species | Origin of a specified receptor | Optional receptor context | Not implemented in canonical model |
| Repertoire-system species | Species of the system whose repertoire is queried | Repertoire recognition/response | Not independently implemented |
| Antigen species | Source organism of antigen | Origin and repertoire-relative context | Source-origin override exists; full relative context pending |
| APC cellular state | Lineage/type, primary versus cell line, disease | Scoped APC biology | Coarse categories implemented |
| APC interventions | Component plus KO, knockdown, inhibition, etc.; concurrent interventions | Affected processing/loading/assembly components | Coarse single category; richer schema pending |
| APC cytokines | Concurrent exposures, dose/time where observed | APC-state conditioning | Coarse single category; richer schema pending |
| Capture/digestion | Ligand capture versus protein digest and enzyme | Explicit process routing | Partial |
| Expression/source protein | Protein identity and sample abundance | Explicit abundance/observation contribution | Mapping lineage exists; abundance model pending |
| Assay descriptors | Family, method, prep, geometry, readout, protocol | Output tracks and supervision only | Binding and T-cell per-axis panels |
| Acquisition descriptors | Instrument, fragmentation, depth, FDR, labeling | Observation tracks only | Mostly pending |
| Explicit annotations | Role-specific class/species information and its provenance | Typed overrides; conflicts reported | Legacy overrides only |
| Optional TCR sequences | Paired alpha/beta or other explicitly supported receptor chains | Sequence-only receptor encoder and specific matcher | Future, not active canonical matching |

Do not infer APC species from MHC species, TCR species from repertoire species,
or selection-system MHC from the presenting molecule. A human MHC in a mouse APC
must be expressible without relabeling either. Antigen species alone is not a
definitive self/nonself label.

Unknown is not wild type, untreated, a protein terminus, a negative observation,
or an annotation copied from another role. Preserve observed/inferred status and
concurrent interventions. Missing optional context permits a prediction under
declared unknown context, not a claim of calibration for that missingness pattern.

### Desired dependency boundaries

| Component | Sequence access | Additional biological context |
|---|---|---|
| Residue encoders | Each segment alone | None |
| Intrinsic binding/kinetics/stability | Peptide and presenting groove pair | Molecular identity/routing; no APC cytokines, source flanks or host-state shortcut |
| Processing/junctions | Peptide processing features and source flanks | APC species/state, affected machinery and interventions |
| Presentation | Processing and binding/assembly information | Presenting APC/MHC set and loading/assembly state |
| Repertoire recognition | Presented pMHC, not peptide-only | Selection-system MHC and species, antigen-origin context |
| Immunogenicity | Presented pMHC and recognition | Relevant response/repertoire context |
| Assay observations | Relevant biological representations | Fixed output descriptors, never observed assay-ID features |

Recognition may use presented pMHC plus repertoire context. This supersedes the
old peptide-plus-foreignness-only restriction. A specific-TCR matcher is a
separate future capability; pMHC-only receptor evidence is not that matcher.

## 2. Implemented sequence and forwarding interface

The main entrypoint is `Presto.forward` in [models/presto.py](https://github.com/pirl-unc/presto/blob/main/models/presto.py).
The current stream is `Nflank | peptide | Cflank | mhc_a | mhc_b`, with shared
residue weights and segment-blocked attention. There are no global
species/completeness embeddings. Host context enters processing downstream.

`mhc_a`/`mhc_b` mean **groove segments**, not two arbitrary full chains:

| Class | `mhc_a` | `mhc_b` |
|---|---|---|
| I | Heavy-chain alpha1 / groove1 | Same heavy-chain alpha2 / groove2 |
| II | Alpha-chain alpha1 / groove1 | Beta-chain beta1 / groove2 |

Class-I beta2m is not a token input. A beta-only DQ/DP/DR observation can yield
an empty first segment and a resolved second segment. Do not discard its resolved
lineage or invent an alpha partner. `mhcseqs` provides resolution/extraction;
source alleles and model-facing resolved alleles remain separate.

Raw tensor interface (batch size B):

| Argument | Current meaning |
|---|---|
| `pep_tok`, `mhc_a_tok`, `mhc_b_tok` | Integer residue tensors [B,L]; padding 0; groove semantics above |
| `flank_n_tok`, `flank_c_tok` | Optional [B,L] tokens; N-flank truncation keeps the junction-nearest residues |
| `flank_n_is_terminus`, `flank_c_is_terminus` | Optional [B] booleans: supplied short flank reaches a known protein boundary |
| `mhc_class` | Optional class annotation/override of inferred class probabilities |
| `mhc_species` | Explicit molecular-species override, never implicitly taken from host species |
| `species`, `immune_species` | Existing host/processing context; explicit immune_species takes precedence; not independent APC/repertoire roles |
| `species_of_origin`, `peptide_species` | Source-origin override and existing old alias; not full repertoire-relative foreignness |
| `machinery` | Downstream excision machinery selector |
| `provenance` | Current coarse process/cell-state tensor dictionary, detailed below |
| `binding_context`, `tcell_context` | Existing accepted but ignored forward arguments; metadata is used by losses instead |
| `return_binding_attention` | Diagnostic output request, not a biological input |

The existing aliases are listed to describe reality, not to endorse another
compatibility layer. Replacing them with the role-specific schema is pending.

Current `provenance` keys: `peptide_source_idx`, `enzymatic_digest_idx`,
`apm_perturbation_idx`, `processing_stimulus_idx`, `cell_lineage_idx`,
`sample_origin_idx`, `disease_state_idx`. Cell-lineage/origin/disease and host
embeddings reach processing latents. APM/stimulus profiles act in excision;
capture/digest selects the relevant process. This dictionary is not the full
desired typed context.

`PrestoBatch.model_inputs()` is the one row-level input assembly used by canonical
training and held-out scoring. MIL materialization, capping and contrastive
construction preserve source boundary flags; swapped MHC does not change the
anchor's boundary facts. Model convenience forwards and presentation prediction
carry flags too. Protein tiling derives flags when a supplied flank reaches the
protein end, including short nonempty flanks.

Current boundary padding uses residue `X`, unknown-window padding uses `?`, and
missing segments use `<MISSING>`. `X` also represents an ambiguous residue:
these are not a fully disjoint structural alphabet. No null may be stringified
into `"NAN"`.

The public `Predictor` returns selected structured results, not the whole raw
dictionary. Presentation accepts flanks and boundary flags. Its complete
biological-context API and multi-allele preparation are still incomplete.
`predict_recognition` is repertoire-level, not TCR-specific.

## 3. Outputs: measurements first

A tensor's existence does not prove label support, identifiability, calibration
or held-out validity. Count aliases once. Panel entries are per-axis outputs,
not independently validated assays or a Cartesian registry of all joint setups.

### Quantitative peptide–MHC assays

Default `assays` dictionary in the log10 binding configuration:

| Key | Physical quantity/unit | Raw tensor space |
|---|---|---|
| `KD_nM` | Dissociation constant, nM | log10(nM) |
| `IC50_nM` | Half-maximal inhibition, nM | log10(nM) |
| `EC50_nM` | Half-maximal effect, nM | log10(nM) |
| `KD_proxy_ic50_nM` | KD reported via IC50 proxy, nM | log10(nM); aliases KD in default merged grouping |
| `KD_proxy_ec50_nM` | KD reported via EC50 proxy, nM | log10(nM); aliases KD in default merged grouping |
| `kon` | Association rate, M^-1 s^-1 | log10(rate) |
| `koff` | Dissociation rate, s^-1 | log10(rate) |
| `t_half` | Dissociation half-life, minutes | log10(minutes) |
| `Tm` | Melting temperature, Celsius | (T - 50) / 15 |

The `_nM` suffixes do **not** mean raw tensors already contain physical nM.
Public affinity results perform conversion. Other binding target modes must be
interpreted using the saved model configuration, never by guessing from names.
Optional binding configurations add IC50/EC50 anchors and
`__method__...` or `__prep__...__readout__...` leaves. These are the same
measurement families, not additional biological quantities.

Fixed binding panels:

| Output | Ordered vocabulary |
|---|---|
| `binding_assay_panel_assay_type` | unknown, KD, KD_PROXY_IC50, KD_PROXY_EC50, IC50, EC50, OTHER, T_HALF, TM, KOFF, KON |
| `binding_assay_panel_assay_prep` | unknown, PURIFIED, CELLULAR, LYSATE, BINDING_ASSAY, OTHER |
| `binding_assay_panel_assay_geometry` | unknown, COMPETITIVE, DIRECT, T_CELL_INHIBITION, OTHER |
| `binding_assay_panel_assay_readout` | unknown, RADIOACTIVITY, FLUORESCENCE, OTHER |

These 26 entries currently use the binding target space and the binding-row
objective. T_HALF/TM/KOFF/KON vocabulary entries do not imply family-specific
kinetics/stability panel training. Generic binding loss still also pushes
heterogeneous measurements toward KD; full measurement-family separation is
open work. Every consumer of a censored measurement must respect its qualifier.

### MS/elution observations

- `elution_logit`, `elution_prob`: MHC-elution observation.
- `ms_logit`, `ms_prob`: current aliases of elution, not another assay.
- `ms_detectability_logit`: detectability score; optional bulk-MS uses proxy
  supervision. It is not independently identified by naming the latent.
- Acquisition-specific tracks and an explicit abundance contribution: pending.

### T-cell response assays

`tcell_logit`/`tcell_prob` predict the unknown-reference response. This is
**not** a mathematical marginal over all assay protocols.
`tcell_panel_logits` (old alias `tcell_context_logits`) emits response logits
under six fixed per-axis sweeps, with other axes at the same reference baseline:

| Axis | Ordered vocabulary |
|---|---|
| `assay_method` | unknown, ELISPOT, ICS, MULTIMER, ELISA, CYTOTOXICITY_ASSAY, PROLIFERATION_ASSAY, IN_VITRO_ASSAY, IN_VIVO_ASSAY, BIOASSAY, OTHER |
| `assay_readout` | unknown, IFNG, TNFA, IL2, IL4, IL5, IL10, GMCSF, CYTOTOXICITY, PROLIFERATION, ACTIVATION, QUAL_BINDING, KD, MULTIMER_BINDING, OTHER |
| `apc_type` | unknown, DENDRITIC, B_CELL, PBMC, SPLENOCYTE, T2_B_CELL, B_LCL, T_CELL, OTHER |
| `culture_context` | unknown, DIRECT_EX_VIVO, SHORT_RESTIM, IN_VITRO, IN_VIVO, ENGINEERED, CELL_LINE_CLONE, NON_SPECIFIC_ACTIVATION, OTHER |
| `stim_context` | unknown, EX_VIVO, IN_VITRO_STIM, IN_VIVO, ENGINEERED, OTHER |
| `peptide_format` | unknown, MINIMAL_EPITOPE, LONG_PEPTIDE, PEPTIDE_POOL, WHOLE_PROTEIN, OTHER |

There are 56 entries. Metadata selects a column for BCE against the observed
response; it is **not** a category-classification CE target. Training and
hold-out extraction use the same selector. Unobserved/unknown context columns
are masked from these per-axis objectives. This remains a per-axis approximation,
not joint configuration modeling.

No explicit IL17, GRANZYME_B, PERFORIN, CD107A, LUMINEX, CYTOKINE_CAPTURE,
DEGRANULATION, ACTIVATION_MARKER, TMG or PEPTIDE_MIX track exists today.
The categorical KD readout is not a quantitative TCR–pMHC affinity head.
APC/culture/stimulation categories currently mix roles; extracting genuinely
biological state into the new input schema remains required.

### Receptor evidence (pMHC-only)

- `tcr_evidence_logit`/`tcr_evidence_prob`: evidence that some receptor was
  reported for the pMHC.
- `tcr_evidence_method_logits`/`tcr_evidence_method_probs`: three evidence
  categories, `multimer_binding`, `target_cell_functional`, `functional_readout`.

This is not specific TCR recognition. Canonical `tcr_vec`, `match_logit`,
`match_prob` and receptor-specific affinity are absent.

## 4. Biological predictions and diagnostics

Biological readouts (often proxy-supervised rather than directly measured):

- Processing: `processing_class1_*`, `processing_class2_*`,
  `processing_logit/prob`, and mixed aliases.
- Binding: class-I/class-II/mixed `binding_*` scores/logits/probabilities.
- Excision: `excision_logit/prob`, `excision_n_terminus_score`,
  `excision_c_terminus_score`, `excision_missed_cleavage_score`,
  `excision_length_score`.
- Presentation: class-I/class-II/mixed `presentation_*` logits/probabilities;
  external MIL aggregates per-molecule values.
- Recognition: `recognition_cd8_*`, `recognition_cd4_*`,
  `recognition_repertoire_*` and mixed aliases. Expanded CD4/CD8 readouts use
  their own latents; the old collapsed topology shares one recognition latent.
- Immunogenicity: `immunogenicity_cd8_*`, `immunogenicity_cd4_*`,
  `immunogenicity_logit/prob` and mixture/mixed aliases.

Counterfactual excision panels coexist with scalar biological conditioning:

| Output | Ordered vocabulary |
|---|---|
| `excision_panel_apm` | none, peptide_supply, n_term_trimming, loading_complex, mhc_null, class_ii_loading, other, unknown |
| `excision_panel_stimulus` | none, ifn_gamma, ifn_type1, tnf_alpha, tlr, cell_activation, cytokine_unspecified |

These are coarse axes, not a simultaneous-intervention schema. Current stimulus
normalization conflates missing and none; the desired contract forbids that.

Auxiliary/diagnostic families:

- MHC class probabilities, per-segment fine-type/species logits and probabilities,
  inferred versus overridden values, chain compatibility. Default category counts:
  class 2, fine type 5, molecular species 6.
- `species_of_origin_logits` (11 source-organism categories), foreignness
  logits/probabilities. These are not validated self/nonself measurements.
- Core/PFR candidate starts, lengths, masks, priors, scores and posteriors;
  core membership/positions and core/N-PFR/C-PFR lengths.
- `pep_vec`, `mhc_a_vec`, `mhc_b_vec`, `groove_vec`,
  `pmhc_vec`/`pmhc_interaction_vec`, `latent_vecs`.
  Segment vectors have dimension d_model; pMHC interaction width is separately
  configured (default 256), not necessarily d_model.
- Binding affinity/stability/probe/core/mixed-KD internals, kinetic latents,
  assay/sequence summary vectors, bias/gate/mix diagnostics, optional attention
  and direct-segment diagnostics.
- `presentation_invivo_excision_term`, `immunogenicity_recognition_term`,
  excision machinery selection and mode diagnostics.
- `apc_cell_type_context_vec` is currently **MHC-derived** class/species/
  compatibility context, despite its name; it is not measured APC cell type.

Diagnostic keys vary with model configuration. No timeless total tensor count
is part of the public contract.

## 5. Supervision, lineage and validation

Targets, units, qualifiers and observed output selectors are not predictive
features. Likewise keep dataset/version, observation ID, assay/reference IDs,
PMID/DOI, sample identity, peptide, original/resolved allele sets, mapping
candidates/category, selected protein/transcript/gene/coordinates/proteome,
synthetic parentage, curation fingerprints and split assignment as traceable
record lineage through held-out dumps.

Row task routing is `LOSS_TASK_SPECS` in
[training/supervision.py](https://github.com/pirl-unc/presto/blob/main/training/supervision.py),
alongside explicit binding/excision panel specs. The trainer retains compatibility
imports and the established uncertainty-parameter order.
Bag tasks and effective target/prediction resolution are shared in
[training/mil.py](https://github.com/pirl-unc/presto/blob/main/training/mil.py):

- Binding/family affinities and half-life/Tm: censor-aware quantitative loss.
  Binding descriptor panels now use the same qualifier-aware loss.
- kon/koff: currently MSE; consistent kinetic censoring remains pending.
- Processing, presentation/elution, T-cell/immunogenicity, receptor evidence:
  binary objectives when labels exist; some are proxies sharing the same label.
- T-cell panels: selected-column response BCE, not method-ID classification.
- MHC identity, source species and core position: categorical auxiliary tasks.
- Evidence-method membership: three-component binary auxiliary objective.
- MIL: bag-level Noisy-OR objectives on supported channels; not an assertion that
  every molecule in a positive bag is positive.

T-cell pathway bags supervise each observed panel axis by selecting the same
fixed column at every candidate molecule, then applying Noisy-OR against the
bag response. Unknown selectors have zero panel support. These six
`tcell_<axis>_mil` objectives have explicit base weight 1, separate from ordinary
row objectives, like the existing scalar T-cell MIL objectives. They do not add
learned uncertainty parameters. Bag labels never enable ordinary row masks,
and observed assay descriptors never enter the forward inputs.

Selected-checkpoint validation/test loss and prediction dumps traverse complete
bags in full precision with 128-instance forward chunks. Training and in-loop
validation may retain their configured sampling cap. Dumps preserve the bag ID,
source-row position and lineage, candidate membership indices within each
batch/channel, original/evaluated counts, selected column name/index and bag
BCE. Two 0.5 instances therefore yield 0.75 in both the objective and the dump.
Existing scalar proxy and ms/elution alias objectives retain their weights.

Row exports preserve a categorical class vector per sample, named evidence-method
components with repeated source identity, and each supervised panel's selected
column. Records carry raw numeric source targets, transformed targets, units,
qualifiers, output paths and per-observation loss/reduction weight. In particular,
source half-life values are hours and their training target is log10(minutes).
Masked observations do not participate in loss calculation; malformed active
shapes, nonfinite objectives and missing censor qualifiers fail explicitly.

`<split>_loss_ledger.json` verifies the persisted prediction CSV against effective
observation counts and the selected-checkpoint evaluation pass. It reconstructs
each batch's supervised losses, including the mean across binding/excision panel
axes, and combines them with recorded task weights and regularization terms.
Task summaries average batches where the task is present; overall loss averages
all batches. Missing/corrupt required predictions fail closure while preserving
the checkpoint and an error artifact. Empty splits still produce header-only
prediction/metric files. Summaries enumerate declared output/column support,
including zero support, absent optional predictions and the ms/elution alias.

Split-support schema v4 adds `row_targets` and `row_split_support.csv` alongside
`mil_targets` and `mil_split_support.csv`, all using the shared target resolvers.
These include derived categorical targets, vector components, selected columns,
response/qualifier counts and source counts. The legacy `targets` table still
counts row masks; these tables are not additive endpoint counts.

The executable declaration in `training/output_contract.py` canonicalizes
published aliases, shapes, units, columns and the existing row/MIL objectives.
`output_coverage.json` and `.csv` count effective observations per split and
canonical column, including zeros. The JSON also separates original source
identities, fallback identities, distinct peptides/alleles, context, response
balance, censoring and raw source/evidence families. Repeated alias objectives
remain visible in the objective ledger without multiplying unique evidence.
Class-split elution bags retain their separate effective observations while
sharing their original source identity.

Target provenance travels as host metadata. Typed assay records, generated
controls, organism-derived foreignness and bulk-MS proxies remain distinct.
Bulk observed products do not establish isolated cleavage measurements;
fractionation-depth labels are ordinal detectability proxies. Wrong-enzyme
controls carry an explicit generation kind. Passing protein sequences to the
current bulk loader does not produce unobserved detectability candidates.

A prospective `--supported-output-manifest` declares a subset of canonical
columns, required splits, admissible evidence and explicit count requirements.
It is frozen before data loading and checked against model configuration and
curated-input/evidence fingerprints before fitting. Unknown origins, aliases as
independent claims and missing support cannot satisfy the gate. Passing means
the declared requirements were met; it establishes neither prediction quality
nor adequacy of thresholds chosen by the experimenter. Exploratory runs without
a manifest report all outputs as undeclared. See the
[training guide](training_spec.md#output-coverage-and-prospective-claims) for use.

`output_updates.json` separately records column-label exposures, output
derivatives, initialization/trainability, final optimizer gradients and actual
dedicated parameter-row changes. Fixed-rule replacements and outputs without a
dedicated parameter row are explicit. Weight decay, shared gradients and indirect
updates cannot establish direct supervision. Tracking is automatic for declared
runs and otherwise opt-in with `--track-output-updates`.

The real-source census/update evidence remains #48/#50; the fresh fitted and
held-out baseline remains #53. Infrastructure tests do not close those claims.

Row counts in historical experiments do not describe today's default corpus.
Merged TSV and Hitlist are explicit mutually exclusive primary source choices;
optional bulk-MS is an addition. Loading all assay families, having balanced
support, and having scientifically validated targets are separate claims.

## 6. Remaining drift and acceptance gates

This first repair addresses encoder conditioning, CD4 readout wiring, fixed
T-cell panels/response losses, binding-panel censoring and boundary forwarding.
It does not close #46.

Remaining work includes:

1. One typed role-specific schema for presenting versus repertoire MHC, all five
   species roles, concurrent interventions/cytokines, provenance and missingness.
2. No role copying in source normalization or auxiliary targets. Full API/MIL/
   tiling parity for biological context and a shared MHC preparation path.
3. Repertoire recognition using presented pMHC plus selection context; optional
   receptor-specific matching only with appropriate data and explicit semantics.
4. Groove-correct augmentation and priors: current `no_mhc_beta` deletes a groove
   segment, **not beta2m**. Do not interpret it as an assembly KO experiment.
5. Family-specific quantitative routing, kinetic censoring and joint assay
   configurations; no unsupported claims from panel vocabulary entries.
6. Reconcile processing versus excision and class-I-only excision-to-presentation
   contribution; justify proxy supervision of biological latents.
7. Complete source/observation lineage and funnel/support gates for declared
   datasets; separate real from synthetic negatives and validation metrics.
8. Bulk candidate negatives, abundance/acquisition modeling and validation of
   detectability; no identifiability claim from a running code path.

Acceptance requires invariance tests with nonzero context parameters, distinct
APC/repertoire and cross-species examples, observed controls versus missingness,
per-track gradients, loader-to-held-out traceability and declared split support.
Scientific experiments must follow [the experiment workflow](https://github.com/pirl-unc/presto/blob/main/experiments/EXPERIMENT_WORKFLOW.md);
a unit training smoke test is not validation of the biology.

## 7. Checkpoints and contract changes

Removing the encoder conditioning parameters changes model state. New checkpoints
round-trip with their saved architecture configuration. Old checkpoints containing
`species_cond_embed`/`chain_completeness_embed` are incompatible and should be
used with their original code or retrained; no silent key dropping, parameter
migration or checkpoint fallback is introduced.
