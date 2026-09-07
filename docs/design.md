# Architecture guide

The single desired input/output design is [the model I/O contract](model_io_contract.md).
This guide describes the current architecture, not a competing specification.
[Issue #46](https://github.com/pirl-unc/presto/issues/46) tracks remaining drift.
Old full-chain, beta2m-token, peptide-only-recognition and context-conditioned
assay pseudocode has been superseded.

## Purpose and boundaries

Presto connects peptide processing, MHC binding, presentation, repertoire
recognition and immunogenicity, with output-side models of their observations.
Sequence-only encoding does not mean context-free biological prediction.

The intended design keeps presenting APC MHC separate from the MHC that shaped
the queried repertoire. APC, molecular MHC, individual TCR, repertoire-system and
antigen species are independent roles. APC interventions and cytokines are allowed
downstream biological context. They are not residue-encoder features or generic
assay IDs. See the contract for the complete input and output inventories.

## Current computation

The canonical implementation is [Presto](https://github.com/pirl-unc/presto/blob/main/models/presto.py). The default latent
topology is `expanded`; `collapsed` remains an explicitly selectable older
architecture, not an automatic fallback. Architecture configuration is saved in
checkpoints.

1. Normalize sequence inputs. `mhcseqs` resolves alleles and extracts groove
   segments. Class I uses alpha1 and alpha2 from the same heavy chain; class II
   uses alpha1 and beta1 from paired chains. Neither class-I input is beta2m.
2. Encode `Nflank | peptide | Cflank | mhc_a | mhc_b` with shared amino-acid
   embeddings, segment embeddings and segment-specific positions.
   Base self-attention is blocked between segments. Species and global
   completeness embeddings are absent.
3. Pool early sequence representations for auxiliary MHC identity and antigen
   origin tasks. Class/species may be inferred or explicitly overridden
   downstream. A host-species value does not override molecular MHC species.
4. Integrate sequences in task-scoped latent queries. Groove cross-attention and
   peptide core/PFR summaries support binding. Processing receives junction
   features and coarse host/cellular context downstream.
5. Compute class-specific processing and presentation, shared molecular
   affinity/stability, separate CD8/CD4 recognition and immunogenicity, and
   observation heads. All fixed assay panels are evaluated independently of
   observed assay descriptors.
6. Aggregate supported multi-MHC observations at the bag level using Noisy-OR in
   the training/inference aggregation code. The core forward is per molecule;
   peptide encoding reuse across a bag is not a current guarantee.

### Expanded latent graph

The executable access/dependency tables are `Presto.EXPANDED_LATENT_SEGMENTS`
and `Presto.EXPANDED_LATENT_DEPS`. They are tested in
[tests/test_latent_topology.py](https://github.com/pirl-unc/presto/blob/main/tests/test_latent_topology.py).

| Current component | Sequence/context access | Important limitation |
|---|---|---|
| `species_of_origin` | Peptide | Source species is not definitive foreignness |
| `foreignness` | Origin path | Does not implement the full repertoire-relative target |
| `processing_class1/class2` | Flanks and permitted peptide features; downstream sample context | Also receives MHC-derived class/species context today; not absolutely MHC-independent |
| `binding_affinity/stability` | Peptide and presenting groove pair; molecular context | No host/flank-presence shortcut through encoder |
| `presentation_class1/class2` | Upstream processing/binding dependencies | Proxy supervision and excision coupling need reconciliation |
| `recognition_cd8/cd4` | Current peptide/foreignness path | Missing presented-pMHC and selection-system context required by target |
| `immunogenicity_cd8/cd4` | Lineage-specific upstream representations and recognition | Distinct names do not prove independently identified biology |
| `ms_detectability` | Peptide | Optional bulk-MS labels are proxies |

The expanded CD4 head reads the CD4 recognition latent, and CD8 reads CD8.
The collapsed topology shares a recognition latent by design.

The output named `apc_cell_type_context_vec` is currently derived from MHC
class/species/compatibility. It must not be interpreted as a measured APC
phenotype. Measured coarse lineage/origin/disease enters a different processing
context token.

### Core and PFR

The model enumerates candidate peptide core windows and computes candidate masks,
priors, scores and posterior weights. It exposes start/length, membership and
PFR summaries. Positional strategies and candidate-window settings are explicit
model configuration, not guarantees that every candidate has direct labels.

The model uses positional frames appropriate to each segment: peptide endpoints
and/or fractional position, junction-relative flank positions, and groove
positions. See constructor options and checkpoint configuration for selected
variants. Do not treat old approximate parameter counts as current measurements.

### Processing, excision and observation

`ExcisionHead` exposes junction, missed-cleavage and length scores. Machinery and
capture/digestion routing act downstream. APM/stimulus profiles can condition the
observed biological state and provide counterfactual output tracks.

An explicit in-vivo excision contribution currently feeds **class-I**
presentation only. There are still both processing latents and an excision
score; their reconciliation is open work, not two independently validated models
of the same event. Elution combines presentation and detectability.
`ms_logit` is an elution alias, not a separate abundance/detection measurement.

### Output-side assay structure

Binding quantities and fixed binding descriptor panels are emitted from the same
molecular representations. T-cell panels sweep one axis at a time with all
other axes at an unknown-reference baseline. They are not a registry of every
attested joint assay configuration. Metadata selects the supervised column
against the measured response and never changes what a fixed column computes.

The reference T-cell scalar is not a mathematical marginal over assay setups.
APC exposure and measured T-cell cytokine secretion have distinct meanings,
even when source annotations use similar words.

### Missingness and boundaries

Unknown optional segments use `<MISSING>`. Flank-window construction distinguishes
known protein boundaries from unknown context via explicit flags. Current
boundary padding reuses `X`; unknown window padding uses `?`. This is an
implementation limitation, not a claim that ambiguous residues and boundaries
are biologically identical. Missing MHC/sequence is not evidence of nonbinding.

## Training and serving contract

[PrestoBatch.model_inputs](https://github.com/pirl-unc/presto/blob/main/data/collate.py) is shared by canonical row training
and held-out scoring. MIL carries per-instance context and terminus flags through
capping and contrastive construction. Presentation inference accepts boundary
flags; tiling derives them from observed protein coordinates.

[Training notes](training_spec.md) explain source selection and objectives.
[Assay learning notes](assay_learning_scheme.md) separate direct labels from
proxy supervision. Neither a class name nor a passing training step proves
identifiability, calibration or biological validity.

Full role-specific context parity across loading, MIL, tiling and public
prediction is pending. Receptor-specific matching is
[future work](tcr_spec.md); canonical receptor-evidence outputs are pMHC-only.

## Verification and checkpoint policy

Behavioral tests cover encoder/host isolation, fixed-panel invariance, correct
CD4 wiring, selected-output response gradients, qualifier-aware binding panels,
terminus forwarding and checkpoint round trips. Broader acceptance requirements
are listed in the I/O contract and #46.

Old checkpoints with removed encoder conditioning parameters must use the
original code or be retrained. There is no silent state-dict migration.
Historical experiment directories preserve their own code/data/configuration
contract and do not override today's design.
