# Modular directly supervised model

The optional `modular` model implements the software foundation of
[issue #130](https://github.com/pirl-unc/presto/issues/130). It has no qualified
biological checkpoint yet. Passing software tests does not establish improved
IC50 accuracy, useful TCR specificity or separately calibrated MS detectability.
The default model and its checkpoint format remain unchanged.

## Auxiliary annotations are not the affinity model

The earlier shorthand "identity plus IC50" describes one diagnostic subset:
sequence-annotation classification and measured IC50 regression share an encoder.
"Identity" means labels such as molecular species and MHC class, not observation
IDs, allele-ID embeddings or a peptide match lookup. It does not describe the
complete processing, MS and receptor system requested in issue #130.

The assay path retains the rich sequence vectors. For affinity, each peptide/core
and each MHC groove contributes a 128-dimensional vector to a learned fusion,
followed by register aggregation and an assay-specific scalar readout. Supplied
or inferred class/species values provide additional inputs and class routing;
the annotation predictions do not replace the sequence vectors. IC50 targets
are standardized log10(nM), with one-sided losses for censored bounds.

Auxiliary-only pretraining is supported, as is assay fitting with auxiliary
replay: mean assay loss plus `identity_weight` times mean auxiliary-label loss
(default weight 0.1). The weight is a configurable software default, not a
validated scientific choice. Annotation labels train their classifiers and the
shared encoder; assay labels train the encoder, fusion, active register pooling
and assay readouts. Detaching categorical probabilities blocks the direct assay
gradient into classifier weights. It does not freeze classifier behavior: assay
updates to the shared encoder can still change its annotation predictions.

Auxiliary tasks are optional. An affinity diagnostic can use supplied class and
molecular species without allocating annotation heads. Learning these annotations
does not establish useful binding representations, transfer, or a remedy for
constant predictions. Those claims need matched affinity comparisons and retained
annotation evaluation. No such real-data comparison exists for this model yet.

A registered assay-only capacity check has now fitted 32 real IC50 observations
(16 per class, with explicit class-II partners) without any auxiliary task.
At 100 updates, class-I/class-II training RMSE was 0.0433/0.0171 log10(nM), with
prediction/target SD ratios close to one and all allocated parameters updated.
This establishes memorization on those inputs, not held-out accuracy, correct
register localization or auxiliary benefit. The
[complete experiment record](https://github.com/pirl-unc/presto/tree/codex/modular-presto-model-20261020/experiments/2026-10-21_0353_codex_modular-memorization-control)
preserves source pins, predictions, fixed gate and fresh-process verification.

A separate, registered peptide-source pretraining comparison has also completed.
All 7,000 selected reference memberships were audited against the pinned protein
reference. The auxiliary-only network beat the prevalence baseline, but its
prespecified validation comparison against amino-acid composition failed;
composition had lower test cross entropy (0.300864 versus 0.302032). Training
coverage and fresh-process checkpoint replay passed. This is no evidence of
improved affinity prediction. The
[reference-pretraining experiment](https://github.com/pirl-unc/presto/tree/codex/modular-presto-model-20261020/experiments/2026-10-21_0512_codex_weirdo-reference-pretraining)
preserves the negative result, baselines and every held-out prediction.

A separate MHC reference audit now supplies 13,218 source-traceable single-groove
species annotations; it has not trained a model. Its exact-domain/original-chain
partition has a large concentration: one component contains 91.95% of human test
annotations. The [admission evidence](https://github.com/pirl-unc/presto/tree/codex/modular-presto-model-20261020/experiments/2026-10-21_0618_codex_mhc-reference-admission)
preserves this limitation. Neither passing source checks nor identifying species
establishes a useful affinity representation.

## Shape and information flow

One shared sequence encoder uses 20 canonical residues plus padding, width 128,
two Transformer layers, four attention heads, feed-forward width 512, sinusoidal
positions and masked mean pooling. It encodes each sequence independently.

| Requested outcome | Required sequences | Additional admitted inputs |
| --- | --- | --- |
| Per-chain molecular species | Available MHC groove or TCR chain, alone or paired | Actual species label for each present input |
| MHC class/pairing; TCR pairing/lineage | Complete MHC groove pair or paired TCR chains | Actual class, pairing or phenotyped lineage labels |
| Peptide source membership | Peptide alone | Reference category labels; categories can overlap |
| Purified affinity/stability | Peptide, MHC groove 1, MHC groove 2 | Molecular class/species, supplied or inferred |
| Cellular response/elution | Same pMHC inputs | APC/immune species, independent component states, antigen delivery, source flanks/boundaries |
| Bulk MS observation | Peptide; source flanks when available | Digest and prospective acquisition context |
| Independent MS anchor | Peptide | Prospective acquisition context |
| Digest anchor | Peptide; source flanks when available | Digest identity |
| Paired receptor binding | Peptide, two MHC grooves, full TCR alpha and beta | Receptor species/repertoire context |
| Paired receptor functional response | Same five sequences | Also admitted cellular context and actual coreceptor state |

Class I groove segments are alpha1 and alpha2 of the heavy chain. Class II
uses alpha1 of the alpha chain and beta1 of the beta chain. Full-MHC preparation
uses the existing strict `mhcseqs` extraction; no partner is guessed. Allele IDs,
study/sample IDs and cell-line names are provenance, never learned embeddings.

### Common core across MHC classes and assays

Both MHC classes use the same `SequenceEncoder`, `pmhc_fusion`, cellular
`context_fusion`, trainer, observation schema, checkpoint and inference API.
Class-specific candidate construction retains whole peptides for class I and
nine-residue registers with flanking features for class II. This routing does
not allocate another encoder or class-specific training pipeline. Ordinal
context scales, quantitative domains, censoring and coverage checks apply to
either class. The MS feature block is also shared with bulk and component assays;
domain-specific scalar readouts preserve distinct measurement scales.

New source adapters should translate evidence into these common inputs. Source
parsing, source-relative scale definitions and experiment controls belong in
curation/configuration; study-specific predictors or trainer branches do not.
Different biological routes may retain different candidate construction and
readouts. Cellular context affects cellular outcomes, while purified binding and
the MS detector remain invariant to that context.

Software regressions exercise both classes updating the same parameters, one
mixed-class quantity fit including bulk observations, and replay from one
checkpoint. They establish reuse and routing, not biological transfer. MHC-II
development has not yet demonstrated an MHC-I accuracy gain. Such a claim needs
a matched predictive comparison with retained MHC-I evaluation and controls for
negative transfer; shared parameters alone are insufficient.

### Explicit reference assumptions

`HumanMHCCatalog.resolve(reported_restriction, mhc_class)` remains strict by
default: missing or ambiguous typing is deferred. When a caller intentionally
models an unverified reference, pass an explicit `MHCReferenceAssumption`:

```python
from presto.data.modular_hitlist import MHCReferenceAssumption

pair = catalog.resolve(
    "HLA-DRB1*04:01",  # Actual source-reported typing; alpha remains unknown.
    "II",
    reference_assumption=MHCReferenceAssumption(
        restriction="HLA-DRA*01:01/DRB1*04:01",
        reason="Explicit reference alpha; source alpha identity unresolved",
        reference="curation-record:reference-choice-v1",
    ),
)
```

The same API permits a declared class-I reference when reported identity is
`None`. The catalog's existing sequence, class, chain and alias checks still
apply. Every reported allele must be retained; broader or conflicting source
typing is not silently reinterpreted. Reason and reference are required, and
there is no automatic reference allele. `MHCInput.provenance` records the actual
`reported_restriction`, selected `catalog_alleles`, `assumed_alleles` and complete
`reference_assumption`. These survive observation serialization, evaluation
records and prediction diagnostics after checkpoint reload; they never become
learned IDs or source-verified typing. Source evidence stays unchanged, and
predictive conclusions remain conditional on the assumption.

Auxiliary observations may contain a single MHC groove or TCR chain. Missing
partners are not encoded or fabricated; the existing shared species readout
supervises the present input. Class/pairing/lineage outputs require their full
input pairs, and targets naming absent inputs fail validation. Every assay still
requires its complete resolved MHC and, where applicable, paired TCR sequences.
Single-chain class metadata does not train the paired MHC-class head. Auxiliary
partitions reject a repeated chain even if its partner or input slot changes;
this exact-sequence guard does not establish homology independence or a qualified
species inference domain.

Optional identity heads predict MHC class, per-chain molecular species, TCR
species and observed CD4/CD8 association. These use actual independent identity
labels. Assay losses cannot train these heads through the resolver: inferred
probabilities are detached, while the encoder receives both identity and assay
supervision through its sequence representations.

Resolution keeps raw predictions, supplied annotations, overrides and effective
values separately. Explicit override wins over supplied metadata, then supported
prediction. A declared convenience policy can default immune species from a
consistent MHC pair and APC/repertoire species from immune species. These are
marked assumptions. Strict mode disables those defaults; antigen origin is never
copied from the host. An uncertain or mixed pair cannot establish common species.
A TCR lineage prediction does not establish actual coreceptor expression.

Additional early tasks are `peptide_source`, `mhc_pairing` and `tcr_pairing`.
Peptide-source prediction uses a frozen `source_categories` vocabulary and one
independent sigmoid per category. Identity-label keys such as
`peptide_source:human` take `present`/`absent`; omitted category labels are missing,
not negative. A two-logit `[0, logit]` representation uses the same categorical CE
loss and evaluator, exactly matching binary cross entropy. Each category retains
its own supervision status; training one category cannot authorize the others.

Chain-pairing heads consume the two corresponding sequence vectors and predict
`matched`/`mismatched` for the declared source pairing task. Preparation requires
explicit pairing evidence. An observed pair supplies a matched label; it cannot
supply a measured mismatch. Generated pairing controls require a policy and parent
lineage and do not establish biochemical incompatibility.

`Context.auxiliary_values` and `auxiliary_overrides` preserve independent supplied
and override values. Optional `auxiliary_context` slots declare downstream use:
MHC pairing enters pMHC fusion, TCR pairing enters receptor fusion, and peptide-
source/lineage priors enter applicable cellular outputs. Actual coreceptor state
remains separate. Source/lineage overrides leave purified binding and fixed
TCR–pMHC compatibility unchanged; none of these metadata enter the shared MS
detector. Probabilities are detached from assay gradients; rich shared encoder
representations remain trainable by all admitted objectives.

The assay representation uses one linear/GELU fusion of sequence vectors, class,
molecular species and 43 fixed peptide-flank features:

- Class I contributes one full-peptide candidate.
- Class II contributes every contiguous nine-residue core, with positions reset
  in each core and up to three neighboring residues on each side summarized by
  composition. Full flank lengths and total peptide length are retained.
- An endpoint consumes learned register-weighted features, normalized separately
  within each class, then mixed by effective class probability.
- TCR sequence vectors and the full peptide enter the optional context fusion
  **before** register pooling. Binding of a fixed pMHC is unchanged when TCR inputs
  are supplied to a separate receptor request.

The optional cellular/receptor context fusion is one linear/GELU layer. Machinery
components retain independent `reference`, `intact`, `reduced`, `lost` or `unknown`
states. Explicit component overrides beat sample annotations and declared species
reference profiles. Profiles are priors; an empty perturbation list is unknown.
Optional measured activities require declared units. Purified binding never consumes
APC state. Paired receptor binding to supplied pMHC also excludes APC processing.

Optional `cell_types` and `cytokines` vocabularies add APC context to this same
fusion. `Context.cell_type_override` takes precedence over the supplied cell type.
Each cytokine is an independent `Exposure` with `present`, `absent` or `unknown`
state and optional dose/unit and duration in hours. `cytokine_overrides` replace
individual exposures, retaining the supplied values in diagnostics. Missing
cytokine metadata is unknown; it never creates an untreated-control label.

Cell type uses one-hot columns plus unknown, and each cytokine has three state
columns. `cytokine_dose_units` and `cytokine_duration_features` explicitly allocate
optional `log1p(value)` and known-mask columns. Doses must match their declared
units; no potency or unit conversion is guessed. These inputs are masked from
purified affinity/stability, fixed receptor binding, digestion and shared MS
response. Unallocated context families are listed as ignored; unsupported values
within an allocated family are rejected on cellular requests.

`processing_pathways` and `interventions` allocate separate named cellular
assertions. Each uses a `ContextAssertion` with `present`, `absent` or `unknown`
state and retained evidence. Multiple pathways or interventions can be present
at once. `pathway_overrides` and `intervention_overrides` replace individual
claims, including replacing a supplied claim with explicit unknown. Diagnostics
retain the supplied claim, override and source of the effective value.

These fields distinguish, for example, a reported added-DM intervention from
measured DM activity or loss, and a source-supported processing-pathway claim
from antigen delivery. The model supplies no pathway default from MHC class,
cell identity, species or gene expression. Evidence text is provenance, not a
predictive input. The independent categorical inputs enter cellular context
fusion; they do not assert independent physical yields or a causal mechanism.
They are masked from purified binding, fixed molecular receptor binding,
laboratory digestion and the separate MS-response component.

`ordinal_scales` adds source-relative ordered cellular observations, such as a
reported low/medium/high DO:DM ratio. Each `OrdinalScale` declares ordered labels
and an explicit comparison or calibration reference. A supplied `OrdinalValue`
must match that reference; equally named levels from unrelated scales are not
silently combined. The model uses separate columns for each level and unknown,
retaining the order without assuming numerical spacing or a monotonic response.

```python
from presto.data.modular_ordinal import OrdinalScale, OrdinalValue

scale = OrdinalScale(
    ("low", "medium", "high"),
    "PXD024392:T2-series:reported-DO-DM-ratio:v1",
)
# Configure ordinal_scales={"do_dm_ratio": scale} on a cellular model.
# Supply this in Context.ordinal_values; evidence stays out of tensors.
reported = OrdinalValue("medium", scale.reference, {"source_label": "DR4DMDO++"})
```

`Context.ordinal_overrides` takes precedence, including `OrdinalValue()` to
replace a supplied level with explicit unknown. Diagnostics retain supplied,
override and effective values. No level is inferred from cell type, MHC class or
species. Scale references and evidence are provenance/validation fields, never
learned identifiers. These inputs affect cellular outcomes only; purified
binding, fixed-pMHC receptor binding, digestion and independent MS response stay
invariant. Preflight requires all declared levels in training cellular rows;
held-out levels and unknowns do not establish training support. Strict prediction
also rejects levels whose input weights have not been trained.

For the cited DO:DM source, this field represents the graded series only. DOKO's
reported reduction relative to its parent does not identify one of these three
levels, and DO absence is separate from an unknown or lowest ratio. DM also varies
across that series. This representation supplies neither absolute activities nor
an isolated DO effect. Resolving the paired MHC requires either reported typing
or an explicitly documented reference assumption through the common catalog
adapter; predictive qualification remains separate. Empty ordinal fields preserve
existing model shapes and historical semantic fingerprints.

`machinery_references` contains typed `MachineryProfile` records with an ID,
species, optional cell type, component priors, reference and version. Species-wide
defaults are followed by a matching cell-specific profile, sample annotations
and explicit component overrides, in that order. Even an explicit unknown state
overrides a default. Duplicate profile scopes are rejected. Resolved priors retain
their provenance and are marked assumptions. No curated profile catalog is shipped
yet, and declaring a source reference does not qualify its biological evidence.
The older `machinery_profiles` name lists remain unverified reference priors.

Preflight requires training examples for each configured cell category, actual
present/absent cytokine contrasts and varied values for declared numeric channels.
Every allocated trainable pathway/intervention likewise requires present and
absent training examples. Unknown and held-out examples cannot supply that
contrast. This check does not establish matched biological arms; source-specific
qualification and predictive ablations are still required. A uniformly annotated
intervention cannot establish an intervention effect.
Profile defaults do not substitute for sample-attested machinery contrasts. Input
column support is saved in checkpoints; unseen states remain unsupported at
inference. These checks establish input support, not causal identification or
predictive value for perturbations.

## MS observation

The native Hitlist positive-MS adapter (`data/modular_hitlist_ms.py`) retains
complete class-II candidate pairs, original observations and source-scoped
intervention/cytokine claims. It preserves unknown chemistry, missing partners
and pooled conditions as explicit dispositions. Its admitted rows remain
unpartitioned `predict` evidence until a separate contributor-aware cohort
contract supplies training membership and any declared contrast sampling.
Positive ligand lists alone do not provide measured negatives or calibrated
detection probabilities.

The proposed [processing distribution and MS design](processing_model_design.md)
extends this observation model with source-coordinate spans and machinery-conditioned
product pools, including nonbinders. It separates the proposed abundance/sampling
contract from the statistical component scores implemented below.

The same detector feature block and scalar readout are used in both routes:

```text
bulk logit    = digest score + shared MS score + bulk observation offset
elution logit = cellular score + shared MS score + elution observation offset
```

Digest features consume peptide, optional source context, enzyme and declared
prospective preparation covariates (`preparation_features`/`preparation_units`,
with row values in `Context.preparation`/`preparation_units`). Each numeric field
has a missingness indicator and needs varied training observations. Preparation
does not enter the detector or cellular pMHC path. Shared MS
features consume peptide and declared prospective acquisition covariates, each
with a missingness indicator. Numeric acquisition/observation fields require matching
unit or prepared-scale declarations in both configuration and input context. Enzyme, MHC, APC, species and TCR metadata cannot
enter this detector directly. An endpoint offset contains one intercept and
optional measured observation covariates, such as amount with its declared scale.

One elution bag has one loss. Identical MHC candidates are deduplicated; distinct
candidate cellular scores use log-mean-exp before the detector is added. This is
an explicit statistical aggregator, not an independence or occupancy assumption.
One observed elution does not become several measured positive peptide–MHC pairs.

Without independent anchor labels, component scores are indirectly trained
diagnostics. The combined outcome can learn within-search observability without
identifying digestion and detection separately. `capabilities()` reports those
statuses independently. There is no automatic calibration or transfer claim.

Quantitative observations use `bulk_quantity:<domain>` and
`elution_quantity:<domain>`. Independent component measurements use
`ms_quantity:<domain>`, `digest_quantity:<domain>` or
`cellular_quantity:<domain>`. The last measures pMHC availability; it is not a
generic cleavage/transport assay. Those biochemical-specific adapters remain
unfinished. The same sequence and feature blocks support binary and quantitative
subsets, with no unused binary readouts allocated to quantitative-only models.

Each quantitative domain has an explicit `QuantitativeDomain` contract: measured
quantity, canonical unit, `identity`/`log2`/`log10` target transform, normalization
and source/reference. `ModularConfig.quantity_domains` maps domain names to these
contracts; each `Observation.quantity_contract` must match exactly. Raw values
retain their unit, qualifier and `value_encoding` (`linear`, `log2` or `log10`).
For example, an already-log2 intensity can be converted directly to a log10
target without exponentiating. No clipping, inferred unit conversion or raw-
versus-normalized fallback is applied. Source inspection must establish whether
two datasets really share a scale; matching caller-provided strings is not proof.

For each domain the model allocates a bias-free scalar readout per reachable
component. Bulk and elution share that domain's MS readout, and component anchors
use those same weights. Different domains share the feature blocks but have
separate readouts. Domain names select supervised outputs, not predictive dataset
embeddings. One offset belongs to each endpoint; declared observation covariates
can condition it. These are statistical additive models, with no automatic
cross-domain calibration or independent component interpretation.

Quantitative outputs are in transformed measurement units. Their squared/censored
residuals are divided by the training-only target scale inside the loss, rather
than changing the shared components' units for each endpoint. Public predictions
return transformed and inverse-transformed values and the measurement contract.
The quantity bag aggregator is a statistical mean in linear space, calculated in
the declared log base when appropriate; duplicate molecules still do not create
extra votes. Existing affinity/stability heads retain standardized internal outputs.

Canonical quantity admission requires an attested measured outcome. Component
anchor endpoints additionally require `component_measurement` provenance naming
the component, `independent_component_measurement` basis and a source reference.
`capabilities().quantity_components` distinguishes direct anchor supervision from
indirect training for each domain. These declarations and software checks do not
establish identification of digestion, availability or MS bias: source curation,
anchors, held-out predictive evaluation and transfer evidence remain required.

## Observation and target contract

`presto.data.modular.Observation` is the single train/inference input. JSONL rows
include the explicit `presto.modular.observation.v1` schema, stable observation and
source IDs, endpoint, sequences, raw value/unit/qualifier, frozen split/group ID,
context and provenance. Identity rows carry independent `identity_labels`; their
sequences depend on the task and they never carry a scalar assay value. Peptide-
source rows need only the peptide. A candidate MHC bag belongs to one observation.

Supported direct regressions are `ic50`, `kd`, `ec50`, `kd_via_ic50`, `kd_via_ec50`,
`koff`, `half_life` and `tm`. Proxy measurements remain separate. Concentration,
rate and duration targets use log10 physical values; Tm uses Celsius. Scaling uses
only exact training values and requires two distinct values per regression endpoint.
There is no legacy 50,000 nM clipping. Checkpoints preserve means, scales and floors.

Exact measurements use squared standardized error. An upper bound penalizes only
predictions above the bound; a lower bound penalizes only predictions below it.
Binary endpoints require exact 0/1 outcomes and use one BCE at the final output.
Missing labels never become zeros. Quantitative MS intensities are not silently
accepted as binary labels.

Source adapters in `presto.data.modular_sources` provide:

- Raw binding, kinetic and stability conversion before historical clipping.
- Verified search-scoped bulk observation conversion. Raw positives without a
  denominator and generated wrong-enzyme controls are rejected by this adapter.
- Strict groove preparation and curated normalized observation admission.
- Paired-TCR admission requiring source-attested full paired chains and a measured
  outcome. The normalized contract is not a claim about the unavailable TCRvdb CSV.

`presto.data.modular_auxiliary.weirdo_observations` reads Weirdo's prepared
`splits.npz`/`splits.json` format. It requires hashes, the reference sequence hash,
explicit per-split sample limits and a seed. Sampling is uniform without replacement
and independent of labels; original folds, family buckets, row indices and overlapping
membership labels are retained. Byte/schema verification does not certify the
upstream biological independence or create an exact species-of-origin label.
Older archives without family buckets are rejected by default. Explicit
`allow_missing_groups=True` preserves their absence and leaves the rows
unadmittable until grouping is established. `audit_weirdo_reference` independently
reconstructs selected memberships, all matching proteins/positions and source
folds from a pinned protein reference. Its explicit missing-group reconstruction
records inferred buckets separately from supplied ones; any membership or fold
discrepancy fails admission. Connected source-family groups reproduce mnemonic
prefix grouping, which is not a guarantee of homology independence.

`presto.data.modular_preparation.prepare_observations` deduplicates original
measured outcomes across exports, defers conflicting copies, preserves independent
replicates and writes per-row dispositions and a census. Source adapters still
need to establish original measurement lineage and cell-line/parent/clone
attribution. The training index verifies hashes, schema, unique original outcomes
and group/peptide-disjoint splits; it does not certify donor or receptor-cluster
independence from IDs alone.
It indexes file offsets and batches sequences/registers instead of retaining every
candidate representation for the entire corpus. Inputs changed after admission fail.

## Training through the canonical CLI

Both commands use the same dispatch:

```bash
presto train unified --model-kind modular \
  --observations-manifest /absolute/path/manifest.json --modular-preflight

python -m presto.scripts.train_iedb --model-kind modular \
  --observations-manifest /absolute/path/manifest.json --modular-preflight
```

Remove `--modular-preflight` to run a registered fit. Model and optimization settings
are in the manifest. Explicit legacy CLI training options are rejected in modular
mode, rather than silently ignored. CPU is the default; `--device` is explicit.
A new timestamped directory under `experiments/` is mandatory for fitting.

The manifest shape is:

```json
{
  "schema": "presto.modular.manifest.v1",
  "model": {
    "endpoints": ["ic50"],
    "identity_tasks": ["mhc_class", "mhc_species"],
    "mhc_classes": ["I", "II"],
    "species": ["human", "mouse"],
    "dropout": 0.1
  },
  "training": {
    "seed": 42,
    "batch_size": 64,
    "max_epochs": 100,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "identity_weight": 0.1,
    "test_policy": "evaluate_selected_once"
  },
  "sources": [
    {
      "path": "qualified-observations.jsonl",
      "sha256": "REPLACE_WITH_ACTUAL_FILE_HASH",
      "source": "REPLACE_WITH_EXACT_ROW_SOURCE",
      "population": "REPLACE_WITH_QUALIFIED_POPULATION",
      "label_policy": "REPLACE_WITH_PINNED_CURATION_POLICY",
      "split_policy": "REPLACE_WITH_FROZEN_INDEPENDENCE_POLICY"
    }
  ]
}
```

The source files must include real identity supervision for those allocated tasks.
An assay-only subset may instead omit identity tasks and supply class/species.
Identity-only and receptor-auxiliary-only pretraining allocate no assay model.
Bulk-only runs allocate no MHC or APC modules. A class-II subset containing only
nine-residue training peptides omits register attention entirely; longer-peptide
requests then fail explicitly. Declared MHC branches without assay training support
are rejected. Context vocabularies and feature slots are declared in `ModularConfig`.

Training makes deterministic shuffled passes over assay observations and uses
observation-weighted mean losses. A separate identity batch contributes 0.1 times
mean identity CE by default. Replay exposure counts are reported separately from
unique labels. A hash-pinned `pretrained` entry with `path` and `sha256` initializes
compatible modules. `allow_task_removal: true` permits explicit task subsetting;
removed heads/domains/scalers/support entries are omitted. Task-driven changes
to receptor/cellular feature width copy retained columns by their canonical names.
New columns and heads remain unsupported until trained. Retained feature meanings,
units, categorical-output vocabularies and class branches cannot silently change.
Retained regression endpoints use their checkpoint's original scalers; only new
endpoints fit new training-only scalers.

`pretrained.freeze` lists output tasks or component targets. For example:

```json
{
  "pretrained": {
    "path": "prior.pt",
    "sha256": "ACTUAL_CHECKPOINT_SHA256",
    "allow_task_removal": true,
    "freeze": ["component:ms_response"]
  }
}
```

Freezing that detector function freezes its shared encoder, MS feature block and
readout. A quantitative component uses a domain-qualified name such as
`component:ms_response:declared_domain`. Freezing a complete output such as `ic50`
also includes its fusion, register pooling, applicable annotation dependencies
and scalar readout. All frozen dependencies need inherited gradient/update support.
Encoder dropout stays disabled while sibling modules train. This restricts what
the other tasks can learn through their shared encoder; head-only freezing would
not preserve the function. Checkpoints store and verify the dependency declarations
and parameter sets. `pretrained_frozen` is training support, not scientific validation.

Fully frozen observations remain available for evaluation but are excluded from
optimization sampling and checkpoint selection. Retained frozen outputs need no
new labels; a frozen annotation function needs no replay. Unfrozen outputs still
need their own supervision, and shared-parameter updates invalidate dependent
validation status. Optimizers contain only trainable parameters. Coverage closure
requires updates for trainable parameters and inherited support for frozen ones;
unseen features cannot be activated in a frozen path. Arbitrary feature/vocabulary
migrations remain unimplemented.

An interrupted modular fit resumes with the same frozen manifest and run directory:

```bash
python -m presto.cli.main train unified --model-kind modular \
  --observations-manifest experiments/RUN/reproduce/source-manifest.json \
  --run-dir experiments/RUN --modular-resume --device cpu
```

The original hash-pinned manifest is also accepted. `--modular-resume` cannot be
combined with preflight or legacy training. It reuses the registered experiment;
an ordinary launch still refuses to overwrite a prior run. Resume attempts and
failures are recorded separately without replacing the original launch recipe.

`resume.pt` preserves the current model and optimizer, ordered optimizer parameter
names, support/freeze/scaler evidence, independent assay and replay RNGs and
cursors, global random states, partial-epoch accounting, coverage, patience and
the selected checkpoint. Snapshots are atomic at initialization, every
`checkpoint_every_steps` updates (default 100), each completed epoch and before
final evaluation. Uncommitted updates after the last snapshot are replayed.
An exclusive process lock prevents concurrent trainers in one run directory.

Resume requires matching data, resolved settings, source-code hashes, package
versions, device and recorded backend/thread settings. Changing these is a new
training stage, not exact resume. Fresh-process equivalence is tested on CPU with
dropout, auxiliary replay and partial epochs; GPU/MPS bitwise equivalence has not
been demonstrated. Recorded elapsed time includes committed work and the resumed
invocation, not unrecorded time lost after an earlier snapshot.

Final evaluation begins only after a committed end of optimization. Each split's
metrics and prediction records have an atomic cache bound to that run and selected
checkpoint. Reporting retries reuse caches. Test evaluation is marked before
consumption: an interruption without a complete cache fails rather than silently
repeating it. A completed resume verifies artifact hashes and returns the cached
result; a failed coverage gate remains failed. These guarantees do not authorize
new model selection using test results.

Every run saves initial/selected/last checkpoints, optimizer state, resolved config,
source manifest, source snapshots, admission census, history, parameter gradient
and update evidence, per-example train/validation/test predictions and metrics.
Prediction records retain the complete normalized observation, including source
and MHC provenance, plus resolved-field and register diagnostics in JSON form.
These reporting fields preserve input identity without changing the fitted loss.
Validation exact IC50 RMSE selects checkpoints when that output is trainable;
otherwise selection uses the observation-weighted loss of trainable tasks.
Frozen-task metrics are still reported. The test split is
only evaluated after selection. A training-only diagnostic must explicitly declare
`no_test_training_diagnostic` and explain why no test split exists.

## Inference and current limits

`presto.inference.modular.ModularPredictor` loads the new strict
`presto.modular.v2` format. Version 2 requires input-feature training evidence;
version 1 experimental checkpoints require an explicit migration and are rejected.
Its `predict()` returns physical units and transformed
values for regression, logits and explicitly uncalibrated scores for binary
outcomes, identity probabilities, effective context, register diagnostics and
training-support status. Random or unavailable requested outputs fail. A trained endpoint head cannot conceal
untrained parameters along its requested path; longer class-II peptides require
actual gradient/update evidence for register pooling.

Input support is also checked at the column/category level. Training one machinery
state or observing only missing acquisition metadata does not support a previously
unseen state/value. Checkpoints retain exposed and gradient-updated input columns
and amino-acid embedding rows. Fixed padding and unobserved entries are reported
separately in `coverage.json`. Numeric feature slots with no observed training
values are rejected at preflight. Inference refuses to activate unsupported columns.

Run closure requires tensor and exposed-feature coverage. A failed gate preserves
the selected checkpoint, prediction dumps, summary and coverage evidence, then
fails the run. An intentionally incomplete training-only diagnostic can declare
`coverage_policy="diagnostic"` and a nonempty `coverage_exception_reason`; this
does not validate any output. Ordinary held-out runs cannot use that exception.

`presto.data.modular_preparation.prepare_observations` writes source-specific pinned
JSONL files, a per-row disposition report and a census. Original measurements need
an explicit `measurement_lineage` (namespace, measurement ID, optional replicate)
or preserved source assay/evidence IDs; an export-row ID alone is insufficient.
Copied outcomes are deduplicated, conflicting exports of the same outcome are
deferred together, and independently measured replicates are retained with their
outcome variation reported. The training index rejects residual original-outcome
duplicates and checks retained group aliases across frozen splits. This preparation
primitive does not itself qualify source evidence or invent missing source adapters.

`rank_pmhc()` scores a supplied candidate library for paired TCR chains. pMHC is
an input to the scorer, not a generated output. Ranking cannot discover restriction
or peptide identity outside the library, and a lineage prior is not an allele-level
restriction claim. No trained TCRvdb checkpoint is included.

The assay-only 32-example fitting gate has passed. Supervised annotation
qualification, matched pretraining/replay comparisons, replicated convergence and
held-out IC50 qualification remain pending. A later local one-epoch IC50 fit
improves ranking and RMSE over a fixed composition ridge baseline, with varying
predictions; it leaves class-II sensitivity and censored observations unresolved. Other data gates cover matched APC
perturbations, quantitative MS component anchors/transfer and authorized paired
TCR records. Noncanonical or short class-II cores, PTMs, direct biochemical
cleavage/transport supervision and automatic full-cohort Hitlist preparation
remain unsupported. The training-only diagnostic does not qualify a serving model
or complete the full architecture and data plan.


## Experimental peptide-product distribution

`presto.models.peptide_availability.PeptideAvailabilityPolicy` provides a small
variable-length span policy for the availability learning program. It takes a
full parent sequence and a declared numeric context vector, then represents
`p(start, length | parent, context)`. Default support includes lengths 1–40 and
all valid protein positions. MHC binding does not filter this product pool.

Fixed residue/composition/flank features feed separate start and conditional
length MLPs. Exact enumeration exposes every valid span. Sampling evaluates all
start scores, draws starts, then scores lengths only at distinct sampled starts.
Repeated sampled products remain repeated observations. Selected log probabilities
retain the exact gradient; sampling choices themselves are discrete.

```python
from presto.models.peptide_availability import PeptideAvailabilityPolicy

policy = PeptideAvailabilityPolicy(min_length=1, max_length=40, context_size=0)
state = policy.prepare("ACDEFGHIKLMNPQRSTVWY")
samples = policy.sample(state, 64)  # zero-based starts, lengths, log probabilities
all_spans = list(policy.enumerate(state, chunk_size=256))
```

These are untrained scores until a qualified fit supplies weights. The state
contains an autograd graph and must be rebuilt after optimizer updates. Chunked
enumeration bounds inference working tensors; retaining every output for backward
still retains its graph. The policy saves conditional-network work during
sampling, while start scoring remains linear in parent length.

The [computational comparison](https://github.com/pirl-unc/presto/blob/codex/modular-presto-model-20261020/experiments/2026-10-22_1938_codex_variable-span-computation/README.md)
verifies the same probabilities and gradients against full enumeration and an
independent NumPy implementation. Its full-protein fixtures use synthetic
mathematical costs and fixed weights, with no biological training or predictive
validation. This primitive is not yet wired into the mixed-assay trainer.

Per-precursor product yield, clearance/residence time, calibrated concentration,
MS/capture response and MHC selection remain separate model/data requirements.
Biological context must come from the shared curation/override path; the numeric
vector does not infer species, assign gene knockouts, or establish feature-training
support. A future trained integration must enforce the same tensor/feature
coverage and source-admission gates as the canonical model.
