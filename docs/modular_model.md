# Modular directly supervised model

The optional `modular` model implements the software foundation of
[issue #130](https://github.com/pirl-unc/presto/issues/130). It has no qualified
biological checkpoint yet. Passing software tests does not establish improved
IC50 accuracy, useful TCR specificity or separately calibrated MS detectability.
The default model and its checkpoint format remain unchanged.

## Shape and information flow

One shared sequence encoder uses 20 canonical residues plus padding, width 128,
two Transformer layers, four attention heads, feed-forward width 512, sinusoidal
positions and masked mean pooling. It encodes each sequence independently.

| Requested outcome | Required sequences | Additional admitted inputs |
| --- | --- | --- |
| Molecular identity | MHC groove pair or paired TCR chains | Actual identity labels during training |
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

## MS observation

The same detector feature block and scalar readout are used in both routes:

```text
bulk logit    = digest score + shared MS score + bulk observation offset
elution logit = cellular score + shared MS score + elution observation offset
```

Digest features consume peptide, optional source context and enzyme. Shared MS
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

## Observation and target contract

`presto.data.modular.Observation` is the single train/inference input. JSONL rows
include the explicit `presto.modular.observation.v1` schema, stable observation and
source IDs, endpoint, sequences, raw value/unit/qualifier, frozen split/group ID,
context and provenance. Identity rows carry independent `identity_labels` and do
not need a peptide or assay value. A candidate MHC bag belongs to one observation.

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

Source-specific preparation still owns cross-dataset deduplication, contradiction
review, cell-line/parent/clone attribution and exclusion/defer reports. The training
index verifies hashes, schema, unique observation IDs and group/peptide-disjoint
splits; it does not certify donor or receptor-cluster independence from IDs alone.
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
unique labels. A compatible pretrained subset may initialize a larger declared
subset through a hash-pinned `pretrained` entry with `path` and `sha256`; new heads
remain unsupported. Existing shared feature vocabularies and scalers cannot be
silently changed. Exact resume, arbitrary migrations and frozen-submodule training
are not implemented in this first version.

Every run saves initial/selected/last checkpoints, optimizer state, resolved config,
source manifest, source snapshots, admission census, history, parameter gradient
and update evidence, per-example train/validation/test predictions and metrics.
Validation exact IC50 RMSE selects checkpoints when available. The test split is
only evaluated after selection. A training-only diagnostic must explicitly declare
`no_test_training_diagnostic` and explain why no test split exists.

## Inference and current limits

`presto.inference.modular.ModularPredictor` loads the new strict
`presto.modular.v1` format. Its `predict()` returns physical units and transformed
values for regression, logits and explicitly uncalibrated scores for binary
outcomes, identity probabilities, effective context, register diagnostics and
training-support status. Random or unavailable requested outputs fail. A trained endpoint head cannot conceal
untrained parameters along its requested path; longer class-II peptides require
actual gradient/update evidence for register pooling.

`rank_pmhc()` scores a supplied candidate library for paired TCR chains. pMHC is
an input to the scorer, not a generated output. Ranking cannot discover restriction
or peptide identity outside the library, and a lineage prior is not an allele-level
restriction claim. No trained TCRvdb checkpoint is included.

The next experiments remain the supervised identity qualification, registered
32-example fitting gate, matched mean/ridge baseline comparison and held-out IC50
qualification from issue #130. Later data gates cover matched APC perturbations,
MS component anchors/transfer and authorized paired TCR records. Noncanonical or
short class-II cores, PTMs, quantitative MS endpoints, reaction/stimulus-specific
features and automatic full-source Hitlist export preparation remain unsupported.
No scientific milestone is marked complete by the software integration fixtures.
