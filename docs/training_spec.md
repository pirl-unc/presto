# Training guide

The authoritative desired model semantics are in
[model_io_contract.md](model_io_contract.md). This guide describes the current
canonical training path and its remaining limitations; it does not define a
second I/O contract.

## One production path

```bash
python -m presto train unified --data-dir ./data --epochs 5 --checkpoint presto.pt
```

This delegates to [scripts/train_iedb.py](https://github.com/pirl-unc/presto/blob/main/scripts/train_iedb.py), using the
shared loss loop and task registry in
[scripts/train_synthetic.py](https://github.com/pirl-unc/presto/blob/main/scripts/train_synthetic.py). Unified mixed-source
training uses time-varying weights. Older staged/utility trainers and historical
experiment launchers are not alternate production contracts.

CLI and YAML/JSON `--config` options merge through `IEDB_DEFAULTS`. New training
options must be registered there as well as in the CLI parser.

## Data and curation

- The default primary source is `merged_tsv`, normally
  `data/merged_deduped.tsv`. Build it with `presto data merge --datadir ./data`.
- `--data-source hitlist` explicitly selects the curated Hitlist pMHC indexes
  instead. It does not union them with merged T-cell/TCR rows.
- The existing `--allow-raw-fallback` is an explicit debug option, not the
  production source-selection design. Missing required primary data must not
  silently pick another corpus.
- `mhcseqs` is the default sequence resolver. An index CSV is an optional
  supplement, not a prerequisite. Strict MHC resolution is the normal contract;
  unresolved reports identify offending alleles.
- `mhc_a/mhc_b` are groove segments. Preserve the original source allele set,
  selected class-II pairing, resolved subset and resolution lineage separately.
- Load caps are per-source/modality curation limits, not examples per epoch.
  A cap of zero means uncapped for that modality.
- Bulk non-MHC MS is optional and off by default. Its labels/negatives and
  detectability supervision are not interchangeable with MHC-elution evidence.

Source normalization, reference-aware deduplication, null-safe sequence
normalization and mapping selection precede batching. The run directory should
retain source statistics, normalized data funnel, preflight support/lineage
audits, dataset fingerprints and curation configuration. Do not quote historical
corpus counts as current defaults.

The complete source and target map is in
[assay_learning_scheme.md](assay_learning_scheme.md). Availability of a parser
does not mean every assay family has been loaded or has held-out support.

## Output coverage and prospective claims

First run the intended curation and split configuration without fitting:

```bash
python -m presto train unified --data-dir ./data --run-dir ./artifacts/preflight \
  --data-preflight-only --data-seed 17 --seed 42 --val-frac 0.1 --test-frac 0.1
```

`output_coverage.json` contains the executable output contract, source contract,
per-split counts and evidence strata. `output_coverage.csv` is the flat canonical
count view. Existing `split_support.json` and its tables retain their schema and
fingerprints. `supported_outputs.json` marks every canonical cell undeclared
until a prospective manifest is supplied. Output existence is not support.

Create a JSON manifest with exactly these top-level fields:

| Field | Required value |
|---|---|
| `schema_version` | Integer `1` |
| `model_contract` | The preflight's `contract.configuration` object |
| `source_contract` | The preflight's `source_contract` object, including fingerprints |
| `claims` | A nonempty list of explicitly scoped claims |

Each claim contains exactly `endpoint`, `columns`, `required_splits`,
`evidence_roles`, `source_families`, `raw_sources`, and `minimums`. Use canonical
endpoint names and named panel/class/component columns; a scalar column is `""`.
The split/role/family/source lists must be explicit. Roles are `direct`, `proxy`,
`auxiliary`, and `synthetic`; unknown provenance cannot pass. Regular families
come from the declared endpoint, while generated families use the exact
`generated:<kind>` string with the `synthetic` role. Each claim's filters apply
jointly before exact distinct counting.

Choose and justify minimums in the experiment plan before fitting; the code
does not choose scientifically adequate thresholds. Every claim requires
positive `unique_observations` and `distinct_peptides` minimums. Binary and
categorical objectives additionally require explicit `unique_positive` and
`unique_negative` minimums; quantitative objectives require `unique_exact`.
Those balance/exact minimums may be zero when explicitly justified. Original
source identity coverage is available as `traceable_observations` and
`untraceable_rows`; unique counts can otherwise include fallback sample IDs.

Run the same curation/split configuration with
`--supported-output-manifest path/to/claims.json` and a fresh run directory.
The manifest is copied to `supported_output_manifest.json` before source loading;
an existing different declaration cannot be overwritten. Missing splits,
incompatible contracts, malformed claims or insufficient counts stop fitting
and leave reports for inspection. Undeclared columns remain undeclared even
when the selected subset passes. This gate does not disable other objectives;
it scopes evidence claims about the run.

Declared training runs automatically record `output_updates.json`; exploratory
runs can opt in with `--track-output-updates`. Both options require a run directory
or checkpoint path. The tracker observes initialization, frozen stages, label
exposures, column output derivatives and actual optimizer changes. Output
derivatives are observed before PCGrad, while parameter gradients are observed
after clipping/projection. Zero-gradient rows may change through weight decay
or optimizer state, and shared readouts may have no dedicated column parameter.
Counts are training exposures rather than distinct source measurements. Tracking
adds diagnostic overhead and does not supply prediction-quality evidence.

Model output variants may be set in YAML/JSON `train.unified` configuration,
including `affinity_assay_residual_mode`, `kd_grouping_mode`,
`affinity_target_encoding`, `core_window_lengths`, `binding_direct_segment_mode`
and `binding_kinetic_input_mode`. The manifest must match the selected variant.
The output contract fixes `max_affinity_nM` to the shared loss convention, 50000.

## Batches and model inputs

The collator builds sparse targets, masks and qualifiers. Canonical row training
and held-out scoring both call `PrestoBatch.model_inputs()`, so boundary flags
and biological context reach the same forward. Assay descriptors stay outside
that input dictionary and select output tracks for supervision.

Balanced mini-batches are enabled by default. They balance available
assay/source/label/allele/synthetic strata; they cannot create missing real
negative observations or validation support for rare targets.

MIL carries per-instance sequences, class, context and source boundary flags.
Noisy-OR supervises supported elution/presentation/MS and T-cell pathway bags.
A positive bag does not label every member positive. Source flags/context must
survive capping and synthetic contrastive construction.

## Loss semantics

The executable task registry is the source of truth. In particular:

| Target | Current objective |
|---|---|
| Binding and affinity-family concentrations | Censor-aware regression in configured target space |
| Binding descriptor panels | Same qualifier-aware regression at selected output columns |
| kon/koff | MSE in log-rate space; kinetic censoring still pending |
| Half-life/Tm | Censor-aware regression in log-minutes/normalized-Celsius space |
| Processing, presentation, elution/MS | BCE or supported bag-level objective; several labels are proxies |
| T-cell response/immunogenicity | BCE |
| T-cell fixed panels | BCE against response at the observed column, not CE against metadata IDs |
| Receptor evidence | Binary evidence plus categorical method auxiliary, not specific TCR matching |
| MHC class/species/fine type | Auxiliary categorical objectives |
| Core start | CE only where actual labels/masks are present |

Censor qualifiers are `-1` for an upper bound, `0` for exact, `1` for a lower
bound. Exact values use squared error; inequalities penalize only violations of
the bound in the transformed target space. Every consumer of a measurement must
honor that qualifier. Regression/threshold metrics must distinguish censored
from exact observations.

Default supervised loss aggregation is `task_mean`; configured uncertainty
weights and regularizers are recorded with the run. Regularizer names alone
are not evidence that their scientific premise is correct.

## Synthetic negatives and priors: current versus intended

Current defaults enable synthetic pMHC negatives (ratio 1.0), an additional
class-I `no_mhc_beta` category (ratio 0.25), and processing corruption (ratio
0.5); elution/cascade ratios are derived unless explicitly configured.
Random/scrambled sequences and mismatched pairs are synthetic hypotheses,
not measured nonbinding or nonpresentation observations.

**Known drift:** `no_mhc_beta` blanks the second groove segment. With class-I
groove inputs this is alpha2, not beta2m. It cannot be interpreted as a beta2m
KO. The corresponding assembly priors and missing-segment negative semantics
must be revisited under #46. Missing information must not become a biological
negative label in the intended design.

Report real and synthetic support/metrics separately. Freeze synthetic kinds,
ratios, parent observations and loss weights in every experiment contract.
Do not claim all defaults have scientific validation merely because training
runs successfully.

## Validation and closure

Default `test_frac` is 0: explicitly reserve a test split when needed. Preflight
support/lineage gates are configurable, not a guarantee every invocation passes
a mandatory full-corpus audit. Set and record required task/split support and
balance criteria for the study being run.

Held-out scoring loads the selected best-validation checkpoint when available
and records its identity. Task extraction uses the same target transforms and
selected-column resolver as training. Preserve validation/test loss,
per-example predictions, qualifier-aware regression/threshold metrics and
source/mapping lineage. State explicitly when a test split is absent.

All scientific experiments must follow
[EXPERIMENT_WORKFLOW.md](https://github.com/pirl-unc/presto/blob/main/experiments/EXPERIMENT_WORKFLOW.md), freeze launch
configuration and dataset curation, and close out artifacts and metrics in the
canonical registry. Unit training/checkpoint tests demonstrate wiring, not
scientific performance.

## Remaining work

[Issue #46](https://github.com/pirl-unc/presto/issues/46) tracks independent context
roles, no cross-role default copying, full training/serving/MIL parity,
groove-correct priors, family-specific quantitative routing, joint assay panels,
full-corpus support and scientific validation. Checkpoint migration fallbacks
are not part of the new design.
