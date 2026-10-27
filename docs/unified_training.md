# One model and training interface

New training uses `presto.models.Presto` (`PrestoConfig`) and one observation
manifest. `ModularPresto` and `ModularConfig` are compatibility aliases to those
same classes. They do not select another architecture.

```bash
python -m presto train --manifest /absolute/path/manifest.json --preflight
python -m presto train --manifest /absolute/path/manifest.json \
  --run-dir experiments/YYYY-MM-DD_HHMM_codex_description --device cpu
```

The [shared model guide](modular_model.md) specifies the observation schema,
manifest, endpoint meanings, coverage gates, transfer/freezing and continuation.
Read `experiments/EXPERIMENT_WORKFLOW.md` before an experiment. Model settings,
optimizer settings and hash-pinned data sources live in the manifest. Source
adapters prepare observations; they do not own models or optimization loops.

## Common execution path

All supported launchers call `cli.train.run`, then `training.modular_run.run_manifest`,
`fit_index` and `training.modular.train_step`. The public `training.Trainer` facade
calls the same `fit_index`. One shared Transformer encodes peptide, MHC and paired
receptor sequences. Endpoint configuration allocates the required readouts and
context modules within this same model; every subset need not allocate unused heads.

| Training use | Manifest contract |
| --- | --- |
| Identity pretraining | `identity` observations; declared identity tasks; no assay endpoints required |
| MHC-I or MHC-II binding | Affinity/stability/kinetic observations and resolved MHC grooves |
| Cellular elution, bulk MS and component anchors | Corresponding binary or quantitative endpoints, with their declared contexts and measurement units |
| Paired TCR binding or response | `tcr_binding` / `tcr_response`, paired sequences and appropriate measured labels |
| Joint training | A union of admitted observation types; shared encoder and optimizer step |
| Fine-tuning | Hash-pinned compatible `pretrained` checkpoint; optional explicit dependency freezing |
| Continuation | Same frozen manifest and run directory with `--resume` |
| Synthetic software checks | Explicit synthetic observations and provenance in the same schema; no implicit negative generation |

There is no source-specific or class-specific trainer. An elution label remains
an elution label: it does not establish purified binding or independently
identified antigen processing. Component scores require independent anchors and
validation for a mechanistic interpretation. Paired receptor scores do not replace
repertoire-level recognition probabilities.

## Observation sampling

The manifest `training.balance_by` field defaults to `[]`, preserving shuffled
passes over observations. An explicit `["mhc_class"]`, `["endpoint"]`, or their
combination groups training rows by those declared fields. Each epoch cycles
smaller shuffled groups to the largest group's size and interleaves groups.
This changes population weights and epoch length; comparisons must record actual
exposures, optimizer updates, validation cadence and runtime. Validation/test
observations and train-derived target scalers are not rebalanced. MHC-class
balancing requires one explicit class per sampled observation; unknown or mixed
class bags fail admission. Single-group sampling is identical to ordinary shuffling.
The durable order and RNG state support exact interrupted-run continuation.

This is an option in the common trainer, not a class-specific model or optimizer.
Do not assume that equal class exposure improves every endpoint or population.

## Binding registers and flanks

MHC-I contributes the entire peptide as one candidate. MHC-II contributes every
contiguous nine-residue core, giving `L - 8` registers for a length-`L` ligand.
Each candidate uses the same encoder and endpoint readout. Its peptide
representation averages an independently encoded core (positions restart at the
core) with contextual token states at those core positions in the full peptide.
One shared four-head cross-attention then reads both groove token sequences,
followed by residual normalization and residue pooling. Either class trains every
interaction parameter; the endpoint readout is also shared. All
flanks can affect those states through attention; there is no flank-only projection
or class-one-hot input. For MHC-I, both views reduce to the full peptide encoding.
The rule is fixed across training runs, with no selectable alternate backend.

The endpoint readout scores each candidate before pooling. A normalized log-sum-exp
uses soft-min for affinity/concentration/off-rate outputs and soft-max for
stability/cellular/receptor outputs, with temperature one in model target units.
The derivative weights are a variable-length softmax. One candidate returns its
score exactly, and duplicating equal candidates does not change the prediction.
There is no independently learned register-attention head. This is a statistical
pooling assumption, not a physical occupancy equation or a mechanistic kinetic model.

Register enumeration is discrete; the encoder, scores and pooling are differentiable.
This is a sequence-level approximation, not an explicit open/closed-groove structure
or a model of class-I peptide bulging. Shared parameters permit cross-class scoring
without newly initialized modules. The fixed-budget development comparison in
`experiments/2026-10-27_2100_codex_transfer-improvements/README.md` supports the
shared interaction package for joint IC50; zero-shot transfer remains modest.
This is not evidence of generalization to other assay families.

These are peptide residues outside the proposed binding core, distinct from
source-protein residues outside the observed ligand. Source-protein context keeps
its own explicit observation fields and downstream route.

Three-residue flank composition has precedent in
[NetMHCIIpan (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2430535/) and the
[2015 core-identification study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4637192/).
That does not establish sufficiency for Presto's endpoints. Full context removes
that input restriction. The fixed-budget H100 comparison recorded in
`experiments/2026-10-27_1930_codex_modal-cross-class/README.md` measured modest
cross-class transfer and no shared-v4 joint-affinity improvement.
Those architectures differ beyond flank context, so the comparison does not isolate
its causal benefit. No scientific qualification transfers from an older architecture.

## Migration and historical checkpoints

The current checkpoint format is `presto.shared.v5`; its fixed peptide-context
contract is `shared_core_context_v2`. New fits must use this contract.
Old shared-v4, shared-v3 and modular-v2 checkpoints dispatch to the historical implementation for inference;
they cannot silently initialize or resume a v5 training run. Their input feature
layout differs. A new model requires new training and scientific qualification.

`configs/ic50_joint_development.json` provides the tested natural-sampling,
2,000-update recipe for the frozen human IC50 development cohort. It expects the
checksummed inputs preserved under `artifacts/2026-10-27_2100_codex_transfer-improvements/upload`.
Run it through `presto train --observations-manifest configs/ic50_joint_development.json`.
It is a bounded comparison recipe, not a convergence claim or a new general-purpose
training default. The frozen experiment contains the exact H100 source/runtime contract.

`presto.inference.ObservationPredictor` (also named
`presto.inference.modular.ModularPredictor`) reads shared-v5, shared-v4, shared-v3 and modular-v2 checkpoints
and returns observation predictions in declared units, with support diagnostics.
The older default-model predictor remains at `presto.inference.predictor.Predictor`
for historical checkpoints. The older `predict presentation`, `predict recognition`,
`predict tile` and `evaluate synthetic` CLI commands are historical-model interfaces;
they are not readers for shared-v5 checkpoints.

Old training flags such as `--epochs`, `--data-dir` and `--config` are rejected.
Use manifest model/training/source fields instead. The optional command names
`unified`, `synthetic`, `pretrain` and `finetune` all dispatch identically; their
names do not generate data or alter the model. The accepted `--model-kind shared`
and `--model-kind modular` spellings select the same implementation.

The former train/pretrain/probe/benchmark/sweep script names are thin aliases to
this interface. They no longer provision remote machines, create sweeps, select
experimental architectures or generate synthetic datasets. Run the common CLI
on a prepared worker and describe each comparison with explicit manifests.

Historical model, trainer and helper code is isolated under `presto.legacy` for
checkpoint reading and compatibility tests. Archived script execution exits with
a migration message. Exact old experiments must use their pinned code/environment;
the immediate pre-consolidation source is commit
`9dd71348124faa49a4c6a60872385c86895d50b0`. The archive inventory and original source
hashes are in `tasks/unified_training_core_20261027_archive.json`. Historical
experiment directories and recorded results are unchanged.

## Deliberate simplifications and remaining gaps

The old configurable core-length machinery and optional core-supervision loss
are retired from new training; the current class-II core is nine residues with
latent register weights. Legacy architecture bakeoffs, custom optimizer/schedule
flags, implicit synthetic negatives and independent TCR wrappers are not new
training options. Old cleavage/transport latent heads and repertoire recognition
outputs are not automatically equivalent to the shared model's observation or
component endpoints. Retaining their historical checkpoint reader does not claim
that those outputs have been ported or scientifically validated.

Software checks cover shared execution, full-flank order sensitivity and gradients,
variable register counts, endpoint isolation, checkpoint replay, transfer and
resume. They do not measure improved MHC-I/MHC-II binding, processing or TCR
accuracy. Prospective predictive comparisons remain separate registered work.

### Scaling the shared model

The observation manifest accepts `d_model`, `n_layers`, `n_heads` and `d_ff` in
`model`; defaults128/2/4/512 preserve the shared-v5 checkpoint contract. Every
reachable task uses these shared dimensions. In `training`, optional
`validation_every_steps` schedules validation and checkpoint selection on update
counts across data-epoch boundaries. Unset preserves epoch-based validation.
The final budget is always evaluated; interval-mode patience counts validation
events, with `min_epochs` retaining its epoch gate. Partial epochs and pending
validation survive exact resume without dropping data or skipping evaluation.

Coordinate-level gradient/update masks are checkpointed and summarized in
`coverage.json`. Historical checkpoints without this evidence retain unknown
historical coordinate coverage. Fixed padding and redundant attention key biases
are distinguished from ordinary weights without relaxing existing output support
and input-exposure checks.

`training.selection_endpoint` makes checkpoint priority explicit. Name a supervised
assay endpoint to select its exact-value RMSE when available, or its validation
loss otherwise. With null (the general default), select the observation-weighted
mean validation assay loss; identity-only runs use identity loss. Frozen or absent
primary endpoints are rejected. The carried-forward IC50 recipe explicitly names
`ic50`, preserving its measured selection contract. Pooled objectives can trade
performance between classes; inspect the class/output tables as well as the scalar
selection score. This option does not create a separate model or training path.
