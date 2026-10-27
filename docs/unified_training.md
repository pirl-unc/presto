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

## Binding registers and flanks

MHC-I contributes the entire peptide as one candidate. MHC-II contributes every
contiguous nine-residue core, giving `L - 8` registers for a length-`L` ligand.
Learned weights use a softmax over that ligand's candidates. Register enumeration
is discrete; the encoder, candidate scoring and weighted pooling are differentiable.
This is a sequence-level approximation, not an explicit open/closed-groove structure
or a model of class-I peptide bulging.

Each class-II register combines the encoded core and MHC grooves with separate
128-dimensional N- and C-flank summaries and three length features. Summaries
pool contextual token states from one encoding of the **entire ordered peptide**.
Every residue outside the candidate core contributes; there is no three-residue
cutoff. Sinusoidal positions and self-attention preserve order before pooling.
Empty flanks are zero vectors; padding does not enter the averages. Class-I
candidates use the full peptide representation and empty flank vectors.

These are peptide residues outside the proposed binding core, distinct from
source-protein residues outside the observed ligand. Source-protein context keeps
its own explicit observation fields and downstream route.

Three-residue flank composition has precedent in
[NetMHCIIpan (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2430535/) and the
[2015 core-identification study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4637192/).
That does not establish sufficiency for Presto's endpoints. Full context removes
that input restriction, but its accuracy benefit has **not** been measured.
No scientific qualification is transferred from an older architecture by this change.

## Migration and historical checkpoints

The current checkpoint format is `presto.shared.v3`; its fixed peptide-context
contract is `full_peptide_contextual_tokens_v1`. New fits must use this contract.
Old modular-v2 checkpoints dispatch to the historical implementation for inference;
they cannot silently initialize or resume a v3 training run. Their input feature
layout differs. A new model requires new training and scientific qualification.

`presto.inference.ObservationPredictor` (also named
`presto.inference.modular.ModularPredictor`) reads shared-v3 and modular-v2 checkpoints
and returns observation predictions in declared units, with support diagnostics.
The older default-model predictor remains at `presto.inference.predictor.Predictor`
for historical checkpoints. The older `predict presentation`, `predict recognition`,
`predict tile` and `evaluate synthetic` CLI commands are historical-model interfaces;
they are not readers for shared-v3 checkpoints.

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
