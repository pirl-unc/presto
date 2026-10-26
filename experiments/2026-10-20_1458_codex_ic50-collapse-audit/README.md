# IC50 collapse: training-only artifact audit

Agent/model: Codex / GPT-6. Registered 2026-10-20 before analysis.

Analyze the published 2026-10-16 frozen fit (Presto 0.1.16, Hitlist 1.78.0,
manifest b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372).
No new fitting or held-out selection. Use existing training predictions,
checkpoint weights and optimizer/update ledgers to separate input/target
variation, head compression and objective/optimization hypotheses.

Budget: local CPU only, up to 20 minutes of diagnostic computation, no GPU,
no generated biological labels. First inspect existing artifacts. Receptor
work and data discovery are separate from this experiment. Historical validation
and test metrics remain in the source experiment; do not rerun held-out
predictions or claim new validation/test results here. Training prediction
exports belong in results when recomputed.

Source artifacts: https://github.com/pirl-unc/presto/releases/tag/effective-target-fit-qualification-v1
Source publications are pinned by the existing repository publication.json.
Downloads go into a separate datacache bundle, never the local live Hitlist store.

Status: completed training-only artifact audit; see results below.

## Prospective diagnostic extension — model/training-path inventory

Registered before execution on 2026-10-20 in response to the request for a deep
architecture and underfitting audit. Specification:
`../../tasks/model_training_path_audit_spec.md`.

Compare every named parameter in the same initial and selected checkpoints;
aggregate exact changes and optimizer moment evidence by module and latent.
Extract training support, historical direct-output gradient counts and objective
weights. Parameter change is not evidence of predictive quality, and weight
decay can move tensors without task-specific learning. The selected checkpoint
is epoch 1, while the historical output-update ledger spans both epochs; keep
those observation windows separate. No new optimization or held-out inference.
Local CPU-only analysis budget remains 20 minutes of diagnostic computation.
Source files and resulting tables will be frozen alongside the existing replay.

Replay scope fixed before execution: all 877 training class-I IC50 rows, both
initial and selected checkpoints. Resolve grooves from the archived catalog.
Require <=2e-5 log10(nM) agreement with archived predictions before interpreting
feature contraction. Class-II replay is outside this bounded diagnostic.

## Completed result

The archived predictions replay on all **877 class-I training observations**
with maximum absolute errors **9.54e-7** (initial) and **1.67e-6** (selected)
log10(nM). No checkpoint was modified. Final analysis runtime was about seven
seconds on local CPU (one Torch/BLAS thread); no GPU was requested or used.
The first two attempts failed before inference because the raw catalog uses
`HomoSapiens-` allele keys. The final script uses the canonical resolver against
the archived catalog. Failure logs are preserved. After the first successful
replay, an additive read-only covariance diagnostic measured the existing
core/probe terms; there was no optimizer, target or partition change.

| Between-example standard deviation | Initial | Selected |
| --- | ---: | ---: |
| Peptide representation, RMS across coordinates | 0.134199 | 0.126011 |
| pMHC interaction representation, RMS across coordinates | 0.173150 | 0.052700 |
| Affinity readout representation, RMS across coordinates | 0.037235 | 0.025270 |
| IC50 prediction, log10(nM) | 0.002809 | 0.007991 |
| Target/bound, log10(nM) | 1.526579 | 1.526579 |

Selected predictions on this subset span **3.183600–3.206863 log10(nM)**.
The direct IC50 path has finite nonzero output gradients in all 200 historical
optimizer batches, with 3,645 repeated direct-label exposures. Thus an absent
loss mask or detached IC50 output does not explain this result. The historical
learned uncertainty weight is about 0.95/0.94 at the two epoch summaries; it
has not reduced this objective to near zero. Its task-normalized batch
weight is about 0.074, within a multi-objective fit; that is a separate factor.

The selected core-KD and bounded probe-KD terms have Pearson **-0.999313**.
Their weighted standard deviations are 0.011074 and 0.021564 log units, but the
mixture's is only 0.010506. The final IC50 standard deviation is 0.007991.
The affinity representation's mean-vector RMS grows from 0.17923 to 2.72896
while its between-example RMS standard deviation shrinks. Flattened tensor
variance would obscure this shared-offset behavior.

**Decision:** the existing data are sufficient to start debugging fitting.
Priority is a separately registered training-only capacity/optimization check:
exact-IC50-only fitting and a small-cohort memorization check, followed by
controlled objective/representation comparisons if needed. No production
architecture is changed on this observational evidence alone. The covariance
and contraction locate where variation is lost; they do not prove which loss,
initialization, schedule or architectural component caused it. More data or an
export refresh is not an established cure.

## Reproducibility and limits

- `reproduce/launch.sh`, `reproduce/launch.json` and
  `reproduce/source/audit.py` preserve the invocation, implementation and local
  versions. Input artifact hashes/URLs and raw cache paths are in
  `results/audit.json`. The public archive remains the authoritative raw source.
- `results/train-class-I-replay.csv` preserves source IDs, target/bound,
  qualifier and both replayed predictions for every included training row.
  `results/optimization-summary.csv` extracts the historical objective weights.
- The model implementation is unchanged from the historical producer apart
  from a docstring in `models/presto.py`; inference parity guards current
  collation/dependency differences. Historical producer: Presto 0.1.16 /
  Hitlist 1.78.0; this analysis runs the current checkout, not a claimed 0.1.16
  installation. See the exact environment in `launch.json`.
- This is a training-only forensic analysis of class I. Class II (1,197 IC50
  training rows) was not replayed. No held-out evaluation or new predictive
  experiment took place, so no new validation/test metrics or prediction dumps
  exist. The historical failed qualification, including its held-out predictions,
  remains in `../2026-10-16_1030_codex_effective-target-fit/` and its public release.
- No training/pretraining parameters, loss terms/weights, assay-label mapping,
  synthetic-data contract or selection policy changed. Exact IC50 and censored
  bounds keep the original log10(clip(nM, .001, 50000)) target contract.
  This diagnosis does not qualify any TCR or other output.

## Completed training-path extension

The final read-only inventory completed in 0.32 seconds of local CPU analysis,
excluding imports, with no model forward/backward or optimizer step. The
[model training audit](../../docs/model_training_audit.md) traces inputs,
representations, assay losses and current TCR-wrapper limitations.

- 476 parameter tensors / 409,850 parameters. At the selected epoch-1
  checkpoint, 108 tensors containing 94,578 parameters have no optimizer state.
  116 tensors containing 95,453 parameters remain exactly unchanged; 359 tensors
  have nonzero Adam second moments. Parameter movement is not useful learning.
- Recognition, immunogenicity, T-cell and receptor-evidence modules are unchanged.
  Processing representations changed through downstream losses, while both
  scalar processing heads remained unchanged.
- The 7,241 training observations partition into 3,589 concentration records,
  19 koff, 434 half-life, 111 Tm and 3,088 positive-only elution bags.
  There are no kon, measured processing, T-cell, TCR or core-position labels.
  Source-origin targets are all human and foreignness targets all zero.
- Source IC50 labels additionally train generic KD, an affinity probe and a
  binding panel. MIL elution and MS are identical predictions counted as separate
  objectives. These verified couplings motivate controlled tests; they do not
  establish the cause of underfit.
- `results/training-paths/` contains parameter/module inventories, train support,
  historical objective and output-gradient tables, source endpoint counts, input
  hashes, execution environment and limitations.
- `reproduce/training-paths.sh`, `reproduce/training-paths.json` and the frozen
  `reproduce/source/training_paths.py` reproduce this extension. The existing
  replay invocation and results remain separate.
- Checkpoint deltas/moments cover epoch 1 only; historical gradients/objectives
  cover both epochs. No validation/test inference or new metrics were produced.
  All original target transforms, synthetic-data exclusions and loss weights
  remain unchanged. The proposed next comparison separates loss routing from
  head complexity before attributing the contraction.
