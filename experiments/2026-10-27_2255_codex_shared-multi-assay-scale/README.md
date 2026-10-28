# Shared architecture: larger data, distinct outputs and capacity

**Status: running eight frozen H100 fits; inputs independently verified.**
Agent/model: Codex / GPT-6. Date2026-10-27. Follows the validation-selected
shared-v5 interaction architecture in `2026-10-27_2100_codex_transfer-improvements`.
Detailed implementation plan: `../agents/codex/plans/2026-10-27_scale-shared-model.md`.
Source audit: `../2026-10-27_2235_codex_shared-support-audit/README.md`.

## Frozen comparison before fitting

Eight fits: four conditions, seeds42 and43. Each starts from scratch and uses the
same observation-manifest trainer, shared encoder, residue interaction, ordered
full peptide context, candidate aggregation and endpoint-specific physical readout.
No pretraining, synthetic labels, identity losses, class-specific heads, learning
rate schedule, AMP, TF32, clipping or changed target caps. Mean standardized
one-sided squared loss per observed assay row; each literal endpoint has its own
scale fitted on exact training outcomes only. AdamW3e-4, decay1e-4, batch64,
dropout0.1;6000 updates, validation every500 updates and at final budget,
checkpoint every100. Early stopping disabled within that budget. Natural sampling
without replacement per data epoch; no endpoint or class balancing. Exposures are
reported separately, because matched updates do not give matched IC50 exposure.

| Condition | Training observations | Shared capacity |
| --- | --- | --- |
| `ic50-exact` | Frozen original exact IC50 training rows |128/2layers/4heads/512FF|
| `ic50-bounds` | Same IC50 rows plus original admitted training bounds |128/2/4/512|
| `multi-assay` | All qualified IC50, directKD, EC50, KD(~IC50), KD(~EC50), half-life |128/2/4/512|
| `multi-assay-wide` | Identical six-output data |192/3/6/768|

The first contrast isolates original bounds. The second adds distinctly labeled
measured endpoints (including proxies); it does not isolate number of rows from
new supervision. The third isolates model capacity. No label-shuffle mechanism
claim is made. All models use peptide and both verified groove sequences; the
larger source adds peptide/groove diversity, not invented cellular/TCR context.
The seven-endpoint admission retains10off-rate training rows, but no off-rate
validation/test rows exist. Off-rate is excluded from these fits and is not a
validated output. EC50 class-II evaluation coverage must be reported explicitly.

## Data and leakage contract

Pinned Hitlistv4 binding/other-assays exports, original purified-MHC quantitative
text, qualifiers and units, canonical unmodified peptides, complete source-named
human MHC partners from the pinned mhcseqs catalog. No guessed DRA or partner.
The original27,364-row IC50 cohort files remain byte-identical. Across endpoints,
peptide and original-assay connected components inherit frozen historical/IC50
splits; conflicting components are deferred. Original measured-outcome IDs have
one vote; independent replicate disagreements remain. Manifest/source hashes and
exact counts are frozen in `reproduce/source-pins.json`; independent source-byte,
unit/target, catalog and cross-endpoint leakage verification precedes launch.
These are development holdouts, with study and sequence-family overlap possible;
new endpoint readouts do not create an independent prospective population.

## Selection, reporting and promotion

Trainer checkpoint selection: lowest validation exact IC50 RMSE (the existing
common selection rule), at identical update opportunities. Final predictive tests
are evaluated once on that selected checkpoint. The local controller exposes only
validation during fitting; condition selection is sealed before opening test dumps.
Primary promotion comparison uses **original IC50 validation IDs**, exact values,
separately by class. New-condition two-seed mean macro class Spearman must improve
by at least0.01 over `ic50-exact`, with neither class losing more than0.005Spearman
or worsening RMSE by more than2%. At least one seed must improve each class and
both seeds must improve macro Spearman. Select the qualifying condition with the
highest macro Spearman; ties within0.005 favor smaller parameter count, then fewer
outputs. If none qualifies, retain the existing IC50 recipe and publish the added
outputs as experimental with their measured limits. No observed test result can
change that decision. Also report all six per-endpoint/class heldout metrics,
train-mean baselines, bounds, calibration, per-groove strata, input/weight evidence
and exact endpoint/class label exposures. Sparse or missing strata are not qualified.

GPU: requested **H100!**, at most2concurrent workers;4CPU,32GiBhost RAM,4h timeout
per worker. Record actual device and memory from PyTorch and nvidia-smi. This budget
extends prior2000-update fits; hardware/runtime contracts remain explicitly separate.

## Reproduction and closure

`code/prepare.py` freezes data manifests, exact runtime files and checksums;
`code/launch.py` runs the frozen bundle on Modal and collects complete archives.
`reproduce/launch.json` and `launch.sh` record the invocation and environment.
Large data and raw checkpoints live in repo-level `data/` and `artifacts/`;
canonical per-example heldout dumps, summary tables and decisions belong in
`results/`. Add exact runtime, validation/test metrics, support limits and final
decision here and in `../experiment_log.md` before declaring this family complete.

Prelaunch software check: admit every condition with the frozen common runtime,
then execute one CPU update on at most two TRAIN observations per endpoint/class
stratum of the wider model. Preserve its IDs, loss, gradients and checkpoint.
This checks mixed-output execution and is not a predictive fit or test evaluation.
The pinned original IC50 preparer numbers source rows from1; the new scanner
numbers them from0. Independent verification explicitly accounts for both, checks
raw-record fingerprints and does not rewrite the frozen original observations.
