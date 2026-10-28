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

## Expanded data and input support

The predictive six-output cohort contains168,200observations:134,561training,
17,503validation and16,136test. Training includes24,792distinct peptides and146
verified groove pairs (147across all partitions). The separately admitted ten
off-rate rows are excluded. The original IC50 controls use21,636exact training
rows, with2,925original validation and2,803original test rows. The bounds arm
uses23,512training rows. Primary comparisons always use original IC50 heldout IDs.

All targets remain separately named. IC50, KD, EC50, KD(~IC50) and KD(~EC50)
use log10(nM); half-life uses log10(minutes). For each endpoint, exact training
outcomes fit its mean and population SD. Each observed row contributes squared
standardized residual; upper/lower bounds contribute only on the violating side.
All endpoint weights are1before averaging over the sampled rows. There are no
cross-assay equality constraints, synthetic labels, pseudo-negatives or pretraining.
For concentration outputs, the500nM table is an endpoint-specific threshold
classification; an EC50/proxy threshold is not a calibrated binding probability.

Counts below are **all / exact** observations.

| Output | Train I | Train II | Validation I | Validation II | Test I | Test II |
|---|---:|---:|---:|---:|---:|---:|
| ic50 | 15,246 / 14,375 | 8,266 / 7,261 | 2,060 / 1,968 | 886 / 758 | 1,856 / 1,758 | 969 / 842 |
| kd | 14,759 / 4,097 | 422 / 350 | 1,898 / 567 | 36 / 32 | 1,606 / 561 | 96 / 82 |
| ec50 | 288 / 288 | 19 / 19 | 31 / 31 | 3 / 3 | 47 / 47 | 0 / 0 |
| kd_via_ic50 | 41,940 / 39,835 | 1,104 / 886 | 5,459 / 5,182 | 207 / 183 | 5,069 / 4,808 | 168 / 123 |
| kd_via_ec50 | 30,628 / 9,113 | 14,218 / 11,307 | 4,020 / 1,175 | 1,935 / 1,529 | 3,819 / 1,154 | 1,570 / 1,231 |
| half_life | 7,659 / 7,650 | 12 / 12 | 968 / 968 | 0 / 0 | 934 / 934 | 2 / 2 |

Training class proportions shift from66.4%I/33.6%II in the exact IC50 control
to82.1%I/17.9%II in the six-output data. About98.6%of class-I training rows are
9–10mers;89.4%of class-II rows are15mers. Long/unusual ligands have sparse support.
All observed grooves are classical human HLA-A/B/C or DR/DP/DQ; no nonclassical
or nonhuman transfer is tested. Peptide and verified groove sequences gain diversity;
cellular machinery, source-protein context, TCR and MS inputs receive no new data.

Training-only exact pMHC overlap is highly asymmetric. IC50 and KD(~IC50) share
6,676class-I contexts (correlation of per-context mean measurements0.963), but
only two class-II contexts. IC50/directKD overlap is69I/zeroII. This is a possible
source of unequal auxiliary benefit, not a causal finding: assay conditions are
not matched, and separate original assay IDs are not guaranteed independent studies.
The two labeled assays remain distinct. `training-output-overlap-by-class.csv`
and the source-disposition ledger preserve the full accounting.

No new source-only/zero-shot class-transfer experiment is included in this family.
The earlier single-class comparisons remain the relevant evidence for that question.
