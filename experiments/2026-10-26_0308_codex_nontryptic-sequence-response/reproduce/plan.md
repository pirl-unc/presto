# Prospective nontryptic sequence-response diagnostic

Prepared by Codex/GPT-6 after source family0236 independently verified the
shared PXD024917 source package. This plan does not authorize opening any old
biological held-out partition. Register a new timestamped family and freeze
code, exact input hashes, recipe and this plan before executing any numerical
admission or fitting stage. The full objective remains precursor-relative APC
peptide abundance with machinery context and a matched exact/discrete comparison.
This diagnostic qualifies only an observation component required by that model.

## Hypothesis and limits

A small sequence-dependent response term improves prediction of nominal-dose
MS2 signal for unseen precursor groups over a shared dose-response curve.
Use only S3A free1H standards spiked after purification:18 selected sequences,
71 dose rows and213 positive slots. Embedded2H S3B adds capture/purification and
is a different observation process; preserve its metadata but exclude it from
this fit. Do not use source-predicted MHC affinity as a feature or binding target.

The target is log10(source-reported summed product-ion signal), conditional on
this selected panel, matrix, isotope form and protocol. It is not unconditional
detection probability, calibrated isotope molarity, native MS1 response or APC
concentration. No zeros, LODs, censor bounds or absent dose rows are manufactured.
Source S3 signals are the endpoint; derived transition sums are not substituted
or selected by numerical agreement. S6 isotope conflicts and unestablished
acquisition links remain explicit exclusions from stronger interpretations.

## Shared admission contract

Use public EvidenceStore.open on the verified0236 attempt2 package and the
existing schema1 materialize API (not a new workbook parser). Freeze the exact
recipe and complete query requests. The recipe returns all348 candidates with
accepted/excluded/ambiguous dispositions. Require reported_positive signal,
source unit, unique standard/nominal-amount links, positive finite nominal fmol
and free-postpurification context. The135 embedded slots must remain in the
ledger with protocol-exclusion reasons, including their9missing signals.
All213 direct slots should be admitted; a mismatch stops admission for diagnosis.

Preserve source coordinates/raw quantities/status, source panel gene, calculated
and reported lengths/conflicts, isotope and protocol assertions, source slots and
complete lineage. C/D/E remain source slots, never invented acquisition IDs.
Unknown physical sample identity prohibits an independent-sample claim.
The missing low-dose row for one peptide stays absent. Features use calculated
sequence length, while the contradictory reported length remains an annotation.

## Features and three fixed arms

Known design input x=log10(nominal_fmol). Output y=log10(raw positive signal).
No pretraining, neural species/class prediction, MHC affinity, APC machinery or
processing parameters are allocated in this isolated observation diagnostic.

1. Unit-dose baseline: y=a+x; fit only a.
2. Shared-slope baseline: y=a+b*x; fit a,b.
3. Sequence response: y=a+b*x+w*z(sequence); z consists of the20 amino-acid
   frequencies and calculated length. Standardize only using training peptides.
   Constant feature columns are removed in each fit, explicitly recorded and
   never represented as trained coefficients. Penalize only w with ridge
   alpha in[0.1,1,10,100]; intercept and dose slope are unpenalized.

Fit all models by weighted least squares. Each peptide receives equal total
weight; within peptide each nominal dose receives equal weight and its reported
slots share that weight. Do not estimate replicate covariance from unknown run
identities. Report all fitted coefficients, active columns, rank, selected alpha
and train-only scaling. No hidden feature selection, clipping or epsilon transform.

## Prospective grouped development evaluation

Outer folds leave one reported precursor gene out; all peptides sharing that
gene, their doses and source slots stay together. This also enforces disjoint
peptide strings. Require every standard to have an unambiguous reported gene.
Freeze fold membership from sequence/gene metadata before reading signal targets
for fitting. Inner leave-one-training-gene-out validation selects alpha for arm3
by equal-peptide MSE, ties choosing the larger alpha. Outer labels never select
alpha, columns or normalization. Training folds must contain at least two doses
and enough groups for inner validation; fail rather than invent unseen intercepts.

No independent biological test split is claimed: the18selected standards and
source signals were already inspected during source qualification, and there
are no independent preparations or cohorts here. Call the outer predictions
out-of-group development estimates. Retain every inner validation prediction
and outer prediction with exact training membership hashes; this is the required
reason for omitting independent biological test metrics. Do not tune again on
these outer groups after closing this experiment.

Report pooled and equal-peptide RMSE/MAE/MSE, Pearson/Spearman, prediction and
target SD, residual after subtracting known logdose, per-peptide errors and
per-dose errors for all three arms. Correlations of constant vectors are null
with reasons. Report within-peptide dose sensitivity separately from between-
peptide response variation so dose alone cannot masquerade as sequence learning.
Binding-threshold metrics and log10(nM) errors do not apply to this endpoint.

Primary gate: arm3 lowers equal-peptide outer RMSE by at least10% versus arm2,
all fitted outer dose slopes positive, and predicted dose-adjusted SD at least
25% of observed dose-adjusted SD. Report the baseline's performance, not only a
relative gain. No biological-model promotion follows even if the gate passes.
Failure stops expansion/tuning on this panel; record the result and move to a
better-supported observation contract, rather than enlarging the same network.

## Verification and resource contract

Before original fitting, test the actual fit path on synthetic known responses,
missing/zero rejection, unequal row counts, strict group exclusion and nested
selection leakage. These fixtures are software checks, not biological training.
Independently verify admitted rows against shared source tables and prior0236
reconciliation. Independently solve ridge via augmented least squares (primary
solver uses normal equations), replay every prediction and recompute metrics.
Preserve coefficients/scalers and all inner/outer predictions, including failures.

Use one local CPU numerical thread, unchanged3GiB startup free+speculative and
process-tree RSS,20min/stage,1GiB disk reserve plus32MiB startup margin,50ms
sampling plus OS peak. No Modal, environment installs, new source retrieval or
rebuilding the source package per fit. Serialize original scientific stages with
software suites. Freeze all invoked Hitlist recipe/query dependencies.

## Connection to the availability objective

This estimates one protocol-specific observation factor: signal roughly equals
amount times response, with non-unit dose slope allowed. It must not absorb
processing/capture loss or be carried unchanged to native MS1. A passing result
can motivate a separately registered cross-protocol calibration test, and a
frozen compatible observation component can later be combined with processing
and pMHC selection. Exact enumeration and discrete trajectories then compare
under the same observation likelihood, precursor normalization, machinery
features, candidate support and held-out biological groups. Keep short/unobserved
products in the latent mass accounting and do not infer their concentrations
from absent MS signals. Current data do not yet prove those end-state requirements.
