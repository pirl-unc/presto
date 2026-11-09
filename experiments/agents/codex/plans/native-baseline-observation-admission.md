# Native baseline amounts and observation calibration

Prepared after family0747 failed all four prospectively fixed gates. Stop feature
and optimizer expansion on the native response-only cohort. The full objective
remains precursor-relative peptide concentrations/distributions in APCs, including
short and unobserved products, machinery-conditioned processing and clearance,
and a matched differentiable versus discrete/RL comparison on biological data.
This plan does not redefine that objective as prediction of an observed peptide set.

## Identifiability before another predictor

For a peptide p from parent q in sample s, use the observation equation

`log I[p,s] = log A[q,s] + log availability[p|q,context_s]
              + MS_response[p,protocol_s] + run/recovery_scale[s] + error`.

MHC-bound measurements need an additional selection/capture component. Native
non-MHC signals do not need that component, but retain extraction, recovery and
instrument selection. Precursor abundance, availability and MS response are
different quantities. Parent centering removes shared precursor/run factors;
it does not remove peptide-specific detectability or recover total peptide yield.
Adding h(p) to log availability and subtracting it from MS response leaves the
likelihood unchanged unless independent anchors constrain the decomposition.

A fixed MS prior followed by a flexible residual using the same sequence features
can simply shift a regression prior. Neither nonconstant output nor better
intensity prediction proves that the residual represents intracellular abundance.
Audit component identifiability and transport before interpreting any latent.

## Stage A: source and calibration qualification, no model fitting

Register a timestamped family before numerical source analysis. Inventory public
processed assets and original metadata before downloading raw MS. Pin URLs,
access conditions, byte sizes, checksums, original coordinates and source versions.
Keep raw inputs once under data/, referenced by experiment bundles. The existing
Hitlist typed evidence/catalog/recipe contracts own any reusable ingestion.
No private access, author outreach, raw-MS search or new package installation.

Candidate: [PXD024809 primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12406853/)
describes ten heavy tryptic standards in a JY HLA-eluate dilution experiment.
Methods/Table1 describe twofold dilution; Results says1:10. Preserve that conflict.
It may test dose response and missingness; ten standards cannot establish broad
sequence generalization. Post-isolation standards do not calibrate intracellular
amounts or MHC recovery. The processed deposit is not yet independently qualified.

Before accepting any nominal dose/measurement join, recover:

- Original standard identity, actual isotope/modification form, reported purity,
  concentration assignment and uncertainty. Unknown certification stays unknown.
- Original sample/preparation/aliquot and acquisition identities; dilution order,
  volume, injection volume and on-column amount are separately typed quantities.
  Technical injections are not biological replication. A name-derived join must
  remain explicitly name-derived until supported by the source design.
- Peptide/precursor/charge/transition aggregation and raw versus normalized response;
  DDA versus DIA, instrument, LC, matrix, extraction and spiking stage.
- Intended query universe, scored candidates, eligibility thresholds, missingness,
  saturation and lower-limit states. Missing table entries are never numeric zero.
- Whether libraries use all dilution levels or the evaluation examples; retain
  outcome-informed selection and distinguish library-conditioned response from
  unconditional detection. Keep standards separate from the native background.
- Source resolution of the dilution inconsistency. Do not silently pick one grid
  or estimate dilution factors from the responses and call them supplied amounts.

Independently replay every admitted quantitative join and arithmetic conversion
from source cells. Source absence/conflict closes as an informative negative result.
No validation/test predictions apply to admission-only work. No source is called
calibrated merely because a download, schema or conservation test passes.

Reuse prior measuredMS1 source/observer evidence (PXD021013, families1546/1702)
as a protocol-scoped prior: crude synthetic pools confound synthesis/handling with
response. PXD004886/#762 has distinct form/run/dilution limitations; preserve its
recorded diagnostics. PXD057588 fails the earlier fresh-parent support gate and
measures MHC-selected TMT products; do not relax that gate or equate its chemistry
with native unmodified MS1. Qualify alternatives against these actual limits.

## Stage B: native baseline admission and direct baseline registration

Inspect baseline support through shared sources without reopening original0630
validation/test values. Their identities are already known and cannot become new
blind data for a baseline endpoint. Initial development may use only the55original
training components; any predictive confirmation needs prospectively frozen new
biological components/cohort and protocol-transfer evidence. Report physical
replication separately from sequence/component counts.

Before fitting, write a separate fixed contract identifying observation grain,
parent/span ambiguity, precursor quantities, recovery calibration, peptide forms,
candidate universe, supported lengths, censoring and exclusion policy. Preserve
the original raw signal alongside every derived target. Do not select new parents
or eligibility using response magnitude or a prospective model's predictions.

Register the simplest direct baseline on the actually identifiable endpoint first:
uniform and length-only controls, a parsimonious fixed sequence/context model,
the frozen protocol-qualified MS observer alone, and an observer-plus-availability
model with a justified separation of parameters. If the data identify only
conditional observed-set composition, label that diagnostic explicitly and require
additional anchors before promotion. Evaluate parent/component generalization,
calibration, spread, held-out likelihood/error and all baseline gains; evaluate
observer transfer separately from native predictor improvement. Prespecify folds,
parameters, losses, selection, uncertainty and materiality gates before any fit.
Do not copy response-only hyperparameters into a new target without a new contract.

## Stage C: matched full candidate distribution comparison

Only after the direct endpoint and observation path qualify, enumerate all supported
contiguous spans of each complete precursor for a small exact reference. Distinguish
the biological candidate universe from the MS query universe; preserve short
sub-ligand products and an explicit unrepresented-length/mass term. Unsupported
lengths are not demonstrated absent. The distribution requires a yield/lifetime
scale as well as normalized composition to represent precursor-relative amount.

Use the same small processing/clearance scorer, APC machinery inputs with explicit
unknown/KO status, observation likelihood and training data for exact marginalization
and sampled discrete trajectories. Evaluate probability/mass conservation, numerical
gradient agreement, score calls, runtime and biological predictive transfer. Retain
original exact/REINFORCE references; add clipping, group normalization, reuse or KL
only in matched controlled comparisons. A surrogate simulator alone is insufficient.
Neither RL nor marginalization supplies missing quantitative supervision.

## Limits, closure and decision

Local CPU, one BLAS thread; no Modal or shared-environment changes. Numerical stages
retain20min,3GiB RSS,1GiB disk reserve+32MiB startup margin,50ms process-tree monitoring
and OS peak checks. Network metadata retrieval is separately bounded by size/time;
do not download multi-gigabyte raw files to discover that processed assets are absent.
Freeze executed source and exact invocation, retain failed attempts, independently
verify summaries, publish canonical registry entries and precise Hitlist data asks.

Success for StageA is a verified admission or exclusion decision that determines
the next modeling action. It is not completion of the biological availability goal.
Do not declare the full model achieved until calibrated precursor-relative amounts,
short/unobserved products, APC machinery effects and independent biological transfer
are demonstrated with the actual trained system.
