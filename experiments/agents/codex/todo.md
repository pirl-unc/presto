# Codex Experiment TODO

Use this file for coarse-grained, agent-specific experimental backlog items.

## Active

- [ ] H100! batch-size sweep on broad 7-allele class-I contract
  - Fixed model/data contract, vary only batch size on `H100!`
  - Candidate designs:
    - `A03`
    - `A05`
    - `A06`
    - `A07`
  - Batch sizes:
    - `64`
    - `128`
    - `192`
    - `256`
  - Fixed training contract:
    - `numeric_no_qualitative`
    - qualifiers `all`
    - warm start
    - no synthetics
    - no ranking
    - fixed epoch count
  - Record:
    - success / failure
    - setup time
    - epoch wallclock
    - GPU utilization / memory
    - validation loss
    - probe affinities and ratios

- [ ] Assay-head structure and target-space bakeoff on broad 7-allele class-I contract
  - Compare:
    - one pooled affinity output
    - shared base logit + assay-specific residual
    - shared base logit + assay-context residual
    - shared base logit + segment-summary residual
    - shared base logit + factorized assay-context residual
    - shared base logit + factorized assay-context + pooled segment residual
  - Compare target spaces:
    - `log10`, `50k`
    - `log10`, `100k`
    - `mhcflurry`, `50k`
    - `mhcflurry`, `100k`
  - Keep assay/output mapping explicit:
    - `IC50` -> `IC50_nM`
    - `EC50` -> `EC50_nM`
    - direct/proxy `KD` -> `KD_nM` for merged condition
    - direct `KD`, `KD (~IC50)`, `KD (~EC50)` split in the split-KD conditions
  - Record whether inequalities help or hurt under each target space.
  - Record whether factorized assay context helps over flat assay-method conditioning.

## Pending

- [x] Positional composition bakeoff on broad 7-allele class-I contract
  - Result:
    - canonical Presto carry-forward candidates: `P04`, `P07`, `P03`
    - groove-transformer control carry-forward candidate: `G06`
  - Main takeaway:
    - explicit start/end/fraction composition is better than one-sided position
    - `P04` is the strongest new canonical candidate by validation loss
    - `P07` remains a strong broad-contract baseline

- [ ] Broad-contract canonical Presto vs groove-control follow-up after assay-head / target-space bakeoff
- [ ] Wider target-encoding / cap sweep with broad numeric contract
- [ ] Clean rerun of distributional-vs-regression benchmark on fixed broad contract
  - Keep:
    - `mhcflurry`
    - `log_mse`
    - `twohot_d2_logit`
    - `hlgauss_d2_logit`
  - Drop `D1-affine` until censoring/assay integration is rederived
  - Use fixed:
    - encoder/backbone
    - data contract
    - batch size
    - epoch budget
    - optimizer
    - LR schedule

- [ ] Distributional bin-count sweep after the clean rerun
  - Compare `K={8,16,32,64,128}`
  - Compare `MAX={50k,200k}`
  - Use only sane distributional families:
    - `twohot_d2_logit`
    - `hlgauss_d2_logit`
  - Keep the broad 7-allele numeric contract fixed
  - Record full val/test metric bundle plus probe metrics

- [x] Run and close [human-corpus frozen preflight](plans/human-corpus-preflight.md): actual source admission, partitions, supervision coverage and leakage limitations.

- [x] Run and close [component allocation audit](plans/component-allocation.md) on the frozen public human cohort; preserve historical memberships.

- [x] Run and close [corrected human preflight](plans/corrected-human-preflight.md) with new membership and a frozen candidate IC50 support declaration.

- [ ] Run and close the [corrected human IC50 fitting diagnostic](plans/corrected-human-fit.md) under the unchanged prospective manifest.

- Corrected-human IC50 attempt stopped before optimization: 211 raw/effective target mismatches; preserve/publish audit in #109, canonical target-policy qualification next in #110.

- [x] Audit canonical affinity policies on all 2,614 public frozen IC50 rows; retain 211 clipped IDs and 54 exact-value changes (2026-10-16_1000).
- [ ] Complete the separate prospective effective-target fit after #110 software release, preserving raw-measurement evaluation.

- [x] Native-presenter audit published: all 18 public assets and source/target joins independently verify; initial failed packaging evidence preserved.

- [x] Qualify fragment solver on complete required training support (2026-10-23_2029); all84fixed states pass, with unobserved-tail limitations retained.
- [x] Qualify [matched minibatch objective](plans/fragment-minibatch-objective.md), freeze and run a bounded biological length-effect comparison under [full-observation plan](plans/fragment-observation-domain.md).

- [x] Close 2026-10-23_2108: both real biological fits independently verify; small matched gain, no new best model.
- [x] Run [exactly profiled condition-offset comparison](plans/fragment-profiled-minibatches.md) with unchanged physical models/data/schedule; retain joint-offset results.

- [x] Training-only residual decomposition (2026-10-23_2214): persistent curve means account for ~77% of remaining error; independent replay passes. Paired fit now complete: weak improvement, no concentration validation. Next hypothesis should address persistent cross-peptide prediction error.

- [x] Run [nested sequence-context/response comparison](plans/digestion-sequence-context.md): four exact fits plus one prospectively gated conditional sampled hybrid; biological error and source scope remain primary.

- [ ] After2358closure, qualify and run [frozen cleavage-prior cellular transfer](plans/digestion-to-mapp-transfer.md), with an explicit input/removal bridge and unchanged canonical MAPP targets.
- [ ] Admit native intracellular peptidome panels through Hitlist788 before any workbook-based fit; preserve censored ratios and reused-measurement lineage.

## Frozen digestion-to-cellular transfer closure

- [x] Complete the registered0043 transfer comparison on unchanged MAPP data: all3exact2000stepfits independently verified; no method beats uniform composition.
- [x] Preserve sampled numerical success and original cost rejection; do not report an unperformed biological RL comparison.
- [x] Execute the bounded train-only monotone amplitude diagnostic in plans/cellular-transfer-amplitude.md, with controls and stop rule fixed before arithmetic.
- [ ] Prioritize canonical native cellular source788 and independent response/precursor evidence; do not repeatedly tune the same MAPP development set.

## Cellular amplitude diagnostic decision

- [x] Execute0106 and independently verify all scalar fits and prediction/metric outputs.
- [x] Enforce the stop rule: neither sequence prior beats calibrated uniform cleavage; no more tuning of this MAPP transfer bridge.
- [x] Record calibrated uniform cleavage as a provisional development control, with historical exposure and measurement limits explicit.
- [ ] Execute plans/native-cellular-peptidome-admission.md through the canonical Hitlist system, beginning with registered public-source/lineage audit. No Presto-only source adapter or shared-environment change.

## Native intracellular source audit and corrections

- [x] Complete available-source0117audit through the merged canonical package; all raw cells/ratios replay, semantic admission fails.
- [x] File concrete upstream findings as Hitlist791, including reuse matching, sequence/modification errors, heatmap projection and protocol provenance.
- [ ] Execute plans/native-peptidome-semantic-corrections.md against current upstream; preserve source values and explicit unknowns.
- [ ] Admit a canonical native-response recipe and preregister the next bounded real-data exact-versus-sampled comparison. No APC concentration claim from source admission alone.

## Native-to-APC response outcome

- [x] Canonical native recipe admitted0259; native controls0314 and exact/sampled processing0410 completed. Native processing gain small: testexactRMSE.243207 versus condition.250953.
- [x] Actual macrophage precursor/processing family0449 completed and independently verified: condition testRMSE.120011, precursor+sequence.139278, withoutprecursor.120008. Negative; no useful processing increment, no concentration claim.
- [x] Preserve all fixed outcomes and stop tuning inspected native/MAPP/APC test partitions. Exact and conditional sampled agree; no sampling advantage.
- [x] Diagnose precursor calibration and kinetic identifiability using training data only (0527); independent biological evaluation remains required. Do not conflate context offsets or nonconstant surrogate inputs with learned peptide-specific biology.

## APC diagnostic closure and independent machinery source

- [x] Complete0527training-only calibration/penalty/condition-preference factorial and independent closure. Gamma.0516; flexible models fit training but worsen innerCV. No outertest use or baseline promotion.
- [x] Enforce predeclared sampled biological gate: candidate worsens innerCVRMSE10.77%,so no sampled run. Preserve unexecuted path and failure, do not claim a new RLcomparison.
- [x] File/readback Hitlist798 for independent primaryDC immunoproteasome-loss sourcePMID20484733; existingbroadmouse13does not admit its quantitative contract.
- [ ] Admit that source through shared Hitlist APIs with original measurements/replicate/context and explicit RNA-versus-protein/machinery uncertainties; freeze independent evaluation before fitting. No Presto workbook curator or APC-concentration claim from elution alone.


- [x] After independently closing1007, execute the [frozen HAP1 outer evaluation](plans/hap1-frozen-outer-evaluation.md), without further fitting or selecting on held-out outcomes. This confirms elution-response generalization; APC concentration qualification remains separate.
- [x] Qualify original APC-derived endolysosomal numeric source under Hitlist812: no numeric peptide cohort in five official supplements; synthetic demo excluded. Source gap remains upstream.
- [x] Qualify broader measured non-MHC peptide MS1-response data underHitlist762: PXD021013 original samples and full intended-pool mapping recovered/verified in1546.
- [x] Implement and independently qualify canonical PXD021013 source/recipe in 1635; Hitlist PR814 passes 3,458 local tests. Release pending.
- [ ] Preregister a larger grouped relative-MS1 predictive comparison. This is an observation prior until independent transfer, not calibrated molarity.

- [x] Close 1635 canonical PXD021013 admission against original source quantities and intended-product groups.
- [ ] Freeze additional pool selection and held-out relative-response evaluation.
