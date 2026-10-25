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
- [x] Implement and independently qualify canonical PXD021013 source/recipe in1635; Hitlist814 is merged and2.0.26 is published with matching tested artifact hashes.
- [x] Preregister a larger grouped relative-MS1 predictive comparison. This is an observation prior until independent transfer, not calibrated molarity.

- [x] Close 1635 canonical PXD021013 admission against original source quantities and intended-product groups.
- [x] Freeze additional pool selection and held-out relative-response evaluation.

- [x] Run 1702 grouped nontryptic MS1 observer from the frozen 24-pool plan; preserve
  prospective selection, canonical admission, sealed test predictions and all outcomes.

- [x] Publish1702 and complete Hitlist814 clean-main release. Primary sequence
  observer passes held-out screen (13.57%test gain; Spearman0.52564), but pool-fixed
  C termini and unknown purity/load prevent pure detector/molar interpretation.
- [ ] Qualify a new prospective biological transfer test before treating this
  observer as a processing correction; exact/sampled arms must share its frozen
  parameters/objective. Previously inspected biological held-outs remain closed.

1702 is published at03f03c5; Hitlist762 and Presto130 comments are exact-body
verified in tasks/modular_model/ms1-observer-publication.json. Release40115753764
passed and2.0.26 is verified on PyPI; shared editables unchanged. Next admission specification:
[observer composition transfer](plans/observer-composition-transfer-admission.md).
A static peptide MS response cancels from same-peptide condition ratios; do not
claim that simply adding it can remedy the prior response-model failures.

- [x]1938independent lymph composition:21fits, numerical/data/prediction replay and report closed; no biological promotion.
- [x] Finish Hitlist818/819 source and recipe releases:2.0.27/2.0.28 bytes independently verified.
- [ ] After this failed transfer, specify a training-only or new-cohort test of separately calibrated observation and processing/selection assumptions; keep1938heldoutsclosed and preserve the full APCavailability objective.

- [x] 2018 HAP1 context diagnostic: frozen nine-arm training-only comparison; retain old outer sets closed. Plan: plans/hap1-context-ablation.md.

- [x] 2041 frozen MHC-score diagnostic: qualify available predictor, then fixed six-arm training-only comparison (plans/hap1-affinity-control.md).

- [x] Execute2106 shared TAP-route response exact/conditional-sampled comparison (plans/hap1-tap-route-response.md); independent concentration qualification remains required.

- [ ] After2106 failed its matched-feature gate, qualify baseline peptide amounts and their observation/selection contract before another response-only mechanism fit. Preserve old test closures and the full precursor-relative APC availability goal.

- [x]2210 baseline source and candidate decisions independently verify; Hitlist827 CI/release pending.
- [x] Execute2244 prospective baseline-composition comparison from the shared recipe; preserve full APC availability goal and prior test closures.

- [x] 2244: original baseline MAPP model and exact/REINFORCE comparison; modest signal, primary testgatefails, outcomes independently verified.
- [x] Finish Hitlist829CI/merge/release and827testedrelease40140479531; published bytes verified.
- [x] Qualify newly merged PXD009438 PBMC source828: support fails in2315 before scoring. Preserve PBMC-versus-serum conflict and unknown specimen/precursor links.


## Intact-degradomics constraint and recent source closure

- [x] 2315 PBMC qualification: support fails before scoring; no external-validation result.
- [x] Hitlist827 source2.0.30 published;829and830passed hosted checks and merged.
- [x] Finish exact existing recipe release jobs40143097533and40144589370; published bytes verified.
- [x] 2338 PXD058121 original kinetic-source audit and Hitlist831 with concrete exceptions.
- [x] Shared intact-degradomics source adapter and independent canonical replay.
- [x] Separately register generation/destruction exact-versus-sampled prediction,
  constrained by original quantitative time courses and honest holdout scope.
  Full precursor-relative APC availability/concentration remains the objective.

## Current intact-kinetic outcome and next representation question

- [x]0021 canonical source independently verified; source832merged18bbf81d with all hosted checks passing.
- [x]0046 measured temporal exact/conditional-score comparison completed and independently closed; physical testRMSE.359038 loses to direct.228184. No concentration result or RL advantage.
- [x]0116 training-only shape/optimizer diagnosis: six fits and independent replay; both registered materiality thresholds fail. No new held-out access.
- [x]2.0.31 and2.0.32 published artifacts independently byte-verified.
- [x] Finish exact source release40149903223 and recipe833 merge/release40151233071; Hitlist2.0.33/2.0.34 published bytes match successful builds.
- [x] Test parsimonious sequence features in0143:8.772%development RMSE gain overlength; paired uncertainty includeszero. Removing physical penalty gains1.362%, failing5%gate. All predictions/selection/metrics independently replay.
- [ ] Qualify broader prospective biological transfer of the frozen direct sequence comparator, then a matched exact/discrete processing comparison; do not tune original100/120minute outcomes or claim blind transfer from same-parent development CV.
- [ ] Extend to precursor-relative intracellular/APC quantities with measured initial/observation calibration; do not substitute observed-set or response-only success for the full availability goal.

## Native PSMD2 processing constraint after temporal development

- [x]0344recovers Sahu2021/PXD018722 native intracellular peptide quantities:
  10,647sequences,34,036positive cells; original run/search metadata and293,009
  source cells independently verified with13date-shaped gene annotations flagged.
- [ ] Implement shared Hitlist admission of the native Hi20S panel, exact reference,
  quantity aggregation and homogeneous preparation/machinery/search metadata.
- [ ] Register same-parent control/KD contrasts to test sequence-dependent relative
  product changes while cancelling shared precursor/run scale and fixed MS response.
  Plan:plans/native-psmd2-canonical-and-contrast.md. Model assumptions, source zeros,
  unsearched short products and APC/general concentration limits stay explicit.
- [ ] Follow a qualified direct representation with a matched exact/discrete
  generative comparison; preserve prior references and simulator/hybrid distinctions.

- [x] 0424 verifies the exact deposited reference, five original processed members,
  all quantities/IDs and all candidate origins. Raw/LFQ and modified-form quantities
  remain separate; original text resolves the 13 date-corrupted gene annotations.
- [x] Replay all 10,647 dependence assignments and descriptive support strata:
  complete-acquisition support has 82 parents, 81 components and 228 potential
  contrasts, before choosing a recipe or computing effects.
- [ ] Implement/release the canonical source in the durable isolated checkout
  `/Users/iskander/code/presto/.worktrees/hitlist-native-psmd2`, following
  `tasks/native_psmd2_source_spec.md`. Then register and run the biological comparison.

- Native contrast screen0630 failed the fixed outer gate. Next: register training-only acquisition sensitivity per plans/native-contrast-training-reliability.md; preserve inspected outer partitions and finish recipe2.0.38 tested publication.

- [x] 0711 acquisition sensitivity: median training Pearson0.8382, all9positive; independently replayed. Hitlist2.0.38published/byte-verified.
- [ ] Execute plans/native-boundary-context-development.md with fixed training-only nestedCV; outer groups remain closed and full APC availability goal unachieved.
