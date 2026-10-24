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
