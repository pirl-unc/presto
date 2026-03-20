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
