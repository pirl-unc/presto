# Modal Hardware Bakeoff

- Date: 2026-03-12
- Agent: codex
- Purpose: compare A100 vs H100! vs H200 on a fixed broad-contract class-I binding benchmark at fixed batch size, measuring success/failure, runtime, memory, and early accuracy.

## Dataset Contract
- Source: merged IEDB-derived binding rows
- Alleles: HLA-A*02:01, HLA-A*24:02, HLA-A*03:01, HLA-A*11:01, HLA-A*01:01, HLA-B*07:02, HLA-B*44:02
- Assay families: IC50, direct KD, KD (~IC50), KD (~EC50), EC50
- Excluded: qualitative binding, presentation, immunogenicity
- Qualifier policy: all qualifiers with censor-aware loss
- Measurement profile: numeric_no_qualitative
- Train/val: expected 32,855 / 8,194

## Training Contract
- Epochs: 5
- Batch size: 140 for every condition
- Warm start: /checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt when applicable
- No synthetics
- No ranking losses
- GPU math/runtime defaults stay fixed; only hardware changes

## Designs
- DP00: heavy directness variant that previously OOMed on A100
- DP01: heavy directness variant that previously OOMed on A100
- A03: best canonical Presto by probe score from the broad frontier
- A07: best canonical Presto by val loss from the broad frontier

## Conditions
- 4 designs x 3 hardware = 12 conditions
- Hardware:
  - A100
  - H100!
  - H200

## Success Criteria
- Record for every condition:
  - success/failure
  - setup wall time
  - epoch wall time
  - GPU peak allocated/reserved GiB
  - GPU utilization metrics if emitted
  - final/best val loss
  - probe values for SLLQHLIGL, FLRYLLFGI, NFLIKFLLI
- Main question:
  - which hardware gives the best runtime for stable models?
  - which hardware makes DP00/DP01 viable without changing batch size?

## Output
- Experiment family under experiments/YYYY-MM-DD_HHMM_codex_modal-hardware-bakeoff
- README with matrix and takeaway
- options_vs_perf.{md,json}
- update experiments/experiment_log.md
