# Groove-corrected data-trust rerun

## Goal

Repair and then rerun Claude's stopped HLA-A*02:01 2x3 source-mapping family
without changing any supervision, split membership, or synthetic membership
between the legacy and masked policies.

## Preconditions

- Null-safe peptide/flank normalization is covered at every Hitlist entrypoint.
- Stable observation and selected-mapping lineage reaches held-out dumps.
- `hitlist==1.55.8`, `mhcseqs==2.6.10`, `mhcgnomes==3.41.0`, Hitlist artifacts,
  and the MHC index are frozen and checked.
- The exact public `python -m presto train unified` invocation accepts and
  enforces every preflight flag.
- All active targets occur in train/validation/test and every active binary
  target contains both classes in every split.
- Data membership is identical across model seeds; policy arms differ only in
  flank strings and terminus state.

## Matrix and contract

- Policies: `legacy_global_canonical`, `mask_unresolved`.
- Model/split seeds: 42, 43, 44; fixed data-seed base 42.
- HLA-A*02:01 numeric binding, capped half-life stability, and real-positive
  versus generated-decoy elution only. Sparse kinetics and Tm are excluded;
  no observed T-cell, processing, TCR, bulk-MS, MHC-only, or UniProt data.
- Expanded d128/l2/h4 Presto, 10 epochs, batch 256, AdamW 2.8e-4, H100!.

## Execution and closure

1. Run `modal run experiments/2026-09-04_1121_claude_groove-corrected-baseline/code/launch.py`.
2. Allow the launcher to complete all six exact CPU preflights before any GPU spawn.
3. Collect all required run artifacts, including data funnels, split support,
   hardware, summaries, and per-example lineage-bearing predictions.
4. Aggregate validation/test loss, affinity regression and 500 nM metrics,
   mapping strata, and elution decoy discrimination.
5. Update the experiment README and canonical experiment log. Do not call the
   family complete until all six conditions and post-run closure are finished.
