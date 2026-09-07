# Implementation status and remaining design drift

Status after the first model-I/O repair. “Implemented” means a code path exists,
not that its outputs are scientifically validated.

The single intended design and full I/O inventory are in
[docs/model_io_contract.md](docs/model_io_contract.md).
[Issue #46](https://github.com/pirl-unc/presto/issues/46) is the tracked gap list.

## Implemented

- Segment-blocked sequence encoder without host/species/completeness metadata.
- mhcseqs-backed groove inputs: class I alpha1/alpha2; class II alpha1/beta1.
- Expanded class-specific processing/presentation and separate CD8/CD4 readouts;
  an explicitly configured collapsed topology also remains.
- Coarse downstream biological context for processing/excision.
- Quantitative binding/kinetics/stability outputs, fixed binding descriptor
  panels, T-cell response panels, elution/MS observations and pMHC-only receptor
  evidence. Output units/aliases are documented in the I/O contract.
- Selected-column T-cell response BCE; qualifier-aware binding panel loss.
- Sparse-label batches, supported MIL bags, shared canonical row forward inputs,
  boundary forwarding, selected-checkpoint hold-out prediction artifacts.
- Explicit merged-TSV or Hitlist primary source, optional bulk-MS, lineage/funnel/
  support audits and configurable launch gates.

## Partial or missing

| Area | Remaining work |
|---|---|
| Context schema | Independent APC/repertoire MHC sets and APC/MHC/TCR/repertoire/antigen species; no role copying |
| Cellular interventions | Concurrent components and cytokines, dose/time, measured controls versus unknown |
| Recognition | Presented-pMHC plus repertoire-selection context |
| Serving parity | Full biological-context API and common MHC preparation for row/MIL/tiling/multi-allele prediction |
| Quantitative assays | Family-specific routing, kinetic censoring and justified method-specific calibration |
| Assay panels | Joint configurations and supported missing readouts; current panels are per-axis |
| Groove semantics | Remove/rework beta2m interpretations of second-groove deletion and missing-sequence negative priors |
| Processing | Reconcile processing/excision representations and class-I-only explicit coupling |
| Corpus | Complete declared source contract, context/lineage coverage, supported negatives, task/split balance |
| Bulk MS | Candidate negatives, abundance/acquisition effects and validation of detectability |
| TCR-specific matching | No active sequence matcher or multi-TCR training; pMHC-only evidence is not a substitute |
| Scientific validity | Per-output held-out support/calibration; shared proxy labels do not identify distinct biological latents |

Historical benchmark counts and discarded prototypes are not current capability
claims. Experiments belong in [experiments/experiment_log.md](experiments/experiment_log.md);
current task execution notes belong in [tasks/todo.md](tasks/todo.md).
