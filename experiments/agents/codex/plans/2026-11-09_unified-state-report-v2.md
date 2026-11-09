# Unified state report v2

Date: 2026-11-09  
Agent/model: Codex / GPT-5  
Source state: clean commit `1cca535e99a20e58c0ce00bc27edcbefbba3ad7d`

## Objective

Generate a deterministic, timestamped report for the closed comparator-parity
retrain. The report must name the retained reference checkpoint, distinguish it
from the rejected data-only parity arm, preserve assay-aware affinity endpoints,
include standardized performance panels and complete compact tables, and state
the exact development-only evidence boundary.

## Frozen inputs

- Closed experiment
  `experiments/2026-11-08_1800_codex_unified-parity-retrain`.
- Reference and parity selected-checkpoint receipts, all TRAIN/development data
  hashes, and the frozen Stage A parent.
- Final matched development tables, figures, prediction dumps, evaluator
  receipt, and independent verification receipt.
- NetMHCpan 4.2c, NetMHCIIpan 4.3e, MHCflurry 2.3.16/weights 2.3.0 receipts.
- PRIME 2.1 through mhctools/MixMHCpred 3.0 smoke receipt; accuracy remains
  unevaluated.
- Historical assay-aware T-cell development results are context only and are
  not represented as training of the retained reference checkpoint.

## Report contract

1. Pin Git to the clean closure commit above and mark the report development.
2. Select the reference checkpoint SHA-256
   `1d0d4122abf022830ea97fcc247b1d28c0645d7d42d206130cdf8a5d7762815a`.
3. Record the four original reference TRAIN/development bundles and the two BA
   v2 TRAIN bundles used only by the rejected parity arm. Do not call the
   comparator-derived BA rows independent evidence.
4. Preserve exact assay/output mapping: EC50, IC50, KD-via-EC50, KD-via-IC50,
   presentation, and operational processing remain distinct.
5. Use equal-study bootstrap estimates as primary. Include pooled exact-IC50
   Spearman only as a labeled secondary diagnostic. Preserve complete metrics,
   per-study tables, and all generated panels as supplemental artifacts.
6. State that no independent TEST or locked confirmation cohort was accessed.
7. Gate results explicitly: retain reference; processing passes versus
   MHCflurry; presentation, MHC-I affinity, MHC-II affinity, PRIME, and an
   all-task TEST claim do not pass.
8. Generate the bundle twice into separate roots and require byte equality for
   every report file.

## Reproduction and storage

- Store the manifest builder, frozen manifest, launch script/JSON, README, and
  generated report bundle in
  `experiments/2026-11-09_1000_codex_unified-state-report-v2`.
- Large checkpoints, source rows, comparator outputs, and per-example
  predictions stay on `presto-checkpoints` and are referenced by URI, byte
  count, SHA-256, and license note.
- Commit only compact report material and standardized figures/tables.

## Stop rules

- Stop if any local supplemental artifact fails its recorded hash/size.
- Stop if the two report renders differ by any byte.
- Do not relabel development as TEST or infer PRIME accuracy from the smoke.
- Do not claim selected-reference paired superiority when only parity paired
  differences were bootstrapped; use the exact method-value intervals and
  retain the explicit evidence boundary.
